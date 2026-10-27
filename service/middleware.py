"""Pure ASGI middleware (D77, D79), outermost first:

1. ``RequestContextMiddleware``: request id in and out, access log, HTTP metrics, and a last-resort
   500 envelope for anything that escaped the handlers.
2. (optional CORS, then gzip, added in ``app.py``)
3. ``GuardMiddleware``: API key, readiness, rate limit and body size, in that order, before the
   router reads a byte of the body.
"""

import hmac
import logging
import math
import re
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.routing import BaseRoute, Match
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from service.errors import error_response
from service.logs import REQUEST_ID, log

ACCESS_LOG = logging.getLogger("litsense.access")
LOG = logging.getLogger("litsense.service")
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
CLIENT_ID_MAX = 128
QUIET_PATHS = frozenset({"/health", "/ready", "/metrics"})


class BodyTooLarge(StarletteHTTPException):
    """An HTTPException so FastAPI's body reader re-raises it as is (413, not 400)."""

    def __init__(self, limit: int) -> None:
        super().__init__(413, f"request body is larger than {limit} bytes")


def route_label(scope: Scope, routes: Sequence[BaseRoute]) -> str:
    """The route template (bounded label set for metrics and logs), or ``unmatched``."""
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str):
        return path
    for candidate in routes:
        match, _ = candidate.matches(scope)
        if match in (Match.FULL, Match.PARTIAL):
            return str(getattr(candidate, "path", "unmatched"))
    return "unmatched"


class RequestContextMiddleware:
    def __init__(
        self, app: ASGIApp, metrics: Any, routes: Callable[[], Sequence[BaseRoute]]
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.routes = routes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = Headers(scope=scope).get("x-request-id", "")
        request_id = incoming if REQUEST_ID_RE.match(incoming) else uuid.uuid4().hex
        token = REQUEST_ID.set(request_id)
        started = time.perf_counter()
        response = {"status": 500, "started": False}

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
                response["started"] = True
                MutableHeaders(scope=message)["X-Request-Id"] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        except Exception:
            LOG.error("unhandled error", exc_info=True)
            if not response["started"]:
                await error_response(500, "internal", "internal error")(
                    scope, receive, send_with_id
                )
        finally:
            seconds = time.perf_counter() - started
            route = route_label(scope, self.routes())
            status = int(response["status"])
            method = scope["method"]
            self.metrics.requests.labels(route=route, method=method, status=str(status)).inc()
            self.metrics.latency.labels(route=route).observe(seconds)
            quiet = scope["path"] in QUIET_PATHS and status < 400
            log(
                ACCESS_LOG,
                logging.DEBUG if quiet else logging.INFO,
                "request",
                method=method,
                route=route,
                status=status,
                duration_ms=round(seconds * 1000, 2),
            )
            REQUEST_ID.reset(token)


def client_key(headers: Headers, scope: Scope) -> str:
    """X-Client-Id when the backend sends one (a per-reader pseudonym, never an email), else
    the client address (behind Render's proxy, uvicorn takes it from X-Forwarded-For)."""
    client_id = headers.get("x-client-id", "").strip()
    if client_id:
        return "id:" + client_id[:CLIENT_ID_MAX]
    client = scope.get("client")
    return "ip:" + (client[0] if client else "unknown")


def key_matches(given: str | None, expected: str) -> bool:
    if not given:
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


class GuardMiddleware:
    def __init__(self, app: ASGIApp, svc: Any) -> None:
        self.app = app
        self.svc = svc

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        svc = self.svc
        settings = svc.settings
        path: str = scope["path"]
        headers = Headers(scope=scope)
        is_v1 = path == "/v1" or path.startswith("/v1/")
        expected = settings.api_key
        needs_key = (is_v1 or path == "/metrics") and expected is not None
        if needs_key and not key_matches(headers.get("x-api-key"), expected):
            refusal = error_response(401, "unauthorized", "missing or wrong X-API-Key header")
            await refusal(scope, receive, send)
            return
        if is_v1:
            if not svc.ready:
                message = (
                    "the model failed to load; see /health"
                    if svc.status == "failed"
                    else "the model is still loading"
                )
                await error_response(503, "not_ready", message, headers={"Retry-After": "5"})(
                    scope, receive, send
                )
                return
            wait = svc.rate_limiter.take(client_key(headers, scope))
            if wait > 0:
                svc.metrics.rate_limited.inc()
                retry = str(max(1, math.ceil(wait)))
                refusal = error_response(
                    429, "rate_limited", "too many requests", headers={"Retry-After": retry}
                )
                await refusal(scope, receive, send)
                return
        limit = settings.max_body_bytes
        length = headers.get("content-length", "")
        if length.isdigit() and int(length) > limit:
            refusal = error_response(
                413, "payload_too_large", f"request body is larger than {limit} bytes"
            )
            await refusal(scope, receive, send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge(limit)
            return message

        await self.app(scope, limited_receive, send)
