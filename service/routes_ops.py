"""Operations routes, outside ``/v1``: liveness, readiness, Prometheus metrics (D78)."""

import os
import platform
import time
from typing import Any

from fastapi import APIRouter, Request, Response, Security
from fastapi.responses import JSONResponse

import litsense_core as core
from service import SERVICE_VERSION
from service.metrics import CONTENT_TYPE
from service.routes_v1 import API_KEY_HEADER
from service.state import ServiceState

router = APIRouter(tags=["ops"])


def _svc(request: Request) -> ServiceState:
    return request.app.state.svc  # type: ignore[no-any-return]


@router.get("/health", summary="Liveness and component status (always 200)")
async def health(request: Request) -> dict[str, Any]:
    """`litsense_core.health()` (bundle version, cutoff, rss_mb, ranker, ranker_error, llm, ...)
    once loaded, plus `ready`, boot timings and process facts. 200 while the process is alive,
    so use `/ready` for traffic decisions."""
    svc = _svc(request)
    settings = svc.settings
    body: dict[str, Any] = core.health() if svc.ready else {"status": svc.status}
    body.update(
        ready=svc.ready,
        bundle_version=svc.bundle_version,
        boot={
            "status": svc.status,
            "attempts": svc.attempts,
            "error": svc.load_error,
            **svc.timings,
        },
        service={
            "version": SERVICE_VERSION,
            "api_version": core.__api_version__,
            "pid": os.getpid(),
            "python": platform.python_version(),
            "uptime_s": round(time.time() - svc.started_at, 1),
            "auth": settings.api_key is not None,
            "core_threads": settings.core_threads,
            "recommend_concurrency": settings.recommend_concurrency,
            "rate_limit_rps": settings.rate_limit_rps,
        },
    )
    return body


@router.get(
    "/ready",
    summary="Readiness: 200 once the bundle is loaded and warm, else 503",
    responses={503: {"description": "Still loading, or the boot failed"}},
)
async def ready(request: Request) -> JSONResponse:
    svc = _svc(request)
    body = {"ready": svc.ready, "status": svc.status, "bundle_version": svc.bundle_version}
    if not svc.ready:
        body["error"] = svc.load_error
        return JSONResponse(body, status_code=503, headers={"Retry-After": "5"})
    return JSONResponse(body)


@router.get(
    "/metrics",
    summary="Prometheus metrics (text format, needs X-API-Key)",
    dependencies=[Security(API_KEY_HEADER)],
    response_class=Response,
)
async def metrics(request: Request) -> Response:
    return Response(_svc(request).metrics.render(), media_type=CONTENT_TYPE)
