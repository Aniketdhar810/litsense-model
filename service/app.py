"""App factory (D76): ``create_app(settings, loader)``.

The lifespan starts the boot in a background thread and returns at once, so the port opens
immediately (Render's free tier wants an open port quickly) and ``/ready`` reports 503 until the
bundle is loaded and warm. ``uvicorn --factory service.app:create_app`` also works.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

import litsense_core as core
from service import SERVICE_VERSION
from service.boot import Loader, load_bundle
from service.config import Settings
from service.errors import install_error_handlers
from service.logs import log
from service.metrics import Metrics
from service.middleware import GuardMiddleware, RequestContextMiddleware
from service.routes_ops import router as ops_router
from service.routes_v1 import router as v1_router
from service.state import ServiceState

LOG = logging.getLogger("litsense.service")

DESCRIPTION = """\
HTTP face of `litsense_core` for the LitSense Node backend. Every `/v1` route is one public
function of `docs/API_CONTRACT.md` with the same request fields and response schema.

* **Auth**: send `X-API-Key: <SERVICE_API_KEY>` on every `/v1` call and on `/metrics`.
* **Request id**: send `X-Request-Id` (letters, digits, `._:-`, up to 64) to correlate logs; the
  response echoes it, or a generated one.
* **Rate limit**: per `X-Client-Id` when sent (a per-reader pseudonym, never an email), else per
  client address. 429 carries `Retry-After`.
* **Errors**: `{"error": {"code", "message", "request_id", "details"?}}`. 400 invalid input
  (including an unknown `book_id`), 401, 404, 413, 429, 503 not ready or overloaded, 500.
* **LLM routes** (`explain`, `chat-turn`, `parse-onboarding`) answer with the template path after
  the hard timeout and then send `X-LitSense-Fallback: timeout`.
* **Readiness**: `/ready` is 503 until the bundle is loaded; `/v1` calls get 503 `not_ready`.
"""


def create_app(settings: Settings | None = None, loader: Loader | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.check()
    metrics = Metrics()
    svc = ServiceState(settings, loader or load_bundle, metrics)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        svc.start()
        log(
            LOG,
            logging.INFO,
            "service starting",
            version=SERVICE_VERSION,
            api_version=core.__api_version__,
            auth=settings.api_key is not None,
            core_threads=settings.core_threads,
            recommend_concurrency=settings.recommend_concurrency,
        )
        try:
            yield
        finally:
            log(LOG, logging.INFO, "service stopping", ready=svc.ready)

    app = FastAPI(
        title="LitSense model service",
        version=SERVICE_VERSION,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "model", "description": "The contract functions (docs/API_CONTRACT.md)."},
            {"name": "ops", "description": "Health, readiness and metrics."},
        ],
    )
    app.state.svc = svc
    install_error_handlers(app)
    app.include_router(ops_router)
    app.include_router(v1_router)
    # add_middleware puts each new one outside the previous: guard is innermost.
    app.add_middleware(GuardMiddleware, svc=svc)
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-Id", "X-Client-Id"],
            expose_headers=["X-Request-Id", "X-LitSense-Fallback", "X-LitSense-Cache"],
        )
    app.add_middleware(RequestContextMiddleware, metrics=metrics, routes=lambda: app.router.routes)
    return app
