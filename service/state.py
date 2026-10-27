"""Process state: background boot, readiness, the core thread pool, slots and caches (D78, D79).

Thread safety: the ``ArtifactStore`` is read-only once ``ready`` is set (memory-mapped arrays,
FAISS flat-index search, the ONNX session and LightGBM ``predict`` all allow concurrent reads), and
the core's LLM client locks its cache, budget and limiter (contract section 5). The service adds
its own locks around the response caches and the rate limiter. Core calls run on a dedicated
anyio thread limiter of ``CORE_THREADS`` so the event loop stays free for health checks.
"""

import contextvars
import hashlib
import logging
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, TypeVar

import anyio
import anyio.to_thread

import litsense_core as core
from litsense_core.schemas import Overrides, UserState
from service.boot import BootResult, Loader
from service.cache import TTLCache
from service.catalog import CatalogView
from service.config import Settings
from service.errors import ApiError
from service.logs import log
from service.metrics import Metrics
from service.ratelimit import TokenBucketLimiter

LOG = logging.getLogger("litsense.service")
LLM_LOG = logging.getLogger("litsense.llm")
T = TypeVar("T")
LLM_EVENT_FIELDS = (
    "prompt_id",
    "model",
    "latency_ms",
    "cached",
    "ok",
    "fallback_reason",
    "attempts",
)


def without_ai(state: UserState) -> UserState:
    """The same reader with ``disabled_ai`` set: the core answers with its template path."""
    overrides = state.overrides.model_copy(update={"disabled_ai": True})
    return state.model_copy(update={"overrides": overrides})


def digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


class ServiceState:
    def __init__(self, settings: Settings, loader: Loader, metrics: Metrics) -> None:
        self.settings = settings
        self.loader = loader
        self.metrics = metrics
        self.ready = False
        self.status = "starting"  # starting, loading, ready, failed
        self.store: Any = None
        self.catalog: CatalogView | None = None
        self.load_error: str | None = None
        self.attempts = 0
        self.timings: dict[str, float] = {}
        self.started_at = time.time()
        self.boot_done = threading.Event()
        self.explain_cache: TTLCache[Any] = TTLCache(
            settings.cache_max_entries, settings.cache_ttl_s
        )
        self.insights_cache: TTLCache[Any] = TTLCache(
            settings.cache_max_entries, settings.cache_ttl_s
        )
        self.rate_limiter = TokenBucketLimiter(settings.rate_limit_rps, settings.rate_limit_burst)
        self.limiter: anyio.CapacityLimiter | None = None
        self.recommend_slots: anyio.Semaphore | None = None
        self._thread: threading.Thread | None = None

    @property
    def bundle_version(self) -> str | None:
        return str(self.store.version) if self.store is not None else None

    # ------------------------------------------------------------------ boot
    def start(self) -> None:
        """Called from the lifespan, inside the event loop: build the async primitives there and
        start the boot thread once (a second lifespan on the same app does not reboot)."""
        self.limiter = anyio.CapacityLimiter(self.settings.core_threads)
        self.recommend_slots = anyio.Semaphore(self.settings.recommend_concurrency)
        if self._thread is None:
            self._thread = threading.Thread(target=self._boot, name="litsense-boot", daemon=True)
            self._thread.start()

    def _boot(self) -> None:
        retries = self.settings.boot_retries
        for attempt in range(1, retries + 1):
            self.attempts = attempt
            self.status = "loading"
            try:
                result: BootResult = self.loader(self.settings, self.on_llm_event)
                timings = dict(result.timings)
                started = time.perf_counter()
                catalog = CatalogView(result.store)
                if self.settings.warmup:
                    self._warm(result.store)
                timings["warm_s"] = round(time.perf_counter() - started, 3)
            except Exception as exc:  # noqa: BLE001 (reported in /health, retried)
                self.load_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                LOG.error(
                    "boot attempt failed",
                    exc_info=True,
                    extra={"fields": {"attempt": attempt, "retries": retries}},
                )
                if attempt < retries:
                    time.sleep(self.settings.boot_backoff_s * 3 ** (attempt - 1))
                continue
            self.store, self.catalog, self.timings = result.store, catalog, timings
            self.load_error = None
            self.status = "ready"
            self.ready = True
            self.metrics.ready.set(1)
            for phase, seconds in timings.items():
                self.metrics.boot.labels(phase=phase.removesuffix("_s")).set(seconds)
            health = core.health()
            log(
                LOG,
                logging.INFO,
                "ready",
                bundle_version=self.bundle_version,
                attempt=attempt,
                rss_mb=health.get("rss_mb"),
                ranker=health.get("ranker"),
                llm=health.get("llm"),
                **timings,
            )
            self.boot_done.set()
            return
        self.status = "failed"
        log(LOG, logging.ERROR, "boot failed, staying up without a model", error=self.load_error)
        self.boot_done.set()

    @staticmethod
    def _warm(store: Any) -> None:
        """First calls pay for lazy imports, page faults on the memory maps and ONNX graph setup;
        pay them before the platform routes traffic here."""
        core.recommend(UserState(), k=10)
        first = int(store.trending_top[0]) if len(store.trending_top) else 0
        core.explain(UserState(overrides=Overrides(disabled_ai=True)), first)
        if store.encoder is not None:
            core.semantic_search("a gripping mystery with a twist", k=5)

    def on_llm_event(self, event: dict[str, Any]) -> None:
        """``configure_llm``'s hook: one counter, one histogram and one log line per LLM call.
        The event never carries the prompt or the key; only the listed fields are logged."""
        if event.get("cached"):
            outcome = "cached"
        elif event.get("ok"):
            outcome = "ok"
        else:
            outcome = str(event.get("fallback_reason") or "error")
        prompt_id = str(event.get("prompt_id"))
        self.metrics.llm_call(prompt_id, outcome, float(event.get("latency_ms") or 0.0) / 1000)
        log(LLM_LOG, logging.INFO, "llm_call", **{k: event.get(k) for k in LLM_EVENT_FIELDS})

    # ------------------------------------------------------------------ calls into the core
    async def _run(self, job: Callable[[], T], abandon: bool) -> T:
        context = contextvars.copy_context()  # request id reaches logs written in the thread
        try:
            return await anyio.to_thread.run_sync(
                context.run, job, limiter=self.limiter, abandon_on_cancel=abandon
            )
        except ApiError:
            raise
        except ValueError as exc:  # invalid input per contract section 5: always 400
            raise ApiError(400, "bad_request", str(exc)[:500]) from exc
        except Exception as exc:
            LOG.error("core call failed", exc_info=True)
            raise ApiError(500, "internal", "internal error") from exc

    async def call(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return await self._run(partial(fn, *args, **kwargs), abandon=False)

    async def call_llm(
        self, route: str, primary: Callable[[], T], fallback: Callable[[], T]
    ) -> tuple[T, str | None]:
        """``primary`` under LLM_TIMEOUT_S; past it, answer with ``fallback`` (the template path)
        as the contract's hard timeout promises. The abandoned thread finishes on its own."""
        try:
            with anyio.fail_after(self.settings.llm_timeout_s):
                return await self._run(primary, abandon=True), None
        except TimeoutError:
            self.metrics.llm_route_timeouts.labels(route=route).inc()
            log(
                LOG,
                logging.WARNING,
                "llm route timed out, answering with the template",
                route=route,
                timeout_s=self.settings.llm_timeout_s,
            )
            return await self._run(fallback, abandon=False), "timeout"

    @asynccontextmanager
    async def recommend_slot(self) -> AsyncIterator[None]:
        """At most RECOMMEND_CONCURRENCY ranking calls at once; a caller waits up to
        QUEUE_TIMEOUT_S for a slot, then gets 503 overloaded."""
        assert self.recommend_slots is not None
        try:
            with anyio.fail_after(self.settings.queue_timeout_s):
                await self.recommend_slots.acquire()
        except TimeoutError:
            self.metrics.overloaded.inc()
            raise ApiError(
                503,
                "overloaded",
                "too many concurrent recommend calls, retry shortly",
                headers={"Retry-After": "1"},
            ) from None
        try:
            yield
        finally:
            self.recommend_slots.release()

    def cache_lookup(self, cache: TTLCache[Any], name: str, key: str) -> Any:
        value = cache.get(key)
        if cache.enabled:
            self.metrics.cache.labels(
                cache=name, result="hit" if value is not None else "miss"
            ).inc()
        return value
