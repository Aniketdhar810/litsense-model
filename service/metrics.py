"""Prometheus metrics on a private registry (one per app, so tests can build several apps)."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)

from litsense_core.store import rss_mb

HTTP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
LLM_BUCKETS = (0.01, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0)
CONTENT_TYPE = CONTENT_TYPE_LATEST


def _rss_bytes() -> float:
    mb = rss_mb()
    return float(mb) * 1024 * 1024 if mb is not None else 0.0


class Metrics:
    def __init__(self) -> None:
        r = self.registry = CollectorRegistry()
        ProcessCollector(registry=r)  # process_resident_memory_bytes, CPU seconds, fds (Linux)
        self.requests = Counter(
            "litsense_http_requests",
            "HTTP requests by route template, method and status.",
            ["route", "method", "status"],
            registry=r,
        )
        self.latency = Histogram(
            "litsense_http_request_duration_seconds",
            "Request latency by route template.",
            ["route"],
            buckets=HTTP_BUCKETS,
            registry=r,
        )
        self.llm_calls = Counter(
            "litsense_llm_calls",
            "LLM calls from the core's on_event hook; outcome: ok, cached or the fallback reason.",
            ["prompt_id", "outcome"],
            registry=r,
        )
        self.llm_latency = Histogram(
            "litsense_llm_call_duration_seconds",
            "LLM call latency by prompt id.",
            ["prompt_id"],
            buckets=LLM_BUCKETS,
            registry=r,
        )
        self.llm_route_timeouts = Counter(
            "litsense_llm_route_timeouts",
            "LLM-backed requests that hit LLM_TIMEOUT_S and answered with the template.",
            ["route"],
            registry=r,
        )
        self.cache = Counter(
            "litsense_cache_lookups",
            "Response cache lookups by cache and result (hit or miss).",
            ["cache", "result"],
            registry=r,
        )
        self.rate_limited = Counter(
            "litsense_rate_limited", "Requests refused by the token bucket.", registry=r
        )
        self.overloaded = Counter(
            "litsense_overloaded",
            "Recommend calls refused after waiting QUEUE_TIMEOUT_S for a slot.",
            registry=r,
        )
        self.ready = Gauge("litsense_ready", "1 once the bundle is loaded and warm.", registry=r)
        self.rss = Gauge("litsense_process_rss_bytes", "Resident memory (VmRSS).", registry=r)
        self.rss.set_function(_rss_bytes)
        self.boot = Gauge("litsense_boot_seconds", "Boot phase durations.", ["phase"], registry=r)

    def llm_call(self, prompt_id: str, outcome: str, seconds: float) -> None:
        self.llm_calls.labels(prompt_id=prompt_id, outcome=outcome).inc()
        if outcome in ("ok", "cached") or seconds > 0:
            self.llm_latency.labels(prompt_id=prompt_id).observe(seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)
