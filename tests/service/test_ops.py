"""Cross-cutting behaviour: auth, readiness, request ids, rate limit, body limit, metrics,
logs, gzip, CORS (D77 to D79)."""

import io
import json
import logging
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.boot import BootResult
from service.config import ConfigError, Settings
from service.logs import JsonFormatter
from tests.service.conftest import KEY, TS
from tests.service.test_routes import assert_envelope


def test_auth_guards_v1_and_metrics_only(make_app: Any) -> None:
    app = make_app()
    client = app.client
    assert_envelope(client.post("/v1/recommend", json={}), 401, "unauthorized")
    wrong = client.post("/v1/recommend", json={}, headers={"X-API-Key": KEY + "x"})
    assert_envelope(wrong, 401, "unauthorized")
    assert_envelope(client.get("/v1/nope"), 401, "unauthorized")  # no route probing without a key
    assert_envelope(client.get("/metrics"), 401, "unauthorized")
    assert client.get("/health").status_code == 200 and client.get("/ready").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert app.post("/v1/recommend", {}).status_code == 200
    assert app.get("/metrics").status_code == 200


def test_settings_refuse_a_missing_key_unless_auth_is_off(make_app: Any) -> None:
    with pytest.raises(ConfigError, match="SERVICE_API_KEY"):
        create_app(Settings(service_api_key=None))
    with pytest.raises(ConfigError):
        create_app(Settings(service_api_key="   "))
    open_app = make_app(service_api_key=None, allow_no_auth=True)
    assert open_app.client.post("/v1/recommend", json={}).status_code == 200
    assert open_app.client.get("/health").json()["service"]["auth"] is False


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BUNDLE_URL", "https://example.org/serve_bundle.tar.gz")
    monkeypatch.setenv("SERVICE_API_KEY", "from-env")
    monkeypatch.setenv("RATE_LIMIT_RPS", "2.5")
    monkeypatch.setenv("ONNX_THREADS", "2")
    monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test")
    settings = Settings()
    assert settings.bundle_url == "https://example.org/serve_bundle.tar.gz"
    assert settings.api_key == "from-env" and "from-env" not in repr(settings)
    assert settings.rate_limit_rps == 2.5 and settings.onnx_threads == 2
    assert settings.cors_origin_list == ["http://a.test", "http://b.test"]


def test_readiness_before_and_after_load(make_app: Any) -> None:
    gate = threading.Event()
    app = make_app(gate=gate, wait=False)
    ready = app.client.get("/ready")
    assert ready.status_code == 503 and ready.json()["ready"] is False
    health = app.client.get("/health").json()
    assert health["ready"] is False and health["boot"]["status"] == "loading"
    blocked = app.post("/v1/recommend", {})
    assert_envelope(blocked, 503, "not_ready")
    assert blocked.headers["retry-after"] == "5"
    gate.set()
    assert app.svc.boot_done.wait(10) and app.svc.ready
    assert app.client.get("/ready").json() == {
        "ready": True,
        "status": "ready",
        "bundle_version": "fake-1",
    }
    health = app.client.get("/health").json()
    assert health["ready"] is True and health["bundle_version"] == "fake-1"
    assert health["n_books"] == 12 and health["mock"] is False and "llm" in health
    assert {"load_s", "warm_s"} <= set(health["boot"]) and health["boot"]["error"] is None
    assert health["service"]["version"] and health["service"]["api_version"] == "0.1.0"


def test_a_failed_boot_stays_up_and_says_why(make_app: Any) -> None:
    calls: list[int] = []

    def broken(settings: Settings, on_event: Any) -> BootResult:
        calls.append(1)
        raise FileNotFoundError("no manifest.json in /var/litsense")

    app = make_app(loader=broken, wait=False, boot_retries=2)
    assert app.svc.boot_done.wait(10)
    assert len(calls) == 2 and app.svc.status == "failed"
    health = app.client.get("/health").json()
    assert health["boot"]["attempts"] == 2 and "no manifest.json" in health["boot"]["error"]
    assert app.client.get("/ready").status_code == 503
    error = assert_envelope(app.post("/v1/recommend", {}), 503, "not_ready")
    assert "failed" in error["message"]


def test_request_id_is_echoed_or_generated(make_app: Any) -> None:
    app = make_app()
    echoed = app.client.get("/health", headers={"X-Request-Id": "node-7f3a.1"})
    assert echoed.headers["x-request-id"] == "node-7f3a.1"
    replaced = app.client.get("/health", headers={"X-Request-Id": "bad id with spaces"})
    assert len(replaced.headers["x-request-id"]) == 32
    generated = app.client.get("/health")
    assert len(generated.headers["x-request-id"]) == 32
    assert generated.headers["x-request-id"] != replaced.headers["x-request-id"]
    error = app.post("/v1/recommend", {"k": 0}, headers={"X-Request-Id": "abc-1"}).json()
    assert error["error"]["request_id"] == "abc-1"


def test_rate_limit_per_client(make_app: Any) -> None:
    app = make_app(rate_limit_rps=0.01, rate_limit_burst=2)
    assert [app.get("/v1/onboarding-grid").status_code for _ in range(2)] == [200, 200]
    limited = app.get("/v1/onboarding-grid")
    assert_envelope(limited, 429, "rate_limited")
    assert int(limited.headers["retry-after"]) >= 1
    other = app.get("/v1/onboarding-grid", headers={"X-Client-Id": "reader-b"})
    assert other.status_code == 200  # its own bucket
    assert app.client.get("/health").status_code == 200  # ops routes are never limited
    assert "litsense_rate_limited_total 1.0" in app.get("/metrics").text


def test_body_size_limit(make_app: Any) -> None:
    app = make_app(max_body_bytes=2048)
    big = {"state": {}, "message": "x" * 1500, "session": None}
    assert app.post("/v1/chat-turn", big).status_code == 200
    huge = {"text": "y" * 3000}
    assert_envelope(app.post("/v1/analyze-user-review", huge), 413, "payload_too_large")

    def chunks() -> Iterator[bytes]:  # no Content-Length: the streamed bytes are counted
        yield b'{"text": "'
        for _ in range(10):
            yield b"z" * 400
        yield b'"}'

    streamed = app.client.post(
        "/v1/analyze-user-review",
        content=chunks(),
        headers={"X-API-Key": KEY, "Content-Type": "application/json"},
    )
    assert_envelope(streamed, 413, "payload_too_large")


def test_metrics_are_prometheus_text(make_app: Any) -> None:
    app = make_app()
    app.post("/v1/recommend", {})
    app.post("/v1/explain", {"book_id": 1})
    app.get("/v1/does-not-exist")
    r = app.get("/metrics")
    assert r.headers["content-type"].startswith("text/plain")
    text = r.text
    requests = 'litsense_http_requests_total{method="POST",route="/v1/recommend",status="200"} 1.0'
    assert requests in text  # prometheus_client 0.26 sorts label names
    assert 'litsense_http_request_duration_seconds_bucket{le="0.08",route="/v1/recommend"}' in text
    assert 'route="unmatched"' in text and 'status="404"' in text
    assert "litsense_llm_calls_total" in text and 'prompt_id="explain"' in text
    assert "litsense_ready 1.0" in text and "litsense_process_rss_bytes" in text
    assert 'litsense_boot_seconds{phase="warm"}' in text


def test_logs_are_json_and_never_carry_user_content(make_app: Any) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("litsense")
    logger.addHandler(handler)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        app = make_app()
        state = {"user_id": 424242, "ratings": [{"book_id": 0, "stars": 5, "ts": TS}]}
        app.post(
            "/v1/chat-turn",
            {"state": state, "message": "SECRET-PHRASE-7781 like Dune"},
            headers={"X-Request-Id": "log-check-1"},
        )
        app.post("/v1/explain", {"state": state, "book_id": 1})
        app.post("/v1/analyze-user-review", {"text": "PRIVATE-REVIEW-TEXT was great"})
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    raw = stream.getvalue()
    lines = [json.loads(line) for line in raw.splitlines()]
    assert "SECRET-PHRASE-7781" not in raw and "424242" not in raw and "PRIVATE-REVIEW" not in raw
    access = [x for x in lines if x["msg"] == "request" and x.get("route") == "/v1/chat-turn"]
    assert access and access[0]["request_id"] == "log-check-1" and access[0]["status"] == 200
    assert set(access[0]) == {
        "ts",
        "level",
        "logger",
        "msg",
        "request_id",
        "method",
        "route",
        "status",
        "duration_ms",
    }
    llm = [x for x in lines if x["msg"] == "llm_call"]
    assert llm and {"prompt_id", "latency_ms", "ok", "fallback_reason"} <= set(llm[0])
    assert any(x.get("request_id") == "log-check-1" for x in llm)  # the id crosses into threads


def test_gzip_and_cors(make_app: Any) -> None:
    app = make_app()
    page = app.get("/v1/catalog", params={"limit": 12}, headers={"Accept-Encoding": "gzip"})
    assert page.headers.get("content-encoding") == "gzip"
    plain = app.get("/v1/onboarding-grid", headers={"Origin": "http://evil.test"})
    assert "access-control-allow-origin" not in plain.headers  # CORS off by default
    cors = make_app(cors_origins="http://localhost:5173")
    preflight = cors.client.options(
        "/v1/recommend",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "x-api-key,content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_two_lifespans_on_one_app_boot_once(make_app: Any) -> None:
    app = make_app()
    with TestClient(app.app) as again:
        assert again.get("/ready").status_code == 200
    assert app.svc.attempts == 1
