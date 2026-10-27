"""The real boot path on the tiny pipeline bundle: release URL -> download -> load_store ->
warm -> every route, and recommend's p95 inside the contract budget (D81). Needs torch to
build the bundle (CI installs the CPU wheel); skipped otherwise."""

import functools
import gzip
import json
import tarfile
import threading
import time
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from litsense_core.llm.client import set_client
from litsense_core.schemas import RecResponse
from litsense_core.store import set_store
from service.app import create_app
from service.config import Settings
from tests.service.conftest import AUTH, KEY, TS

RECOMMEND_P95_BUDGET_MS = 80.0  # contract section 2 (CLAUDE.md s15.4)


@pytest.fixture
def release(
    tiny_bundle_tar: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, dict[str, Any]]]:
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.delenv("LITSENSE_TEST_NO_KEY", raising=False)
    with tarfile.open(tiny_bundle_tar) as tar:
        manifest = json.load(tar.extractfile("serve_bundle/manifest.json"))  # type: ignore[arg-type]
        text = tar.extractfile("serve_bundle/catalog_text.json.gz").read()  # type: ignore[union-attr]
    manifest["_isbn13"] = json.loads(gzip.decompress(text))["isbn13"]
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(tiny_bundle_tar.parent))
    handler.log_message = lambda *args: None  # type: ignore[method-assign]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/serve_bundle.tar.gz", manifest
    server.shutdown()
    server.server_close()
    set_store(None)
    set_client(None)


def boot(settings: Settings) -> tuple[TestClient, Any]:
    app = create_app(settings)
    client = TestClient(app)
    client.__enter__()
    svc = app.state.svc
    assert svc.boot_done.wait(120), "boot did not finish"
    assert svc.ready, svc.load_error
    return client, svc


def p95(values: list[float]) -> float:
    return float(np.percentile(np.asarray(values), 95))


def test_boot_from_release_url_and_serve(
    release: tuple[str, dict[str, Any]], tmp_path: Path
) -> None:
    url, manifest = release
    var = tmp_path / "var"
    settings = Settings(
        bundle_url=url,
        bundle_dir=str(var),
        service_api_key=KEY,
        rate_limit_rps=0,
        recommend_concurrency=4,
    )
    client, svc = boot(settings)
    try:
        health = client.get("/health").json()
        assert health["bundle_version"] == manifest["bundle_version"]
        assert health["ranker"] is True, health["ranker_error"]  # libgomp present (D74)
        assert health["encoder"] is True and health["llm"] is False  # no key in the test bundle
        assert {"download_s", "load_s", "warm_s"} <= set(health["boot"])
        # tar and staging gone; the LLM cache sits next to the bundle
        assert sorted(p.name for p in var.iterdir()) == ["llm.sqlite", "serve_bundle"]
        n_items = manifest["n_items"]
        page = client.get("/v1/catalog", params={"limit": 1000}, headers=AUTH).json()
        assert page["total"] == n_items == len(page["items"])
        assert [b["isbn13"] for b in page["items"]] == [v or None for v in manifest["_isbn13"]]
        grid = client.get("/v1/onboarding-grid", params={"n": 4}, headers=AUTH).json()
        reader = {
            "ratings": [{"book_id": b, "stars": 5, "ts": TS + k} for k, b in enumerate(grid[:2])],
            "profile_version": 1,
        }

        # Latency through the whole HTTP stack (in-process client), warm, sequential.
        wall: list[float] = []
        server: list[float] = []
        for k in range(60):
            body = {"state": reader if k % 2 else {}, "k": 10}
            started = time.perf_counter()
            r = client.post("/v1/recommend", json=body, headers=AUTH)
            wall.append((time.perf_counter() - started) * 1000)
            assert r.status_code == 200, r.text
            feed = RecResponse.model_validate(r.json())
            assert feed.degraded is False and feed.items
            assert feed.items[0].evidence.signals["ranked_by"] == "ranker"
            server.append(feed.latency_ms)
        print(f"recommend tiny bundle: core p95 {p95(server):.1f} ms, http p95 {p95(wall):.1f} ms")
        assert p95(server) < RECOMMEND_P95_BUDGET_MS
        assert p95(wall) < RECOMMEND_P95_BUDGET_MS

        exp = client.post(
            "/v1/explain",
            json={"state": reader, "book_id": int(feed.items[0].book_id)},
            headers=AUTH,
        )
        assert exp.status_code == 200 and exp.json()["method"] == "template"
        hits = client.post(
            "/v1/semantic-search", json={"query": "plot characters dark", "k": 5}, headers=AUTH
        ).json()
        assert any(h["matched"] == "semantic" for h in hits)
        chat = client.post(
            "/v1/chat-turn",
            json={"state": reader, "message": "recommend a dark plot heavy book"},
            headers=AUTH,
        )
        assert chat.status_code == 200 and chat.json()["intent"] == "recommend_general"
        insights = client.get(f"/v1/book-insights/{grid[0]}", headers=AUTH).json()
        assert len(insights["aspect_bars"]) == 12
        review = client.post(
            "/v1/analyze-user-review",
            headers=AUTH,
            json={"text": "The plot kept me guessing. It turns out the butler did it."},
        )
        assert review.json()["spoiler_sentences"] >= 1
        rails = client.post("/v1/home-rails", json={"state": reader}, headers=AUTH)
        assert rails.status_code == 200 and rails.json()["top_picks"]
        bad = client.post(
            "/v1/recommend",
            headers=AUTH,
            json={"state": {"ratings": [{"book_id": n_items + 5, "stars": 4, "ts": TS}]}},
        )
        assert bad.status_code == 400 and bad.json()["error"]["code"] == "bad_request"
        metrics = client.get("/metrics", headers=AUTH).text
        assert 'litsense_boot_seconds{phase="download"}' in metrics
    finally:
        client.__exit__(None, None, None)

    # A restart on a persistent disk finds the extracted bundle and skips the download.
    again, svc2 = boot(Settings(bundle_dir=str(var), service_api_key=KEY))
    try:
        assert (
            "download_s" not in svc2.timings and svc2.bundle_version == manifest["bundle_version"]
        )
    finally:
        again.__exit__(None, None, None)
