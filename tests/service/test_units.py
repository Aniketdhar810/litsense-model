"""Units without an app: cache, token bucket, boot helpers, catalogue rows, OpenAPI file."""

import functools
import io
import json
import tarfile
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from service import openapi
from service.boot import BootError, fetch_bundle, llm_config, load_bundle, locate_bundle
from service.cache import TTLCache
from service.catalog import CatalogView
from service.config import Settings
from service.ratelimit import TokenBucketLimiter
from tests.litsense_core.fake_store import build_store


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_ttl_cache_expires_and_evicts() -> None:
    clock = Clock()
    cache: TTLCache[str] = TTLCache(2, 10.0, clock)
    cache.set("a", "1")
    cache.set("b", "2")
    assert cache.get("a") == "1"  # a is now the most recent
    cache.set("c", "3")
    assert cache.get("b") is None and cache.get("a") == "1" and len(cache) == 2
    clock.now += 10.0
    assert cache.get("a") is None
    off: TTLCache[str] = TTLCache(0, 10.0, clock)
    off.set("a", "1")
    assert off.get("a") is None and not off.enabled


def test_token_bucket_refills_per_key() -> None:
    clock = Clock()
    bucket = TokenBucketLimiter(rate=2.0, burst=2, clock=clock)
    assert [bucket.take("a"), bucket.take("a")] == [0.0, 0.0]
    assert bucket.take("a") == pytest.approx(0.5)
    assert bucket.take("b") == 0.0
    clock.now += 0.5
    assert bucket.take("a") == 0.0
    assert TokenBucketLimiter(rate=0, burst=1).take("x") == 0.0  # off
    small = TokenBucketLimiter(rate=1.0, burst=1, max_keys=2, clock=clock)
    for key in ("k1", "k2", "k3"):
        small.take(key)
    assert len(small._buckets) == 2


def test_locate_bundle_and_llm_config(tmp_path: Path) -> None:
    assert locate_bundle(tmp_path) is None
    (tmp_path / "serve_bundle").mkdir()
    (tmp_path / "serve_bundle" / "manifest.json").write_text("{}")
    assert locate_bundle(tmp_path) == tmp_path / "serve_bundle"
    assert locate_bundle(tmp_path / "serve_bundle") == tmp_path / "serve_bundle"
    manifest = {
        "config": {
            "llm": {
                "enabled": True,
                "provider": "gemini",
                "model": "m-1",
                "cache_path": ".cache/llm.sqlite",
            }
        }
    }
    cfg = llm_config(manifest, Settings(bundle_dir=str(tmp_path), allow_no_auth=True))
    assert cfg["cache_path"] == str(tmp_path / "llm.sqlite") and cfg["model"] == "m-1"
    assert "api_key" not in cfg and cfg["api_key_env"] == "GEMINI_API_KEY"
    fake = llm_config(
        manifest, Settings(bundle_dir=str(tmp_path), llm_provider="fake", llm_cache_path=":memory:")
    )
    assert (
        fake["provider"] == "fake" and fake["model"] == "fake" and fake["cache_path"] == ":memory:"
    )


def test_load_bundle_needs_a_bundle_or_a_url(tmp_path: Path) -> None:
    with pytest.raises(BootError, match="BUNDLE_URL"):
        load_bundle(Settings(bundle_dir=str(tmp_path)), lambda event: None)


@pytest.fixture
def served(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Path, str]]:
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    root = tmp_path / "www"
    root.mkdir()
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *args: None  # type: ignore[method-assign]
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _tar(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_fetch_bundle_stages_then_moves_into_place(
    served: tuple[Path, str], tmp_path: Path
) -> None:
    root, base = served
    _tar(
        root / "serve_bundle.tar.gz",
        {
            "serve_bundle/manifest.json": b'{"n_items": 1}',
            "serve_bundle/catalog.npz": b"x",
            "other/skip.txt": b"y",
        },
    )
    dest = tmp_path / "var"
    folder = fetch_bundle(f"{base}/serve_bundle.tar.gz", dest)
    assert folder == dest / "serve_bundle" and (folder / "manifest.json").is_file()
    assert sorted(p.name for p in dest.iterdir()) == ["serve_bundle"]  # no tar, no staging
    assert locate_bundle(dest) == folder
    with pytest.raises(Exception):  # noqa: B017 (httpx raises its own HTTPStatusError)
        fetch_bundle(f"{base}/missing.tar.gz", tmp_path / "other")
    assert locate_bundle(tmp_path / "other") is None  # a failed fetch leaves nothing loadable


def test_catalog_rows_map_unknowns_to_null(tmp_path: Path) -> None:
    store = build_store(tmp_path / "b")
    store.catalog.year = np.where(np.arange(12) == 2, -1, store.catalog.year).astype(np.int32)
    store.catalog.pages = np.where(np.arange(12) == 2, -1, store.catalog.pages).astype(np.int32)
    import gzip

    with gzip.open(tmp_path / "b" / "catalog_text.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"isbn13": ["9780593135204"] + [None] * 11}, f)
    view = CatalogView(store)
    assert view.book(0).isbn13 == "9780593135204" and view.book(1).isbn13 is None
    assert view.book(2).year is None and view.book(2).pages is None
    assert view.book(0).series_pos is None and view.book(6).series_pos == 2.0
    assert [b.book_id for b in view.page(10, 5)] == [10, 11]


def test_committed_openapi_matches_the_app() -> None:
    committed = json.loads(openapi.OUT.read_text(encoding="utf-8"))
    current: dict[str, Any] = openapi.build()
    assert committed == current, "service/openapi.json is stale: run python -m service.openapi"
    paths = set(current["paths"])
    expected = {
        "/v1/recommend",
        "/v1/home-rails",
        "/v1/similar-books",
        "/v1/more-by-author",
        "/v1/semantic-search",
        "/v1/resolve-entities",
        "/v1/onboarding-grid",
        "/v1/parse-onboarding",
        "/v1/explain",
        "/v1/chat-turn",
        "/v1/book-insights/{book_id}",
        "/v1/analyze-user-review",
        "/v1/metrics-report",
        "/v1/catalog",
        "/v1/catalog/{book_id}",
        "/health",
        "/ready",
        "/metrics",
    }
    assert paths == expected
    assert "APIKeyHeader" in current["components"]["securitySchemes"]
    rec = current["paths"]["/v1/recommend"]["post"]
    assert rec["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith(
        "/RecResponse"
    )
    assert {"400", "401", "429", "503", "500"} <= set(rec["responses"])
