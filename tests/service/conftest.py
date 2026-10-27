"""Service fixtures. ``make_app`` boots the app over the hand-built 12-book store
(tests/litsense_core/fake_store.py) with the fake LLM provider: fast, no torch, no network.
``tiny_bundle_tar`` builds the real tiny pipeline bundle once per session for
test_bundle_boot.py."""

import itertools
import os
import shutil
import tarfile
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.llm.adapters.fake import FakeAdapter
from litsense_core.llm.client import set_client
from litsense_core.store import set_store
from service.app import create_app
from service.boot import BootResult, Loader
from service.config import Settings
from service.state import ServiceState
from tests.litsense_core.fake_store import build_store

KEY = "test-service-key"
AUTH = {"X-API-Key": KEY}
TS = 1_650_000_000_000
_N = itertools.count()


@dataclass
class Booted:
    app: FastAPI
    client: TestClient
    adapter: FakeAdapter

    @property
    def svc(self) -> ServiceState:
        return self.app.state.svc  # type: ignore[no-any-return]

    def post(self, path: str, body: Any, **kwargs: Any) -> Any:
        headers = {**AUTH, **kwargs.pop("headers", {})}
        return self.client.post(path, json=body, headers=headers, **kwargs)

    def get(self, path: str, **kwargs: Any) -> Any:
        headers = {**AUTH, **kwargs.pop("headers", {})}
        return self.client.get(path, headers=headers, **kwargs)


def fake_loader(
    root: Path, adapter: FakeAdapter, llm: bool = True, gate: threading.Event | None = None
) -> Loader:
    def load(settings: Settings, on_event: Callable[[dict[str, Any]], None]) -> BootResult:
        if gate is not None:
            assert gate.wait(30), "test gate never opened"
        n = next(_N)
        store = build_store(root / f"bundle{n}")
        set_store(store)
        cfg = LLMConfig.from_mapping(
            {
                "enabled": llm,
                "provider": "fake",
                "model": "fake",
                "cache_path": str(root / f"llm{n}.db"),
            }
        )
        set_client(LLMClient(cfg, on_event=on_event, adapter=adapter, sleep=lambda s: None))
        return BootResult(store, {"load_s": 0.001})

    return load


@pytest.fixture
def make_app(tmp_path: Path) -> Iterator[Callable[..., Booted]]:
    """``make_app(script=..., llm=True, gate=None, wait=True, loader=None, **settings)``."""
    opened: list[TestClient] = []

    def factory(
        script: dict[str, list[Any]] | None = None,
        llm: bool = True,
        gate: threading.Event | None = None,
        wait: bool = True,
        loader: Loader | None = None,
        **overrides: Any,
    ) -> Booted:
        adapter = FakeAdapter(script or {})
        values: dict[str, Any] = {
            "service_api_key": KEY,
            "bundle_dir": str(tmp_path),
            "rate_limit_rps": 0,
            "boot_backoff_s": 0,
            **overrides,
        }
        app = create_app(
            Settings(**values), loader=loader or fake_loader(tmp_path, adapter, llm, gate)
        )
        client = TestClient(app)
        client.__enter__()
        opened.append(client)
        if wait:
            assert app.state.svc.boot_done.wait(30), "boot did not finish"
            assert app.state.svc.ready, app.state.svc.load_error
        return Booted(app, client, adapter)

    yield factory
    for client in opened:
        client.__exit__(None, None, None)
    set_store(None)
    set_client(None)


@pytest.fixture(scope="session")
def tiny_bundle_tar(
    tmp_path_factory: pytest.TempPathFactory, repo_config: dict[str, Any], tiny_dataset: Path
) -> Path:
    """``serve_bundle.tar.gz`` of the real tiny pipeline bundle, built once per session. Set
    LITSENSE_KEEP_TINY_BUNDLE=<folder> to keep a copy (the load test in service/README.md)."""
    pytest.importorskip("torch")
    from ml.stages import load_stages

    # Stage order is registration order: register every stage before bundle_build imports
    # ml.nlp.onnx_export, or nlp.onnx lands before eval.val (tests/litsense_core does the same).
    load_stages()
    from tests.litsense_core.bundle_build import tiny_bundle_session

    root = tmp_path_factory.mktemp("service_bundle")
    with tiny_bundle_session(root, repo_config, tiny_dataset) as (_cfg, paths):
        tar = paths.resolve("artifacts:serve_bundle.tar.gz")
        with tarfile.open(tar) as t:
            assert "serve_bundle/manifest.json" in t.getnames()
        out = root / "release"
        out.mkdir()
        shutil.copy(tar, out / "serve_bundle.tar.gz")
    set_store(None)
    set_client(None)
    keep = os.environ.get("LITSENSE_KEEP_TINY_BUNDLE")
    if keep:
        Path(keep).mkdir(parents=True, exist_ok=True)
        shutil.copy(out / "serve_bundle.tar.gz", Path(keep) / "serve_bundle.tar.gz")
    return out / "serve_bundle.tar.gz"
