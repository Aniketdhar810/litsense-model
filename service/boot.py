"""The default loader: find or download the bundle, configure the LLM client, load the store (D78).

Order matters: ``configure_llm`` must run before ``load_store`` so the event hook and the
writable cache path win over the bundle's own ``llm`` slice (contract section 1b, D68).
"""

import json
import logging
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import litsense_core as core
from litsense_core.llm.config import find_config
from service.config import Settings
from service.logs import log

LOG = logging.getLogger("litsense.boot")
EventHook = Callable[[dict[str, Any]], None]


class BootError(RuntimeError):
    """The bundle cannot be found or fetched."""


@dataclass
class BootResult:
    store: Any  # litsense_core.store.ArtifactStore
    timings: dict[str, float] = field(default_factory=dict)


Loader = Callable[[Settings, EventHook], BootResult]


def locate_bundle(bundle_dir: str | Path) -> Path | None:
    """The folder holding ``manifest.json``: ``bundle_dir`` itself or its ``serve_bundle/``."""
    root = Path(bundle_dir)
    for candidate in (root, root / "serve_bundle"):
        if (candidate / "manifest.json").is_file():
            return candidate
    return None


def fetch_bundle(url: str, bundle_dir: str | Path) -> Path:
    """Download and extract into a staging folder, then move ``serve_bundle/`` into place, so an
    interrupted download never leaves a half bundle that a restart would try to load."""
    root = Path(bundle_dir)
    staging = root / ".staging"
    shutil.rmtree(staging, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    extracted = Path(core.download_bundle(url, str(staging)))
    target = root / "serve_bundle"
    shutil.rmtree(target, ignore_errors=True)
    os.replace(extracted, target)
    shutil.rmtree(staging, ignore_errors=True)  # also drops the 94 MB tar
    return target


def llm_config(manifest: dict[str, Any], settings: Settings) -> dict[str, Any]:
    """The bundle's ``llm`` slice (or the repo YAML for older bundles) with a writable cache
    path and, for tests and demos, a provider override. The API key stays in the environment."""
    cfg = find_config((manifest.get("config") or {}).get("llm") or None).to_dict()
    cfg["cache_path"] = settings.llm_cache_path or str(Path(settings.bundle_dir) / "llm.sqlite")
    if settings.llm_provider:
        cfg["provider"] = settings.llm_provider
        if settings.llm_provider == "fake":
            cfg["model"] = cfg["model_quality"] = "fake"
    return cfg


def _set_onnx_threads(store: Any, folder: Path, threads: int) -> None:
    """Rebuild the query encoder when ONNX_THREADS differs from the bundle's ``onnx_threads``.
    Costs a second ONNX session for a moment at boot; leave ONNX_THREADS unset on 512 MB."""
    current = int((store.config.get("serving") or {}).get("onnx_threads", 1))
    if store.encoder is None or threads == current:
        return
    from litsense_core.encoder import MiniLMEncoder

    max_len = int((store.manifest.get("onnx") or {}).get("max_seq_length", 256))
    store.encoder = MiniLMEncoder(
        folder / "minilm.onnx", folder / "tokenizer.json", max_len, threads
    )


def load_bundle(settings: Settings, on_event: EventHook) -> BootResult:
    timings: dict[str, float] = {}
    folder = locate_bundle(settings.bundle_dir)
    if folder is None:
        if not settings.bundle_url:
            raise BootError(
                f"no bundle in BUNDLE_DIR={settings.bundle_dir} and BUNDLE_URL is not set"
            )
        parts = urlsplit(settings.bundle_url)
        log(LOG, logging.INFO, "downloading bundle", host=parts.netloc, path=parts.path)
        started = time.perf_counter()
        folder = fetch_bundle(settings.bundle_url, settings.bundle_dir)
        timings["download_s"] = round(time.perf_counter() - started, 3)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    core.configure_llm(llm_config(manifest, settings), on_event=on_event)
    started = time.perf_counter()
    store = core.load_store(str(folder))
    if settings.onnx_threads:
        _set_onnx_threads(store, folder, settings.onnx_threads)
    timings["load_s"] = round(time.perf_counter() - started, 3)
    return BootResult(store, timings)
