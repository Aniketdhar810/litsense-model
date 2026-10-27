"""Stage framework (CLAUDE.md s7.0): declared inputs and outputs, fingerprints, atomic writes.

A stage is a function registered with ``@stage(name, inputs, outputs, config_keys)``. Before it
runs, the runner computes a fingerprint from

* the declared config values, the seed, the data version and ``--sample``;
* each input's identity: the sha256 its producing stage recorded, or size+mtime for files no
  stage produces;
* the source of the stage's module and, transitively, of every ``ml``/``litsense_core`` module it
  imports (framework plumbing excluded, see ``docs/decisions.md``).

If a complete manifest with that fingerprint exists and every output is present, the stage is
skipped. Otherwise the stage writes into a staging directory (``ctx.out``), the runner moves the
outputs into place only after the function returns, and the manifest is written last with
``status: "complete"``. A failed run leaves no manifest, so nothing downstream trusts it.

Path specs are ``namespace:relative/path``; a trailing ``/`` marks a directory output.

* ``url:<key>`` (inputs only): a source streamed from its URL and never stored. ``<key>`` is a
  ``sources`` entry or a Goodreads file name (``sources.goodreads_base`` + ``/`` + name). With
  ``--sample``, ``ctx.source`` returns the same-named local copy under ``project.local_raw_root``
  when it exists; its identity is the URL plus that copy's size and mtime.
* ``raw:<key>``: a raw download. ``<key>`` is a ``sources`` entry (file = URL basename) or a
  file name, under ``project.raw_root`` (``project.local_raw_root`` with ``--sample``).
* ``data:<rel>``, ``artifacts:<rel>``, ``results:<rel>``: the file
  ``<root>/<namespace>/<data_version>/<rel>``, where ``<root>`` is ``project.drive_root``
  (``project.local_root`` with ``--sample``).
"""

import ast
import importlib.util
import inspect
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import numpy as np

from ml.common.config import REPO_ROOT, config_get
from ml.common.errors import StageError
from ml.common.log import get_logger
from ml.common.manifest import (
    describe_path,
    fingerprint,
    git_sha,
    is_complete,
    read_manifest,
    write_json_atomic,
)
from ml.common.seed import set_global_seed

NAMESPACES = ("url", "raw", "data", "artifacts", "results")
_PROJECT_PACKAGES = frozenset({"ml", "litsense_core"})
# Plumbing that cannot change what a stage writes; hashing it would rerun everything on any edit.
_FRAMEWORK_MODULES = frozenset(
    {"ml.common.stage", "ml.common.manifest", "ml.common.log", "ml.stages"}
)


@dataclass(frozen=True)
class Stage:
    """A registered stage: its function and what it reads and writes."""

    name: str
    fn: Callable[["StageContext"], None]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    config_keys: tuple[str, ...]


class StageRegistry:
    """Stages by name, in registration order. Each output spec has exactly one producer."""

    def __init__(self) -> None:
        self._stages: dict[str, Stage] = {}

    def stage(
        self,
        name: str,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        config_keys: Sequence[str] = (),
    ) -> Callable[[Callable[["StageContext"], None]], Callable[["StageContext"], None]]:
        """Decorator registering the function as stage ``name``."""

        def register(fn: Callable[["StageContext"], None]) -> Callable[["StageContext"], None]:
            self.add(Stage(name, fn, tuple(inputs), tuple(outputs), tuple(config_keys)))
            return fn

        return register

    def add(self, st: Stage) -> None:
        """Register ``st``; raises StageError on duplicate names, bad specs or shared outputs."""
        if st.name in self._stages:
            raise StageError(f"stage {st.name!r} is registered twice")
        if not st.outputs:
            raise StageError(f"stage {st.name!r} declares no outputs")
        for spec in (*st.inputs, *st.outputs):
            split_spec(spec)
        for out in st.outputs:
            if split_spec(out)[0] == "url":
                raise StageError(
                    f"stage {st.name!r} declares url output {out!r}; url is input-only"
                )
            other = self.producer(out)
            if other is not None:
                raise StageError(
                    f"output {out!r} of {st.name!r} is already produced by {other.name!r}"
                )
        self._stages[st.name] = st

    def get(self, name: str) -> Stage:
        """Return the stage called ``name``."""
        if name not in self._stages:
            raise StageError(f"unknown stage {name!r}; known: {sorted(self._stages)}")
        return self._stages[name]

    def producer(self, spec: str) -> Stage | None:
        """Return the stage that declares ``spec`` as an output, if any."""
        return next((s for s in self._stages.values() if spec in s.outputs), None)

    def ordered(self) -> list[Stage]:
        """Stages in dependency order; registration order breaks ties."""
        deps = {
            s.name: {p.name for spec in s.inputs if (p := self.producer(spec)) and p is not s}
            for s in self._stages.values()
        }
        done: list[str] = []
        while len(done) < len(deps):
            ready = [n for n in deps if n not in done and deps[n] <= set(done)]
            if not ready:
                raise StageError(f"dependency cycle among stages {sorted(set(deps) - set(done))}")
            done.append(ready[0])
        return [self._stages[n] for n in done]


REGISTRY = StageRegistry()
stage = REGISTRY.stage


def split_spec(spec: str) -> tuple[str, str]:
    """Split ``"data:books.parquet"`` into ``("data", "books.parquet")``, validating both parts."""
    ns, sep, rel = spec.partition(":")
    if not sep or ns not in NAMESPACES or not rel.strip("/"):
        raise StageError(f"bad path spec {spec!r}; expected one of {NAMESPACES} + ':<path>'")
    return ns, rel


def source_url(sources: Mapping[str, Any], key: str) -> str:
    """URL of streamed source ``key``: a ``sources`` URL entry or a Goodreads file name."""
    value = sources.get(key)
    if isinstance(value, str) and "://" in value:
        return value
    if key in sources.get("goodreads_files", []):
        return f"{str(sources['goodreads_base']).rstrip('/')}/{key}"
    raise StageError(f"no URL for source {key!r} in config sources")


def raw_filename(sources: Mapping[str, Any], key: str) -> str:
    """File name of raw input ``key``: the URL basename for a ``sources`` entry, else ``key``."""
    value = sources.get(key)
    if isinstance(value, str) and "://" in value:
        return PurePosixPath(urlparse(value).path).name
    return key


@dataclass(frozen=True)
class RunPaths:
    """Where one run reads raw files and writes everything else."""

    raw_dir: Path
    out_root: Path
    version: str
    sources: Mapping[str, Any]
    sampling: bool = False

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], sample: int | None) -> "RunPaths":
        """Full runs use the Colab roots; ``--sample`` runs use the local ones."""
        project = cfg["project"]
        raw_key, out_key = (
            ("raw_root", "drive_root") if sample is None else ("local_raw_root", "local_root")
        )
        return cls(
            _repo_path(project[raw_key]),
            _repo_path(project[out_key]),
            str(project["data_version"]),
            cfg["sources"],
            sample is not None,
        )

    def resolve(self, spec: str) -> Path:
        """Final location of ``spec``."""
        ns, rel = split_spec(spec)
        if ns == "url":
            raise StageError(f"{spec!r} is streamed, not stored; read it with ctx.source")
        if ns == "raw":
            return self.raw_dir / raw_filename(self.sources, rel)
        return self.out_root / ns / self.version / rel

    def source(self, spec: str, *, full: bool = False) -> str:
        """URL of a ``url:`` input, or its local copy in a ``--sample`` run (unless ``full``)."""
        ns, key = split_spec(spec)
        if ns != "url":
            raise StageError(f"{spec!r} is not a url input")
        url = source_url(self.sources, key)
        local = self.local_copy(url)
        return str(local) if self.sampling and not full and local.exists() else url

    def local_copy(self, url: str) -> Path:
        """Where a ``--sample`` run looks for a local head of ``url``."""
        return self.raw_dir / PurePosixPath(urlparse(url).path).name

    def staged(self, stage_name: str, spec: str) -> Path:
        """Where ``ctx.out`` writes ``spec`` before it is moved into place."""
        ns, rel = split_spec(spec)
        return self.staging_dir(stage_name, ns) / ns / rel

    def staging_dir(self, stage_name: str, ns: str) -> Path:
        """Per-stage staging directory on the same filesystem as namespace ``ns``."""
        root = (
            self.raw_dir / ".staging" if ns == "raw" else self.out_root / ".staging" / self.version
        )
        return root / stage_name

    def manifest(self, stage_name: str) -> Path:
        """Manifest file of ``stage_name``."""
        return self.out_root / "manifests" / self.version / f"{stage_name}.json"

    def work_dir(self, stage_name: str) -> Path:
        """Checkpoint directory of ``stage_name``."""
        return self.out_root / "work" / self.version / stage_name


class StageContext:
    """What a running stage may touch: declared inputs, staged outputs, declared config."""

    def __init__(
        self, st: Stage, cfg: Mapping[str, Any], paths: RunPaths, sample: int | None, fp: str
    ) -> None:
        self.stage = st
        self.sample = sample  # read only the first N lines of each raw file when set
        self.seed = int(cfg["project"]["seed"])
        self.rng = np.random.default_rng(self.seed)
        self.log = get_logger(st.name)
        self.notes: dict[str, Any] = {}  # copied into the manifest
        self.cutoff: int | None = None  # ms; set by stages whose outputs depend on a cutoff
        self._cfg = cfg
        self._paths = paths
        self._fp = fp
        self._work_dir: Path | None = None

    def param(self, key: str) -> Any:
        """Config value at a dotted key covered by the stage's ``config_keys``."""
        if not any(key == k or key.startswith(k + ".") for k in self.stage.config_keys):
            raise StageError(
                f"stage {self.stage.name!r} read config key {key!r} not in its config_keys"
            )
        return config_get(self._cfg, key)

    def path(self, spec: str) -> Path:
        """Location of a declared stored input."""
        if spec not in self.stage.inputs:
            raise StageError(f"stage {self.stage.name!r} read undeclared input {spec!r}")
        return self._paths.resolve(spec)

    def source(self, spec: str, *, full: bool = False) -> str:
        """URL (or, with ``--sample``, local head) of a declared ``url:`` input. ``full=True``
        always returns the URL, for stages that must read the whole file even in sample runs."""
        if spec not in self.stage.inputs:
            raise StageError(f"stage {self.stage.name!r} read undeclared input {spec!r}")
        return self._paths.source(spec, full=full)

    def out(self, spec: str) -> Path:
        """Staging location to write a declared output to (parent directories created)."""
        if spec not in self.stage.outputs:
            raise StageError(f"stage {self.stage.name!r} wrote undeclared output {spec!r}")
        path = self._paths.staged(self.stage.name, spec)
        (path if spec.endswith("/") else path.parent).mkdir(parents=True, exist_ok=True)
        return path

    @property
    def work_dir(self) -> Path:
        """Checkpoint directory for long streams. It survives a failed run, is wiped when the
        fingerprint changes (checkpoints from other code or config are never reused) and is
        deleted once the stage completes."""
        if self._work_dir is None:
            d = self._paths.work_dir(self.stage.name)
            marker = d / ".fingerprint"
            if d.exists() and (not marker.exists() or marker.read_text() != self._fp):
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            marker.write_text(self._fp)
            self._work_dir = d
        return self._work_dir

    def count(self, label: str, n: int) -> None:
        """Record a row count in the manifest notes (feeds the attrition table) and log it."""
        self.notes.setdefault("counts", {})[label] = int(n)
        self.log.info("%s: %d", label, n)

    def upstream_notes(self, stage_name: str) -> dict[str, Any]:
        """The notes of an upstream stage's complete manifest (its row counts, for attrition
        tables). Declare one of that stage's outputs as an input so its changes rerun this one."""
        manifest = read_manifest(self._paths.manifest(stage_name))
        if not is_complete(manifest):
            raise StageError(f"stage {stage_name!r} has no complete manifest")
        return dict(manifest.get("notes", {}))


def compute_fingerprint(
    st: Stage,
    cfg: Mapping[str, Any],
    paths: RunPaths,
    registry: StageRegistry,
    sample: int | None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(fingerprint, material)`` for ``st``; raises StageError if an input is missing."""
    material = {
        "stage": st.name,
        "data_version": paths.version,
        "sample": sample,
        "seed": cfg["project"]["seed"],
        "config": {k: config_get(cfg, k) for k in st.config_keys},
        "inputs": {spec: _input_id(spec, paths, registry) for spec in st.inputs},
        "code": {_display_path(p): _source_hash(p) for p in code_files(st.fn)},
    }
    return fingerprint(material), material


def stage_status(
    name: str,
    cfg: Mapping[str, Any],
    *,
    registry: StageRegistry = REGISTRY,
    sample: int | None = None,
) -> str:
    """``"complete"``, ``"stale"``, ``"never run"`` or ``"blocked: <reason>"``."""
    st = registry.get(name)
    paths = RunPaths.from_config(cfg, sample)
    try:
        fp, _ = compute_fingerprint(st, cfg, paths, registry, sample)
    except StageError as e:
        return f"blocked: {e}"
    if _is_current(st, paths, fp):
        return "complete"
    return "stale" if paths.manifest(name).exists() else "never run"


def run_stage(
    name: str,
    cfg: Mapping[str, Any],
    *,
    registry: StageRegistry = REGISTRY,
    sample: int | None = None,
    force: bool = False,
) -> bool:
    """Run one stage unless it is up to date (or ``force``). Returns True if it ran."""
    st = registry.get(name)
    paths = RunPaths.from_config(cfg, sample)
    log = get_logger(name)
    for spec in st.inputs:
        producer = registry.producer(spec)
        if (
            producer
            and stage_status(producer.name, cfg, registry=registry, sample=sample) == "stale"
        ):
            log.warning("upstream stage %r is stale; run it first or use `run all`", producer.name)
    fp, material = compute_fingerprint(st, cfg, paths, registry, sample)
    if not force and _is_current(st, paths, fp):
        log.info("up to date (%s)", fp[:12])
        return False

    paths.manifest(name).unlink(missing_ok=True)
    _clear_staging(st, paths)
    ctx = StageContext(st, cfg, paths, sample, fp)
    set_global_seed(ctx.seed)
    log.info("running (sample=%s, fingerprint %s)", sample, fp[:12])
    started = time.perf_counter()
    try:
        st.fn(ctx)
        missing = [o for o in st.outputs if not paths.staged(name, o).exists()]
        if missing:
            raise StageError(f"stage {name!r} did not write declared outputs {missing}")
        for out in st.outputs:
            _move_into_place(paths.staged(name, out), paths.resolve(out))
    finally:
        _clear_staging(st, paths)

    manifest = {
        "stage": name,
        "status": "complete",
        "fingerprint": fp,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO_ROOT),
        "data_version": paths.version,
        "sample": sample,
        "cutoff": ctx.cutoff,
        "duration_s": round(time.perf_counter() - started, 3),
        "config": material["config"],
        "inputs": material["inputs"],
        "code": material["code"],
        "files": {out: describe_path(paths.resolve(out)) for out in st.outputs},
        "notes": ctx.notes,
    }
    write_json_atomic(paths.manifest(name), manifest)
    shutil.rmtree(paths.work_dir(name), ignore_errors=True)
    log.info("complete in %.1fs", manifest["duration_s"])
    return True


def run_all(
    cfg: Mapping[str, Any],
    *,
    registry: StageRegistry = REGISTRY,
    until: str | None = None,
    sample: int | None = None,
    force: bool = False,
) -> list[str]:
    """Run stages in dependency order, stopping after ``until``. Returns the names that ran."""
    order = [s.name for s in registry.ordered()]
    if until is not None:
        registry.get(until)
        order = order[: order.index(until) + 1]
    return [n for n in order if run_stage(n, cfg, registry=registry, sample=sample, force=force)]


def code_files(fn: Callable[..., Any]) -> list[Path]:
    """Source files that can change what ``fn`` writes: its own module plus, transitively, every
    ``ml``/``litsense_core`` module it imports, minus the framework plumbing."""
    source = inspect.getsourcefile(fn)
    if source is None:
        raise StageError(f"cannot locate the source of {fn!r}")
    seen: set[Path] = set()
    queue = [(fn.__module__, Path(source).resolve())]
    while queue:
        module, path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in _project_imports(module, path):
            dep = None if name in _FRAMEWORK_MODULES else _module_source(name)
            if dep is not None and dep not in seen:
                queue.append((name, dep))
    return sorted(seen)


def _project_imports(module: str, path: Path) -> set[str]:
    """Dotted names of ml/litsense_core modules (or module attributes) imported by a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parent = ".".join(parts[: len(parts) - node.level + 1])
                base = f"{parent}.{base}" if base else parent
            names.add(base)
            names.update(f"{base}.{a.name}" for a in node.names)  # `from pkg import submodule`
    return {n for n in names if n.split(".")[0] in _PROJECT_PACKAGES}


def _module_source(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):  # the name is an attribute of a module, not a module
        return None
    if spec is None or not spec.origin or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin).resolve()


def _source_hash(path: Path) -> str:
    # Normalize line endings so a Windows checkout and Colab agree on the fingerprint.
    return sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _input_id(spec: str, paths: RunPaths, registry: StageRegistry) -> str:
    if split_spec(spec)[0] == "url":
        url = paths.source(spec, full=True)
        local = paths.local_copy(url)
        if paths.sampling and local.exists():
            stat = local.stat()
            return f"{url};local size={stat.st_size};mtime_ns={stat.st_mtime_ns}"
        return url
    producer = registry.producer(spec)
    location = paths.resolve(spec)
    if producer is not None:
        manifest = read_manifest(paths.manifest(producer.name))
        if not is_complete(manifest) or not location.exists():
            raise StageError(f"input {spec!r} needs stage {producer.name!r} to run first")
        entry = manifest.get("files", {}).get(spec)
        if entry is None:  # spec was added to the producer's outputs after its last run
            raise StageError(f"input {spec!r} needs stage {producer.name!r} to run again")
        return entry["sha256"]
    if not location.exists():
        raise StageError(f"missing input {spec!r} (expected at {location})")
    stat = location.stat()
    return f"size={stat.st_size};mtime_ns={stat.st_mtime_ns}"


def _is_current(st: Stage, paths: RunPaths, fp: str) -> bool:
    manifest = read_manifest(paths.manifest(st.name))
    return (
        is_complete(manifest)
        and manifest.get("fingerprint") == fp
        and all(_intact(paths.resolve(out), manifest["files"].get(out)) for out in st.outputs)
    )


def _intact(path: Path, entry: Mapping[str, Any] | None) -> bool:
    """The output exists with the size its manifest recorded. Google Drive can keep a file's name
    while losing its content (a 0-byte parquet after a session ended, D42), so existence alone is
    not enough; sizes are cheap to check, hashes are not."""
    if not path.exists() or entry is None:
        return False
    if path.is_dir() or "bytes" not in entry:
        return True
    return path.stat().st_size == entry["bytes"]


def _clear_staging(st: Stage, paths: RunPaths) -> None:
    for ns in {split_spec(out)[0] for out in st.outputs}:
        shutil.rmtree(paths.staging_dir(st.name, ns), ignore_errors=True)


def _move_into_place(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.exists():
        shutil.rmtree(dst)
    os.replace(src, dst)  # staging and destination share a filesystem


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.anchor else REPO_ROOT / path
