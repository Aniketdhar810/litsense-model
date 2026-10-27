"""Stage framework: skipping, fingerprints, atomic outputs and manifests (CLAUDE.md s7.0)."""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.common.config import REPO_ROOT
from ml.common.errors import StageError
from ml.common.manifest import read_manifest, sha256_file
from ml.common.stage import (
    RunPaths,
    StageContext,
    StageRegistry,
    code_files,
    raw_filename,
    run_all,
    run_stage,
    stage_status,
)


def write_raw(cfg: dict[str, Any], text: str, *, sample: bool = False) -> Path:
    root = Path(cfg["project"]["local_raw_root" if sample else "raw_root"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ratings.csv"
    path.write_text(text)
    return path


def pipeline(calls: list[str]) -> StageRegistry:
    """Two stages: raw CSV -> parquet -> (npy, results directory)."""
    reg = StageRegistry()

    @reg.stage(
        "t.first",
        inputs=["raw:ratings.csv"],
        outputs=["data:first.parquet"],
        config_keys=["data.min_pages"],
    )
    def first(ctx: StageContext) -> None:
        calls.append("t.first")
        rows = ctx.path("raw:ratings.csv").read_text().splitlines()[1:]
        if ctx.sample:
            rows = rows[: ctx.sample]
        frame = pd.DataFrame({"n": [len(rows)], "min_pages": [ctx.param("data.min_pages")]})
        frame.to_parquet(ctx.out("data:first.parquet"))
        ctx.count("rows_in", len(rows))

    @reg.stage(
        "t.second",
        inputs=["data:first.parquet"],
        outputs=["artifacts:emb.npy", "results:report/"],
    )
    def second(ctx: StageContext) -> None:
        calls.append("t.second")
        n = int(pd.read_parquet(ctx.path("data:first.parquet"))["n"].iloc[0])
        np.save(ctx.out("artifacts:emb.npy"), np.full((3, 4), n, dtype=np.float32))
        (ctx.out("results:report/") / "summary.json").write_text(json.dumps({"n": n}))

    return reg


@pytest.fixture
def cfg(tmp_config: dict[str, Any]) -> dict[str, Any]:
    write_raw(tmp_config, "user,item\nu1,a\nu2,b\n")
    return tmp_config


def test_run_writes_outputs_and_complete_manifest(cfg: dict[str, Any], tmp_path: Path) -> None:
    calls: list[str] = []
    reg = pipeline(calls)
    assert run_all(cfg, registry=reg) == ["t.first", "t.second"]

    data = tmp_path / "drive" / "data" / "v0" / "first.parquet"
    emb = tmp_path / "drive" / "artifacts" / "v0" / "emb.npy"
    report = tmp_path / "drive" / "results" / "v0" / "report" / "summary.json"
    assert data.exists() and emb.exists() and report.exists()
    assert not emb.with_name("emb.npy.npy").exists()  # staging keeps np.save's extension intact

    m1 = read_manifest(tmp_path / "drive" / "manifests" / "v0" / "t.first.json")
    assert m1["status"] == "complete"
    assert m1["files"]["data:first.parquet"] == {
        "sha256": sha256_file(data),
        "bytes": data.stat().st_size,
        "rows": 1,
    }
    assert m1["config"] == {"data.min_pages": 40}
    assert m1["notes"] == {"counts": {"rows_in": 2}}
    assert m1["inputs"]["raw:ratings.csv"].startswith("size=")
    assert "tests/ml/common/test_stage.py" in m1["code"]
    assert m1["sample"] is None and m1["data_version"] == "v0"

    m2 = read_manifest(tmp_path / "drive" / "manifests" / "v0" / "t.second.json")
    assert m2["files"]["artifacts:emb.npy"]["shape"] == [3, 4]
    assert m2["files"]["artifacts:emb.npy"]["dtype"] == "float32"
    assert m2["files"]["results:report/"]["n_files"] == 1
    assert m2["inputs"] == {"data:first.parquet": m1["files"]["data:first.parquet"]["sha256"]}
    assert not (tmp_path / "drive" / ".staging" / "v0" / "t.first").exists()


def test_second_run_skips(cfg: dict[str, Any]) -> None:
    calls: list[str] = []
    reg = pipeline(calls)
    run_all(cfg, registry=reg)
    assert run_all(cfg, registry=reg) == []
    assert run_stage("t.first", cfg, registry=reg) is False
    assert calls == ["t.first", "t.second"]


def test_force_reruns(cfg: dict[str, Any]) -> None:
    calls: list[str] = []
    reg = pipeline(calls)
    run_all(cfg, registry=reg)
    assert run_stage("t.first", cfg, registry=reg, force=True) is True
    assert calls == ["t.first", "t.second", "t.first"]


def test_declared_config_change_reruns_but_undeclared_does_not(cfg: dict[str, Any]) -> None:
    calls: list[str] = []
    reg = pipeline(calls)
    run_all(cfg, registry=reg)
    cfg["data"]["max_review_chars"] = 1  # not a config_key of either stage
    assert run_all(cfg, registry=reg) == []
    cfg["data"]["min_pages"] = 41  # declared by t.first; changes its output, so t.second reruns
    assert run_all(cfg, registry=reg) == ["t.first", "t.second"]


def test_downstream_reruns_only_when_upstream_output_changes(cfg: dict[str, Any]) -> None:
    calls: list[str] = []
    reg = pipeline(calls)
    run_all(cfg, registry=reg)
    raw = write_raw(cfg, "user,item\nu1,a\nu2,b\n")  # same bytes, new mtime
    os.utime(raw, ns=(raw.stat().st_atime_ns, raw.stat().st_mtime_ns + 10**9))
    assert run_all(cfg, registry=reg) == ["t.first"]  # identical parquet: t.second stays current
    write_raw(cfg, "user,item\nu1,a\n")
    assert run_all(cfg, registry=reg) == ["t.first", "t.second"]


def test_failed_run_leaves_no_manifest_and_no_outputs(cfg: dict[str, Any], tmp_path: Path) -> None:
    reg = StageRegistry()

    @reg.stage("t.fail", inputs=["raw:ratings.csv"], outputs=["data:out.parquet"])
    def fail(ctx: StageContext) -> None:
        pd.DataFrame({"a": [1]}).to_parquet(ctx.out("data:out.parquet"))
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_stage("t.fail", cfg, registry=reg)
    assert not (tmp_path / "drive" / "data" / "v0" / "out.parquet").exists()
    assert not (tmp_path / "drive" / "manifests" / "v0" / "t.fail.json").exists()
    assert not (tmp_path / "drive" / ".staging" / "v0" / "t.fail").exists()
    assert stage_status("t.fail", cfg, registry=reg) == "never run"


def test_failed_rerun_invalidates_previous_manifest(cfg: dict[str, Any], tmp_path: Path) -> None:
    calls: list[str] = []
    reg = pipeline(calls)
    run_stage("t.first", cfg, registry=reg)
    manifest = tmp_path / "drive" / "manifests" / "v0" / "t.first.json"
    assert manifest.exists()
    Path(cfg["project"]["raw_root"], "ratings.csv").unlink()
    with pytest.raises(StageError, match="missing input"):
        run_stage("t.first", cfg, registry=reg)
    assert manifest.exists()  # nothing ran, so the old complete manifest is untouched

    def explode(ctx: StageContext) -> None:
        raise RuntimeError("boom")

    reg2 = StageRegistry()
    reg2.stage("t.first", outputs=["data:first.parquet"])(explode)
    with pytest.raises(RuntimeError):
        run_stage("t.first", cfg, registry=reg2, force=True)
    assert not manifest.exists()  # a started rerun withdraws the old manifest first


def test_missing_declared_output_fails(cfg: dict[str, Any], tmp_path: Path) -> None:
    reg = StageRegistry()
    reg.stage("t.lazy", outputs=["data:a.parquet", "data:b.parquet"])(
        lambda ctx: pd.DataFrame({"a": [1]}).to_parquet(ctx.out("data:a.parquet"))
    )
    with pytest.raises(StageError, match="did not write declared outputs"):
        run_stage("t.lazy", cfg, registry=reg)
    assert not (tmp_path / "drive" / "data" / "v0" / "a.parquet").exists()
    assert not (tmp_path / "drive" / "manifests" / "v0" / "t.lazy.json").exists()


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (lambda ctx: ctx.out("data:other.parquet"), "undeclared output"),
        (lambda ctx: ctx.path("raw:meta_books"), "undeclared input"),
        (lambda ctx: ctx.param("data.kcore"), "not in its config_keys"),
    ],
)
def test_undeclared_access_raises(cfg: dict[str, Any], body: Any, message: str) -> None:
    reg = StageRegistry()
    reg.stage("t.sneaky", outputs=["data:a.parquet"], config_keys=["data.min_pages"])(body)
    with pytest.raises(StageError, match=message):
        run_stage("t.sneaky", cfg, registry=reg)


def test_param_allows_children_of_declared_keys(cfg: dict[str, Any]) -> None:
    reg = StageRegistry()
    seen: list[Any] = []

    @reg.stage("t.kcore", outputs=["data:a.parquet"], config_keys=["data.kcore"])
    def kcore(ctx: StageContext) -> None:
        seen.append(ctx.param("data.kcore.user"))
        pd.DataFrame({"a": [1]}).to_parquet(ctx.out("data:a.parquet"))

    run_stage("t.kcore", cfg, registry=reg)
    assert seen == [5]


def test_upstream_must_complete_first(cfg: dict[str, Any]) -> None:
    reg = pipeline([])
    with pytest.raises(StageError, match="needs stage 't.first' to run first"):
        run_stage("t.second", cfg, registry=reg)
    assert stage_status("t.second", cfg, registry=reg).startswith("blocked")


def test_output_added_after_producer_ran_blocks_cleanly(cfg: dict[str, Any]) -> None:
    run_stage("t.first", cfg, registry=pipeline([]))
    reg = StageRegistry()  # t.first now also declares an output its old manifest lacks
    reg.stage("t.first", outputs=["data:first.parquet", "data:extra.txt"])(lambda ctx: None)
    reg.stage("t.next", inputs=["data:first.parquet", "data:extra.txt"], outputs=["data:n.txt"])(
        lambda ctx: None
    )
    Path(RunPaths.from_config(cfg, None).resolve("data:extra.txt")).write_text("stray")
    assert stage_status("t.next", cfg, registry=reg) == (
        "blocked: input 'data:extra.txt' needs stage 't.first' to run again"
    )


def test_sample_mode_uses_local_roots(tmp_config: dict[str, Any], tmp_path: Path) -> None:
    write_raw(tmp_config, "user,item\nu1,a\nu2,b\nu3,c\n", sample=True)
    calls: list[str] = []
    reg = pipeline(calls)
    run_stage("t.first", tmp_config, registry=reg, sample=2)
    out = tmp_path / "local" / "data" / "v0" / "first.parquet"
    assert pd.read_parquet(out)["n"].iloc[0] == 2
    assert not (tmp_path / "drive").exists()
    assert read_manifest(tmp_path / "local" / "manifests" / "v0" / "t.first.json")["sample"] == 2
    assert run_stage("t.first", tmp_config, registry=reg, sample=2) is False
    assert run_stage("t.first", tmp_config, registry=reg, sample=3) is True  # N is fingerprinted


def test_stage_status_lifecycle(cfg: dict[str, Any]) -> None:
    reg = pipeline([])
    assert stage_status("t.first", cfg, registry=reg) == "never run"
    run_stage("t.first", cfg, registry=reg)
    assert stage_status("t.first", cfg, registry=reg) == "complete"
    cfg["data"]["min_pages"] = 99
    assert stage_status("t.first", cfg, registry=reg) == "stale"
    Path(cfg["project"]["raw_root"], "ratings.csv").unlink()
    assert stage_status("t.first", cfg, registry=reg).startswith("blocked: missing input")


def test_ordered_follows_dependencies_and_until_stops(cfg: dict[str, Any]) -> None:
    calls: list[str] = []
    reg = StageRegistry()

    def writer(name: str, out: str) -> Any:
        def fn(ctx: StageContext) -> None:
            calls.append(name)
            ctx.out(out).write_text(name)

        return fn

    reg.stage("c", inputs=["data:b.txt"], outputs=["data:c.txt"])(writer("c", "data:c.txt"))
    reg.stage("b", inputs=["data:a.txt"], outputs=["data:b.txt"])(writer("b", "data:b.txt"))
    reg.stage("a", outputs=["data:a.txt"])(writer("a", "data:a.txt"))
    assert [s.name for s in reg.ordered()] == ["a", "b", "c"]
    assert run_all(cfg, registry=reg, until="b") == ["a", "b"]
    assert calls == ["a", "b"]
    with pytest.raises(StageError, match="unknown stage"):
        run_all(cfg, registry=reg, until="z")


def test_registry_rejects_bad_declarations() -> None:
    reg = StageRegistry()
    reg.stage("a", outputs=["data:a.txt"])(lambda ctx: None)
    with pytest.raises(StageError, match="registered twice"):
        reg.stage("a", outputs=["data:z.txt"])(lambda ctx: None)
    with pytest.raises(StageError, match="already produced by 'a'"):
        reg.stage("b", outputs=["data:a.txt"])(lambda ctx: None)
    with pytest.raises(StageError, match="declares no outputs"):
        reg.stage("c")(lambda ctx: None)
    with pytest.raises(StageError, match="bad path spec"):
        reg.stage("d", outputs=["drive:x.txt"])(lambda ctx: None)
    reg.stage("e", inputs=["data:f.txt"], outputs=["data:e.txt"])(lambda ctx: None)
    reg.stage("f", inputs=["data:e.txt"], outputs=["data:f.txt"])(lambda ctx: None)
    with pytest.raises(StageError, match="dependency cycle"):
        reg.ordered()


def test_work_dir_survives_failure_until_fingerprint_changes(cfg: dict[str, Any]) -> None:
    reg = StageRegistry()
    attempts: list[bool] = []

    @reg.stage("t.stream", outputs=["data:a.txt"], config_keys=["data.min_pages"])
    def stream(ctx: StageContext) -> None:
        part = ctx.work_dir / "part_0000.txt"
        attempts.append(part.exists())
        part.write_text("checkpoint")
        if len(attempts) in (1, 3):
            raise RuntimeError("disconnected")
        ctx.out("data:a.txt").write_text("done")

    with pytest.raises(RuntimeError):
        run_stage("t.stream", cfg, registry=reg)
    run_stage("t.stream", cfg, registry=reg)  # resumes from the checkpoint
    work = RunPaths.from_config(cfg, None).work_dir("t.stream")
    assert not work.exists()  # removed once the stage completes
    cfg["data"]["min_pages"] = 7
    with pytest.raises(RuntimeError):
        run_stage("t.stream", cfg, registry=reg)
    cfg["data"]["min_pages"] = 8  # another fingerprint: the old checkpoint must not be reused
    run_stage("t.stream", cfg, registry=reg)
    assert attempts == [False, True, False, False]


def test_code_change_reruns(cfg: dict[str, Any], tmp_path: Path, monkeypatch: Any) -> None:
    module_dir = tmp_path / "mods"
    module_dir.mkdir()
    source = module_dir / "tmp_stage_mod.py"
    source.write_text(
        "def build(ctx):\n    ctx.out('data:a.txt').write_text('x')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(module_dir))
    module = importlib.import_module("tmp_stage_mod")
    reg = StageRegistry()
    reg.stage("t.code", outputs=["data:a.txt"])(module.build)
    run_stage("t.code", cfg, registry=reg)
    assert stage_status("t.code", cfg, registry=reg) == "complete"
    source.write_text(source.read_text() + "# edited\n", encoding="utf-8")
    assert stage_status("t.code", cfg, registry=reg) == "stale"
    assert run_stage("t.code", cfg, registry=reg) is True


def test_code_files_follow_project_imports_but_skip_plumbing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    (tmp_path / "tmp_dep_mod.py").write_text(
        "from ml.common.config import config_get\nimport ml.common.stage\n\n"
        "def build(ctx):\n    return config_get\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("tmp_dep_mod")
    in_repo = [p for p in code_files(module.build) if p.is_relative_to(REPO_ROOT)]
    files = {p.relative_to(REPO_ROOT).as_posix() for p in in_repo}
    assert {"ml/common/config.py", "ml/common/errors.py"} <= files  # config imports errors
    assert "ml/common/stage.py" not in files and "ml/common/manifest.py" not in files


def test_url_inputs_resolve_to_urls_or_local_heads(tmp_config: dict[str, Any]) -> None:
    cfg = tmp_config
    meta_url = "https://example.invalid/data/meta_Books.jsonl.gz"
    cfg["sources"]["meta_books"] = meta_url
    gr_url = f"{cfg['sources']['goodreads_base']}/goodreads_books.json.gz"
    reg = StageRegistry()
    seen: list[tuple[str, str, str]] = []

    @reg.stage(
        "t.url", inputs=["url:meta_books", "url:goodreads_books.json.gz"], outputs=["data:a.txt"]
    )
    def read(ctx: StageContext) -> None:
        seen.append(
            (
                ctx.source("url:meta_books"),
                ctx.source("url:meta_books", full=True),
                ctx.source("url:goodreads_books.json.gz"),
            )
        )
        ctx.out("data:a.txt").write_text("x")

    run_stage("t.url", cfg, registry=reg)
    assert seen[-1] == (meta_url, meta_url, gr_url)

    local = Path(cfg["project"]["local_raw_root"])
    local.mkdir(parents=True)
    (local / "meta_Books.jsonl.gz").write_text("head")
    run_stage("t.url", cfg, registry=reg, sample=10)
    assert seen[-1] == (str(local / "meta_Books.jsonl.gz"), meta_url, gr_url)  # no local GR copy
    assert stage_status("t.url", cfg, registry=reg, sample=10) == "complete"
    (local / "meta_Books.jsonl.gz").write_text(
        "a longer head"
    )  # the local copy is part of identity
    assert stage_status("t.url", cfg, registry=reg, sample=10) == "stale"

    assert stage_status("t.url", cfg, registry=reg) == "complete"
    cfg["sources"]["meta_books"] = "https://example.invalid/v2/meta_Books.jsonl.gz"
    assert stage_status("t.url", cfg, registry=reg) == "stale"


def test_url_specs_are_input_only_and_have_no_path(tmp_config: dict[str, Any]) -> None:
    reg = StageRegistry()
    with pytest.raises(StageError, match="url is input-only"):
        reg.stage("t.out", outputs=["url:meta_books"])(lambda ctx: None)
    reg.stage("t.path", inputs=["url:meta_books"], outputs=["data:a.txt"])(
        lambda ctx: ctx.path("url:meta_books")
    )
    with pytest.raises(StageError, match="read it with ctx.source"):
        run_stage("t.path", tmp_config, registry=reg)
    reg.stage("t.unknown", inputs=["url:nowhere"], outputs=["data:b.txt"])(lambda ctx: None)
    with pytest.raises(StageError, match="no URL for source 'nowhere'"):
        run_stage("t.unknown", tmp_config, registry=reg)


def test_raw_filename_uses_url_basename(repo_config: dict[str, Any]) -> None:
    sources = repo_config["sources"]
    assert raw_filename(sources, "ratings_5core") == "Books.csv.gz"
    assert raw_filename(sources, "meta_books") == "meta_Books.jsonl.gz"
    assert raw_filename(sources, "reviews_books") == "Books.jsonl.gz"
    assert raw_filename(sources, "goodreads_books.json.gz") == "goodreads_books.json.gz"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml.stages", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_list_and_errors() -> None:
    listed = run_cli("list")
    assert listed.returncode == 0, listed.stderr
    unknown = run_cli("run", "no.such.stage")
    assert unknown.returncode == 2
    assert "unknown stage" in unknown.stderr
    misuse = run_cli("run", "no.such.stage", "--until", "x")
    assert misuse.returncode == 2


def test_a_truncated_output_makes_the_stage_stale(tmp_path: Path) -> None:
    """Google Drive can keep an output's name but lose its bytes (D42)."""
    registry = StageRegistry()

    @registry.stage("demo.write", outputs=["data:out.txt"])
    def write(ctx: StageContext) -> None:
        ctx.out("data:out.txt").write_text("hello", encoding="utf-8")

    cfg = {"project": {"seed": 1, "data_version": "v0", "drive_root": str(tmp_path / "d"),
                       "raw_root": str(tmp_path / "r"), "local_root": str(tmp_path / "l"),
                       "local_raw_root": str(tmp_path / "lr")}, "sources": {}}  # fmt: skip
    assert run_stage("demo.write", cfg, registry=registry)
    assert stage_status("demo.write", cfg, registry=registry) == "complete"
    out = RunPaths.from_config(cfg, None).resolve("data:out.txt")
    out.write_bytes(b"")
    assert stage_status("demo.write", cfg, registry=registry) == "stale"
    assert run_stage("demo.write", cfg, registry=registry)
    assert out.read_text(encoding="utf-8") == "hello"
