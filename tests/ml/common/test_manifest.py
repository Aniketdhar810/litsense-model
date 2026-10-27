"""Manifest helpers: hashes, file descriptions, fingerprints, atomic writes."""

import gzip
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.common.config import REPO_ROOT
from ml.common.manifest import (
    describe_path,
    fingerprint,
    git_sha,
    is_complete,
    read_manifest,
    sha256_dir,
    write_json_atomic,
)


def test_describe_known_formats(tmp_path: Path) -> None:
    pd.DataFrame({"a": range(5)}).to_parquet(tmp_path / "t.parquet")
    np.save(tmp_path / "e.npy", np.zeros((4, 3), dtype=np.float16))
    (tmp_path / "t.csv").write_text("a,b\n1,2\n3,4")  # no trailing newline
    with gzip.open(tmp_path / "r.jsonl.gz", "wt") as f:
        f.write('{"a": 1}\n{"a": 2}\n{"a": 3}\n')
    assert describe_path(tmp_path / "t.parquet")["rows"] == 5
    npy = describe_path(tmp_path / "e.npy")
    assert npy["shape"] == [4, 3] and npy["dtype"] == "float16"
    assert describe_path(tmp_path / "t.csv")["rows"] == 2
    assert describe_path(tmp_path / "r.jsonl.gz")["rows"] == 3


def test_directory_hash_tracks_names_and_content(tmp_path: Path) -> None:
    d = tmp_path / "d"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "x.txt").write_text("1")
    before = sha256_dir(d)
    assert describe_path(d)["n_files"] == 1
    (d / "sub" / "x.txt").write_text("2")
    changed = sha256_dir(d)
    (d / "sub" / "x.txt").rename(d / "sub" / "y.txt")
    assert len({before, changed, sha256_dir(d)}) == 3


def test_fingerprint_ignores_key_order() -> None:
    assert fingerprint({"a": 1, "b": [1, 2]}) == fingerprint({"b": [1, 2], "a": 1})
    assert fingerprint({"a": 1}) != fingerprint({"a": 2})


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "m" / "stage.json"
    assert read_manifest(path) is None and not is_complete(None)
    write_json_atomic(path, {"status": "complete", "files": {}})
    assert is_complete(read_manifest(path))
    assert not path.with_name("stage.json.tmp").exists()
    write_json_atomic(path, {"status": "running"})
    assert not is_complete(read_manifest(path))


def test_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    assert re.fullmatch(r"[0-9a-f]{7,}(-dirty)?", git_sha(REPO_ROOT))

    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_sha(REPO_ROOT) == "unknown"
