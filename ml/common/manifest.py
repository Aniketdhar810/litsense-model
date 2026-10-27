"""Manifests: hashing, file descriptions, fingerprints and atomic JSON writes (CLAUDE.md s7.0).

A manifest is the JSON record a stage writes last, after all its outputs are in place. Readers
(the next stage, the evaluator, the bundle export) trust an artifact only if its manifest says
``status == "complete"``.
"""

import gzip
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def sha256_dir(path: Path) -> str:
    """Hex sha256 over every file under ``path`` (relative path and content, in sorted order)."""
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(f.relative_to(path).as_posix().encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def describe_path(path: Path) -> dict[str, Any]:
    """Return the manifest entry for an output: sha256 plus rows, or shape and dtype, when known.

    Parquet reports rows from its footer, ``.npy`` its shape and dtype, CSV its data lines
    (header excluded), JSONL its lines. Directories report a combined hash and a file count.
    """
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.is_file()]
        return {
            "sha256": sha256_dir(path),
            "n_files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
        }
    info: dict[str, Any] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    name = path.name.lower()
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        info["rows"] = pq.read_metadata(path).num_rows
    elif name.endswith(".npy"):
        arr = np.load(path, mmap_mode="r")
        info["shape"] = list(arr.shape)
        info["dtype"] = str(arr.dtype)
    elif name.endswith((".csv", ".csv.gz")):
        info["rows"] = max(_count_lines(path) - 1, 0)
    elif name.endswith((".jsonl", ".jsonl.gz")):
        info["rows"] = _count_lines(path)
    return info


def _count_lines(path: Path) -> int:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    n, last = 0, b"\n"
    with opener(path, "rb") as f:
        while block := f.read(_CHUNK):
            n += block.count(b"\n")
            last = block[-1:]
    return n + (last != b"\n")


def fingerprint(material: Mapping[str, Any]) -> str:
    """Hex sha256 of the canonical JSON encoding of ``material``."""
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON to ``<path>.tmp`` and rename it over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_manifest(path: Path) -> dict[str, Any] | None:
    """Return the manifest at ``path``, or None if it does not exist."""
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_complete(manifest: Mapping[str, Any] | None) -> bool:
    """True only for a manifest written after a successful run."""
    return manifest is not None and manifest.get("status") == "complete"


def git_sha(repo_root: Path) -> str:
    """Short HEAD sha, suffixed ``-dirty`` when tracked files under ``ml/`` or ``litsense_core/``
    differ from HEAD; ``"unknown"`` outside a git checkout. Provenance only, not a fingerprint.
    """
    try:
        sha = _git(repo_root, "rev-parse", "--short", "HEAD")
        dirty = _git(
            repo_root, "status", "--porcelain", "--untracked-files=no", "--", "ml", "litsense_core"
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def _git(repo_root: Path, *args: str) -> str:
    out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True)
    return out.stdout.strip()
