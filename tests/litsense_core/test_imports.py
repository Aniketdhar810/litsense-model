"""Serving-package import rules (CLAUDE.md s3.5, s4.4)."""

import ast
import subprocess
import sys
from pathlib import Path

from ml.common.config import REPO_ROOT

CORE = REPO_ROOT / "litsense_core"
# Allowed third-party imports for litsense_core: requirements-core.txt (import names).
ALLOWED_THIRD_PARTY = {
    "numpy", "faiss", "lightgbm", "onnxruntime", "tokenizers", "rapidfuzz", "pydantic",
    "yaml", "httpx", "google", "vaderSentiment",
}  # fmt: skip


def imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_importing_core_does_not_import_torch() -> None:
    code = (
        "import sys, litsense_core, litsense_core.schemas, litsense_core.mock\n"
        "bad = sorted(m for m in ('torch', 'pandas', 'transformers', 'sklearn', 'ml') "
        "if m in sys.modules)\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_core_imports_only_stdlib_and_core_requirements() -> None:
    stdlib = set(sys.stdlib_module_names)
    for path in CORE.rglob("*.py"):
        foreign = imported_roots(path) - stdlib - ALLOWED_THIRD_PARTY - {"litsense_core"}
        assert not foreign, f"{path.relative_to(REPO_ROOT)} imports {sorted(foreign)}"


def test_core_requirements_match_allowed_list() -> None:
    names = {
        line.split("==")[0].lower()
        for line in (REPO_ROOT / "requirements-core.txt").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert names == {
        "numpy", "faiss-cpu", "lightgbm", "onnxruntime", "tokenizers", "rapidfuzz", "pydantic",
        "pyyaml", "httpx", "google-genai", "vadersentiment",
    }  # fmt: skip
