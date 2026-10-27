"""Notebooks are committed without outputs (CLAUDE.md rule 0.2.11: thin wrappers, no logic).

Outputs belong in results/ and Drive, not in git. Clear them before committing
(Colab: Edit > Clear all outputs; Jupyter: Cell > All Output > Clear) and never commit from Colab's
"Save in GitHub".
"""

import json
from pathlib import Path

import pytest

from ml.common.config import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def test_notebooks_exist() -> None:
    assert NOTEBOOKS, "expected the Colab notebooks under notebooks/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_no_outputs(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    dirty = [
        cell.get("id", str(i))
        for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)  # 0 is dirty too
    ]
    assert not dirty, f"{path.name}: cells with outputs or execution counts: {dirty}; clear them"
