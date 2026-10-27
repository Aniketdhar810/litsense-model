"""Shared fixtures."""

import copy
from pathlib import Path
from typing import Any

import pytest

from ml.common.config import load_config


@pytest.fixture(scope="session")
def repo_config() -> dict[str, Any]:
    """The repository's ml/config.yaml, loaded once (treat as read-only)."""
    return load_config()


def rooted_config(repo_config: dict[str, Any], root: Path) -> dict[str, Any]:
    """A copy of the repo config whose every root points into ``root``."""
    cfg = copy.deepcopy(repo_config)
    cfg["project"].update(
        raw_root=str(root / "raw"),
        drive_root=str(root / "drive"),
        local_raw_root=str(root / "local_raw"),
        local_root=str(root / "local"),
    )
    return cfg


@pytest.fixture
def tmp_config(repo_config: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    """A copy of the repo config whose every root points into ``tmp_path``."""
    return rooted_config(repo_config, tmp_path)


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The tiny raw fixture (tests/fixtures/tiny), generated once per session."""
    from tests.fixtures.tiny.make_tiny import make_tiny

    out = tmp_path_factory.mktemp("tiny")
    make_tiny(out)
    return out
