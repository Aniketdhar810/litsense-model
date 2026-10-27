"""One global seed makes every RNG reproducible (CLAUDE.md s2.2 #8)."""

import os
import random

import numpy as np
import pytest

from ml.common.seed import set_global_seed


def draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())


def test_same_seed_same_draws() -> None:
    set_global_seed(42)
    first = draw()
    set_global_seed(42)
    assert draw() == first
    set_global_seed(43)
    assert draw() != first


def test_sets_pythonhashseed() -> None:
    set_global_seed(7)
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_seeds_torch_when_installed() -> None:
    torch = pytest.importorskip("torch")
    set_global_seed(42)
    first = torch.rand(3)
    set_global_seed(42)
    assert torch.equal(torch.rand(3), first)
