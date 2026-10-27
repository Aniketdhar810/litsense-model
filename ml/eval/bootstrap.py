"""Seeded bootstrap confidence intervals over users (CLAUDE.md s8.6)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A mean with its 95% percentile interval."""

    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class PairedDiff:
    """Mean of (A - B) over the same users, its 95% interval and the share of resamples <= 0."""

    mean: float
    low: float
    high: float
    share_le_zero: float
    users: int


def _resample_means(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.integers(0, len(values), size=(n, len(values)))
    return values[idx].mean(axis=1)


def mean_ci(values: np.ndarray, n: int, rng: np.random.Generator) -> Interval:
    """95% percentile bootstrap interval of the mean."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    means = _resample_means(values, n, rng)
    return Interval(float(values.mean()), *map(float, np.percentile(means, [2.5, 97.5])))


def paired(a: np.ndarray, b: np.ndarray, n: int, rng: np.random.Generator) -> PairedDiff:
    """Paired bootstrap of A - B; ``a`` and ``b`` are per-user values for the same users in the
    same order."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(diff) == 0:
        return PairedDiff(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    means = _resample_means(diff, n, rng)
    low, high = np.percentile(means, [2.5, 97.5])
    return PairedDiff(
        float(diff.mean()), float(low), float(high), float((means <= 0).mean()), len(diff)
    )
