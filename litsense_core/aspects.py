"""Aspect-profile arithmetic shared by the offline profiles stage and the online profile
(CLAUDE.md s11.8, principle 4: same code offline and online). numpy only.

Item side (offline, per cutoff): ``shrink`` pulls a per-item mean toward the aspect's global mean
with weight ``k`` pseudo-reviews. User side (both): the implicit vector is the rating- and
time-weighted mean of the *centered* item profiles of the books the reader rated; the explicit
vector is the decay-weighted mean polarity of the reader's own review sentences, with a confidence
that grows with the number of reviews; ``combine`` blends them.
"""

from collections.abc import Sequence

import numpy as np

DAY_MS = 86_400_000


def shrink(raw: np.ndarray, n: np.ndarray, means: np.ndarray, k: float) -> np.ndarray:
    """``(n * raw + k * mean) / (n + k)`` per cell; cells with ``n == 0`` return the mean."""
    raw = np.where(n > 0, raw, 0.0)
    return ((n * raw + k * means) / (n + k)).astype(np.float32)


def decay(ages_ms: np.ndarray, half_life_days: float) -> np.ndarray:
    """``0.5 ** (age_days / half_life)``; ages below zero count as zero."""
    days = np.clip(np.asarray(ages_ms, dtype=np.float64), 0, None) / DAY_MS
    return (0.5 ** (days / half_life_days)).astype(np.float32)


def implicit_aspect_vec(
    item_aspects: np.ndarray, means: np.ndarray, item_ids: np.ndarray, weights: np.ndarray
) -> np.ndarray | None:
    """``sum w * (shrunk_i - mean) / sum |w|`` over rated items; None when there is no weight."""
    weights = np.asarray(weights, dtype=np.float32)
    total = float(np.abs(weights).sum())
    if len(weights) == 0 or total == 0.0:
        return None
    centered = np.asarray(item_aspects[np.asarray(item_ids, dtype=np.int64)], dtype=np.float32)
    centered = centered - means.astype(np.float32)
    return ((weights[:, None] * centered).sum(axis=0) / total).astype(np.float32)


def explicit_aspect_vec(
    polarities: np.ndarray,
    weights: np.ndarray,
    n_aspects: int,
    conf_k: float,
) -> tuple[np.ndarray, np.ndarray]:
    """From per-review aspect polarities (``(n_reviews, n_aspects)``, NaN where the review does not
    mention the aspect) and per-review weights (time decay): the weighted mean polarity per aspect
    (0 where never mentioned) and the confidence ``n_a / (n_a + conf_k)``."""
    polarities = np.asarray(polarities, dtype=np.float32).reshape(-1, n_aspects)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    present = ~np.isnan(polarities)
    w = present * weights[:, None]
    total = w.sum(axis=0)
    vec = np.where(
        total > 0, (np.nan_to_num(polarities) * w).sum(axis=0) / np.where(total > 0, total, 1), 0.0
    )
    n = present.sum(axis=0).astype(np.float32)
    return vec.astype(np.float32), (n / (n + conf_k)).astype(np.float32)


def combine(
    implicit: np.ndarray | None,
    explicit: np.ndarray | None,
    explicit_conf: np.ndarray | None,
    explicit_weight: float,
    positive: Sequence[int] = (),
    negative: Sequence[int] = (),
    pinned: Sequence[int] = (),
    n_aspects: int = 12,
) -> np.ndarray | None:
    """``w * explicit + (1 - w) * implicit`` where the explicit side has confidence, else implicit;
    onboarding aspects add +-0.5 and pinned aspects +0.5; clipped to [-1, 1]. None when nothing is
    known at all."""
    if implicit is None and explicit is None and not (positive or negative or pinned):
        return None
    vec = np.zeros(n_aspects, dtype=np.float32) if implicit is None else implicit.copy()
    if explicit is not None and explicit_conf is not None:
        w = explicit_weight * explicit_conf  # confidence scales how much the explicit side counts
        vec = w * explicit + (1 - w) * vec
    for a in positive:
        vec[a] += 0.5
    for a in negative:
        vec[a] -= 0.5
    for a in pinned:
        vec[a] += 0.5
    return np.clip(vec, -1.0, 1.0).astype(np.float32)
