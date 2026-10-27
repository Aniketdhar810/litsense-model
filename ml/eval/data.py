"""Evaluation data per mode (CLAUDE.md s8.1).

| mode | fit data    | cutoff | masked items      | ground truth                       |
| val  | train       | t1     | the user's train  | val interactions, rating >= rel    |
| test | train + val | t2     | train + val items | test interactions, rating >= rel   |

Evaluated users have at least one relevant ground-truth item and at least one fit interaction;
users without fit interactions belong to the cold-start simulation (s8.5).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from ml.eval.errors import LeakageError

Mode = Literal["val", "test"]
FIT_SPLITS: dict[str, tuple[str, ...]] = {"val": ("train",), "test": ("train", "val")}
CUTOFF_KEY = {"val": "t1", "test": "t2"}


@dataclass
class FitData:
    """What a model may learn from: fit-window interactions (``u``, ``i``, ``rating``, ``ts``)."""

    interactions: pd.DataFrame
    n_users: int
    n_items: int
    cutoff: int  # ms; every fit row is older

    def __post_init__(self) -> None:
        if len(self.interactions) and int(self.interactions["ts"].max()) >= self.cutoff:
            raise LeakageError("fit interactions include rows at or after the cutoff")


@dataclass
class EvalData:
    """Fit data plus the evaluated users, their masks and their ground truth."""

    mode: Mode
    fit: FitData
    users: np.ndarray  # evaluated user ids, sorted
    mask: dict[int, np.ndarray]  # user -> fit items (never recommended)
    truth: dict[int, np.ndarray]  # user -> relevant ground-truth items
    fit_counts: np.ndarray  # (n_users,) fit interactions per user
    item_counts: np.ndarray  # (n_items,) fit interactions per item
    tail: np.ndarray = field(repr=False)  # (n_items,) bool: long-tail items (s8.4)

    @property
    def cutoff(self) -> int:
        return self.fit.cutoff


def longtail_items(item_counts: np.ndarray, fraction: float) -> np.ndarray:
    """The least-interacted ``fraction`` of items (zero-count items included); ties by item id."""
    order = np.lexsort((np.arange(len(item_counts)), item_counts))
    tail = np.zeros(len(item_counts), dtype=bool)
    tail[order[: int(round(fraction * len(item_counts)))]] = True
    return tail


def _by_user(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    return {int(u): np.sort(g["i"].to_numpy(np.int64)) for u, g in frame.groupby("u")}


def build_eval_data(
    interactions: pd.DataFrame,
    splits: Mapping[str, object],
    mode: Mode,
    n_users: int,
    n_items: int,
    relevance_min: float,
    longtail_fraction: float,
) -> EvalData:
    """``EvalData`` for ``mode`` from ``interactions.parquet`` and ``splits.json``."""
    cutoff = int(splits[CUTOFF_KEY[mode]]["ms"])  # type: ignore[index]
    in_fit = interactions["split"].isin(FIT_SPLITS[mode])
    fit_rows = interactions.loc[in_fit, ["u", "i", "rating", "ts"]].reset_index(drop=True)
    target = "val" if mode == "val" else "test"
    truth_rows = interactions[
        (interactions["split"] == target) & (interactions["rating"] >= relevance_min)
    ]
    fit = FitData(fit_rows, n_users, n_items, cutoff)
    fit_counts = np.bincount(fit_rows["u"], minlength=n_users)
    item_counts = np.bincount(fit_rows["i"], minlength=n_items)
    mask = _by_user(fit_rows)
    truth = {}
    for u, items in _by_user(truth_rows).items():
        if fit_counts[u] == 0:
            continue
        items = np.setdiff1d(items, mask.get(u, np.empty(0, np.int64)))
        if len(items):
            truth[u] = items
    users = np.array(sorted(truth), dtype=np.int64)
    return EvalData(
        mode=mode,
        fit=fit,
        users=users,
        mask={u: mask.get(u, np.empty(0, np.int64)) for u in users.tolist()},
        truth=truth,
        fit_counts=fit_counts,
        item_counts=item_counts,
        tail=longtail_items(item_counts, longtail_fraction),
    )


def sparsity_bucket(count: int, buckets: Sequence[Sequence[int]]) -> str:
    """``"lo-hi"`` label of the bucket holding ``count`` (``data.sparsity_buckets``)."""
    for lo, hi in buckets:
        if lo <= count <= hi:
            return f"{lo}-{hi}" if hi < 1_000_000 else f"{lo}+"
    return "0"
