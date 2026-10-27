"""The one evaluation function every model goes through (CLAUDE.md s8.3)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ml.eval.data import EvalData, FitData
from ml.eval.errors import DegenerateScoresError, LeakageError
from ml.eval.metrics import coverage, gini, ild, novelty, per_user_metrics, tail_share


class Scorer(Protocol):
    """A model: fit once per mode, then score users against the whole catalogue."""

    name: str
    cutoff: int | None  # set by fit() from FitData; checked against the evaluation mode

    def fit(self, data: FitData) -> None: ...

    def score(self, users: np.ndarray) -> np.ndarray: ...  # (len(users), n_items) float32


@dataclass
class EvalResult:
    """Per-user metric rows, their means, beyond-accuracy metrics and the top-K lists."""

    name: str
    rows: pd.DataFrame  # one row per evaluated user: "u" + every metric
    means: dict[str, float]
    beyond: dict[str, float]
    tops: np.ndarray  # (users, K) item ids, best first


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` best scores per row, best first (stable for ties)."""
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1, kind="stable")
    return np.take_along_axis(part, order, 1)


def check_degenerate(scores: np.ndarray, name: str) -> None:
    """Raise if any row's finite scores all tie (or a row has none): with every score equal the
    "ranking" is item-index order (the zero-user-vector bug, s2.2 #7). Non-candidates at -inf are
    allowed (candidate-based rankers, s8.3)."""
    finite = np.isfinite(scores)
    hi = np.where(finite, scores, -np.inf).max(axis=1)
    lo = np.where(finite, scores, np.inf).min(axis=1)
    if np.any(finite.sum(axis=1) < 2) or np.any(hi == lo):
        raise DegenerateScoresError(f"{name}: a user's scores are all equal (or missing)")


def evaluate(
    scorer: Scorer,
    data: EvalData,
    ks: Sequence[int] = (5, 10, 20, 50),
    batch: int = 1024,
    embeddings: np.ndarray | None = None,
) -> EvalResult:
    """Full-catalogue ranking of ``data.users`` with their fit items masked."""
    if scorer.cutoff != data.cutoff:
        raise LeakageError(
            f"{scorer.name} was fit with cutoff {scorer.cutoff}, evaluation needs {data.cutoff}"
        )
    k_max = max(ks)
    rows, tops = [], []
    for start in range(0, len(data.users), batch):
        users = data.users[start : start + batch]
        scores = np.asarray(scorer.score(users), dtype=np.float32).copy()
        check_degenerate(scores, scorer.name)
        for r, u in enumerate(users.tolist()):
            if len(np.intersect1d(data.mask[u], data.truth[u])):
                raise LeakageError(f"user {u}: ground truth overlaps the masked fit items")
            scores[r, data.mask[u]] = -np.inf
        top = top_k(scores, k_max)
        tops.append(top)
        for r, u in enumerate(users.tolist()):
            rows.append({"u": u, **per_user_metrics(top[r], data.truth[u], ks)})
    tops_all = np.concatenate(tops) if tops else np.empty((0, k_max), np.int64)
    frame = pd.DataFrame(rows)
    n_items = data.fit.n_items
    beyond: dict[str, float] = {}
    for k in ks:
        beyond[f"coverage@{k}"] = coverage(tops_all, k, n_items)
        beyond[f"gini@{k}"] = gini(tops_all, k, n_items)
        beyond[f"novelty@{k}"] = novelty(tops_all, k, data.item_counts)
        beyond[f"tail_share@{k}"] = tail_share(tops_all, k, data.tail)
        if embeddings is not None:
            beyond[f"ild@{k}"] = ild(tops_all, k, embeddings)
    beyond["catalogue_items"] = float(n_items)  # the coverage denominator (s8.2)
    means = frame.drop(columns="u").mean().to_dict() if len(frame) else {}
    return EvalResult(scorer.name, frame, means, beyond, tops_all)
