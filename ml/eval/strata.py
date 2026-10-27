"""Strata (CLAUDE.md s8.4): user sparsity, long-tail truth, cold-item truth. Scoring and the top-K
lists are unchanged; only the users grouped or the truth counted changes."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ml.eval.data import EvalData, sparsity_bucket
from ml.eval.evaluator import EvalResult
from ml.eval.metrics import per_user_metrics


def by_sparsity(
    result: EvalResult, data: EvalData, buckets: Sequence[Sequence[int]]
) -> pd.DataFrame:
    """Mean metrics and user counts per fit-interaction bucket."""
    rows = result.rows.copy()
    rows["bucket"] = [sparsity_bucket(int(data.fit_counts[u]), buckets) for u in rows["u"]]
    grouped = rows.drop(columns="u").groupby("bucket", sort=False)
    out = grouped.mean()
    out.insert(0, "users", grouped.size())
    order = [sparsity_bucket(lo, buckets) for lo, _ in buckets]
    return out.reindex([b for b in order if b in out.index]).reset_index()


def restricted_rows(
    result: EvalResult, data: EvalData, keep: np.ndarray, ks: Sequence[int]
) -> pd.DataFrame:
    """Per-user metrics with each user's truth restricted to items where ``keep`` is True; users
    left without truth are dropped."""
    rows = []
    for r, u in enumerate(result.rows["u"].tolist()):
        truth = data.truth[u][keep[data.truth[u]]]
        if len(truth):
            rows.append({"u": u, **per_user_metrics(result.tops[r], truth, ks)})
    return pd.DataFrame(rows)


def longtail_rows(result: EvalResult, data: EvalData, ks: Sequence[int]) -> pd.DataFrame:
    """Truth restricted to long-tail items."""
    return restricted_rows(result, data, data.tail, ks)


def cold_item_rows(result: EvalResult, data: EvalData, ks: Sequence[int]) -> pd.DataFrame:
    """Truth restricted to items with zero fit interactions."""
    return restricted_rows(result, data, data.item_counts == 0, ks)
