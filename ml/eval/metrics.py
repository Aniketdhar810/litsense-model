"""Ranking and beyond-accuracy metrics (CLAUDE.md s8.2). Binary relevance; ranks start at 1."""

from collections.abc import Sequence

import numpy as np

ACCURACY = ("precision", "recall", "hit", "ndcg", "mrr")


def per_user_metrics(top: np.ndarray, truth: np.ndarray, ks: Sequence[int]) -> dict[str, float]:
    """Accuracy metrics of one ranked list ``top`` (length >= max(ks)) against ``truth``."""
    hits = np.isin(top[: max(ks)], truth)
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    first = int(np.argmax(hits)) if hits.any() else None
    out: dict[str, float] = {}
    for k in ks:
        h = hits[:k]
        n_hits = int(h.sum())
        ideal = discounts[: min(len(truth), k)].sum()
        out[f"precision@{k}"] = n_hits / k
        out[f"recall@{k}"] = n_hits / len(truth) if len(truth) else 0.0
        out[f"hit@{k}"] = float(n_hits > 0)
        out[f"ndcg@{k}"] = float(discounts[:k][h].sum() / ideal) if ideal else 0.0
        out[f"mrr@{k}"] = 1.0 / (first + 1) if first is not None and first < k else 0.0
    return out


def coverage(tops: np.ndarray, k: int, n_items: int) -> float:
    """Distinct recommended items in the top-k lists / catalogue size ``n_items``."""
    return len(np.unique(tops[:, :k])) / n_items


def gini(tops: np.ndarray, k: int, n_items: int) -> float:
    """Gini coefficient of how often each catalogue item is recommended (0 = perfectly even)."""
    counts = np.sort(np.bincount(tops[:, :k].ravel(), minlength=n_items).astype(float))
    if counts.sum() == 0:
        return 0.0
    n = len(counts)
    return float((2 * np.arange(1, n + 1) - n - 1) @ counts / (n * counts.sum()))


def novelty(tops: np.ndarray, k: int, item_counts: np.ndarray) -> float:
    """Mean -log2 of each recommended item's share of fit interactions (unseen items count 1)."""
    share = np.maximum(item_counts, 1) / max(item_counts.sum(), 1)
    return float(np.mean(-np.log2(share[tops[:, :k]])))


def tail_share(tops: np.ndarray, k: int, tail: np.ndarray) -> float:
    """Fraction of recommended items in the long-tail set."""
    return float(tail[tops[:, :k]].mean())


def ild(tops: np.ndarray, k: int, embeddings: np.ndarray) -> float:
    """Mean pairwise 1 - cosine within each top-k list (``embeddings`` rows L2-normalized)."""
    off = ~np.eye(k, dtype=bool)
    total, count = 0.0, 0
    for start in range(0, len(tops), 512):  # bounded memory: users x k x d per batch
        lists = embeddings[tops[start : start + 512, :k]].astype(np.float32)
        sims = np.einsum("ukd,ujd->ukj", lists, lists)[:, off]
        total += float((1 - sims).sum())
        count += sims.size
    return total / count if count else 0.0
