"""Week-4 baselines (CLAUDE.md s9.1-9.3): random, popularity, ItemKNN, ALS, content-only.

Every scorer implements ``ml.eval.evaluator.Scorer``: ``fit`` records the fit cutoff, ``score``
returns (users, n_items) float32. Collaborative scorers cannot score items without fit
interactions: they get -inf and show up in the cold-item stratum (s8.1).
"""

import contextlib
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ml.eval.data import FitData

DAY_MS = 86_400_000


def user_item_matrix(data: FitData, binary: bool = True) -> sp.csr_matrix:
    """(n_users, n_items) CSR of fit interactions (1.0, or the rating)."""
    rows = data.interactions
    values = np.ones(len(rows), np.float32) if binary else rows["rating"].to_numpy(np.float32)
    return sp.csr_matrix(
        (values, (rows["u"].to_numpy(), rows["i"].to_numpy())), shape=(data.n_users, data.n_items)
    )


class RandomScorer:
    """Seeded uniform scores: the sanity floor."""

    name = "random"

    def __init__(self, seed: int) -> None:
        self.seed, self.cutoff, self.n_items = seed, None, 0

    def fit(self, data: FitData) -> None:
        self.cutoff, self.n_items = data.cutoff, data.n_items

    def score(self, users: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng([self.seed, int(users[0]) if len(users) else 0])
        return rng.random((len(users), self.n_items), dtype=np.float32)


class PopularityScorer:
    """Fit-window interaction counts. Also keeps the Bayesian mean rating and the trending score
    used later by features and rails (s9.1)."""

    name = "popularity"

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.bayes_m = float(params["bayes_m"])
        self.half_life = float(params["trend_half_life_days"])
        self.cutoff: int | None = None

    def fit(self, data: FitData) -> None:
        rows = data.interactions
        self.cutoff = data.cutoff
        self.counts = np.bincount(rows["i"], minlength=data.n_items).astype(np.float32)
        sums = np.bincount(rows["i"], weights=rows["rating"], minlength=data.n_items)
        global_mean = float(rows["rating"].mean()) if len(rows) else 0.0
        self.bayes_rating = (sums + self.bayes_m * global_mean) / (self.counts + self.bayes_m)
        age_days = (data.cutoff - rows["ts"].to_numpy()) / DAY_MS
        weights = np.exp(-age_days / self.half_life)
        self.trending = np.bincount(rows["i"], weights=weights, minlength=data.n_items)
        self._scores = np.where(self.counts > 0, self.counts, -np.inf).astype(np.float32)

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self._scores, (len(users), len(self._scores))).copy()


class ItemKNNScorer:
    """Item-item cosine with shrinkage on the binary user-item matrix, top-``k`` neighbours per
    item; score = the user's row times the similarity matrix (s9.3)."""

    name = "itemknn"

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.k, self.shrink = int(params["k"]), float(params["shrink"])
        self.cutoff: int | None = None

    def fit(self, data: FitData) -> None:
        self.cutoff = data.cutoff
        self.x = user_item_matrix(data)
        co = (self.x.T @ self.x).tocsr().astype(np.float32)
        norms = np.sqrt(co.diagonal())
        co.setdiag(0)
        co.eliminate_zeros()
        co = co.tocoo()
        sim = co.data / (norms[co.row] * norms[co.col] + self.shrink)
        full = sp.csr_matrix((sim, (co.row, co.col)), shape=co.shape)
        self.sim = keep_top_k(full, self.k)
        self.degree = np.asarray(self.x.sum(axis=0)).ravel()

    def score(self, users: np.ndarray) -> np.ndarray:
        scores = (self.x[users] @ self.sim).toarray().astype(np.float32)
        scores[:, self.degree == 0] = -np.inf
        return scores


def keep_top_k(matrix: sp.csr_matrix, k: int) -> sp.csr_matrix:
    """Keep the ``k`` largest entries of each row: row r holds item r's neighbours, so
    ``user_row @ result`` scores item c by summing sim(r, c) over the user's items r."""
    rows, cols, vals = [], [], []
    for r in range(matrix.shape[0]):
        start, end = matrix.indptr[r], matrix.indptr[r + 1]
        if end == start:
            continue
        data, idx = matrix.data[start:end], matrix.indices[start:end]
        keep = np.argsort(-data, kind="stable")[:k]
        rows.append(np.full(len(keep), r))
        cols.append(idx[keep])
        vals.append(data[keep])
    if not rows:
        return sp.csr_matrix(matrix.shape, dtype=np.float32)
    # entry (source item r, neighbour c); user scores for item c sum sim(r, c) over rated r
    return sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=matrix.shape,
        dtype=np.float32,
    )


def _single_blas_thread() -> Any:
    """``threadpoolctl`` limit to one BLAS thread when available, else a no-op context."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(1, "blas")


class ALSScorer:
    """Implicit ALS on confidence 1 + alpha * binary (s9.3). ``implicit`` >= 0.5 fits on the
    (users x items) matrix; the output shapes are asserted because older versions differ."""

    name = "als"

    def __init__(self, params: Mapping[str, Any], seed: int) -> None:
        self.params, self.seed = dict(params), seed
        self.cutoff: int | None = None

    def fit(self, data: FitData) -> None:
        from implicit.cpu.als import AlternatingLeastSquares

        self.cutoff = data.cutoff
        model = AlternatingLeastSquares(
            factors=int(self.params["factors"]),
            regularization=float(self.params["regularization"]),
            iterations=int(self.params["iterations"]),
            random_state=self.seed,
        )
        x = user_item_matrix(data) * float(self.params["alpha"])
        with _single_blas_thread():  # implicit warns that threaded BLAS slows ALS down
            model.fit(x.tocsr(), show_progress=False)
        self.user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self.item_factors = np.asarray(model.item_factors, dtype=np.float32)
        factors = int(self.params["factors"])
        assert self.user_factors.shape == (data.n_users, factors), self.user_factors.shape
        assert self.item_factors.shape == (data.n_items, factors), self.item_factors.shape
        self.degree = np.bincount(data.interactions["i"], minlength=data.n_items)

    def score(self, users: np.ndarray) -> np.ndarray:
        scores = self.user_factors[users] @ self.item_factors.T
        scores[:, self.degree == 0] = -np.inf
        return scores.astype(np.float32)


class ContentScorer:
    """Content-only (s9.2): the user profile is the rating-weighted mean of the content vectors of
    fit items rated >= ``relevance_min`` (all fit items if none), L2-normalized; never zero."""

    name = "content"

    def __init__(self, embeddings: np.ndarray, relevance_min: float) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.relevance_min = relevance_min
        self.cutoff: int | None = None

    def fit(self, data: FitData) -> None:
        self.cutoff = data.cutoff
        rows = data.interactions
        liked = rows[rows["rating"] >= self.relevance_min]
        chosen = pd.concat([liked, rows[~rows["u"].isin(liked["u"].unique())]])
        weights = sp.csr_matrix(
            (
                chosen["rating"].to_numpy(np.float32),
                (chosen["u"].to_numpy(), chosen["i"].to_numpy()),
            ),
            shape=(data.n_users, data.n_items),
        )
        profile = np.asarray(weights @ self.embeddings, dtype=np.float32)
        norms = np.linalg.norm(profile, axis=1, keepdims=True)
        self.profiles = np.divide(profile, norms, out=np.zeros_like(profile), where=norms > 0)

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.profiles[users] @ self.embeddings.T).astype(np.float32)
