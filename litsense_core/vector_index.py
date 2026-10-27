"""Exact inner-product search over L2-normalized vectors (CLAUDE.md s10.2).

FAISS ``IndexFlatIP`` when ``faiss`` is importable, else ``matrix @ q`` with ``argpartition``.
Both paths return the same ids for the same query; tests run without faiss.
"""

from collections.abc import Iterable

import numpy as np

try:  # pragma: no cover - exercised only where faiss is installed
    import faiss  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001 (any import failure means "no faiss")
    faiss = None


class VectorIndex:
    """Top-k by inner product over the rows of ``matrix`` (float32, rows L2-normalized)."""

    def __init__(self, matrix: np.ndarray, use_faiss: bool = True) -> None:
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.n, self.dim = self.matrix.shape
        self._faiss = None
        if use_faiss and faiss is not None and self.n:
            index = faiss.IndexFlatIP(self.dim)
            index.add(self.matrix)
            self._faiss = index

    @property
    def backend(self) -> str:
        return "faiss" if self._faiss is not None else "numpy"

    def search(
        self,
        query: np.ndarray,
        k: int,
        exclude: Iterable[int] | None = None,
        allow: Iterable[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(ids, scores)`` of the ``k`` best rows, best first. ``exclude`` removes ids;
        ``allow`` restricts to ids. Fewer than ``k`` rows come back when the pool is small."""
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has {q.shape[0]} dims, index has {self.dim}")
        if k < 1 or self.n == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        excluded = set() if exclude is None else {int(i) for i in exclude}
        if allow is not None:
            pool = np.fromiter((int(i) for i in allow if int(i) not in excluded), dtype=np.int64)
            pool = np.unique(pool[(pool >= 0) & (pool < self.n)])
            if pool.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.float32)
            scores = self.matrix[pool] @ q
            order = np.argsort(-scores, kind="stable")[:k]
            return pool[order], scores[order].astype(np.float32)
        want = min(self.n, k + len(excluded))
        if self._faiss is not None:
            scores, ids = self._faiss.search(q[None, :], want)
            ids, scores = ids[0], scores[0]
            keep = ids >= 0
            ids, scores = ids[keep], scores[keep]
        else:
            scores = self.matrix @ q
            part = np.argpartition(-scores, want - 1)[:want] if want < self.n else np.arange(self.n)
            order = np.argsort(-scores[part], kind="stable")
            ids, scores = part[order], scores[part][order]
        if excluded:
            keep = np.fromiter((int(i) not in excluded for i in ids), dtype=bool, count=len(ids))
            ids, scores = ids[keep], scores[keep]
        return ids[:k].astype(np.int64), scores[:k].astype(np.float32)
