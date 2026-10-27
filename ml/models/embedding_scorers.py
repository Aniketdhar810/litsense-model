"""Scorers over stored user and item embeddings (LightGCN now, the projection hybrid next).

Numpy only, so evaluation never imports torch. A stored scorer is not refit: ``fit`` checks that
the artifact belongs to the evaluation's fit data (shapes, and which items have fit interactions),
and the cutoff comes from the artifact itself, so the evaluator's cutoff check catches an artifact
from another artifact set (s2.2 #5, s3.2).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.eval.data import FitData
from ml.eval.errors import LeakageError
from ml.models.baselines import ContentScorer


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Rows scaled to unit length; all-zero rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


@dataclass
class EmbeddingArtifact:
    """What a CF training stage stores: embeddings, item degrees and its metadata."""

    user: np.ndarray  # (n_users, dim) float32
    item: np.ndarray  # (n_items, dim) float32
    degree: np.ndarray  # (n_items,) fit interactions per item
    meta: dict[str, Any]  # cutoff, model, config, training history

    @classmethod
    def load(cls, folder: Path, user: str, item: str) -> "EmbeddingArtifact":
        """Read ``<folder>/{user,item}`` plus ``item_degree.npy`` and ``meta.json``."""
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(
            np.load(folder / user).astype(np.float32),
            np.load(folder / item).astype(np.float32),
            np.load(folder / "item_degree.npy"),
            meta,
        )


class EmbeddingScorer:
    """score = user . item, or user . item / |item| with ``normalize_items`` (the L2-normalized
    LightGCN of s9.4). Items with no fit interactions score -inf (s8.1) unless ``score_cold``:
    models that represent them from content (the projection hybrid, s9.5) rank them too."""

    def __init__(
        self,
        name: str,
        artifact: EmbeddingArtifact,
        normalize_items: bool,
        score_cold: bool = False,
    ) -> None:
        self.name = name
        self.artifact = artifact
        self.user = artifact.user
        self.item = l2_normalize_rows(artifact.item) if normalize_items else artifact.item
        self.cold = np.zeros(len(artifact.degree), bool) if score_cold else artifact.degree == 0
        self.cutoff: int | None = int(artifact.meta["cutoff"])

    def fit(self, data: FitData) -> None:
        """Check the stored model was trained on exactly this fit data; nothing is learned."""
        check_artifact(self.name, self.artifact, data)

    def score(self, users: np.ndarray) -> np.ndarray:
        scores = (self.user[users] @ self.item.T).astype(np.float32)
        scores[:, self.cold] = -np.inf
        return scores


def check_artifact(name: str, artifact: EmbeddingArtifact, data: FitData) -> None:
    """Raise ``LeakageError`` unless ``artifact`` was trained on exactly ``data``: same id map
    sizes, same cutoff, same item degrees."""
    if artifact.user.shape[0] != data.n_users or artifact.item.shape[0] != data.n_items:
        raise LeakageError(f"{name}: artifact shapes do not match the id map")
    if int(artifact.meta["cutoff"]) != data.cutoff:
        raise LeakageError(f"{name}: artifact cutoff {artifact.meta['cutoff']} != {data.cutoff}")
    counts = np.bincount(data.interactions["i"], minlength=data.n_items)
    if not np.array_equal(counts, artifact.degree):
        raise LeakageError(f"{name}: artifact item degrees differ from the fit data")


def pca(matrix: np.ndarray, dim: int) -> np.ndarray:
    """Rows of ``matrix`` projected onto its top ``dim`` principal components (centered SVD)."""
    centered = matrix - matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:dim].T).astype(np.float32)


class StaticFusionScorer:
    """Static fusion control (s9.5): alpha * cos(user, LightGCN item) + (1 - alpha) * cos(content
    profile, item content), with content reduced by PCA to the LightGCN dim. Built as one dot
    product of concatenated vectors, so each half compares like with like (D45). Items without
    fit interactions have a zero LightGCN half and are ranked by content alone."""

    def __init__(
        self, artifact: EmbeddingArtifact, content: np.ndarray, alpha: float, relevance_min: float
    ) -> None:
        self.name = f"static_fusion_{alpha:g}"
        self.artifact, self.alpha, self.relevance_min = artifact, alpha, relevance_min
        self.content = l2_normalize_rows(pca(content, artifact.item.shape[1]))
        gcn = l2_normalize_rows(artifact.item) * (artifact.degree > 0)[:, None]
        self.item = np.hstack([gcn, self.content])
        self.cutoff: int | None = int(artifact.meta["cutoff"])

    def fit(self, data: FitData) -> None:
        """Check the LightGCN artifact and build the content half of each user vector (the
        content-only profile of s9.2, in PCA space)."""
        check_artifact(self.name, self.artifact, data)
        profile = ContentScorer(self.content, self.relevance_min)
        profile.fit(data)
        self.user = np.hstack(
            [
                self.alpha * l2_normalize_rows(self.artifact.user),
                (1 - self.alpha) * profile.profiles,
            ]
        )

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.user[users] @ self.item.T).astype(np.float32)
