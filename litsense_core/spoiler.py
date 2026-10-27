"""Runtime spoiler scorer (CLAUDE.md s11.7 steps 5-7), numpy only.

``score = sigmoid(emb . coef + intercept) + regex_boost * [a spoiler pattern matches]
+ late_position_boost * [the sentence sits in the last ``late_fraction`` of its review]``;
a sentence is flagged when ``score >= threshold``. The same class scores Amazon sentences offline
(``ml/nlp/spoiler_train.py``) and LLM text or snippets online. A model file with an empty
coefficient vector is a regex-only scorer (the classifier is not trained yet): the probability
term is 0 and the threshold equals ``regex_boost``, so a pattern match alone flags.
"""

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class SpoilerScorer:
    """Logistic-regression probability over MiniLM embeddings plus regex and position boosts."""

    def __init__(
        self,
        coef: np.ndarray | None,
        intercept: float,
        threshold: float,
        patterns: Sequence[str],
        regex_boost: float,
        late_position_boost: float,
        late_fraction: float,
    ) -> None:
        self.coef = None if coef is None or coef.size == 0 else np.asarray(coef, dtype=np.float32)
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.patterns = [str(p) for p in patterns]
        self.regex = (
            re.compile("|".join(f"(?:{p})" for p in self.patterns), re.IGNORECASE)
            if self.patterns
            else None
        )
        self.regex_boost = float(regex_boost)
        self.late_position_boost = float(late_position_boost)
        self.late_fraction = float(late_fraction)

    @property
    def has_model(self) -> bool:
        return self.coef is not None

    @classmethod
    def regex_only(
        cls, patterns: Sequence[str], regex_boost: float, late_position_boost: float,
        late_fraction: float,
    ) -> "SpoilerScorer":  # fmt: skip
        """A scorer without a classifier: a regex match alone flags."""
        return cls(
            None, 0.0, regex_boost, patterns, regex_boost, late_position_boost, late_fraction
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "SpoilerScorer":
        """Load ``spoiler_lr.npz`` as written by ``ml/nlp/spoiler_train.py``."""
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["coef"],
                float(data["intercept"]),
                float(data["threshold"]),
                [str(p) for p in data["regex_patterns"].tolist()],
                float(data["regex_boost"]),
                float(data["late_position_boost"]),
                float(data["late_fraction"]),
            )

    def to_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            coef=np.zeros(0, dtype=np.float32) if self.coef is None else self.coef,
            intercept=np.float32(self.intercept),
            threshold=np.float32(self.threshold),
            regex_patterns=np.array(self.patterns, dtype=str),
            regex_boost=np.float32(self.regex_boost),
            late_position_boost=np.float32(self.late_position_boost),
            late_fraction=np.float32(self.late_fraction),
        )

    def probability(self, emb: np.ndarray) -> np.ndarray:
        """Classifier probability per row; zeros without a model."""
        emb = np.asarray(emb, dtype=np.float32)
        if self.coef is None:
            return np.zeros(len(emb), dtype=np.float32)
        if emb.shape[1] != self.coef.shape[0]:
            raise ValueError(f"embedding dim {emb.shape[1]} != model dim {self.coef.shape[0]}")
        z = emb @ self.coef + self.intercept
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    def regex_hits(self, texts: Sequence[str]) -> np.ndarray:
        if self.regex is None:
            return np.zeros(len(texts), dtype=bool)
        return np.array([self.regex.search(t) is not None for t in texts], dtype=bool)

    def late(self, pos: np.ndarray | None, n_in_review: np.ndarray | None) -> np.ndarray:
        """True where ``pos / n_in_review`` falls in the last ``late_fraction`` of the review."""
        if pos is None or n_in_review is None:
            return np.zeros(0, dtype=bool)
        pos = np.asarray(pos, dtype=np.float64)
        n = np.asarray(n_in_review, dtype=np.float64)
        return (pos + 1) / np.clip(n, 1, None) > 1.0 - self.late_fraction

    def score(
        self,
        emb: np.ndarray,
        texts: Sequence[str],
        pos: np.ndarray | None = None,
        n_in_review: np.ndarray | None = None,
    ) -> np.ndarray:
        """Final score per sentence (probability plus boosts)."""
        out = self.probability(emb) + self.regex_boost * self.regex_hits(texts)
        if pos is not None and n_in_review is not None:
            out = out + self.late_position_boost * self.late(pos, n_in_review)
        return out.astype(np.float32)

    def flags(
        self,
        emb: np.ndarray,
        texts: Sequence[str],
        pos: np.ndarray | None = None,
        n_in_review: np.ndarray | None = None,
    ) -> np.ndarray:
        return self.score(emb, texts, pos, n_in_review) >= self.threshold - 1e-6
