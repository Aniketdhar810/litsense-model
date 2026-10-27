"""Stand-ins for MiniLM, spaCy and DistilBERT so the fast tests never load a model.

The stub encoder hashes each text into a fixed unit vector, so identical texts get identical
embeddings (cosine 1) and different texts are nearly orthogonal in expectation. The stub polarity
reads the tiny fixture's template sentences and a few clause keywords.
"""

import hashlib
import re

import numpy as np

from tests.fixtures.tiny.make_tiny import ASPECT_SENTENCES

DIM = 16
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TEMPLATE_POLARITY = {
    text: 0.9 * sign for pairs in ASPECT_SENTENCES.values() for text, sign in pairs
}
_POSITIVE = ("wonderful", "loved", "great", "gorgeous")
_NEGATIVE = ("dragged", "boring", "flat", "annoying")


class StubEncoder:
    """Deterministic ``DIM``-d unit vectors from a text hash; whitespace tokens."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[int] = []  # batch sizes seen, for resume tests

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(len(texts))
        rows = []
        for t in texts:
            seed = int(hashlib.sha1(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim)
            rows.append(v / np.linalg.norm(v))
        return np.array(rows, dtype=np.float32).reshape(len(texts), self.dim)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


def stub_splitter(texts: list[str]) -> list[list[str]]:
    """Regex sentence split on terminal punctuation followed by whitespace."""
    return [[s for s in _SENTENCE_END.split(t.strip()) if s] for t in texts]


def stub_polarity(texts: list[str]) -> np.ndarray:
    """Fixture template sentences get their planted sign; clauses are read by keyword."""
    out = np.zeros(len(texts), dtype=np.float32)
    for k, text in enumerate(texts):
        if text in _TEMPLATE_POLARITY:
            out[k] = _TEMPLATE_POLARITY[text]
            continue
        lower = text.lower()
        if any(w in lower for w in _POSITIVE):
            out[k] = 0.9
        elif any(w in lower for w in _NEGATIVE):
            out[k] = -0.9
    return out
