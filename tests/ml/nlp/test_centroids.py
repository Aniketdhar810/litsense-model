"""Review centroids (CLAUDE.md s11.6) by hand, and the leakage guard (s16.2)."""

import numpy as np
import pandas as pd
import pytest

from ml.eval.errors import LeakageError
from ml.nlp.centroids import build_centroids


def _unit(*values: float) -> np.ndarray:
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_centroids_are_normalized_means_of_liked_sentences() -> None:
    emb = np.stack([_unit(1, 0, 0), _unit(0, 1, 0), _unit(0, 0, 1), _unit(1, 1, 0)]).astype(
        np.float16
    )
    sentences = pd.DataFrame(
        {
            "sent_id": [0, 1, 2, 3],
            "u": [0, 0, 1, 1],
            "i": [0, 0, 1, 2],
            "ts": [10, 20, 30, 40],
            "rating": [5.0, 4.0, 2.0, 5.0],  # sentence 2 is below the rating floor
        }
    )
    item, user, counts = build_centroids(
        sentences, emb, 3, 2, cutoff=100, min_rating=4, chunk_rows=2
    )
    assert item.shape == (3, 3) and user.shape == (2, 3) and item.dtype == np.float32
    assert np.allclose(item[0], _unit(1, 1, 0), atol=1e-3)  # mean of e0 and e1, normalized
    assert np.allclose(item[1], 0)  # its only sentence is rated 2
    assert np.allclose(item[2], _unit(1, 1, 0), atol=1e-3)
    assert np.allclose(user[0], _unit(1, 1, 0), atol=1e-3) and np.allclose(
        user[1], _unit(1, 1, 0), atol=1e-3
    )
    assert counts == {
        "sentences_before_cutoff": 4,
        "sentences_used": 3,
        "items_with_centroid": 2,
        "users_with_centroid": 2,
    }
    with pytest.raises(LeakageError):  # a sentence at the cutoff leaks
        build_centroids(sentences, emb, 3, 2, cutoff=40, min_rating=4, chunk_rows=2)
    with pytest.raises(ValueError, match="id map"):
        build_centroids(sentences, emb, 2, 2, cutoff=100, min_rating=4, chunk_rows=2)


def test_centroids_with_no_sentences() -> None:
    empty = pd.DataFrame({"sent_id": [], "u": [], "i": [], "ts": [], "rating": []}, dtype=np.int64)
    item, user, counts = build_centroids(empty, np.zeros((0, 4), np.float16), 2, 3, 5, 4, 10)
    assert item.shape == (2, 4) and not item.any() and user.shape == (3, 4) and not user.any()
    assert counts["sentences_used"] == 0 and counts["items_with_centroid"] == 0
