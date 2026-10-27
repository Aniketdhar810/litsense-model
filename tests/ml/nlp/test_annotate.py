"""Annotation sample export (CLAUDE.md s11.5 step 1)."""

import numpy as np
import pandas as pd
import pytest

from ml.nlp import annotate
from ml.nlp.aspect_seeds import ASPECTS


def _tables(n: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    sentences = pd.DataFrame({"sent_id": np.arange(n), "text": [f"sentence {k}" for k in range(n)]})
    masks = pd.DataFrame(
        {"sent_id": np.arange(n), "aspect_mask": (np.arange(n) % 2).astype("uint16")}
    )
    return sentences, masks


def test_sample_has_the_requested_mix_and_blank_label_columns() -> None:
    sentences, masks = _tables()
    out = annotate.sample_for_annotation(sentences, masks, 12, 8, np.random.default_rng(0))
    assert list(out.columns) == list(annotate.LABEL_COLUMNS) and len(out) == 12
    assert out["sent_id"].is_unique
    assert (out["sent_id"] % 2 == 1).sum() == 8  # odd ids carry a hit
    assert (out[list(ASPECTS)] == "").all().all()
    again = annotate.sample_for_annotation(sentences, masks, 12, 8, np.random.default_rng(0))
    assert again["sent_id"].tolist() == out["sent_id"].tolist()  # seeded
    assert not out["sent_id"].is_monotonic_increasing  # shuffled, hits and misses mixed


def test_sample_refuses_small_pools() -> None:
    sentences, masks = _tables(10)
    with pytest.raises(ValueError, match="have 5 and 5"):
        annotate.sample_for_annotation(sentences, masks, 12, 8, np.random.default_rng(0))
    with pytest.raises(ValueError, match="between 0 and total"):
        annotate.sample_for_annotation(sentences, masks, 4, 5, np.random.default_rng(0))
