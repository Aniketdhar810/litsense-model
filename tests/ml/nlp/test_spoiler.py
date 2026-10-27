"""Spoiler scorer (CLAUDE.md s11.7, s16.2) and the training helpers."""

import json
from pathlib import Path

import numpy as np
import pytest

from litsense_core.spoiler import SpoilerScorer
from ml.nlp import spoiler_train
from tests.ml.nlp.stubs import StubEncoder

PATTERNS = [r"\bdies\b", r"\bdied\b", "turns out", "the killer", r"\bmarries\b", "in the end"]
SPOILER = "It turns out the narrator was dead the whole time."
SAFE = "A young girl discovers a hidden world beneath the city."


def test_regex_only_scorer_flags_by_pattern(tmp_path: Path) -> None:  # s16.2 NLP
    scorer = SpoilerScorer.regex_only(PATTERNS, 0.15, 0.05, 0.3)
    emb = StubEncoder().encode([SPOILER, SAFE])
    assert not scorer.has_model and scorer.probability(emb).tolist() == [0.0, 0.0]
    flags = scorer.flags(emb, [SPOILER, SAFE])
    assert flags.tolist() == [True, False]
    # a late position alone never flags in regex-only mode (0.05 < 0.15)
    late = scorer.flags(emb, [SAFE, SAFE], np.array([9, 0]), np.array([10, 10]))
    assert late.tolist() == [False, False]
    assert scorer.late(np.array([6, 7, 0]), np.array([10, 10, 1])).tolist() == [False, True, True]
    path = tmp_path / "spoiler_lr.npz"
    scorer.to_npz(path)
    again = SpoilerScorer.from_npz(path)
    assert not again.has_model and again.patterns == PATTERNS
    assert again.threshold == pytest.approx(0.15, abs=1e-6)
    assert again.flags(emb, [SPOILER, SAFE]).tolist() == [True, False]


def test_model_scorer_adds_boosts_and_roundtrips(tmp_path: Path) -> None:
    coef = np.zeros(16, dtype=np.float32)
    coef[0] = 4.0
    scorer = SpoilerScorer(coef, -2.0, 0.6, PATTERNS, 0.15, 0.05, 0.3)
    emb = np.zeros((3, 16), dtype=np.float32)
    emb[0, 0] = 1.0  # z = 2 -> p 0.88
    emb[1, 0] = 0.5  # z = 0 -> p 0.5
    p = scorer.probability(emb)
    assert p[0] == pytest.approx(1 / (1 + np.exp(-2)), abs=1e-5) and p[1] == pytest.approx(0.5)
    texts = [SAFE, SAFE, SAFE]
    assert scorer.flags(emb, texts).tolist() == [True, False, False]
    # 0.5 + late boost 0.05 is still under 0.6; 0.5 + regex 0.15 passes
    assert scorer.flags(emb, texts, np.array([0, 9, 0]), np.array([10, 10, 10])).tolist() == [
        True,
        False,
        False,
    ]
    assert scorer.flags(emb, [SAFE, SPOILER, SAFE]).tolist() == [True, True, False]
    scorer.to_npz(tmp_path / "m.npz")
    again = SpoilerScorer.from_npz(tmp_path / "m.npz")
    assert again.has_model and np.allclose(again.coef, coef) and again.intercept == -2.0
    with pytest.raises(ValueError, match="dim"):
        scorer.probability(np.zeros((1, 8), dtype=np.float32))


def test_training_helpers() -> None:
    record = {"book_id": "b1", "review_sentences": [[0, "a b c d e"], [1, "f g h i j"]]}
    assert spoiler_train.review_sentences(record) == [(0, "a b c d e"), (1, "f g h i j")]
    with pytest.raises(ValueError, match="unexpected"):
        spoiler_train.review_sentences({"review_sentences": [["yes", "x"]]})
    lines = [
        json.dumps(
            {
                "book_id": f"b{k % 5}",
                "review_sentences": [
                    [0, f"plain sentence number {k} here"],
                    [1, f"spoiler sentence number {k} here"],
                ],
            }
        )
        for k in range(50)
    ]
    rng = np.random.default_rng(0)
    sample, counts = spoiler_train.sample_sentences(lines, 20, rng, 4, 80)
    assert counts["reviews_streamed"] == 50 and counts["sentences_seen_spoiler"] == 50
    assert (sample["label"] == 1).sum() == 10 and (sample["label"] == 0).sum() == 10
    assert (
        sample["text"].str.startswith("spoiler").to_numpy().tolist()
        == (sample["label"] == 1).tolist()
    )
    held = spoiler_train.group_holdout(sample["book_id"].to_numpy(), 0.4, np.random.default_rng(1))
    held_books = set(sample.loc[held, "book_id"])
    assert len(held_books) == 2 and not (held_books & set(sample.loc[~held, "book_id"]))
    probs = np.array([0.9, 0.8, 0.3, 0.1, 0.7, 0.2])
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert spoiler_train.recall_threshold(probs, labels, 0.9) == pytest.approx(
        0.3
    )  # all 3 positives
    assert spoiler_train.recall_threshold(probs, labels, 0.6) == pytest.approx(
        0.8
    )  # 2 of 3 suffice
    metrics = spoiler_train.holdout_metrics(probs, labels, 0.3)
    assert metrics["recall"] == 1.0 and metrics["precision"] == pytest.approx(0.75)
    assert metrics["flagged_share"] == pytest.approx(4 / 6, abs=1e-4)
    assert metrics["auc"] == pytest.approx(8 / 9, abs=1e-4)  # 8 of 9 (pos, neg) pairs ordered
    # a tie counts half, as in sklearn: pairs 0.5 vs 0.5 (half), 0.5 > 0.1, 0.9 > both
    assert spoiler_train.roc_auc(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.9, 0.1])) == 0.875
    assert np.isnan(spoiler_train.roc_auc(np.array([1, 1]), np.array([0.2, 0.9])))
