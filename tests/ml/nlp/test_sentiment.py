"""Clause splitting and per-aspect polarity (CLAUDE.md s11.4)."""

import numpy as np
import pytest

from ml.nlp import aspects, sentiment
from ml.nlp.aspect_seeds import ASPECTS
from tests.ml.nlp.stubs import StubEncoder, stub_polarity

A = {name: k for k, name in enumerate(ASPECTS)}
MARKERS = [" but ", " although ", " though ", " however ", " yet ", " whereas "]
MIXED = "Wonderful characters, but the middle third really dragged."


def _lexicon_only_detector(dim: int = 16) -> aspects.AspectDetector:
    """Zero prototypes: cosines are 0, so only the lexicon can fire."""
    return aspects.AspectDetector(
        np.zeros((12, 8, dim), dtype=np.float32),
        np.zeros((3, 2, dim), dtype=np.float32),
        np.full(12, 0.5, dtype=np.float32),
        dim,
    )


def test_split_clauses() -> None:
    markers = sentiment.compile_markers(MARKERS)
    assert sentiment.split_clauses(MIXED, markers) == [
        "Wonderful characters",
        "the middle third really dragged.",
    ]
    assert sentiment.split_clauses("Great BUT slow, yet moving.", markers) == [
        "Great",
        "slow",
        "moving.",
    ]
    assert sentiment.split_clauses("No marker in this one.", markers) == ["No marker in this one."]
    assert sentiment.split_clauses("Butter is but a spread", markers) == ["Butter is", "a spread"]


def _polarities(texts: list[str]) -> np.ndarray:
    det = _lexicon_only_detector()
    encoder = StubEncoder()
    hits = det.detect(encoder.encode(texts), texts).hits
    assert hits.any(axis=1).all(), "every test sentence must hit an aspect"
    return sentiment.aspect_polarities(
        np.arange(len(texts)),
        texts,
        hits,
        stub_polarity,
        det,
        encoder,
        sentiment.compile_markers(MARKERS),
    )


def test_mixed_sentence_gives_characters_positive_and_pacing_negative() -> None:  # s16.2 NLP
    rows = _polarities([MIXED])
    by_aspect = {int(r["aspect"]): r for r in rows}
    assert set(by_aspect) == {A["characters"], A["pacing"]}
    assert by_aspect[A["characters"]]["polarity"] > 0 > by_aspect[A["pacing"]]["polarity"]
    assert all(bool(r["from_clause"]) for r in rows)


def test_single_aspect_sentences_use_the_sentence_polarity() -> None:
    rows = _polarities(["The plot kept me guessing until the very last page."])
    assert len(rows) == 1 and rows[0]["aspect"] == A["plot"]
    assert rows[0]["polarity"] == pytest.approx(0.9) and not rows[0]["from_clause"]


def test_aspect_no_clause_mentions_falls_back_to_the_sentence() -> None:
    # Two aspects (plot, humor) but a clause split in which no clause mentions humor; "funny"
    # sits in the first clause together with "story" so both clauses are checked.
    text = "A funny story, but the dragged middle was tedious."
    rows = _polarities([text])
    by_aspect = {int(r["aspect"]): r for r in rows}
    assert set(by_aspect) == {A["plot"], A["humor"], A["pacing"]}
    assert by_aspect[A["pacing"]]["polarity"] == pytest.approx(-0.9)
    assert by_aspect[A["plot"]]["from_clause"] and by_aspect[A["humor"]]["from_clause"]
    # a two-aspect sentence without any marker is never split
    rows = _polarities(["Wonderful characters and sharp dialogue."])
    assert len(rows) == 2 and not any(bool(r["from_clause"]) for r in rows)
    assert all(r["polarity"] == pytest.approx(0.9) for r in rows)


def test_empty_chunk() -> None:
    rows = sentiment.aspect_polarities(
        np.zeros(0, dtype=np.int64),
        [],
        np.zeros((0, 12), dtype=bool),
        stub_polarity,
        _lexicon_only_detector(),
        StubEncoder(),
        sentiment.compile_markers(MARKERS),
    )
    assert rows.shape == (0,) and rows.dtype == sentiment.PART_DTYPE


@pytest.mark.slow
def test_real_models_on_the_mixed_sentence(repo_config: dict) -> None:  # s16.2 NLP
    from ml.nlp import embed

    content, nlp = repo_config["content"], repo_config["nlp"]
    encoder = embed.load_encoder(content["model_name"], 32, int(content["max_seq_length"]))
    det = aspects.AspectDetector(
        aspects.embed_prototypes(encoder),
        aspects.embed_poles(encoder),
        aspects.tau_vector(repo_config["aspects"]),
        int(nlp["embedding_dim"]),
    )
    polarity = sentiment.load_polarity_model(
        nlp["sentiment_model"], int(nlp["sentiment_batch"]), int(nlp["sentiment_max_length"])
    )
    hits = det.detect(encoder.encode([MIXED]), [MIXED]).hits
    rows = sentiment.aspect_polarities(
        np.array([0]), [MIXED], hits, polarity, det, encoder,
        sentiment.compile_markers(nlp["clause_split_markers"]),
    )  # fmt: skip
    by_aspect = {int(r["aspect"]): float(r["polarity"]) for r in rows}
    assert by_aspect[A["characters"]] > 0 > by_aspect[A["pacing"]]
