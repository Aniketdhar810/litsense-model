"""Sentence table pieces (CLAUDE.md s11.1), checkpoint parts and cutoff helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.eval.errors import LeakageError
from ml.nlp import cutoffs, sentences
from ml.nlp.parts import PartStore
from tests.ml.nlp.stubs import DIM, StubEncoder, stub_splitter


def test_keep_sentence_bounds() -> None:
    assert sentences.keep_sentence("one two three four", 4, 80)
    assert not sentences.keep_sentence("one two three", 4, 80)
    assert not sentences.keep_sentence(" ".join(["w"] * 81), 4, 80)


def test_sentence_rows_keep_review_positions() -> None:
    reviews = pd.DataFrame(
        {
            "review_row": [7, 9],
            "u": [1, 2],
            "i": [3, 4],
            "ts": [100, 200],
            "rating": [5.0, 2.0],
            "helpful_vote": [0, 3],
            "text": [
                "Short. The plot kept me guessing until the very last page. Wow!",
                "The writing was clunky and amateurish.",
            ],
        }
    )
    rows, dropped = sentences.sentence_rows(reviews, stub_splitter, 4, 80, first_id=10)
    assert dropped == 2  # "Short." and "Wow!"
    assert rows["sent_id"].tolist() == [10, 11]
    assert rows["pos"].tolist() == [1, 0]  # index among every sentence of the review
    assert rows["n_in_review"].tolist() == [3, 1]
    assert rows["review_row"].tolist() == [7, 9] and rows["helpful_vote"].tolist() == [0, 3]
    assert list(rows.columns) == sentences.SENTENCE_SCHEMA.names


def test_part_store_resumes_and_rejects_bad_parts(tmp_path: Path) -> None:
    parts = PartStore(tmp_path, "emb", n_rows=10, chunk_rows=4)
    assert parts.n_parts == 3 and parts.bounds(2) == (8, 10)
    assert parts.first_pending() == 0
    parts.save(0, np.ones((4, 2), dtype=np.float16))
    with pytest.raises(ValueError, match="expected 4"):
        parts.save(1, np.ones((3, 2), dtype=np.float16))
    parts.save(1, np.full((4, 2), 2, dtype=np.float16))
    assert parts.first_pending() == 2
    parts.path(1).write_bytes(parts.path(1).read_bytes()[:-5])  # a part lost by Drive
    assert parts.first_pending() == 1 and not parts.path(1).exists()
    parts.save(1, np.full((4, 2), 2, dtype=np.float16))
    parts.save(2, np.full((2, 2), 3, dtype=np.float16))
    out = np.zeros((10, 2), dtype=np.float16)
    parts.concat(out)
    assert out[:, 0].tolist() == [1] * 4 + [2] * 4 + [3] * 2
    with pytest.raises(ValueError, match="rows"):
        parts.concat(np.zeros((9, 2), dtype=np.float16))
    loose = PartStore(tmp_path / "loose", "p", n_rows=10, chunk_rows=4, fixed_rows=False)
    loose.save(0, np.arange(7))  # a long table part may hold any number of rows
    assert loose.is_done(0) and loose.first_pending() == 1
    with pytest.raises(ValueError, match="fixed-size"):
        loose.concat(np.zeros(10))


def test_embed_sentences_resumes_after_the_last_good_part(tmp_path: Path) -> None:
    texts = [f"sentence number {k} of the review" for k in range(10)]
    encoder = StubEncoder()
    parts = PartStore(tmp_path, "emb", n_rows=10, chunk_rows=4)
    parts.save(0, encoder.encode(texts[:4]).astype(np.float16))
    encoder.calls.clear()
    sentences.embed_sentences(texts, encoder, parts, DIM)
    assert encoder.calls == [4, 2]  # part 0 was reused
    out = np.zeros((10, DIM), dtype=np.float16)
    parts.concat(out)
    assert np.allclose(out.astype(np.float32), StubEncoder().encode(texts), atol=1e-3)
    with pytest.raises(ValueError, match="nlp.embedding_dim"):
        sentences.embed_sentences(texts, encoder, PartStore(tmp_path / "x", "e", 10, 4), DIM + 1)


def test_cutoffs() -> None:
    splits = {"t1": {"ms": 1_000}, "t2": {"ms": 2_000}}
    assert cutoffs.cutoff_for("eval_val", splits, 5_000) == 1_000
    assert cutoffs.cutoff_for("eval_test", splits, 5_000) == 2_000
    assert cutoffs.cutoff_for("serve", splits, 5_000) == 5_001
    frame = pd.DataFrame({"ts": [500, 999, 1_000, 1_500]})
    assert cutoffs.rows_before(frame, 1_000)["ts"].tolist() == [500, 999]
    cutoffs.check_cutoff(frame["ts"][:2], 1_000)
    cutoffs.check_cutoff(np.array([], dtype=np.int64), 1_000)
    with pytest.raises(LeakageError, match="at or after the cutoff"):
        cutoffs.check_cutoff(frame["ts"], 1_000)
