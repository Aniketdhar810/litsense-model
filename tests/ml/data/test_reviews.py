"""data.reviews: review text for the 5-core ratings of kept works, joined on parent_asin and
deduplicated per (user, work) and per (user, text) (CLAUDE.md s7.5, s16.2)."""

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

import ml.data.canonical  # noqa: F401  (registers the stage)
import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.goodreads  # noqa: F401  (registers the stage)
import ml.data.meta  # noqa: F401  (registers the stage)
import ml.data.reviews as reviews
from ml.common.errors import DataError
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all

Config = dict[str, Any]


def test_text_hash_ignores_markup_case_and_punctuation() -> None:
    assert reviews.text_hash("Great read!<br />Loved it.") == reviews.text_hash(
        "great read loved it"
    )
    assert reviews.text_hash("Great read") != reviews.text_hash("Great reads")
    assert reviews.text_hash("") is None and reviews.text_hash("<br/> !!") is None


def test_review_row_rejects_second_timestamps() -> None:
    record = {"user_id": "u", "parent_asin": "a", "rating": 5.0, "timestamp": 1546300800}
    with pytest.raises(DataError, match="milliseconds"):
        reviews.review_row(record, "am_a")


def test_dedupe_keeps_latest_per_work_then_one_per_text() -> None:
    keys = pd.DataFrame(
        [
            ("u1", "w1", 100, "h1"),  # same work twice: the later one wins
            ("u1", "w1", 200, "h2"),
            ("u1", "w2", 150, "h2"),  # same text pasted on another work: dropped
            ("u1", "w3", 160, None),  # empty texts are never text duplicates
            ("u1", "w4", 170, None),
            ("u2", "w2", 100, "h2"),  # another user may write the same text
        ],
        columns=reviews.KEY_COLUMNS,
    )
    kept, counts = reviews.dedupe(keys)
    assert list(kept) == [1, 3, 4, 5]
    assert counts == {"dropped_same_work": 1, "dropped_copied_text": 1}


def test_stage_on_the_tiny_fixture(tiny_cfg: Config, tiny_dataset: Path) -> None:
    run_all(tiny_cfg, until="data.reviews")
    truth = json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))
    paths = RunPaths.from_config(tiny_cfg, None)
    table = pq.read_table(paths.resolve("data:reviews_raw.parquet"))
    assert table.schema.equals(reviews.REVIEW_SCHEMA)
    rows = table.to_pandas()
    mapping = pd.read_parquet(paths.resolve("data:asin_to_canonical.parquet"))
    kept = mapping.loc[mapping["status"] == "kept", "parent_asin"]
    with gzip.open(tiny_dataset / "raw" / "Books.csv.gz", "rt") as f:
        ratings = pd.read_csv(f)
    rated = set(zip(ratings["user_id"], ratings["parent_asin"], strict=True))

    assert rows["parent_asin"].isin(kept).all()
    assert all(pair in rated for pair in zip(rows["user_id"], rows["parent_asin"], strict=True))
    assert not rows["user_id"].isin(truth["users_outside_5core"]).any()
    assert not rows.duplicated(["user_id", "canonical_id"]).any()
    keys = set(rows["user_id"] + "|" + rows["parent_asin"])
    assert not keys & set(truth["pairs_without_review"])
    child = truth["special_reviews"]["child_asin"]
    assert child in keys  # its review is filed under a child asin; the join uses parent_asin

    first, second = truth["special_reviews"]["cross_edition_duplicate"]  # same text, two editions
    user = first.split("|")[0]
    same_work = rows[
        (rows["user_id"] == user)
        & rows["parent_asin"].isin([k.split("|")[1] for k in (first, second)])
    ]
    assert len(same_work) == 1 and same_work["parent_asin"].iloc[0] == second.split("|")[1]
    assert rows["ts"].between(8e11, 1.8e12).all()

    counts = read_manifest(paths.manifest("data.reviews"))["notes"]["counts"]
    assert counts["reviews"] == len(rows)
    assert counts["dropped_same_work"] >= 1
    assert counts["pairs_without_review"] >= 1
    assert (
        counts["5core_pairs_of_kept_works"]
        == counts["pairs_with_review"] + counts["pairs_without_review"]
    )
