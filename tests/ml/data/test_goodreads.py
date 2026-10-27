"""data.goodreads: off by default (no network, empty tables); when enabled it keeps only Goodreads
books whose ISBN matches books_raw, including Kindle editions' page-source ISBNs."""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.goodreads as goodreads
import ml.data.meta  # noqa: F401  (registers the stage)
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all
from ml.data.parsing import isbn10_to_13
from tests.ml.data.conftest import FileServer

Config = dict[str, Any]


def test_disabled_by_default_and_offline(tiny_cfg: Config, file_server: FileServer) -> None:
    assert tiny_cfg["data"]["use_goodreads_graph"] is False
    run_all(tiny_cfg, until="data.goodreads")
    paths = RunPaths.from_config(tiny_cfg, None)
    index = pq.read_table(paths.resolve("data:gr_isbn_index.parquet"))
    series = pq.read_table(paths.resolve("data:gr_series.parquet"))
    assert index.num_rows == series.num_rows == 0
    assert index.schema.equals(goodreads.GR_INDEX_SCHEMA)
    assert series.schema.equals(goodreads.GR_SERIES_SCHEMA)
    assert not [name for name, _ in file_server.requests if name.startswith("goodreads")]
    assert read_manifest(paths.manifest("data.goodreads"))["notes"]["enabled"] is False


def test_enabled_keeps_only_books_raw_isbns(tiny_cfg: Config, tiny_dataset: Path) -> None:
    tiny_cfg["data"]["use_goodreads_graph"] = True
    run_all(tiny_cfg, until="data.goodreads")
    truth = json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))
    paths = RunPaths.from_config(tiny_cfg, None)
    table = pq.read_table(paths.resolve("data:gr_isbn_index.parquet"))
    assert table.schema.equals(goodreads.GR_INDEX_SCHEMA)
    index = table.to_pandas()
    books = truth["books"]

    novella = index[index["title"] == truth["series"]["novella_only_in_goodreads"]]
    assert len(novella) == 1  # found only through a Kindle edition's page-source ISBN
    assert novella["isbn13"].iloc[0] == isbn10_to_13(truth["series"]["novella_isbn10"])
    assert novella["series_pos"].iloc[0] == 1.5
    assert len(index) == len(truth["goodreads_matched"]) + 1
    assert not index["title"].str.startswith("Unrelated").any()

    for asin, position in truth["series"]["volumes"].items():
        row = index[index["isbn13"] == isbn10_to_13(books[asin]["isbn10"])].iloc[0]
        assert (row["series_name"], row["series_pos"]) == ("The Ember Cycle", position)
        assert row["series_id"] == truth["series"]["series_id"]
        assert row["work_title"] == books[asin]["title"].split(" (")[0].split(":")[0]
    box = index[index["isbn13"] == isbn10_to_13(books[truth["boxsets"][0]]["isbn10"])].iloc[0]
    assert box["series_name"] == "The Ember Cycle" and box["series_pos"] != box["series_pos"]  # NaN
    pair = [isbn10_to_13(books[a]["isbn10"]) for a in truth["editions_goodreads"][0]]
    assert index.set_index("isbn13").loc[pair, "work_id"].nunique() == 1
    for asin in truth["editions_heuristic"][0]:
        assert isbn10_to_13(books[asin]["isbn10"]) not in set(index["isbn13"])
    for asin in truth["goodreads_isbn10_only"] + truth["goodreads_isbn13_only"]:
        assert isbn10_to_13(books[asin]["isbn10"]) in set(index["isbn13"])

    series = pq.read_table(paths.resolve("data:gr_series.parquet")).to_pylist()
    assert series == [
        {
            "series_id": "90001",
            "title": "The Ember Cycle",
            "numbered": True,
            "primary_work_count": 3,
        }
    ]
    counts = read_manifest(paths.manifest("data.goodreads"))["notes"]["counts"]
    assert counts["gr_rows_matched"] == len(index) and counts["gr_series"] == 1
    assert counts["books_raw_matched"] == len(truth["goodreads_matched"]) + 2  # + both Kindles
