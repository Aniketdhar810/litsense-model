"""data.meta: parse_meta (CLAUDE.md s16.2), the streamed stage on the tiny fixture, the slice
proposal, the STOP guard, and resuming after a crash."""

import copy
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.meta as meta
from ml.common.errors import ConfigError
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all, run_stage
from ml.data.parsing import isbn10_to_13

Config = dict[str, Any]
RECORD = {
    "main_category": "Books",
    "title": "Chaucer",
    "subtitle": "Hardcover – Import, January 1, 2004",
    "author": {"avatar": "a.jpg", "name": "Peter Ackroyd", "about": []},
    "categories": ["Books", "Literature & Fiction", "History & Criticism"],
    "description": [],
    "features": ["A <b>life</b> of Chaucer."],
    "details": {
        "Publisher": "Chatto & Windus; First Edition (January 1, 2004)",
        "Language": "English",
        "Hardcover": "196 pages",
        "ISBN 10": "0701169850",
        "ISBN 13": "978-0701169855",
    },
    "images": [{"large": "c.jpg", "variant": "MAIN"}],
    "store": "Peter Ackroyd (Author)",
    "parent_asin": "0701169850",
}


def test_parse_meta_real_layout() -> None:
    book = meta.parse_meta(RECORD)
    assert book == {
        "parent_asin": "0701169850",
        "title": "Chaucer",
        "subtitle": "Hardcover – Import, January 1, 2004",
        "author": "Peter Ackroyd",
        "categories": ["Books", "Literature & Fiction", "History & Criticism"],
        "main_category": "Books",
        "description": None,
        "features": "A life of Chaucer.",
        "isbn10": "0701169850",
        "isbn13": "9780701169855",  # dashes removed
        "page_source_isbn13": None,
        "pages": 196,
        "format": "Hardcover",
        "edition_kind": "print",
        "year": 2004,
        "language": "English",
        "image": "c.jpg",
    }


def test_parse_meta_variants() -> None:
    as_strings = RECORD | {
        "details": json.dumps(RECORD["details"]),
        "author": json.dumps(RECORD["author"]),
    }
    assert meta.parse_meta(as_strings) == meta.parse_meta(RECORD)
    store_only = meta.parse_meta(RECORD | {"author": None, "store": "Ann Lee (Author)"})
    assert store_only is not None and store_only["author"] == "Ann Lee"
    no_pages = meta.parse_meta(RECORD | {"details": {"Publication date": "May 1, 2019"}})
    assert no_pages is not None and no_pages["pages"] is None and no_pages["year"] == 2019
    kindle = meta.parse_meta(
        RECORD
        | {
            "main_category": "Buy a Kindle",
            "subtitle": "Kindle Edition",
            "parent_asin": "B00KFOP3RG",
            "details": {
                "File size": "2 MB",
                "Print length": "352 pages",
                "Page numbers source ISBN": "0701169850",
            },
        }
    )
    assert kindle is not None
    assert (kindle["edition_kind"], kindle["isbn10"], kindle["isbn13"]) == ("kindle", None, None)
    assert (kindle["page_source_isbn13"], kindle["pages"]) == ("9780701169855", 352)
    assert meta.parse_meta(RECORD | {"title": " <br> "}) is None


def test_stage_on_the_tiny_fixture(tiny_cfg: Config, tiny_dataset: Path) -> None:
    assert run_all(tiny_cfg, until="data.meta") == ["data.download", "data.lid", "data.meta"]
    truth = json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))
    paths = RunPaths.from_config(tiny_cfg, None)
    table = pq.read_table(paths.resolve("data:books_raw.parquet"))
    assert table.schema.equals(meta.BOOK_SCHEMA)
    parquet = pq.ParquetFile(paths.resolve("data:books_raw.parquet"))
    assert parquet.metadata.row_group(0).column(0).compression == "ZSTD"
    books = table.to_pandas().set_index("parent_asin")

    dropped = {*truth["non_narrative"], *truth["too_short"], *truth["empty_title"]}
    assert set(books.index) == set(truth["books"]) - dropped
    counts = read_manifest(paths.manifest("data.meta"))["notes"]["counts"]
    assert counts == {
        "5core_items": 80,
        "meta_lines_read": 85,
        "meta_matched": 80,
        "dropped_title_blocklist": 2,
        "dropped_min_pages": 1,
        "dropped_no_title": 1,
        "duplicate_parent_asins": 0,
        "books_raw": 76,
    }

    for asin, source in truth["kindle_editions"].items():  # F1: kept and tagged, linkable
        assert books.loc[asin, "edition_kind"] == "kindle"
        assert books.loc[asin, "page_source_isbn13"] == isbn10_to_13(source)
        assert books.loc[asin, "author"] in {"Daniel Okoro", "Imogen Hale"}
    assert books.loc[truth["audiobooks"][0], "edition_kind"] == "audio"
    for asin, value in truth["isbn13_in_isbn10_field"].items():
        assert books.loc[asin, "isbn13"] == value
    for asin, name in {**truth["store_author"], **truth["author_without_name"]}.items():
        assert books.loc[asin, "author"] == name

    ratings = pd.read_csv(tiny_dataset / "raw" / "Books.csv.gz")
    per_item = ratings["parent_asin"].value_counts()
    assert (books["n_interactions"] == per_item.reindex(books.index)).all()
    cats = pd.read_csv(paths.resolve("data:category_counts.csv"), keep_default_na=False)
    for level in ("top", "pair"):
        rows = cats[cats["level"] == level]
        assert rows["n_interactions"].sum() == books["n_interactions"].sum()
        assert rows["n_items"].sum() == len(books)
    kindle_total = books.loc[books["edition_kind"] == "kindle", "n_interactions"].sum()
    assert cats.loc[cats["level"] == "top", "n_interactions_kindle"].sum() == kindle_total
    assert "<none>" in set(cats["category"])  # the book without categories

    proposal = json.loads(paths.resolve("data:genre_slice_proposal.json").read_text("utf-8"))
    assert proposal["meets_threshold"] is False  # 600 ratings cannot reach 1,000,000
    assert len(proposal["proposed"]) == 3
    assert {c["category"] for c in proposal["candidates"]} == set(
        tiny_cfg["data"]["fiction_allowlist"]
    )


def test_proposal_rule() -> None:
    counts = pd.DataFrame(
        {
            "level": "top",
            "category": [
                "Romance",
                "Literature & Fiction",
                "Mystery, Thriller & Suspense",
                "Cooking",
            ],
            "n_items": [10, 20, 30, 40],
            "n_interactions": [900_000, 700_000, 400_000, 5_000_000],
        }
    )
    allow = ["Literature & Fiction", "Romance", "Mystery, Thriller & Suspense"]
    rule = {"min_slices": 2, "max_slices": 3, "target_multiple": 4}
    two = meta.propose_genre_slices(counts, allow, 250_000, rule)
    assert two["proposed"] == ["Romance", "Literature & Fiction"] and two["meets_threshold"]
    three = meta.propose_genre_slices(counts, allow, 450_000, rule)
    assert len(three["proposed"]) == 3 and three["meets_threshold"]
    none = meta.propose_genre_slices(counts, allow, 600_000, rule)
    assert len(none["proposed"]) == 3 and not none["meets_threshold"]
    assert none["threshold"] == 2_400_000


def test_require_genre_slices(repo_config: Config) -> None:
    assert meta.require_genre_slices(repo_config["data"]["genre_slices"]) == [
        "Literature & Fiction",
        "Mystery, Thriller & Suspense",
        "Science Fiction & Fantasy",
    ]  # the user's choice (D26)
    for unset in ([], None, [""]):
        with pytest.raises(ConfigError, match="STOP: choose data.genre_slices"):
            meta.require_genre_slices(unset)


def test_stops_with_a_report(
    tiny_cfg: Config,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_all(tiny_cfg, until="data.meta")
    assert meta.STOP_MESSAGE in caplog.text and "Proposal (" in caplog.text
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(tiny_cfg), encoding="utf-8")
    assert meta.main(["--config", str(config_file)]) == 0
    printed = capsys.readouterr().out
    assert "Top-level categories" in printed and meta.STOP_MESSAGE in printed


def test_sample_mode_reads_the_local_head(tiny_cfg: Config, tiny_dataset: Path) -> None:
    head = Path(tiny_cfg["project"]["local_raw_root"])
    head.mkdir(parents=True)
    with gzip.open(tiny_dataset / "raw" / "meta_Books.jsonl.gz", "rt", encoding="utf-8") as f:
        lines = [next(f) for _ in range(30)]
    with gzip.open(head / "meta_Books.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(lines)
    run_all(tiny_cfg, until="data.meta", sample=12)
    notes = read_manifest(RunPaths.from_config(tiny_cfg, 12).manifest("data.meta"))["notes"]
    assert notes["counts"]["meta_lines_read"] == 12
    assert notes["source"]["source"] == str(head / "meta_Books.jsonl.gz")


def test_resumes_after_a_crash(
    tiny_cfg: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean_cfg = copy.deepcopy(tiny_cfg)
    clean_cfg["project"]["drive_root"] = str(tmp_path / "clean_drive")
    run_all(clean_cfg, until="data.meta")

    real_parse, calls = meta.parse_meta, []

    def crashing(record: dict[str, Any]) -> dict[str, Any] | None:
        calls.append(record["parent_asin"])
        if len(calls) == 45:
            raise RuntimeError("colab disconnected")
        return real_parse(record)

    monkeypatch.setattr(meta, "parse_meta", crashing)
    with pytest.raises(RuntimeError):
        run_all(tiny_cfg, until="data.meta")
    paths = RunPaths.from_config(tiny_cfg, None)
    progress = json.loads((paths.work_dir("data.meta") / "progress.json").read_text())
    assert progress["parts"] >= 3
    monkeypatch.setattr(meta, "parse_meta", real_parse)
    run_stage("data.meta", tiny_cfg)
    clean_paths = RunPaths.from_config(clean_cfg, None)
    resumed = pd.read_parquet(paths.resolve("data:books_raw.parquet"))
    clean = pd.read_parquet(clean_paths.resolve("data:books_raw.parquet"))
    assert resumed.equals(clean)
    assert (
        read_manifest(paths.manifest("data.meta"))["notes"]["counts"]
        == read_manifest(clean_paths.manifest("data.meta"))["notes"]["counts"]
    )
    assert not paths.work_dir("data.meta").exists()


def test_find_titles(
    tiny_cfg: Config, tiny_dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_all(tiny_cfg, until="data.meta")
    truth = json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))
    books = pd.read_parquet(RunPaths.from_config(tiny_cfg, None).resolve("data:books_raw.parquet"))
    found = meta.find_titles(books, ["salt orchard", "No Such Book"])
    salt = found[found["query"] == "salt orchard"]
    expected = {*truth["editions_goodreads"][0], *truth["kindle_editions"]} - {
        a for a, src in truth["kindle_editions"].items() if src == truth["series"]["novella_isbn10"]
    }
    assert set(salt["parent_asin"]) == expected  # hardcover, paperback and Kindle editions
    assert list(salt["n_interactions"]) == sorted(salt["n_interactions"], reverse=True)
    assert set(salt["category"]) == {"Literature & Fiction"}
    missing = found[found["query"] == "No Such Book"]
    assert len(missing) == 1 and missing["n_interactions"].iloc[0] == 0

    config_file = tiny_dataset.parent / "find_config.yaml"
    config_file.write_text(yaml.safe_dump(tiny_cfg), encoding="utf-8")
    assert meta.main(["--config", str(config_file), "--find", "Salt Orchard"]) == 0
    assert "'Salt Orchard':" in capsys.readouterr().out
