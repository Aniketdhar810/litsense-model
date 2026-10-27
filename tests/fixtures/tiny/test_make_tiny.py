"""The tiny fixture is deterministic, well-formed, and every planted case is really there."""

import csv
import gzip
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from ml.common.manifest import sha256_file
from tests.fixtures.tiny.make_tiny import make_tiny

SERIES_RE = re.compile(r"\(([^()]+?),?\s*#(\d+(?:\.\d+)?)\)\s*$")  # CLAUDE.md s7.3
Config = dict[str, Any]


def lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read().splitlines()


def records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in lines(path)]


@pytest.fixture(scope="module")
def raw(tiny_dataset: Path) -> Path:
    return tiny_dataset / "raw"


@pytest.fixture(scope="module")
def truth(tiny_dataset: Path) -> dict[str, Any]:
    return json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ratings(raw: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO("\n".join(lines(raw / "Books.csv.gz")))))


@pytest.fixture(scope="module")
def meta(raw: Path) -> dict[str, dict[str, Any]]:
    return {m["parent_asin"]: m for m in records(raw / "meta_Books.jsonl.gz")}


def test_generation_is_byte_identical(tiny_dataset: Path, tmp_path: Path) -> None:
    make_tiny(tmp_path)
    for path in sorted((tiny_dataset / "raw").iterdir()):
        assert sha256_file(path) == sha256_file(tmp_path / "raw" / path.name), path.name
    assert sha256_file(tiny_dataset / "truth.json") == sha256_file(tmp_path / "truth.json")


def test_raw_files_use_pipeline_names(raw: Path, repo_config: Config) -> None:
    names = {p.name for p in raw.iterdir()}
    assert {"Books.csv.gz", "meta_Books.jsonl.gz", "Books.jsonl.gz"} <= names
    assert set(repo_config["sources"]["goodreads_files"]) <= names


def test_ratings_shape(raw: Path, ratings: list[dict[str, str]]) -> None:
    assert lines(raw / "Books.csv.gz")[0] == "user_id,parent_asin,rating,timestamp"
    pairs = {(r["user_id"], r["parent_asin"]) for r in ratings}
    assert len(ratings) == len(pairs) == 600
    per_user = Counter(r["user_id"] for r in ratings)
    assert len(per_user) == 50 and min(per_user.values()) >= 5
    assert len({r["parent_asin"] for r in ratings}) == 80
    ts = [int(r["timestamp"]) for r in ratings]
    assert all(8e11 < t < 1.8e12 for t in ts)  # milliseconds (CLAUDE.md s6.1)
    assert (max(ts) - min(ts)) / 86_400_000 > 2.9 * 365
    assert {float(r["rating"]) for r in ratings} <= {1.0, 2.0, 3.0, 4.0, 5.0}


def test_meta_covers_rated_books_plus_unrated(
    meta: dict[str, dict[str, Any]], ratings: list[dict[str, str]]
) -> None:
    rated = {r["parent_asin"] for r in ratings}
    assert rated < set(meta) and len(meta) == 85
    fields = {"title", "author", "store", "categories", "details", "description", "images"}
    assert all(fields <= set(m) for m in meta.values())


def test_planted_metadata_cases(
    meta: dict[str, dict[str, Any]], truth: dict[str, Any], repo_config: Config
) -> None:
    (json_asin,) = truth["json_string_fields"]
    assert isinstance(meta[json_asin]["details"], str)
    assert json.loads(meta[json_asin]["author"])["name"] == "Tess Arden"
    (images_asin,) = truth["images_as_dict"]
    assert isinstance(meta[images_asin]["images"], dict)
    (no_author,) = truth["no_author"]
    assert meta[no_author]["author"] is None and meta[no_author]["store"] is None
    for asin, name in truth["store_author"].items():  # F4: several credits, author not first
        assert meta[asin]["author"] is None and f"{name} (Author)" in meta[asin]["store"]
        assert not meta[asin]["store"].startswith(name)
    for asin, page_source in truth["kindle_editions"].items():  # F1
        kindle = meta[asin]
        assert kindle["main_category"] == "Buy a Kindle" and asin.startswith("B0")
        assert "ISBN 10" not in kindle["details"] and "ISBN 13" not in kindle["details"]
        assert kindle["details"]["Page numbers source ISBN"] == page_source
        assert "Format: Kindle Edition" in kindle["store"]
    (audio,) = truth["audiobooks"]
    assert meta[audio]["main_category"] == "Audible Audiobooks"
    assert meta[audio]["categories"][0] == "Audible Books & Originals"
    for asin, value in truth["isbn13_in_isbn10_field"].items():  # F3
        assert meta[asin]["details"]["ISBN 10"] == value and len(value) == 13
    for asin in truth["author_without_name"]:
        assert meta[asin]["author"] == {"avatar": "https://example.invalid/avatar.jpg"}
    assert meta[truth["empty_title"][0]]["title"] == ""
    (dated,) = truth["publication_date_key"]
    assert "Publication date" in meta[dated]["details"]
    blocklist = repo_config["data"]["title_blocklist"]
    for asin in truth["non_narrative"]:
        title = meta[asin]["title"].lower()
        assert any(re.search(rf"\b{re.escape(term)}\b", title) for term in blocklist)
    (short,) = truth["too_short"]
    pages = re.search(r"(\d+) pages", json.dumps(meta[short]["details"]))
    assert pages and int(pages.group(1)) < repo_config["data"]["min_pages"]
    (cookbook,) = truth["out_of_slice"]
    assert meta[cookbook]["categories"][1] not in repo_config["data"]["fiction_allowlist"]
    assert meta[truth["no_categories"][0]]["categories"] == []


def test_series_boxset_and_editions(
    raw: Path, meta: dict[str, dict[str, Any]], truth: dict[str, Any], repo_config: Config
) -> None:
    goodreads = records(raw / "goodreads_books.json.gz")
    by_isbn = {g["isbn"]: g for g in goodreads if g["isbn"]}
    by_isbn13 = {g["isbn13"]: g for g in goodreads if g["isbn13"]}
    for asin, pos in truth["series"]["volumes"].items():
        record = by_isbn[meta[asin]["details"]["ISBN 10"]]
        match = SERIES_RE.search(record["title"])
        assert match and (match.group(1), float(match.group(2))) == ("The Ember Cycle", pos)
        assert record["series"] == [truth["series"]["series_id"]]
    novella = SERIES_RE.search(truth["series"]["novella_only_in_goodreads"])
    assert novella and float(novella.group(2)) == 1.5
    assert by_isbn[truth["series"]["novella_isbn10"]]["title"] == novella.string

    box_patterns = [re.compile(p, re.I) for p in repo_config["canonical"]["boxset_patterns"]]
    (box,) = truth["boxsets"]
    assert any(p.search(meta[box]["title"]) for p in box_patterns)
    volumes = [meta[a]["title"] for a in truth["series"]["volumes"]]
    assert not any(p.search(title) for title in volumes for p in box_patterns)
    assert SERIES_RE.search(by_isbn[meta[box]["details"]["ISBN 10"]]["title"]) is None  # "#1-3"

    (gr_pair,) = truth["editions_goodreads"]
    assert len({by_isbn[meta[a]["details"]["ISBN 10"]]["work_id"] for a in gr_pair}) == 1
    (heuristic_pair,) = truth["editions_heuristic"]
    assert not any(meta[a]["details"]["ISBN 10"] in by_isbn for a in heuristic_pair)
    for asin in truth["goodreads_isbn10_only"]:
        assert by_isbn[meta[asin]["details"]["ISBN 10"]]["isbn13"] == ""
    for asin in truth["goodreads_isbn13_only"]:
        assert by_isbn13[meta[asin]["details"]["ISBN 13"].replace("-", "")]["isbn"] == ""


def test_goodreads_values_are_strings(raw: Path) -> None:
    for g in records(raw / "goodreads_books.json.gz"):
        assert all(isinstance(v, str) for v in g.values() if not isinstance(v, list))
    spoilers = records(raw / "goodreads_reviews_spoiler.json.gz")
    assert {label for r in spoilers for label, _ in r["review_sentences"]} == {0, 1}
    for r in spoilers:
        assert r["has_spoiler"] == any(label for label, _ in r["review_sentences"])


def test_reviews_and_planted_review_cases(
    raw: Path, ratings: list[dict[str, str]], truth: dict[str, Any], repo_config: Config
) -> None:
    reviews = records(raw / "Books.jsonl.gz")
    in_core = {(r["user_id"], r["parent_asin"]) for r in ratings}
    keyed = {f"{r['user_id']}|{r['parent_asin']}": r for r in reviews}
    outsiders = [r for r in reviews if (r["user_id"], r["parent_asin"]) not in in_core]
    assert len(outsiders) == 30
    assert {r["user_id"] for r in outsiders} == set(truth["users_outside_5core"])
    assert len(reviews) - 30 + len(truth["pairs_without_review"]) == 600
    assert not set(truth["pairs_without_review"]) & set(keyed)

    special = truth["special_reviews"]
    first, second = special["cross_edition_duplicate"]
    assert keyed[first]["text"] == keyed[second]["text"]
    child = keyed[special["child_asin"]]
    assert child["asin"] != child["parent_asin"]
    shipping = [re.compile(p, re.I) for p in repo_config["data"]["shipping_patterns"]]
    assert any(p.search(keyed[special["shipping"]]["text"]) for p in shipping)
    assert len(keyed[special["too_short"]]["text"]) < repo_config["data"]["min_review_chars"]
    html = keyed[special["html"]]["text"]
    assert "<br" in html and "&#39;" in html
    assert "libro" in keyed[special["non_english"]]["text"]

    spoiler = [re.compile(p, re.I) for p in repo_config["spoiler"]["regex_patterns"]]
    n_spoilers = 0
    for key, sentences in truth["reviews"].items():
        assert keyed[key]["text"] == " ".join(s["text"] for s in sentences)
        for s in sentences:
            if s["spoiler"]:
                n_spoilers += 1
                assert s is sentences[-1]  # spoilers sit at the end of a review
                assert any(p.search(s["text"]) for p in spoiler)
    assert n_spoilers >= 20
