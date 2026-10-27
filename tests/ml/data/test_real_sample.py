"""Parsers against real records in local/raw_sample/ (CLAUDE.md s16.1); skipped when absent.

The bounds are parsing sanity checks measured on the first 50,000 meta records (see
docs/progress.md), not statistics of the dataset.
"""

import gzip
import json
from collections import Counter

import pytest

from ml.common.config import REPO_ROOT
from ml.data.meta import parse_meta
from ml.data.parsing import extract_parent_asin

SAMPLE = REPO_ROOT / "local" / "raw_sample"


def lines(name: str) -> list[str]:
    path = SAMPLE / name
    if not path.exists():
        pytest.skip(f"{path} not present (create it with curl | gunzip | head)")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return f.read().splitlines()


def test_meta_records_parse() -> None:
    books = []
    for line in lines("meta_Books.jsonl.gz"):
        record = json.loads(line)
        assert extract_parent_asin(line) == record["parent_asin"]
        books.append(parse_meta(record))
    parsed = [b for b in books if b is not None]
    assert len(parsed) >= 0.95 * len(books)
    kinds = Counter(b["edition_kind"] for b in parsed)
    assert kinds["kindle"] > 0 and kinds["audio"] > 0  # F1
    assert kinds["unknown"] < 0.05 * len(parsed)
    print_books = [b for b in parsed if b["edition_kind"] == "print"]
    assert sum(bool(b["isbn10"] or b["isbn13"]) for b in print_books) >= 0.9 * len(print_books)
    assert sum(bool(b["author"]) for b in parsed) >= 0.95 * len(parsed)


def test_review_lines_expose_parent_asin() -> None:
    for line in lines("Books.jsonl.gz"):
        assert extract_parent_asin(line) == json.loads(line)["parent_asin"]


def test_sample_folder_is_gitignored() -> None:
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "local/" in ignore  # covers local/raw_sample/
