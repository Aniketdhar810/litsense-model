"""Stage data.goodreads (CLAUDE.md s7.3): an ISBN index into the UCSD Goodreads book graph.

Optional (D20). With ``data.use_goodreads_graph: false`` (the default) the stage reads nothing
from the network and writes both tables empty but with their final schema, so data.canonical has
a single code path. When enabled it streams ``goodreads_books`` (2.08 GB) and keeps only the books
whose ISBN-13 matches a ``books_raw`` ISBN (print ISBN-10/13, or a Kindle edition's page-source
ISBN), then streams the series and works files for the ids those rows reference. Only the two
small tables are stored.
"""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.log import Progress
from ml.common.stage import StageContext, stage
from ml.data.parsing import (
    as_isbn13,
    isbn10_to_13,
    json_str_field,
    series_suffix,
    split_isbns,
)
from ml.data.stream import StreamSettings, StreamStats, iter_lines

BOOKS, SERIES, WORKS = (
    "url:goodreads_books.json.gz",
    "url:goodreads_book_series.json.gz",
    "url:goodreads_book_works.json.gz",
)
GR_INDEX_SCHEMA = pa.schema(
    [
        ("isbn13", pa.string()),  # the ISBN-13 that matched books_raw
        ("isbn10", pa.string()),
        ("gr_book_id", pa.string()),
        ("work_id", pa.string()),
        ("series_id", pa.string()),  # first of series_ids (s7.3)
        ("series_ids", pa.list_(pa.string())),
        ("series_name", pa.string()),  # series file title, else the title suffix
        ("series_pos", pa.float32()),  # from "(Name, #2.5)"; null for ranges like "#1-3"
        ("title", pa.string()),
        ("title_without_series", pa.string()),
        ("num_pages", pa.int32()),
        ("publication_year", pa.int16()),
        ("ratings_count", pa.int32()),
        ("work_title", pa.string()),
        ("work_year", pa.int16()),
    ]
)
GR_SERIES_SCHEMA = pa.schema(
    [
        ("series_id", pa.string()),
        ("title", pa.string()),
        ("numbered", pa.bool_()),
        ("primary_work_count", pa.int32()),
    ]
)


def book_isbn13s(books: pd.DataFrame) -> list[set[str]]:
    """Per ``books_raw`` row, every ISBN-13 it can be found under: its print ISBN (ISBN-10s
    converted) and, for Kindle editions, the page-source ISBN of the print edition (F1, D17)."""
    columns = (books[c].map(_text) for c in ("isbn10", "isbn13", "page_source_isbn13"))
    return [
        {k for k in (as_isbn13(i10, i13), source) if k}
        for i10, i13, source in zip(*columns, strict=True)
    ]


def match_key(line: str, wanted: set[str]) -> tuple[str, str | None] | None:
    """``(matched ISBN-13, ISBN-10)`` of a goodreads_books line if it is wanted, else None.
    Reads the two ISBN fields without parsing the (long) line."""
    isbn10, isbn13 = split_isbns(json_str_field(line, "isbn"), json_str_field(line, "isbn13"))
    for key in (isbn13, isbn10_to_13(isbn10) if isbn10 else None):
        if key in wanted:
            return key, isbn10
    return None


def gr_row(record: Mapping[str, Any], isbn13: str, isbn10: str | None) -> dict[str, Any]:
    """One index row from a goodreads_books record (every value in the dump is a string, F12)."""
    series_ids = [str(s) for s in record.get("series") or []]
    suffix = series_suffix(record.get("title") or "")
    return {
        "isbn13": isbn13,
        "isbn10": isbn10,
        "gr_book_id": record.get("book_id") or None,
        "work_id": record.get("work_id") or None,
        "series_id": series_ids[0] if series_ids else None,
        "series_ids": series_ids,
        "series_name": suffix[0] if suffix else None,
        "series_pos": suffix[1] if suffix else None,
        "title": record.get("title") or None,
        "title_without_series": record.get("title_without_series") or None,
        "num_pages": _int(record.get("num_pages")),
        "publication_year": _int(record.get("publication_year")),
        "ratings_count": _int(record.get("ratings_count")),
        "work_title": None,
        "work_year": None,
    }


@stage(
    "data.goodreads",
    inputs=["data:books_raw.parquet", BOOKS, SERIES, WORKS],
    outputs=["data:gr_isbn_index.parquet", "data:gr_series.parquet"],
    config_keys=["data.use_goodreads_graph", "data.stream"],
)
def build_goodreads(ctx: StageContext) -> None:
    """Index the Goodreads books matching books_raw ISBNs, with series and work details."""
    index_path, series_path = (
        ctx.out("data:gr_isbn_index.parquet"),
        ctx.out("data:gr_series.parquet"),
    )
    ctx.notes["enabled"] = bool(ctx.param("data.use_goodreads_graph"))
    if not ctx.notes["enabled"]:
        _write(pd.DataFrame(columns=GR_INDEX_SCHEMA.names), GR_INDEX_SCHEMA, index_path)
        _write(pd.DataFrame(columns=GR_SERIES_SCHEMA.names), GR_SERIES_SCHEMA, series_path)
        ctx.log.info("data.use_goodreads_graph is false: wrote empty Goodreads tables")
        return

    settings = StreamSettings.from_config(ctx.param("data.stream"))
    books = pd.read_parquet(
        ctx.path("data:books_raw.parquet"), columns=["isbn10", "isbn13", "page_source_isbn13"]
    )
    keys = book_isbn13s(books)
    wanted = set().union(*keys)
    rows = []
    for line in _lines(ctx, BOOKS, settings):
        if matched := match_key(line, wanted):
            rows.append(gr_row(json.loads(line), *matched))
    index = pd.DataFrame(rows, columns=GR_INDEX_SCHEMA.names)

    series = pd.DataFrame(
        [
            {
                "series_id": r["series_id"],
                "title": r.get("title") or None,
                "numbered": r.get("numbered") == "true",
                "primary_work_count": _int(r.get("primary_work_count")),
            }
            for r in _records(ctx, SERIES, settings, "series_id", set(index["series_id"].dropna()))
        ],
        columns=GR_SERIES_SCHEMA.names,
    )
    works = {
        r["work_id"]: r
        for r in _records(ctx, WORKS, settings, "work_id", set(index["work_id"].dropna()))
    }
    names = dict(zip(series["series_id"], series["title"], strict=True))
    index["series_name"] = [
        names.get(sid) or parsed
        for sid, parsed in zip(index["series_id"], index["series_name"], strict=True)
    ]
    index["work_title"] = [works.get(w, {}).get("original_title") or None for w in index["work_id"]]
    index["work_year"] = [
        _int(works.get(w, {}).get("original_publication_year")) for w in index["work_id"]
    ]

    _write(index, GR_INDEX_SCHEMA, index_path)
    _write(series, GR_SERIES_SCHEMA, series_path)
    matched = set(index["isbn13"])
    ctx.count("books_raw", len(books))
    ctx.count("books_raw_isbn13s", len(wanted))
    ctx.count("gr_rows_matched", len(index))
    ctx.count("books_raw_matched", sum(bool(k & matched) for k in keys))
    ctx.count("gr_series", len(series))
    ctx.count("gr_works", len(works))


def _lines(ctx: StageContext, spec: str, settings: StreamSettings) -> Iterable[str]:
    stats = StreamStats(ctx.source(spec))
    progress = Progress(ctx.log, spec.removeprefix("url:"))
    for line in iter_lines(stats.source, settings, limit=ctx.sample, stats=stats):
        progress.update(stats.bytes_read, stats.size, f"; {stats.lines:,} lines")
        yield line
    ctx.notes.setdefault("sources", {})[spec] = stats.as_note()


def _records(
    ctx: StageContext, spec: str, settings: StreamSettings, key: str, wanted: set[str]
) -> Iterable[dict[str, Any]]:
    """Parsed records of ``spec`` whose ``key`` field is in ``wanted`` (prefiltered on the raw
    line)."""
    for line in _lines(ctx, spec, settings):
        if json_str_field(line, key) in wanted:
            yield json.loads(line)


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text.isdigit() else None


def _write(frame: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    frame = frame.copy()
    for f in schema:
        if pa.types.is_integer(f.type):
            frame[f.name] = pd.to_numeric(frame[f.name]).astype(f"Int{f.type.bit_width}")
        elif pa.types.is_floating(f.type):
            frame[f.name] = pd.to_numeric(frame[f.name]).astype("float32")
        elif pa.types.is_boolean(f.type):
            frame[f.name] = frame[f.name].astype("boolean")
    pq.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False), path)
