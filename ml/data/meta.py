"""Stage data.meta (CLAUDE.md s7.2): metadata for every 5-core book, streamed from meta_Books.

meta_Books (4.94 GB, about 4.4M records) is streamed from its URL and filtered on the fly; it is
never stored (D18). A line is JSON-parsed only when its ``parent_asin`` is a 5-core item (about
15% of records). Parsing follows the format findings F1-F8. Kindle and audiobook records are kept
and tagged in ``edition_kind``: they are editions of the same work, and data.canonical merges them
with print editions (D17).

This is where the automatic pipeline pauses: the stage writes ``category_counts.csv`` and a slice
proposal, and every later stage calls ``require_genre_slices`` before doing anything.

    python -m ml.data.meta [--sample N]                 # category report of the last run
    python -m ml.data.meta --find "Circe" "Gone Girl"   # look titles up in books_raw
"""

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.config import load_config
from ml.common.errors import ConfigError
from ml.common.log import Progress
from ml.common.stage import RunPaths, StageContext, stage
from ml.data.parsing import (
    DATE_RE,
    ISBN10_RE,
    as_isbn13,
    clean_html,
    extract_parent_asin,
    maybe_json,
    parse_author,
    parse_format,
    parse_image,
    parse_pages,
    parse_year,
    split_isbns,
)
from ml.data.stream import PartWriter, StreamSettings, StreamStats, iter_lines

BOOK_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("title", pa.string()),
        ("subtitle", pa.string()),
        ("author", pa.string()),
        ("categories", pa.list_(pa.string())),
        ("main_category", pa.string()),
        ("description", pa.string()),
        ("features", pa.string()),
        ("isbn10", pa.string()),
        ("isbn13", pa.string()),
        ("page_source_isbn13", pa.string()),  # F1: a Kindle edition's print ISBN
        ("pages", pa.int32()),
        ("format", pa.string()),
        ("edition_kind", pa.string()),  # print | kindle | audio | unknown
        ("year", pa.int16()),
        ("language", pa.string()),
        ("image", pa.string()),
        ("n_interactions", pa.int32()),  # 5-core ratings of this parent_asin
    ]
)
BOOK_COLUMNS = BOOK_SCHEMA.names
COUNT_COLUMNS = [
    "level", "category", "subcategory", "n_items", "n_interactions",
    "n_interactions_kindle", "n_interactions_audio", "interaction_share",
]  # fmt: skip
NO_CATEGORY = "<none>"
PARQUET_COMPRESSION = "zstd"  # D24: 30% smaller than snappy on books_raw


def parse_meta(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """One meta_Books record as a ``books_raw`` row (``n_interactions`` excluded); None without
    a title."""
    title = clean_html(record.get("title"))
    if not title:
        return None
    details = maybe_json(record.get("details"))
    details = details if isinstance(details, dict) else {}
    asin = record["parent_asin"]
    categories = [c for c in record.get("categories") or [] if isinstance(c, str)]
    subtitle = clean_html(record.get("subtitle")) or None
    isbn10, isbn13 = split_isbns(details.get("ISBN 10"), details.get("ISBN 13"))
    if isbn10 is None and ISBN10_RE.match(asin):
        isbn10 = asin  # most print parent_asins are ISBN-10s (F9: 84%)
    page_source = split_isbns(details.get("Page numbers source ISBN"))
    publisher_date = DATE_RE.search(str(details.get("Publisher", "")))
    year = (
        parse_year(publisher_date.group(1))
        if publisher_date
        else parse_year(details.get("Publication date")) or parse_year(subtitle)  # F8
    )
    fmt, edition_kind = parse_format(details, record.get("main_category"), subtitle, categories)
    return {
        "parent_asin": asin,
        "title": title,
        "subtitle": subtitle,
        "author": parse_author(record.get("author"), record.get("store")),
        "categories": categories,
        "main_category": record.get("main_category") or None,
        "description": clean_html(" ".join(map(str, record.get("description") or []))) or None,
        "features": clean_html(" ".join(map(str, record.get("features") or []))) or None,
        "isbn10": isbn10,
        "isbn13": isbn13,
        "page_source_isbn13": as_isbn13(*page_source),
        "pages": parse_pages(details),
        "format": fmt,
        "edition_kind": edition_kind,
        "year": year,
        "language": details.get("Language"),
        "image": parse_image(record.get("images")),
    }


def compile_blocklist(terms: Sequence[str]) -> re.Pattern[str]:
    """One case-insensitive, word-bounded pattern for ``data.title_blocklist``."""
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)


def drop_reason(book: Mapping[str, Any], blocklist: re.Pattern[str], min_pages: int) -> str | None:
    """Why a parsed book is not narrative (s7.2 step 4), or None to keep it."""
    if blocklist.search(book["title"]):
        return "title_blocklist"
    if book["pages"] is not None and book["pages"] < min_pages:
        return "min_pages"
    return None


def load_item_interactions(path: Path) -> dict[str, int]:
    """5-core interactions per ``parent_asin``."""
    items = pd.read_csv(path, usecols=["parent_asin"], dtype={"parent_asin": "string[pyarrow]"})
    return {str(k): int(v) for k, v in items["parent_asin"].value_counts().items()}


def category_counts(books: pd.DataFrame) -> pd.DataFrame:
    """Items and 5-core interactions per top-level category (``categories[1]``) and per
    (top-level, ``categories[2]``) pair, with Kindle and audio interactions broken out."""
    n = books["n_interactions"].astype("int64")
    frame = pd.DataFrame(
        {
            "category": [c[1] if len(c) > 1 else NO_CATEGORY for c in books["categories"]],
            "subcategory": [c[2] if len(c) > 2 else NO_CATEGORY for c in books["categories"]],
            "n": n,
            "kindle": n.where(books["edition_kind"] == "kindle", 0),
            "audio": n.where(books["edition_kind"] == "audio", 0),
        }
    )
    total = max(int(n.sum()), 1)
    levels = []
    for level, keys in (("top", ["category"]), ("pair", ["category", "subcategory"])):
        grouped = (
            frame.groupby(keys)
            .agg(
                n_items=("n", "size"),
                n_interactions=("n", "sum"),
                n_interactions_kindle=("kindle", "sum"),
                n_interactions_audio=("audio", "sum"),
            )
            .reset_index()
        )
        grouped["level"] = level
        if "subcategory" not in grouped:
            grouped["subcategory"] = ""
        grouped["interaction_share"] = (grouped["n_interactions"] / total).round(6)
        levels.append(grouped.sort_values(["n_interactions", "category"], ascending=[False, True]))
    return pd.concat(levels, ignore_index=True).reindex(columns=COUNT_COLUMNS)


def propose_genre_slices(
    counts: pd.DataFrame,
    allowlist: Sequence[str],
    target_v1: int,
    rule: Mapping[str, int],
) -> dict[str, Any]:
    """s7.2 step 6 default: the fewest (``min_slices``..``max_slices``) allowlisted categories with
    the most interactions whose total reaches ``target_multiple`` x the v1 target. The user
    decides; this is only the starting proposal."""
    top = counts[counts["level"] == "top"].set_index("category")
    candidates = sorted(
        (
            {
                "category": c,
                "n_interactions": int(top["n_interactions"].get(c, 0)),
                "n_items": int(top["n_items"].get(c, 0)),
            }
            for c in allowlist
        ),
        key=lambda c: (-c["n_interactions"], c["category"]),
    )
    threshold = int(rule["target_multiple"]) * int(target_v1)
    chosen = next(
        (
            candidates[:k]
            for k in range(int(rule["min_slices"]), int(rule["max_slices"]) + 1)
            if sum(c["n_interactions"] for c in candidates[:k]) >= threshold
        ),
        None,
    )
    meets = chosen is not None
    chosen = chosen if meets else candidates[: int(rule["max_slices"])]
    return {
        "rule": (
            f"fewest of {rule['min_slices']}-{rule['max_slices']} data.fiction_allowlist "
            f"categories with >= {rule['target_multiple']} x target_interactions.v1 interactions"
        ),
        "threshold": threshold,
        "candidates": candidates,
        "proposed": [c["category"] for c in chosen],
        "combined_interactions": sum(c["n_interactions"] for c in chosen),
        "meets_threshold": meets,
    }


def category_report(counts: pd.DataFrame, proposal: Mapping[str, Any], top: int = 25) -> str:
    """Human-readable summary of ``category_counts.csv`` and the proposal."""
    allow = {c["category"] for c in proposal["candidates"]}
    rows = counts[counts["level"] == "top"].head(top)
    lines = [
        "Top-level categories by 5-core interactions (* = in data.fiction_allowlist):",
        f"  {'interactions':>12} {'share':>7} {'items':>8} {'kindle':>7} {'audio':>6}  category",
    ]
    for r in rows.itertuples():
        kindle = r.n_interactions_kindle / max(r.n_interactions, 1)
        audio = r.n_interactions_audio / max(r.n_interactions, 1)
        mark = "*" if r.category in allow else " "
        lines.append(
            f"  {r.n_interactions:>12,} {r.interaction_share:>7.1%} {r.n_items:>8,} "
            f"{kindle:>7.1%} {audio:>6.1%} {mark}{r.category}"
        )
    verdict = "meets" if proposal["meets_threshold"] else "does NOT meet"
    lines += [
        f"Proposal ({proposal['rule']}): {', '.join(proposal['proposed']) or '-'}",
        f"  combined {proposal['combined_interactions']:,} interactions; "
        f"{verdict} the threshold of {proposal['threshold']:,}",
    ]
    return "\n".join(lines)


STOP_MESSAGE = (
    "STOP: choose data.genre_slices in ml/config.yaml from category_counts.csv (proposal above), "
    "record the choice in docs/decisions.md, then continue with the next stage."
)


def require_genre_slices(slices: Sequence[str] | None) -> list[str]:
    """Return the ``data.genre_slices`` value; raise ConfigError while the user has not chosen
    them."""
    if not slices or not all(isinstance(s, str) and s for s in slices):
        raise ConfigError(f"data.genre_slices is not set. {STOP_MESSAGE}")
    return list(slices)


@stage(
    "data.meta",
    inputs=["data:ratings_5core.csv.gz", "url:meta_books"],
    outputs=[
        "data:books_raw.parquet",
        "data:category_counts.csv",
        "data:genre_slice_proposal.json",
    ],
    config_keys=[
        "data.title_blocklist",
        "data.min_pages",
        "data.fiction_allowlist",
        "data.target_interactions",
        "data.genre_slice_rule",
        "data.stream",
    ],
)
def build_meta(ctx: StageContext) -> None:
    """Stream meta_Books, keep 5-core books, drop non-narrative titles, count categories."""
    interactions = load_item_interactions(ctx.path("data:ratings_5core.csv.gz"))
    blocklist = compile_blocklist(ctx.param("data.title_blocklist"))
    min_pages = int(ctx.param("data.min_pages"))
    stream = ctx.param("data.stream")
    source = ctx.source("url:meta_books")
    stats = StreamStats(source)
    writer = PartWriter(ctx.work_dir, every=int(stream["checkpoint_rows"]))
    if writer.resume_lines:
        ctx.log.info("resuming after line %d from checkpoint", writer.resume_lines)
    lines = iter_lines(source, StreamSettings.from_config(stream), limit=ctx.sample, stats=stats)
    progress = Progress(ctx.log, "meta_Books")
    for n, line in enumerate(lines, start=1):
        progress.update(
            stats.bytes_read,
            stats.size,
            f"; {n:,} lines, {writer.counters['meta_matched']:,} matched the 5-core",
        )
        if n <= writer.resume_lines:
            continue
        asin = extract_parent_asin(line)
        if asin in interactions:
            writer.counters["meta_matched"] += 1
            book = parse_meta(json.loads(line))
            reason = "no_title" if book is None else drop_reason(book, blocklist, min_pages)
            if reason:
                writer.counters[f"dropped_{reason}"] += 1
            else:
                writer.add(book | {"n_interactions": interactions[asin]})
        writer.checkpoint(n)

    out = ctx.out("data:books_raw.parquet")
    duplicates = write_books(writer.close(stats.lines), out)
    books = pd.read_parquet(
        out, columns=["parent_asin", "categories", "edition_kind", "n_interactions"]
    )
    counts = category_counts(books)
    counts.to_csv(ctx.out("data:category_counts.csv"), index=False)
    proposal = propose_genre_slices(
        counts,
        ctx.param("data.fiction_allowlist"),
        ctx.param("data.target_interactions.v1"),
        ctx.param("data.genre_slice_rule"),
    )
    path = ctx.out("data:genre_slice_proposal.json")
    path.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")

    ctx.count("5core_items", len(interactions))
    ctx.count("meta_lines_read", stats.lines)
    for key, value in sorted(writer.counters.items()):
        ctx.count(key, value)
    ctx.count("duplicate_parent_asins", duplicates)
    ctx.count("books_raw", len(books))
    ctx.notes["source"] = stats.as_note()
    ctx.notes["edition_kind_interactions"] = (
        books.groupby("edition_kind")["n_interactions"].sum().astype(int).to_dict()
    )
    ctx.log.info("\n%s", category_report(counts, proposal))
    ctx.log.warning(STOP_MESSAGE)


def write_books(parts: Sequence[Path], path: Path) -> int:
    """Write checkpoint parts to one parquet file with ``BOOK_SCHEMA``, one part in memory at a
    time, keeping the first record of each ``parent_asin``. Returns the number of duplicates."""
    seen: set[str] = set()
    duplicates = 0
    with pq.ParquetWriter(path, BOOK_SCHEMA, compression=PARQUET_COMPRESSION) as out:
        for part in parts:
            frame = pd.read_parquet(part)
            keep = ~frame["parent_asin"].isin(seen) & ~frame["parent_asin"].duplicated()
            duplicates += int((~keep).sum())
            frame = frame[keep].reindex(columns=BOOK_COLUMNS)
            seen.update(frame["parent_asin"])
            for column, dtype in (
                ("pages", "Int32"),
                ("year", "Int16"),
                ("n_interactions", "Int32"),
            ):
                frame[column] = pd.to_numeric(frame[column]).astype(dtype)
            frame["categories"] = frame["categories"].map(list)
            out.write_table(pa.Table.from_pandas(frame, schema=BOOK_SCHEMA, preserve_index=False))
    return duplicates


def find_titles(books: pd.DataFrame, queries: Sequence[str], limit: int = 10) -> pd.DataFrame:
    """Editions whose title contains each query (case-insensitive), most interactions first.

    ``books`` needs ``parent_asin``, ``title``, ``author``, ``categories``, ``edition_kind`` and
    ``n_interactions``. Matching is plain substring search: editions of one work appear as separate
    rows until data.canonical merges them.
    """
    titles = books["title"].str.casefold()
    rows = []
    for query in queries:
        hits = books[titles.str.contains(query.casefold(), regex=False)]
        hits = hits.sort_values("n_interactions", ascending=False).head(limit)
        for book in hits.itertuples():
            rows.append(
                {
                    "query": query,
                    "title": book.title,
                    "author": book.author,
                    "category": book.categories[1] if len(book.categories) > 1 else NO_CATEGORY,
                    "edition_kind": book.edition_kind,
                    "n_interactions": int(book.n_interactions),
                    "parent_asin": book.parent_asin,
                }
            )
        if hits.empty:
            rows.append({"query": query, "title": None, "n_interactions": 0})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the category report of the last data.meta run, or look titles up in books_raw."""
    parser = argparse.ArgumentParser(description="Report on data.meta's outputs.")
    parser.add_argument("--config", help="config YAML path (default ml/config.yaml)")
    parser.add_argument("--sample", type=int, metavar="N", help="read the --sample run")
    parser.add_argument("--find", nargs="+", metavar="TITLE", help="look titles up in books_raw")
    args = parser.parse_args(argv)
    paths = RunPaths.from_config(load_config(args.config), args.sample)
    if args.find:
        books = pd.read_parquet(
            paths.resolve("data:books_raw.parquet"),
            columns=[
                "parent_asin",
                "title",
                "author",
                "categories",
                "edition_kind",
                "n_interactions",
            ],
        )
        found = find_titles(books, args.find)
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            for query, group in found.groupby("query", sort=False):
                total = int(group["n_interactions"].sum())
                editions = int(group["title"].notna().sum())
                print()
                print(f"{query!r}: {total:,} interactions over {editions} edition(s)")
                if group["title"].notna().any():
                    print(group.drop(columns="query").to_string(index=False))
        return 0
    counts = pd.read_csv(paths.resolve("data:category_counts.csv"), keep_default_na=False)
    proposal = json.loads(paths.resolve("data:genre_slice_proposal.json").read_text("utf-8"))
    print(category_report(counts, proposal))
    print(STOP_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
