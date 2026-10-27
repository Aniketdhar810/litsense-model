"""Stage data.canonical (CLAUDE.md s7.4, docs/decisions.md D28-D29): editions -> works.

Every ``books_raw`` edition (print, Kindle, audio) is mapped to one work, never merging different
works, different volumes, or a box set with a volume. Merges come from, in order:

1. ``isbn``: two parent_asins with the same ISBN-13 (ISBN-10s converted) and the same author, or
   an author missing on one side (D31);
2. ``page_source``: a Kindle edition whose "Page numbers source ISBN" is a print edition's ISBN
   (same author, or an author missing on one side; F1, D17);
3. ``goodreads``: the same Goodreads work id (only with ``data.use_goodreads_graph``);
4. ``title``: blocked by (first token of the normalized title, author surname), merged when
   ``token_set_ratio >= fuzzy_threshold`` AND ``token_sort_ratio >= subset_guard_threshold`` (a
   word subset scores 100 on token_set: "martian" vs "martian chronicles")
   AND the same author key AND compatible volume numbers AND, for non-identical titles, a year gap
   of at most ``max_year_gap``. Two components with different Goodreads works or different known
   volumes are never joined.
5. ``subtitle`` (D32): "Title" joins "Title: subtitle" when the bare title equals the other's text
   before the colon, the author key is the same, the volume guard passes, neither is a box set,
   and their blurbs (``features``, else ``description``) are similar (``token_sort_ratio`` on the
   first ``subtitle_description_chars`` characters >= ``subtitle_description_threshold``). Without
   a blurb on either side there is no merge. Runs last, so every such edge joins two works that the
   other rules kept apart; ``merge_audit.csv`` samples them in its own section.

All 454K editions are merged first; a work is kept when any of its editions sits in a genre slice
(rule A, D28), so a Kindle edition filed under another category still counts. Box sets are dropped.
The audit files go to ``results:`` so they reach the repo (D28, user rule B).

    python -m ml.data.canonical [--sample N]    # summary of the last run's canonical_report.json
"""

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from ml.common.config import load_config
from ml.common.errors import DataError
from ml.common.stage import RunPaths, StageContext, stage
from ml.data.genres import work_genres
from ml.data.meta import NO_CATEGORY, PARQUET_COMPRESSION, require_genre_slices
from ml.data.parsing import as_isbn13
from ml.data.titles import (
    TitleRules,
    author_key,
    display_title,
    is_boxset,
    normalize_title,
    number_tokens,
    series_from_title,
    series_key,
    title_head,
    volume_number,
)

KEY_COLUMNS = [
    "parent_asin", "title", "author", "categories", "isbn10", "isbn13", "page_source_isbn13",
    "pages", "edition_kind", "year", "image", "n_interactions",
]  # fmt: skip
MAP_SCHEMA = pa.schema(
    [
        ("parent_asin", pa.string()),
        ("canonical_id", pa.string()),
        ("status", pa.string()),  # kept | boxset | out_of_slice
        ("merge_rule", pa.string()),  # singleton | isbn | page_source | goodreads | title
        ("in_slice_edition", pa.bool_()),  # this edition's own categories[1] is a slice
    ]
)
CANONICAL_SCHEMA = pa.schema(
    [
        ("canonical_id", pa.string()),
        ("title", pa.string()),
        ("author", pa.string()),
        ("author_key", pa.string()),
        ("description", pa.string()),
        ("features", pa.string()),
        ("genres", pa.list_(pa.string())),
        ("primary_genre", pa.string()),
        ("slices", pa.list_(pa.string())),  # categories[1] of the editions, most-rated first
        ("primary_slice", pa.string()),  # first of ``slices`` that is a genre slice
        ("categories_raw", pa.list_(pa.string())),  # every edition's path, " > "-joined
        ("year", pa.int16()),
        ("pages", pa.int32()),
        ("image", pa.string()),
        ("isbn13", pa.string()),
        ("member_asins", pa.list_(pa.string())),
        ("edition_kinds", pa.list_(pa.string())),
        ("series_id", pa.string()),
        ("series_name", pa.string()),
        ("series_pos", pa.float32()),
        ("gr_matched", pa.bool_()),
        ("n_members", pa.int32()),
        ("n_interactions", pa.int32()),  # 5-core ratings over all editions
    ]
)
NEAR_MISS_FLOOR = 80  # s7.4 step 9: near-threshold non-merges are audited from this score
NEAR_MISS_RESERVOIR = 2000  # near-miss pairs kept per reason for the audit (all are counted)
AUDIT_GROUPS, AUDIT_NEAR_MISSES, AUDIT_MAX_MEMBERS, AUDIT_SUBTITLE = 100, 50, 15, 30
SUBTITLE_PAIRS_PER_WORKS = 3  # description comparisons tried per pair of works (most-rated first)
Blurbs = Callable[[Sequence[int]], Mapping[int, str]]  # edition rows -> blurb text
AUDIT_EDITION_COLUMNS = [
    "canonical_id", "parent_asin", "title", "author", "year", "edition_kind", "category",
    "n_interactions", "merge_rule",
]  # fmt: skip
AUDIT_COLUMNS = [
    "audit_kind", "audit_id", "verdict", "reason", "token_set", "token_sort", "canonical_id",
    "parent_asin", "title", "author", "year", "edition_kind", "category", "n_interactions",
    "merge_rule",
]  # fmt: skip


class Works:
    """Union-find over editions. Each root carries the guard state of its component: the known
    volume numbers and Goodreads work ids of its members."""

    def __init__(self, volumes: Sequence[float | None], work_ids: Sequence[str | None]) -> None:
        self.parent = list(range(len(volumes)))
        self.volumes: list[set[float]] = [set() if v is None else {v} for v in volumes]
        self.work_ids: list[set[str]] = [{w} if w else set() for w in work_ids]
        self.rule: list[str | None] = [None] * len(volumes)  # how an edition first joined
        self.edges: Counter[str] = Counter()

    def find(self, i: int) -> int:
        """Root of ``i``'s component (with path compression)."""
        root = i
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[i] != root:
            self.parent[i], i = root, self.parent[i]
        return root

    def conflict(self, a: int, b: int) -> str | None:
        """Why the components of ``a`` and ``b`` must stay apart, or None."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return None
        if self.work_ids[ra] and self.work_ids[rb] and self.work_ids[ra] != self.work_ids[rb]:
            return "goodreads_work"
        if len(self.volumes[ra] | self.volumes[rb]) > 1:
            return "volume"
        return None

    def union(self, a: int, b: int, rule: str) -> bool:
        """Join the components of ``a`` and ``b``; False if they already were one."""
        ra, rb = sorted((self.find(a), self.find(b)))
        if ra == rb:
            return False
        self.parent[rb] = ra
        self.volumes[ra] |= self.volumes[rb]
        self.work_ids[ra] |= self.work_ids[rb]
        self.volumes[rb], self.work_ids[rb] = set(), set()
        for i in (a, b):
            self.rule[i] = self.rule[i] or rule
        self.edges[rule] += 1
        return True

    def roots(self) -> np.ndarray:
        """Component root of every edition."""
        return np.fromiter((self.find(i) for i in range(len(self.parent))), np.int64)


class NearMisses:
    """Title pairs that were not merged: exact counts per reason plus a seeded reservoir sample
    of ``NEAR_MISS_RESERVOIR`` pairs per reason (there can be millions)."""

    def __init__(self, rng: np.random.Generator, size: int = NEAR_MISS_RESERVOIR) -> None:
        self.rng, self.size = rng, size
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, list[dict[str, Any]]] = {}

    def add(self, pair: dict[str, Any]) -> None:
        """Count ``pair`` and keep it with reservoir probability."""
        reason = pair["reason"]
        self.counts[reason] += 1
        sample = self.samples.setdefault(reason, [])
        if len(sample) < self.size:
            sample.append(pair)
        elif (j := int(self.rng.integers(self.counts[reason]))) < self.size:
            sample[j] = pair


@dataclass
class SubtitleMerges:
    """What the ``subtitle`` rule did: its merge edges (two edition rows and their description
    similarity) and counts of candidate pairs by outcome."""

    edges: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)


def volume_conflict(a: float | None, b: float | None) -> bool:
    """D28 volume guard: both known and different, or one unknown and the other >= 2 ("Mistborn"
    may be book 1 but is never book 3)."""
    if a is not None and b is not None:
        return a != b
    known = a if a is not None else b
    return known is not None and known >= 2


def isbn_unions(books: pd.DataFrame, works: Works, akeys: Sequence[str | None]) -> None:
    """``isbn`` and ``page_source`` merges (box sets excluded by the caller via ``keys``)."""

    def same_author(i: int, j: int) -> bool:  # unknown on either side counts as a match
        return not akeys[i] or not akeys[j] or akeys[i] == akeys[j]

    first_by_isbn: dict[str, int] = {}
    for i, key in enumerate(books["isbn_key"]):
        if key:
            if key not in first_by_isbn:
                first_by_isbn[key] = i
            elif same_author(first_by_isbn[key], i):  # Amazon reuses some ISBNs (audit G018)
                works.union(first_by_isbn[key], i, "isbn")
    for i, source in enumerate(books["page_source_isbn13"]):
        j = first_by_isbn.get(source) if source else None
        if j is not None and same_author(i, j):
            works.union(j, i, "page_source")


def goodreads_unions(books: pd.DataFrame, works: Works) -> None:
    """``goodreads`` merges: editions sharing a Goodreads work id."""
    first_by_work: dict[str, int] = {}
    for i, work in enumerate(books["work_id"]):
        if work:
            if work in first_by_work:
                works.union(first_by_work[work], i, "goodreads")
            else:
                first_by_work[work] = i


def title_unions(
    books: pd.DataFrame, works: Works, params: Mapping[str, Any], near: NearMisses
) -> None:
    """``title`` merges within (first title token, author surname) blocks. Title pairs scoring
    >= ``NEAR_MISS_FLOOR`` on token_set that are not merged go to ``near`` with the reason."""
    set_min, sort_min = int(params["fuzzy_threshold"]), int(params["subset_guard_threshold"])
    max_gap = int(params["max_year_gap"])
    norms, akeys = books["norm_title"].tolist(), books["author_key"].tolist()
    volumes, years = books["volume"].tolist(), books["year"].tolist()
    order = np.argsort(-books["n_interactions"].to_numpy(), kind="stable")

    def blocked(i: int, j: int, identical: bool) -> str | None:
        if akeys[i] != akeys[j]:
            return "author"
        if volume_conflict(volumes[i], volumes[j]):
            return "volume"
        if not identical and number_tokens(norms[i]) != number_tokens(norms[j]):
            return "number"  # "nypd red 2" vs "nypd red 3" score 90 (D33)
        known = years[i] is not None and years[j] is not None
        if not identical and known and abs(years[i] - years[j]) > max_gap:
            return "year"
        return works.conflict(i, j)

    boxsets = books["boxset"].tolist()
    blocks: dict[tuple[str, str], dict[str, list[int]]] = {}
    for i in order.tolist():  # most-rated first: each title's first edition represents it
        if boxsets[i] or not norms[i] or not akeys[i]:
            continue  # D28: heuristic merges need an author
        key = (norms[i].split()[0], akeys[i].split()[-1])
        blocks.setdefault(key, {}).setdefault(norms[i], []).append(i)

    for titles in blocks.values():
        for members in titles.values():  # identical normalized titles
            for j in members[1:]:
                for i in members:
                    if i == j:
                        break
                    if not blocked(i, j, identical=True):
                        works.union(i, j, "title")
                        break
        if len(titles) < 2:
            continue
        names = list(titles)
        scores = cdist(
            names,
            names,
            scorer=fuzz.token_set_ratio,
            dtype=np.uint8,
            workers=-1 if len(names) > 200 else 1,
        )
        for u, v in zip(*np.nonzero(np.triu(scores >= NEAR_MISS_FLOOR, k=1)), strict=True):
            a, b = names[u], names[v]
            set_score, sort_score = int(scores[u, v]), round(fuzz.token_sort_ratio(a, b))
            if set_score < set_min:
                reason = "below_threshold"
            elif sort_score < sort_min:
                reason = "subset_guard"
            else:
                reasons = []
                for i in titles[a]:
                    for j in titles[b]:
                        why = blocked(i, j, identical=False)
                        if why is None:
                            works.union(i, j, "title")
                        reasons.append(why)
                if None in reasons:
                    continue
                reason = Counter(reasons).most_common(1)[0][0]
            if reason != "author":
                near.add(
                    {
                        "a": titles[a][0],
                        "b": titles[b][0],
                        "token_set": set_score,
                        "token_sort": sort_score,
                        "reason": reason,
                    }
                )


def blurb_key(text: str | None, chars: int) -> str:
    """Comparable start of a blurb: its first ``chars`` characters, lowercased, punctuation
    collapsed to spaces."""
    return " ".join(re.sub(r"[\W_]+", " ", (text or "")[:chars].lower()).split())


def pick_blurb(features: Any, description: Any) -> str | None:
    """The edition's blurb: ``features`` (usually the publisher's premise, D24), else
    ``description``."""
    for text in (features, description):
        if isinstance(text, str) and text.strip():
            return text
    return None


def frame_blurbs(books: pd.DataFrame) -> Blurbs:
    """Blurbs read from ``books`` itself (its ``features``/``description`` columns, if any)."""

    def read(rows: Sequence[int]) -> dict[int, str]:
        texts = {}
        for i in rows:
            get = books.iloc[i].get
            if text := pick_blurb(get("features"), get("description")):
                texts[i] = text
        return texts

    return read


def parquet_blurbs(path: Path, asins: Sequence[str]) -> Blurbs:
    """Blurbs read from ``books_raw`` in batches, only for the requested editions (``asins`` is
    the edition row -> parent_asin list)."""

    def read(rows: Sequence[int]) -> dict[int, str]:
        wanted = {asins[i]: i for i in rows}
        texts: dict[int, str] = {}
        parquet = pq.ParquetFile(path)
        columns = ["parent_asin", "features", "description"]
        for batch in parquet.iter_batches(columns=columns, batch_size=50_000):
            values = (batch.column(c).to_pylist() for c in columns)
            for asin, features, description in zip(*values, strict=True):
                row = wanted.get(asin)
                if row is not None and (text := pick_blurb(features, description)):
                    texts[row] = text
        return texts

    return read


def subtitle_unions(
    books: pd.DataFrame,
    works: Works,
    params: Mapping[str, Any],
    near: NearMisses,
    blurbs: Blurbs,
) -> SubtitleMerges:
    """``subtitle`` merges (D32): an edition whose title has no subtitle joins an edition whose
    title is that text plus ``: subtitle`` (same author key, volume guard, no box set) when their
    blurbs are similar. Pairs with dissimilar blurbs go to ``near`` (reason
    ``subtitle_description``)."""
    threshold = float(params["subtitle_description_threshold"])
    chars = int(params["subtitle_description_chars"])
    norms, heads = books["norm_title"].tolist(), books["head"].tolist()
    akeys, volumes, boxsets = (books[c].tolist() for c in ("author_key", "volume", "boxset"))
    order = np.argsort(-books["n_interactions"].to_numpy(), kind="stable").tolist()

    bare: dict[tuple[str, str], list[int]] = {}
    for i in order:  # most-rated first
        if heads[i] is None and norms[i] and akeys[i] and not boxsets[i]:
            bare.setdefault((norms[i], akeys[i]), []).append(i)
    tried: Counter[tuple[int, int]] = Counter()
    pairs: list[tuple[int, int]] = []
    for j in order:
        if heads[j] is None or not akeys[j] or boxsets[j]:
            continue
        for i in bare.get((heads[j], akeys[j]), []):
            key = (works.find(i), works.find(j))
            if key[0] != key[1] and tried[key] < SUBTITLE_PAIRS_PER_WORKS:
                tried[key] += 1
                pairs.append((i, j))

    result = SubtitleMerges()
    texts = blurbs(sorted({k for pair in pairs for k in pair}))
    for i, j in pairs:
        result.counts["candidates"] += 1
        if works.find(i) == works.find(j):
            result.counts["already_merged"] += 1
            continue
        if volume_conflict(volumes[i], volumes[j]) or works.conflict(i, j):
            result.counts["blocked_volume_or_work"] += 1
            continue
        a, b = blurb_key(texts.get(i), chars), blurb_key(texts.get(j), chars)
        if not a or not b:
            result.counts["no_description"] += 1
            continue
        similarity = round(fuzz.token_sort_ratio(a, b))
        if similarity < threshold:
            result.counts["dissimilar_description"] += 1
            near.add(
                {
                    "a": i,
                    "b": j,
                    "token_set": round(fuzz.token_set_ratio(norms[i], norms[j])),
                    "token_sort": round(fuzz.token_sort_ratio(norms[i], norms[j])),
                    "reason": "subtitle_description",
                    "description_sim": similarity,
                }
            )
            continue
        works.union(i, j, "subtitle")
        result.counts["merged"] += 1
        result.edges.append({"a": i, "b": j, "description_sim": similarity})
    return result


def prepare(books: pd.DataFrame, rules: TitleRules, gr_index: pd.DataFrame) -> pd.DataFrame:
    """Per-edition keys used by the merge: ISBN key, normalized title, author key, volume, box-set
    flag and Goodreads work/series."""
    books = books.reset_index(drop=True).copy()
    books["isbn_key"] = [
        as_isbn13(a, b) for a, b in zip(books["isbn10"], books["isbn13"], strict=True)
    ]
    books["norm_title"] = [normalize_title(t, rules) for t in books["title"]]
    books["head"] = pd.Series([title_head(t, rules) for t in books["title"]], dtype=object)
    books["author_key"] = [author_key(a) for a in books["author"]]
    books["volume"] = pd.Series(
        [volume_number(t) for t in books["title"]], dtype=object
    )  # None, not NaN
    books["boxset"] = [is_boxset(t, rules) for t in books["title"]]
    for column in ("year", "pages"):
        books[column] = pd.Series(
            [None if pd.isna(v) else int(v) for v in books[column]], dtype=object
        )
    books["n_interactions"] = books["n_interactions"].astype("int64")
    books["category"] = [c[1] if len(c) > 1 else NO_CATEGORY for c in books["categories"]]
    gr = gr_index.drop_duplicates("isbn13")
    work = {k: w for k, w in zip(gr["isbn13"], gr["work_id"], strict=True) if isinstance(w, str)}
    series = {
        k: (sid, name, None if pd.isna(pos) else float(pos))
        for k, sid, name, pos in zip(
            gr["isbn13"], gr["series_id"], gr["series_name"], gr["series_pos"], strict=True
        )
        if isinstance(sid, str)
    }
    lookup = [  # a Kindle edition is found through its print edition's ISBN
        key if key in work else source
        for key, source in zip(books["isbn_key"], books["page_source_isbn13"], strict=True)
    ]
    books["work_id"] = [work.get(k) if k else None for k in lookup]
    books["gr_series"] = [series.get(k) if k else None for k in lookup]
    return books


def merge(
    books: pd.DataFrame,
    params: Mapping[str, Any],
    rng: np.random.Generator,
    blurbs: Blurbs | None = None,
) -> tuple[Works, NearMisses, SubtitleMerges]:
    """Run every merge rule over the prepared editions; box sets stay alone. ``blurbs`` reads
    edition blurbs for the subtitle rule (default: the ``books`` columns)."""
    work_ids = [
        None if box else w for w, box in zip(books["work_id"], books["boxset"], strict=True)
    ]
    works = Works(books["volume"].tolist(), work_ids)
    keyed = books.assign(
        isbn_key=books["isbn_key"].where(~books["boxset"], None),
        page_source_isbn13=books["page_source_isbn13"].where(~books["boxset"], None),
        work_id=work_ids,
    )
    isbn_unions(keyed, works, books["author_key"].tolist())
    goodreads_unions(keyed, works)
    near = NearMisses(rng)
    title_unions(books, works, params, near)
    subtitles = subtitle_unions(books, works, params, near, blurbs or frame_blurbs(books))
    return works, near, subtitles


def canonical_records(
    books: pd.DataFrame, slices: Sequence[str], rules: TitleRules
) -> pd.DataFrame:
    """One row per work (texts are filled in by ``fill_texts``); editions most-rated first."""
    wanted = set(slices)
    order = np.lexsort((-books["n_interactions"].to_numpy("int64"), books["root"].to_numpy()))
    roots = books["root"].to_numpy()[order]
    starts = np.flatnonzero(np.r_[True, roots[1:] != roots[:-1]]).tolist()
    col = {c: books[c].tolist() for c in books.columns}
    rows = []
    for start, end in zip(starts, [*starts[1:], len(order)], strict=True):
        idx = order[start:end].tolist()
        top = idx[0]
        akeys = [col["author_key"][i] for i in idx if col["author_key"][i]]
        akey = Counter(akeys).most_common(1)[0][0] if akeys else None
        author = next((col["author"][i] for i in idx if col["author_key"][i] == akey), None)
        paths = [list(col["categories"][i]) for i in idx]
        genres, primary = work_genres(paths)
        work_slices = list(dict.fromkeys(col["category"][i] for i in idx))
        series_id = series_name = series_pos = None
        if gr_series := next((col["gr_series"][i] for i in idx if col["gr_series"][i]), None):
            series_id, series_name, series_pos = f"gr_{gr_series[0]}", gr_series[1], gr_series[2]
        elif parsed := next((p for i in idx if (p := series_from_title(col["title"][i]))), None):
            series_name, series_pos = parsed
            series_id = f"am_{akey or ''}:{series_key(series_name)}"
        pages = [col["pages"][i] for i in idx if col["pages"][i] is not None]
        years = [col["year"][i] for i in idx if col["year"][i] is not None]
        work_ids = {col["work_id"][i] for i in idx if col["work_id"][i]}
        members = [col["parent_asin"][i] for i in idx]
        rows.append(
            {
                "canonical_id": (
                    f"gr_{next(iter(work_ids))}" if len(work_ids) == 1 else f"am_{min(members)}"
                ),
                "title": display_title(col["title"][top], rules) or col["title"][top],
                "author": author,
                "author_key": akey,
                "description": None,
                "features": None,
                "genres": genres,
                "primary_genre": primary,
                "slices": work_slices,
                "primary_slice": next((s for s in work_slices if s in wanted), None),
                "categories_raw": list(dict.fromkeys(" > ".join(p) for p in paths)),
                "year": min(years) if years else None,
                "pages": int(np.median(pages)) if pages else None,
                "image": next((col["image"][i] for i in idx if col["image"][i]), None),
                "isbn13": next((col["isbn_key"][i] for i in idx if col["isbn_key"][i]), None),
                "member_asins": members,
                "edition_kinds": list(dict.fromkeys(col["edition_kind"][i] for i in idx)),
                "series_id": series_id,
                "series_name": series_name,
                "series_pos": series_pos,
                "gr_matched": bool(work_ids),
                "n_members": len(idx),
                "n_interactions": sum(col["n_interactions"][i] for i in idx),
                "root": col["root"][top],
                "boxset": col["boxset"][top],
            }
        )
    return pd.DataFrame(rows)


def fill_texts(books_raw: Path, records: pd.DataFrame, row_of_asin: Mapping[str, int]) -> None:
    """Set each work's ``description`` and ``features`` to the longest among its editions,
    reading ``books_raw`` in batches (the text columns are most of its size)."""
    best: dict[str, list[str | None]] = {
        column: [None] * len(records) for column in ("description", "features")
    }
    parquet = pq.ParquetFile(books_raw)
    for batch in parquet.iter_batches(columns=["parent_asin", *best], batch_size=50_000):
        asins = batch.column("parent_asin").to_pylist()
        for column, values in best.items():
            for asin, text in zip(asins, batch.column(column).to_pylist(), strict=True):
                row = row_of_asin.get(asin)
                if row is not None and text and len(text) > len(values[row] or ""):
                    values[row] = text
    for column, values in best.items():
        records[column] = values


def assign_works(
    books: pd.DataFrame,
    works: Works,
    slices: Sequence[str],
    rules: TitleRules,
    drop_boxsets: bool,
) -> pd.DataFrame:
    """Build the work records and write each edition's ``root``, ``merge_rule``,
    ``in_slice_edition``, ``canonical_id`` and ``status`` into ``books``."""
    books["root"] = works.roots()
    books["merge_rule"] = [r or "singleton" for r in works.rule]
    books["in_slice_edition"] = books["category"].isin(slices)
    records = canonical_records(books, slices, rules)
    records["status"] = work_status(records, drop_boxsets)
    if records["canonical_id"].duplicated().any():
        raise DataError("data.canonical produced duplicate canonical ids")
    by_root = records.set_index("root")
    for column in ("canonical_id", "status"):
        books[column] = by_root.loc[books["root"], column].to_numpy()
    return records


def work_status(records: pd.DataFrame, drop_boxsets: bool) -> list[str]:
    """``boxset`` (dropped), ``out_of_slice`` (no edition in a genre slice) or ``kept``."""
    return [
        "boxset" if box and drop_boxsets else "out_of_slice" if s is None else "kept"
        for box, s in zip(records["boxset"], records["primary_slice"], strict=True)
    ]


def members_histogram(sizes: pd.Series) -> dict[str, int]:
    """Works per edition count, binned 1, 2, 3, 4-5, 6-10, 11+."""
    labels = ["1", "2", "3", "4-5", "6-10", "11+"]
    bins = pd.cut(sizes, [0, 1, 2, 3, 5, 10, np.inf], labels=labels)
    return {str(k): int(v) for k, v in bins.value_counts(sort=False).items()}


def build_report(
    books: pd.DataFrame,
    records: pd.DataFrame,
    works: Works,
    near: NearMisses,
    subtitles: SubtitleMerges,
    slices: Sequence[str],
    goodreads_enabled: bool,
) -> dict[str, Any]:
    """``canonical_report.json`` (s7.4 step 9, plus the slice mix for the EDA attrition table)."""
    kept = records[records["status"] == "kept"]
    kept_editions = books[books["status"] == "kept"]
    rescued = kept_editions[~kept_editions["in_slice_edition"]]

    def by(frame: pd.DataFrame, key: str) -> dict[str, dict[str, int]]:
        grouped = frame.groupby(key)["n_interactions"].agg(["size", "sum"])
        return {
            str(k): {"works": int(r["size"]), "interactions": int(r["sum"])}
            for k, r in grouped.sort_values("sum", ascending=False).iterrows()
        }

    gr_editions = int(books["work_id"].notna().sum())
    largest = kept.sort_values(["n_members", "n_interactions"], ascending=False).head(10)
    return {
        "genre_slices": list(slices),
        "editions": len(books),
        "editions_by_kind": {k: int(v) for k, v in books["edition_kind"].value_counts().items()},
        "boxsets": int(books["boxset"].sum()),
        "merge_edges": dict(sorted(works.edges.items())),
        "works": len(records),
        "multi_member_works": int((records["n_members"] > 1).sum()),
        "status": by(records, "status"),
        "kept": {
            "works": len(kept),
            "editions": len(kept_editions),
            "interactions": int(kept["n_interactions"].sum()),
            "members_per_work": members_histogram(kept["n_members"]),
            "rescued_editions": len(rescued),  # rule A: edition outside the slices, work inside
            "rescued_interactions": int(rescued["n_interactions"].sum()),
            "interactions_by_edition_kind": {
                k: int(v)
                for k, v in kept_editions.groupby("edition_kind")["n_interactions"].sum().items()
            },
            "by_primary_slice": by(kept, "primary_slice"),
            "by_primary_genre": by(kept, "primary_genre"),
            "with_series": int(kept["series_id"].notna().sum()),
            "with_author": int(kept["author_key"].notna().sum()),
        },
        "goodreads": {
            "enabled": goodreads_enabled,
            "editions_matched": gr_editions,
            "edition_match_rate": round(gr_editions / max(len(books), 1), 4),
            "kept_works_matched": int(kept["gr_matched"].sum()),
            "kept_works_unmatched": int((~kept["gr_matched"]).sum()),
        },
        "near_misses": dict(near.counts.most_common()),
        "subtitle_rule": dict(sorted(subtitles.counts.items()))
        | {
            "kept_works_joined": sum(books["status"].iat[e["a"]] == "kept" for e in subtitles.edges)
        },
        "largest_kept_works": [
            {
                "canonical_id": r.canonical_id,
                "title": r.title,
                "author": r.author,
                "n_members": int(r.n_members),
                "n_interactions": int(r.n_interactions),
            }
            for r in largest.itertuples()
        ],
    }


def build_audit(
    books: pd.DataFrame,
    records: pd.DataFrame,
    near: NearMisses,
    subtitles: SubtitleMerges,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """``merge_audit.csv``: ``AUDIT_GROUPS`` random kept works with 2+ editions (one row per
    edition, up to ``AUDIT_MAX_MEMBERS``, scored against the most-rated one),
    ``AUDIT_NEAR_MISSES`` title pairs that were not merged, spread evenly over the reasons, and
    ``AUDIT_SUBTITLE`` random kept works joined only by the subtitle rule (the two editions of the
    joining edge, ``merge_rule`` = subtitle). The reviewer fills ``verdict``: Y = right decision,
    N = wrong."""

    def edition_row(i: int, **extra: Any) -> dict[str, Any]:
        return {c: books[c].iat[i] for c in AUDIT_EDITION_COLUMNS} | extra

    rows: list[dict[str, Any]] = []
    groups = records[(records["status"] == "kept") & (records["n_members"] > 1)]
    picked = np.sort(rng.choice(len(groups), size=min(AUDIT_GROUPS, len(groups)), replace=False))
    row_of_asin = {a: i for i, a in enumerate(books["parent_asin"])}
    for n, group in enumerate(groups.iloc[picked].itertuples(), start=1):
        members = [row_of_asin[a] for a in group.member_asins[:AUDIT_MAX_MEMBERS]]
        anchor = books["norm_title"].iat[members[0]]
        for i in members:
            other = books["norm_title"].iat[i]
            rows.append(
                edition_row(
                    i,
                    audit_kind="merged_work",
                    audit_id=f"G{n:03d}",
                    reason=f"{group.n_members} editions",
                    token_set=round(fuzz.token_set_ratio(anchor, other)),
                    token_sort=round(fuzz.token_sort_ratio(anchor, other)),
                )
            )
    status = books["status"].tolist()
    queues: dict[str, list[Mapping[str, Any]]] = {}
    for m in (pair for sample in near.samples.values() for pair in sample):
        if "kept" in (status[m["a"]], status[m["b"]]):
            queues.setdefault(m["reason"], []).append(m)
    queues = {r: [q[k] for k in rng.permutation(len(q))] for r, q in sorted(queues.items())}
    chosen: list[Mapping[str, Any]] = []
    while len(chosen) < AUDIT_NEAR_MISSES and any(queues.values()):
        for queue in queues.values():  # round-robin over reasons
            if queue and len(chosen) < AUDIT_NEAR_MISSES:
                chosen.append(queue.pop())
    for n, m in enumerate(chosen, start=1):
        for i in (m["a"], m["b"]):
            rows.append(
                edition_row(
                    i,
                    audit_kind="not_merged",
                    audit_id=f"N{n:03d}",
                    reason=m["reason"],
                    token_set=m["token_set"],
                    token_sort=m["token_sort"],
                )
            )
    by_work: dict[str, Mapping[str, Any]] = {}
    for edge in subtitles.edges:  # one joining edge per kept work
        if status[edge["a"]] == "kept":
            by_work.setdefault(books["canonical_id"].iat[edge["a"]], edge)
    edges = [by_work[w] for w in sorted(by_work)]
    picked = np.sort(rng.choice(len(edges), size=min(AUDIT_SUBTITLE, len(edges)), replace=False))
    for n, k in enumerate(picked.tolist(), start=1):
        edge = edges[k]
        a, b = books["norm_title"].iat[edge["a"]], books["norm_title"].iat[edge["b"]]
        for i in (edge["a"], edge["b"]):
            rows.append(
                edition_row(
                    i,
                    audit_kind="subtitle_merge",
                    audit_id=f"S{n:03d}",
                    reason=f"description similarity {edge['description_sim']}",
                    token_set=round(fuzz.token_set_ratio(a, b)),
                    token_sort=round(fuzz.token_sort_ratio(a, b)),
                )
                | {"merge_rule": "subtitle"}
            )
    audit = pd.DataFrame(rows).reindex(columns=AUDIT_COLUMNS)
    audit["year"] = audit["year"].astype("Int64")  # 2015, not 2015.0, when a year is missing
    return audit


def write_table(frame: pd.DataFrame, schema: pa.Schema, path: Path) -> None:
    """Write ``frame``'s ``schema`` columns as zstd parquet."""
    frame = frame.reindex(columns=schema.names).copy()
    for f in schema:
        if pa.types.is_integer(f.type):
            frame[f.name] = pd.to_numeric(frame[f.name]).astype(f"Int{f.type.bit_width}")
        elif pa.types.is_floating(f.type):
            frame[f.name] = pd.to_numeric(frame[f.name]).astype("float32")
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False)
    pq.write_table(table, path, compression=PARQUET_COMPRESSION)


@stage(
    "data.canonical",
    inputs=["data:books_raw.parquet", "data:gr_isbn_index.parquet"],
    outputs=[
        "data:asin_to_canonical.parquet",
        "data:books_canonical.parquet",
        "results:canonical/canonical_report.json",
        "results:canonical/merge_audit.csv",
    ],
    config_keys=["data.genre_slices", "data.use_goodreads_graph", "canonical"],
)
def build_canonical(ctx: StageContext) -> None:
    """Merge editions into works, keep works with an edition in a genre slice, write the audit."""
    slices = require_genre_slices(ctx.param("data.genre_slices"))
    params = ctx.param("canonical")
    rules = TitleRules.from_config(params)
    raw_path = ctx.path("data:books_raw.parquet")
    books = prepare(
        pd.read_parquet(raw_path, columns=KEY_COLUMNS),
        rules,
        pd.read_parquet(ctx.path("data:gr_isbn_index.parquet")),
    )
    ctx.count("editions", len(books))
    blurbs = parquet_blurbs(raw_path, books["parent_asin"].tolist())
    works, near, subtitles = merge(books, params, ctx.rng, blurbs)
    records = assign_works(books, works, slices, rules, bool(params["drop_boxsets"]))

    kept = records[records["status"] == "kept"].sort_values("canonical_id").reset_index(drop=True)
    fill_texts(raw_path, kept, {a: r for r, ms in enumerate(kept["member_asins"]) for a in ms})
    write_table(kept, CANONICAL_SCHEMA, ctx.out("data:books_canonical.parquet"))
    write_table(books, MAP_SCHEMA, ctx.out("data:asin_to_canonical.parquet"))

    enabled = bool(ctx.param("data.use_goodreads_graph"))
    report = build_report(books, records, works, near, subtitles, slices, enabled)
    path = ctx.out("results:canonical/canonical_report.json")
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    audit = build_audit(books, records, near, subtitles, ctx.rng)
    # utf-8-sig: Excel on Windows reads a CSV as UTF-8 only with a BOM
    audit.to_csv(ctx.out("results:canonical/merge_audit.csv"), index=False, encoding="utf-8-sig")

    for key in ("works", "multi_member_works", "boxsets"):
        ctx.count(key, report[key])
    for status, value in report["status"].items():
        ctx.count(f"works_{status}", value["works"])
        ctx.count(f"interactions_{status}", value["interactions"])
    ctx.count("rescued_editions", report["kept"]["rescued_editions"])
    ctx.count("audit_rows", len(audit))
    ctx.notes["merge_edges"] = report["merge_edges"]
    ctx.notes["near_misses"] = report["near_misses"]
    ctx.notes["subtitle_rule"] = report["subtitle_rule"]
    ctx.log.info("\n%s", report_summary(report))
    ctx.log.warning(AUDIT_MESSAGE)


AUDIT_MESSAGE = (
    "STOP: review results/<version>/canonical/merge_audit.csv (fill verdict Y/N), record the "
    "merge precision in docs/decisions.md, then run data.reviews."
)


def report_summary(report: Mapping[str, Any]) -> str:
    """Human-readable summary of ``canonical_report.json``."""
    kept = report["kept"]
    lines = [
        f"editions {report['editions']:,} -> works {report['works']:,} "
        f"({report['multi_member_works']:,} with 2+ editions); box sets {report['boxsets']:,}",
        "merge edges: " + ", ".join(f"{k} {v:,}" for k, v in report["merge_edges"].items()),
        f"kept: {kept['works']:,} works, {kept['editions']:,} editions, "
        f"{kept['interactions']:,} interactions; rescued by rule A: "
        f"{kept['rescued_editions']:,} editions, {kept['rescued_interactions']:,} interactions",
        "editions per kept work: "
        + ", ".join(f"{k}: {v:,}" for k, v in kept["members_per_work"].items()),
        "kept works by primary slice:",
        *(
            f"  {v['interactions']:>10,} interactions {v['works']:>8,} works  {k}"
            for k, v in kept["by_primary_slice"].items()
        ),
        "not merged (near misses by reason): "
        + ", ".join(f"{k} {v:,}" for k, v in report["near_misses"].items()),
        "subtitle rule (D32): "
        + ", ".join(f"{k} {v:,}" for k, v in report.get("subtitle_rule", {}).items()),
        "largest kept works:",
        *(
            f"  {w['n_members']:>4} editions {w['n_interactions']:>7,} interactions  "
            f"{w['title']} / {w['author']}"
            for w in report["largest_kept_works"]
        ),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the summary of the last data.canonical run."""
    parser = argparse.ArgumentParser(description="Report on data.canonical's outputs.")
    parser.add_argument("--config", help="config YAML path (default ml/config.yaml)")
    parser.add_argument("--sample", type=int, metavar="N", help="read the --sample run")
    args = parser.parse_args(argv)
    paths = RunPaths.from_config(load_config(args.config), args.sample)
    report_path = paths.resolve("results:canonical/canonical_report.json")
    print(report_summary(json.loads(report_path.read_text(encoding="utf-8"))))
    print(AUDIT_MESSAGE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
