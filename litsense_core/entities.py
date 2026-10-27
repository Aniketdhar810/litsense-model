"""Entity resolution (CLAUDE.md s14.5): titles and authors named in onboarding text or chat.

``resolve_title`` / ``resolve_author`` use rapidfuzz with the brief's thresholds (WRatio >= 88
and a 5-point margin over the runner-up for titles; token_sort_ratio >= 85 with the same margin
for authors) and return ``Resolution(kind="ambiguous")`` with up to three options otherwise.
Indices are built once per store. The public ``resolve_entities(query)`` keeps its pre-Phase 6
behaviour (exact normalized match, then a difflib ratio) so author-mode search is unchanged;
the chat and onboarding paths use the rapidfuzz resolvers and log every resolution.
"""

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from rapidfuzz import fuzz, process

from litsense_core.schemas import EntityMatch
from litsense_core.store import ArtifactStore

TITLE_ACCEPT = 88.0
AUTHOR_ACCEPT = 85.0
MARGIN = 5.0
OPTIONS = 3
_INDEX_CACHE: dict[int, tuple[ArtifactStore, "EntityIndex"]] = {}


def norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"^(?:the|a|an) ", "", " ".join(text.split()))


def author_key(name: str) -> str:
    """Lowercase, initials joined: "J.K. Rowling" and "JK Rowling" -> "jk rowling"."""
    text = re.sub(r"\.", "", name.lower())
    return " ".join(text.split())


@dataclass
class Resolution:
    kind: Literal["title", "author", "ambiguous", "none"]
    query: str
    book_ids: list[int] = field(default_factory=list)
    author: str | None = None
    author_id: int | None = None
    options: list[str] = field(default_factory=list)
    match: str | None = None
    score: float = 0.0

    def log(self) -> dict[str, Any]:
        """One debug record (query, match, score) for the reply's debug field."""
        return {"query": self.query, "kind": self.kind, "match": self.match,
                "score": round(self.score, 1)}  # fmt: skip


class EntityIndex:
    def __init__(self, store: ArtifactStore) -> None:
        cat = store.catalog
        self.store = store
        self.title_ids: dict[str, list[int]] = {}
        for i, t in enumerate(cat.title):
            self.title_ids.setdefault(norm(t), []).append(i)
        for ids in self.title_ids.values():  # most popular first
            ids.sort(key=lambda i: (-float(cat.popularity[i]), i))
        self.title_keys = list(self.title_ids)
        self.author_ids: dict[str, int] = {}
        for k, a in enumerate(cat.author_labels):
            self.author_ids.setdefault(author_key(a), k)
        self.author_keys = list(self.author_ids)

    def display_title(self, key: str) -> str:
        return self.store.catalog.title[self.title_ids[key][0]]

    def display_author(self, key: str) -> str:
        return self.store.catalog.author_labels[self.author_ids[key]]


def index_for(store: ArtifactStore) -> EntityIndex:
    entry = _INDEX_CACHE.get(id(store))
    if entry is None or entry[0] is not store:
        entry = (store, EntityIndex(store))
        _INDEX_CACHE[id(store)] = entry
    return entry[1]


def _pick(
    scored: list[tuple[str, float, Any]], accept: float, same: Any = None
) -> tuple[str | None, float, list[str]]:
    """Apply the accept-and-margin rule; returns (key, score, ambiguous keys)."""
    if not scored or scored[0][1] < accept:
        return None, scored[0][1] if scored else 0.0, []
    best, score, _ = scored[0]
    rivals = [k for k, s, _ in scored[1:] if s >= accept and score - s < MARGIN]
    if same is not None:
        rivals = [k for k in rivals if not same(best, k)]
    if rivals:
        return None, score, [best, *rivals][:OPTIONS]
    return best, score, []


def resolve_title(q: str, store: ArtifactStore, author_hint: str | None = None) -> Resolution:
    """Exact normalized match first, else rapidfuzz WRatio with the 88 / 5 rule (s14.5). The
    same title by several authors uses ``author_hint``, else the most popular edition."""
    index = index_for(store)
    cat = store.catalog
    key = norm(q)
    if not key:
        return Resolution("none", q)
    score = 100.0
    if key not in index.title_ids:
        scored = process.extract(key, index.title_keys, scorer=fuzz.WRatio, limit=5)
        chosen, score, options = _pick([(k, s, None) for k, s, _ in scored], TITLE_ACCEPT)
        if chosen is None:
            if options:
                return Resolution(
                    "ambiguous", q, options=[index.display_title(k) for k in options], score=score
                )
            return Resolution("none", q, score=score)
        key = chosen
    ids = list(index.title_ids[key])
    if len(ids) > 1 and author_hint:
        hint = author_key(author_hint)
        preferred = [i for i in ids if cat.author[i] and author_key(cat.author[i]) == hint]
        ids = preferred or ids
    return Resolution("title", q, book_ids=ids, match=cat.title[ids[0]], score=score)


def resolve_author(q: str, store: ArtifactStore) -> Resolution:
    """Exact normalized author, else token_sort_ratio >= 85 with the margin rule."""
    index = index_for(store)
    cat = store.catalog
    key = author_key(q)
    if not key:
        return Resolution("none", q)
    score = 100.0
    if key not in index.author_ids:
        scored = process.extract(key, index.author_keys, scorer=fuzz.token_sort_ratio, limit=5)
        chosen, score, options = _pick([(k, s, None) for k, s, _ in scored], AUTHOR_ACCEPT)
        if chosen is None:
            if options:
                return Resolution(
                    "ambiguous", q, options=[index.display_author(k) for k in options], score=score
                )
            return Resolution("none", q, score=score)
        key = chosen
    k = index.author_ids[key]
    return Resolution(
        "author", q, book_ids=list(cat.author_items[k]), author=cat.author_labels[k],
        author_id=k, match=cat.author_labels[k], score=score,
    )  # fmt: skip


def resolve_any(q: str, store: ArtifactStore, prefer: str = "title") -> Resolution:
    """Try one kind, then the other ("like Andy Weir" names an author after "like")."""
    first, second = (resolve_title, resolve_author) if prefer == "title" else (
        resolve_author, resolve_title)  # fmt: skip
    out = first(q, store)
    if out.kind == "none":
        alt = second(q, store)
        if alt.kind != "none":
            return alt
    return out


def resolve_entities(query: str, store: ArtifactStore) -> EntityMatch:
    """The public s15.1 function, unchanged from Phase 4 (exact match, then difflib)."""
    cat = store.catalog
    q = norm(query)
    by_title: dict[str, list[int]] = {}
    for i, t in enumerate(cat.title):
        by_title.setdefault(norm(t), []).append(i)
    by_author = {norm(a): k for k, a in enumerate(cat.author_labels)}
    if q in by_title:
        return EntityMatch(kind="title", book_ids=by_title[q])
    if q in by_author:
        k = by_author[q]
        return EntityMatch(kind="author", book_ids=cat.author_items[k], author=cat.author_labels[k])
    scored = sorted(
        ((difflib.SequenceMatcher(None, q, key).ratio(), kind, key)
         for kind, index in (("title", by_title), ("author", by_author)) for key in index),
        reverse=True,
    )  # fmt: skip
    close = [s for s in scored[:5] if s[0] >= 0.8]
    if not close:
        return EntityMatch(kind="none")
    if len(close) > 1 and close[0][0] - close[1][0] < 0.05:
        options = [
            cat.title[by_title[key][0]] if kind == "title" else cat.author_labels[by_author[key]]
            for _, kind, key in close[:3]
        ]
        return EntityMatch(kind="ambiguous", options=options)
    _, kind, key = close[0]
    if kind == "title":
        return EntityMatch(kind="title", book_ids=by_title[key])
    k = by_author[key]
    return EntityMatch(kind="author", book_ids=cat.author_items[k], author=cat.author_labels[k])
