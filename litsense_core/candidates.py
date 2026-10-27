"""Candidate sources and the filters applied before ranking (CLAUDE.md s13.1, s13.2).

Sources: ``cf``, ``semantic`` (quota shared with the onboarding text vector when present),
``item_item``, ``popularity``, ``author``, ``aspect`` (D63: ``aspect_topitems`` lists for the
reader's top-2 positive aspects, intersected with preferred genres when that leaves enough) and
``longtail`` (by content similarity, since D49 found the projection does not rank cold items).
Quotas come from ``ranking.quotas[bucket]``; the union is deduplicated by round-robin over the
sources in quota order and capped at ``ranking.candidates_total``.
"""

from collections.abc import Iterable, Mapping

import numpy as np

from litsense_core.profile import Profile
from litsense_core.schemas import Filters
from litsense_core.store import ArtifactStore

SOURCE_ORDER = ("cf", "semantic", "item_item", "popularity", "author", "aspect", "longtail")
ITEM_ITEM_PER_ANCHOR = 15


def _top(scores: np.ndarray, k: int, exclude: set[int]) -> list[int]:
    """Ids of the ``k`` highest scores, best first, skipping ``exclude``."""
    if k <= 0 or scores.size == 0:
        return []
    want = min(scores.size, k + len(exclude))
    part = (
        np.argpartition(-scores, want - 1)[:want] if want < scores.size else np.arange(scores.size)
    )
    order = part[np.argsort(-scores[part], kind="stable")]
    return [int(i) for i in order if int(i) not in exclude][:k]


def generate(
    profile: Profile, store: ArtifactStore, quotas: Mapping[str, int]
) -> dict[str, list[int]]:
    """Per-source ordered candidate lists (provenance kept); ``profile.seen`` is excluded."""
    cat = store.catalog
    seen = profile.seen
    out: dict[str, list[int]] = {}

    q = int(quotas.get("cf", 0))
    if q and profile.cf_vec is not None:
        scores = np.asarray(store.lgcn_item, dtype=np.float32) @ profile.cf_vec
        out["cf"] = _top(scores, q, seen)

    q = int(quotas.get("semantic", 0))
    if q and profile.semantic_vec is not None:
        if profile.onboarding_vec is not None:  # split the quota with the raw text vector
            half = max(1, q // 2)
            a, _ = store.content_index.search(profile.semantic_vec, half, exclude=seen)
            b, _ = store.content_index.search(profile.onboarding_vec, q - half, exclude=seen)
            merged: list[int] = []
            for pair in zip(a.tolist(), b.tolist(), strict=False):
                merged.extend(int(i) for i in pair)
            merged.extend(int(i) for i in a.tolist()[len(b) :] + b.tolist()[len(a) :])
            out["semantic"] = list(dict.fromkeys(merged))[:q]
        else:
            ids, _ = store.content_index.search(profile.semantic_vec, q, exclude=seen)
            out["semantic"] = [int(i) for i in ids]

    q = int(quotas.get("item_item", 0))
    if q and profile.anchors:
        rows: list[int] = []
        per = min(ITEM_ITEM_PER_ANCHOR, max(1, q // max(1, len(profile.anchors))))
        for anchor in profile.anchors:
            rows.extend(
                int(j) for j in store.item_nn[anchor][: per + len(seen)] if int(j) not in seen
            )
        out["item_item"] = list(dict.fromkeys(rows))[:q]

    q = int(quotas.get("popularity", 0))
    if q:
        preferred = set(profile.preferred_genres[:5]) | set(profile.onboarding_genres)
        pool = [int(i) for i in store.trending_top if int(i) not in seen]
        if preferred:
            genre_pool = [i for i in pool if set(cat.genres_of(i)) & preferred]
            pool = genre_pool if len(genre_pool) >= q // 2 else pool
        out["popularity"] = pool[:q]

    q = int(quotas.get("author", 0))
    if q and profile.favorite_authors:
        rows = []
        for author_id in profile.favorite_authors:  # affinity order
            books = [i for i in cat.author_items[author_id] if i not in seen]
            books.sort(key=lambda i: (-float(cat.popularity[i]), i))
            rows.extend(books)
        out["author"] = list(dict.fromkeys(rows))[:q]

    q = int(quotas.get("aspect", 0))
    if q and store.aspect_topitems and profile.aspect_vec is not None:
        preferred = set(profile.preferred_genres[:5]) | set(profile.onboarding_genres)
        lists = []
        for a, _value in profile.top_aspects(threshold=0.1)[:2]:
            name = store.aspect_names[a]
            items = [i for i in store.aspect_topitems.get(name, []) if i not in seen]
            if preferred:
                in_genre = [i for i in items if set(cat.genres_of(i)) & preferred]
                items = in_genre if len(in_genre) >= q // 2 else items
            lists.append(items)
        rows = []
        for pair in zip(*lists, strict=False):
            rows.extend(pair)
        for lst in lists:
            rows.extend(lst)
        if rows:
            out["aspect"] = list(dict.fromkeys(rows))[:q]

    q = int(quotas.get("longtail", 0))
    if q:
        tail = np.flatnonzero(cat.is_longtail)
        if profile.semantic_vec is not None and tail.size:
            ids, _ = store.content_index.search(profile.semantic_vec, q, exclude=seen, allow=tail)
            out["longtail"] = [int(i) for i in ids]
        elif tail.size:
            scores = cat.trending[tail]
            out["longtail"] = [int(tail[i]) for i in _top(scores, q, set())
                               if int(tail[i]) not in seen][:q]  # fmt: skip
    return out


def union(sources: Mapping[str, list[int]], total: int) -> tuple[list[int], dict[int, list[str]]]:
    """Round-robin merge over sources in quota order; returns ids and each id's source flags."""
    provenance: dict[int, list[str]] = {}
    for name, ids in sources.items():
        for i in ids:
            provenance.setdefault(i, []).append(name)
    ordered: list[int] = []
    queues = {name: list(ids) for name, ids in sources.items() if ids}
    names = [n for n in SOURCE_ORDER if n in queues] + [n for n in queues if n not in SOURCE_ORDER]
    taken: set[int] = set()
    while queues and len(ordered) < total:
        for name in names:
            if name not in queues:
                continue
            while queues[name]:
                i = queues[name].pop(0)
                if i not in taken:
                    taken.add(i)
                    ordered.append(i)
                    break
            if not queues[name]:
                del queues[name]
            if len(ordered) >= total:
                break
    return ordered, {i: provenance[i] for i in ordered}


def passes_filters(store: ArtifactStore, book_id: int, filters: Filters | None) -> bool:
    cat = store.catalog
    if filters is None:
        return True
    year, pages = int(cat.year[book_id]), int(cat.pages[book_id])
    genres = set(cat.genres_of(book_id))
    if filters.min_year is not None and (year < 0 or year < filters.min_year):
        return False
    if filters.max_year is not None and (year < 0 or year > filters.max_year):
        return False
    if filters.min_pages is not None and (pages < 0 or pages < filters.min_pages):
        return False
    if filters.max_pages is not None and (pages < 0 or pages > filters.max_pages):
        return False
    if filters.genres_include and not genres & set(filters.genres_include):
        return False
    return not genres & set(filters.genres_exclude)


def series_allows(store: ArtifactStore, book_id: int, profile: Profile) -> bool:
    """Series guard (s13.2): volume p > 1 passes only if the reader has reached p - 1."""
    cat = store.catalog
    s, pos = int(cat.series_id[book_id]), float(cat.series_pos[book_id])
    if s < 0 or np.isnan(pos) or pos <= 1.0:
        return True
    return profile.series_progress.get(s, 0.0) >= pos - 1.0


def next_in_series(store: ArtifactStore, profile: Profile) -> list[int]:
    """The exact next volume of every series the reader is in, when it is unseen (s12.4)."""
    cat = store.catalog
    out = []
    for s, progress in profile.series_progress.items():
        for i in cat.series_items[s]:
            if i not in profile.seen and float(cat.series_pos[i]) == progress + 1.0:
                out.append(i)
    return out


def apply_filters(
    ids: Iterable[int],
    provenance: Mapping[int, list[str]],
    profile: Profile,
    store: ArtifactStore,
    filters: Filters | None,
    exclude: Iterable[int] | None,
    min_reviews: int,
) -> list[int]:
    """s13.2: drop seen, excluded, hard-filtered, muted, thin-review (except longtail-source) and
    series-guarded items; the exact next volume is kept out of the main feed too."""
    cat = store.catalog
    blocked = set(profile.seen) | ({int(i) for i in exclude} if exclude else set())
    nxt = set(next_in_series(store, profile))
    kept = []
    for i in ids:
        if i in blocked or i in nxt:
            continue
        if set(cat.genres_of(i)) & profile.muted_genres:
            continue
        if not passes_filters(store, i, filters):
            continue
        thin = int(cat.n_reviews[i]) < min_reviews
        if thin and "longtail" not in provenance.get(i, []):
            continue
        if not series_allows(store, i, profile):
            continue
        kept.append(i)
    return kept
