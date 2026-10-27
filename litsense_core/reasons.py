"""Evidence and template reasons (CLAUDE.md s12.5).

Reasons are chosen in the brief's priority order. Since D63 the aspect template (#2) and the safe
snippet come from the bundled aspect profiles and evidence snippets; on an older bundle without
them those parts are skipped. Every number in a reason comes from an evidence field.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from litsense_core.profile import Profile
from litsense_core.schemas import ASPECT_LABELS, AspectStrength, Evidence, LikedItem, Snippet
from litsense_core.store import ArtifactStore

SIMILAR_THRESHOLD = 0.6  # template #3: content cosine to a liked book
# Template #2 thresholds on centered values (D65): the reader's aspect vector is already centered
# on the aspect means; the item's shrunk value is centered here. Raw polarity is mostly positive
# (means 0.4 to 0.56 on v0), so the brief's "> 0.3" on raw values fired for almost every pair.
USER_ASPECT_THRESHOLD = 0.15
ITEM_ASPECT_THRESHOLD = 0.08


def similar_liked(
    book_id: int, profile: Profile, store: ArtifactStore, top: int = 3
) -> list[tuple[int, float]]:
    """Liked books most similar (content cosine) to ``book_id``, best first."""
    if not profile.liked:
        return []
    liked = np.asarray(profile.liked[:50], dtype=np.int64)
    sims = store.content_rows(liked) @ store.content_rows(np.array([book_id]))[0]
    order = np.argsort(-sims, kind="stable")[:top]
    return [(int(liked[o]), float(sims[o])) for o in order]


def label_of(name: str) -> str:
    return ASPECT_LABELS.get(name, name.replace("_", " "))


def matching_aspect(book_id: int, profile: Profile, store: ArtifactStore) -> int | None:
    """The strongest aspect the reader values more than average that the item is also praised
    for more than the average book (both on centered values, D65)."""
    if profile.aspect_vec is None or store.item_aspects is None or store.aspect_means is None:
        return None
    for a, _value in profile.top_aspects(threshold=USER_ASPECT_THRESHOLD):
        centered = float(store.item_aspects[book_id, a] - store.aspect_means[a])
        if centered > ITEM_ASPECT_THRESHOLD:
            return a
    return None


def item_top_aspect(book_id: int, store: ArtifactStore) -> int | None:
    """The item's most praised aspect weighted by how often it is mentioned (for snippets)."""
    if store.item_pos_rate is None or store.item_mention_rate is None:
        return None
    score = store.item_pos_rate[book_id] * store.item_mention_rate[book_id]
    if not np.any(score > 0):
        return None
    return int(np.argmax(score))


def snippet_for(book_id: int, aspect: int | None, store: ArtifactStore) -> dict[str, Any] | None:
    """The best stored safe snippet for ``aspect`` (positive side), as a ``Snippet`` dump."""
    if aspect is None or store.snippets is None:
        return None
    name = store.aspect_names[aspect]
    slots = store.snippets.for_item(book_id).get(name)
    if not slots or not slots.get("pos"):
        return None
    best = slots["pos"][0]
    return Snippet(
        text=best["text"],
        review_ref=int(best["review_row"]),
        helpful_votes=int(best["helpful_votes"]),
        aspect=name,
    ).model_dump()


def _strengths(profile: Profile, store: ArtifactStore, sign: int) -> list[dict[str, Any]]:
    conf = profile.aspect_conf
    out = []
    for a, value in profile.top_aspects(sign=sign, threshold=USER_ASPECT_THRESHOLD)[:3]:
        name = store.aspect_names[a]
        n = (
            int(round(float(conf[a]) * 3 / max(1e-6, 1 - float(conf[a]))))
            if conf is not None
            else 0
        )
        out.append(
            AspectStrength(
                aspect=name, label=label_of(name), strength=round(value, 3), n=n
            ).model_dump()
        )
    return out


def build_evidence(
    book_id: int,
    profile: Profile,
    store: ArtifactStore,
    sources: Sequence[str],
    signals: Mapping[str, Any] | None = None,
) -> Evidence:
    cat = store.catalog
    author_id = int(cat.author_id[book_id])
    similar = [
        LikedItem(book_id=i, title=cat.title[i], stars=profile.stars.get(i, 5.0)).model_dump()
        | {"similarity": round(s, 3)}
        for i, s in similar_liked(book_id, profile, store)
    ]
    match = matching_aspect(book_id, profile, store)
    chosen = match if match is not None else item_top_aspect(book_id, store)
    names = store.aspect_names
    item_evidence: dict[str, Any] = {
        "aspect_profile": {},
        "mention_rates": {},
        "pos_rates": {},
        "snippet": snippet_for(book_id, chosen, store),
    }
    if store.item_aspects is not None and names:
        item_evidence["aspect_profile"] = {
            n: round(float(v), 3) for n, v in zip(names, store.item_aspects[book_id], strict=True)
        }
    if store.item_mention_rate is not None and names:
        item_evidence["mention_rates"] = {
            n: round(float(v), 3)
            for n, v in zip(names, store.item_mention_rate[book_id], strict=True)
        }
    if store.item_pos_rate is not None and names:
        item_evidence["pos_rates"] = {
            n: round(float(v), 3) for n, v in zip(names, store.item_pos_rate[book_id], strict=True)
        }
    return Evidence(
        book={
            "book_id": book_id,
            "title": cat.title[book_id],
            "author": cat.author[book_id],
            "genres": list(cat.genres_of(book_id)),
            "year": int(cat.year[book_id]) if cat.year[book_id] >= 0 else None,
            "bayes_rating": round(float(cat.bayes_rating[book_id]), 2),
            "rating_count": int(cat.rating_count[book_id]),
            "is_longtail": bool(cat.is_longtail[book_id]),
        },
        user_evidence={
            "top_positive_aspects": _strengths(profile, store, 1),
            "top_negative_aspects": _strengths(profile, store, -1),
            "similar_liked_items": similar,
            "favorite_author": cat.author[book_id]
            if author_id in profile.favorite_authors
            else None,
            "preferred_genres": profile.preferred_genres[:3],
        },
        item_evidence=item_evidence,
        signals={
            "sources": list(sources),
            "bucket": profile.bucket,
            "cf_available": profile.cf_available,
            "aspect_match": names[match] if match is not None else None,
            **dict(signals or {}),
        },
    )


def template_reason(
    book_id: int, profile: Profile, store: ArtifactStore, evidence: Evidence
) -> str:
    """First template of s12.5 whose condition holds."""
    cat = store.catalog
    author_id = int(cat.author_id[book_id])
    if author_id in profile.favorite_authors and author_id in profile.favorite_author_liked:
        liked = profile.favorite_author_liked[author_id]
        return f"Because you loved {cat.author[book_id]}'s {cat.title[liked]}"
    matched = evidence.signals.get("aspect_match")
    if matched and store.item_pos_rate is not None:
        a = store.aspect_index(matched)
        if a is not None:
            pos_pct = int(round(100 * float(store.item_pos_rate[book_id, a])))
            return (
                f"Readers praise its {label_of(matched)} ({pos_pct}% of mentions), "
                "something you value"
            )
    similar = evidence.user_evidence.get("similar_liked_items", [])
    if similar and similar[0].get("similarity", 0.0) >= SIMILAR_THRESHOLD:
        stars = similar[0]["stars"]
        return f"Similar to {similar[0]['title']}, which you rated {stars:g}★"
    theme = next((g for g in cat.genres_of(book_id) if g in profile.onboarding_genres), None)
    if theme:
        return f"Matches what you described: {theme.lower()}"
    sources = evidence.signals.get("sources", [])
    if "longtail" in sources and profile.n >= 5:
        return "A lesser-known pick that fits your taste"
    if "cf" in sources and profile.cf_available > 0:
        # Not in the s12.5 list: the collaborative source needs its own line when no liked title
        # is similar enough, otherwise every CF pick reads as "popular" (D56).
        return "Readers with a taste like yours loved this"
    if profile.n == 0 and not profile.onboarding_genres:
        return "Popular with readers right now"
    return f"Popular with {cat.primary_genre_of(book_id)} readers right now"
