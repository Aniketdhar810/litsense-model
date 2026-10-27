"""Onboarding text -> ``OnboardingProfile`` (CLAUDE.md s14.4).

LLM parse (``onboarding_parse`` prompt) with the vocabularies enforced after the fact: genres
must be catalogue genre labels, aspects must be the twelve aspect names, descriptors are clipped
to [-1, 1]; named titles and authors go through ``entities`` and resolved titles join the picks.
The keyword parser that served before Phase 6 is the fallback and is unchanged (``method`` is
``embed_only`` then). The raw text is always kept by the caller as ``UserState.onboarding_text``,
so the ONNX encoder still embeds it whatever the parse did (s12.2).
"""

import re
from typing import Any

from litsense_core import entities
from litsense_core.llm import PrivacyError, get_client
from litsense_core.schemas import ASPECT_LABELS, OnboardingProfile, OnboardingResult
from litsense_core.store import ArtifactStore

MAX_THEMES = 6
MAX_NAMED = 5
DESCRIPTOR_WORDS = {
    "darkness": ("cozy", "dark"),
    "pace": ("slow", "fast"),
    "complexity_level": ("easy", "dense"),
}


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"^(?:the|a|an) ", "", " ".join(text.split()))


def keyword_parse(text: str, picks: list[int], store: ArtifactStore) -> OnboardingResult:
    """The Phase 4 keyword parse (unchanged): catalogue genre labels and titles in the text."""
    cat = store.catalog
    for i in picks:
        cat.check_id(i)
    low = text.lower()
    profile = OnboardingProfile()
    for clause in re.split(r"[.;!?]|,|\bbut\b", low):
        negated = re.search(r"\b(?:not|no|never|hate|avoid|dislike|nothing)\b", clause) is not None
        for g in cat.genre_labels:
            if g and g.lower() in clause:
                (profile.disliked_genres if negated else profile.liked_genres).append(g)
    norm_text = _norm(text)
    profile.named_titles = [t for t in cat.title if len(t) > 3 and _norm(t) in norm_text][:5]
    profile.named_authors = [a for a in cat.author_labels if len(a) > 3 and _norm(a) in norm_text][
        :5
    ]
    resolved = list(dict.fromkeys([*picks, *(cat.title.index(t) for t in profile.named_titles)]))
    chips = [f"Likes: {g}" for g in profile.liked_genres] + [
        f"Avoids: {g}" for g in profile.disliked_genres
    ]
    return OnboardingResult(
        profile=profile, resolved_picks=resolved, display_chips=chips, method="embed_only"
    )


def clean_profile(raw: OnboardingProfile, store: ArtifactStore) -> OnboardingProfile:
    """Drop anything outside the vocabularies; clip descriptors; cap list lengths."""
    cat = store.catalog
    genre_by_norm = {g.lower(): g for g in cat.genre_labels if g}
    aspects = set(store.aspect_names) or set(ASPECT_LABELS)
    descriptors = list((store.config.get("aspects") or {}).get("descriptors") or DESCRIPTOR_WORDS)

    def genres(values: list[str]) -> list[str]:
        out = [
            genre_by_norm[v.strip().lower()] for v in values if v.strip().lower() in genre_by_norm
        ]
        return list(dict.fromkeys(out))

    def aspect_list(values: list[str]) -> list[str]:
        return list(dict.fromkeys(v for v in values if v in aspects))

    return OnboardingProfile(
        liked_genres=genres(raw.liked_genres),
        disliked_genres=[
            g for g in genres(raw.disliked_genres) if g not in genres(raw.liked_genres)
        ],
        positive_aspects=aspect_list(raw.positive_aspects),
        negative_aspects=[
            a
            for a in aspect_list(raw.negative_aspects)
            if a not in aspect_list(raw.positive_aspects)
        ],  # fmt: skip
        descriptors={
            k: float(max(-1.0, min(1.0, float(v))))
            for k, v in raw.descriptors.items()
            if k in descriptors and isinstance(v, int | float)
        },
        themes=[str(t).strip() for t in raw.themes if str(t).strip()][:MAX_THEMES],
        named_titles=[str(t).strip() for t in raw.named_titles if str(t).strip()][:MAX_NAMED],
        named_authors=[str(a).strip() for a in raw.named_authors if str(a).strip()][:MAX_NAMED],
    )


def display_chips(profile: OnboardingProfile) -> list[str]:
    chips = [f"Likes: {g}" for g in profile.liked_genres]
    chips += [f"Avoids: {g}" for g in profile.disliked_genres]
    chips += [f"Values: {ASPECT_LABELS.get(a, a)}" for a in profile.positive_aspects]
    chips += [f"Avoids: {ASPECT_LABELS.get(a, a)}" for a in profile.negative_aspects]
    for name, value in profile.descriptors.items():
        words = DESCRIPTOR_WORDS.get(name)
        if words and abs(value) >= 0.2:
            chips.append(f"Prefers: {words[value > 0]}")
    chips += [f"Theme: {t}" for t in profile.themes[:3]]
    return chips


def parse_onboarding(
    text: str, picks: list[int], store: ArtifactStore, disabled_ai: bool = False
) -> OnboardingResult:
    cat = store.catalog
    for i in picks:
        cat.check_id(i)
    client = get_client()
    if disabled_ai or not client.available or not text.strip():
        return keyword_parse(text, picks, store)
    variables: dict[str, Any] = {
        "genres": ", ".join(g for g in cat.genre_labels if g),
        "text": text.strip()[:1000],
    }
    try:
        result = client.json("onboarding_parse", variables, OnboardingProfile, "parse")
    except PrivacyError:
        return keyword_parse(text, picks, store)
    if not result.ok or result.data is None:
        return keyword_parse(text, picks, store)
    profile = clean_profile(result.data, store)  # type: ignore[arg-type]
    resolved: list[int] = list(picks)
    kept_titles: list[str] = []
    for title in profile.named_titles:
        res = entities.resolve_title(title, store)
        if res.kind == "title":
            resolved.append(res.book_ids[0])
            kept_titles.append(cat.title[res.book_ids[0]])
    kept_authors: list[str] = []
    for name in profile.named_authors:
        res = entities.resolve_author(name, store)
        if res.kind == "author" and res.author:
            kept_authors.append(res.author)
    profile.named_titles, profile.named_authors = kept_titles, kept_authors
    return OnboardingResult(
        profile=profile,
        resolved_picks=list(dict.fromkeys(resolved)),
        display_chips=display_chips(profile),
        method="llm",
    )
