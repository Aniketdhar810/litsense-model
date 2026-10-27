"""One chat turn (CLAUDE.md s14.7): intent parse, anchor resolution, session update, the same
candidate and ranking path as ``recommend``, then the reply text.

Intent comes from the LLM (``intent_parse`` prompt, validated as ``IntentParse``) when the
client is available, else from ``rule_parse`` (regexes for "like <title>", "by <author>", "more by
that author", "why ...", "something else", and the modifier words of the prompt's field guide).
Anchors named by either parser must appear in the message and resolve through ``entities``;
ambiguity returns a clarification with up to three options. The reply text comes from the LLM
(``chat_reply`` prompt) only when its book ids equal the results in order and every reason
passes the validators; otherwise the template intro and template reasons answer. With the LLM
off, every turn reports ``method="template"``.
"""

import contextlib
import importlib
import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from litsense_core import entities, validators
from litsense_core.llm import PrivacyError, get_client
from litsense_core.profile import Profile, build_profile
from litsense_core.reasons import build_evidence, template_reason
from litsense_core.schemas import (
    ASPECT_LABELS,
    ChatItem,
    ChatMessage,
    ChatReply,
    Chip,
    Clarification,
    ClarificationOption,
    Evidence,
    Filters,
    Intent,
    SessionState,
    UserState,
)
from litsense_core.store import ArtifactStore

CHAT_K = 5
HISTORY = 6
DEFAULT_SHORT_PAGES = 300
RECENT_YEARS = 5
CLASSIC_MAX_YEAR = 1980
DELTA_WEIGHT = 0.5  # aspect_vec += 0.5 * aspect_deltas (s14.7 step 6)
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
# "like X", "similar to X", "in the vein of X", "close to X"; X ends at a comma or "but".
_LIKE_RE = re.compile(
    r"\b(?:like|similar to|in the vein of|close to|in the spirit of|in the style of)\s+"
    r"(.+?)(?:\s*(?:,|\bbut\b|\bthough\b).*)?$",
    re.IGNORECASE,
)
_VIBES_RE = re.compile(r"^(?:give me |i want |something with )?(.+?)\s+vibes\b", re.IGNORECASE)
_LOVED_NEXT_RE = re.compile(
    r"\b(?:loved|liked|enjoyed|finished)\s+(.+?)\s*[,.!?;]?\s*(?:what(?:'s)? next|what now|"
    r"more like (?:it|that)|anything (?:else )?like (?:it|that))",
    re.IGNORECASE,
)
_BY_RE = re.compile(r"\b(?:by|from)\s+(.+?)(?:\s*(?:,|\bbut\b).*)?$", re.IGNORECASE)
_MORE_RE = re.compile(r"^\s*more\s+(?:by\s+|from\s+)?(.+?)\s*[.!?]?$", re.IGNORECASE)
_WHAT_ELSE_RE = re.compile(
    r"\bwhat else (?:did|has|have)\s+(.+?)\s+(?:write|written|wrote|publish|published)",
    re.IGNORECASE,
)
_SESSION_AUTHOR_RE = re.compile(
    r"\b(?:more (?:by|from)|other books? (?:by|from)|books? (?:by|from)|author of|"
    r"the author of)\s+(?:that|this|the same)\s+(?:author|one|book)\b"
    r"|\b(?:the )?author of the (first|second|third|fourth|fifth|last) one\b"
    r"|\bother books? (?:by|from) the same author\b"
)
_EXPLAIN_RE = re.compile(r"\bwhy\b|\bexplain\b|\bwhat makes\b.*\b(?:fit|match|good)\b|\breason\b")
_BOOKISH = re.compile(
    r"\b(?:book|books|read|reads|novel|novels|recommend|recommendation|story|stories|something|"
    r"anything|author|surprise me|doorstopper|classic|classics|thrillers?|mysteries|romances?)\b"
)
_DESCRIPTOR_KEYWORDS = (  # keyword regex, descriptor, value; later matches win ("less dark")
    (r"\bdark(?:er)?\b|\bgrim(?:dark)?\b|\bbleak\b", "darkness", 0.6),
    (r"\bcoz(?:y|ier)|\bcosy\b|\bgentle\b|\blighter\b|\bless (?:bleak|dark)", "darkness", -0.6),
    (r"\bfast(?:er|-paced)?\b|\bgripping\b|\bpage[- ]turner", "pace", 0.6),
    (r"\bslow(?:er)?\b|\bslow[- ]burn", "pace", -0.5),
    (r"\beas(?:y|ier)\b|\blight read\b|\bless complicated\b|\bsimpler\b|\bless confusing\b",
     "complexity_level", -0.5),
    (r"\bdense\b|\bcomplex\b|\bmore literary\b", "complexity_level", 0.3),
)  # fmt: skip
_ASPECT_KEYWORDS = ((r"\bfunn(?:y|ier)\b|\bmore humou?r\b", "humor", 0.5),
                    (r"\bmore literary\b", "writing_style", 0.5))  # fmt: skip
# Genre words readers type -> the catalogue label they mean (labels themselves also match).
_GENRE_SYNONYMS = {
    "sci-fi": "Science Fiction", "scifi": "Science Fiction", "science fiction": "Science Fiction",
    "fantasy": "Fantasy", "mystery": "Mystery", "mysteries": "Mystery", "thriller": "Thriller",
    "thrillers": "Thriller", "horror": "Horror", "romance": "Romance", "romances": "Romance",
    "historical": "Historical Fiction", "historical fiction": "Historical Fiction",
    "literary": "Literary Fiction", "literary fiction": "Literary Fiction",
    "young adult": "Young Adult", "ya": "Young Adult",
}  # fmt: skip
_PAGES_MAX_RE = re.compile(
    r"\b(?:under|below|less than|fewer than|no more than|nothing (?:longer|more) than|max(?:imum)?|"
    r"at most|up to)\s+(\d{2,4})\s*pages?\b"
)
_PAGES_MIN_RE = re.compile(
    r"\b(?:over|above|more than|at least|longer than|min(?:imum)?)\s+(\d{2,4})\s*pages?\b"
)
_FILTER_NOISE = re.compile(
    r"\b(?:only|just|no|not|nothing|exclude|excluding|without|please|published|books?|or|and|"
    r"between|from|to|under|over|than|pages?|before|after|since|older|newer|longer|shorter|"
    r"more|less|than|the|a|an|of|me|show|give|i|want|year|years|range)\b|\d+"
)
# Mixed-opinion vocabulary: what a reader praises or faults in a named book (s14.7 step 4).
_MIXED_ASPECTS = {
    "characters": r"\bcharacters?\b|\bprotagonist\b",
    "plot": r"\bplot\b|\bstory(?:line)?\b|\btwists?\b",
    "humor": r"\bhumou?r\b|\bfunny\b|\bjokes\b", "ending": r"\bending\b",
    "writing_style": r"\bprose\b|\bwriting\b|\bwritten\b", "dialogue": r"\bdialog(?:ue)?\b",
    "world_building": r"\bworld[- ]?building\b|\bsetting\b", "pacing": r"\bpacing\b|\bpace\b",
    "emotional_impact": r"\bemotional\b|\bmoving\b", "complexity": r"\bdepth\b",
}  # fmt: skip
_MIXED_DESCRIPTORS = (  # a fault with one of these -> what the reader wants instead
    (r"\bslow\b|\bdragged\b|\bdrags\b|\bboring\b", "pace", 0.4),
    (r"\bdense\b|\bcomplicated\b|\bconfusing\b|\bhard to follow\b", "complexity_level", -0.4),
    (r"\btoo dark\b|\bbleak\b|\bdepressing\b|\bgrim\b", "darkness", -0.4),
)
_POSITIVE = re.compile(
    r"\b(?:loved|love|liked|like|enjoyed|great|amazing|worked|clever|wonderful|brilliant|good|"
    r"fantastic|excellent|strong|beautiful)\b"
)
_NEGATIVE = re.compile(
    r"\b(?:not|didn't|did not|wasn't|was not|too|flat|slow|dragged|boring|dense|weak|poor|bad|"
    r"disappointing|confusing|complicated|hated|disliked)\b"
)
_MIXED_CUE = re.compile(r"\bbut\b|\bthough\b|\bhowever\b|\bnot\b|\bdid not\b|\bdidn't\b|\bflat\b")
INTROS = {
    "similar_to": "Here are books in the spirit of {anchor}.",
    "similar_to_with_modifier": "Here are books in the spirit of {anchor}, adjusted as asked.",
    "more_by_author": "More from {author}.",
    "refine_previous": "Here's a refined set.",
    "recommend_general": "Here are some picks for that.",
    "mixed_opinion": "Taking that into account, here are some picks.",
    "filter_by_attribute": "Here are picks within those limits.",
}
OUT_OF_SCOPE = "I only help with finding books. Try: like <a title>."
FOLLOWUP = "Want them shorter, cozier or more recent?"
FOLLOWUP_EMPTY = "Try removing one of the active filters."


class IntentParse(BaseModel):
    """The ``intent_parse`` prompt's JSON (s14.3); also what ``rule_parse`` returns."""

    intent: Intent
    anchor_titles: list[str] = Field(default_factory=list)
    anchor_authors: list[str] = Field(default_factory=list)
    target_index: int | None = None
    anchor_from_session: bool = False
    aspect_deltas: dict[str, float] = Field(default_factory=dict)
    descriptor_deltas: dict[str, float] = Field(default_factory=dict)
    relative_length: Literal["shorter", "longer"] | None = None
    filters: Filters = Field(default_factory=Filters)
    exclude_previous: bool = False
    free_text_query: str | None = None
    needs_clarification: str | None = None


class ChatReplyItemLLM(BaseModel):
    book_id: int
    reason: str


class ChatReplyLLM(BaseModel):
    intro: str
    items: list[ChatReplyItemLLM] = Field(default_factory=list)
    followup: str | None = None


# ---------------------------------------------------------------------- intent parsing


def modifiers(low: str) -> IntentParse:
    """Modifier words of the field guide applied to a blank ``IntentParse``."""
    out = IntentParse(intent="refine_previous")
    for pattern, name, value in _DESCRIPTOR_KEYWORDS:
        if re.search(pattern, low):
            out.descriptor_deltas[name] = value
    for pattern, name, value in _ASPECT_KEYWORDS:
        if re.search(pattern, low):
            out.aspect_deltas[name] = value
    if re.search(r"\bshorter\b|\bshort\b|\bquick read\b|\bunder \d+ pages\b", low):
        out.relative_length = "shorter"
    elif re.search(r"\blonger\b|\bdoorstopper\b", low):
        out.relative_length = "longer"
    if re.search(r"\brecent\b|\bnewer\b|\bnew releases?\b", low):
        out.filters.min_year = datetime.now(UTC).year - RECENT_YEARS
    if re.search(r"\bclassics?\b", low):
        out.filters.max_year = CLASSIC_MAX_YEAR
    if m := re.search(r"\bbetween ((?:19|20)\d\d) and ((?:19|20)\d\d)\b", low):
        out.filters.min_year, out.filters.max_year = int(m.group(1)), int(m.group(2))
    if m := re.search(
        r"\b(?:nothing|not) older than ((?:19|20)\d\d)|\b(?:after|since|from) ((?:19|20)\d\d)"
        r"|\b((?:19|20)\d\d) (?:or later|onwards?|and later)",
        low,
    ):
        out.filters.min_year = int(m.group(1) or m.group(2) or m.group(3))
    if m := re.search(r"(?<!nothing )(?<!not )\b(?:older than|before|up to) ((?:19|20)\d\d)", low):
        out.filters.max_year = int(m.group(1))
    if m := _PAGES_MAX_RE.search(low):
        out.filters.max_pages = int(m.group(1))
        out.relative_length = None  # an exact limit, not "shorter"
    if m := _PAGES_MIN_RE.search(low):
        out.filters.min_pages = int(m.group(1))
        out.relative_length = None
    return out


def genre_filters(low: str, store: ArtifactStore) -> tuple[list[str], list[str]]:
    """``(include, exclude)`` genre labels named in the message: "no horror", "exclude sci-fi",
    "mystery or thriller only", or a bare genre word (include)."""
    labels = {g.lower(): g for g in store.catalog.genre_labels if g}
    words = {**_GENRE_SYNONYMS, **{k: v for k, v in labels.items()}}
    found: list[tuple[int, bool, str]] = []  # (position, negated, label), message order
    for word in sorted(words, key=len, reverse=True):
        for m in re.finditer(rf"\b{re.escape(word)}\b", low):
            label = words[word]
            if label.lower() not in labels and label not in labels.values():
                continue  # a synonym for a genre this catalogue does not have
            before = low[max(0, m.start() - 20) : m.start()]
            negated = re.search(r"\b(?:no|not|exclude|excluding|without|nothing|avoid|skip)\s*$",
                                before) is not None  # fmt: skip
            found.append((m.start(), negated, label))
            low = low[: m.start()] + " " * (m.end() - m.start()) + low[m.end() :]
            break
    include = list(dict.fromkeys(label for _, neg, label in sorted(found) if not neg))
    exclude = list(dict.fromkeys(label for _, neg, label in sorted(found) if neg))
    return include, [g for g in exclude if g not in include]


def filter_only(low: str) -> bool:
    """True when nothing but filter phrases remains after the filter words are removed."""
    rest = _FILTER_NOISE.sub(" ", low)
    for word in list(_GENRE_SYNONYMS) + ["science fiction", "historical fiction",
                                          "literary fiction", "young adult"]:  # fmt: skip
        rest = re.sub(rf"\b{re.escape(word)}\b", " ", rest)
    return len(re.findall(r"[a-z]{3,}", rest)) == 0


def find_title(text: str, store: ArtifactStore) -> str | None:
    """A catalogue title written inside a longer message: the longest word n-gram that is an
    exact normalized title (single words only when capitalised and four or more letters)."""
    found = validators.title_matcher(store).mentioned(text)
    if found:
        return found[0]
    index = entities.index_for(store)
    words = re.findall(r"[A-Za-z0-9'’:-]+", text)
    for n in range(5, 0, -1):
        for start in range(0, len(words) - n + 1):
            span = words[start : start + n]
            if n == 1 and (len(span[0]) < 4 or not span[0][0].isupper()):
                continue
            key = entities.norm(" ".join(span))
            if key in index.title_ids:
                return index.display_title(key)
    return None


def mixed_opinion_deltas(low: str) -> tuple[dict[str, float], dict[str, float]]:
    """Aspect and descriptor deltas from the praised and faulted parts of a message."""
    aspects: dict[str, float] = {}
    descriptors: dict[str, float] = {}
    for clause in re.split(r"\bbut\b|\bthough\b|\bhowever\b|[,;:]", low):
        positive = _POSITIVE.search(clause) is not None and _NEGATIVE.search(clause) is None
        for pattern, name, value in _MIXED_DESCRIPTORS:
            if not positive and re.search(pattern, clause):
                descriptors[name] = value
        for name, pattern in _MIXED_ASPECTS.items():
            if re.search(pattern, clause):
                if name == "pacing" and not positive and "pace" in descriptors:
                    continue  # "the pacing was slow" is the pace descriptor, not the aspect
                aspects[name] = 0.5 if positive else -0.5
    return aspects, descriptors


def explain_index(low: str, session: SessionState) -> int | None:
    if re.search(r"\blast one\b|\bthe last\b", low) and session.last_shown:
        return len(session.last_shown)
    index = next((v for w, v in ORDINALS.items() if re.search(rf"\b{w}\b", low)), None)
    if index is None and (m := re.search(r"\b(?:#|number |no\.? ?)?([1-9])(?:st|nd|rd|th)?\b",
                                         low)):  # fmt: skip
        index = int(m.group(1))
    return index


def has_modifiers(parsed: IntentParse) -> bool:
    f = parsed.filters
    return bool(
        parsed.aspect_deltas
        or parsed.descriptor_deltas
        or parsed.relative_length
        or any(v is not None for v in (f.min_year, f.max_year, f.min_pages, f.max_pages))
        or f.genres_include
        or f.genres_exclude
    )


def rule_parse(message: str, session: SessionState, store: ArtifactStore) -> IntentParse:
    """The non-LLM parser (s14.7). Never calls the network."""
    text = message.strip()
    low = text.lower()
    if not text:
        return IntentParse(intent="refine_previous")
    if _EXPLAIN_RE.search(low) and not _LIKE_RE.search(text):
        return IntentParse(
            intent="explain_recommendation", target_index=explain_index(low, session)
        )
    parsed = modifiers(low)
    include, exclude = genre_filters(low, store)
    parsed.filters.genres_include, parsed.filters.genres_exclude = include, exclude
    if re.search(r"\b(?:something else|not these|other ones?|different ones?)\b", low):
        parsed.intent, parsed.exclude_previous = "refine_previous", True
        return parsed
    if m := _SESSION_AUTHOR_RE.search(low):
        parsed.intent, parsed.anchor_from_session = "more_by_author", True
        if m.group(1):
            parsed.target_index = explain_index(m.group(1), session)
        return parsed
    if m := _WHAT_ELSE_RE.search(text):
        parsed.anchor_authors, parsed.intent = [m.group(1).strip()], "more_by_author"
        return parsed
    if m := _LIKE_RE.search(text) or _VIBES_RE.search(text) or _LOVED_NEXT_RE.search(text):
        parsed.anchor_titles = [m.group(1).strip()]
        parsed.intent = "similar_to_with_modifier" if has_modifiers(parsed) else "similar_to"
        return parsed
    title = find_title(text, store)
    if title and _MIXED_CUE.search(low) and (_POSITIVE.search(low) or _NEGATIVE.search(low)):
        parsed = IntentParse(intent="mixed_opinion", anchor_titles=[title])
        parsed.aspect_deltas, parsed.descriptor_deltas = mixed_opinion_deltas(low)
        return parsed
    if title and re.search(r"\bbut\b|\bthough\b", low):
        parsed.anchor_titles = [title]  # "The Martian but funnier"
        parsed.intent = "similar_to_with_modifier" if has_modifiers(parsed) else "similar_to"
        return parsed
    if m := _BY_RE.search(text):
        parsed.anchor_authors = [m.group(1).strip()]
        parsed.intent = "more_by_author"
        return parsed
    if (m := _MORE_RE.search(text)) and entities.resolve_author(m.group(1), store).kind == "author":
        parsed.anchor_authors, parsed.intent = [m.group(1).strip()], "more_by_author"
        return parsed
    match = entities.resolve_entities(text, store)  # a bare title or author (a picked option)
    if match.kind == "title":
        parsed.anchor_titles, parsed.intent = [text], "similar_to"
        return parsed
    if match.kind == "author":
        parsed.anchor_authors, parsed.intent = [text], "more_by_author"
        return parsed
    if (
        has_modifiers(parsed)
        and filter_only(low)
        and not (parsed.aspect_deltas or parsed.descriptor_deltas or parsed.relative_length)
    ):
        parsed.intent = "filter_by_attribute"
        return parsed
    if _BOOKISH.search(low) or (include and not filter_only(low)):
        parsed.intent, parsed.free_text_query = "recommend_general", text
        return parsed
    if has_modifiers(parsed):
        parsed.intent = "refine_previous" if session.turn else "recommend_general"
        return parsed
    return IntentParse(intent="out_of_scope")


def _session_summary(session: SessionState, store: ArtifactStore) -> dict[str, Any]:
    cat = store.catalog
    return {
        "turn": session.turn,
        "anchor_titles": [cat.title[i] for i in session.anchor_ids],
        "anchor_authors": list(session.anchor_authors),
        "aspect_deltas": session.aspect_deltas,
        "descriptor_deltas": session.descriptor_deltas,
        "filters": session.filters.model_dump(exclude_none=True),
    }


def llm_parse(message: str, session: SessionState, store: ArtifactStore) -> IntentParse | None:
    """``intent_parse`` through the client; None on any failure (the rule parser answers)."""
    client = get_client()
    if not client.available or not message.strip():
        return None
    cat = store.catalog
    variables = {
        "aspects": ", ".join(store.aspect_names or list(ASPECT_LABELS)),
        "genres": ", ".join(g for g in cat.genre_labels if g),
        "recent_year": datetime.now(UTC).year - RECENT_YEARS,
        "session_summary": _session_summary(session, store),
        "last_shown": [
            {"index": k, "title": cat.title[i]} for k, i in enumerate(session.last_shown, 1)
        ],  # fmt: skip
        "message": message.strip()[:1000],
    }
    try:
        result = client.json("intent_parse", variables, IntentParse, "parse")
    except PrivacyError:
        return None
    if not result.ok or result.data is None:
        return None
    parsed: IntentParse = result.data  # type: ignore[assignment]
    norm_message = entities.norm(message)
    # Anchors must be copied from the message (s3.1: never a title the user did not name).
    parsed.anchor_titles = [t for t in parsed.anchor_titles if entities.norm(t) in norm_message]
    parsed.anchor_authors = [a for a in parsed.anchor_authors if entities.norm(a) in norm_message]
    aspects = set(store.aspect_names or ASPECT_LABELS)
    descriptors = set((store.config.get("aspects") or {}).get("descriptors")
                      or ("darkness", "pace", "complexity_level"))  # fmt: skip
    parsed.aspect_deltas = {k: float(np.clip(v, -1, 1)) for k, v in parsed.aspect_deltas.items()
                            if k in aspects}  # fmt: skip
    parsed.descriptor_deltas = {
        k: float(np.clip(v, -1, 1)) for k, v in parsed.descriptor_deltas.items() if k in descriptors
    }
    genres = {g for g in cat.genre_labels if g}
    parsed.filters.genres_include = [g for g in parsed.filters.genres_include if g in genres]
    parsed.filters.genres_exclude = [g for g in parsed.filters.genres_exclude if g in genres]
    if parsed.target_index is not None and not 1 <= parsed.target_index <= len(session.last_shown):
        parsed.target_index = None
    anchored = parsed.intent in ("similar_to", "similar_to_with_modifier", "mixed_opinion")
    if anchored and not (parsed.anchor_titles or parsed.anchor_authors or parsed.target_index
                         or parsed.anchor_from_session):  # fmt: skip
        return None  # every anchor was invented: the rule parser decides instead
    return parsed


def parse_intent(
    message: str, session: SessionState, store: ArtifactStore, disabled_ai: bool = False
) -> tuple[IntentParse, str]:
    """``(intent, "llm" | "rule")``."""
    if not disabled_ai:
        parsed = llm_parse(message, session, store)
        if parsed is not None:
            return parsed, "llm"
    return rule_parse(message, session, store), "rule"


# ---------------------------------------------------------------------- session update


def merge_filters(base: Filters, extra: Filters) -> Filters:
    out = base.model_copy(deep=True)
    for name in ("min_year", "max_year", "min_pages", "max_pages"):
        if getattr(extra, name) is not None:
            setattr(out, name, getattr(extra, name))
    out.genres_include = list(dict.fromkeys([*out.genres_include, *extra.genres_include]))
    out.genres_exclude = list(dict.fromkeys([*out.genres_exclude, *extra.genres_exclude]))
    return out


def _clip_merge(base: dict[str, float], extra: dict[str, float]) -> dict[str, float]:
    out = dict(base)
    for k, v in extra.items():
        out[k] = float(np.clip(out.get(k, 0.0) + v, -1.0, 1.0))
    return out


def _length_filter(session: SessionState, relative: str | None, store: ArtifactStore) -> None:
    if relative is None:
        return
    pages = int(store.catalog.pages[session.anchor_ids[0]]) if session.anchor_ids else -1
    if relative == "shorter":
        session.filters.max_pages = int(0.8 * pages) if pages > 0 else DEFAULT_SHORT_PAGES
    elif pages > 0:
        session.filters.min_pages = int(1.2 * pages)


def _resolve(
    parsed: IntentParse, session: SessionState, store: ArtifactStore
) -> tuple[list[int], list[str], list[str], list[dict[str, Any]]]:
    """Anchor book ids, anchor author labels, ambiguity options and resolution logs."""
    ids: list[int] = []
    authors: list[str] = []
    options: list[str] = []
    logs: list[dict[str, Any]] = []
    for title in parsed.anchor_titles:
        res = entities.resolve_any(title, store, "title")
        logs.append(res.log())
        if res.kind == "title":
            ids.extend(res.book_ids[:1])
        elif res.kind == "author" and res.author:
            authors.append(res.author)
        elif res.kind == "ambiguous":
            options = res.options
    for name in parsed.anchor_authors:
        res = entities.resolve_any(name, store, "author")
        logs.append(res.log())
        if res.kind == "author" and res.author:
            authors.append(res.author)
        elif res.kind == "title":
            ids.extend(res.book_ids[:1])
        elif res.kind == "ambiguous":
            options = res.options
    if parsed.target_index is not None and 1 <= parsed.target_index <= len(session.last_shown):
        ids.append(session.last_shown[parsed.target_index - 1])
    if parsed.anchor_from_session and not ids and not authors:
        ids.extend(session.anchor_ids[:1] or session.last_shown[:1])
        if not ids:
            authors.extend(session.anchor_authors[:1])
    return list(dict.fromkeys(ids)), list(dict.fromkeys(authors)), options, logs


def update_session(
    session: SessionState, parsed: IntentParse, ids: list[int], authors: list[str],
    store: ArtifactStore,
) -> None:  # fmt: skip
    """s14.7 step 4."""
    cat = store.catalog
    intent = parsed.intent
    if intent in ("similar_to", "similar_to_with_modifier", "mixed_opinion"):
        session.anchor_ids, session.anchor_authors = ids[:1], []
        session.aspect_deltas, session.descriptor_deltas, session.filters = {}, {}, Filters()
        if intent == "mixed_opinion":
            session.aspect_deltas = dict(parsed.aspect_deltas)
            session.descriptor_deltas = dict(parsed.descriptor_deltas)
        else:
            session.aspect_deltas = _clip_merge({}, parsed.aspect_deltas)
            session.descriptor_deltas = _clip_merge({}, parsed.descriptor_deltas)
        session.filters = merge_filters(session.filters, parsed.filters)
        if not ids and authors:  # "like Andy Weir": an author after "like"
            session.anchor_authors = authors[:1]
    elif intent == "more_by_author":
        label = authors[:1]
        if not label and ids:
            author = cat.author[ids[0]]
            label = [author] if author else []
        session.anchor_ids, session.anchor_authors = [], label
        session.filters = merge_filters(session.filters, parsed.filters)
    elif intent == "refine_previous":
        session.aspect_deltas = _clip_merge(session.aspect_deltas, parsed.aspect_deltas)
        session.descriptor_deltas = _clip_merge(session.descriptor_deltas, parsed.descriptor_deltas)
        session.filters = merge_filters(session.filters, parsed.filters)
    elif intent == "filter_by_attribute":
        session.filters = merge_filters(session.filters, parsed.filters)
    elif intent == "recommend_general":
        session.anchor_ids, session.anchor_authors = [], []
        session.aspect_deltas = _clip_merge(session.aspect_deltas, parsed.aspect_deltas)
        session.descriptor_deltas = _clip_merge(session.descriptor_deltas, parsed.descriptor_deltas)
        session.filters = merge_filters(session.filters, parsed.filters)
    if parsed.exclude_previous:
        session.excluded_ids = list(dict.fromkeys([*session.excluded_ids, *session.last_shown]))
    _length_filter(session, parsed.relative_length, store)


# ---------------------------------------------------------------------- results


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)


def apply_modifiers(profile: Profile, session: SessionState, store: ArtifactStore,
                    query: str | None) -> Profile:  # fmt: skip
    """s14.7 step 6: anchors into the item_item source, deltas into the aspect and descriptor
    vectors, the free-text query into the semantic vector."""
    if session.anchor_ids:
        profile.anchors = list(session.anchor_ids)
        anchor = store.content_rows(np.asarray(session.anchor_ids[:1], dtype=np.int64))[0]
        profile.semantic_vec = _unit(
            anchor if profile.semantic_vec is None else 0.5 * profile.semantic_vec + 0.5 * anchor
        )
    if query and store.encoder is not None:
        q = store.encoder.encode([query])[0]
        profile.semantic_vec = _unit(
            q if profile.semantic_vec is None else 0.5 * profile.semantic_vec + 0.5 * q
        )
    if session.aspect_deltas and store.aspect_names:
        vec = (
            np.zeros(store.n_aspects, np.float32)
            if profile.aspect_vec is None
            else profile.aspect_vec.copy()
        )
        for name, delta in session.aspect_deltas.items():
            a = store.aspect_index(name)
            if a is not None:
                vec[a] += DELTA_WEIGHT * float(delta)
        profile.aspect_vec = np.clip(vec, -1.0, 1.0).astype(np.float32)
        if profile.aspect_conf is None:
            profile.aspect_conf = np.zeros(store.n_aspects, np.float32)
    if session.descriptor_deltas and store.item_descriptors is not None:
        names = list((store.config.get("aspects") or {}).get("descriptors") or [])
        pref = (
            np.zeros(store.item_descriptors.shape[1], np.float32)
            if profile.descriptor_pref is None
            else profile.descriptor_pref.copy()
        )
        for k, name in enumerate(names[: len(pref)]):
            if name in session.descriptor_deltas:
                pref[k] += float(session.descriptor_deltas[name])
        profile.descriptor_pref = np.clip(pref, -1.0, 1.0).astype(np.float32)
    return profile


def results(profile: Profile, session: SessionState, store: ArtifactStore,
            query: str | None) -> list[int]:  # fmt: skip
    """Book ids for this turn: the author's books for an author anchor, else the shared
    candidates -> filters -> ranker/blend -> post-processing path."""
    from litsense_core import candidates as cand
    from litsense_core.recommend import _ranked

    cat = store.catalog
    blocked = profile.seen | set(session.excluded_ids) | set(session.anchor_ids)
    if session.anchor_authors:
        res = entities.resolve_author(session.anchor_authors[0], store)
        if res.kind != "author":
            return []
        pool = [i for i in res.book_ids if i not in blocked
                and cand.passes_filters(store, i, session.filters)
                and not set(cat.genres_of(i)) & profile.muted_genres]  # fmt: skip
        pool.sort(key=lambda i: (
            float(cat.series_pos[i]) if not np.isnan(cat.series_pos[i]) else 99.0,
            -float(cat.popularity[i]), i,
        ))  # fmt: skip
        return pool[:CHAT_K]
    profile = apply_modifiers(profile, session, store, query)
    top, _, _, _ = _ranked(profile, store, session.filters, sorted(blocked - profile.seen), CHAT_K)
    return [i for i, _ in top]


# ---------------------------------------------------------------------- reply text


def chips(session: SessionState, store: ArtifactStore) -> list[Chip]:
    cat = store.catalog
    out = [Chip(key=f"anchor:book:{i}", label=f"like {cat.title[i]}") for i in session.anchor_ids]
    out += [Chip(key="anchor:author", label=f"by {a}") for a in session.anchor_authors]
    labels = {"darkness": ("cozier", "darker"), "pace": ("slower", "faster"),
              "complexity_level": ("easier", "denser")}  # fmt: skip
    out += [
        Chip(key=f"descriptor:{d}", label=labels.get(d, ("less " + d, "more " + d))[v > 0])
        for d, v in session.descriptor_deltas.items()
    ]
    out += [
        Chip(key=f"aspect:{a}", label=f"{'more' if v > 0 else 'less'} {ASPECT_LABELS.get(a, a)}")
        for a, v in session.aspect_deltas.items()
    ]
    f = session.filters
    for key, value, label in (
        ("max_pages", f.max_pages, "under {} pages"),
        ("min_pages", f.min_pages, "over {} pages"),
        ("min_year", f.min_year, "from {}"),
        ("max_year", f.max_year, "before {}"),
    ):
        if value is not None:
            out.append(Chip(key=f"filter:{key}", label=label.format(value)))
    out += [Chip(key=f"filter:genres_include:{g}", label=f"only {g}") for g in f.genres_include]
    out += [Chip(key=f"filter:genres_exclude:{g}", label=f"no {g}") for g in f.genres_exclude]
    return out


def llm_reply(
    request: str, session: SessionState, evidences: list[Evidence], store: ArtifactStore
) -> ChatReplyLLM | None:
    """The validated ``chat_reply`` JSON, or None (template reply)."""
    client = get_client()
    if not client.available:
        return None
    variables = {
        "request": request[:500] or (session.last_request or "")[:500],
        "active_filters": [c.label for c in chips(session, store)],
        "results_json": [
            {"book_id": e.book["book_id"], "evidence": e.model_dump()} for e in evidences
        ],  # fmt: skip
    }
    try:
        result = client.json("chat_reply", variables, ChatReplyLLM, "chat")
    except PrivacyError:
        return None
    if not result.ok or result.data is None:
        return None
    reply: ChatReplyLLM = result.data  # type: ignore[assignment]
    if [it.book_id for it in reply.items] != [e.book["book_id"] for e in evidences]:
        return None
    limit = client.config.chat_max_words
    if validators.check(reply.intro, [], None, limit, store):
        return None
    if reply.followup and validators.check(reply.followup, [], None, limit, store):
        return None
    for item, ev in zip(reply.items, evidences, strict=True):
        allowed = ev.allowed_titles() - {ev.book.get("title")}  # the app renders the title
        if validators.check(item.reason, allowed, ev, limit, store):
            return None
    return reply


def _finish(
    session: SessionState,
    text: str,
    intro: str,
    items: list[ChatItem],
    intent: Intent,
    store: ArtifactStore,
    method: Literal["template", "llm"] = "template",
    followup: str | None = None,
    clarification: Clarification | None = None,
) -> ChatReply:
    session.turn += 1
    if intent not in ("explain_recommendation", "out_of_scope") and clarification is None:
        session.last_shown = [it.book_id for it in items]
        session.last_request = text or session.last_request
    if text:
        session.history.append(ChatMessage(role="user", text=text[:300]))
    session.history.append(ChatMessage(role="assistant", text=intro[:300]))
    session.history = session.history[-HISTORY:]
    if followup is None and intent != "out_of_scope" and clarification is None:
        followup = FOLLOWUP if items else FOLLOWUP_EMPTY
    return ChatReply(
        intro=intro,
        items=items,
        followup=followup,
        chips=chips(session, store),
        clarification=clarification,
        session=session,
        method=method,
        intent=intent,
    )


def chat_turn(
    state: UserState, session: SessionState | None, message: str, store: ArtifactStore
) -> ChatReply:
    cat = store.catalog
    profile = build_profile(state, store)
    session = (
        session.model_copy(deep=True) if session else SessionState(session_id=uuid.uuid4().hex)
    )
    text = message.strip()
    disabled_ai = state.overrides.disabled_ai
    parsed, _parser = parse_intent(text, session, store, disabled_ai)
    intent = parsed.intent
    if intent == "out_of_scope":
        return _finish(session, text, OUT_OF_SCOPE, [], intent, store)
    if intent == "explain_recommendation":
        return _explain_turn(state, session, text, parsed, store)
    ids, authors, options, _logs = _resolve(parsed, session, store)
    if options:
        return _finish(
            session, text, "I found more than one match.", [], intent, store,
            clarification=Clarification(
                question="Which one did you mean?",
                options=[ClarificationOption(label=o, value=o) for o in options[:3]],
            ),
        )  # fmt: skip
    if (
        parsed.needs_clarification
        and not ids
        and not authors
        and intent in ("similar_to", "similar_to_with_modifier", "more_by_author")
    ):
        return _finish(session, text, parsed.needs_clarification, [], intent, store)
    if intent in ("similar_to", "similar_to_with_modifier") and not ids and not authors:
        return _finish(session, text, "I could not find that book in the catalogue.", [], intent,
                       store)  # fmt: skip
    if intent == "more_by_author" and not ids and not authors and not session.anchor_authors:
        return _finish(session, text, "Which book's author do you mean?", [], intent, store)
    update_session(session, parsed, ids, authors, store)
    picks = results(profile, session, store, parsed.free_text_query)
    intro = INTROS.get(intent, INTROS["recommend_general"]).format(
        anchor=cat.title[session.anchor_ids[0]] if session.anchor_ids else "that",
        author=session.anchor_authors[0] if session.anchor_authors else "that author",
    )
    if not picks:
        intro = "Nothing in the catalogue matched those constraints."
    evidences = [build_evidence(i, profile, store, ["chat"]) for i in picks]
    reply = None if disabled_ai else llm_reply(text, session, evidences, store)
    if reply is not None:
        items = [ChatItem(book_id=i, reason=it.reason, evidence=ev)
                 for i, it, ev in zip(picks, reply.items, evidences, strict=True)]  # fmt: skip
        return _finish(session, text, reply.intro, items, intent, store, "llm",
                       reply.followup or None)  # fmt: skip
    items = [ChatItem(book_id=i, reason=template_reason(i, profile, store, ev), evidence=ev)
             for i, ev in zip(picks, evidences, strict=True)]  # fmt: skip
    return _finish(session, text, intro, items, intent, store)


def _explain_turn(
    state: UserState, session: SessionState, text: str, parsed: IntentParse, store: ArtifactStore
) -> ChatReply:
    intent: Intent = "explain_recommendation"
    if not session.last_shown:
        intro = "I haven't suggested anything yet; ask me for some books first."
        return _finish(session, text, intro, [], intent, store)
    index = min(parsed.target_index or 1, len(session.last_shown))
    book_id = session.last_shown[index - 1]
    explanation = importlib.import_module("litsense_core.explain").explain(state, book_id, store)
    items = [ChatItem(book_id=book_id, reason=explanation.text, evidence=explanation.evidence)]
    return _finish(session, text, explanation.text, items, intent, store, explanation.method)


def debug_json(value: Any) -> str:
    """Compact JSON for logs (never raises)."""
    with contextlib.suppress(TypeError, ValueError):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)
