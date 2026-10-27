"""Week-1 stand-in for the litsense_core public API (CLAUDE.md s15.5).

Same signatures and schemas as the real package, over a deterministic synthetic catalogue of 200
books, so the backend and frontend can be built before any model exists. Nothing is learned: the
feed is popularity plus genre and author affinity, reasons are the s12.5 templates, chat is a few
regular expressions, review analysis is a keyword lexicon. Titles, authors and every number are
invented; never report anything this module returns as a result.

Usage: ``from litsense_core import mock as core`` now; switch to the real package when the
bundle ships (Week 5). Signatures do not change.
"""

import difflib
import math
import random
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from litsense_core import __api_version__
from litsense_core.schemas import (
    ASPECT_LABELS,
    AspectBar,
    BecauseYouLiked,
    BookInsights,
    Bucket,
    CatalogBook,
    ChatItem,
    ChatMessage,
    ChatReply,
    Chip,
    Clarification,
    ClarificationOption,
    EntityMatch,
    Evidence,
    Explanation,
    Filters,
    HomeRails,
    Intent,
    LikedItem,
    OnboardingProfile,
    OnboardingResult,
    RecItem,
    RecResponse,
    ReviewAnalysis,
    SearchHit,
    SessionState,
    Snippet,
    UserState,
)

MODEL_VERSION = "mock-0.1"
N_BOOKS = 200
CHAT_K = 5
# Lower bound of each interaction bucket; mirrors ranking.buckets in ml/config.yaml. A reader is
# in the last bucket whose lower bound is <= n, so fractional n (list adds count 0.5) is covered.
BUCKET_LOWER: tuple[tuple[Bucket, int], ...] = (
    ("cold", 0),
    ("warming", 1),
    ("blending", 5),
    ("warm", 20),
)


def _words(text: str) -> tuple[str, ...]:
    """Whitespace-separated word list (keeps long vocabularies compact)."""
    return tuple(text.split())


GENRES = (
    "Fantasy",
    "Science Fiction",
    "Mystery",
    "Thriller",
    "Romance",
    "Historical Fiction",
    "Horror",
    "Literary Fiction",
    "Young Adult",
    "Contemporary Fiction",
)

_SEED = 7
_SECOND_GENRE = dict(
    zip(
        GENRES,
        (
            "Young Adult",
            "Thriller",
            "Thriller",
            "Mystery",
            "Contemporary Fiction",
            "Romance",
            "Thriller",
            "Historical Fiction",
            "Fantasy",
            "Literary Fiction",
        ),
        strict=True,
    )
)
_FIRST_NAMES = _words(
    "Mara Elias Noor Theo Ines Callum Priya Jonah Freya Tomas Leila Arjun Hazel Ravi Selma Owen "
    "Yara Felix"
)
_LAST_NAMES = _words(
    "Ellison Varga Okafor Lindqvist Ashdown Moreau Castellan Iyer Brannigan Holloway Sato Reyes "
    "Whitlock Duarte Fenwick Quill"
)
_ADJECTIVES = _words(
    "Glass Hollow Silent Burning Last Hidden Salt Iron Paper Winter Crimson Drowned Golden Broken "
    "Midnight Wandering Quiet Distant Lantern Ashen"
)
_NOUNS = _words(
    "Cartographer Orchard Harbor Archive Garden Lighthouse Kingdom Library Station River Crown "
    "Signal Daughter Atlas Choir Tide Engine Masquerade Meridian Hour"
)
_TONES = ("sweeping", "quiet", "tense", "warm", "darkly funny", "slow-burning", "fast-paced")
_THEMES = (
    "a family secret that resurfaces after a funeral",
    "a crew stranded far from home",
    "two rivals forced to share a single inheritance",
    "a town that forgets one day a year",
    "an apprentice who inherits a dangerous map",
    "a detective returning to her hometown",
    "letters found inside a library book",
    "a friendship tested by a long winter",
)
_SERIES_NAMES = (
    "The Ember Saga",
    "The Tidewater Trilogy",
    "The Northern Lights Cycle",
    "The Clockwork Quartet",
    "The Hollow Crown Chronicles",
    "The Salt Road Books",
)

_STOPWORDS = frozenset(
    _words(
        "a an and about book books for i in like me novel novels of on or some something the to "
        "with want"
    )
)
_DESCRIPTOR_KEYWORDS = (  # keyword regex, descriptor, value; later matches win ("less dark")
    (r"\bdark(?:er)?\b|\bgrim(?:dark)?\b|\bbleak\b", "darkness", 0.6),
    (r"\bcoz(?:y|ier)|\bcosy\b|\bgentle\b|\blighter\b|\bless (?:bleak|dark)", "darkness", -0.6),
    (r"\bfast(?:er|-paced)?\b|\bgripping\b|\bpage[- ]turner", "pace", 0.6),
    (r"\bslow(?:er)?\b|\bslow[- ]burn", "pace", -0.5),
    (r"\beas(?:y|ier)\b|\blight read\b", "complexity_level", -0.5),
    (r"\bdense\b|\bcomplex\b|\bliterary\b", "complexity_level", 0.4),
)
_ASPECT_KEYWORDS = {
    "plot": ("plot", "story", "twist"),
    "writing_style": ("prose", "writing", "written", "sentences"),
    "characters": ("character", "characters", "protagonist", "found family"),
    "world_building": ("world", "worldbuilding", "world-building", "setting", "magic system"),
    "pacing": ("pacing", "pace", "dragged", "drags", "page-turner", "slow", "rushed"),
    "editing_quality": ("typo", "typos", "editing", "translation"),
    "dialogue": ("dialogue", "banter", "conversations"),
    "humor": ("funny", "humor", "humour", "witty", "hilarious"),
    "emotional_impact": ("cried", "moving", "emotional", "heartbreaking", "touching"),
    "ending": ("ending", "conclusion", "finale", "cliffhanger"),
    "complexity": ("confusing", "complex", "dense", "layered"),
    "tone_dark": ("dark", "bleak", "grim", "disturbing", "cozy", "cosy"),
}
_POSITIVE = frozenset(
    _words(
        "wonderful loved love great gorgeous brilliant beautiful beautifully gripping funny moving "
        "perfect fantastic excellent enjoyed satisfying"
    )
)
_NEGATIVE = frozenset(
    _words(
        "dragged boring tedious flat confusing hated weak rushed dull awful disappointing "
        "predictable annoying bad"
    )
)
_NEGATION = re.compile(r"\b(?:not|no|never|hate|avoid|dislike|nothing)\b")
_LIKE_RE = re.compile(r"\b(?:like|similar to)\s+(.+?)(?:\s+but\b.*)?$", re.IGNORECASE)
_BY_RE = re.compile(r"\b(?:by|more)\s+(.+?)(?:\s+but\b.*)?$", re.IGNORECASE)
_SPOILER = re.compile(
    r"\bdies\b|\bdied\b|\bkilled\b|turns out|the ending|twist (?:is|was)|"
    r"revealed to be|the killer|ends up",
    re.IGNORECASE,
)
_DESCRIPTOR_GENRES = {
    "darkness": ("Horror", "Thriller"),
    "pace": ("Thriller", "Mystery"),
    "complexity_level": ("Literary Fiction", "Historical Fiction"),
}
_ASPECT_GENRES = {"humor": ("Contemporary Fiction", "Romance"), "world_building": ("Fantasy",)}
_MODEL_ROWS = (
    ("M0", "Popularity"),
    ("M1", "ALS"),
    ("M2", "LightGCN, raw"),
    ("M2n", "LightGCN, normalized"),
    ("M2p", "Projection hybrid"),
    ("M2s", "Projection, shuffled text"),
    ("M3", "Content-only"),
    ("M4", "Ranker: CF + META + USER + SOURCE"),
    ("M5", "M4 + CONTENT + REVIEW_EMB"),
    ("M6", "M4 + ASPECT"),
    ("M7", "M4 + CONTENT + REVIEW_EMB + ASPECT"),
    ("M8", "Full LitSense"),
    ("M8a", "M8 - AUTHOR"),
    ("M8b", "M8 - ASPECT"),
)


@dataclass(frozen=True)
class _Book:
    book_id: int
    title: str
    author: str
    genres: tuple[str, ...]
    year: int
    pages: int
    description: str
    rating_count: int
    bayes_rating: float
    pop: float  # log-scaled popularity in [0, 1]
    longtail: bool
    mention: tuple[float, ...]  # per aspect, ASPECT_LABELS order
    pos: tuple[float, ...]
    neg: tuple[float, ...]
    series_id: str | None = None
    series_name: str | None = None
    series_pos: float | None = None


def _build_catalog() -> tuple[_Book, ...]:
    """Deterministic synthetic catalogue (``random.Random`` is stable across Python versions)."""
    rng = random.Random(_SEED)
    titles = [f"The {a} {n}" for a in _ADJECTIVES for n in _NOUNS]
    rng.shuffle(titles)
    names = [f"{f} {last}" for f in _FIRST_NAMES for last in _LAST_NAMES]
    rng.shuffle(names)
    authors_by_genre = {g: names[6 * i : 6 * i + 6] for i, g in enumerate(GENRES)}
    counts = sorted((int(4000 / (r + 1) ** 0.9) + 12 for r in range(N_BOOKS)), reverse=True)
    rng.shuffle(counts)
    tail_cut = sorted(counts)[int(0.8 * N_BOOKS)]  # bottom 80% by count is the long tail
    max_log = math.log1p(max(counts))
    drafts = []
    for book_id in range(N_BOOKS):
        genre = GENRES[book_id % len(GENRES)]
        author = rng.choices(authors_by_genre[genre], weights=(6, 4, 3, 2, 1, 1))[0]
        genres = (genre, _SECOND_GENRE[genre]) if rng.random() < 0.5 else (genre,)
        mention = tuple(round(rng.uniform(0.05, 0.6), 3) for _ in ASPECT_LABELS)
        pos = tuple(round(rng.uniform(0.35, 0.9), 3) for _ in ASPECT_LABELS)
        neg = tuple(round(rng.uniform(0.0, 1.0 - p), 3) for p in pos)
        mean = rng.uniform(3.4, 4.7)
        blurb = f"A {rng.choice(_TONES)} {genre.lower()} novel about {rng.choice(_THEMES)}."
        drafts.append(
            dict(
                book_id=book_id,
                title=titles[book_id],
                author=author,
                genres=genres,
                year=rng.randint(1965, 2023),
                pages=rng.randint(180, 720),
                description=blurb,
                rating_count=counts[book_id],
                bayes_rating=round((counts[book_id] * mean + 10 * 4.0) / (counts[book_id] + 10), 3),
                pop=round(math.log1p(counts[book_id]) / max_log, 4),
                longtail=counts[book_id] < tail_cut,
                mention=mention,
                pos=pos,
                neg=neg,
            )
        )
    by_author: dict[str, list[dict[str, Any]]] = {}
    for d in drafts:
        by_author.setdefault(d["author"], []).append(d)
    prolific = [a for a in sorted(by_author) if len(by_author[a]) >= 3]
    for s, author in enumerate(prolific[: len(_SERIES_NAMES)]):
        first_year = min(d["year"] for d in by_author[author][:3])
        for pos, d in enumerate(by_author[author][:3], start=1):
            d.update(
                series_id=f"mock_s{s}",
                series_name=_SERIES_NAMES[s],
                series_pos=float(pos),
                year=first_year + 2 * (pos - 1),
            )
    return tuple(_Book(**d) for d in drafts)


def _norm(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return re.sub(r"^(?:the|a|an) ", "", " ".join(text.split()))


BOOKS = _build_catalog()
_BY_TITLE: dict[str, list[int]] = {}
_BY_AUTHOR: dict[str, list[int]] = {}
for _b in BOOKS:
    _BY_TITLE.setdefault(_norm(_b.title), []).append(_b.book_id)
    _BY_AUTHOR.setdefault(_norm(_b.author), []).append(_b.book_id)


@dataclass
class _Profile:
    n: float
    bucket: Bucket
    seen: set[int]
    stars: dict[int, float]
    liked: list[int]  # rated >= 4, newest first
    genre_aff: dict[str, float]
    fav_authors: dict[str, int]  # author -> most recent liked book by them
    series_progress: dict[str, float]
    muted: set[str]
    onboarding_genres: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class MockStore:
    """What ``load_store`` returns in the mock; the real ArtifactStore replaces it."""

    version: str
    bundle_dir: str
    n_books: int


def load_store(bundle_dir: str) -> MockStore:
    """Mock: nothing to load; the catalogue is built at import."""
    return MockStore(version=MODEL_VERSION, bundle_dir=bundle_dir, n_books=N_BOOKS)


def download_bundle(url: str, dest: str) -> str:
    """Mock: downloads nothing; creates ``dest`` and returns it."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    return dest


def health() -> dict[str, Any]:
    """Component status in the shape the real ``health()`` will return."""
    return {
        "status": "ok",
        "mock": True,
        "api_version": __api_version__,
        "bundle_version": MODEL_VERSION,
        "data_version": "mock",
        "cutoff": None,
        "rss_mb": _rss_mb(),
        "ranker": False,
        "llm": False,
        "n_books": N_BOOKS,
    }


def catalog() -> list[CatalogBook]:
    """PROPOSED addition to s15.1: every catalogue book, for seeding the backend's books table."""
    return [
        CatalogBook(
            book_id=b.book_id,
            canonical_id=f"mock_{b.book_id:03d}",
            title=b.title,
            author=b.author,
            genres=list(b.genres),
            primary_genre=b.genres[0],
            year=b.year,
            pages=b.pages,
            description=b.description,
            image=None,
            isbn13=None,
            series_id=b.series_id,
            series_name=b.series_name,
            series_pos=b.series_pos,
        )
        for b in BOOKS
    ]


def recommend(
    state: UserState,
    k: int = 10,
    filters: Filters | None = None,
    exclude: list[int] | None = None,
    use_ranker: bool = True,
) -> RecResponse:
    """Top-k main feed. ``use_ranker`` is accepted for parity and ignored by the mock."""
    started = time.perf_counter()
    if k < 1:
        raise ValueError("k must be >= 1")
    prof = _profile(state)
    _check_ids(exclude or [])
    ranked = _rank(prof, filters, set(exclude or []))[:k]
    return RecResponse(
        items=[_rec_item(b, rank, prof) for rank, b in enumerate(ranked, 1)],
        continue_series=_continue_series(prof),
        model_version=MODEL_VERSION,
        profile_version=state.profile_version,
        bucket=prof.bucket,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


def home_rails(state: UserState) -> HomeRails:
    """All home rails in one call (PRD s7.3)."""
    prof = _profile(state)
    anchors = prof.liked[:2] or state.onboarding_picks[:2]
    trending = [b for b in sorted(BOOKS, key=lambda b: (-b.pop, b.book_id)) if _allowed(b, prof)]
    favorites = {a for a in prof.fav_authors if len(_BY_AUTHOR[_norm(a)]) >= 2}
    by_fav = [b for b in _rank(prof, None, set()) if b.author in favorites]
    viewed = [_similar(v, prof, set(state.viewed)) for v in state.viewed[:3]]
    wanted = [e.book_id for e in state.reading_list if e.status == "want_to_read"]
    from_list = [_similar(w, prof, set(wanted)) for w in wanted[:3]]
    gems = [b for b in _rank(prof, None, set()) if b.longtail] if prof.n >= 5 else []
    return HomeRails(
        top_picks=recommend(state, k=10).items,
        because_you_liked=[
            BecauseYouLiked(anchor_book_id=a, items=similar_books(a, 10, state)) for a in anchors
        ],
        trending=[
            _rec_item(b, r, prof, "Popular with readers right now", ["popularity"])
            for r, b in enumerate(trending[:20], 1)
        ],
        more_by_authors=[_rec_item(b, r, prof) for r, b in enumerate(by_fav[:10], 1)],
        continue_exploring=_items_from(_interleave(viewed)[:10], prof, "item_item"),
        hidden_gems=[
            _rec_item(b, r, prof, "A lesser-known pick that fits your taste", ["longtail"])
            for r, b in enumerate(gems[:10], 1)
        ],
        from_reading_list=_items_from(_interleave(from_list)[:10], prof, "item_item"),
        continue_series=_continue_series(prof),
    )


def similar_books(book_id: int, k: int = 10, state: UserState | None = None) -> list[RecItem]:
    """Books like ``book_id`` (shared genres, same author, close year); hides seen books if a
    state is given."""
    anchor = _book(book_id)
    prof = _profile(state or UserState())
    stars = prof.stars.get(book_id)
    reason = f"Similar to {anchor.title}" + (f", which you rated {stars:g}★" if stars else "")
    picks = _similar(book_id, prof, set())[:k]
    return [_rec_item(b, r, prof, reason, ["item_item"]) for r, b in enumerate(picks, 1)]


def more_by_author(book_id: int, k: int = 10, state: UserState | None = None) -> list[RecItem]:
    """Other books by the author of ``book_id``, series order first, then popularity."""
    anchor = _book(book_id)
    prof = _profile(state or UserState())
    same = [
        BOOKS[i]
        for i in _BY_AUTHOR[_norm(anchor.author)]
        if i != book_id and i not in prof.seen and BOOKS[i].genres[0] not in prof.muted
    ]
    same.sort(key=lambda b: (b.series_pos or 99, -b.pop, b.book_id))
    reason = f"More by {anchor.author}"
    return [_rec_item(b, r, prof, reason, ["author"]) for r, b in enumerate(same[:k], 1)]


def semantic_search(query: str, k: int = 20, filters: Filters | None = None) -> list[SearchHit]:
    """Keyword overlap stand-in for MiniLM search: title and author hits outrank description."""
    words = _content_words(query)
    if not words:
        return []
    hits = []
    for b in BOOKS:
        if not _passes(b, filters):
            continue
        t = len(words & _content_words(b.title))
        a = len(words & _content_words(b.author))
        d = len(words & _content_words(" ".join(b.genres) + " " + b.description))
        if t or a or d:
            score = min(1.0, (3 * t + 3 * a + d) / (3 * len(words)))
            matched = "title" if t else "author" if a else "semantic"
            hits.append(SearchHit(book_id=b.book_id, score=round(score, 4), matched=matched))
    hits.sort(key=lambda h: (-h.score, -BOOKS[h.book_id].pop, h.book_id))
    return hits[:k]


def resolve_entities(query: str) -> EntityMatch:
    """Resolve a title or author: exact normalized match, else a clear fuzzy winner."""
    q = _norm(query)
    if q in _BY_TITLE:
        return EntityMatch(kind="title", book_ids=_BY_TITLE[q])
    if q in _BY_AUTHOR:
        ids = _BY_AUTHOR[q]
        return EntityMatch(kind="author", book_ids=ids, author=BOOKS[ids[0]].author)
    scored = sorted(
        (
            (difflib.SequenceMatcher(None, q, key).ratio(), kind, key)
            for kind, index in (("title", _BY_TITLE), ("author", _BY_AUTHOR))
            for key in index
        ),
        reverse=True,
    )
    close = [s for s in scored if s[0] >= 0.8]
    if not close:
        return EntityMatch(kind="none")
    if len(close) > 1 and close[0][0] - close[1][0] < 0.05:
        return EntityMatch(
            kind="ambiguous", options=[_display(kind, key) for _, kind, key in close[:3]]
        )
    _, kind, key = close[0]
    return resolve_entities(_display(kind, key))


def onboarding_grid(n: int = 60) -> list[int]:
    """Popular, well-rated books, round-robin across primary genres."""
    per_genre = [
        sorted(
            (b for b in BOOKS if b.genres[0] == g and b.rating_count >= 50),
            key=lambda b: (-b.bayes_rating, b.book_id),
        )
        for g in GENRES
    ]
    return [b.book_id for b in _interleave(per_genre)[:n]]


def parse_onboarding(text: str, picks: list[int]) -> OnboardingResult:
    """Keyword parse of a taste sentence; the real core asks the LLM and falls back to keywords."""
    _check_ids(picks)
    profile = OnboardingProfile()
    for clause in re.split(r"[.;!?]|,|\bbut\b", text.lower()):
        negated = bool(_NEGATION.search(clause))
        for g in GENRES:
            if g.lower() in clause:
                (profile.disliked_genres if negated else profile.liked_genres).append(g)
        for pattern, name, value in _DESCRIPTOR_KEYWORDS:
            if re.search(pattern, clause):
                profile.descriptors[name] = -value if negated else value
        for aspect, words in _ASPECT_KEYWORDS.items():
            if aspect not in ("pacing", "tone_dark") and any(_has_word(clause, w) for w in words):
                target = profile.negative_aspects if negated else profile.positive_aspects
                if aspect not in target:
                    target.append(aspect)
    low = _norm(text)
    profile.named_titles = [BOOKS[ids[0]].title for t, ids in _BY_TITLE.items() if t in low]
    profile.named_authors = [BOOKS[ids[0]].author for a, ids in _BY_AUTHOR.items() if a in low]
    resolved = list(
        dict.fromkeys([*picks, *(_BY_TITLE[_norm(t)][0] for t in profile.named_titles)])
    )
    chips = [f"Likes: {g}" for g in profile.liked_genres]
    chips += [f"Avoids: {g}" for g in profile.disliked_genres]
    chips += [f"Values: {ASPECT_LABELS[a]}" for a in profile.positive_aspects]
    chips += [f"Prefers: {_descriptor_label(d, v)}" for d, v in profile.descriptors.items()]
    return OnboardingResult(
        profile=profile, resolved_picks=resolved, display_chips=chips, method="embed_only"
    )


def explain(state: UserState, book_id: int) -> Explanation:
    """Template explanation; the real core tries the LLM first and falls back to this."""
    prof = _profile(state)
    book = _book(book_id)
    reason, sources = _reason(book, prof)
    return Explanation(
        book_id=book_id, text=reason, method="template", evidence=_evidence(book, prof, sources)
    )


def chat_turn(state: UserState, session: SessionState | None, message: str) -> ChatReply:
    """One rule-parsed chat turn. An empty message re-runs the session as it stands (the UI sends
    one after the reader removes a chip)."""
    prof = _profile(state)
    session = (
        session.model_copy(deep=True) if session else SessionState(session_id=uuid.uuid4().hex)
    )
    text = message.strip()
    low = text.lower()
    if re.search(r"\bwhy\b", low):
        return _chat_explain(state, prof, session, text)

    intent: Intent | None = None
    if not text:
        intent = "refine_previous"
    elif re.search(r"\b(?:something else|not these|other ones?|different)\b", low):
        session.excluded_ids = list(dict.fromkeys([*session.excluded_ids, *session.last_shown]))
        intent = "refine_previous"
    elif re.search(r"\bmore (?:by|from) (?:that|this|the same) author\b", low):
        source = (session.anchor_ids or session.last_shown)[:1]
        if not source:
            return _chat_reply(
                session, text, "Which book's author do you mean?", [], "more_by_author", prof
            )
        session.anchor_ids, session.anchor_authors = [], [BOOKS[source[0]].author]
        intent = "more_by_author"
    else:
        # "like X" / "by Y", or a bare title or author (what the UI sends for a picked option)
        m = _LIKE_RE.search(text) or _BY_RE.search(text)
        match = resolve_entities(m.group(1) if m else text)
        if match.kind == "ambiguous" and m:
            return _clarify(session, text, match.options)
        if match.kind == "title":
            session.anchor_ids, session.anchor_authors = match.book_ids[:1], []
            session.aspect_deltas, session.descriptor_deltas, session.filters = {}, {}, Filters()
            intent = "similar_to"
        elif match.kind == "author":
            session.anchor_ids, session.anchor_authors = [], [match.author or ""]
            intent = "more_by_author"

    modified = _apply_modifiers(session, low)
    query = None
    if intent is None:
        if modified and session.turn:
            intent = "refine_previous"
        elif modified or _looks_bookish(low):
            session.anchor_ids, session.anchor_authors = [], []
            intent, query = "recommend_general", text
        else:
            return _chat_reply(
                session,
                text,
                'I can only help with finding books. Try "something like <a title> but shorter".',
                [],
                "out_of_scope",
                prof,
            )
    elif intent == "similar_to" and modified:
        intent = "similar_to_with_modifier"
    picks = _chat_results(prof, session, query)
    intros = {
        "similar_to": "Here are books in the spirit of {anchor}.",
        "similar_to_with_modifier": "Here are books in the spirit of {anchor}, adjusted as asked.",
        "more_by_author": "More from {author}.",
        "refine_previous": "Here's a refined set.",
        "recommend_general": "Here are some picks for that.",
    }
    intro = intros[intent].format(
        anchor=BOOKS[session.anchor_ids[0]].title if session.anchor_ids else "that",
        author=session.anchor_authors[0] if session.anchor_authors else "that author",
    )
    if not picks:
        intro = "Nothing in the catalogue matched those constraints."
    return _chat_reply(session, text, intro, picks, intent, prof)


def book_insights(book_id: int) -> BookInsights:
    """Aspect bars, a summary and two snippets, all synthetic."""
    b = _book(book_id)
    aspects = list(ASPECT_LABELS)
    bars = [
        AspectBar(
            aspect=a,
            label=ASPECT_LABELS[a],
            pos_rate=b.pos[i],
            neg_rate=b.neg[i],
            mention_rate=b.mention[i],
        )
        for i, a in enumerate(aspects)
    ]
    best = sorted(range(len(aspects)), key=lambda i: -b.pos[i] * b.mention[i])[:2]
    worst = max(range(len(aspects)), key=lambda i: b.neg[i] * b.mention[i])
    praised = " and ".join(ASPECT_LABELS[aspects[i]] for i in best)
    summary = (
        f"Mock summary: readers most often praise its {praised}; "
        f"some criticise its {ASPECT_LABELS[aspects[worst]]}."
    )
    snippets = [
        Snippet(
            text=f"Mock snippet: the {ASPECT_LABELS[aspects[i]]} is what stayed with me.",
            review_ref=book_id * 10 + j,
            helpful_votes=12 - 5 * j,
            aspect=aspects[i],
        )
        for j, i in enumerate(best)
    ]
    return BookInsights(
        book_id=book_id,
        aspect_bars=bars,
        review_summary=summary,
        snippets=snippets,
        n_reviews=b.rating_count,
    )


def analyze_user_review(text: str) -> ReviewAnalysis:
    """Keyword aspect detection with word-list polarity, per clause; counts spoiler sentences."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    polarities: dict[str, list[float]] = {}
    for sentence in sentences:
        for clause in re.split(
            r",?\s+\b(?:but|although|though|however|yet|whereas)\b\s+", sentence
        ):
            words = re.findall(r"[a-z'-]+", clause.lower())
            score = sum(w in _POSITIVE for w in words) - sum(w in _NEGATIVE for w in words)
            if not score:
                continue
            for aspect, keywords in _ASPECT_KEYWORDS.items():
                if any(_has_word(clause.lower(), k) for k in keywords):
                    polarities.setdefault(aspect, []).append(math.tanh(score))
    return ReviewAnalysis(
        aspects={a: round(sum(v) / len(v), 3) for a, v in polarities.items()},
        spoiler_sentences=sum(bool(_SPOILER.search(s)) for s in sentences),
    )


def metrics_report() -> dict[str, Any]:
    """The admin metrics JSON shape (s15.6) with every metric null: the mock has no results."""
    metric_names = ("ndcg@10", "recall@20", "precision@10", "mrr@10", "coverage@10")
    return {
        "is_mock": True,
        "data_version": "mock",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bundle_version": MODEL_VERSION,
        "dataset": {
            "users": None,
            "items": N_BOOKS,
            "interactions": None,
            "attrition": [],
            "splits": {},
        },
        "models": [
            {
                "id": mid,
                "name": name,
                "mode": None,
                "metrics": dict.fromkeys(metric_names),
                "ci95": {},
            }
            for mid, name in _MODEL_ROWS
        ],
        "ablation": [],
        "strata": [],
        "longtail": [],
        "coldstart": [],
        "genai": {"intent_field_accuracy": None, "rejection_rates": {}, "audit": {}},
    }


# ---------------------------------------------------------------- internals


def _book(book_id: int) -> _Book:
    if not 0 <= book_id < N_BOOKS:
        raise ValueError(f"unknown book_id {book_id}")
    return BOOKS[book_id]


def _check_ids(ids: Iterable[int]) -> None:
    for i in ids:
        _book(i)


def _profile(state: UserState) -> _Profile:
    _check_ids(
        [e.book_id for e in state.ratings]
        + [e.book_id for e in state.reading_list]
        + state.dismissed
        + state.viewed
        + state.onboarding_picks
        + [r.book_id for r in state.review_analyses if r.book_id is not None]
    )
    ratings = sorted(state.ratings, key=lambda e: -e.ts)
    n = len(ratings) + 0.5 * len(state.reading_list)
    bucket = [name for name, lower in BUCKET_LOWER if lower <= n][-1]
    genre_aff: dict[str, float] = {}

    def add(book_id: int, weight: float) -> None:
        for g in BOOKS[book_id].genres:
            genre_aff[g] = genre_aff.get(g, 0.0) + weight

    for e in ratings:
        add(e.book_id, (e.stars - 3) / 2)  # 5 -> 1.0 ... 1 -> -1.0, as aspects.rating_weights
    for e in state.reading_list:
        add(e.book_id, 0.5)
    for b in state.onboarding_picks:
        add(b, 1.0)
    onboarding = state.onboarding or OnboardingProfile()
    for g in onboarding.liked_genres:
        genre_aff[g] = genre_aff.get(g, 0.0) + 1.0
    for g in onboarding.disliked_genres:
        genre_aff[g] = genre_aff.get(g, 0.0) - 1.0
    liked = [e.book_id for e in ratings if e.stars >= 4]
    fav_authors: dict[str, int] = {}
    for b in liked:
        fav_authors.setdefault(BOOKS[b].author, b)
    finished = [e.book_id for e in ratings] + [
        e.book_id for e in state.reading_list if e.status == "read"
    ]
    progress: dict[str, float] = {}
    for b in finished:
        book = BOOKS[b]
        if book.series_id and book.series_pos:
            progress[book.series_id] = max(progress.get(book.series_id, 0.0), book.series_pos)
    seen = (  # s12.2, plus onboarding picks: books the reader says they already enjoyed
        {e.book_id for e in ratings}
        | {e.book_id for e in state.reading_list if e.status in ("read", "dropped")}
        | set(state.dismissed)
        | set(state.onboarding_picks)
    )
    return _Profile(
        n=n,
        bucket=bucket,
        seen=seen,
        stars={e.book_id: e.stars for e in ratings},
        liked=liked,
        genre_aff=genre_aff,
        fav_authors=fav_authors,
        series_progress=progress,
        muted=set(state.overrides.muted_genres),
        onboarding_genres=set(onboarding.liked_genres),
    )


def _score(b: _Book, prof: _Profile) -> float:
    affinity = sum(prof.genre_aff.get(g, 0.0) for g in b.genres)
    return round(b.pop + 0.3 * math.tanh(affinity) + 0.5 * (b.author in prof.fav_authors), 6)


def _passes(b: _Book, filters: Filters | None) -> bool:
    if filters is None:
        return True
    return (
        (filters.min_year is None or b.year >= filters.min_year)
        and (filters.max_year is None or b.year <= filters.max_year)
        and (filters.min_pages is None or b.pages >= filters.min_pages)
        and (filters.max_pages is None or b.pages <= filters.max_pages)
        and (not filters.genres_include or bool(set(b.genres) & set(filters.genres_include)))
        and not set(b.genres) & set(filters.genres_exclude)
    )


def _allowed(b: _Book, prof: _Profile) -> bool:
    """Main-feed eligibility: unseen, not muted, and not ahead of the reader in its series."""
    if b.book_id in prof.seen or set(b.genres) & prof.muted:
        return False
    if b.series_pos is None or b.series_pos <= 1:
        return True
    return prof.series_progress.get(b.series_id or "", 0.0) >= b.series_pos  # read a later volume


def _rank(prof: _Profile, filters: Filters | None, exclude: set[int]) -> list[_Book]:
    pool = [
        b for b in BOOKS if b.book_id not in exclude and _allowed(b, prof) and _passes(b, filters)
    ]
    return sorted(pool, key=lambda b: (-_score(b, prof), b.book_id))


def _continue_series(prof: _Profile) -> list[RecItem]:
    nxt = [
        b
        for b in BOOKS
        if b.series_id in prof.series_progress
        and b.series_pos == prof.series_progress[b.series_id] + 1
        and b.book_id not in prof.seen
    ]
    reason = "Next in a series you are reading"
    return [_rec_item(b, r, prof, reason, ["item_item"]) for r, b in enumerate(nxt, 1)]


def _similar(book_id: int, prof: _Profile, also_exclude: set[int]) -> list[_Book]:
    a = BOOKS[book_id]

    def sim(b: _Book) -> float:
        shared = len(set(a.genres) & set(b.genres))
        return (
            2 * (a.genres[0] == b.genres[0])
            + shared
            + 1.5 * (a.author == b.author)
            - abs(a.year - b.year) / 100
            + 0.1 * _score(b, prof)
        )

    pool = [
        b
        for b in BOOKS
        if b.book_id != book_id and b.book_id not in also_exclude and _allowed(b, prof)
    ]
    return sorted(pool, key=lambda b: (-sim(b), b.book_id))


def _interleave(lists: list[list[_Book]]) -> list[_Book]:
    out: list[_Book] = []
    for row in zip(*lists, strict=False) if lists else []:
        out.extend(b for b in row if b not in out)
    for rest in lists:
        out.extend(b for b in rest if b not in out)
    return out


def _items_from(books: list[_Book], prof: _Profile, source: str) -> list[RecItem]:
    return [_rec_item(b, r, prof, None, [source]) for r, b in enumerate(books, 1)]


def _reason(b: _Book, prof: _Profile) -> tuple[str, list[str]]:
    """First matching template of CLAUDE.md s12.5 (the aspect template needs real aspects)."""
    if b.author in prof.fav_authors:
        liked = BOOKS[prof.fav_authors[b.author]]
        return f"Because you loved {b.author}'s {liked.title}", ["author", "popularity"]
    same = next((i for i in prof.liked if BOOKS[i].genres[0] == b.genres[0]), None)
    if same is not None:
        return (
            f"Similar to {BOOKS[same].title}, which you rated {prof.stars[same]:g}★",
            ["item_item", "popularity"],
        )
    theme = next((g for g in b.genres if g in prof.onboarding_genres), None)
    if theme:
        return f"Matches what you described: {theme.lower()}", ["semantic", "popularity"]
    if b.longtail and prof.n >= 5:
        return "A lesser-known pick that fits your taste", ["longtail"]
    if prof.bucket == "cold" and not prof.genre_aff:
        return "Popular with readers right now", ["popularity"]
    return f"Popular with {b.genres[0]} readers right now", ["popularity"]


def _evidence(b: _Book, prof: _Profile, sources: list[str]) -> Evidence:
    similar = [
        LikedItem(book_id=i, title=BOOKS[i].title, stars=prof.stars[i]).model_dump()
        for i in prof.liked
        if set(BOOKS[i].genres) & set(b.genres)
    ][:3]
    aspects = list(ASPECT_LABELS)
    return Evidence(
        book={"book_id": b.book_id, "title": b.title, "author": b.author, "genres": list(b.genres)},
        user_evidence={
            "top_positive_aspects": [],
            "top_negative_aspects": [],
            "similar_liked_items": similar,
            "favorite_author": b.author if b.author in prof.fav_authors else None,
        },
        item_evidence={
            "aspect_profile": {a: round(b.pos[i] - b.neg[i], 3) for i, a in enumerate(aspects)},
            "mention_rates": dict(zip(aspects, b.mention, strict=True)),
            "pos_rates": dict(zip(aspects, b.pos, strict=True)),
            "snippet": None,
        },
        signals={
            "sources": sources,
            "cf_rank": None,
            "semantic_rank": None,
            "aspect_match": None,
            "bucket": prof.bucket,
        },
    )


def _rec_item(
    b: _Book,
    rank: int,
    prof: _Profile,
    reason: str | None = None,
    sources: list[str] | None = None,
) -> RecItem:
    default_reason, default_sources = _reason(b, prof)
    sources = sources or default_sources
    return RecItem(
        book_id=b.book_id,
        rank=rank,
        score=_score(b, prof),
        sources=sources,
        reason=reason or default_reason,
        evidence=_evidence(b, prof, sources),
    )


def _apply_modifiers(session: SessionState, low: str) -> bool:
    """Apply mood, length and era words to the session; True if any matched."""
    found = False
    for pattern, name, value in _DESCRIPTOR_KEYWORDS:
        if re.search(pattern, low):
            session.descriptor_deltas[name] = max(-1.0, min(1.0, value))
            found = True
    if re.search(r"\bfunnier\b", low):
        session.aspect_deltas["humor"] = 0.5
        found = True
    anchor_pages = BOOKS[session.anchor_ids[0]].pages if session.anchor_ids else None
    if re.search(r"\bshorter\b", low):
        session.filters.max_pages = int(0.8 * anchor_pages) if anchor_pages else 300
        found = True
    if re.search(r"\blonger\b", low) and anchor_pages:
        session.filters.min_pages = int(1.2 * anchor_pages)
        found = True
    if re.search(r"\brecent\b", low):
        session.filters.min_year = datetime.now(UTC).year - 5
        found = True
    if re.search(r"\bclassics?\b", low):
        session.filters.max_year = 1980
        found = True
    if m := re.search(
        r"\b(?:nothing|not) older than ((?:19|20)\d\d)|\b(?:after|since) ((?:19|20)\d\d)", low
    ):
        session.filters.min_year = int(m.group(1) or m.group(2))
        found = True
    if m := re.search(r"(?<!nothing )(?<!not )\b(?:older than|before) ((?:19|20)\d\d)", low):
        session.filters.max_year = int(m.group(1))
        found = True
    return found


def _chat_results(prof: _Profile, session: SessionState, query: str | None) -> list[_Book]:
    blocked = set(session.excluded_ids) | set(session.anchor_ids)
    if session.anchor_authors:
        names = {_norm(a) for a in session.anchor_authors}
        base = [BOOKS[i] for n in names for i in _BY_AUTHOR.get(n, []) if _allowed(BOOKS[i], prof)]
    elif session.anchor_ids:
        base = _similar(session.anchor_ids[0], prof, set())
    elif query and (hits := semantic_search(query, k=50)):
        base = [BOOKS[h.book_id] for h in hits if _allowed(BOOKS[h.book_id], prof)]
    else:
        base = _rank(prof, None, set())
    position = {b.book_id: i for i, b in enumerate(base)}

    def nudge(b: _Book) -> float:
        total = sum(
            v * (len(set(b.genres) & set(_DESCRIPTOR_GENRES[d])) > 0)
            for d, v in session.descriptor_deltas.items()
        )
        total += sum(
            v * (len(set(b.genres) & set(_ASPECT_GENRES.get(a, ()))) > 0)
            for a, v in session.aspect_deltas.items()
        )
        return total

    pool = [b for b in base if b.book_id not in blocked and _passes(b, session.filters)]
    pool.sort(key=lambda b: (position[b.book_id] / len(base) - nudge(b), b.book_id))
    return pool[:CHAT_K]


def _chat_explain(state: UserState, prof: _Profile, session: SessionState, text: str) -> ChatReply:
    if not session.last_shown:
        return _chat_reply(
            session,
            text,
            "I haven't suggested anything yet; ask me for some books first.",
            [],
            "explain_recommendation",
            prof,
        )
    ordinals = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    low = text.lower()
    index = next((v for w, v in ordinals.items() if w in low), None)
    if index is None and (
        m := re.search(r"\b(?:#|number |no\.? ?)?([1-9])(?:st|nd|rd|th)?\b", low)
    ):
        index = int(m.group(1))
    index = min(index or 1, len(session.last_shown))
    book_id = session.last_shown[index - 1]
    return _chat_reply(
        session,
        text,
        explain(state, book_id).text,
        [BOOKS[book_id]],
        "explain_recommendation",
        prof,
    )


def _clarify(session: SessionState, text: str, options: list[str]) -> ChatReply:
    session.turn += 1
    session.history = [*session.history, ChatMessage(role="user", text=text[:300])][-6:]
    return ChatReply(
        intro="I found more than one match.",
        clarification=Clarification(
            question="Which one did you mean?",
            options=[ClarificationOption(label=o, value=o) for o in options],
        ),
        session=session,
        method="template",
        intent="similar_to",
    )


def _chat_reply(
    session: SessionState,
    text: str,
    intro: str,
    picks: list[_Book],
    intent: Intent,
    prof: _Profile,
) -> ChatReply:
    items = []
    for b in picks:
        reason, sources = _reason(b, prof)
        items.append(
            ChatItem(book_id=b.book_id, reason=reason, evidence=_evidence(b, prof, sources))
        )
    session.turn += 1
    if intent not in ("explain_recommendation", "out_of_scope"):
        session.last_shown = [b.book_id for b in picks]
        session.last_request = text or session.last_request
    if text:
        session.history.append(ChatMessage(role="user", text=text[:300]))
    session.history.append(ChatMessage(role="assistant", text=intro[:300]))
    session.history = session.history[-6:]
    followup = None if intent == "out_of_scope" else "Want them shorter, cozier or more recent?"
    if intent != "out_of_scope" and not picks:
        followup = "Try removing one of the active filters."
    return ChatReply(
        intro=intro,
        items=items,
        followup=followup,
        chips=_chips(session),
        session=session,
        method="template",
        intent=intent,
    )


def _chips(session: SessionState) -> list[Chip]:
    chips = [
        Chip(key=f"anchor:book:{i}", label=f"like {BOOKS[i].title}") for i in session.anchor_ids
    ]
    chips += [Chip(key="anchor:author", label=f"by {a}") for a in session.anchor_authors]
    chips += [
        Chip(key=f"descriptor:{d}", label=_descriptor_label(d, v))
        for d, v in session.descriptor_deltas.items()
    ]
    chips += [
        Chip(key=f"aspect:{a}", label=f"{'more' if v > 0 else 'less'} {ASPECT_LABELS[a]}")
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
            chips.append(Chip(key=f"filter:{key}", label=label.format(value)))
    chips += [Chip(key=f"filter:genres_include:{g}", label=f"only {g}") for g in f.genres_include]
    chips += [Chip(key=f"filter:genres_exclude:{g}", label=f"no {g}") for g in f.genres_exclude]
    return chips


def _descriptor_label(name: str, value: float) -> str:
    labels = {
        "darkness": ("cozier", "darker"),
        "pace": ("slower", "faster"),
        "complexity_level": ("easier", "denser"),
    }
    return labels[name][value > 0]


def _display(kind: str, key: str) -> str:
    index = _BY_TITLE if kind == "title" else _BY_AUTHOR
    b = BOOKS[index[key][0]]
    return b.title if kind == "title" else b.author


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _looks_bookish(low: str) -> bool:
    if re.search(r"\b(?:book|books|read|reads|novel|novels|recommend|story|stories|author)\b", low):
        return True
    return any(g.lower() in low for g in GENRES) or any(
        re.search(p, low) for p, _, _ in _DESCRIPTOR_KEYWORDS
    )


def _rss_mb() -> float | None:
    """Current resident memory from /proc (Linux, where the backend runs); None elsewhere."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            kb = next(int(line.split()[1]) for line in f if line.startswith("VmRSS:"))
    except (OSError, StopIteration, ValueError):
        return None
    return round(kb / 1024, 1)
