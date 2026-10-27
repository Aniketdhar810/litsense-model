"""Pydantic v2 schemas of the litsense_core public API (CLAUDE.md s15.2).

These models are the contract with the backend. Any change bumps
``litsense_core.__api_version__`` and is announced in ``docs/API_CONTRACT.md``.
``book_id`` is always the integer item index from ``id_map.json``.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Epoch milliseconds. The bounds (years 1973-2286) reject epoch seconds, a classic unit bug.
TimestampMs = Annotated[int, Field(ge=100_000_000_000, le=9_999_999_999_999)]
BookId = Annotated[int, Field(ge=0)]
Stars = Annotated[float, Field(ge=1.0, le=5.0)]

ListStatus = Literal["want_to_read", "currently_reading", "read", "dropped"]
Bucket = Literal["cold", "warming", "blending", "warm"]  # mirrors ranking.buckets in ml/config.yaml
Method = Literal["template", "llm"]
Intent = Literal[
    "recommend_general",
    "similar_to",
    "similar_to_with_modifier",
    "more_by_author",
    "refine_previous",
    "explain_recommendation",
    "filter_by_attribute",
    "mixed_opinion",
    "out_of_scope",
]

# Aspect vector order (index 0-11) and the label the UI shows; mirrors aspects.names in config.
ASPECT_LABELS: dict[str, str] = {
    "plot": "plot",
    "writing_style": "prose",
    "characters": "characters",
    "world_building": "world-building",
    "pacing": "pacing",
    "editing_quality": "editing",
    "dialogue": "dialogue",
    "humor": "humor",
    "emotional_impact": "emotional punch",
    "ending": "ending",
    "complexity": "depth",
    "tone_dark": "tone",
}


class RatingEvent(BaseModel):
    book_id: BookId
    stars: Stars
    ts: TimestampMs


class ListEvent(BaseModel):
    book_id: BookId
    status: ListStatus
    ts: TimestampMs


class Overrides(BaseModel):
    muted_genres: list[str] = Field(default_factory=list)
    pinned_aspects: list[str] = Field(default_factory=list)
    disabled_ai: bool = False


class OnboardingProfile(BaseModel):
    liked_genres: list[str] = Field(default_factory=list)
    disliked_genres: list[str] = Field(default_factory=list)
    positive_aspects: list[str] = Field(default_factory=list)
    negative_aspects: list[str] = Field(default_factory=list)
    descriptors: dict[str, float] = Field(default_factory=dict)  # darkness, pace, complexity_level
    themes: list[str] = Field(default_factory=list)
    named_titles: list[str] = Field(default_factory=list)
    named_authors: list[str] = Field(default_factory=list)


class ReviewAnalysis(BaseModel):
    book_id: BookId | None = None
    ts: TimestampMs | None = None
    aspects: dict[str, float] = Field(default_factory=dict)  # aspect -> polarity in [-1, 1]
    spoiler_sentences: int = 0


class UserState(BaseModel):
    """Everything the core knows about a reader; rebuilt by the backend for every call."""

    user_id: int | None = None  # never sent to the LLM
    corpus_user_id: str | None = None  # demo accounts seeded from the corpus
    ratings: list[RatingEvent] = Field(default_factory=list)
    reading_list: list[ListEvent] = Field(default_factory=list)
    dismissed: list[BookId] = Field(default_factory=list)
    viewed: list[BookId] = Field(default_factory=list)  # detail-page views, newest first (max 50)
    onboarding: OnboardingProfile | None = None
    onboarding_picks: list[BookId] = Field(default_factory=list)
    onboarding_text: str | None = None
    review_analyses: list[ReviewAnalysis] = Field(default_factory=list)
    overrides: Overrides = Field(default_factory=Overrides)
    profile_version: int = 0


class Filters(BaseModel):
    min_year: int | None = None
    max_year: int | None = None
    min_pages: int | None = None
    max_pages: int | None = None
    genres_include: list[str] = Field(default_factory=list)
    genres_exclude: list[str] = Field(default_factory=list)


class AspectStrength(BaseModel):
    aspect: str
    label: str
    strength: float
    n: int


class LikedItem(BaseModel):
    book_id: BookId
    title: str
    stars: float


class Snippet(BaseModel):
    text: str
    review_ref: int
    helpful_votes: int
    aspect: str


class Evidence(BaseModel):
    """Facts behind a recommendation; the only input an explanation may use. Values are plain
    JSON (nested schema objects are stored via ``model_dump()``)."""

    book: dict[str, Any]  # book_id, title, author, genres
    user_evidence: dict[
        str, Any
    ]  # top_positive_aspects, top_negative_aspects, similar_liked_items, favorite_author
    item_evidence: dict[str, Any]  # aspect_profile, mention_rates, pos_rates, snippet
    signals: dict[str, Any]  # sources, cf_rank, semantic_rank, aspect_match, bucket

    def allowed_titles(self) -> set[str]:
        """Titles generated text may mention: this book and the reader's similar liked books."""
        titles = {self.book.get("title")}
        titles.update(
            item.get("title") for item in self.user_evidence.get("similar_liked_items", [])
        )
        return {t for t in titles if t}


class RecItem(BaseModel):
    book_id: BookId
    rank: int
    score: float
    sources: list[str]
    reason: str
    reason_method: Method = "template"
    evidence: Evidence


class RecResponse(BaseModel):
    items: list[RecItem]
    continue_series: list[RecItem] = Field(default_factory=list)
    model_version: str
    profile_version: int
    bucket: Bucket
    degraded: bool = False
    latency_ms: float


class BecauseYouLiked(BaseModel):
    """One "Because you liked {title}" rail."""

    anchor_book_id: BookId
    items: list[RecItem]


class HomeRails(BaseModel):
    top_picks: list[RecItem]
    because_you_liked: list[BecauseYouLiked]  # up to 2 rails
    trending: list[RecItem]
    more_by_authors: list[RecItem] = Field(
        default_factory=list
    )  # empty unless a favorite author has 2+ books
    continue_exploring: list[RecItem] = Field(default_factory=list)  # from state.viewed
    hidden_gems: list[RecItem] = Field(default_factory=list)  # longtail source; empty until n >= 5
    from_reading_list: list[RecItem] = Field(default_factory=list)
    continue_series: list[RecItem] = Field(default_factory=list)


class SearchHit(BaseModel):
    book_id: BookId
    score: float
    matched: Literal["semantic", "title", "author"]


class EntityMatch(BaseModel):
    kind: Literal["title", "author", "none", "ambiguous"]
    book_ids: list[BookId] = Field(default_factory=list)
    author: str | None = None
    options: list[str] = Field(default_factory=list)


class OnboardingResult(BaseModel):
    profile: OnboardingProfile
    resolved_picks: list[BookId]
    display_chips: list[str]
    method: Literal["llm", "embed_only"]


class Explanation(BaseModel):
    book_id: BookId
    text: str
    method: Method
    evidence: Evidence
    rejections: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    text: str  # truncated to 300 characters by the core


class SessionState(BaseModel):
    """Chat session state; the backend persists it and sends it back on the next turn."""

    session_id: str
    turn: int = 0
    aspect_deltas: dict[str, float] = Field(default_factory=dict)
    descriptor_deltas: dict[str, float] = Field(default_factory=dict)
    filters: Filters = Field(default_factory=Filters)
    excluded_ids: list[BookId] = Field(default_factory=list)
    anchor_ids: list[BookId] = Field(default_factory=list)
    anchor_authors: list[str] = Field(default_factory=list)
    last_shown: list[BookId] = Field(default_factory=list)
    last_request: str | None = None
    history: list[ChatMessage] = Field(default_factory=list)  # last 6 messages


class ChatItem(BaseModel):
    book_id: BookId
    reason: str
    evidence: Evidence


class Chip(BaseModel):
    key: str  # e.g. "descriptor:darkness"
    label: str  # e.g. "cozier"


class ClarificationOption(BaseModel):
    label: str
    value: str | int


class Clarification(BaseModel):
    question: str
    options: list[ClarificationOption]


class ChatReply(BaseModel):
    intro: str
    items: list[ChatItem] = Field(default_factory=list)
    followup: str | None = None
    chips: list[Chip] = Field(default_factory=list)
    clarification: Clarification | None = None
    session: SessionState
    method: Method
    intent: Intent


class AspectBar(BaseModel):
    aspect: str
    label: str
    pos_rate: float
    neg_rate: float
    mention_rate: float


class BookInsights(BaseModel):
    book_id: BookId
    aspect_bars: list[AspectBar]
    review_summary: str | None
    snippets: list[Snippet]
    n_reviews: int


class CatalogBook(BaseModel):
    """PROPOSED (not in CLAUDE.md s15): catalogue row the backend seeds its books table from."""

    book_id: BookId
    canonical_id: str
    title: str
    author: str | None
    genres: list[str]
    primary_genre: str
    year: int | None
    pages: int | None
    description: str
    image: str | None
    isbn13: str | None
    series_id: str | None = None
    series_name: str | None = None
    series_pos: float | None = None
