"""HTTP request bodies and service-only responses (D76).

Every response a contract function returns uses its ``litsense_core.schemas`` class unchanged;
this module only wraps the function arguments into JSON bodies and adds the catalogue page and
the error envelope. Wrappers forbid unknown fields so a misspelt key fails with a 400 instead of
being ignored.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from litsense_core.schemas import BookId, CatalogBook, Filters, SessionState, UserState

MAX_K = 100

EXAMPLE_STATE: dict[str, Any] = {
    "ratings": [
        {"book_id": 12, "stars": 5, "ts": 1760000000000},
        {"book_id": 40, "stars": 4, "ts": 1760100000000},
    ],
    "reading_list": [{"book_id": 7, "status": "want_to_read", "ts": 1760200000000}],
    "dismissed": [3],
    "viewed": [12],
    "profile_version": 3,
}


def _examples(*examples: dict[str, Any]) -> ConfigDict:
    return ConfigDict(extra="forbid", json_schema_extra={"examples": list(examples)})


class RecommendRequest(BaseModel):
    model_config = _examples(
        {"state": EXAMPLE_STATE, "k": 10},
        {"state": {}, "k": 5, "filters": {"min_year": 2010, "genres_exclude": ["Horror"]}},
    )
    state: UserState = Field(default_factory=UserState)
    k: int = Field(default=10, ge=1, le=MAX_K)
    filters: Filters | None = None
    exclude: list[BookId] = Field(default_factory=list, max_length=2000)
    use_ranker: bool = True


class StateRequest(BaseModel):
    model_config = _examples({"state": EXAMPLE_STATE})
    state: UserState = Field(default_factory=UserState)


class BookListRequest(BaseModel):
    model_config = _examples(
        {"book_id": 12, "k": 10}, {"book_id": 12, "k": 5, "state": EXAMPLE_STATE}
    )
    book_id: BookId
    k: int = Field(default=10, ge=1, le=MAX_K)
    state: UserState | None = None


class SearchRequest(BaseModel):
    model_config = _examples(
        {"query": "cozy mystery in a small village", "k": 20},
        {"query": "Andy Weir", "k": 10, "filters": {"min_year": 2000}},
    )
    query: str = Field(max_length=500)
    k: int = Field(default=20, ge=1, le=MAX_K)
    filters: Filters | None = None


class EntitiesRequest(BaseModel):
    model_config = _examples({"query": "brandon sanderson"})
    query: str = Field(max_length=500)


class OnboardingRequest(BaseModel):
    model_config = _examples(
        {"text": "Slow-burn fantasy with morally grey characters, nothing too dark", "picks": [12]}
    )
    text: str = Field(default="", max_length=2000)
    picks: list[BookId] = Field(default_factory=list, max_length=200)


class ExplainRequest(BaseModel):
    model_config = _examples({"state": EXAMPLE_STATE, "book_id": 55})
    state: UserState = Field(default_factory=UserState)
    book_id: BookId


class ChatRequest(BaseModel):
    model_config = _examples(
        {"state": EXAMPLE_STATE, "session": None, "message": "something like Circe but shorter"},
        {
            "state": EXAMPLE_STATE,
            "session": {"session_id": "3f2a", "turn": 1, "last_shown": [5, 9]},
            "message": "why the second one?",
        },
    )
    state: UserState = Field(default_factory=UserState)
    session: SessionState | None = None
    message: str = Field(default="", max_length=2000)


class ReviewRequest(BaseModel):
    model_config = _examples(
        {"text": "The characters were wonderful, but the middle third really dragged."}
    )
    text: str = Field(max_length=20000)


class CatalogPage(BaseModel):
    """One page of the catalogue. ``bundle_version`` changes when a new bundle ships, which is
    the backend's signal to reseed."""

    bundle_version: str
    total: int
    offset: int
    limit: int
    items: list[CatalogBook]


class ErrorBody(BaseModel):
    code: str = Field(
        description="bad_request, validation_error, unauthorized, not_found, method_not_allowed, "
        "payload_too_large, rate_limited, not_ready, overloaded or internal"
    )
    message: str
    request_id: str | None
    details: list[dict[str, Any]] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody
