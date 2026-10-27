"""One route per public contract function, under ``/v1`` (D76).

Bodies wrap the function arguments; responses are the ``litsense_core.schemas`` classes the
functions return. ``ValueError`` from the core becomes 400 (contract section 5) in
``ServiceState._run``. Auth, readiness and the rate limit are enforced by ``GuardMiddleware``;
the ``Security`` dependency only documents the header in OpenAPI.
"""

import importlib
from functools import partial
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Path, Query, Request, Response, Security
from fastapi.security import APIKeyHeader

import litsense_core as core
from litsense_core.schemas import (
    BookInsights,
    CatalogBook,
    ChatReply,
    EntityMatch,
    Explanation,
    HomeRails,
    OnboardingResult,
    RecItem,
    RecResponse,
    ReviewAnalysis,
    SearchHit,
)
from service.catalog import MAX_PAGE
from service.errors import ApiError
from service.models import (
    BookListRequest,
    CatalogPage,
    ChatRequest,
    EntitiesRequest,
    ErrorEnvelope,
    ExplainRequest,
    OnboardingRequest,
    RecommendRequest,
    ReviewRequest,
    SearchRequest,
    StateRequest,
)
from service.state import ServiceState, digest, without_ai

API_KEY_HEADER = APIKeyHeader(
    name="X-API-Key", auto_error=False, description="The shared SERVICE_API_KEY."
)
ERRORS: dict[int | str, dict[str, Any]] = {
    400: {"model": ErrorEnvelope, "description": "Invalid input (validation or unknown book_id)"},
    401: {"model": ErrorEnvelope, "description": "Missing or wrong X-API-Key"},
    429: {"model": ErrorEnvelope, "description": "Rate limited; see Retry-After"},
    503: {"model": ErrorEnvelope, "description": "Model still loading, or overloaded"},
    500: {"model": ErrorEnvelope, "description": "Internal error; quote request_id"},
}
FALLBACK_HEADER = "X-LitSense-Fallback"
CACHE_HEADER = "X-LitSense-Cache"

router = APIRouter(
    prefix="/v1", tags=["model"], dependencies=[Security(API_KEY_HEADER)], responses=ERRORS
)


def get_svc(request: Request) -> ServiceState:
    return request.app.state.svc  # type: ignore[no-any-return]


Svc = Annotated[ServiceState, Depends(get_svc)]
BookIdPath = Annotated[int, Path(ge=0, description="Integer item index (books.model_item_idx)")]


@router.post("/recommend", response_model=RecResponse, summary="Main feed (recommend)")
async def recommend(body: RecommendRequest, svc: Svc) -> RecResponse:
    """`recommend(state, k, filters, exclude, use_ranker)`. Never calls the LLM; p95 budget
    80 ms on the ranker path. At most RECOMMEND_CONCURRENCY run at once."""
    async with svc.recommend_slot():
        return await svc.call(
            core.recommend, body.state, body.k, body.filters, body.exclude, body.use_ranker
        )


@router.post("/home-rails", response_model=HomeRails, summary="Every home rail (home_rails)")
async def home_rails(body: StateRequest, svc: Svc) -> HomeRails:
    """`home_rails(state)`: top picks, because-you-liked, trending and the conditional rails."""
    async with svc.recommend_slot():
        return await svc.call(core.home_rails, body.state)


@router.post("/similar-books", response_model=list[RecItem], summary="similar_books")
async def similar_books(body: BookListRequest, svc: Svc) -> list[RecItem]:
    """`similar_books(book_id, k, state)`; with a state, seen books are hidden."""
    return await svc.call(core.similar_books, body.book_id, body.k, body.state)


@router.post("/more-by-author", response_model=list[RecItem], summary="more_by_author")
async def more_by_author(body: BookListRequest, svc: Svc) -> list[RecItem]:
    """`more_by_author(book_id, k, state)`; series order first."""
    return await svc.call(core.more_by_author, body.book_id, body.k, body.state)


@router.post("/semantic-search", response_model=list[SearchHit], summary="semantic_search")
async def semantic_search(body: SearchRequest, svc: Svc) -> list[SearchHit]:
    """`semantic_search(query, k, filters)`: title and author matches, then meaning."""
    return await svc.call(core.semantic_search, body.query, body.k, body.filters)


@router.post("/resolve-entities", response_model=EntityMatch, summary="resolve_entities")
async def resolve_entities(body: EntitiesRequest, svc: Svc) -> EntityMatch:
    """`resolve_entities(query)` for author-mode search; `ambiguous` returns `options`."""
    return await svc.call(core.resolve_entities, body.query)


@router.get("/onboarding-grid", response_model=list[int], summary="onboarding_grid")
async def onboarding_grid(svc: Svc, n: Annotated[int, Query(ge=1, le=500)] = 60) -> list[int]:
    """`onboarding_grid(n)`: popular, well-rated, genre-diverse book ids for the cover picker."""
    return await svc.call(core.onboarding_grid, n)


@router.post("/parse-onboarding", response_model=OnboardingResult, summary="parse_onboarding")
async def parse_onboarding(
    body: OnboardingRequest, svc: Svc, response: Response
) -> OnboardingResult:
    """`parse_onboarding(text, picks)`. LLM-backed: past LLM_TIMEOUT_S the keyword parse answers
    (`method="embed_only"`) and the response carries `X-LitSense-Fallback: timeout`."""
    onboarding = importlib.import_module("litsense_core.onboarding")
    result, fallback = await svc.call_llm(
        "parse-onboarding",
        partial(core.parse_onboarding, body.text, body.picks),
        partial(onboarding.parse_onboarding, body.text, body.picks, svc.store, True),
    )
    if fallback:
        response.headers[FALLBACK_HEADER] = fallback
    return result


@router.post("/explain", response_model=Explanation, summary="Why this book (explain)")
async def explain(body: ExplainRequest, svc: Svc, response: Response) -> Explanation:
    """`explain(state, book_id)`: LLM text when available and valid, else the template.
    LLM-backed with the hard timeout; cached for CACHE_TTL_S per (bundle, state, book)."""
    key = digest(str(svc.bundle_version), body.state.model_dump_json(), str(body.book_id))
    cached = svc.cache_lookup(svc.explain_cache, "explain", key)
    if cached is not None:
        response.headers[CACHE_HEADER] = "hit"
        return cached  # type: ignore[no-any-return]
    result, fallback = await svc.call_llm(
        "explain",
        partial(core.explain, body.state, body.book_id),
        partial(core.explain, without_ai(body.state), body.book_id),
    )
    if fallback:
        response.headers[FALLBACK_HEADER] = fallback
    else:
        svc.explain_cache.set(key, result)
    return result


@router.post("/chat-turn", response_model=ChatReply, summary="One chat turn (chat_turn)")
async def chat_turn(body: ChatRequest, svc: Svc, response: Response) -> ChatReply:
    """`chat_turn(state, session, message)`. Persist `reply.session` and send it back next turn
    (contract section 4). LLM-backed with the hard timeout (then the rule parser and templates)."""
    result, fallback = await svc.call_llm(
        "chat-turn",
        partial(core.chat_turn, body.state, body.session, body.message),
        partial(core.chat_turn, without_ai(body.state), body.session, body.message),
    )
    if fallback:
        response.headers[FALLBACK_HEADER] = fallback
    return result


@router.get("/book-insights/{book_id}", response_model=BookInsights, summary="book_insights")
async def book_insights(book_id: BookIdPath, svc: Svc, response: Response) -> BookInsights:
    """`book_insights(book_id)`: aspect bars, review summary, spoiler-safe snippets. Cached."""
    key = digest(str(svc.bundle_version), str(book_id))
    cached = svc.cache_lookup(svc.insights_cache, "book_insights", key)
    if cached is not None:
        response.headers[CACHE_HEADER] = "hit"
        return cached  # type: ignore[no-any-return]
    result = await svc.call(core.book_insights, book_id)
    svc.insights_cache.set(key, result)
    return result


@router.post("/analyze-user-review", response_model=ReviewAnalysis, summary="analyze_user_review")
async def analyze_user_review(body: ReviewRequest, svc: Svc) -> ReviewAnalysis:
    """`analyze_user_review(text)`: store the result and send it back in
    `UserState.review_analyses` (set `book_id` and `ts` on your side)."""
    return await svc.call(core.analyze_user_review, body.text)


@router.get("/metrics-report", response_model=dict[str, Any], summary="metrics_report")
async def metrics_report(svc: Svc) -> dict[str, Any]:
    """`metrics_report()`: the admin page JSON (contract section 6). Show null as a dash."""
    return await svc.call(core.metrics_report)


@router.get("/catalog", response_model=CatalogPage, summary="Catalogue page (seed books)")
async def catalog(
    svc: Svc,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 100,
) -> CatalogPage:
    """Every book of the loaded bundle in `book_id` order, for seeding the backend's `books`
    table (contract section 7 item 1). Page with `offset` and `limit` until `items` is shorter
    than `limit`; reseed when `bundle_version` changes."""
    view = svc.catalog
    assert view is not None
    items = await svc.call(view.page, offset, limit)
    return CatalogPage(
        bundle_version=view.version, total=view.total, offset=offset, limit=limit, items=items
    )


@router.get(
    "/catalog/{book_id}",
    response_model=CatalogBook,
    summary="One catalogue book",
    responses={404: {"model": ErrorEnvelope, "description": "No book with that id"}},
)
async def catalog_book(book_id: BookIdPath, svc: Svc) -> CatalogBook:
    view = svc.catalog
    assert view is not None
    if book_id >= view.total:
        raise ApiError(404, "not_found", f"unknown book_id {book_id}")
    return view.book(book_id)
