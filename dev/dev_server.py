"""Minimal FastAPI wrapper around ``litsense_core`` for local manual testing (CLAUDE.md s12.8).

    pip install fastapi uvicorn
    LITSENSE_BUNDLE=artifacts/serve_bundle uvicorn dev.dev_server:app --reload

Not deployed; the teammate's backend owns the real routes. Bodies are ``UserState`` JSON.
Phase 6: ``/health`` reports ``llm`` (true only when the client is enabled, has a model id and
the API key named by ``llm.api_key_env`` is set); ``/onboarding/parse``, ``/insights/{book_id}``,
``/review/analyze``, ``/entities`` and ``/author/{book_id}`` show the remaining payloads. Set
``LITSENSE_LOG_LLM=1`` to print every LLM event (prompt id, latency, cached, fallback reason;
never the key or the prompt).
"""

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import litsense_core as core
from litsense_core.schemas import Filters, SessionState, UserState

app = FastAPI(title="LitSense dev server", version=core.__api_version__)
if os.environ.get("LITSENSE_LOG_LLM"):
    core.configure_llm(None, on_event=lambda event: print("llm", event))
core.load_store(os.environ.get("LITSENSE_BUNDLE", "artifacts/serve_bundle"))


class RecommendBody(BaseModel):
    state: UserState = UserState()
    k: int = 10
    filters: Filters | None = None
    exclude: list[int] = []
    use_ranker: bool = True


class ChatBody(BaseModel):
    state: UserState = UserState()
    session: SessionState | None = None
    message: str


class OnboardingBody(BaseModel):
    text: str
    picks: list[int] = []


class ReviewBody(BaseModel):
    text: str


def _guard(fn: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return fn(*args, **kwargs)
    except ValueError as exc:  # invalid input (unknown book_id, malformed state)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    return core.health()


@app.post("/recommend")
def recommend(body: RecommendBody) -> Any:
    return _guard(core.recommend, body.state, body.k, body.filters, body.exclude, body.use_ranker)


@app.post("/rails")
def rails(state: UserState) -> Any:
    return _guard(core.home_rails, state)


@app.post("/explain/{book_id}")
def explain(book_id: int, state: UserState) -> Any:
    return _guard(core.explain, state, book_id)


@app.get("/similar/{book_id}")
def similar(book_id: int, k: int = 10) -> Any:
    return _guard(core.similar_books, book_id, k)


@app.get("/author/{book_id}")
def author(book_id: int, k: int = 10) -> Any:
    return _guard(core.more_by_author, book_id, k)


@app.get("/search")
def search(q: str, k: int = 20) -> Any:
    return core.semantic_search(q, k)


@app.get("/entities")
def entities(q: str) -> Any:
    return core.resolve_entities(q)


@app.post("/chat")
def chat(body: ChatBody) -> Any:
    return _guard(core.chat_turn, body.state, body.session, body.message)


@app.get("/onboarding/grid")
def grid(n: int = 60) -> list[int]:
    return core.onboarding_grid(n)


@app.post("/onboarding/parse")
def onboarding_parse(body: OnboardingBody) -> Any:
    return _guard(core.parse_onboarding, body.text, body.picks)


@app.get("/insights/{book_id}")
def insights(book_id: int) -> Any:
    return _guard(core.book_insights, book_id)


@app.post("/review/analyze")
def review(body: ReviewBody) -> Any:
    return core.analyze_user_review(body.text)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    return core.metrics_report()
