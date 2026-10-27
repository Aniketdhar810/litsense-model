"""Every /v1 route on the 12-book fake store: happy path, schema, error envelope (D76)."""

import threading
import time
from typing import Any

import pytest

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
from tests.service.conftest import TS

READER = {
    "ratings": [{"book_id": 0, "stars": 5, "ts": TS}, {"book_id": 3, "stars": 4, "ts": TS + 1}],
    "reading_list": [{"book_id": 4, "status": "want_to_read", "ts": TS + 2}],
    "dismissed": [8],
    "profile_version": 7,
}


def assert_envelope(response: Any, status: int, code: str) -> dict[str, Any]:
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["code"] == code
    assert error["request_id"] == response.headers["x-request-id"]
    assert error["message"]
    return error  # type: ignore[no-any-return]


def test_feed_routes(make_app: Any) -> None:
    app = make_app()
    r = app.post("/v1/recommend", {"state": READER, "k": 5})
    assert r.status_code == 200, r.text
    feed = RecResponse.model_validate(r.json())
    assert feed.profile_version == 7 and 1 <= len(feed.items) <= 5
    assert not {0, 3, 8} & {i.book_id for i in feed.items}  # rated and dismissed stay out
    assert feed.degraded is True  # the fake store ships no ranker: the blend ranks (s15.3)
    narrow = app.post(
        "/v1/recommend",
        {
            "state": READER,
            "k": 3,
            "use_ranker": False,
            "filters": {"genres_include": ["Fantasy"]},
            "exclude": [5],
        },
    )
    assert narrow.status_code == 200 and narrow.json()["degraded"] is False
    assert 5 not in [i["book_id"] for i in narrow.json()["items"]]
    rails = HomeRails.model_validate(app.post("/v1/home-rails", {"state": READER}).json())
    assert rails.top_picks and rails.trending
    assert app.post("/v1/home-rails", {}).status_code == 200  # an empty state is a cold reader


def test_book_list_and_search_routes(make_app: Any) -> None:
    app = make_app()
    sim = app.post("/v1/similar-books", {"book_id": 0, "k": 3, "state": READER})
    items = [RecItem.model_validate(x) for x in sim.json()]
    assert sim.status_code == 200 and items and 0 not in [i.book_id for i in items]
    plain = app.post("/v1/similar-books", {"book_id": 0, "k": 3}).json()
    assert plain and "★" not in plain[0]["reason"]  # no rating, no star (D75 fixed its encoding)
    rated = app.post("/v1/similar-books", {"book_id": 0, "k": 3, "state": READER}).json()
    assert all(i["reason"].endswith("rated 5★") for i in rated)
    by = app.post("/v1/more-by-author", {"book_id": 0}).json()
    assert sorted(x["book_id"] for x in by) == [1, 2]  # Andy Weir's other books
    hits = [
        SearchHit.model_validate(h)
        for h in app.post("/v1/semantic-search", {"query": "Mistborn", "k": 5}).json()
    ]
    assert hits and hits[0].matched == "title" and hits[0].book_id in (5, 6, 7)
    assert app.post("/v1/semantic-search", {"query": ""}).json() == []
    match = EntityMatch.model_validate(app.post("/v1/resolve-entities", {"query": "Circe"}).json())
    assert match.kind in ("title", "author", "none", "ambiguous")
    grid = app.get("/v1/onboarding-grid", params={"n": 5})
    assert grid.status_code == 200 and 1 <= len(grid.json()) <= 5
    assert all(isinstance(i, int) for i in grid.json())


def test_genai_routes_with_the_fake_llm(make_app: Any) -> None:
    app = make_app()
    onboarding = app.post("/v1/parse-onboarding", {"text": "funny mystery novels", "picks": [9]})
    parsed = OnboardingResult.model_validate(onboarding.json())
    assert parsed.method == "llm" and 9 in parsed.resolved_picks
    exp = Explanation.model_validate(
        app.post("/v1/explain", {"state": READER, "book_id": 1}).json()
    )
    assert exp.method == "llm" and exp.book_id == 1
    first = app.post("/v1/chat-turn", {"state": READER, "message": "something like The Martian"})
    reply = ChatReply.model_validate(first.json())
    assert reply.intent in ("similar_to", "similar_to_with_modifier") and reply.session.turn == 1
    second = app.post(
        "/v1/chat-turn",
        {"state": READER, "session": first.json()["session"], "message": "more by that author"},
    )
    assert ChatReply.model_validate(second.json()).session.turn == 2
    third = app.post(
        "/v1/chat-turn",
        {"state": READER, "session": second.json()["session"], "message": "why the second one?"},
    )
    assert third.status_code == 200 and third.json()["session"]["turn"] == 3


def test_insight_review_and_report_routes(make_app: Any) -> None:
    app = make_app()
    insights = BookInsights.model_validate(app.get("/v1/book-insights/0").json())
    assert len(insights.aspect_bars) == 12 and insights.snippets
    again = app.get("/v1/book-insights/0")
    assert again.headers.get("x-litsense-cache") == "hit" and again.json() == insights.model_dump(
        mode="json"
    )
    review = app.post("/v1/analyze-user-review", {"text": "Great plot. It turns out he dies."})
    assert review.status_code == 200
    ReviewAnalysis.model_validate(review.json())
    report = app.get("/v1/metrics-report").json()
    assert report["bundle_version"] == "fake-1" and report["is_mock"] is False


def test_catalog_pages_and_single_book(make_app: Any) -> None:
    app = make_app()
    first = app.get("/v1/catalog", params={"limit": 5}).json()
    assert first["total"] == 12 and first["bundle_version"] == "fake-1"
    assert [b["book_id"] for b in first["items"]] == [0, 1, 2, 3, 4]
    seen: list[int] = []
    offset = 0
    while True:
        page = app.get("/v1/catalog", params={"offset": offset, "limit": 5}).json()
        seen += [CatalogBook.model_validate(b).book_id for b in page["items"]]
        if len(page["items"]) < 5:
            break
        offset += 5
    assert seen == list(range(12))
    assert app.get("/v1/catalog", params={"offset": 50}).json()["items"] == []
    book = CatalogBook.model_validate(app.get("/v1/catalog/5").json())
    assert book.title == "Mistborn: The Final Empire" and book.series_id == "mistborn"
    assert book.series_pos == 1.0 and book.year == 2006 and book.pages == 541
    assert book.isbn13 is None  # the fake store has no catalog_text.json.gz
    untitled = app.get("/v1/catalog/11").json()
    assert untitled["author"] is None and untitled["series_id"] is None
    assert_envelope(app.get("/v1/catalog/12"), 404, "not_found")
    assert_envelope(app.get("/v1/catalog", params={"limit": 0}), 400, "validation_error")
    assert_envelope(app.get("/v1/catalog", params={"limit": 1001}), 400, "validation_error")


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/recommend", {"state": {"ratings": [{"book_id": 999, "stars": 4, "ts": TS}]}}),
        ("/v1/explain", {"book_id": 999}),
        ("/v1/similar-books", {"book_id": 999}),
        ("/v1/more-by-author", {"book_id": 999}),
        ("/v1/parse-onboarding", {"text": "fantasy", "picks": [999]}),
    ],
)
def test_unknown_book_id_is_400_never_500(make_app: Any, path: str, body: Any) -> None:
    app = make_app()
    error = assert_envelope(app.post(path, body), 400, "bad_request")
    assert "book_id" in error["message"]


def test_validation_and_routing_errors(make_app: Any) -> None:
    app = make_app()
    assert_envelope(app.get("/v1/book-insights/999"), 400, "bad_request")
    assert_envelope(app.get("/v1/book-insights/-1"), 400, "validation_error")
    zero = assert_envelope(app.post("/v1/recommend", {"k": 0}), 400, "validation_error")
    assert zero["details"][0]["loc"] == ["body", "k"]
    assert all(set(d) == {"loc", "msg", "type"} for d in zero["details"])  # never the input
    seconds = {"state": {"ratings": [{"book_id": 1, "stars": 4, "ts": 1_650_000_000}]}}
    assert_envelope(app.post("/v1/recommend", seconds), 400, "validation_error")
    assert_envelope(app.post("/v1/recommend", {"state": {}, "kk": 3}), 400, "validation_error")
    broken = app.client.post(
        "/v1/recommend",
        content=b"{not json",
        headers={"X-API-Key": "test-service-key", "Content-Type": "application/json"},
    )
    assert_envelope(broken, 400, "validation_error")
    assert_envelope(app.get("/v1/nope"), 404, "not_found")
    assert_envelope(app.get("/v1/recommend"), 405, "method_not_allowed")


def test_unexpected_core_errors_are_500_with_request_id(
    make_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app()

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("secret internals")

    monkeypatch.setattr(core, "recommend", boom)
    r = app.post("/v1/recommend", {}, headers={"X-Request-Id": "trace-42"})
    error = assert_envelope(r, 500, "internal")
    assert error["request_id"] == "trace-42" and "secret" not in r.text

    def bad_input(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("k is out of range")

    monkeypatch.setattr(core, "similar_books", bad_input)
    assert_envelope(app.post("/v1/similar-books", {"book_id": 0}), 400, "bad_request")


def test_llm_routes_fall_back_to_the_template_after_the_timeout(make_app: Any) -> None:
    lock = threading.Lock()
    running = [0]

    def slow(prompt: str) -> str:
        with lock:
            running[0] += 1
        time.sleep(1.5)
        with lock:
            running[0] -= 1
        return "Too late to matter."

    app = make_app(
        script={"explain": [slow], "intent_parse": [slow], "onboarding_parse": [slow]},
        llm_timeout_s=0.3,
    )
    started = time.perf_counter()
    r = app.post("/v1/explain", {"state": READER, "book_id": 1})
    assert time.perf_counter() - started < 1.2
    assert r.status_code == 200 and r.headers["x-litsense-fallback"] == "timeout"
    assert r.json()["method"] == "template"
    again = app.post("/v1/explain", {"state": READER, "book_id": 1})
    assert "x-litsense-cache" not in again.headers  # a timeout answer is never cached
    chat = app.post("/v1/chat-turn", {"state": READER, "message": "something like Dune"})
    assert chat.status_code == 200 and chat.headers["x-litsense-fallback"] == "timeout"
    assert chat.json()["method"] == "template"
    onboarding = app.post("/v1/parse-onboarding", {"text": "funny mysteries", "picks": []})
    assert onboarding.headers["x-litsense-fallback"] == "timeout"
    assert onboarding.json()["method"] == "embed_only"
    metrics = app.get("/metrics").text
    assert 'litsense_llm_route_timeouts_total{route="explain"}' in metrics
    deadline = time.monotonic() + 10  # abandoned calls finish on their own; let them log here
    while running[0] and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.1)


def test_explain_cache_is_keyed_by_state_and_book(make_app: Any) -> None:
    app = make_app()
    first = app.post("/v1/explain", {"state": READER, "book_id": 1})
    assert "x-litsense-cache" not in first.headers
    hit = app.post("/v1/explain", {"state": READER, "book_id": 1})
    assert hit.headers["x-litsense-cache"] == "hit" and hit.json() == first.json()
    other = app.post("/v1/explain", {"state": {**READER, "profile_version": 8}, "book_id": 1})
    assert "x-litsense-cache" not in other.headers
    text = app.get("/metrics").text
    assert 'litsense_cache_lookups_total{cache="explain",result="hit"} 1.0' in text
    assert 'litsense_cache_lookups_total{cache="explain",result="miss"} 2.0' in text


def test_recommend_slots_refuse_when_saturated(
    make_app: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = make_app(recommend_concurrency=1, queue_timeout_s=0.2)
    real = core.recommend
    entered = threading.Event()

    def slow(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        time.sleep(1.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(core, "recommend", slow)
    results: list[int] = []
    worker = threading.Thread(
        target=lambda: results.append(app.post("/v1/recommend", {}).status_code)
    )
    worker.start()
    assert entered.wait(5)
    second = app.post("/v1/recommend", {})
    worker.join(10)
    assert results == [200]
    error = assert_envelope(second, 503, "overloaded")
    assert second.headers["retry-after"] == "1" and "concurrent" in error["message"]
