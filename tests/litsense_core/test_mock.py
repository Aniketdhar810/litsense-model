"""litsense_core.mock honours the s15 contract the backend builds against in Week 1."""

import inspect
from pathlib import Path
from typing import Any

import pytest

from litsense_core import mock
from litsense_core.schemas import (
    ASPECT_LABELS,
    ChatReply,
    Filters,
    ListEvent,
    OnboardingProfile,
    Overrides,
    RatingEvent,
    SessionState,
    UserState,
)

TS = 1_700_000_000_000
DAY = 86_400_000

# CLAUDE.md s15.1: name -> (parameters with defaults, return annotation)
SPEC = {
    "load_store": ("(bundle_dir: str)", None),
    "download_bundle": ("(url: str, dest: str)", "str"),
    "health": ("()", "dict"),
    "recommend": (
        "(state: UserState, k: int = 10, filters: Filters | None = None, "
        "exclude: list[int] | None = None, use_ranker: bool = True)",
        "RecResponse",
    ),
    "home_rails": ("(state: UserState)", "HomeRails"),
    "similar_books": ("(book_id: int, k: int = 10, state: UserState | None = None)", "list"),
    "more_by_author": ("(book_id: int, k: int = 10, state: UserState | None = None)", "list"),
    "semantic_search": ("(query: str, k: int = 20, filters: Filters | None = None)", "list"),
    "resolve_entities": ("(query: str)", "EntityMatch"),
    "onboarding_grid": ("(n: int = 60)", "list"),
    "parse_onboarding": ("(text: str, picks: list[int])", "OnboardingResult"),
    "explain": ("(state: UserState, book_id: int)", "Explanation"),
    "chat_turn": (
        "(state: UserState, session: SessionState | None, message: str)",
        "ChatReply",
    ),
    "book_insights": ("(book_id: int)", "BookInsights"),
    "analyze_user_review": ("(text: str)", "ReviewAnalysis"),
    "metrics_report": ("()", "dict"),
}


def rate(*pairs: tuple[int, float]) -> UserState:
    return UserState(
        ratings=[RatingEvent(book_id=b, stars=s, ts=TS + i * DAY) for i, (b, s) in enumerate(pairs)]
    )


def ids(items: list[Any]) -> list[int]:
    return [i.book_id for i in items]


def series_volumes() -> list[list[mock._Book]]:
    by_series: dict[str, list[mock._Book]] = {}
    for b in mock.BOOKS:
        if b.series_id:
            by_series.setdefault(b.series_id, []).append(b)
    return [sorted(v, key=lambda b: b.series_pos or 0) for v in by_series.values()]


@pytest.mark.parametrize("name", sorted(SPEC))
def test_public_functions_match_spec_signatures(name: str) -> None:
    params, returns = SPEC[name]
    sig = inspect.signature(getattr(mock, name))
    shown = str(sig.replace(return_annotation=inspect.Signature.empty))
    assert shown.replace("litsense_core.schemas.", "") == params
    if returns:
        annotation = sig.return_annotation
        assert getattr(annotation, "__name__", str(annotation)).startswith(returns)


def test_catalogue_is_deterministic_and_well_formed() -> None:
    assert mock._build_catalog() == mock.BOOKS
    books = mock.catalog()
    assert len(books) == mock.N_BOOKS == len({b.title for b in books})
    assert [b.book_id for b in books] == list(range(mock.N_BOOKS))
    volumes = series_volumes()
    assert len(volumes) == 6
    for vol in volumes:
        assert [b.series_pos for b in vol] == [1.0, 2.0, 3.0]
        assert len({b.author for b in vol}) == 1
        assert [b.year for b in vol] == sorted(b.year for b in vol)


def test_cold_reader_gets_popular_books() -> None:
    feed = mock.recommend(UserState(), k=10)
    assert feed.bucket == "cold" and len(feed.items) == 10
    assert all(i.reason == "Popular with readers right now" for i in feed.items)
    pops = [mock.BOOKS[i].pop for i in ids(feed.items)]
    assert pops == sorted(pops, reverse=True)
    assert [i.rank for i in feed.items] == list(range(1, 11))


def test_onboarding_picks_alone_give_a_feed_without_the_picks() -> None:
    picks = mock.onboarding_grid()[:3]
    feed = mock.recommend(UserState(onboarding_picks=picks), k=10)
    assert feed.bucket == "cold" and len(feed.items) == 10
    assert not set(picks) & set(ids(feed.items))


def test_rating_changes_the_feed() -> None:
    before = ids(mock.recommend(UserState(), k=10).items)
    anchor = next(b for b in mock.BOOKS if b.genres[0] == "Horror")
    feed = mock.recommend(rate((anchor.book_id, 5)), k=10)
    assert feed.bucket == "warming"
    assert ids(feed.items) != before and anchor.book_id not in ids(feed.items)
    assert any("Because you loved" in i.reason for i in feed.items)


def test_feed_respects_seen_dismissed_exclude_and_filters() -> None:
    state = UserState(
        ratings=[RatingEvent(book_id=0, stars=4, ts=TS)],
        reading_list=[
            ListEvent(book_id=1, status="read", ts=TS),
            ListEvent(book_id=2, status="dropped", ts=TS),
            ListEvent(book_id=4, status="want_to_read", ts=TS),
        ],
        dismissed=[5],
        overrides=Overrides(muted_genres=["Romance"]),
    )
    filters = Filters(min_year=1980, max_year=2015, max_pages=500, genres_exclude=["Horror"])
    feed = mock.recommend(state, k=15, filters=filters, exclude=[6, 7])
    assert len(feed.items) == 15
    for item in feed.items:
        b = mock.BOOKS[item.book_id]
        assert b.book_id not in {0, 1, 2, 5, 6, 7}
        assert 1980 <= b.year <= 2015 and b.pages <= 500
        assert not {"Horror", "Romance"} & set(b.genres)


def test_series_guard() -> None:
    first, second, third = series_volumes()[0]
    state = rate((first.book_id, 5))
    feed = mock.recommend(state, k=mock.N_BOOKS)
    assert second.book_id not in ids(feed.items) and third.book_id not in ids(feed.items)
    assert ids(feed.continue_series) == [second.book_id]
    rails = mock.home_rails(state)
    shown = [*rails.top_picks, *rails.trending, *rails.more_by_authors, *rails.hidden_gems]
    shown += [i for rail in rails.because_you_liked for i in rail.items]
    assert not {second.book_id, third.book_id} & set(ids(shown))
    assert ids(rails.continue_series) == [second.book_id]


@pytest.mark.parametrize(
    ("n_ratings", "n_list", "bucket"),
    [(0, 0, "cold"), (0, 1, "cold"), (1, 0, "warming"), (4, 1, "warming"), (5, 0, "blending"),
     (19, 1, "blending"), (20, 0, "warm")],
)  # fmt: skip
def test_buckets_use_lower_bounds(n_ratings: int, n_list: int, bucket: str) -> None:
    state = UserState(
        ratings=[RatingEvent(book_id=i, stars=4, ts=TS) for i in range(n_ratings)],
        reading_list=[
            ListEvent(book_id=100 + i, status="want_to_read", ts=TS) for i in range(n_list)
        ],
    )
    assert mock.recommend(state, k=1).bucket == bucket


def test_bucket_bounds_mirror_config(repo_config: dict[str, Any]) -> None:
    lowers = {name: lo for name, (lo, _) in repo_config["ranking"]["buckets"].items()}
    assert dict(mock.BUCKET_LOWER) == lowers


def test_invalid_input_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown book_id"):
        mock.recommend(rate((mock.N_BOOKS, 5)))
    with pytest.raises(ValueError, match="unknown book_id"):
        mock.similar_books(10_000)
    with pytest.raises(ValueError):
        mock.recommend(UserState(), k=0)


def test_home_rails_fill_once_there_is_history() -> None:
    liked = [b.book_id for b in mock.BOOKS if b.author == mock.BOOKS[13].author][:2]
    state = UserState(
        ratings=[
            RatingEvent(book_id=b, stars=5, ts=TS + i * DAY) for i, b in enumerate([*liked, 30, 31])
        ],
        reading_list=[ListEvent(book_id=b, status="want_to_read", ts=TS) for b in (50, 51)],
        viewed=[60, 61],
    )
    rails = mock.home_rails(state)
    assert len(rails.top_picks) == 10 and len(rails.trending) == 20
    assert [r.anchor_book_id for r in rails.because_you_liked] == [31, 30]
    for rail in (rails.continue_exploring, rails.hidden_gems, rails.from_reading_list):
        assert len(rail) == 10
    assert rails.more_by_authors
    everything = [i for rail in rails.because_you_liked for i in rail.items] + rails.top_picks
    assert not {*liked, 30, 31} & set(ids(everything))


def test_similar_books_and_more_by_author() -> None:
    anchor = mock.BOOKS[13]
    similar = mock.similar_books(13, k=10)
    assert len(similar) == 10 and 13 not in ids(similar)
    assert mock.BOOKS[similar[0].book_id].genres[0] == anchor.genres[0]
    same = mock.more_by_author(13, k=10)
    assert same and all(mock.BOOKS[i].author == anchor.author for i in ids(same))
    assert 13 not in ids(same)


def test_search_and_entity_resolution() -> None:
    target = mock.BOOKS[32]
    hits = mock.semantic_search(target.title.lower(), k=5)
    assert hits[0].book_id == 32 and hits[0].matched == "title"
    assert mock.semantic_search("the of a") == []
    assert mock.resolve_entities("the glass cartograper").book_ids == [32]  # typo
    author = mock.resolve_entities("hazel elison")
    assert author.kind == "author" and author.author == "Hazel Ellison"
    assert mock.resolve_entities("zzzz qqqq").kind == "none"
    ambiguous = mock.resolve_entities("las meridian")
    assert ambiguous.kind == "ambiguous" and len(ambiguous.options) >= 2


def test_onboarding_grid_and_parse() -> None:
    grid = mock.onboarding_grid()
    assert len(grid) == len(set(grid)) == 60
    first_genres = [mock.BOOKS[b].genres[0] for b in grid[: len(mock.GENRES)]]
    assert sorted(first_genres) == sorted(mock.GENRES)
    title = mock.BOOKS[32].title
    result = mock.parse_onboarding(
        f"I love slow-burn fantasy with found-family characters, like {title}. "
        "Not into grimdark or horror.",
        picks=[1, 2],
    )
    assert result.method == "embed_only"
    assert result.profile.liked_genres == ["Fantasy"] and result.profile.disliked_genres == [
        "Horror"
    ]
    assert result.profile.descriptors == {"pace": -0.5, "darkness": -0.6}
    assert "characters" in result.profile.positive_aspects
    assert result.resolved_picks == [1, 2, 32]
    assert "Likes: Fantasy" in result.display_chips
    with pytest.raises(ValueError):
        mock.parse_onboarding("anything", picks=[999])


def test_explain_is_a_template_grounded_in_evidence() -> None:
    state = rate((13, 5))
    other = next(i for i in mock.more_by_author(13) if i.book_id != 13).book_id
    explanation = mock.explain(state, other)
    assert explanation.method == "template" and explanation.rejections == []
    assert explanation.text.startswith(f"Because you loved {mock.BOOKS[13].author}")
    assert mock.BOOKS[13].title in explanation.evidence.allowed_titles()


def test_three_turn_chat_flow() -> None:
    state = rate((13, 5))
    anchor = mock.BOOKS[40]
    turn1 = mock.chat_turn(state, None, f"something like {anchor.title} but shorter")
    assert turn1.intent == "similar_to_with_modifier" and len(turn1.items) == mock.CHAT_K
    assert all(mock.BOOKS[i].pages <= 0.8 * anchor.pages for i in ids(turn1.items))
    assert {c.key for c in turn1.chips} >= {"anchor:book:40", "filter:max_pages"}
    assert turn1.session.anchor_ids == [40] and turn1.session.turn == 1

    session = SessionState.model_validate_json(turn1.session.model_dump_json())  # backend storage
    turn2 = mock.chat_turn(state, session, "more by that author")
    assert turn2.intent == "more_by_author" and turn2.items
    assert all(mock.BOOKS[i].author == anchor.author for i in ids(turn2.items))

    turn3 = mock.chat_turn(state, turn2.session, "why did you suggest the second one?")
    second = turn2.session.last_shown[1] if len(turn2.items) > 1 else turn2.session.last_shown[0]
    assert turn3.intent == "explain_recommendation"
    assert ids(turn3.items) == [second]
    assert turn3.session.last_shown == turn2.session.last_shown
    assert len(turn3.session.history) == 6


def test_chat_edges() -> None:
    state = UserState()
    off = mock.chat_turn(state, None, "what's the weather tomorrow")
    assert off.intent == "out_of_scope" and off.items == []

    first = mock.chat_turn(state, None, "a cozy mystery")
    again = mock.chat_turn(state, first.session, "something else")
    assert again.intent == "refine_previous"
    assert not set(ids(first.items)) & set(ids(again.items))

    clarify = mock.chat_turn(state, None, "something like las meridian")
    assert clarify.clarification is not None and clarify.items == []
    assert len(clarify.clarification.options) >= 2
    choice = clarify.clarification.options[0]
    picked = mock.chat_turn(state, clarify.session, str(choice.value))  # UI sends the value
    assert picked.intent == "similar_to" and picked.items
    assert mock.BOOKS[picked.session.anchor_ids[0]].title == choice.label


def test_removing_a_chip_and_resending_recomputes() -> None:
    anchor = mock.BOOKS[40]
    turn = mock.chat_turn(UserState(), None, f"like {anchor.title} but shorter")
    session = turn.session.model_copy(deep=True)
    session.filters.max_pages = None  # the UI removed the "under N pages" chip
    rerun = mock.chat_turn(UserState(), session, "")
    assert rerun.intent == "refine_previous"
    assert "filter:max_pages" not in {c.key for c in rerun.chips}
    assert any(mock.BOOKS[i].pages > 0.8 * anchor.pages for i in ids(rerun.items))


def test_book_insights_are_well_formed() -> None:
    insights = mock.book_insights(13)
    assert [b.aspect for b in insights.aspect_bars] == list(ASPECT_LABELS)
    for bar in insights.aspect_bars:
        assert 0 <= bar.mention_rate <= 1 and 0 <= bar.pos_rate + bar.neg_rate <= 1
    assert insights.review_summary.startswith("Mock summary")
    assert all(s.text.startswith("Mock snippet") for s in insights.snippets)


def test_analyze_user_review() -> None:
    mixed = mock.analyze_user_review(
        "Wonderful characters, but the middle third really dragged. "
        "It turns out the narrator was dead the whole time."
    )
    assert mixed.aspects["characters"] > 0 > mixed.aspects["pacing"]
    assert mixed.spoiler_sentences == 1
    assert mock.analyze_user_review("I read it on a train.").aspects == {}


def test_metrics_report_contains_no_numbers() -> None:
    report = mock.metrics_report()
    assert report["is_mock"] is True
    for model in report["models"]:
        assert set(model["metrics"].values()) == {None}
    assert report["genai"]["intent_field_accuracy"] is None


def test_store_health_and_bundle(tmp_path: Path) -> None:
    assert mock.load_store("unused").n_books == mock.N_BOOKS
    assert mock.health()["mock"] is True and mock.health()["llm"] is False
    dest = mock.download_bundle("https://example.invalid/bundle.tar.gz", str(tmp_path / "b"))
    assert Path(dest).is_dir()


def test_reply_schemas_round_trip() -> None:
    reply = mock.chat_turn(rate((13, 5)), None, "something like the glass cartographer")
    assert ChatReply.model_validate_json(reply.model_dump_json()) == reply
    profile = OnboardingProfile(liked_genres=["Fantasy"])
    feed = mock.recommend(UserState(onboarding=profile), k=10)
    fantasy = [i for i in feed.items if "Fantasy" in mock.BOOKS[i.book_id].genres]
    assert fantasy and all(i.reason == "Matches what you described: fantasy" for i in fantasy)
