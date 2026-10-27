"""Chat turns (CLAUDE.md s14.7, s16.2 GenAI): the rule parser, the three-turn demo with the LLM
off, and the LLM reply validators with the fake provider."""

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

import litsense_core as core
from litsense_core import chat
from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.llm.adapters.fake import FakeAdapter, prompt_field
from litsense_core.llm.client import set_client
from litsense_core.schemas import RatingEvent, SessionState, UserState
from litsense_core.store import set_store
from tests.litsense_core.fake_store import build_store

TS = 1_650_000_000_000


@pytest.fixture
def store(tmp_path: Path) -> Any:
    st = build_store(tmp_path / "bundle")
    set_store(st)
    set_client(LLMClient(LLMConfig(cache_path=str(tmp_path / "off.db"))))
    yield st
    set_store(None)
    set_client(None)


_N = itertools.count()


def fake_llm(tmp_path: Path, script: dict[str, list[Any]], **over: Any) -> FakeAdapter:
    adapter = FakeAdapter(script)
    path = str(tmp_path / f"llm{next(_N)}.db")
    cfg = LLMConfig.from_mapping(
        {"enabled": True, "provider": "fake", "model": "fake-1", "cache_path": path, **over}
    )
    set_client(LLMClient(cfg, adapter=adapter, sleep=lambda s: None))
    return adapter


def reader() -> UserState:
    return UserState(ratings=[RatingEvent(book_id=0, stars=5.0, ts=TS)])


def test_rule_parser_cases(store: Any) -> None:
    fresh = SessionState(session_id="s")
    used = SessionState(session_id="s", turn=2, last_shown=[1, 2])
    p = chat.rule_parse("like Project Hail Mary but shorter", fresh, store)
    assert p.intent == "similar_to_with_modifier" and p.anchor_titles == ["Project Hail Mary"]
    assert p.relative_length == "shorter"
    p = chat.rule_parse("more by that author", used, store)
    assert p.intent == "more_by_author" and p.anchor_from_session
    p = chat.rule_parse("something else", used, store)
    assert p.intent == "refine_previous" and p.exclude_previous
    p = chat.rule_parse("why the second one?", used, store)
    assert p.intent == "explain_recommendation" and p.target_index == 2
    p = chat.rule_parse("a bit cozier, and nothing older than 2010", used, store)
    assert p.intent == "refine_previous" and p.descriptor_deltas == {"darkness": -0.6}
    assert p.filters.min_year == 2010
    p = chat.rule_parse("funnier and faster please", used, store)
    assert p.aspect_deltas == {"humor": 0.5} and p.descriptor_deltas == {"pace": 0.6}
    assert chat.rule_parse("what's the weather tomorrow", fresh, store).intent == "out_of_scope"
    p = chat.rule_parse("recommend a dark plot heavy book", fresh, store)
    assert p.intent == "recommend_general" and p.free_text_query
    assert chat.rule_parse("by Andy Weir", fresh, store).anchor_authors == ["Andy Weir"]
    assert chat.rule_parse("Circe", fresh, store).anchor_titles == ["Circe"]
    assert chat.rule_parse("", used, store).intent == "refine_previous"
    # Families from the gold set (D73): filters, mixed opinion, author questions, "vibes".
    p = chat.rule_parse("mystery or thriller only", fresh, store)
    assert p.intent == "filter_by_attribute" and p.filters.genres_include == ["Mystery", "Thriller"]
    p = chat.rule_parse("under 300 pages", used, store)
    assert p.intent == "filter_by_attribute" and p.filters.max_pages == 300
    assert chat.rule_parse("between 2000 and 2010", used, store).filters.max_year == 2010
    p = chat.rule_parse("I loved the characters in Circe but the pacing was slow", fresh, store)
    assert p.intent == "mixed_opinion" and p.anchor_titles == ["Circe"]
    assert p.aspect_deltas == {"characters": 0.5} and p.descriptor_deltas == {"pace": 0.4}
    p = chat.rule_parse("what else did Andy Weir write?", fresh, store)
    assert p.intent == "more_by_author" and p.anchor_authors == ["Andy Weir"]
    p = chat.rule_parse("more by the author of the third one", used, store)
    assert p.intent == "more_by_author" and p.anchor_from_session and p.target_index == 3
    assert chat.rule_parse("give me Mistborn vibes", fresh, store).anchor_titles == ["Mistborn"]
    assert chat.rule_parse("I loved Dune, what next?", fresh, store).intent == "similar_to"
    assert chat.rule_parse("more literary", used, store).intent == "refine_previous"
    assert chat.rule_parse("explain number 3", used, store).target_index == 3
    p = chat.rule_parse("any funny sci-fi?", fresh, store)
    assert p.intent == "recommend_general" and p.filters.genres_include == ["Science Fiction"]
    assert p.aspect_deltas == {"humor": 0.5}


def test_three_turn_demo_with_llm_off(store: Any) -> None:
    cat = store.catalog
    assert core.health()["llm"] is False
    one = core.chat_turn(reader(), None, "something like Project Hail Mary but shorter")
    assert one.intent == "similar_to_with_modifier" and one.method == "template"
    assert one.session.turn == 1 and one.session.anchor_ids == [0]
    assert one.items and 0 not in [i.book_id for i in one.items]
    assert all(cat.pages[i.book_id] <= int(0.8 * 476) for i in one.items)
    labels = {c.label for c in one.chips}
    assert {"like Project Hail Mary", f"under {int(0.8 * 476)} pages"} <= labels
    assert one.session.last_shown == [i.book_id for i in one.items]
    assert all(i.reason and i.evidence.book["book_id"] == i.book_id for i in one.items)

    two = core.chat_turn(reader(), one.session, "more by that author")
    assert two.intent == "more_by_author" and two.session.anchor_authors == ["Andy Weir"]
    assert [i.book_id for i in two.items] == [2, 1]  # Artemis (305 p) before The Martian: pop
    assert {c.label for c in two.chips} >= {"by Andy Weir"}

    three = core.chat_turn(reader(), two.session, "why the second one?")
    assert three.intent == "explain_recommendation" and three.method == "template"
    assert [i.book_id for i in three.items] == [two.session.last_shown[1]]
    assert (
        three.session.last_shown == two.session.last_shown and three.intro == three.items[0].reason
    )
    assert len(three.session.history) == 6


def test_refinements_exclusions_and_rerun(store: Any) -> None:
    first = core.chat_turn(reader(), None, "recommend a gripping book")
    assert first.intent == "recommend_general" and first.items
    shown = list(first.session.last_shown)
    again = core.chat_turn(reader(), first.session, "something else")
    assert again.intent == "refine_previous" and set(again.session.excluded_ids) == set(shown)
    assert not set(i.book_id for i in again.items) & set(shown)
    cozy = core.chat_turn(reader(), again.session, "a bit cozier, and nothing older than 2010")
    assert cozy.session.descriptor_deltas["darkness"] == -0.6  # "gripping" set pace earlier
    assert cozy.session.filters.min_year == 2010
    assert all(store.catalog.year[i.book_id] >= 2010 for i in cozy.items)
    assert {"cozier", "from 2010"} <= {c.label for c in cozy.chips}
    # Chip removal: the backend deletes the field and sends an empty message.
    session = cozy.session.model_copy(deep=True)
    session.descriptor_deltas = {}
    rerun = core.chat_turn(reader(), session, "")
    assert rerun.intent == "refine_previous" and "cozier" not in {c.label for c in rerun.chips}
    assert (
        rerun.session.filters.min_year == 2010
        and rerun.session.last_request == cozy.session.last_request
    )
    off = core.chat_turn(reader(), rerun.session, "what's the weather tomorrow")
    assert off.intent == "out_of_scope" and off.items == [] and off.followup is None


def test_ambiguous_title_asks_then_accepts_the_option(store: Any) -> None:
    ask = core.chat_turn(reader(), None, "like Mistborn")
    assert ask.clarification is not None and ask.items == []
    options = [o.value for o in ask.clarification.options]
    assert "Mistborn: The Final Empire" in options and len(options) <= 3
    picked = core.chat_turn(reader(), ask.session, "Mistborn: The Final Empire")
    assert picked.intent == "similar_to" and picked.session.anchor_ids == [5]
    assert 5 not in [i.book_id for i in picked.items]
    nothing = core.chat_turn(reader(), None, "like Zzz Qqq Nothing")
    assert nothing.intent == "similar_to" and nothing.items == []
    why = core.chat_turn(reader(), None, "why?")
    assert why.intent == "explain_recommendation" and why.items == []


def _intent_json(**fields: Any) -> str:
    base = {
        "intent": "recommend_general", "anchor_titles": [], "anchor_authors": [],
        "target_index": None, "anchor_from_session": False, "aspect_deltas": {},
        "descriptor_deltas": {}, "relative_length": None, "filters": {},
        "exclude_previous": False, "free_text_query": None, "needs_clarification": None,
    }  # fmt: skip
    return json.dumps(base | fields)


def _reply(prompt: str, reorder: bool = False, reason: str = "A grounded, friendly pick.") -> str:
    results = json.loads(prompt_field(prompt, "RESULTS"))
    ids = [r["book_id"] for r in results]
    if reorder:
        ids = ids[::-1]
    return json.dumps({"intro": "Here is a shorter take on that.",
                       "items": [{"book_id": i, "reason": reason} for i in ids],
                       "followup": "Want them cozier?"})  # fmt: skip


def test_llm_intent_and_reply_with_validators(store: Any, tmp_path: Path) -> None:
    intent = _intent_json(intent="similar_to_with_modifier", anchor_titles=["Project Hail Mary"],
                          relative_length="shorter")  # fmt: skip
    adapter = fake_llm(tmp_path, {"intent_parse": [intent], "chat_reply": [_reply]})
    out = core.chat_turn(reader(), None, "something like Project Hail Mary but shorter")
    assert out.method == "llm" and out.intent == "similar_to_with_modifier"
    assert out.intro == "Here is a shorter take on that." and out.followup == "Want them cozier?"
    assert all(i.reason == "A grounded, friendly pick." for i in out.items)
    assert out.session.filters.max_pages == int(0.8 * 476)
    prompt = adapter.calls[0]["prompt"]
    assert "LAST_SHOWN: []" in prompt and "user_id" not in prompt
    assert "RESULTS:" in adapter.calls[1]["prompt"]

    # Extra, missing or reordered ids -> template reply; so does a title or a spoiler in a reason.
    for bad in (
        lambda p: _reply(p, reorder=True),
        lambda p: json.dumps({"intro": "x", "items": [], "followup": "y"}),
        lambda p: _reply(p, reason="Better than Gone Girl in every way."),
        lambda p: _reply(p, reason="The twist is that the narrator dies."),
        "not json",
    ):
        fake_llm(tmp_path, {"intent_parse": [intent], "chat_reply": [bad]})
        out = core.chat_turn(reader(), None, "something like Project Hail Mary but shorter")
        assert out.method == "template" and out.items, bad
        assert out.intro.startswith("Here are books in the spirit of Project Hail Mary")


def test_llm_anchor_not_in_message_is_dropped(store: Any, tmp_path: Path) -> None:
    invented = _intent_json(intent="similar_to", anchor_titles=["Gone Girl"])
    fake_llm(tmp_path, {"intent_parse": [invented], "chat_reply": [_reply]})
    out = core.chat_turn(reader(), None, "something like Project Hail Mary")
    assert out.intent == "similar_to" and out.session.anchor_ids == [0]  # the rule parser won
    disabled = reader()
    disabled.overrides.disabled_ai = True
    out = core.chat_turn(disabled, None, "something like Project Hail Mary")
    assert out.method == "template" and out.session.anchor_ids == [0]


def test_llm_explain_turn_uses_the_explain_path(store: Any, tmp_path: Path) -> None:
    fake_llm(tmp_path, {"intent_parse": [_intent_json(intent="explain_recommendation",
                                                       target_index=1)],
                        "explain": ["A clever, funny read that fits your taste."]})  # fmt: skip
    session = SessionState(session_id="s", turn=1, last_shown=[1, 2])
    out = core.chat_turn(reader(), session, "why the first one?")
    assert out.intent == "explain_recommendation" and out.method == "llm"
    assert out.items[0].book_id == 1 and out.intro == "A clever, funny read that fits your taste."
