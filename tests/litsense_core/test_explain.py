"""Explanation path and validators (CLAUDE.md s14.6, s16.2 GenAI), fake provider only."""

import itertools
from pathlib import Path
from typing import Any

import pytest

import litsense_core as core
from litsense_core import validators
from litsense_core.explain import llm_reason
from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.llm.adapters.fake import FakeAdapter
from litsense_core.llm.client import set_client
from litsense_core.reasons import build_evidence
from litsense_core.schemas import RatingEvent, UserState
from litsense_core.store import set_store
from tests.litsense_core.fake_store import build_store

TS = 1_650_000_000_000


@pytest.fixture
def store(tmp_path: Path) -> Any:
    st = build_store(tmp_path / "bundle")
    set_store(st)
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


def test_llm_off_keeps_the_template_byte_identical(store: Any, tmp_path: Path) -> None:
    set_client(LLMClient(LLMConfig(cache_path=str(tmp_path / "off.db"))))
    assert core.health()["llm"] is False
    before = core.explain(reader(), 1)
    assert before.method == "template" and before.rejections == []
    fake_llm(tmp_path, {"explain": ["Anything"]}, enabled=False)
    assert core.explain(reader(), 1) == before
    fake_llm(tmp_path, {"explain": ["Anything"]}, provider="gemini", model="SET-ME")
    assert core.explain(reader(), 1) == before and core.health()["llm"] is False


def test_llm_explanation_passes_and_is_cached(store: Any, tmp_path: Path) -> None:
    adapter = fake_llm(
        tmp_path,
        {
            "explain": [
                "Like The Martian, it mixes problem-solving science with humor.",
                "A clever, funny read that fits your taste.",
            ]
        },
    )
    assert core.health()["llm"] is True
    out = core.explain(reader(), 1)  # The Martian is similar to the liked Project Hail Mary
    assert out.method == "llm" and out.rejections == [] and "The Martian" in out.text
    assert core.explain(reader(), 1).text == out.text and len(adapter.calls) == 1
    assert "EVIDENCE:" in adapter.calls[0]["prompt"] and "user_id" not in adapter.calls[0]["prompt"]
    assert "Project Hail Mary" in adapter.calls[0]["prompt"]  # the similar liked title
    # The main feed never calls the LLM, whatever the client state (s14).
    feed = core.recommend(reader(), k=3, use_ranker=False)
    assert all(i.reason_method == "template" for i in feed.items) and len(adapter.calls) == 1


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("You will love Gone Girl after Project Hail Mary.", "title:Gone Girl"),
        ("Readers praise its plot: 4321 of 5000 mentions are positive.", "number:4321"),
        ("A clever story. It turns out the captain dies at the end.", "spoiler"),
        ("word " * 61, "length"),
        ("- great plot\n- funny", "format"),
        ("See https://example.com for more.", "format"),
        ('Reviewers say "this book changed my whole life forever" and more.', "quote"),
        ("", "empty"),
    ],
)
def test_validators_reject(store: Any, text: str, code: str) -> None:
    ev = build_evidence(1, _profile(store), store, ["explain"])
    problems = validators.check(text, ev.allowed_titles(), ev, 60, store)
    assert code in problems, problems


def test_validators_accept_grounded_text(store: Any) -> None:
    ev = build_evidence(0, _profile(store), store, ["explain"])
    snippet = ev.item_evidence["snippet"]["text"]
    pct = int(round(100 * ev.item_evidence["pos_rates"]["plot"]))
    text = f'Readers praise its plot ({pct}% positive): "{snippet}" Suits a Project Hail Mary fan.'
    assert validators.check(text, ev.allowed_titles(), ev, 60, store) == []
    assert "Project Hail Mary" in validators.title_matcher(store).mentioned(text)
    matcher = validators.title_matcher(store)
    assert matcher.mentioned("I read the song of achilles and it.") == ["The Song of Achilles"]
    assert matcher.mentioned("Dune, Artemis and The Martian: one word after the article.") == []
    # Weak spoiler patterns do not fire on generated text; strong ones still do.
    assert validators.spoiler_sentences("Some criticise the ending.", store.spoiler) == []
    assert validators.spoiler_sentences("In the end the killer is her twin.", store.spoiler)


def test_rejected_explanation_falls_back_to_template(store: Any, tmp_path: Path) -> None:
    adapter = fake_llm(tmp_path, {"explain": ["Twist: the narrator died the whole time."]})
    out = core.explain(reader(), 1)
    assert out.method == "template" and out.rejections == ["spoiler"]
    assert len(adapter.calls) == 1
    # A reader with AI disabled never reaches the LLM.
    state = reader()
    state.overrides.disabled_ai = True
    assert core.explain(state, 1).method == "template" and len(adapter.calls) == 1


def test_provider_failure_is_degraded_but_valid(store: Any, tmp_path: Path) -> None:
    from litsense_core.llm.adapters import TimedOut

    fake_llm(tmp_path, {"explain": [TimedOut("slow")]})
    out = core.explain(reader(), 1)
    assert out.method == "template" and out.rejections == []
    ev = build_evidence(1, _profile(store), store, ["explain"])
    res = llm_reason(ev, store)
    assert res.text is None and res.fallback_reason == "timeout" and res.failed


def test_privacy_guard_in_evidence_degrades_cleanly(store: Any, tmp_path: Path) -> None:
    fake_llm(tmp_path, {"explain": ["Fine."]})
    ev = build_evidence(1, _profile(store), store, ["explain"])
    ev.item_evidence["snippet"] = {"text": "write to me at a@b.co", "review_ref": 1,
                                  "helpful_votes": 1, "aspect": "plot"}  # fmt: skip
    res = llm_reason(ev, store)
    assert res.text is None and res.rejections == ["privacy"]


def _profile(store: Any) -> Any:
    from litsense_core.profile import build_profile

    return build_profile(reader(), store)
