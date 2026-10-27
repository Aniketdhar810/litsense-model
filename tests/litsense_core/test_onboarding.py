"""Onboarding parse (CLAUDE.md s14.4, s16.2 GenAI): LLM path with the vocabularies enforced,
keyword parse as fallback."""

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

import litsense_core as core
from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.llm.adapters.fake import FakeAdapter
from litsense_core.llm.client import set_client
from litsense_core.store import set_store
from tests.litsense_core.fake_store import build_store


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


TEXT = "I love funny mystery novels like The Thursday Murder Club, nothing too dark."


def test_llm_off_keyword_parse_unchanged(store: Any, tmp_path: Path) -> None:
    set_client(LLMClient(LLMConfig(cache_path=str(tmp_path / "off.db"))))
    out = core.parse_onboarding(TEXT, [3])
    assert out.method == "embed_only" and out.resolved_picks[0] == 3
    assert out.profile.liked_genres == ["Mystery"] and 9 in out.resolved_picks
    assert out.display_chips == ["Likes: Mystery"]
    with pytest.raises(ValueError, match="book_id"):
        core.parse_onboarding(TEXT, [99])


def test_llm_parse_drops_unknown_genres_and_aspects(store: Any, tmp_path: Path) -> None:
    raw = {
        "liked_genres": ["mystery", "Cookbooks"],
        "disliked_genres": ["Horror", "Mystery"],
        "positive_aspects": ["humor", "vibes", "plot"],
        "negative_aspects": ["tone_dark", "plot"],
        "descriptors": {"darkness": -3, "pace": 7, "mood": 1},
        "themes": [f"theme {k}" for k in range(8)],
        "named_titles": ["The Thursday Murder Club", "Not A Real Book Title"],
        "named_authors": ["Richard Osman", "Nobody Known"],
    }
    adapter = fake_llm(tmp_path, {"onboarding_parse": [json.dumps(raw)]})
    out = core.parse_onboarding(TEXT, [3])
    assert out.method == "llm"
    p = out.profile
    assert p.liked_genres == ["Mystery"] and p.disliked_genres == []  # Horror not in catalogue
    assert p.positive_aspects == ["humor", "plot"] and p.negative_aspects == ["tone_dark"]
    assert p.descriptors == {"darkness": -1.0, "pace": 1.0} and len(p.themes) == 6
    assert p.named_titles == ["The Thursday Murder Club"] and p.named_authors == ["Richard Osman"]
    assert out.resolved_picks == [3, 9]
    assert {"Likes: Mystery", "Values: humor", "Avoids: tone", "Prefers: cozy"} <= set(
        out.display_chips
    )
    assert "GENRES: " in adapter.calls[0]["prompt"] and TEXT in adapter.calls[0]["prompt"]


def test_llm_failure_falls_back_to_keywords(store: Any, tmp_path: Path) -> None:
    fake_llm(tmp_path, {"onboarding_parse": ["not json at all"]})
    out = core.parse_onboarding(TEXT, [])
    assert out.method == "embed_only" and out.profile.liked_genres == ["Mystery"]
    fake_llm(tmp_path, {"onboarding_parse": ['{"liked_genres": ["Mystery"]}']})
    assert core.parse_onboarding("", []).method == "embed_only"  # nothing to parse
    assert core.parse_onboarding(TEXT, []).method == "llm"
