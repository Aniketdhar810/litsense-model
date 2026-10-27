"""LLM client (CLAUDE.md s14.2, s16.2 GenAI): fake provider only, never the network."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from litsense_core.llm import LLMClient, LLMConfig, PrivacyError
from litsense_core.llm.adapters import RateLimited, TimedOut
from litsense_core.llm.adapters.fake import FakeAdapter
from litsense_core.llm.budget import RateLimiter
from litsense_core.llm.client import PROMPT_DIR, load_prompts, strip_fences
from litsense_core.llm.config import find_config


class Summary(BaseModel):
    summary: str


_COUNTER = [0]


def make_client(
    tmp_path: Path, script: dict[str, list[Any]] | None = None, **over: Any
) -> tuple[LLMClient, FakeAdapter, list[dict[str, Any]]]:
    """A fake-provider client with its own cache file (the daily budget lives in it)."""
    events: list[dict[str, Any]] = []
    _COUNTER[0] += 1
    cfg = LLMConfig.from_mapping(
        {
            "enabled": True,
            "provider": "fake",
            "model": "fake-1",
            "cache_path": str(tmp_path / f"c{_COUNTER[0]}.db"),
            "daily_request_budget": 5,
            "rpm_limit": 60,
            **over,
        }
    )
    adapter = FakeAdapter(script)
    sleeps: list[float] = []
    client = LLMClient(cfg, events.append, adapter, sleep=sleeps.append)
    client.sleeps = sleeps  # type: ignore[attr-defined]
    return client, adapter, events


VARS = {"title": "A Book", "author": "Someone", "n_reviews": 12, "praised": "plot",
        "criticised": "none", "pos_snippets": "[]", "neg_snippets": "[]"}  # fmt: skip


def test_prompt_files_load_with_front_matter_and_render_json_braces() -> None:
    prompts = load_prompts()
    assert set(prompts) == {"onboarding_parse", "intent_parse", "explain", "chat_reply",
                            "review_synthesis"}  # fmt: skip
    assert all(p.version == 1 for p in prompts.values())
    assert {p.role for p in prompts.values()} == {"parse", "explain", "chat", "synthesis"}
    text = prompts["explain"].render({"evidence_json": '{"book": {"title": "X"}}'})
    assert 'EVIDENCE:\n{"book": {"title": "X"}}' in text and "$evidence_json" not in text
    assert (PROMPT_DIR / "explain.v1.md").exists()


def test_strip_fences_and_prose() -> None:
    assert strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_fences('Sure! {"a": 1} hope this helps') == '{"a": 1}'
    assert strip_fences('{"a": 1}') == '{"a": 1}'


def test_config_defaults_off_and_yaml_block_reads(tmp_path: Path) -> None:
    assert LLMConfig().enabled is False and LLMConfig().model == "SET-ME"
    cfg = find_config(None)  # the repo's ml/config.yaml
    assert cfg.api_key_env == "GEMINI_API_KEY" and cfg.explain_max_words == 60
    cfg = LLMConfig.from_mapping({"temperature": {"chat": 0.9}})
    assert cfg.temperature["chat"] == 0.9 and cfg.temperature["parse"] == 0.0


def test_disabled_set_me_and_missing_key_never_call_the_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITSENSE_TEST_KEY", raising=False)
    for over, reason in (
        ({"enabled": False}, "disabled"),
        ({"provider": "gemini", "model": "SET-ME"}, "not_configured"),
        ({"provider": "gemini", "model": "gemini-x", "api_key_env": "LITSENSE_TEST_KEY"},
         "no_api_key"),
    ):  # fmt: skip
        client, adapter, events = make_client(tmp_path, **over)
        result = client.text("explain", {"evidence_json": "{}"}, "explain")
        assert result.text is None and result.fallback_reason == reason and not result.ok
        assert adapter.calls == [] and events[-1]["fallback_reason"] == reason
        assert client.available is False and client.status()["reason"] == reason


def test_fake_provider_round_trip_cache_hit_and_ttl(tmp_path: Path) -> None:
    client, adapter, events = make_client(tmp_path, {"review_synthesis": ['{"summary": "Good."}']})
    first = client.json("review_synthesis", VARS, Summary, "synthesis")
    assert first.ok and first.data.summary == "Good." and first.cached is False
    second = client.json("review_synthesis", VARS, Summary, "synthesis")
    assert second.cached is True and second.data.summary == "Good." and len(adapter.calls) == 1
    assert client.budget.used() == 1  # the cached call cost nothing
    # Different variables, model or params -> different key
    other = client.json("review_synthesis", {**VARS, "title": "B"}, Summary, "synthesis")
    assert other.cached is False and len(adapter.calls) == 2
    key = client.cache_key(client.prompts["review_synthesis"], VARS, 0.3, 200)
    assert key != client.cache_key(client.prompts["review_synthesis"], VARS, 0.4, 200)
    client.cache.set(key, "review_synthesis", '{"summary": "Old."}', now=0.0)  # expired
    stale = client.json("review_synthesis", VARS, Summary, "synthesis")
    assert stale.cached is False and len(adapter.calls) == 3
    assert events[0]["cached"] is False and events[1]["cached"] is True


def test_daily_budget_exhaustion_falls_back(tmp_path: Path) -> None:
    client, adapter, _ = make_client(tmp_path, daily_request_budget=2)
    for k in range(2):
        assert client.text("explain", {"evidence_json": str(k)}, "explain").ok
    out = client.text("explain", {"evidence_json": "3"}, "explain")
    assert out.fallback_reason == "budget" and len(adapter.calls) == 2
    assert client.status()["budget_used_today"] == 2


def test_429_backoff_then_clean_fallback(tmp_path: Path) -> None:
    script = {"explain": [RateLimited("429"), RateLimited("429"), RateLimited("429")]}
    client, adapter, _ = make_client(tmp_path, script, max_retries=2)
    out = client.text("explain", {"evidence_json": "x"}, "explain")
    assert out.fallback_reason == "rate_limited" and out.text is None and out.attempts == 3
    assert len(client.sleeps) == 3 and all(1 <= s <= 9 for s in client.sleeps)  # type: ignore[attr-defined]
    recovered = {"explain": [RateLimited("429"), "Fine after a retry."]}
    client, adapter, _ = make_client(tmp_path, recovered, max_retries=2)
    out = client.text("explain", {"evidence_json": "y"}, "explain")
    assert out.ok and out.text == "Fine after a retry." and out.attempts == 2


def test_timeout_and_bad_json_fall_back(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, {"explain": [TimedOut("slow")]})
    assert client.text("explain", {"evidence_json": "x"}, "explain").fallback_reason == "timeout"
    bad = {"review_synthesis": ["not json", '```json\n{"summary": 5}\n```', '{"nope": 1}']}
    client, adapter, _ = make_client(tmp_path, bad)
    out = client.json("review_synthesis", VARS, Summary, "synthesis")
    assert out.fallback_reason == "invalid_json" and out.data is None and out.attempts == 3
    assert "not valid JSON" in adapter.calls[1]["prompt"]  # the repair retry carries the error
    repaired = {"review_synthesis": ["oops", '```json\n{"summary": "fixed"}\n```']}
    client, _, _ = make_client(tmp_path, repaired)
    out = client.json("review_synthesis", VARS, Summary, "synthesis")
    assert out.ok and out.data.summary == "fixed" and json.loads(out.text)["summary"] == "fixed"


def test_privacy_guard_raises_on_ids_and_emails(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path)
    with pytest.raises(PrivacyError):
        client.text("explain", {"evidence_json": "x", "user_id": 5}, "explain")
    with pytest.raises(PrivacyError):
        client.text("explain", {"evidence_json": {"a": ["mail me at zee@example.com"]}}, "explain")
    assert client.text("explain", {"evidence_json": "plain"}, "explain").ok


def test_rate_limiter_non_blocking_and_blocking() -> None:
    now = [0.0]
    slept: list[float] = []

    def sleep(s: float) -> None:
        slept.append(s)
        now[0] += s

    limiter = RateLimiter(2, clock=lambda: now[0], sleep=sleep)
    assert limiter.acquire() and limiter.acquire() and not limiter.acquire()
    now[0] += 30.0
    assert limiter.acquire() and not limiter.acquire()
    assert limiter.acquire(block=True) and slept and slept[0] == pytest.approx(30.0)


def test_limiter_exhausted_reports_rate_limited_without_sleeping(tmp_path: Path) -> None:
    client, adapter, _ = make_client(tmp_path, rpm_limit=1)
    assert client.text("explain", {"evidence_json": "1"}, "explain").ok
    out = client.text("explain", {"evidence_json": "2"}, "explain")
    assert out.fallback_reason == "rate_limited" and len(adapter.calls) == 1
    assert client.sleeps == []  # type: ignore[attr-defined]


def test_openai_compat_stub_degrades_cleanly(tmp_path: Path) -> None:
    client, _, _ = make_client(tmp_path, provider="openai_compat", model="llama",
                               api_key_env="PATH")  # fmt: skip
    client._adapter = None
    out = client.text("explain", {"evidence_json": "x"}, "explain")
    assert out.fallback_reason == "provider_error" and "not implemented" in client._adapter_error
