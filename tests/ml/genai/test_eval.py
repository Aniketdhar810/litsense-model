"""Intent scorer and explanation audit (CLAUDE.md s14.9) on the hand-built store, fake provider."""

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.llm.adapters.fake import FakeAdapter
from litsense_core.llm.client import set_client
from litsense_core.schemas import RatingEvent, UserState
from litsense_core.store import set_store
from ml.common.config import REPO_ROOT
from ml.genai import audit, eval_intent
from tests.litsense_core.fake_store import build_store

TS = 1_650_000_000_000


@pytest.fixture
def store(tmp_path: Path) -> Any:
    st = build_store(tmp_path / "bundle")
    set_store(st)
    set_client(LLMClient(LLMConfig(cache_path=":memory:")))
    yield st
    set_store(None)
    set_client(None)


def test_example_gold_set_scores_the_rule_parser(store: Any) -> None:
    rows = eval_intent.load_gold(REPO_ROOT / "docs" / "annotation" / "intent_example.csv")
    assert len(rows) == 6 and rows[1]["session"]["last_shown_titles"] == ["The Martian", "Artemis"]
    result = eval_intent.evaluate(rows, store, "rule")
    assert result["rows"] == 6 and result["intent_accuracy"] == 1.0
    assert result["anchor_resolution_accuracy"] == 1.0 and result["filter_exact_match"] == 1.0
    assert result["field_accuracy"] == 1.0 and result["llm_fallbacks"] == 0
    by_id = {r["id"]: r for r in result["per_row"]}
    assert by_id["e02"]["predicted_intent"] == "more_by_author"
    assert by_id["e04"]["ok_descriptor_signs"] and by_id["e05"]["ok_exclude_previous"]


def test_llm_parser_scored_with_fallback_counted(store: Any, tmp_path: Path) -> None:
    rows = eval_intent.load_gold(REPO_ROOT / "docs" / "annotation" / "intent_example.csv")
    base = {"anchor_titles": [], "anchor_authors": [], "target_index": None,
            "anchor_from_session": False, "aspect_deltas": {}, "descriptor_deltas": {},
            "relative_length": None, "filters": {}, "exclude_previous": False,
            "free_text_query": None, "needs_clarification": None}  # fmt: skip
    cfg = LLMConfig.from_mapping({"enabled": True, "provider": "fake", "model": "f",
                                  "cache_path": ":memory:"})  # fmt: skip
    # Every LLM answer says recommend_general: scored as given, no fallback.
    wrong = json.dumps(base | {"intent": "recommend_general", "free_text_query": "x"})
    set_client(LLMClient(cfg, adapter=FakeAdapter({"intent_parse": [wrong]})))
    result = eval_intent.evaluate(rows, store, "llm")
    assert result["intent_accuracy"] == 0.0 and result["llm_fallbacks"] == 0
    assert 0.0 <= result["field_accuracy"] < 1.0
    # Every LLM answer invents an anchor the message never named: the guard sends each row to
    # the rule parser, which gets the example set right.
    invented = json.dumps(base | {"intent": "similar_to", "anchor_titles": ["Gone Girl"]})
    set_client(LLMClient(cfg, adapter=FakeAdapter({"intent_parse": [invented]})))
    result = eval_intent.evaluate(rows, store, "llm")
    assert result["llm_fallbacks"] == 6 and result["intent_accuracy"] == 1.0


def test_candidates_file_and_gold_format(tmp_path: Path) -> None:
    path = tmp_path / "cand.csv"
    assert eval_intent.write_candidates(path) == len(eval_intent.CANDIDATES) >= 70
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    suggested = {eval_intent.INTENT_WORDS[r["intent"]] for r in rows}
    assert suggested == {
        "recommend_general", "similar_to", "similar_to_with_modifier", "more_by_author",
        "refine_previous", "explain_recommendation", "filter_by_attribute", "mixed_opinion",
        "out_of_scope",
    }  # fmt: skip
    assert eval_intent.load_gold(path) == []  # the sheet is not a gold file: nothing to score


def test_fleiss_kappa_values() -> None:
    assert audit.fleiss_kappa(np.array([[3, 0], [0, 3], [3, 0]])) == pytest.approx(1.0)
    assert audit.fleiss_kappa(np.array([[3, 0], [2, 1]])) == pytest.approx(-0.2)
    assert np.isnan(audit.fleiss_kappa(np.array([[3, 0], [3, 0]])))
    with pytest.raises(ValueError):
        audit.fleiss_kappa(np.array([[3, 0], [2, 0]]))


def test_export_and_score_audit(store: Any, tmp_path: Path) -> None:
    states = [UserState(ratings=[RatingEvent(book_id=b, stars=5.0, ts=TS)]) for b in (0, 3, 8)]
    cfg = LLMConfig.from_mapping({"enabled": True, "provider": "fake", "model": "f",
                                  "cache_path": ":memory:"})  # fmt: skip
    script = {"explain": ["A good fit for you.", "It turns out the hero dies.", "Fits your taste.",
                          "Twist: she died.", "Nicely done.", "He was killed."]}  # fmt: skip
    set_client(LLMClient(cfg, adapter=FakeAdapter(script)))
    rows = audit.collect_explanations(store, states, 9, k=3)
    assert len(rows) == 9 and sum(r["method"] == "llm" for r in rows) == 3  # last answer repeats
    chosen = audit.stratify(rows, 6)
    assert len(chosen) == 6 and sum(r["method"] == "llm" for r in chosen) == 3
    out = tmp_path / "explain_audit.csv"
    assert audit.write_audit_csv(chosen, out) == 6
    with out.open(encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == list(audit.COLUMNS)
    # Three raters: two agree everything is grounded, one dissents on the first row.
    filled = []
    for k in range(3):
        rows_k = []
        with out.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                r["grounded"] = "N" if (k == 2 and r["id"] == "a001") else "Y"
                r["hallucinated_title"] = "N"
                r["spoiler"] = "N"
                r["useful"] = str(3 + k)
                rows_k.append(r)
        path = tmp_path / f"rater{k}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=audit.COLUMNS)
            w.writeheader()
            w.writerows(rows_k)
        filled.append(path)
    report = audit.score_audit(filled)
    assert report["rows"] == 6 and report["raters"] == 3
    assert report["by_method"]["llm"]["grounded_rate"] == 1.0  # majority vote
    assert report["by_method"]["template"]["spoiler_rate"] == 0.0
    assert report["by_method"]["llm"]["useful_mean"] == 4.0
    assert report["kappa"]["spoiler"] is None  # total agreement: chance agreement is total
    assert -1.0 <= report["kappa"]["grounded"] <= 1.0
    rates = audit.rejection_rates(store, states, 9, k=3)
    assert rates["explanations"] == 9 and rates["by_type"]["spoiler"]["count"] >= 1
    assert rates["rejection_rate"] == pytest.approx(rates["rejected"] / 9, abs=1e-4)
