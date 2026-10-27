"""Review synthesis helpers (CLAUDE.md s14.8) with the fake provider; the stage itself runs in
``tests/litsense_core/test_bundle.py`` on the tiny pipeline."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.llm.adapters.fake import FakeAdapter
from litsense_core.spoiler import SpoilerScorer
from litsense_core.validators import TitleMatcher
from ml.genai import synthesis
from ml.serve.export import catalog_text, read_summaries

PARAMS = {
    "top_n": 3,
    "praised": {"pos_rate": 0.6, "mention_rate": 0.1},
    "criticised": {"neg_rate": 0.25, "mention_rate": 0.1},
    "snippets": {"positive": 2, "negative": 1},
    "max_words": 70,
}
NAMES = ["plot", "writing_style", "characters", "pacing"]
VARS = {
    "title": "The Salt Orchard",
    "author": "Daniel Okoro",
    "n_reviews": 40,
    "praised": [{"aspect": "characters", "positive_pct": 74, "mention_pct": 30}],
    "criticised": [{"aspect": "pacing", "negative_pct": 30, "mention_pct": 20}],
    "pos_snippets": ["I fell in love with the characters."],
    "neg_snippets": ["The middle dragged on forever."],
}


def client(tmp_path: Path, script: list) -> LLMClient:
    cfg = LLMConfig.from_mapping(
        {"enabled": True, "provider": "fake", "model": "f", "model_quality": "fq",
         "cache_path": ":memory:"}  # one cache per client: the same variables must not hit
    )  # fmt: skip
    return LLMClient(cfg, adapter=FakeAdapter({"review_synthesis": script}), model="fq")


def test_select_books_and_aspect_stats() -> None:
    assert synthesis.select_books(np.array([5, 0, 9, 9, 1]), 3) == [2, 3, 0]
    assert synthesis.select_books(np.array([0, 0]), 3) == []
    mention = np.array([[0.3, 0.05, 0.2, 0.2]])
    pos = np.array([[0.74, 0.9, 0.5, 0.4]])
    neg = np.array([[0.1, 0.0, 0.1, 0.3]])
    praised, criticised = synthesis.aspect_stats(0, NAMES, mention, pos, neg, PARAMS)
    assert praised == [{"aspect": "plot", "positive_pct": 74, "mention_pct": 30}]  # prose: rare
    assert criticised == [{"aspect": "pacing", "negative_pct": 30, "mention_pct": 20}]
    entry = {"plot": {"pos": [{"text": "a", "helpful_votes": 1}, {"text": "b", "helpful_votes": 9}],
                      "neg": [{"text": "c", "helpful_votes": 0}]},
             "pacing": {"pos": [{"text": "d", "helpful_votes": 5}], "neg": []}}  # fmt: skip
    assert synthesis.pick_snippets(entry, 2, 1) == (["b", "d"], ["c"])
    assert synthesis.template_summary(40, praised, criticised) == (
        "Readers most often praise its plot (74% positive); some criticise its pacing"
        " (30% negative)."
    )
    assert (
        synthesis.template_summary(3, [], []) == "Based on 3 reviews, no single aspect stands out."
    )


def test_synthesize_accepts_rejects_and_falls_back(tmp_path: Path) -> None:
    matcher = TitleMatcher(["The Salt Orchard", "Lanterns Over Kesh"])
    spoiler = SpoilerScorer.regex_only([r"\bdies\b", "turns out"], 0.15, 0.05, 0.3)
    good = json.dumps({"summary": "Readers love the characters (74% positive); some find the"
                                  " pacing slow."})  # fmt: skip
    row = synthesis.synthesize(client(tmp_path, [good]), VARS, matcher, spoiler, 70)
    assert (
        row["method"] == "llm"
        and row["summary"].startswith("Readers love")
        and not row["rejections"]
    )
    cases = {
        "title": json.dumps({"summary": "Better than Lanterns Over Kesh."}),
        "number": json.dumps({"summary": "Rated 4.9 by 999 readers."}),
        "spoiler": json.dumps({"summary": "The hero dies at the end."}),
        "length": json.dumps({"summary": "word " * 71}),
    }
    for code, script in cases.items():
        row = synthesis.synthesize(client(tmp_path, [script]), VARS, matcher, spoiler, 70)
        assert row["method"] == "template" and row["rejections"][0].startswith(code), code
        assert row["summary"].startswith("Readers most often praise its characters (74% positive)")
    row = synthesis.synthesize(client(tmp_path, ["not json"]), VARS, matcher, spoiler, 70)
    assert row["method"] == "template" and row["fallback_reason"] == "invalid_json"
    # "the ending" is a weak pattern (D72): a rejected raw answer re-validates without a call.
    raw = "Readers love the characters; some criticise the ending."
    row = synthesis.validate_row(raw, VARS, matcher, spoiler, 70)
    assert row["method"] == "llm" and row["raw"] == raw


def test_summaries_round_trip_into_catalog_text(tmp_path: Path) -> None:
    path = tmp_path / "review_summaries.jsonl"
    path.write_text(
        json.dumps({"i": 1, "summary": "Readers praise its plot.", "method": "llm"})
        + "\n"
        + json.dumps({"i": 2, "summary": "", "method": "template"})
        + "\n",
        encoding="utf-8",
    )
    assert read_summaries(path) == {1: "Readers praise its plot."}
    assert read_summaries(None) == {} and synthesis.read_summaries(path)[2]["summary"] == ""
    books = pd.DataFrame({
        "i": [0, 1, 2], "canonical_id": ["a", "b", "c"], "title": ["A", "B", "C"],
        "author": ["x", None, "z"], "genres": [["Fantasy"], [], ["Mystery"]],
        "image": [None] * 3, "description": ["d", "", "f"], "features": ["", "", ""],
        "isbn13": [None] * 3, "series_name": [None] * 3,
    })  # fmt: skip
    text = catalog_text(books, read_summaries(path))
    assert text["review_summary"] == [None, "Readers praise its plot.", None]
    assert catalog_text(books)["review_summary"] == [None, None, None]
