"""The labellers' intent sheet and its conversion to the gold CSV (CLAUDE.md s14.9)."""

import csv
import json
from pathlib import Path

import pytest

from ml.genai import eval_intent

ROWS = [
    {"id": "c01", "message": "something like Project Hail Mary but shorter", "keep": "yes",
     "intent": "like + change", "book_named": "Project Hail Mary", "shorter_or_longer": "shorter"},
    {"id": "c02", "message": "more by that author", "keep": "yes", "intent": "more by author",
     "that_one_or_author": "yes", "shown_before": "The Martian; Artemis"},
    {"id": "c03", "message": "why the second one?", "keep": "yes", "intent": "why",
     "which_one": "second", "shown_before": "The Martian; Artemis"},
    {"id": "c04", "message": "a bit cozier, and nothing older than 2010", "keep": "yes",
     "intent": "refine", "mood": "cozier, funnier", "year_from": "2010",
     "shown_before": "Gone Girl"},
    {"id": "c05", "message": "something else", "keep": "yes", "intent": "refine",
     "something_else": "yes", "shown_before": "Gone Girl"},
    {"id": "c06", "message": "what's the weather tomorrow", "keep": "yes",
     "intent": "not about books"},
    {"id": "c07", "message": "only fantasy, no horror", "keep": "", "intent": "filter",
     "only_genre": "Fantasy", "no_genre": "Horror"},
]  # fmt: skip


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=eval_intent.SHEET_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in eval_intent.SHEET_COLUMNS})


def test_sheet_converts_to_gold_rows(tmp_path: Path) -> None:
    sheet, out = tmp_path / "sheet.csv", tmp_path / "gold.csv"
    write_sheet(sheet, ROWS)
    assert eval_intent.sheet_to_gold(sheet, out) == 6  # c07 has keep empty
    gold = {r["id"]: r for r in eval_intent.load_gold(out)}
    assert gold["c01"]["gold"] == {"intent": "similar_to_with_modifier",
                                   "anchor_titles": ["Project Hail Mary"],
                                   "anchor_resolved": "Project Hail Mary",
                                   "relative_length": "shorter"}  # fmt: skip
    assert gold["c02"]["gold"]["anchor_from_session"] is True
    assert gold["c02"]["session"] == {"turn": 1, "last_shown_titles": ["The Martian", "Artemis"],
                                      "anchor_titles": ["The Martian"]}  # fmt: skip
    assert gold["c03"]["gold"]["target_index"] == 2
    assert gold["c04"]["gold"]["descriptor_deltas"] == {"darkness": -0.5}
    assert gold["c04"]["gold"]["aspect_deltas"] == {"humor": 0.5}
    assert gold["c04"]["gold"]["filters"] == {"min_year": 2010}
    assert gold["c05"]["gold"]["exclude_previous"] is True
    assert gold["c06"]["gold"] == {"intent": "out_of_scope"} and gold["c06"]["session"] == {}
    # The example gold file in docs and the sheet route agree on these six rows.
    example = {
        r["id"]: r
        for r in eval_intent.load_gold(
            Path(__file__).resolve().parents[3] / "docs" / "annotation" / "intent_example.csv"
        )
    }
    for sheet_id, ex_id in (("c01", "e01"), ("c03", "e03"), ("c05", "e05"), ("c06", "e06")):
        assert gold[sheet_id]["gold"] == example[ex_id]["gold"], sheet_id


def test_sheet_keeps_every_row_when_keep_is_empty_and_reports_bad_words(tmp_path: Path) -> None:
    sheet, out = tmp_path / "sheet.csv", tmp_path / "gold.csv"
    rows = [dict(r, keep="") for r in ROWS]
    write_sheet(sheet, rows)
    assert eval_intent.sheet_to_gold(sheet, out) == 7
    c07 = {r["id"]: r for r in eval_intent.load_gold(out)}["c07"]
    assert c07["gold"]["filters"] == {"genres_include": ["Fantasy"], "genres_exclude": ["Horror"]}
    write_sheet(sheet, [dict(ROWS[0], intent="sumthing")])
    with pytest.raises(ValueError, match="unknown intent"):
        eval_intent.sheet_to_gold(sheet, out)
    write_sheet(sheet, [dict(ROWS[3], mood="cozier, spicy")])
    with pytest.raises(ValueError, match="unknown mood"):
        eval_intent.sheet_to_gold(sheet, out)


def test_candidates_sheet_has_the_plain_columns(tmp_path: Path) -> None:
    path = tmp_path / "cand.csv"
    n = eval_intent.write_candidates(path)
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n and list(rows[0]) == list(eval_intent.SHEET_COLUMNS)
    assert all(r["intent"] in eval_intent.INTENT_WORDS for r in rows)
    assert json.dumps(rows[0]["message"])  # plain text, no JSON asked of the labeller
