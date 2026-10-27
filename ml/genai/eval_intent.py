"""Intent-parser evaluation (CLAUDE.md s14.9), the gold-set format and the scorer.

Gold set: ``docs/annotation/intent_50.csv`` written by hand (never by the LLM), one row per
message. Columns:

* ``id``: a short label.
* ``message``: the chat message exactly as a reader would type it.
* ``session_json``: optional JSON for the session the message arrives in: ``{"turn": 2,
  "anchor_titles": [...], "anchor_authors": [...], "last_shown_titles": [...]}``. Titles are
  resolved against the bundle when scoring (ids differ between bundles). Empty = a fresh session.
* ``gold_json``: the expected parse: ``intent`` (one of the nine), ``anchor_titles``,
  ``anchor_authors``, ``anchor_resolved`` (the catalogue title or author the anchor should resolve
  to after ``entities``), ``target_index``, ``anchor_from_session``, ``aspect_deltas``,
  ``descriptor_deltas`` (only the sign matters), ``relative_length``, ``filters``,
  ``exclude_previous``. Missing keys mean "none".

Metrics (both parsers are scored, ``rule`` always, ``llm`` when the client is available):
intent accuracy; anchor resolution accuracy (after ``entities``); filter exact match; sign
accuracy of aspect and descriptor deltas; and the overall field accuracy over the eight fields
(target >= 0.8). ``docs/annotation/intent_example.csv`` shows the format on six rows;
``--draft`` writes ~70 candidate messages for the user to pick 50 from and label by hand.

    python -m ml.genai.eval_intent --bundle artifacts/serve_bundle \
        --gold docs/annotation/intent_50.csv --out results/v0/genai/intent_eval.json
    python -m ml.genai.eval_intent --draft docs/annotation/intent_candidates.csv
"""

import argparse
import csv
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import litsense_core as core
from litsense_core import chat, entities
from litsense_core.chat import IntentParse
from litsense_core.schemas import Filters, SessionState
from litsense_core.store import ArtifactStore

FIELDS = (
    "intent",
    "anchor",
    "filters",
    "aspect_signs",
    "descriptor_signs",
    "relative_length",
    "exclude_previous",
    "target_index",
)
CANDIDATES = [
    ("recommend_general", "recommend me a good book"),
    ("recommend_general", "something gripping for a long flight"),
    ("recommend_general", "a cozy mystery for the weekend"),
    ("recommend_general", "I want a dark literary novel"),
    ("recommend_general", "any funny sci-fi?"),
    ("recommend_general", "books with found family and slow burn romance"),
    ("recommend_general", "a short classic I can finish in a day"),
    ("recommend_general", "recent thrillers with a twist"),
    ("similar_to", "something like Project Hail Mary"),
    ("similar_to", "books similar to Circe"),
    ("similar_to", "like The Martian"),
    ("similar_to", "more books in the vein of Gone Girl"),
    ("similar_to", "I loved Dune, what next?"),
    ("similar_to", "anything close to The Thursday Murder Club"),
    ("similar_to", "give me Mistborn vibes"),
    ("similar_to_with_modifier", "something like Project Hail Mary but shorter"),
    ("similar_to_with_modifier", "like Circe but faster paced"),
    ("similar_to_with_modifier", "similar to Gone Girl but less bleak"),
    ("similar_to_with_modifier", "like Dune but easier to read"),
    ("similar_to_with_modifier", "The Martian but funnier"),
    ("similar_to_with_modifier", "like Mistborn but published after 2015"),
    ("similar_to_with_modifier", "something like Artemis, longer though"),
    ("more_by_author", "more by that author"),
    ("more_by_author", "what else did Andy Weir write?"),
    ("more_by_author", "by Madeline Miller"),
    ("more_by_author", "other books from the same author"),
    ("more_by_author", "more from Brandon Sanderson"),
    ("more_by_author", "show me the author of the first one's other books"),
    ("refine_previous", "a bit cozier"),
    ("refine_previous", "something else"),
    ("refine_previous", "not these, something darker"),
    ("refine_previous", "faster paced please"),
    ("refine_previous", "nothing older than 2010"),
    ("refine_previous", "shorter ones"),
    ("refine_previous", "more literary"),
    ("refine_previous", "less complicated"),
    ("refine_previous", "funnier"),
    ("refine_previous", "can you make them more recent and cozier?"),
    ("explain_recommendation", "why the second one?"),
    ("explain_recommendation", "why did you suggest that?"),
    ("explain_recommendation", "what makes the first one a fit for me"),
    ("explain_recommendation", "explain number 3"),
    ("explain_recommendation", "why?"),
    ("filter_by_attribute", "only fantasy"),
    ("filter_by_attribute", "no horror please"),
    ("filter_by_attribute", "under 300 pages"),
    ("filter_by_attribute", "published before 1980"),
    ("filter_by_attribute", "between 2000 and 2010"),
    ("filter_by_attribute", "mystery or thriller only"),
    ("filter_by_attribute", "nothing longer than 400 pages"),
    ("mixed_opinion", "I loved the characters in Circe but the pacing was slow"),
    ("mixed_opinion", "Dune had amazing world-building but I found it dense"),
    ("mixed_opinion", "the humor in The Martian worked, the ending did not"),
    ("mixed_opinion", "Gone Girl was too dark for me but the plot was clever"),
    ("mixed_opinion", "Project Hail Mary: great science, flat dialogue"),
    ("mixed_opinion", "liked the prose of The Song of Achilles, not the ending"),
    ("out_of_scope", "what's the weather tomorrow"),
    ("out_of_scope", "write my essay for me"),
    ("out_of_scope", "who won the match last night"),
    ("out_of_scope", "tell me a joke"),
    ("out_of_scope", "how do I reset my password"),
    ("out_of_scope", "translate this to French"),
    ("recommend_general", "I read mostly thrillers, surprise me"),
    ("recommend_general", "a doorstopper fantasy with great world-building"),
    ("similar_to", "that one again but by someone else"),
    ("refine_previous", "darker and more recent"),
    ("refine_previous", "ok but longer"),
    ("more_by_author", "more by the author of the third one"),
    ("filter_by_attribute", "exclude science fiction"),
    ("explain_recommendation", "why the last one?"),
]


def load_gold(path: Path) -> list[dict[str, Any]]:
    """Rows of the gold CSV with ``session`` and ``gold`` parsed from their JSON columns."""
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            if not raw.get("message", "").strip() or not raw.get("gold_json", "").strip():
                continue  # unlabelled rows (the candidates file) are not scored
            rows.append(
                {
                    "id": raw.get("id") or str(len(rows) + 1),
                    "message": raw["message"].strip(),
                    "session": json.loads(raw["session_json"])
                    if raw.get("session_json", "").strip()
                    else {},
                    "gold": json.loads(raw["gold_json"]),
                }
            )
    return rows


def session_from(spec: Mapping[str, Any], store: ArtifactStore) -> SessionState:
    """A ``SessionState`` from the gold row's session JSON (titles resolved on this bundle)."""

    def ids(titles: list[str]) -> list[int]:
        out = []
        for t in titles:
            res = entities.resolve_title(t, store)
            if res.kind == "title":
                out.append(res.book_ids[0])
        return out

    return SessionState(
        session_id="gold",
        turn=int(spec.get("turn", 0)),
        anchor_ids=ids(spec.get("anchor_titles", [])),
        anchor_authors=list(spec.get("anchor_authors", [])),
        last_shown=ids(spec.get("last_shown_titles", [])),
        aspect_deltas=dict(spec.get("aspect_deltas", {})),
        descriptor_deltas=dict(spec.get("descriptor_deltas", {})),
        filters=Filters.model_validate(spec.get("filters", {})),
    )


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def _filters(f: Mapping[str, Any] | Filters) -> dict[str, Any]:
    raw = f.model_dump() if isinstance(f, Filters) else dict(f)
    return {k: v for k, v in raw.items() if v not in (None, [], {})}


def resolved_anchor(parsed: IntentParse, session: SessionState, store: ArtifactStore) -> str | None:
    """The catalogue title or author the parse's anchor resolves to (``entities``), or None.
    ``more_by_author`` resolves to an author (the author of an anchored book counts);
    ``explain_recommendation`` has no anchor (its target is scored as ``target_index``)."""
    if parsed.intent == "explain_recommendation":
        return None
    ids, authors, _options, _logs = chat._resolve(parsed, session, store)
    if parsed.intent == "more_by_author":
        if authors:
            return authors[0]
        return store.catalog.author[ids[0]] if ids else None
    if ids:
        return store.catalog.title[ids[0]]
    return authors[0] if authors else None


def score_row(
    parsed: IntentParse, gold: Mapping[str, Any], session: SessionState, store: ArtifactStore
) -> dict[str, bool]:
    """Per-field correctness of one parse against its gold row."""
    gold_anchor = gold.get("anchor_resolved")
    predicted_anchor = resolved_anchor(parsed, session, store)
    if "anchor_resolved" not in gold and gold.get("anchor_from_session"):
        # "that author" / "that one": the sheet cannot name the resolved entity, so any resolved
        # anchor from the session counts.
        gold_anchor = predicted_anchor if predicted_anchor is not None else "<missing>"
    gold_aspects = {k: _sign(v) for k, v in gold.get("aspect_deltas", {}).items()}
    gold_desc = {k: _sign(v) for k, v in gold.get("descriptor_deltas", {}).items()}
    return {
        "intent": parsed.intent == gold.get("intent"),
        "anchor": (predicted_anchor or None) == (gold_anchor or None),
        "filters": _filters(parsed.filters) == _filters(gold.get("filters", {})),
        "aspect_signs": {k: _sign(v) for k, v in parsed.aspect_deltas.items()} == gold_aspects,
        "descriptor_signs": {k: _sign(v) for k, v in parsed.descriptor_deltas.items()} == gold_desc,
        "relative_length": parsed.relative_length == gold.get("relative_length"),
        "exclude_previous": bool(parsed.exclude_previous) == bool(gold.get("exclude_previous")),
        "target_index": parsed.target_index == gold.get("target_index"),
    }


def evaluate(rows: list[dict[str, Any]], store: ArtifactStore, parser: str) -> dict[str, Any]:
    """Accuracies for one parser (``rule`` or ``llm``) over the gold rows."""
    per_row = []
    for row in rows:
        session = session_from(row["session"], store)
        if parser == "llm":
            parsed = chat.llm_parse(row["message"], session, store)
            fell_back = parsed is None
            if parsed is None:
                parsed = chat.rule_parse(row["message"], session, store)
        else:
            parsed, fell_back = chat.rule_parse(row["message"], session, store), False
        correct = score_row(parsed, row["gold"], session, store)
        per_row.append(
            {
                "id": row["id"],
                "message": row["message"],
                "predicted_intent": parsed.intent,
                "gold_intent": row["gold"].get("intent"),
                "fell_back_to_rules": fell_back,
                **{f"ok_{k}": v for k, v in correct.items()},
            }  # fmt: skip
        )
    n = max(1, len(per_row))
    field_acc = {f: sum(r[f"ok_{f}"] for r in per_row) / n for f in FIELDS}
    return {
        "parser": parser,
        "rows": len(per_row),
        "intent_accuracy": field_acc["intent"],
        "anchor_resolution_accuracy": field_acc["anchor"],
        "filter_exact_match": field_acc["filters"],
        "aspect_sign_accuracy": field_acc["aspect_signs"],
        "descriptor_sign_accuracy": field_acc["descriptor_signs"],
        "field_accuracy": sum(field_acc.values()) / len(FIELDS),
        "per_field": field_acc,
        "llm_fallbacks": sum(r["fell_back_to_rules"] for r in per_row),
        "per_row": per_row,
    }


# ---------------------------------------------------------------------- the labellers' sheet
# Non-technical labellers fill a plain sheet (one column per question, words not JSON);
# ``sheet_to_gold`` turns it into the gold CSV above. Column meanings are in
# docs/annotation_guides/intent.md.
SHEET_COLUMNS = (
    "id", "message", "keep", "intent", "book_named", "author_named", "that_one_or_author",
    "which_one", "shorter_or_longer", "mood", "year_from", "year_to", "only_genre", "no_genre",
    "something_else", "shown_before", "notes",
)  # fmt: skip
INTENT_WORDS = {  # what the labeller may type -> the intent name
    "general": "recommend_general",
    "recommend": "recommend_general",
    "like a book": "similar_to",
    "similar": "similar_to",
    "like a book with a change": "similar_to_with_modifier",
    "like + change": "similar_to_with_modifier",
    "more by author": "more_by_author",
    "author": "more_by_author",
    "refine": "refine_previous",
    "change": "refine_previous",
    "why": "explain_recommendation",
    "explain": "explain_recommendation",
    "filter": "filter_by_attribute",
    "mixed": "mixed_opinion",
    "mixed opinion": "mixed_opinion",
    "not about books": "out_of_scope",
    "off topic": "out_of_scope",
}
MOOD_WORDS = {  # mood column words -> (kind, name, sign)
    "cozier": ("descriptor", "darkness", -1),
    "lighter": ("descriptor", "darkness", -1),
    "less dark": ("descriptor", "darkness", -1),
    "darker": ("descriptor", "darkness", 1),
    "faster": ("descriptor", "pace", 1),
    "gripping": ("descriptor", "pace", 1),
    "slower": ("descriptor", "pace", -1),
    "slow burn": ("descriptor", "pace", -1),
    "easier": ("descriptor", "complexity_level", -1),
    "simpler": ("descriptor", "complexity_level", -1),
    "denser": ("descriptor", "complexity_level", 1),
    "more literary": ("descriptor", "complexity_level", 1),
    "funnier": ("aspect", "humor", 1),
    "more humor": ("aspect", "humor", 1),
    "less humor": ("aspect", "humor", -1),
    "better characters": ("aspect", "characters", 1),
    "better plot": ("aspect", "plot", 1),
    "better prose": ("aspect", "writing_style", 1),
    "more emotional": ("aspect", "emotional_impact", 1),
}
ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "last": 5}


def write_candidates(path: Path) -> int:
    """The labellers' sheet with ~70 candidate messages across the nine intents prefilled (the
    user keeps 50 and fills the columns by hand; s14.9: LLM-written gold labels are not allowed).
    The ``intent`` column carries a suggestion the labeller must confirm or correct."""
    path.parent.mkdir(parents=True, exist_ok=True)
    words = {v: k for k, v in reversed(list(INTENT_WORDS.items()))}
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        writer.writeheader()
        for k, (intent, message) in enumerate(CANDIDATES, 1):
            writer.writerow({"id": f"c{k:02d}", "message": message, "keep": "",
                             "intent": words.get(intent, intent)})  # fmt: skip
    return len(CANDIDATES)


def _yes(value: str) -> bool:
    return value.strip().lower() in ("y", "yes", "true", "1", "haan", "ha")


def _int(value: str) -> int | None:
    value = value.strip()
    return int(value) if value.isdigit() else None


def sheet_row_to_gold(row: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(session, gold)`` for one filled sheet row; raises ValueError on an unknown intent."""
    get = lambda k: (row.get(k) or "").strip()  # noqa: E731
    intent_word = get("intent").lower()
    intent = INTENT_WORDS.get(intent_word) or (intent_word if intent_word in INTENT_WORDS.values()
                                               else None)  # fmt: skip
    if intent is None:
        raise ValueError(f"row {get('id')!r}: unknown intent {get('intent')!r}")
    gold: dict[str, Any] = {"intent": intent}
    if get("book_named"):
        gold["anchor_titles"] = [get("book_named")]
        gold["anchor_resolved"] = get("book_named")
    if get("author_named"):
        gold["anchor_authors"] = [get("author_named")]
        gold["anchor_resolved"] = get("author_named")
    if _yes(get("that_one_or_author")):
        gold["anchor_from_session"] = True
    which = get("which_one").lower()
    if which:
        gold["target_index"] = _int(which) or ORDINAL_WORDS.get(which)
    if get("shorter_or_longer").lower() in ("shorter", "longer"):
        gold["relative_length"] = get("shorter_or_longer").lower()
    aspects: dict[str, float] = {}
    descriptors: dict[str, float] = {}
    for word in (w.strip().lower() for w in get("mood").split(",") if w.strip()):
        if word not in MOOD_WORDS:
            raise ValueError(f"row {get('id')!r}: unknown mood word {word!r}")
        kind, name, sign = MOOD_WORDS[word]
        (aspects if kind == "aspect" else descriptors)[name] = 0.5 * sign
    if aspects:
        gold["aspect_deltas"] = aspects
    if descriptors:
        gold["descriptor_deltas"] = descriptors
    filters: dict[str, Any] = {}
    if _int(get("year_from")) is not None:
        filters["min_year"] = _int(get("year_from"))
    if _int(get("year_to")) is not None:
        filters["max_year"] = _int(get("year_to"))
    if get("only_genre"):
        filters["genres_include"] = [g.strip() for g in get("only_genre").split(",") if g.strip()]
    if get("no_genre"):
        filters["genres_exclude"] = [g.strip() for g in get("no_genre").split(",") if g.strip()]
    if filters:
        gold["filters"] = filters
    if _yes(get("something_else")):
        gold["exclude_previous"] = True
    session: dict[str, Any] = {}
    shown = [t.strip() for t in get("shown_before").split(";") if t.strip()]
    if shown:
        session = {"turn": 1, "last_shown_titles": shown}
        if intent == "more_by_author" and gold.get("anchor_from_session"):
            session["anchor_titles"] = shown[:1]
    elif gold.get("anchor_from_session") or intent in ("refine_previous", "explain_recommendation"):
        session = {"turn": 1}
    return session, gold


def sheet_to_gold(sheet: Path, out: Path) -> int:
    """Convert the filled labellers' sheet into the gold CSV (``keep`` = yes rows only, or every
    row when the column is empty everywhere); returns the number of rows written."""
    with sheet.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    any_keep = any((r.get("keep") or "").strip() for r in rows)
    kept = [r for r in rows if (r.get("message") or "").strip()
            and (not any_keep or _yes(r.get("keep") or ""))]  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "message", "session_json", "gold_json"])
        for r in kept:
            session, gold = sheet_row_to_gold(r)
            writer.writerow([
                r.get("id") or "", r["message"].strip(),
                json.dumps(session) if session else "", json.dumps(gold),
            ])  # fmt: skip
    return len(kept)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="intent gold-set scorer (CLAUDE.md s14.9)")
    parser.add_argument("--bundle", help="serve bundle folder")
    parser.add_argument("--gold", help="gold CSV (docs/annotation/intent_50.csv)")
    parser.add_argument("--out", help="JSON report path")
    parser.add_argument("--parser", choices=("rule", "llm", "both"), default="both")
    parser.add_argument("--draft", help="write the labellers' sheet with candidate messages")
    parser.add_argument("--from-sheet", help="convert a filled labellers' sheet to the gold CSV")
    args = parser.parse_args(argv)
    if args.draft:
        n = write_candidates(Path(args.draft))
        print(f"wrote {n} candidate messages to {args.draft}")
        return 0
    if args.from_sheet:
        if not args.out:
            parser.error("--from-sheet needs --out (the gold CSV to write)")
        n = sheet_to_gold(Path(args.from_sheet), Path(args.out))
        print(f"wrote {n} gold rows to {args.out}")
        return 0
    if not (args.bundle and args.gold):
        parser.error("--bundle and --gold are required (or --draft)")
    store = core.load_store(args.bundle)
    rows = load_gold(Path(args.gold))
    parsers = ["rule", "llm"] if args.parser == "both" else [args.parser]
    from litsense_core.llm import get_client
    from litsense_core.llm.client import configure

    # Offline scoring: wait for the rate window instead of falling back to the rules (the online
    # limiter never sleeps), and stay under the free tier's 15 requests per minute.
    client = configure({**(store.config.get("llm") or {}), "rpm_limit": 7})
    client.blocking_limiter = True
    if "llm" in parsers and not get_client().available:
        print(f"LLM unavailable ({get_client().unavailable_reason()}); scoring rules only")
        parsers = ["rule"]
    report = {"gold": args.gold, "bundle": store.version, "results": {}}
    for name in parsers:
        result = evaluate(rows, store, name)
        report["results"][name] = result
        print(
            f"{name}: intent {result['intent_accuracy']:.3f}, anchors"
            f" {result['anchor_resolution_accuracy']:.3f}, filters"
            f" {result['filter_exact_match']:.3f}, fields {result['field_accuracy']:.3f}"
            f" over {result['rows']} rows, {result['llm_fallbacks']} fell back to the rules"
        )
        if name == "llm" and result["llm_fallbacks"] > result["rows"] // 10:
            print(
                "WARNING: many LLM rows fell back to the rule parser; the llm numbers are not"
                " an LLM measurement (quota, rate limit or provider errors; check the key)"
            )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
