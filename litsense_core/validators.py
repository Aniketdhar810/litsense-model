"""Deterministic checks on LLM text before it reaches a reader (CLAUDE.md s14.6).

``check(text, allowed_titles, evidence, max_words, store)`` returns a list of problem codes
(empty when the text passes). The caller falls back to a template on any problem and keeps the
codes so the backend can log rejection rates (a hallucination-rate proxy, s14.9).

Codes: ``title:<catalogue title>`` (a title outside the allowed set), ``number:<n>`` (a number
not in the evidence), ``spoiler`` (a sentence the bundled spoiler scorer flags), ``length``
(over the word limit), ``format`` (markdown list, heading, emphasis or URL), ``quote`` (quoted
text that is not the evidence snippet), ``empty``.
"""

import json
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

from litsense_core.schemas import Evidence
from litsense_core.store import ArtifactStore

MIN_TITLE_WORDS = 2
MIN_TITLE_CHARS = 10  # a one-word title must be this long ("Frankenstein"); "Mystery" is skipped
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_QUOTED = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_MARKDOWN = re.compile(r"(\*\*|__|^#+\s|`)", re.M)
_URL = re.compile(r"https?://|www\.", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_TITLE_CACHE: dict[int, tuple[ArtifactStore, "TitleMatcher"]] = {}
_ARTICLE = re.compile(r"^(?:the|a|an) ")
# Spoiler regex patterns too weak for generated text: a summary that says "some criticise the
# ending" or "in the end most readers loved it" reveals nothing (D72). The classifier, when the
# bundle has a trained one, still scores every sentence.
WEAK_SPOILER_PATTERNS = frozenset({"the ending", "in the end", "ends up", "is actually"})
_SCORER_CACHE: dict[int, tuple[Any, Any]] = {}


def normalize(text: str) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
    return " ".join(text.split())


class TitleMatcher:
    """One case-insensitive alternation over the catalogue titles worth matching. After a leading
    article is dropped, a title must have two or more words or a single word of ten or more
    characters: "It", "Dune", "Mystery", "The Plot" and "The Help" are ordinary phrases that would
    reject ordinary sentences (D69, D72). Built once per store."""

    def __init__(self, titles: Iterable[str]) -> None:
        keep = {}
        for t in titles:
            norm = normalize(t)
            core = _ARTICLE.sub("", norm)
            if core and (len(core.split()) >= MIN_TITLE_WORDS or len(core) >= MIN_TITLE_CHARS):
                keep[norm] = t
        self.by_norm = keep
        # Longest first so "Mistborn: The Final Empire" wins over "Mistborn".
        ordered = sorted(keep, key=len, reverse=True)
        self.regex = (
            re.compile(r"\b(?:" + "|".join(re.escape(n) for n in ordered) + r")\b")
            if ordered
            else None
        )

    def mentioned(self, text: str) -> list[str]:
        """Catalogue titles (display form) found in ``text``, in order of appearance."""
        if self.regex is None:
            return []
        norm = normalize(text)
        return [self.by_norm[m.group(0)] for m in self.regex.finditer(norm)]


def title_matcher(store: ArtifactStore) -> TitleMatcher:
    entry = _TITLE_CACHE.get(id(store))
    if entry is None or entry[0] is not store:
        entry = (store, TitleMatcher(store.catalog.title))
        _TITLE_CACHE[id(store)] = entry
    return entry[1]


def evidence_numbers(evidence: Evidence | dict[str, Any]) -> set[str]:
    """Every number a text may use: ints, floats at 0 to 3 decimals, and rates in [0, 1] as
    percentages, taken from the evidence JSON."""
    raw = evidence.model_dump() if isinstance(evidence, Evidence) else evidence
    out: set[str] = set()

    def add_number(v: float) -> None:
        if float(v).is_integer():
            out.add(str(int(v)))
        for d in (1, 2, 3):
            s = f"{v:.{d}f}".rstrip("0").rstrip(".")
            out.add(s)
        out.add(str(int(round(v))))
        if 0.0 <= v <= 1.0:
            out.add(str(int(round(100 * v))))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list | tuple):
            for v in node:
                walk(v)
        elif isinstance(node, bool):
            return
        elif isinstance(node, int | float):
            add_number(float(node))
        elif isinstance(node, str):
            for m in _NUMBER.finditer(node):
                out.add(m.group(0).replace(",", ""))

    walk(raw)
    return out


def numbers_in(text: str) -> list[str]:
    return [m.group(0).replace(",", "") for m in _NUMBER.finditer(text)]


def generated_text_scorer(spoiler: Any) -> Any:
    """``spoiler`` with the weak regex patterns removed (same classifier, boosts and threshold)."""
    if spoiler is None:
        return None
    entry = _SCORER_CACHE.get(id(spoiler))
    if entry is None or entry[0] is not spoiler:
        from litsense_core.spoiler import SpoilerScorer

        patterns = [p for p in spoiler.patterns if p not in WEAK_SPOILER_PATTERNS]
        strict = SpoilerScorer(
            spoiler.coef, spoiler.intercept, spoiler.threshold, patterns, spoiler.regex_boost,
            spoiler.late_position_boost, spoiler.late_fraction,
        )  # fmt: skip
        entry = (spoiler, strict)
        _SCORER_CACHE[id(spoiler)] = entry
    return entry[1]


def spoiler_sentences(text: str, spoiler: Any, encoder: Any = None) -> list[str]:
    """Sentences the scorer flags (strong regex patterns plus the classifier when an encoder is
    given)."""
    spoiler = generated_text_scorer(spoiler)
    if spoiler is None:
        return []
    sentences = [s.strip() for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    if not sentences:
        return []
    if encoder is not None and spoiler.has_model:
        emb = encoder.encode(sentences)
    else:
        emb = np.zeros((len(sentences), 1), dtype=np.float32)
    flags = spoiler.flags(emb, sentences)
    return [s for s, f in zip(sentences, flags, strict=True) if f]


def check(
    text: str | None,
    allowed_titles: Iterable[str],
    evidence: Evidence | dict[str, Any] | None,
    max_words: int,
    store: ArtifactStore,
    snippet: str | None = None,
) -> list[str]:
    """All problems with ``text`` (empty list = accepted), against a loaded store."""
    return check_text(
        text, allowed_titles, evidence, max_words, title_matcher(store), store.spoiler,
        store.encoder, snippet,
    )  # fmt: skip


def check_text(
    text: str | None,
    allowed_titles: Iterable[str],
    evidence: Evidence | dict[str, Any] | None,
    max_words: int,
    matcher: TitleMatcher,
    spoiler: Any = None,
    encoder: Any = None,
    snippet: str | None = None,
) -> list[str]:
    """The checks over explicit title and spoiler scorers (the offline synthesis stage has no
    store; it builds a ``TitleMatcher`` from ``books.parquet`` and a regex-only scorer)."""
    if text is None or not text.strip():
        return ["empty"]
    problems: list[str] = []
    allowed = {normalize(t) for t in allowed_titles}
    for title in matcher.mentioned(text):
        if normalize(title) not in allowed:
            problems.append(f"title:{title}")
    if evidence is not None:
        permitted = evidence_numbers(evidence)
        for n in numbers_in(text):
            if n not in permitted:
                problems.append(f"number:{n}")
    if spoiler_sentences(text, spoiler, encoder):
        problems.append("spoiler")
    if len(text.split()) > max_words:
        problems.append("length")
    if _LIST_LINE.search(text) or _MARKDOWN.search(text) or _URL.search(text):
        problems.append("format")
    if snippet is None and isinstance(evidence, Evidence):
        snip = evidence.item_evidence.get("snippet") or {}
        snippet = snip.get("text") if isinstance(snip, dict) else None
    for m in _QUOTED.finditer(text):
        quoted = normalize(m.group(1))
        if snippet is None or quoted not in normalize(snippet):
            problems.append("quote")
            break
    return list(dict.fromkeys(problems))


def evidence_json(evidence: Evidence) -> str:
    """The evidence as the prompt sees it (compact JSON, stable key order)."""
    return json.dumps(evidence.model_dump(), ensure_ascii=False, sort_keys=True)
