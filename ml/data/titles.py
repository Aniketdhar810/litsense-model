"""Title and author rules for edition canonicalization (CLAUDE.md s7.4, docs/decisions.md D28).

``normalize_title`` builds the matching key two editions of one work share; ``display_title`` is
the human title with format noise removed; ``volume_number`` and ``series_from_title`` read the
series position from Amazon titles (the Goodreads graph is optional, D20).
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
_WORDS += ["eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen"]
_WORDS += ["eighteen", "nineteen", "twenty"]
_ORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"]
_ORDS += ["tenth"]
_ROMANS = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]
WORD_NUMBERS = {w: i for i, w in enumerate(_WORDS, start=1)}
ORDINALS = {w: i for i, w in enumerate(_ORDS, start=1)}
ROMAN = {r: i for i, r in enumerate(_ROMANS, start=1)}
_NUM = r"(\d+(?:\.\d+)?|[ivx]{1,4}|" + "|".join([*WORD_NUMBERS, *ORDINALS]) + r")"
_PAREN_RE = re.compile(r"\s*[(\[]([^()\[\]]*)[)\]]")
_VOLUME_RES = (
    re.compile(rf"\b(?:book|bk|vol|volume|part|no|number)\.?\s*(?:no\.?\s*)?{_NUM}\b"),
    re.compile(rf"#\s*{_NUM}\b"),
    re.compile(rf"\b{_NUM}\s+of\s+\w"),
    re.compile(rf"\bbook\s+{_NUM}\s+of\b"),
)
_TRAILING_DATE_RE = re.compile(r"\s*\b\d{1,2}\s+[a-z]{3,9}\.?\s+\d{4}\s*$", re.IGNORECASE)
_SERIES_WORDS_RE = re.compile(
    r"\b(?:series|saga|trilogy|cycle|quartet|chronicles|novels|books|sequence)\s*$"
)
_SERIES_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\(([^()]+?),?\s*#(\d+(?:\.\d+)?)\)",  # (Name, #3)  Goodreads style
        rf"\(([^()]*?)[,\s]+(?:book|bk\.?|vol\.?|volume|part|no\.?)\s*{_NUM}\)",  # (Name Book 2)
        r"\(([^()]+?),\s*(\d+(?:\.\d+)?)\)",  # (The Mistborn Saga, 4)
        rf"\bbook\s+{_NUM}\s+of\s+(?:the\s+)?([^():]+)",  # Book Three of Mistborn
        rf":\s*([^:()]+?),\s*(?:book|vol\.?|volume)\s*{_NUM}\s*$",  # X: Name, Book 3
    )
)


def number_value(token: str) -> float | None:
    """3, "3", "iii", "three" or "third" as a number."""
    token = token.lower().strip(".")
    if re.fullmatch(r"\d+(\.\d+)?", token):
        return float(token)
    value = WORD_NUMBERS.get(token) or ORDINALS.get(token) or ROMAN.get(token)
    return float(value) if value else None


@dataclass(frozen=True)
class TitleRules:
    """Compiled ``canonical`` config patterns."""

    noise: re.Pattern[str]
    boxset: re.Pattern[str]
    subtitle_noise: re.Pattern[str]

    @classmethod
    def from_config(cls, canonical: Mapping[str, Any]) -> "TitleRules":
        """Build from the ``canonical`` config section."""

        def any_of(patterns: Sequence[str], escape: bool) -> re.Pattern[str]:
            parts = [re.escape(p) if escape else p for p in patterns]
            return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.IGNORECASE)

        return cls(
            noise=any_of(canonical["edition_noise"], escape=True),
            boxset=any_of(canonical["boxset_patterns"], escape=False),
            subtitle_noise=re.compile(
                "|".join(f"(?:{p})" for p in canonical["subtitle_noise_patterns"]), re.IGNORECASE
            ),
        )

    def is_noise(self, text: str) -> bool:
        """True when ``text`` holds nothing but format/edition words, numbers and punctuation."""
        rest = self.noise.sub(" ", text.lower())
        rest = re.sub(r"\b(?:\d+(?:st|nd|rd|th)?|edition|ed|format|version|print)\b", " ", rest)
        return not re.sub(r"[\W_]+", "", rest)

    def is_noise_subtitle(self, text: str) -> bool:
        """A subtitle that is format noise or marketing ("A Novel", "TikTok sensation")."""
        return self.is_noise(text) or bool(self.subtitle_noise.search(text.strip()))


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").lower()
    return text.replace("’", "'").replace("‘", "'").replace("&", " and ")


def normalize_title(title: str, rules: TitleRules) -> str:
    """Matching key: lowercase, no bracketed segments, no noise subtitle or trailing format
    words, no punctuation, no leading article. "Dune (Mass Market Paperback, 40th Anniversary
    Edition)" -> "dune"."""
    text = _PAREN_RE.sub(" ", _fold(title))
    head, sep, tail = text.partition(":")
    if sep and (rules.is_noise_subtitle(tail) or volume_number(tail) is not None):
        text = head
    text = _trim_trailing_noise(text, rules)
    text = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    return re.sub(r"^(?:the|a|an) ", "", text)


def number_tokens(key: str) -> list[str]:
    """Numbers in a matching key, digits and multi-letter roman numerals ("nypd red 2" -> ["2"]).
    Two titles that differ in these are different books (D33)."""
    return sorted(t for t in key.split() if t.isdigit() or (len(t) > 1 and t in ROMAN))


def title_head(title: str, rules: TitleRules) -> str | None:
    """Matching key of the text before the first ':' of a title that has a subtitle ("Cold hand in
    mine: Strange stories" -> "cold hand in mine"); None for a title without one (D32)."""
    head, sep, tail = _PAREN_RE.sub(" ", _fold(title)).partition(":")
    if not sep or not re.sub(r"[\W_]+", "", tail):
        return None
    return normalize_title(head, rules) or None


def display_title(title: str, rules: TitleRules) -> str:
    """The title as shown in the app: format-only brackets and noise subtitles removed, series
    brackets and case kept. "Project Hail Mary: A Novel" -> "Project Hail Mary"."""
    text = unicodedata.normalize("NFKC", title or "")
    text = _PAREN_RE.sub(lambda m: " " if rules.is_noise(m.group(1)) else m.group(0), text)
    head, sep, tail = text.partition(":")
    if sep and rules.is_noise_subtitle(_PAREN_RE.sub(" ", tail)):  # keep the series bracket
        text = " ".join([head, *(m.group(0).strip() for m in _PAREN_RE.finditer(tail))])
    return _trim_trailing_noise(text, rules).strip(" -:,")


def _trim_trailing_noise(text: str, rules: TitleRules) -> str:
    """Drop trailing dates and format words ("Midnight Library Paperback 18 Feb 2021"), never
    the whole title."""
    words = text.split()
    while True:
        words = _TRAILING_DATE_RE.sub("", " ".join(words)).split()
        cut = next(
            (
                i
                for i in range(1, len(words))
                if rules.noise.search(" ".join(words[i:])) and rules.is_noise(" ".join(words[i:]))
            ),
            None,
        )
        if cut is None:
            return " ".join(words)
        words = words[:cut]


def is_boxset(title: str, rules: TitleRules) -> bool:
    """Box sets, omnibuses and "Books 1-3" bundles (never merged with a single volume)."""
    return bool(rules.boxset.search(title or ""))


def volume_number(title: str) -> float | None:
    """Volume/position marker in a title ("Book 3", "#2.5", "Vol. III", "Book Three of")."""
    text = _fold(title)
    for pattern in _VOLUME_RES:
        if m := pattern.search(text):
            value = number_value(m.group(1))
            if value is not None:
                return value
    return None


def series_from_title(title: str) -> tuple[str, float] | None:
    """``(series name, position)`` from an Amazon title, or None."""
    text = unicodedata.normalize("NFKC", title or "")
    for i, pattern in enumerate(_SERIES_PATTERNS):
        if not (m := pattern.search(text)):
            continue
        name, number = (m.group(2), m.group(1)) if i == 3 else (m.group(1), m.group(2))
        position = number_value(number)
        name = name.strip(" ,:-")
        if position is not None and name and not re.fullmatch(r"(?:book|vol|no)\.?", name, re.I):
            return name, position
    return None


def series_key(name: str) -> str:
    """Comparable series name: "The Mistborn Saga" and "Mistborn" give "mistborn"."""
    text = " ".join(re.sub(r"[^\w\s]", " ", _fold(name)).split())
    text = re.sub(r"^(?:the|a|an) ", "", text)
    return _SERIES_WORDS_RE.sub("", text).strip() or text


def author_key(author: str | None) -> str | None:
    """Lowercase author with initials and punctuation normalized: "J.K. Rowling" and
    "JK Rowling" give "jk rowling" (s7.7)."""
    if not author:
        return None
    text = _fold(author).replace(".", " ")
    merged: list[str] = []
    initials = False  # whether merged[-1] is a run of single-letter initials
    for word in re.sub(r"[^\w\s'-]", " ", text).split():
        if len(word) == 1 and initials:  # "j k rowling" -> "jk rowling"
            merged[-1] += word
        else:
            merged.append(word)
            initials = len(word) == 1
    return " ".join(merged) or None
