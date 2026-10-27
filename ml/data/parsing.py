"""Field parsers shared by the data stages (CLAUDE.md s7.2-s7.3, docs/decisions.md F1-F12).

Every rule here was checked against the first records of the real files; the comments name the
finding that forced it.
"""

import html
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

ISBN10_RE = re.compile(r"^\d{9}[\dX]$")
ISBN13_RE = re.compile(r"^97[89]\d{10}$")
DATE_RE = re.compile(r"\(([A-Z][a-z]+ \d{1,2}, \d{4})\)")  # Publisher "...(January 1, 2004)"
YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2})\b")
PAGES_RE = re.compile(r"(\d+)\s+pages", re.IGNORECASE)
SERIES_RE = re.compile(r"\(([^()]+?),?\s*#(\d+(?:\.\d+)?)\)\s*$")  # Goodreads "(Name, #3)"
_TAG_RE = re.compile(r"<[^>]+>")
_STORE_PART_RE = re.compile(r"\s*([^,()]+?)\s*\(([^)]*)\)")
_SUBTITLE_DASH_RE = re.compile(r"\s[–-]\s")  # "Paperback – January 1, 1994"
_STORE_FORMAT_RE = re.compile(r"\s+Format:.*$")  # F4: "Name (Author)   Format: Kindle Edition"
_PARENT_ASIN_RE = re.compile(r'"parent_asin":\s*"([^"]*)"')

# details keys that name a binding (F1); the first one present wins
PRINT_FORMATS = (
    "Hardcover", "Paperback", "Mass Market Paperback", "Board book", "Library Binding",
    "Spiral-bound", "Spiral bound", "Perfect Paperback", "Leather Bound", "Loose Leaf",
    "School Library Binding", "Plastic Comb", "Imitation Leather", "Unknown Binding",
    "Flexibound", "Pocket Book", "Comic", "Single Issue Magazine", "Staple Bound",
    "Calendar", "Cards", "Map", "Pamphlet", "Sheet music", "Diary",
)  # fmt: skip
AUDIO_FORMATS = ("Audio CD", "MP3 CD", "Audio Cassette", "Audible Audiobook")


def clean_html(text: Any) -> str:
    """Unescape entities, drop tags, NFKC-normalize and collapse whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", html.unescape(str(text)))
    return " ".join(unicodedata.normalize("NFKC", text).split())


def maybe_json(value: Any) -> Any:
    """Decode a JSON-string field (some dataset copies encode ``details``/``author``)."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value if value is not None else {}


def split_isbns(*values: Any) -> tuple[str | None, str | None]:
    """First valid ISBN-10 and ISBN-13 among ``values``, ignoring dashes and spaces.

    A value is classified by its shape, not by the field it came from: ``details["ISBN 10"]``
    sometimes holds a 13-digit ISBN (F3).
    """
    isbn10 = isbn13 = None
    for value in values:
        digits = re.sub(r"[^0-9Xx]", "", str(value or "")).upper()
        if isbn10 is None and ISBN10_RE.match(digits):
            isbn10 = digits
        elif isbn13 is None and ISBN13_RE.match(digits):
            isbn13 = digits
    return isbn10, isbn13


def isbn10_to_13(isbn10: str) -> str:
    """The 978-prefixed ISBN-13 of an ISBN-10."""
    core = "978" + isbn10[:9]
    total = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(core))
    return core + str((10 - total % 10) % 10)


def as_isbn13(isbn10: str | None, isbn13: str | None) -> str | None:
    """ISBN-13 if known, else the converted ISBN-10, else None."""
    return isbn13 or (isbn10_to_13(isbn10) if isbn10 else None)


def parse_year(text: Any) -> int | None:
    """Last plausible four-digit year in ``text``."""
    years = YEAR_RE.findall(str(text or ""))
    return int(years[-1]) if years else None


def parse_pages(details: Mapping[str, Any]) -> int | None:
    """First "N pages" value in ``details`` (binding keys and Kindle's ``Print length``)."""
    for value in details.values():
        if isinstance(value, str) and (m := PAGES_RE.search(value)):
            return int(m.group(1))
    return None


def parse_author(author: Any, store: Any) -> str | None:
    """``author.name``, else the first "(Author)" credit in ``store``, else its first credit."""
    record = maybe_json(author)
    if isinstance(record, dict) and isinstance(record.get("name"), str) and record["name"].strip():
        return clean_html(record["name"])
    text = _STORE_FORMAT_RE.sub("", str(store or "")).strip()
    credits = _STORE_PART_RE.findall(text)
    for name, roles in credits:
        if "author" in roles.lower():
            return clean_html(name)
    if credits:
        return clean_html(credits[0][0])
    return clean_html(text) or None


def parse_format(
    details: Mapping[str, Any],
    main_category: str | None,
    subtitle: str | None,
    categories: Sequence[str],
) -> tuple[str | None, str]:
    """``(format, edition_kind)``; kind is ``print``, ``kindle``, ``audio`` or ``unknown`` (F1).

    The binding is a ``details`` key for most print books; about a tenth only name it in the
    subtitle ("Paperback – January 1, 1994", "Unknown Binding").
    """
    category, sub = main_category or "", subtitle or ""
    for key in [*details, _SUBTITLE_DASH_RE.split(sub)[0].strip()]:
        if key in PRINT_FORMATS:
            return key, "print"
        if key in AUDIO_FORMATS:
            return key, "audio"
    if category == "Buy a Kindle" or "File size" in details or "Kindle" in sub:
        return "Kindle", "kindle"
    top = categories[0] if categories else ""
    if category.startswith("Audible") or top.startswith("Audible") or "Audiobook" in sub:
        return "Audible", "audio"
    return None, "unknown"


def parse_image(images: Any) -> str | None:
    """``large`` URL of the MAIN image, else the first ``large``; images may be a dict of lists."""
    if isinstance(images, dict):
        keys = list(images)
        images = [
            dict(zip(keys, values, strict=False)) for values in zip(*images.values(), strict=False)
        ]
    images = [i for i in images or [] if isinstance(i, dict)]
    main = next((i["large"] for i in images if i.get("variant") == "MAIN" and i.get("large")), None)
    return main or next((i["large"] for i in images if i.get("large")), None)


def series_suffix(title: str) -> tuple[str, float] | None:
    """``("The Stormlight Archive", 1.0)`` from "The Way of Kings (The Stormlight Archive, #1)".

    Ranges such as "#1-2" (box sets) do not match on purpose.
    """
    m = SERIES_RE.search(title or "")
    return (m.group(1).strip(), float(m.group(2))) if m else None


def json_str_field(line: str, key: str, *, last: bool = False) -> str | None:
    """Value of a top-level string field read straight from a JSON line, without parsing it.

    Safe because a quote inside a JSON string is always escaped, so ``"key": "`` cannot occur in
    a value. ``last`` searches from the end (``parent_asin`` sits at the end of meta records).
    """
    marker = f'"{key}": "'
    start = line.rfind(marker) if last else line.find(marker)
    if start < 0:
        return None
    start += len(marker)
    return line[start : line.find('"', start)]


def extract_parent_asin(line: str) -> str | None:
    """``parent_asin`` of a meta or review line, without JSON-parsing the line."""
    value = json_str_field(line, "parent_asin", last=True)
    if value is not None:
        return value
    m = _PARENT_ASIN_RE.search(line)  # tolerate other separators
    return m.group(1) if m else None
