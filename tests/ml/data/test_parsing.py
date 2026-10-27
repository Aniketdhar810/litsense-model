"""Field parsers (CLAUDE.md s16.2 data tests plus the format findings F1-F12)."""

import json

import pytest

from ml.data.parsing import (
    as_isbn13,
    clean_html,
    extract_parent_asin,
    isbn10_to_13,
    json_str_field,
    maybe_json,
    parse_author,
    parse_format,
    parse_image,
    parse_pages,
    parse_year,
    series_suffix,
    split_isbns,
)


def test_clean_html() -> None:
    text = "Great read.<br /><br />I&#39;d read it&nbsp;again &amp; again. <b>Five</b> ﬁve"
    assert clean_html(text) == "Great read. I'd read it again & again. Five five"
    assert clean_html(None) == "" and clean_html("") == ""


def test_maybe_json() -> None:
    assert maybe_json('{"a": 1}') == {"a": 1}
    assert maybe_json("not json") == {} and maybe_json(None) == {}
    assert maybe_json({"a": 1}) == {"a": 1}


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (("0701169850", "978-0701169855"), ("0701169850", "9780701169855")),
        (("9780316185363", "978-0316185363"), (None, "9780316185363")),  # F3: 13 digits in ISBN 10
        (("0-8050-6909-x", None), ("080506909X", None)),
        (("12345", "abc"), (None, None)),
        ((None,), (None, None)),
    ],
)
def test_split_isbns(values: tuple[object, ...], expected: tuple[str | None, str | None]) -> None:
    assert split_isbns(*values) == expected


def test_isbn_conversion() -> None:
    assert isbn10_to_13("0701169850") == "9780701169855"
    assert isbn10_to_13("080506909X") == "9780805069099"
    assert as_isbn13("0701169850", None) == "9780701169855"
    assert as_isbn13(None, "9781111111111") == "9781111111111"
    assert as_isbn13(None, None) is None


@pytest.mark.parametrize(
    ("text", "year"),
    [
        ("January 1, 2004", 2004),
        ("Hardcover – Import, January 1, 2004", 2004),
        ("1st Edition", None),
        (None, None),
    ],
)
def test_parse_year(text: str | None, year: int | None) -> None:
    assert parse_year(text) == year


def test_parse_pages() -> None:
    assert parse_pages({"Language": "English", "Hardcover": "196 pages"}) == 196
    assert parse_pages({"File size": "2345 KB", "Print length": "352 pages"}) == 352  # Kindle
    assert parse_pages({"Language": "English"}) is None


@pytest.mark.parametrize(
    ("author", "store", "expected"),
    [
        (
            {"avatar": "x", "name": "Peter Ackroyd", "about": []},
            "Peter Ackroyd (Author)",
            "Peter Ackroyd",
        ),
        ('{"name": "Tess Arden"}', None, "Tess Arden"),  # JSON-string author
        (None, "Sandra Wilde (Editor)", "Sandra Wilde"),
        (None, "Ada Moss (Illustrator), Rowan Pike (Author)", "Rowan Pike"),
        (None, "Remington Kane (Author)   Format: Kindle Edition", "Remington Kane"),
        ({"avatar": "x"}, "Sergio Ruzzier (Author)", "Sergio Ruzzier"),  # dict without name
        (None, "Stephen King", "Stephen King"),  # no role
        (None, None, None),
    ],
)
def test_parse_author(author: object, store: object, expected: str | None) -> None:
    assert parse_author(author, store) == expected


@pytest.mark.parametrize(
    ("details", "main_category", "subtitle", "categories", "expected"),
    [
        (
            {"Hardcover": "196 pages"},
            "Books",
            "Hardcover – May 8, 2012",
            ["Books"],
            ("Hardcover", "print"),
        ),
        (
            {"File size": "2 MB", "Print length": "352 pages"},
            "Buy a Kindle",
            "Kindle Edition",
            ["Books"],
            ("Kindle", "kindle"),
        ),
        (
            {"Publisher": "Audible"},
            "Audible Audiobooks",
            None,
            ["Audible Books & Originals"],
            ("Audible", "audio"),
        ),
        ({"Audio CD": "8 pages"}, "Books", None, ["Books"], ("Audio CD", "audio")),
        ({"Language": "English"}, "Books", "1st Edition", ["Books"], (None, "unknown")),
        (
            {"Language": "English"},
            "Books",
            "Paperback – January 1, 1994",
            ["Books"],
            ("Paperback", "print"),
        ),
        (
            {"Item Weight": "1 pound"},
            "Books",
            "Unknown Binding",
            ["Books"],
            ("Unknown Binding", "print"),
        ),
        ({"Flexibound": "120 pages"}, "Books", None, ["Books"], ("Flexibound", "print")),
    ],
)
def test_parse_format(
    details: dict[str, str],
    main_category: str,
    subtitle: str | None,
    categories: list[str],
    expected: tuple[str | None, str],
) -> None:
    assert parse_format(details, main_category, subtitle, categories) == expected


def test_parse_image_list_and_dict_forms() -> None:
    images = [{"large": "b.jpg", "variant": "PT01"}, {"large": "a.jpg", "variant": "MAIN"}]
    assert parse_image(images) == "a.jpg"
    as_dict = {"large": ["b.jpg", "a.jpg"], "variant": ["PT01", "MAIN"]}
    assert parse_image(as_dict) == "a.jpg"
    assert parse_image([{"large": "c.jpg", "variant": "PT01"}]) == "c.jpg"
    assert parse_image([]) is None and parse_image(None) is None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("The Way of Kings (The Stormlight Archive, #1)", ("The Stormlight Archive", 1.0)),
        ("Edgedancer (The Stormlight Archive, #2.5)", ("The Stormlight Archive", 2.5)),
        ("The House of Memory (Pluto's Snitch #2)", ("Pluto's Snitch", 2.0)),
        ("The Unschooled Wizard (Sun Wolf and Starhawk, #1-2)", None),  # a range: a box set
        ("The Martian", None),
    ],
)
def test_series_suffix(title: str, expected: tuple[str, float] | None) -> None:
    assert series_suffix(title) == expected


def test_reading_fields_without_parsing() -> None:
    record = {"title": 'He said "parent_asin": "X"', "store": "a", "parent_asin": "0701169850"}
    line = json.dumps(record)
    assert extract_parent_asin(line) == "0701169850"
    assert extract_parent_asin(json.dumps(record, separators=(",", ":"))) == "0701169850"
    assert extract_parent_asin('{"title": "x"}') is None
    gr = json.dumps({"isbn": "0312853122", "isbn13": "9780312853129", "title": "x"})
    assert json_str_field(gr, "isbn") == "0312853122"
    assert json_str_field(gr, "isbn13") == "9780312853129"
