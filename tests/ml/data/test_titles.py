"""Title rules behind edition canonicalization (CLAUDE.md s7.4, s16.2; D28)."""

from typing import Any

import pytest

from ml.data.titles import (
    TitleRules,
    author_key,
    display_title,
    is_boxset,
    normalize_title,
    series_from_title,
    series_key,
    title_head,
    volume_number,
)


@pytest.fixture(scope="module")
def rules(repo_config: dict[str, Any]) -> TitleRules:
    return TitleRules.from_config(repo_config["canonical"])


@pytest.mark.parametrize(
    ("title", "key"),
    [
        ("Dune (Mass Market Paperback, 40th Anniversary Edition)", "dune"),  # s16.2
        ("Project Hail Mary: A Novel", "project hail mary"),
        (
            "The Silent Patient: The record-breaking, multimillion copy TikTok sensation",
            "silent patient",
        ),
        ("The Midnight Library Paperback 18 Feb 2021", "midnight library"),
        ("Where the Crawdads Sing Deluxe Edition", "where the crawdads sing"),
        ("The Well of Ascension (Mistborn, Book 2)", "well of ascension"),
        ("The Hero of Ages: Book Three of Mistborn", "hero of ages"),
        ("Mistborn: The Final Empire", "mistborn the final empire"),  # a real subtitle stays
        ("Catch-22", "catch 22"),
        ("Gone Girl", "gone girl"),
    ],
)
def test_normalize_title(rules: TitleRules, title: str, key: str) -> None:
    assert normalize_title(title, rules) == key


@pytest.mark.parametrize(
    ("title", "shown"),
    [
        ("Project Hail Mary: A Novel", "Project Hail Mary"),
        ("The Midnight Library Paperback 18 Feb 2021", "The Midnight Library"),
        ("The Well of Ascension (Mistborn, Book 2)", "The Well of Ascension (Mistborn, Book 2)"),
        ("Q Is For Quarry (Kinsey Millhone, #17)", "Q Is For Quarry (Kinsey Millhone, #17)"),
        (
            "Finders Keepers: A Novel (The Bill Hodges Trilogy Book 2)",
            "Finders Keepers (The Bill Hodges Trilogy Book 2)",
        ),
        ("Dune (Mass Market Paperback)", "Dune"),
        ("1984", "1984"),
    ],
)
def test_display_title_keeps_series_brackets(rules: TitleRules, title: str, shown: str) -> None:
    assert display_title(title, rules) == shown


def test_title_head(rules: TitleRules) -> None:  # D32
    assert title_head("Cold hand in mine: Strange stories", rules) == "cold hand in mine"
    assert title_head("Mistborn: The Final Empire", rules) == "mistborn"
    assert title_head("Thrones, Dominations (A Lord Peter Wimsey: Mystery)", rules) is None
    assert title_head("Cold Hand in Mine", rules) is None
    assert title_head("Trailing colon:", rules) is None


def test_boxsets(rules: TitleRules) -> None:
    assert is_boxset("The Mistborn Trilogy Box Set", rules)  # s16.2
    assert is_boxset("The Ember Cycle Box Set: Books 1-3", rules)
    assert is_boxset("Harry Potter Boxed Set", rules)
    assert not is_boxset("Mistborn: The Final Empire", rules)
    assert not is_boxset("The Box", rules)


@pytest.mark.parametrize(
    ("title", "volume"),
    [
        ("Emberlight (The Ember Cycle Book 2)", 2.0),
        ("The Hero of Ages: Book Three of Mistborn", 3.0),
        ("Q Is For Quarry (Kinsey Millhone, #17)", 17.0),
        ("The Dragon Reborn, Vol. III", 3.0),
        ("Embers Between (The Ember Cycle, #1.5)", 1.5),
        ("Mistborn: The Final Empire", None),
        ("Catch-22", None),
    ],
)
def test_volume_number(title: str, volume: float | None) -> None:
    assert volume_number(title) == volume


@pytest.mark.parametrize(
    ("title", "series"),
    [
        ("The Way of Kings (The Stormlight Archive, #1)", ("The Stormlight Archive", 1.0)),  # s16.2
        ("Edgedancer (The Stormlight Archive, #2.5)", ("The Stormlight Archive", 2.5)),
        ("The Well of Ascension (Mistborn, Book 2)", ("Mistborn", 2.0)),
        ("Ashfall (The Ember Cycle Book 1)", ("The Ember Cycle", 1.0)),
        ("The Hero of Ages: Book Three of Mistborn", ("Mistborn", 3.0)),
        ("Cinderthrone: The Ember Cycle, Book 3", ("The Ember Cycle", 3.0)),
        ("Gone Girl", None),
    ],
)
def test_series_from_title(title: str, series: tuple[str, float] | None) -> None:
    assert series_from_title(title) == series


def test_series_and_author_keys() -> None:
    assert series_key("The Mistborn Saga") == series_key("Mistborn") == "mistborn"
    assert author_key("J.K. Rowling") == author_key("JK Rowling") == "jk rowling"  # s7.7
    assert author_key("J. R. R. Tolkien") == "jrr tolkien"
    assert author_key(None) is None and author_key("") is None
