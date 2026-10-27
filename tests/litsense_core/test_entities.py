"""Entity resolution (CLAUDE.md s14.5, s16.2 GenAI) on the hand-built store."""

from typing import Any

import pytest

from litsense_core import entities
from tests.litsense_core.fake_store import build_store


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Any:
    return build_store(tmp_path_factory.mktemp("bundle"))


def test_exact_and_fuzzy_titles(store: Any) -> None:
    exact = entities.resolve_title("project hail mary", store)
    assert exact.kind == "title" and exact.book_ids == [0] and exact.score == 100.0
    fuzzy = entities.resolve_title("Projct Hail Mary", store)
    assert fuzzy.kind == "title" and fuzzy.book_ids == [0] and fuzzy.score >= 88
    assert fuzzy.log()["match"] == "Project Hail Mary"
    assert entities.resolve_title("the martian", store).book_ids == [1]
    assert entities.resolve_title("zzz qqq nothing here", store).kind == "none"
    assert entities.resolve_title("", store).kind == "none"


def test_authors_exact_fuzzy_and_initials(store: Any) -> None:
    exact = entities.resolve_author("andy weir", store)
    assert exact.kind == "author" and exact.author == "Andy Weir"
    assert sorted(exact.book_ids) == [0, 1, 2] and exact.author_id is not None
    fuzzy = entities.resolve_author("Andy Wier", store)
    assert fuzzy.kind == "author" and fuzzy.author == "Andy Weir"
    assert entities.author_key("J.K. Rowling") == entities.author_key("JK Rowling")
    assert entities.resolve_author("someone unknown", store).kind == "none"


def test_ambiguous_title_returns_options(store: Any) -> None:
    out = entities.resolve_title("Mistborn", store)
    assert out.kind == "ambiguous" and 2 <= len(out.options) <= 3
    assert all(o.startswith("Mistborn") for o in out.options)
    assert entities.resolve_title("Mistborn: The Final Empire", store).book_ids == [5]


def test_resolve_any_tries_the_other_kind(store: Any) -> None:
    assert entities.resolve_any("Andy Weir", store, "title").kind == "author"
    assert entities.resolve_any("Circe", store, "author").kind == "title"


def test_public_resolve_entities_is_unchanged(store: Any) -> None:
    assert entities.resolve_entities("Circe", store).kind == "title"
    out = entities.resolve_entities("madeline miller", store)
    assert out.kind == "author" and out.author == "Madeline Miller"
    assert entities.resolve_entities("xyzzy plugh", store).kind == "none"
