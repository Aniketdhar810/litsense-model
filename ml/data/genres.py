"""Fixed genre labels from Amazon category paths (CLAUDE.md s7.4 step 8, docs/decisions.md D29).

A path such as ``["Books", "Science Fiction & Fantasy", "Fantasy", "Epic"]`` gets the labels of
its deepest element that matches any keyword, so "Fantasy" wins over the compound top level
"Science Fiction & Fantasy" (which would match both). Audience labels (Young Adult, Children's)
are added from any level. Within one element, ``GENRE_RULES`` order breaks ties.
"""

import re
from collections.abc import Iterable, Sequence

FALLBACK_GENRE = "Fiction (general)"
AUDIENCE_GENRES = ("Young Adult", "Children's")
GENRE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, re.compile(pattern, re.IGNORECASE))
    for label, pattern in (
        ("Young Adult", r"\bteen|young adult"),
        ("Children's", r"\bchildren"),
        ("Graphic Novels", r"graphic novel|comics|manga"),
        ("Fantasy", r"fantasy|sword & sorcery|dragons"),
        ("Science Fiction", r"science fiction|sci-fi|space opera|cyberpunk|dystopi|time travel"),
        ("Horror", r"horror|ghost"),
        ("Thriller", r"thriller|suspense|espionage|spy"),
        ("Mystery", r"myster|detective|crime|noir|cozy|sleuth"),
        ("Romance", r"romance|romantic|erotica"),
        ("Historical Fiction", r"historical"),
        ("Action & Adventure", r"action|adventure"),
        ("War & Military", r"military|\bwar\b"),
        ("Westerns", r"western"),
        ("Literary Fiction", r"\bliterary(?! criticism)"),
        ("Classics", r"classic"),
        ("Mythology & Folklore", r"mytholog|folk tale|folklore"),
        ("Short Stories", r"short stor|antholog"),
        ("Women's Fiction", r"women's fiction"),
        ("Humor & Satire", r"humou?r|satire"),
        ("Religious & Inspirational", r"christian|religious|inspirational"),
        ("LGBTQ+", r"lgbt|gay|lesbian"),
        ("Poetry", r"poetry"),
        ("Drama & Plays", r"drama|plays\b"),
        ("Biography & Memoir", r"biograph|memoir"),
        ("Criticism & Essays", r"criticism|essays"),
        ("Contemporary Fiction", r"contemporary"),
    )
)
GENRE_LABELS: tuple[str, ...] = (*(label for label, _ in GENRE_RULES), FALLBACK_GENRE)


def _labels(element: str) -> list[str]:
    return [label for label, pattern in GENRE_RULES if pattern.search(element)]


def path_genres(path: Sequence[str]) -> list[str]:
    """Labels of one category path, most specific first; empty when nothing matches."""
    body = list(path[1:]) if path and path[0] == "Books" else list(path)
    deepest: list[str] = next((found for e in reversed(body) if (found := _labels(e))), [])
    audience = [g for e in body for g in _labels(e) if g in AUDIENCE_GENRES]
    return list(dict.fromkeys([*deepest, *audience]))


def work_genres(paths: Iterable[Sequence[str]]) -> tuple[list[str], str]:
    """``(genres, primary_genre)`` of a work from its members' paths, ordered by importance (the
    caller passes the most-rated edition first). The primary genre is the first non-audience
    label, else the first label, else ``FALLBACK_GENRE``."""
    genres = list(dict.fromkeys(g for path in paths for g in path_genres(path)))
    primary = next((g for g in genres if g not in AUDIENCE_GENRES), None)
    primary = primary or (genres[0] if genres else FALLBACK_GENRE)
    return genres or [FALLBACK_GENRE], primary
