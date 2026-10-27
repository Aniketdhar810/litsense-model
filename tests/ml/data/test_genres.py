"""Genre labels from Amazon category paths (CLAUDE.md s7.4 step 8, D29)."""

from ml.data.genres import FALLBACK_GENRE, GENRE_LABELS, path_genres, work_genres


def test_deepest_level_wins_over_a_compound_top_level() -> None:
    assert path_genres(["Books", "Science Fiction & Fantasy", "Fantasy", "Epic"]) == ["Fantasy"]
    assert path_genres(["Books", "Science Fiction & Fantasy", "Science Fiction"]) == [
        "Science Fiction"
    ]
    assert path_genres(["Books", "Mystery, Thriller & Suspense", "Thrillers & Suspense"]) == [
        "Thriller"
    ]


def test_audience_labels_come_from_any_level() -> None:
    path = ["Books", "Teen & Young Adult", "Science Fiction & Fantasy", "Fantasy"]
    assert path_genres(path) == ["Fantasy", "Young Adult"]
    assert work_genres([path]) == (["Fantasy", "Young Adult"], "Fantasy")
    assert work_genres([["Books", "Teen & Young Adult"]]) == (["Young Adult"], "Young Adult")


def test_fallback_and_ordering() -> None:
    assert path_genres(["Books", "Literature & Fiction", "Genre Fiction"]) == []
    assert work_genres([["Books", "Literature & Fiction", "Genre Fiction"]]) == (
        [FALLBACK_GENRE],
        FALLBACK_GENRE,
    )
    assert work_genres([]) == ([FALLBACK_GENRE], FALLBACK_GENRE)
    genres, primary = work_genres(
        [
            ["Books", "Literature & Fiction", "Literary"],  # the most-rated edition comes first
            ["Books", "Mystery, Thriller & Suspense", "Mystery"],
        ]
    )
    assert (genres, primary) == (["Literary Fiction", "Mystery"], "Literary Fiction")
    assert "History & Criticism" not in GENRE_LABELS
    assert path_genres(["Books", "Literature & Fiction", "History & Criticism"]) == [
        "Criticism & Essays"
    ]
