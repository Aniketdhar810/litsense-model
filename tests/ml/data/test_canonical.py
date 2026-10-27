"""data.canonical: editions merge into works, different works and volumes never do (CLAUDE.md
s7.4, s16.2; D28)."""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import ml.data.canonical as canonical
import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.goodreads  # noqa: F401  (registers the stage)
import ml.data.meta  # noqa: F401  (registers the stage)
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all
from ml.data.goodreads import GR_INDEX_SCHEMA
from ml.data.parsing import isbn10_to_13
from ml.data.titles import TitleRules

Config = dict[str, Any]
FICTION = ["Books", "Literature & Fiction", "Literary"]
_asins = (f"A{n:09d}" for n in itertools.count())


def edition(title: str, author: str | None, **extra: Any) -> dict[str, Any]:
    """One ``books_raw`` row with neutral defaults."""
    return {
        "parent_asin": next(_asins),
        "title": title,
        "author": author,
        "categories": FICTION,
        "isbn10": None,
        "isbn13": None,
        "page_source_isbn13": None,
        "pages": 300,
        "edition_kind": "print",
        "year": 2015,
        "image": None,
        "n_interactions": 10,
    } | extra


def merged(repo_config: Config, *rows: dict[str, Any], gr: pd.DataFrame | None = None) -> pd.Series:
    """Component root per edition after every merge rule."""
    params = repo_config["canonical"]
    gr = gr if gr is not None else pd.DataFrame(columns=GR_INDEX_SCHEMA.names)
    books = canonical.prepare(pd.DataFrame(rows), TitleRules.from_config(params), gr)
    works = canonical.merge(books, params, np.random.default_rng(0))[0]
    return pd.Series(works.roots(), index=[r["title"] for r in rows])


def same_work(roots: pd.Series) -> bool:
    return roots.nunique() == 1


def test_editions_merge(repo_config: Config) -> None:  # s16.2
    assert same_work(
        merged(
            repo_config,
            edition("Project Hail Mary", "Andy Weir", year=2021, format="Hardcover"),
            edition("Project Hail Mary: A Novel", "Andy Weir", year=2023),
            edition("Project Hail Mary (Large Print)", "Andy Weir", year=2021),
        )
    )
    roots = merged(
        repo_config,
        edition("Gone Girl", "Gillian Flynn"),
        edition("Gone Girl", "GILLIAN FLYNN"),
        edition("Sharp Objects (Book 2)", "Gillian Flynn"),  # a known volume beside unknown ones
    )
    assert roots["Gone Girl"].nunique() == 1 and roots.nunique() == 2  # regression: None, not NaN


def test_series_volumes_never_merge(repo_config: Config) -> None:  # s16.2
    roots = merged(
        repo_config,
        edition("Mistborn: The Final Empire", "Brandon Sanderson"),
        edition("Mistborn: The Well of Ascension", "Brandon Sanderson"),
        edition("Mistborn (Book 3)", "Brandon Sanderson"),
        edition("Mistborn", "Brandon Sanderson"),
    )
    assert roots.nunique() == 4  # "mistborn" is a word subset of "mistborn the final empire"
    assert roots["Mistborn (Book 3)"] != roots["Mistborn"]  # unknown volume vs volume 3


def test_boxsets_never_merge(repo_config: Config) -> None:  # s16.2
    shared = "9780765311788"
    roots = merged(
        repo_config,
        edition("The Mistborn Trilogy Box Set", "Brandon Sanderson", isbn13=shared),
        edition("Mistborn: The Final Empire", "Brandon Sanderson", isbn13=shared),
        edition("Mistborn: The Final Empire", "Brandon Sanderson"),
    )
    assert roots.nunique() == 2
    assert roots["The Mistborn Trilogy Box Set"] != roots["Mistborn: The Final Empire"].iloc[0]


def test_numbered_titles_never_merge(repo_config: Config) -> None:  # D33, run-2 finding
    roots = merged(
        repo_config,
        edition("NYPD Red 2", "James Patterson", year=2014),
        edition("NYPD Red 2", "James Patterson", year=2015),
        edition("NYPD Red 3", "James Patterson", year=2015),
        edition("NYPD Red 4", "James Patterson", year=2016),
        edition("Hell Divers II", "Nicholas Sansbury Smith"),
        edition("Hell Divers III", "Nicholas Sansbury Smith"),
        edition("Catch-22", "Joseph Heller"),
        edition("Catch-22 (Kindle Edition)", "Joseph Heller"),
    )
    assert roots["NYPD Red 2"].nunique() == 1
    assert roots.nunique() == 6  # NYPD Red 2 / 3 / 4, Hell Divers II / III, Catch-22


def test_subset_guard_and_other_authors(repo_config: Config) -> None:
    roots = merged(
        repo_config,
        edition("Alex Cross", "James Patterson"),
        edition("Alex Cross's Trial", "James Patterson"),  # token_set 100: a word subset
        edition("Circe", "Madeline Miller"),
        edition("Circe", "Peter Miller"),  # same block (surname), different author
        edition("The Shining Girls", "Lauren Beukes", year=2013),
        edition("The Shining Girl", "Lauren Beukes", year=2020),  # fuzzy match, 7-year gap
        edition("The Quiet Engine", None),
        edition("The Quiet Engine", None),  # D28: title merges need an author
    )
    assert roots.nunique() == 8


def test_kindle_page_source(repo_config: Config) -> None:
    print_isbn = isbn10_to_13("0316556343")
    roots = merged(
        repo_config,
        edition("Circe", "Madeline Miller", isbn10="0316556343"),
        edition(
            "Circe (Kindle Edition)",
            "Madeline Miller",
            page_source_isbn13=print_isbn,
            edition_kind="kindle",
            year=2018,
        ),
        edition(
            "Unrelated Ebook", "Someone Else", page_source_isbn13=print_isbn, edition_kind="kindle"
        ),
    )
    assert roots.iloc[0] == roots.iloc[1] != roots.iloc[2]


AICKMAN = (
    "Robert Aickman's strange stories are among the most unsettling in the language: a woman "
    "visits a house where nothing is quite what it seems, a traveller is stranded in a town that "
    "will not let him go, and ordinary lives tilt into the uncanny."
)
MISTBORN_1 = (
    "For a thousand years the ash fell and no flowers bloomed. The Lord Ruler reigns with absolute "
    "power, and a street thief named Vin is recruited by a crew planning an impossible heist."
)
MISTBORN_SERIES = (
    "The complete guide to the cosmere: maps, art and essays covering every era of the saga, "
    "from the Final Empire to the Wax and Wayne novels, with notes from the author."
)


def test_subtitle_rule_merges_a_bare_title_with_similar_blurbs(repo_config: Config) -> None:
    roots = merged(
        repo_config,
        edition("Cold Hand in Mine", "Robert Aickman", features=AICKMAN, year=2014),
        edition(
            "Cold hand in mine: Strange stories",
            "Robert Aickman",
            description=AICKMAN.replace("language", "English language"),
            year=1975,
        ),
    )
    assert same_work(roots)


def test_subtitle_rule_needs_similar_blurbs(repo_config: Config) -> None:
    roots = merged(
        repo_config,
        edition("Mistborn", "Brandon Sanderson", description=MISTBORN_SERIES),
        edition("Mistborn: The Final Empire", "Brandon Sanderson", description=MISTBORN_1),
    )
    assert roots.nunique() == 2


@pytest.mark.parametrize(
    ("bare", "long", "blurb_bare", "blurb_long"),
    [
        (
            ("Cold Hand in Mine", "Robert Aickman"),
            ("Cold Hand in Mine: Stories", "R Aickman"),
            1,
            1,
        ),
        (
            ("Cold Hand in Mine", "Robert Aickman"),
            ("Cold Hand in Mine: Stories", "Robert Aickman"),
            1,
            0,
        ),
        (
            ("Mistborn (Book 3)", "Brandon Sanderson"),
            ("Mistborn: The Final Empire", "Brandon Sanderson"),
            1,
            1,
        ),
        (
            ("Mistborn Box Set", "Brandon Sanderson"),
            ("Mistborn Box Set: Books 1-3", "Brandon Sanderson"),
            1,
            1,
        ),
    ],
    ids=["other_author", "empty_description", "volume_guard", "boxset"],
)
def test_subtitle_rule_guards(
    repo_config: Config,
    bare: tuple[str, str],
    long: tuple[str, str],
    blurb_bare: int,
    blurb_long: int,
) -> None:
    roots = merged(
        repo_config,
        edition(*bare, description=AICKMAN if blurb_bare else None),
        edition(*long, description=AICKMAN if blurb_long else ""),
    )
    assert roots.nunique() == 2


def test_subtitle_merges_get_their_own_audit_section(repo_config: Config) -> None:
    params = repo_config["canonical"]
    rules = TitleRules.from_config(params)
    rows = [
        edition("Cold Hand in Mine", "Robert Aickman", features=AICKMAN),
        edition("Cold hand in mine: Strange stories", "Robert Aickman", features=AICKMAN),
        edition("The Wine-Dark Sea", "Robert Aickman"),
    ]
    books = canonical.prepare(
        pd.DataFrame(rows), rules, pd.DataFrame(columns=GR_INDEX_SCHEMA.names)
    )
    rng = np.random.default_rng(0)
    works, near, subtitles = canonical.merge(books, params, rng)
    records = canonical.assign_works(books, works, [FICTION[1]], rules, drop_boxsets=True)
    assert subtitles.counts["merged"] == 1 and len(subtitles.edges) == 1
    audit = canonical.build_audit(books, records, near, subtitles, rng)
    section = audit[audit["audit_kind"] == "subtitle_merge"]
    assert list(section["audit_id"]) == ["S001", "S001"]
    assert (section["merge_rule"] == "subtitle").all()
    assert set(section["title"]) == {r["title"] for r in rows[:2]}


def test_shared_isbn_needs_the_same_author(repo_config: Config) -> None:
    shared = "9781590173466"  # audit G018: Amazon gives a crafts book a novel's ISBN
    roots = merged(
        repo_config,
        edition("The Long Ships", "Frans Gunnar Bengtsson", isbn13=shared),
        edition("Log Cabin Variations", "Annies", isbn13=shared),
        edition("The Long Ships (NYRB Classics)", None, isbn13=shared),  # unknown author: merges
    )
    assert roots.iloc[0] == roots.iloc[2] != roots.iloc[1]


def test_goodreads_works_join_and_separate(repo_config: Config) -> None:
    a, b, c = (isbn10_to_13(i) for i in ("0000000019", "0000000027", "0000000035"))
    gr = pd.DataFrame(
        [
            {"isbn13": a, "work_id": "1", "series_id": "9", "series_name": "S", "series_pos": 1.0},
            {"isbn13": b, "work_id": "1"},
            {"isbn13": c, "work_id": "2"},
        ]
    ).reindex(columns=GR_INDEX_SCHEMA.names)
    roots = merged(
        repo_config,
        edition("The Salt Orchard", "Daniel Okoro", isbn13=a),
        edition("A Garden of Salt", "Daniel Okoro", isbn13=b),  # different title, same GR work
        edition("The Salt Orchard", "Daniel Okoro", isbn13=c),  # same title, other GR work
        gr=gr,
    )
    assert roots.iloc[0] == roots.iloc[1] != roots.iloc[2]


def test_near_miss_reservoir_is_bounded() -> None:
    near = canonical.NearMisses(np.random.default_rng(0), size=5)
    for n in range(100):
        near.add({"reason": "subset_guard" if n % 4 else "volume", "a": n, "b": n + 1})
    assert near.counts == {"subset_guard": 75, "volume": 25}
    assert {r: len(s) for r, s in near.samples.items()} == {"volume": 5, "subset_guard": 5}


def test_volume_conflict_rule() -> None:
    assert not canonical.volume_conflict(None, None)
    assert not canonical.volume_conflict(1.0, None)
    assert not canonical.volume_conflict(2.0, 2.0)
    assert canonical.volume_conflict(2.0, None)
    assert canonical.volume_conflict(1.0, 2.0)


def load_outputs(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    paths = RunPaths.from_config(cfg, None)
    mapping = pd.read_parquet(paths.resolve("data:asin_to_canonical.parquet"))
    works = pq.read_table(paths.resolve("data:books_canonical.parquet"))
    assert works.schema.equals(canonical.CANONICAL_SCHEMA)
    report_path = paths.resolve("results:canonical/canonical_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    audit = pd.read_csv(paths.resolve("results:canonical/merge_audit.csv"), encoding="utf-8-sig")
    return mapping, works.to_pandas(), report, audit


@pytest.mark.parametrize("use_goodreads", [False, True])
def test_stage_on_the_tiny_fixture(
    tiny_cfg: Config, tiny_dataset: Path, use_goodreads: bool
) -> None:
    tiny_cfg["data"]["use_goodreads_graph"] = use_goodreads
    run_all(tiny_cfg, until="data.canonical")
    truth = json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))
    mapping, works, report, audit = load_outputs(tiny_cfg)
    work_of = mapping.set_index("parent_asin")["canonical_id"]
    status = mapping.set_index("parent_asin")["status"]

    books_raw = pd.read_parquet(
        RunPaths.from_config(tiny_cfg, None).resolve("data:books_raw.parquet")
    )
    assert sorted(mapping["parent_asin"]) == sorted(books_raw["parent_asin"])
    assert works["canonical_id"].is_unique
    assert set(works["canonical_id"]) == set(work_of[status == "kept"])
    assert (
        works["n_interactions"].sum()
        == books_raw.set_index("parent_asin")
        .loc[mapping.loc[mapping["status"] == "kept", "parent_asin"], "n_interactions"]
        .sum()
    )

    salt_print = truth["editions_goodreads"][0]
    print_isbn = truth["books"][salt_print[0]]["isbn10"]
    salt = [*salt_print, *(a for a, s in truth["kindle_editions"].items() if s == print_isbn)]
    assert work_of[salt].nunique() == 1 and len(salt) == 3
    lanterns = [*truth["editions_heuristic"][0], *truth["audiobooks"]]
    assert work_of[lanterns].nunique() == 1
    assert work_of[list(truth["series"]["volumes"])].nunique() == 3
    assert status[truth["boxsets"][0]] == "boxset"
    assert status[truth["out_of_slice"][0]] == "out_of_slice"
    assert status[truth["no_categories"][0]] == "out_of_slice"
    assert status[truth["no_author"][0]] == "kept"

    by_id = works.set_index("canonical_id")
    ember = by_id.loc[work_of[list(truth["series"]["volumes"])]]
    assert sorted(ember["series_pos"]) == [1.0, 2.0, 3.0]
    assert ember["series_id"].nunique() == 1
    salt_work = by_id.loc[work_of[salt[0]]]
    assert salt_work["n_members"] == 3 and salt_work["title"] == "The Salt Orchard"
    assert sorted(salt_work["edition_kinds"]) == ["kindle", "print"]
    assert salt_work["description"]  # filled from the editions' longest description
    if use_goodreads:
        assert salt_work.name.startswith("gr_") and ember.index.str.startswith("gr_").all()
        assert ember["series_id"].iloc[0] == "gr_" + truth["series"]["series_id"]
        novella_isbn = truth["series"]["novella_isbn10"]
        novella_kindle = next(a for a, s in truth["kindle_editions"].items() if s == novella_isbn)
        assert by_id.loc[work_of[novella_kindle], "series_pos"] == 1.5
    else:
        assert (works["canonical_id"].str.startswith("am_")).all()
        assert ember["series_name"].iloc[0] == "The Ember Cycle"

    assert report["editions"] == len(books_raw) and report["boxsets"] == 1
    assert report["kept"]["works"] == len(works)
    assert report["goodreads"]["enabled"] is use_goodreads
    assert list(audit.columns) == canonical.AUDIT_COLUMNS
    assert set(audit["audit_kind"]) <= {"merged_work", "not_merged", "subtitle_merge"}
    assert audit["verdict"].isna().all()
    years = pd.read_csv(
        RunPaths.from_config(tiny_cfg, None).resolve("results:canonical/merge_audit.csv"),
        encoding="utf-8-sig",
        dtype=str,
    )["year"]
    assert not years.str.endswith(".0", na=False).any()  # 2012, not 2012.0
    counts = read_manifest(RunPaths.from_config(tiny_cfg, None).manifest("data.canonical"))[
        "notes"
    ]["counts"]
    assert counts["works_kept"] == len(works)


def test_audit_is_seeded(tiny_cfg: Config) -> None:
    run_all(tiny_cfg, until="data.canonical")
    first = load_outputs(tiny_cfg)[3]
    run_all(tiny_cfg, until="data.canonical", force=True)
    pd.testing.assert_frame_equal(first, load_outputs(tiny_cfg)[3])


# Real editions from the Week 2 gate lookup (`python -m ml.data.meta --find` on the full Colab run):
# (expected work, title, author, edition kind).
REAL_EDITIONS = [
    ("phm", "Project Hail Mary: A Novel", "Andy Weir", "print"),
    ("phm", "Project Hail Mary", "Andy Weir", "audio"),
    ("martian", "The Martian", "Andy Weir", "print"),
    ("martian", "The Martian: A Novel", "Andy Weir", "print"),
    ("martian", "The Martian (Mass Market MTI): A Novel", "Andy Weir", "print"),
    ("chronicles", "The Martian Chronicles", "Ray Bradbury", "print"),
    ("chronicles", "The Martian Chronicles (The Grand Master Editions)", "Ray Bradbury", "print"),
    ("mistborn1", "Mistborn: The Final Empire (Book No. 1)", "Brandon Sanderson", "print"),
    ("mistborn1", "Mistborn: The Final Empire", "Brandon Sanderson", "print"),
    (
        "boxset",
        "Mistborn Trilogy Boxed Set (Mistborn, The Hero of Ages, & The Well of Ascension)",
        "Brandon Sanderson",
        "print",
    ),
    ("mistborn2", "The Well of Ascension (Mistborn, Book 2)", "Brandon Sanderson", "print"),
    (
        "mistborn3",
        "The Hero of Ages: Book Three of Mistborn (Mistborn, 3)",
        "Brandon Sanderson",
        "print",
    ),
    ("mistborn3", "The Hero of Ages (Mistborn, Book 3)", "Brandon Sanderson", "print"),
    (
        "alloy",
        "The Alloy of Law: A Mistborn Novel (The Mistborn Saga, 4)",
        "Brandon Sanderson",
        "print",
    ),
    (
        "shadows",
        "Shadows of Self: A Mistborn Novel (The Mistborn Saga, 5)",
        "Brandon Sanderson",
        "print",
    ),
    ("circe", "CIRCE", "Madeline Miller", "kindle"),
    ("circe", "Circe", "Madeline Miller", "print"),
    ("circe_penot", "Circe", "Jessica Penot", "kindle"),
    ("gone", "Gone Girl: A Novel", "Gillian Flynn", "kindle"),
    ("gone", "Gone Girl", "Gillian Flynn", "print"),
    ("gone", "Gone girl", "Gillian Flynn", "print"),
    (
        "complete",
        "The Complete Gillian Flynn: Gone Girl, Dark Places, Sharp Objects",
        "Gillian Flynn",
        "kindle",
    ),
    ("grownup", "The Grownup: A Story by the Author of Gone Girl", "Gillian Flynn", "print"),
    (
        "dark",
        "Dark Places: The New York Times bestselling phenomenon from the author of Gone Girl",
        "Gillian Flynn",
        "kindle",
    ),
    ("patient", "The Silent Patient", "Alex Michaelides", "kindle"),
    (
        "patient",
        "The Silent Patient: The record-breaking, multimillion copy Sunday Times bestselling "
        "thriller and TikTok sensation",
        "Alex Michaelides",
        "kindle",
    ),
    ("library", "The Midnight Library: A Novel", "Matt Haig", "print"),
]


def test_real_titles_from_the_gate_lookup(repo_config: Config) -> None:
    rows = [edition(t, a, edition_kind=k) for _, t, a, k in REAL_EDITIONS]
    roots = merged(repo_config, *rows).to_numpy()
    expected = [label for label, *_ in REAL_EDITIONS]
    for i, j in itertools.combinations(range(len(rows)), 2):
        same = expected[i] == expected[j]
        assert (roots[i] == roots[j]) == same, (REAL_EDITIONS[i][1], REAL_EDITIONS[j][1])
