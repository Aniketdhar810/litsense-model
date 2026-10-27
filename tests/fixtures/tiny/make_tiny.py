"""Deterministic synthetic raw data in the Amazon Reviews 2023 and UCSD Goodreads formats.

    python -m tests.fixtures.tiny.make_tiny --out local/tiny

Writes ``<out>/raw/`` using the file names the pipeline resolves for each source
(``raw_filename``) and ``<out>/truth.json`` describing every planted case, so data-stage tests can
assert exact outcomes. Field layouts follow CLAUDE.md s6, which is partly from memory: when
Phase 1 inspects the real files, update this generator to match them (rule 0.2.5).

Contents: 50 users, 80 rated books and 600 five-core interactions over 3 years, with reviews built
from template sentences whose aspects, polarities and spoiler labels are recorded. Planted cases:
a 3-book series (plus a #1.5 novella only in Goodreads), a box set, two editions of one work joined
through Goodreads and two of another that only the heuristics can merge, a book without author,
an author only in ``store``, ``details``/``author``/``images`` in their alternative encodings,
non-narrative titles, a too-short book, an out-of-slice category, empty categories, a child
``asin``, a cross-edition duplicate review, a shipping complaint, a non-English review, a too-short
review, HTML entities, 5-core pairs without a review and reviews outside the 5-core.

The serving-bundle half of CLAUDE.md s16.1 is added in Phase 4, when ``serve.export`` fixes the
bundle format.
"""

import argparse
import csv
import gzip
import io
import json
import random
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.common.config import load_config
from ml.common.stage import raw_filename

SEED = 2026
N_USERS = 50
N_BOOKS = 80
N_INTERACTIONS = 600
DAY_MS = 86_400_000
START_MS = 1_546_300_800_000  # 2019-01-01T00:00:00Z
SPAN_MS = 3 * 365 * DAY_MS

SLICES = {
    "sff": ["Books", "Science Fiction & Fantasy", "Fantasy", "Epic"],
    "scifi": ["Books", "Science Fiction & Fantasy", "Science Fiction", "Space Opera"],
    "thriller": ["Books", "Mystery, Thriller & Suspense", "Thrillers & Suspense", "Psychological"],
    "mystery": ["Books", "Mystery, Thriller & Suspense", "Mystery", "Cozy"],
    "literary": ["Books", "Literature & Fiction", "Literary"],
    "romance": ["Books", "Romance", "Contemporary"],
    "ya": ["Books", "Teen & Young Adult", "Science Fiction & Fantasy"],
    "cookbook": ["Books", "Cookbooks, Food & Wine", "Baking"],
}
FORMATS = ("Hardcover", "Paperback", "Mass Market Paperback")
MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
PUBLISHERS = ("Tor Books", "Orbit", "Penguin Books", "Vintage", "Harper Voyager", "Del Rey")
FIRST = (
    "Imogen",
    "Daniel",
    "Priya",
    "Rowan",
    "Tess",
    "Marcus",
    "Lena",
    "Omar",
    "Clara",
    "Hugo",
    "Signe",
    "Malik",
    "Rosa",
    "Felix",
    "Anya",
    "Bram",
)
LAST = (
    "Hale",
    "Okoro",
    "Raman",
    "Pike",
    "Arden",
    "Vale",
    "Strand",
    "Haddad",
    "Moss",
    "Brenner",
    "Lund",
    "Carver",
    "Ibarra",
    "Wren",
    "Petrov",
    "Quinn",
)
ADJ = (
    "Silver",
    "Sunken",
    "Northern",
    "Lost",
    "Scarlet",
    "Fallen",
    "Shattered",
    "Velvet",
    "Bitter",
    "Hollow",
    "Painted",
    "Restless",
)
NOUN = (
    "Compass",
    "Garden",
    "Lantern",
    "Kingdom",
    "Harbor",
    "Mirror",
    "Letters",
    "Season",
    "Tower",
    "Promise",
    "Archive",
    "Daughters",
)

# Template sentences: (text, {aspect: polarity}). Polarity sign is the label, magnitude is nominal.
ASPECT_SENTENCES: dict[str, tuple[tuple[str, int], ...]] = {
    "plot": (
        ("The plot kept me guessing until the very last page.", 1),
        ("The storyline was predictable and full of holes.", -1),
    ),
    "writing_style": (
        ("The prose is gorgeous and lyrical.", 1),
        ("The writing was clunky and amateurish.", -1),
    ),
    "characters": (
        ("I fell in love with the characters.", 1),
        ("The main character was annoying and hard to root for.", -1),
    ),
    "world_building": (
        ("The world-building is detailed and immersive.", 1),
        ("I never understood how this world worked.", -1),
    ),
    "pacing": (
        ("A real page-turner from start to finish.", 1),
        ("The middle dragged on forever.", -1),
    ),
    "editing_quality": (
        ("Well edited and cleanly produced.", 1),
        ("The book is full of typos and grammatical errors.", -1),
    ),
    "dialogue": (
        ("The dialogue is sharp and witty.", 1),
        ("The conversations felt stilted and unnatural.", -1),
    ),
    "humor": (("This book made me laugh out loud.", 1), ("The jokes fell flat for me.", -1)),
    "emotional_impact": (
        ("A deeply moving and emotional read.", 1),
        ("It left me feeling nothing at all.", -1),
    ),
    "ending": (("The ending was perfect.", 1), ("The ending felt rushed and unsatisfying.", -1)),
    "complexity": (
        ("A complex, layered novel that rewards attention.", 1),
        ("It was confusing and hard to follow.", -1),
    ),
    "tone_dark": (("Gritty and unflinching in the best way.", 1), ("Too depressing for me.", -1)),
}
MIXED_SENTENCE = (
    "Wonderful characters, but the middle third really dragged.",
    {"characters": 1, "pacing": -1},
)
FILLERS = (
    "I picked this up on a friend's recommendation.",
    "I read it over a long weekend.",
    "This was my book club's pick this month.",
    "I listened to part of it on a road trip and finished it in print.",
)
SPOILERS = (
    "It turns out the narrator was dead the whole time.",
    "In the end the captain dies saving the crew.",
    "The twist is that the two sisters were the same person.",
    "She ends up marrying the villain in the last chapter.",
    "The killer was the vicar all along.",
)
REVIEW_TITLES = {5: "Loved it", 4: "Really good", 3: "It was fine", 2: "Not for me", 1: "Skip it"}
SPECIAL_TEXTS = {
    "shipping": "The book arrived damaged and the cover was bent. Very upset with the seller.",
    "non_english": "Me encantó este libro. Los personajes son maravillosos y muy humanos.",
    "too_short": "Loved it!",
    "html": "Great read.<br /><br />I fell in love with the characters. I&#39;d read it again.",
}


def isbn10(rng: random.Random) -> str:
    """Random ISBN-10 in an English-language group, with a valid check digit."""
    body = rng.choice("01") + "".join(rng.choice("0123456789") for _ in range(8))
    total = sum((10 - i) * int(c) for i, c in enumerate(body))
    check = (11 - total % 11) % 11
    return body + ("X" if check == 10 else str(check))


def isbn13(isbn_10: str) -> str:
    """ISBN-13 (978 prefix) of an ISBN-10."""
    core = "978" + isbn_10[:9]
    total = sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(core))
    return core + str((10 - total % 10) % 10)


def kindle_style_asin(rng: random.Random) -> str:
    return "B0" + "".join(rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(8))


def date_text(rng: random.Random, year: int) -> str:
    return f"{rng.choice(MONTHS)} {rng.randint(1, 28)}, {year}"


def build_books(rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """80 book specs (planted cases first) and the truth about them."""
    specs: list[dict[str, Any]] = []

    def add(
        title: str,
        author: str | None,
        slice_: str,
        year: int,
        *,
        isbn: bool = True,
        pages: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        number = isbn10(rng) if isbn else None
        spec = {
            "title": title,
            "author": author,
            "categories": SLICES[slice_] if slice_ else [],
            "year": year,
            "pages": pages or rng.randint(220, 640),
            "isbn10": number,
            "asin": number if number and rng.random() < 0.85 else kindle_style_asin(rng),
            "format": rng.choice(FORMATS),
        } | extra
        specs.append(spec)
        return spec

    series = [
        add(
            "Ashfall (The Ember Cycle Book 1)",
            "Imogen Hale",
            "sff",
            2009,
            gr_pos=1.0,
            gr_title="Ashfall (The Ember Cycle, #1)",
        ),
        add(
            "Emberlight (The Ember Cycle Book 2)",
            "Imogen Hale",
            "sff",
            2011,
            gr_pos=2.0,
            gr_title="Emberlight (The Ember Cycle, #2)",
        ),
        add(
            "Cinderthrone: The Ember Cycle, Book 3",
            "Imogen Hale",
            "sff",
            2013,
            gr_pos=3.0,
            gr_title="Cinderthrone (The Ember Cycle, #3)",
        ),
    ]
    boxset = add(
        "The Ember Cycle Box Set: Books 1-3",
        "Imogen Hale",
        "sff",
        2015,
        pages=1450,
        gr_title="The Ember Cycle Boxed Set (The Ember Cycle, #1-3)",
    )
    gr_pair = [
        add(
            "The Salt Orchard",
            "Daniel Okoro",
            "literary",
            2012,
            format="Hardcover",
            gr_title="The Salt Orchard",
        ),
        add(
            "The Salt Orchard: A Novel",
            "Daniel Okoro",
            "literary",
            2013,
            format="Paperback",
            gr_title="The Salt Orchard",
        ),
    ]
    heuristic_pair = [
        add("Lanterns Over Kesh", "Priya Raman", "scifi", 2019, format="Hardcover"),
        add("Lanterns Over Kesh (Large Print)", "Priya Raman", "scifi", 2020, format="Paperback"),
    ]
    no_author = add("The Quiet Engine", None, "thriller", 2016)
    store_only = add(
        "Harbor of Glass",
        None,
        "mystery",
        2014,
        store="Ada Moss (Illustrator), Rowan Pike (Author)",
    )
    json_strings = add("The Winter Station", "Tess Arden", "thriller", 2017, json_strings=True)
    images_dict = add("A Season of Letters", "Lena Strand", "romance", 2018, images_as_dict=True)
    pub_date_key = add("Northern Promise", "Omar Haddad", "romance", 2010, publication_date=True)
    journal = add("My Gratitude Journal: 52 Weeks of Reflection", "Clara Moss", "literary", 2020)
    planner = add("Daily Planner 2021: Weekly and Monthly", "Clara Moss", "literary", 2020)
    too_short = add("Pip the Little Fox", "Hugo Brenner", "literary", 2015, pages=32)
    cookbook = add("Weeknight Baking", "Signe Lund", "cookbook", 2018)
    no_categories = add("The Painted Tower", "Malik Carver", "", 2011)
    untitled = add("Untitled Record", "Malik Carver", "literary", 2014, empty_title=True)
    # F1: Kindle and audio editions live in meta_Books; Kindle ones have no ISBN but often point to
    # the print edition through "Page numbers source ISBN"
    kindle_of_print = add(
        "The Salt Orchard",
        "Daniel Okoro",
        "literary",
        2012,
        isbn=False,
        edition="kindle",
        page_source=gr_pair[0]["isbn10"],
    )
    novella_isbn = isbn10(rng)  # the novella's print edition exists only in Goodreads
    kindle_of_gr_only = add(
        "Embers Between",
        "Imogen Hale",
        "sff",
        2010,
        isbn=False,
        edition="kindle",
        page_source=novella_isbn,
    )
    audiobook = add("Lanterns Over Kesh", "Priya Raman", "scifi", 2019, isbn=False, edition="audio")
    isbn13_field = add("The Ninth Lighthouse", "Signe Lund", "mystery", 2012, isbn13_in_isbn10=True)
    nameless = add("The Glass Atlas", "Sergio Ruzzier", "ya", 2016, author_without_name=True)
    for spec in specs:
        spec["planted"] = True

    names = [f"{f} {last}" for f in FIRST for last in LAST]
    rng.shuffle(names)
    authors = names[:24]
    titles = [f"The {a} {n}" for a in ADJ for n in NOUN]
    rng.shuffle(titles)
    fiction = [s for s in SLICES if s != "cookbook"]
    while len(specs) < N_BOOKS:
        add(
            titles.pop(),
            rng.choice(authors),
            rng.choice(fiction),
            rng.randint(1985, 2021),
            isbn=rng.random() < 0.9,
        )

    truth = {
        "series": {
            "series_id": "90001",
            "name": "The Ember Cycle",
            "volumes": {s["asin"]: s["gr_pos"] for s in series},
            "novella_only_in_goodreads": "Embers Between (The Ember Cycle, #1.5)",
            "novella_isbn10": novella_isbn,
        },
        "boxsets": [boxset["asin"]],
        "editions_goodreads": [[b["asin"] for b in gr_pair]],
        "editions_heuristic": [[b["asin"] for b in heuristic_pair]],
        "no_author": [no_author["asin"]],
        "store_author": {store_only["asin"]: "Rowan Pike"},
        "json_string_fields": [json_strings["asin"]],
        "images_as_dict": [images_dict["asin"]],
        "publication_date_key": [pub_date_key["asin"]],
        "non_narrative": {journal["asin"]: "journal", planner["asin"]: "planner"},
        "too_short": [too_short["asin"]],
        "out_of_slice": [cookbook["asin"]],
        "no_categories": [no_categories["asin"]],
        "empty_title": [untitled["asin"]],
        "kindle_editions": {
            b["asin"]: b["page_source"] for b in (kindle_of_print, kindle_of_gr_only)
        },
        "audiobooks": [audiobook["asin"]],
        "isbn13_in_isbn10_field": {isbn13_field["asin"]: isbn13(isbn13_field["isbn10"])},
        "author_without_name": {nameless["asin"]: "Sergio Ruzzier"},
    }
    return specs, truth


def meta_record(spec: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """One meta_Books.jsonl line for a book spec."""
    publisher = rng.choice(PUBLISHERS)
    details: dict[str, Any] = {
        "Publisher": f"{publisher}; 1st edition ({date_text(rng, spec['year'])})",
        "Language": "English",
        spec["format"]: f"{spec['pages']} pages",
        "Item Weight": f"{round(rng.uniform(0.4, 2.2), 2)} pounds",
        "Dimensions": "6 x 1.2 x 9 inches",
    }
    main_category, subtitle = "Books", f"{spec['format']} – {date_text(rng, spec['year'])}"
    categories, store = spec["categories"], spec.get("store")
    if spec.get("publication_date"):
        details["Publisher"] = publisher
        details["Publication date"] = date_text(rng, spec["year"])
    if spec["isbn10"]:
        details["ISBN 10"] = spec["isbn10"]
        details["ISBN 13"] = f"978-{isbn13(spec['isbn10'])[3:]}"
    if spec.get("isbn13_in_isbn10"):  # F3
        details["ISBN 10"] = isbn13(spec["isbn10"])
    if spec.get("edition") == "kindle":  # F1: real Kindle records carry no ISBN fields
        details = {
            "Publisher": publisher,
            "Publication date": date_text(rng, spec["year"]),
            "Language": "English",
            "File size": f"{rng.randint(800, 4000)} KB",
            "Print length": f"{spec['pages']} pages",
            "Page numbers source ISBN": spec["page_source"],
        }
        main_category, subtitle = "Buy a Kindle", "Kindle Edition"
        store = f"{spec['author']} (Author)   Format: Kindle Edition"
    if spec.get("edition") == "audio":
        details = {"Publisher": "Audible Studios", "Language": "English"}
        main_category, subtitle = "Audible Audiobooks", "Audible Audiobook – Unabridged"
        categories = ["Audible Books & Originals", *spec["categories"][1:3]]
    author: Any = None
    if spec["author"]:
        author = {
            "avatar": "https://example.invalid/avatar.jpg",
            "name": spec["author"],
            "about": [f"{spec['author']} writes novels."],
        }
    if spec.get("author_without_name"):  # F4: one real author dict had only an avatar
        author = {"avatar": "https://example.invalid/avatar.jpg"}
    large = f"https://example.invalid/images/{spec['asin']}.jpg"
    images: Any = [
        {
            "thumb": large.replace(".jpg", "_t.jpg"),
            "large": large,
            "variant": "MAIN",
            "hi_res": None,
        },
        {
            "thumb": None,
            "large": large.replace(".jpg", "_b.jpg"),
            "variant": "PT01",
            "hi_res": None,
        },
    ]
    if spec.get("images_as_dict"):
        images = {key: [img[key] for img in images] for key in images[0]}
    if spec.get("json_strings"):
        details, author = json.dumps(details), json.dumps(author)
    description = [
        f"<b>{spec['title'].split(':')[0]}</b> is a story about choices that echo for years.",
        "&quot;Unforgettable.&quot; &mdash; a reviewer",
    ]
    return {
        "main_category": main_category,
        "title": "" if spec.get("empty_title") else spec["title"],
        "subtitle": subtitle,
        "author": author,
        "average_rating": round(rng.uniform(3.2, 4.9), 1),
        "rating_number": rng.randint(5, 40_000),
        "features": [],
        "description": description,
        "price": round(rng.uniform(6, 30), 2),
        "images": images,
        "videos": [],
        "store": store or (f"{spec['author']} (Author)" if spec["author"] else None),
        "categories": categories,
        "details": details,
        "parent_asin": spec["asin"],
        "bought_together": None,
    }


def build_interactions(
    rng: random.Random, users: list[str], books: list[dict[str, Any]], truth: dict[str, Any]
) -> list[dict[str, Any]]:
    """600 unique (user, book) ratings; every user >= 5, every book >= 1, planted pairs first."""
    asins = [b["asin"] for b in books]
    weights = [1 / (r + 1) ** 0.7 for r in range(len(asins))]
    rng.shuffle(weights)
    salt_a, salt_b = truth["editions_goodreads"][0]
    ember = list(truth["series"]["volumes"])
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    def put(user: str, asin: str, ts: int | None = None, rating: float | None = None) -> None:
        if (user, asin) not in rows:
            stars = rating or rng.choices((5.0, 4.0, 3.0, 2.0, 1.0), weights=(50, 25, 12, 7, 6))[0]
            rows[(user, asin)] = {
                "user_id": user,
                "parent_asin": asin,
                "rating": stars,
                "timestamp": ts or START_MS + rng.randrange(SPAN_MS),
            }

    put(users[0], salt_a, START_MS + 100 * DAY_MS, 5.0)  # cross-edition duplicate review pair
    put(users[0], salt_b, START_MS + 400 * DAY_MS, 5.0)
    truth["cross_edition_reviewer"] = users[0]
    for i, asin in enumerate(ember):  # a reader working through the series in order
        put(users[1], asin, START_MS + (200 + 150 * i) * DAY_MS, 5.0)
    for i, asin in enumerate(asins):
        put(users[(i * 7) % len(users)], asin)
    for user in users:
        while sum(1 for u, _ in rows if u == user) < 5:
            put(user, rng.choices(asins, weights=weights)[0])
    activity = [1 / (r + 1) ** 0.8 for r in range(len(users))]
    while len(rows) < N_INTERACTIONS:
        put(rng.choices(users, weights=activity)[0], rng.choices(asins, weights=weights)[0])
    return sorted(rows.values(), key=lambda r: (r["user_id"], r["timestamp"]))


def review_text(rng: random.Random, rating: float) -> tuple[str, list[dict[str, Any]]]:
    """2-4 template sentences; returns the text and per-sentence truth."""
    sign = 1 if rating >= 4 else -1 if rating <= 2 else rng.choice((1, -1))
    sentences: list[dict[str, Any]] = []
    if rng.random() < 0.6:
        sentences.append({"text": rng.choice(FILLERS), "aspects": {}, "spoiler": False})
    if rng.random() < 0.1:
        sentences.append(
            {"text": MIXED_SENTENCE[0], "aspects": MIXED_SENTENCE[1], "spoiler": False}
        )
    for aspect in rng.sample(sorted(ASPECT_SENTENCES), rng.randint(1, 3)):
        polarity = sign if rng.random() < 0.8 else -sign
        text = next(t for t, p in ASPECT_SENTENCES[aspect] if p == polarity)
        sentences.append({"text": text, "aspects": {aspect: polarity}, "spoiler": False})
    if rng.random() < 0.12:  # spoilers cluster at the end of a review
        sentences.append({"text": rng.choice(SPOILERS), "aspects": {}, "spoiler": True})
    return " ".join(s["text"] for s in sentences), sentences


def review_record(rng: random.Random, row: dict[str, Any], text: str) -> dict[str, Any]:
    return {
        "rating": row["rating"],
        "title": REVIEW_TITLES[int(row["rating"])],
        "text": text,
        "images": [],
        "asin": row["parent_asin"],
        "parent_asin": row["parent_asin"],
        "user_id": row["user_id"],
        "timestamp": row["timestamp"],
        "helpful_vote": min(int(rng.paretovariate(1.3)) - 1, 250),
        "verified_purchase": rng.random() < 0.8,
    }


def build_reviews(
    rng: random.Random, interactions: list[dict[str, Any]], truth: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reviews for most 5-core pairs (with planted specials) plus reviews outside the 5-core."""
    key = "{user_id}|{parent_asin}".format
    salt_pair = set(truth["editions_goodreads"][0])
    duplicate_text, duplicate_truth = review_text(rng, 5.0)
    specials = iter(SPECIAL_TEXTS.items())
    reviews, review_truth, special_keys, missing = [], {}, {}, []
    for i, row in enumerate(interactions):
        k = key(**row)
        if row["user_id"] == truth["cross_edition_reviewer"] and row["parent_asin"] in salt_pair:
            text, sentences = duplicate_text, duplicate_truth
            special_keys.setdefault("cross_edition_duplicate", []).append(k)
        elif i % 97 == 13 and (special := next(specials, None)):
            name, text = special
            sentences = []
            special_keys[name] = k
        elif rng.random() < 0.08:
            missing.append(k)
            continue
        else:
            text, sentences = review_text(rng, row["rating"])
            if "child_asin" not in special_keys:  # filed under a child edition: join on parent_asin
                special_keys["child_asin"] = k
        record = review_record(rng, row, text)
        if special_keys.get("child_asin") == k:
            record["asin"] = kindle_style_asin(rng)
        reviews.append(record)
        if sentences:
            review_truth[k] = sentences
    outsiders = [f"AX{rng.randrange(16**24):024X}TINY" for _ in range(10)]
    asins = sorted({r["parent_asin"] for r in interactions})
    for j in range(30):
        row = {
            "user_id": outsiders[j % 10],
            "parent_asin": rng.choice(asins),
            "rating": float(rng.randint(1, 5)),
            "timestamp": START_MS + rng.randrange(SPAN_MS),
        }
        reviews.append(review_record(rng, row, review_text(rng, row["rating"])[0]))
    truth.update(
        reviews=review_truth,
        special_reviews=special_keys,
        pairs_without_review=missing,
        users_outside_5core=outsiders,
    )
    return reviews


def build_goodreads(
    rng: random.Random, books: list[dict[str, Any]], truth: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Goodreads books/series/works/spoiler-review records (string-typed values, as in the dump)."""
    gr_books, works = [], {}
    next_book, next_work = iter(range(1_000_001, 2_000_000)), iter(range(70_001, 80_000))

    def gr_book(
        title: str,
        isbn_10: str | None,
        year: int,
        pages: int,
        work: str,
        series: list[str],
        *,
        drop_isbn10: bool = False,
        drop_isbn13: bool = False,
    ) -> dict[str, Any]:
        shelves = [{"count": "120", "name": "to-read"}, {"count": "40", "name": "fantasy"}]
        record = {
            "isbn": "" if drop_isbn10 or not isbn_10 else isbn_10,
            "text_reviews_count": str(rng.randint(1, 900)),
            "series": series,
            "country_code": "US",
            "language_code": "eng",
            "popular_shelves": shelves,
            "asin": "",
            "is_ebook": "false",
            "average_rating": f"{rng.uniform(3, 4.8):.2f}",
            "kindle_asin": "",
            "similar_books": [],
            "description": "",
            "format": "Paperback",
            "link": "",
            "authors": [{"author_id": str(rng.randint(1000, 99999)), "role": ""}],
            "publisher": rng.choice(PUBLISHERS),
            "num_pages": str(pages),
            "publication_day": "",
            "isbn13": "" if drop_isbn13 or not isbn_10 else isbn13(isbn_10),
            "publication_month": "",
            "edition_information": "",
            "publication_year": str(year),
            "url": "",
            "image_url": "",
            "book_id": str(next(next_book)),
            "ratings_count": str(rng.randint(10, 50_000)),
            "work_id": work,
            "title": title,
            "title_without_series": title.split(" (")[0],
        }
        gr_books.append(record)
        return record

    def work_for(original_title: str, year: int) -> str:
        work = str(next(next_work))
        works[work] = {
            "work_id": work,
            "best_book_id": "",
            "original_title": original_title,
            "original_publication_year": str(year),
            "books_count": "1",
            "ratings_count": str(rng.randint(10, 50_000)),
            "media_type": "book",
        }
        return work

    matched: dict[str, str] = {}
    series_id = truth["series"]["series_id"]
    by_asin = {b["asin"]: b for b in books}
    for asin in truth["series"]["volumes"]:
        b = by_asin[asin]
        work = work_for(b["gr_title"].split(" (")[0], b["year"])
        rec = gr_book(b["gr_title"], b["isbn10"], b["year"], b["pages"], work, [series_id])
        works[work]["best_book_id"] = rec["book_id"]
        matched[asin] = work
    novella = work_for("Embers Between", 2010)
    novella_isbn = truth["series"]["novella_isbn10"]  # also the Kindle edition's source ISBN
    gr_book(
        truth["series"]["novella_only_in_goodreads"], novella_isbn, 2010, 96, novella, [series_id]
    )
    box = by_asin[truth["boxsets"][0]]
    matched[box["asin"]] = work_for("The Ember Cycle Boxed Set", 2015)
    gr_book(box["gr_title"], box["isbn10"], 2015, 1450, matched[box["asin"]], [series_id])
    salt_work = work_for("The Salt Orchard", 2012)
    for asin in truth["editions_goodreads"][0]:
        b = by_asin[asin]
        gr_book(b["gr_title"], b["isbn10"], b["year"], b["pages"], salt_work, [])
        matched[asin] = salt_work
    regular = [b for b in books if not b.get("planted") and b["isbn10"] and b["year"] <= 2017]
    for n, b in enumerate(regular[:15]):  # 0-4 match on isbn10 only, 5-9 on isbn13 only
        work = work_for(b["title"], b["year"])
        gr_book(
            b["title"],
            b["isbn10"],
            b["year"],
            b["pages"],
            work,
            [],
            drop_isbn13=n < 5,
            drop_isbn10=5 <= n < 10,
        )
        matched[b["asin"]] = work
    for n in range(10):  # Goodreads books that are not in the Amazon slice
        gr_book(
            f"Unrelated Book {n}", isbn10(rng), 2005 + n, 300, work_for(f"Unrelated {n}", 2005), []
        )

    series = [
        {
            "numbered": "true",
            "note": "",
            "description": "",
            "title": "The Ember Cycle",
            "series_works_count": "5",
            "series_id": series_id,
            "primary_work_count": "3",
        }
    ]
    spoiler_reviews = []
    for n in range(40):
        lines = [[0, rng.choice(FILLERS)]]
        lines += [
            [0, rng.choice(ASPECT_SENTENCES[a])[0]] for a in rng.sample(sorted(ASPECT_SENTENCES), 2)
        ]
        if n % 2 == 0:
            lines.append([1, rng.choice(SPOILERS)])
        spoiler_reviews.append(
            {
                "user_id": f"{rng.randrange(16**32):032x}",
                "timestamp": f"2017-0{1 + n % 9}-1{n % 10}",
                "review_sentences": lines,
                "rating": rng.randint(1, 5),
                "has_spoiler": n % 2 == 0,
                "book_id": rng.choice(gr_books)["book_id"],
                "review_id": f"{rng.randrange(16**32):032x}",
            }
        )
    truth["goodreads_matched"] = matched
    truth["goodreads_isbn10_only"] = [b["asin"] for b in regular[:5]]
    truth["goodreads_isbn13_only"] = [b["asin"] for b in regular[5:10]]
    return {
        "goodreads_books.json.gz": gr_books,
        "goodreads_book_series.json.gz": series,
        "goodreads_book_works.json.gz": list(works.values()),
        "goodreads_reviews_spoiler.json.gz": spoiler_reviews,
    }


def write_gz_lines(path: Path, lines: Iterable[str]) -> None:
    """Gzip with a zeroed header timestamp so the bytes are reproducible."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        for line in lines:
            gz.write(line.encode("utf-8") + b"\n")


def make_tiny(out_dir: Path, seed: int = SEED) -> dict[str, Any]:
    """Write the tiny raw dataset under ``out_dir/raw`` and return (and save) its truth."""
    rng = random.Random(seed)
    sources = load_config()["sources"]
    raw = out_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    books, truth = build_books(rng)
    users = [f"AE{rng.randrange(36**26):026X}"[:28] for _ in range(N_USERS)]
    interactions = build_interactions(rng, users, books, truth)
    reviews = build_reviews(rng, interactions, truth)
    goodreads = build_goodreads(rng, books, truth)
    extra_meta = [
        dict(meta_record(b, rng), title=f"Unrated {b['title']}", parent_asin=kindle_style_asin(rng))
        for b in books[-5:]
    ]

    ratings_csv = io.StringIO()
    writer = csv.DictWriter(
        ratings_csv, ["user_id", "parent_asin", "rating", "timestamp"], lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(interactions)
    write_gz_lines(
        raw / raw_filename(sources, "ratings_5core"), ratings_csv.getvalue().splitlines()
    )
    meta = [meta_record(b, rng) for b in books] + extra_meta
    write_gz_lines(raw / raw_filename(sources, "meta_books"), (json.dumps(m) for m in meta))
    write_gz_lines(raw / raw_filename(sources, "reviews_books"), (json.dumps(r) for r in reviews))
    for name, records in goodreads.items():
        write_gz_lines(raw / name, (json.dumps(r) for r in records))

    first, last = (
        min(r["timestamp"] for r in interactions),
        max(r["timestamp"] for r in interactions),
    )
    truth.update(
        seed=seed,
        schema_note="Layouts follow CLAUDE.md s6 (partly from memory); verify against real files.",
        counts={
            "users": N_USERS,
            "books": N_BOOKS,
            "interactions": len(interactions),
            "reviews_5core": len(reviews) - 30,
            "reviews_outside_5core": 30,
            "meta_records": len(meta),
            "meta_not_rated": len(extra_meta),
        },
        time_range=[datetime.fromtimestamp(t / 1000, UTC).isoformat() for t in (first, last)],
        books={
            b["asin"]: {
                "title": b["title"],
                "author": b["author"],
                "isbn10": b["isbn10"],
                "year": b["year"],
                "pages": b["pages"],
            }
            for b in books
        },
    )
    (out_dir / "truth.json").write_text(
        json.dumps(truth, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("local/tiny"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    truth = make_tiny(args.out, args.seed)
    print(f"wrote {args.out / 'raw'}: {truth['counts']}")


if __name__ == "__main__":
    main()
