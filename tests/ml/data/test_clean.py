"""data.clean: cleaned review text, every row kept, text_ok and its reason (CLAUDE.md s7.6)."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

import ml.data.canonical  # noqa: F401  (registers the stage)
import ml.data.download  # noqa: F401  (registers the stage)
import ml.data.goodreads  # noqa: F401  (registers the stage)
import ml.data.meta  # noqa: F401  (registers the stage)
import ml.data.reviews  # noqa: F401  (registers the stage)
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all
from ml.data import clean

Config = dict[str, Any]
SPANISH = ("me encantó", "los personajes")


def fake_language_id(text: str) -> tuple[str, float]:
    """Stands in for fastText (not installed in CI)."""
    return ("es", 0.95) if any(w in text.lower() for w in SPANISH) else ("en", 0.9)


@pytest.fixture
def params(repo_config: Config) -> dict[str, Any]:
    data = repo_config["data"]
    return {k: data[k] for k in ("min_review_chars", "max_review_chars", "lid_min_confidence")}


@pytest.fixture
def shipping(repo_config: Config) -> Any:
    return clean.compile_patterns(repo_config["data"]["shipping_patterns"])


def test_clean_text() -> None:
    assert clean.clean_text("Great read.<br /><br />I&#39;d  read\nit again.") == (
        "Great read. I'd read it again."
    )
    assert clean.clean_text(None) == ""


def test_truncate_at_a_sentence_end() -> None:
    assert clean.truncate("Short.", 50) == ("Short.", False)
    text = "One sentence here. Two sentences here. Three sentences here."
    assert clean.truncate(text, 45) == ("One sentence here. Two sentences here.", True)
    assert clean.truncate("no sentence end at all in this text", 20) == ("no sentence end at", True)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("Loved it!", "too_short"),
        ("Me encantó este libro. Los personajes son maravillosos.", "non_english"),
        (
            "The book arrived damaged and the cover was bent. Very upset with the seller.",
            "shipping",
        ),
        ("Arrived damaged, but the characters were wonderful and the plot gripping.", None),
        ("I fell in love with the characters and the ending was perfect.", None),
    ],
)
def test_drop_reasons(params: dict[str, Any], shipping: Any, text: str, reason: str | None) -> None:
    assert clean.drop_reason(text, params, fake_language_id, shipping) == reason


def test_low_confidence_language_is_kept(params: dict[str, Any], shipping: Any) -> None:
    unsure = lambda text: ("fr", 0.3)  # noqa: E731
    assert clean.drop_reason("Great read, would recommend it.", params, unsure, shipping) is None


def test_stage_on_the_tiny_fixture(
    tiny_cfg: Config, tiny_dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(clean, "load_language_id", lambda path: fake_language_id)
    run_all(tiny_cfg, until="data.clean")
    truth = json.loads((tiny_dataset / "truth.json").read_text(encoding="utf-8"))
    paths = RunPaths.from_config(tiny_cfg, None)
    raw = pd.read_parquet(paths.resolve("data:reviews_raw.parquet"))
    table = pq.read_table(paths.resolve("data:reviews_clean.parquet"))
    assert table.schema.equals(clean.CLEAN_SCHEMA)
    rows = table.to_pandas()
    assert len(rows) == len(raw)  # every review keeps its row
    by_key = rows.set_index(rows["user_id"] + "|" + rows["parent_asin"])
    special = truth["special_reviews"]
    expected = {"shipping": "shipping", "non_english": "non_english", "too_short": "too_short"}
    for name, reason in expected.items():
        if name == "shipping" and special[name] not in by_key.index:
            continue  # the planted shipping review rates an out-of-slice book
        assert by_key.loc[special[name], "text_drop_reason"] == reason
        assert not by_key.loc[special[name], "text_ok"]
    cleaned = by_key.loc[special["html"], "text"]
    assert "<br" not in cleaned and "I'd" in cleaned
    assert (rows["text_ok"] == rows["text_drop_reason"].isna()).all()
    counts = read_manifest(paths.manifest("data.clean"))["notes"]["counts"]
    assert counts["reviews"] == len(rows) and counts["text_ok"] == int(rows["text_ok"].sum())
