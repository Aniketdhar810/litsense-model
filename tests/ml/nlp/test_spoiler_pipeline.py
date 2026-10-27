"""nlp.spoiler_train with the Goodreads stream enabled, on the tiny fixture's spoiler file."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ml.data.clean as clean
import ml.nlp.sentences as sentences_module
import ml.nlp.sentiment as sentiment_module
from litsense_core.spoiler import SpoilerScorer
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all, run_stage
from ml.nlp import embed
from tests.ml.nlp.stubs import DIM, StubEncoder, stub_polarity, stub_splitter

Config = dict[str, Any]


def test_spoiler_training_on_the_fixture_stream(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    monkeypatch.setattr(clean, "load_language_id", lambda path: lambda text: ("en", 0.9))
    monkeypatch.setattr(embed, "load_encoder", lambda *args: StubEncoder())
    monkeypatch.setattr(sentences_module, "load_splitter", lambda *args, **kwargs: stub_splitter)
    monkeypatch.setattr(sentiment_module, "load_polarity_model", lambda *args: stub_polarity)
    tiny_cfg["data"]["kcore"] = {"user": 2, "item": 2}
    tiny_cfg["nlp"]["embedding_dim"] = DIM
    tiny_cfg["spoiler"] |= {
        "train_on_goodreads": True,
        "sample_size": 30,
        "holdout_fraction": 0.3,
        "C_grid": [1.0],
    }
    run_all(tiny_cfg, until="data.kcore_split")
    for name in ("nlp.sentences", "nlp.spoiler_train", "nlp.spoiler_flags"):
        assert run_stage(name, tiny_cfg) is True
    paths = RunPaths.from_config(tiny_cfg, None)
    scorer = SpoilerScorer.from_npz(paths.resolve("artifacts:serve/spoiler/spoiler_lr.npz"))
    assert scorer.has_model and scorer.coef.shape == (DIM,)
    metrics = json.loads(paths.resolve("results:nlp/spoiler_metrics.json").read_text("utf-8"))
    assert metrics["model"] == "logistic_regression" and metrics["C"] == 1.0
    assert metrics["sampled_spoiler"] == 15 and metrics["sampled_non_spoiler"] == 15
    assert metrics["reviews_streamed"] == 40  # the fixture's spoiler file
    assert metrics["holdout"]["recall"] >= tiny_cfg["spoiler"]["target_recall"] - 1e-9
    assert metrics["holdout"]["threshold"] == pytest.approx(scorer.threshold, abs=1e-4)
    assert 0 < metrics["holdout_rows"] < 30 and metrics["holdout_books"] >= 1
    assert read_manifest(paths.manifest("nlp.spoiler_train"))["notes"]["train_on_goodreads"]
    flags = pd.read_parquet(paths.resolve("data:sentence_spoiler.parquet"))
    sentences = pd.read_parquet(paths.resolve("data:sentences.parquet"))
    assert len(flags) == len(sentences) and flags["spoiler_p"].between(0, 1).all()
    assert (flags["spoiler_score"] >= flags["spoiler_p"] - 1e-6).all()
    # the hand check of s16.2 holds for the regex part even with a stub-trained model
    emb = StubEncoder().encode(["It turns out the narrator was dead the whole time."])
    assert scorer.score(emb, ["It turns out the narrator was dead the whole time."])[0] >= 0.15
    assert isinstance(np.asarray(flags["spoiler_flag"]).sum(), np.integer)
