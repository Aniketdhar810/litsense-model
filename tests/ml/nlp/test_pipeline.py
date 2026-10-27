"""The Phase 3 stages end to end on the tiny fixture (CLAUDE.md s11.1-11.6) with stubbed models."""

import gzip
import json
import re
from typing import Any

import numpy as np
import pandas as pd
import pytest

import ml.data.clean as clean
import ml.nlp.sentences as sentences_module
import ml.nlp.sentiment as sentiment_module
from ml.common.manifest import read_manifest
from ml.common.stage import RunPaths, run_all, run_stage
from ml.nlp import aspects, embed
from ml.nlp.aspect_seeds import ASPECTS, PROTOTYPES
from ml.nlp.sentences import SENTENCE_SCHEMA
from tests.fixtures.tiny.make_tiny import ASPECT_SENTENCES, MIXED_SENTENCE
from tests.ml.nlp.stubs import DIM, StubEncoder, stub_polarity, stub_splitter

Config = dict[str, Any]
A = {name: k for k, name in enumerate(ASPECTS)}
STAGES = (
    "nlp.sentences",
    "nlp.aspects",
    "nlp.sentiment",
    "nlp.centroids_val",
    "nlp.centroids_serve",
    "nlp.profiles_val",
    "nlp.profiles_serve",
    "nlp.spoiler_train",
    "nlp.spoiler_flags",
    "nlp.snippets",
    "nlp.annotation_export",
)


@pytest.fixture
def nlp_cfg(tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> Config:
    """Tiny-fixture config with every model stubbed and small chunks so parts are exercised.

    The stub encoder gives identical texts cosine 1 and unrelated texts cosines near 0, so
    ``tau_default`` 0.95 means "the prototype sentence itself"; most of the fixture's review
    sentences are prototypes, so prototype detection is exercised without a real model.
    """
    monkeypatch.setattr(clean, "load_language_id", lambda path: lambda text: ("en", 0.9))
    monkeypatch.setattr(embed, "load_encoder", lambda *args: StubEncoder())
    monkeypatch.setattr(sentences_module, "load_splitter", lambda *args, **kwargs: stub_splitter)
    monkeypatch.setattr(sentiment_module, "load_polarity_model", lambda *args: stub_polarity)
    tiny_cfg["data"]["kcore"] = {"user": 2, "item": 2}
    tiny_cfg["nlp"]["embedding_dim"] = DIM
    tiny_cfg["nlp"]["chunk_rows"] = 64
    tiny_cfg["aspects"]["tau_default"] = 0.95
    return tiny_cfg


def test_stages_on_the_tiny_fixture(nlp_cfg: Config) -> None:
    run_all(nlp_cfg, until="data.kcore_split")
    for name in STAGES:
        assert run_stage(name, nlp_cfg) is True
    paths = RunPaths.from_config(nlp_cfg, None)

    # s11.1: the sentence table and its embeddings
    reviews = pd.read_parquet(paths.resolve("data:reviews.parquet"))
    sentences = pd.read_parquet(paths.resolve("data:sentences.parquet"))
    emb = np.load(paths.resolve("data:sent_emb.f16.npy"), mmap_mode="r")
    assert list(sentences.columns) == SENTENCE_SCHEMA.names and len(sentences) > 64  # > 1 part
    assert np.array_equal(sentences["sent_id"].to_numpy(), np.arange(len(sentences)))
    assert emb.shape == (len(sentences), DIM) and emb.dtype == np.float16
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-2)
    usable = reviews.loc[reviews["text_ok"], "review_row"]
    assert sentences["review_row"].isin(usable).all()
    tokens = sentences["text"].str.split().str.len()
    assert tokens.between(
        nlp_cfg["nlp"]["min_sentence_tokens"], nlp_cfg["nlp"]["max_sentence_tokens"]
    ).all()
    assert (sentences["pos"] < sentences["n_in_review"]).all()
    joined = sentences.merge(reviews[["review_row", "u", "i", "ts", "rating"]], on="review_row",
                             suffixes=("", "_r"))  # fmt: skip
    for col in ("u", "i", "ts", "rating"):
        assert (joined[col] == joined[f"{col}_r"]).all()
    manifest = read_manifest(paths.manifest("nlp.sentences"))
    assert manifest["notes"]["counts"]["sentences"] == len(sentences)
    assert manifest["notes"]["counts"]["reviews_text_ok"] == int(reviews["text_ok"].sum())
    assert not paths.work_dir("nlp.sentences").exists()  # checkpoints removed on completion

    # s11.2-11.3: prototypes, poles and detection
    proto = np.load(paths.resolve("artifacts:nlp/aspect_prototypes.npy"))
    poles = np.load(paths.resolve("artifacts:nlp/descriptor_poles.npy"))
    assert proto.shape == (12, 8, DIM) and poles.shape == (3, 2, DIM)
    hits_table = pd.read_parquet(paths.resolve("data:sentence_aspects.parquet"))
    assert hits_table["sent_id"].tolist() == sentences["sent_id"].tolist()
    hits = aspects.unpack_mask(hits_table["aspect_mask"].to_numpy())
    text_to_aspect = {t: a for a, pairs in ASPECT_SENTENCES.items() for t, _ in pairs}
    planted = sentences["text"].map(text_to_aspect)
    assert planted.notna().sum() > 50
    for row, aspect in zip(np.flatnonzero(planted.notna()), planted.dropna(), strict=True):
        assert hits[row, A[aspect]], (sentences["text"].iloc[row], aspect)
    # most planted sentences equal a prototype, so the prototype branch fires at cosine 1
    is_prototype = sentences["text"].isin([t for texts in PROTOTYPES.values() for t in texts])
    assert is_prototype.sum() > 50 and (hits_table.loc[is_prototype, "top_sim"] > 0.99).all()
    lexicon = aspects.unpack_mask(hits_table["lexicon_mask"].to_numpy())
    assert (hits | lexicon == hits).all()  # a lexicon hit is always a hit
    dark = hits[:, A["tone_dark"]]
    assert hits_table["darkness"].notna().to_numpy().tolist() == dark.tolist()
    checks = json.loads(paths.resolve("results:nlp/aspect_checks.json").read_text(encoding="utf-8"))
    assert checks["sentences"] == len(sentences) and set(checks["per_aspect"]) == set(ASPECTS)
    assert checks["sentences_with_hit"] == int(hits.any(axis=1).sum())

    # s11.4: polarity per (sentence, aspect), clause split for the mixed sentence
    polarity = pd.read_parquet(paths.resolve("data:aspect_polarity.parquet"))
    expected_rows = int(hits.sum())
    assert len(polarity) == expected_rows
    assert set(zip(polarity["sent_id"], polarity["aspect"], strict=True)) == {
        (int(sentences["sent_id"].iloc[r]), int(a)) for r, a in zip(*np.nonzero(hits), strict=True)
    }
    sign = {t: s for pairs in ASPECT_SENTENCES.values() for t, s in pairs}
    merged = polarity.merge(sentences[["sent_id", "text"]], on="sent_id")
    single = merged[merged["text"].isin(sign) & ~merged["from_clause"]]
    assert len(single) > 50
    assert (np.sign(single["polarity"]) == single["text"].map(sign)).all()
    mixed = merged[merged["text"] == MIXED_SENTENCE[0]]
    if len(mixed):  # the fixture plants it with probability 0.1 per review
        assert set(mixed["aspect"]) == {A["characters"], A["pacing"]}
        assert (mixed.loc[mixed["aspect"] == A["characters"], "polarity"] > 0).all()
        assert (mixed.loc[mixed["aspect"] == A["pacing"], "polarity"] < 0).all()
        assert mixed["from_clause"].all()
    sentiment_checks = json.loads(
        paths.resolve("results:nlp/sentiment_checks.json").read_text(encoding="utf-8")
    )
    assert sentiment_checks["aspect_mentions"] == expected_rows

    # s11.6: centroids per cutoff; manifests record the cutoff; no newer sentence is used
    splits = json.loads(paths.resolve("data:splits.json").read_text(encoding="utf-8"))
    t1 = splits["t1"]["ms"]
    val = paths.resolve("artifacts:eval_val/review_centroids/")
    serve = paths.resolve("artifacts:serve/review_centroids/")
    val_meta = json.loads((val / "meta.json").read_text(encoding="utf-8"))
    serve_meta = json.loads((serve / "meta.json").read_text(encoding="utf-8"))
    assert val_meta["cutoff"] == t1 == read_manifest(paths.manifest("nlp.centroids_val"))["cutoff"]
    assert serve_meta["cutoff"] == int(sentences["ts"].max()) + 1
    assert read_manifest(paths.manifest("nlp.centroids_serve"))["cutoff"] == serve_meta["cutoff"]
    assert val_meta["sentences_before_cutoff"] == int((sentences["ts"] < t1).sum())
    assert serve_meta["sentences_before_cutoff"] == len(sentences)
    ids = json.loads(paths.resolve("data:id_map.json").read_text(encoding="utf-8"))
    item = np.load(val / "review_centroid_item.npy")
    user = np.load(val / "review_centroid_user.npy")
    assert item.shape == (len(ids["items"]), DIM) and user.shape == (len(ids["users"]), DIM)
    norms = np.linalg.norm(item, axis=1)
    assert np.all((np.abs(norms - 1) < 1e-4) | (norms == 0))
    liked = sentences[(sentences["ts"] < t1) & (sentences["rating"] >= 4)]
    some_item = int(liked["i"].iloc[0])
    rows = liked.loc[liked["i"] == some_item, "sent_id"].to_numpy()
    mean = emb[np.sort(rows)].astype(np.float32).mean(axis=0)
    assert np.allclose(item[some_item], mean / np.linalg.norm(mean), atol=1e-3)
    cold = np.setdiff1d(np.arange(len(ids["items"])), liked["i"].unique())
    assert not item[cold].any()
    assert val_meta["items_with_centroid"] == int((norms > 0).sum())

    # s11.8: item and user aspect profiles per cutoff
    n_items, n_users = len(ids["items"]), len(ids["users"])
    val_asp = paths.resolve("artifacts:eval_val/aspects/")
    serve_asp = paths.resolve("artifacts:serve/aspects/")
    item_aspects = np.load(val_asp / "item_aspects.npy")
    assert item_aspects.shape == (n_items, 12) and item_aspects.dtype == np.float32
    for name in ("item_mention_rate.npy", "item_pos_rate.npy", "item_neg_rate.npy"):
        rates = np.load(val_asp / name)
        assert rates.shape == (n_items, 12) and rates.min() >= 0 and rates.max() <= 1
    descriptors = np.load(val_asp / "item_descriptors.npy")
    assert descriptors.shape == (n_items, 3) and np.abs(descriptors).max() <= 1
    assert np.load(val_asp / "aspect_means.npy").shape == (12,)
    assert np.load(val_asp / "user_aspects.npy").shape == (n_users, 12)
    conf = np.load(val_asp / "user_aspect_conf.npy")
    assert conf.shape == (n_users, 12) and conf.min() >= 0 and conf.max() < 1
    per_review = pd.read_parquet(val_asp / "user_review_aspects.parquet")
    assert (per_review["ts"] < t1).all() and per_review["u"].max() < n_users
    assert not per_review.duplicated(["review_row", "aspect"]).any()
    asp_meta = json.loads((val_asp / "meta.json").read_text(encoding="utf-8"))
    assert asp_meta["cutoff"] == t1 == read_manifest(paths.manifest("nlp.profiles_val"))["cutoff"]
    assert asp_meta["sentences_before_cutoff"] == int((sentences["ts"] < t1).sum())
    # an (item, aspect) cell with no mention before t1 sits exactly at the aspect mean
    means = np.load(val_asp / "aspect_means.npy")
    mentioned = per_review.loc[per_review["aspect"] == A["humor"], "i"].unique()
    silent = np.setdiff1d(np.arange(n_items), mentioned)
    assert len(silent) and np.allclose(item_aspects[silent, A["humor"]], means[A["humor"]])
    assert not np.allclose(item_aspects[mentioned, A["humor"]], means[A["humor"]])
    topitems = json.loads((serve_asp / "aspect_topitems.json").read_text(encoding="utf-8"))
    assert set(topitems) == set(ASPECTS)
    serve_n = pd.read_parquet(serve_asp / "user_review_aspects.parquet")
    assert len(serve_n) >= len(per_review)
    val_top = json.loads((val_asp / "aspect_topitems.json").read_text(encoding="utf-8"))
    assert set(val_top) == set(ASPECTS)  # D66: the val set has its own lists for ranker training

    # s11.7 with train_on_goodreads false: regex-only model, every sentence scored
    spoiler_flags = pd.read_parquet(paths.resolve("data:sentence_spoiler.parquet"))
    assert spoiler_flags["sent_id"].tolist() == sentences["sent_id"].tolist()
    assert (spoiler_flags["spoiler_p"] == 0).all()
    spoiler_re = re.compile(
        "|".join(f"(?:{p})" for p in nlp_cfg["spoiler"]["regex_patterns"]), re.I
    )
    regex_hit = sentences["text"].map(lambda t: spoiler_re.search(t) is not None).to_numpy()
    assert spoiler_flags["spoiler_flag"].to_numpy().tolist() == regex_hit.tolist()
    assert regex_hit.sum() > 0  # the fixture plants spoiler sentences
    spoiler_metrics = json.loads(
        paths.resolve("results:nlp/spoiler_metrics.json").read_text("utf-8")
    )
    assert spoiler_metrics["model"] == "regex_only"
    flag_report = json.loads(paths.resolve("results:nlp/spoiler_flags.json").read_text("utf-8"))
    assert flag_report["flagged"] == int(regex_hit.sum()) and flag_report["model"] == "regex_only"

    # s11.9: evidence snippets, serve only, spoiler-flagged sentences never shown
    snip_dir = paths.resolve("artifacts:serve/snippets/")
    with gzip.open(snip_dir / "evidence_snippets.json.gz", "rt", encoding="utf-8") as f:
        snips = json.load(f)
    snip_meta = json.loads((snip_dir / "meta.json").read_text(encoding="utf-8"))
    assert snip_meta["snippets"] > 0 and len(snips) == snip_meta["items_with_snippets"]
    limits = nlp_cfg["aspects"]["snippets"]
    for item, per_aspect in snips.items():
        assert 0 <= int(item) < n_items
        for aspect, slots in per_aspect.items():
            assert aspect in ASPECTS
            assert (
                len(slots["pos"]) <= limits["positive"] and len(slots["neg"]) <= limits["negative"]
            )
            for s in slots["pos"] + slots["neg"]:
                assert limits["min_chars"] <= len(s["text"]) <= limits["max_chars"]
                assert not spoiler_re.search(s["text"])
            assert all(s["polarity"] > nlp_cfg["nlp"]["pos_threshold"] for s in slots["pos"])
            assert all(s["polarity"] < nlp_cfg["nlp"]["neg_threshold"] for s in slots["neg"])
    assert snip_meta["dropped_spoiler"] > 0  # the fixture plants spoiler sentences

    # s11.5 step 1: the blank annotation sample (200 with a predicted hit, 100 without)
    csv = pd.read_csv(
        paths.resolve("results:nlp/annotation/aspects_300.csv"), keep_default_na=False
    )
    assert list(csv.columns) == ["sent_id", "text", *ASPECTS] and len(csv) == 300
    assert csv["sent_id"].is_unique and csv["sent_id"].isin(sentences["sent_id"]).all()
    predicted = hits_table.set_index("sent_id").loc[csv["sent_id"], "aspect_mask"] > 0
    assert int(predicted.sum()) == 200
    assert (csv[list(ASPECTS)] == "").all().all()

    # s11.5 steps 3-5: a "labelled" copy in the raw dir (labels = the fixture's planted aspects
    # for prototype sentences, blank elsewhere), a second labeller on the first 100 rows
    labelled = csv.copy()
    planted = {text: name for name, texts in PROTOTYPES.items() for text in texts}
    for k, text in enumerate(labelled["text"]):
        name = planted.get(text)
        if name is not None:
            labelled.loc[k, name] = "+"
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    labelled.to_csv(paths.raw_dir / "aspects_300.csv", index=False, lineterminator="\n")
    second = labelled.head(100).copy()
    second.loc[second.index[:5], "humor"] = "-"  # some disagreement
    second.to_csv(paths.raw_dir / "aspects_300_second.csv", index=False, lineterminator="\n")
    assert run_stage("nlp.validate_aspects", nlp_cfg) is True
    validation = json.loads(
        paths.resolve("results:nlp/aspect_validation.json").read_text(encoding="utf-8")
    )
    assert validation["sentences"] == 300 and validation["tune_rows"] == 150
    assert set(validation["proposed_tau"]) == set(ASPECTS)
    assert validation["agreement"]["shared_rows"] == 100
    humor_kappa = validation["agreement"]["kappa"]["humor"]
    assert humor_kappa is None or humor_kappa < 1  # the second labeller disagreed on humor
    table = pd.read_csv(paths.resolve("results:nlp/aspect_validation.csv"))
    assert list(table["aspect"]) == list(ASPECTS)
    scored = table[table["test_positives"] > 0]
    assert len(scored) >= 1 and (scored["recall"] > 0).all()  # prototypes are detectable

    # every stage is up to date afterwards
    for name in (*STAGES, "nlp.validate_aspects"):
        assert run_stage(name, nlp_cfg) is False
