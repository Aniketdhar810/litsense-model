"""Serve artifact set, bundle export and the real litsense_core over it (s12, D56, D63)."""

import gzip
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

import litsense_core as core  # noqa: E402
import ml.serve.export  # noqa: E402, F401  (registers the stages)
from litsense_core import candidates  # noqa: E402
from litsense_core import features as feats  # noqa: E402
from litsense_core import store as store_mod  # noqa: E402
from litsense_core.features import FEATURES  # noqa: E402
from litsense_core.llm.client import set_client  # noqa: E402
from litsense_core.profile import bucket_for, build_profile  # noqa: E402
from litsense_core.rank import rank_scores  # noqa: E402
from litsense_core.schemas import (  # noqa: E402
    Filters,
    ListEvent,
    OnboardingProfile,
    RatingEvent,
    ReviewAnalysis,
    UserState,
)
from litsense_core.vector_index import VectorIndex  # noqa: E402
from ml.common.manifest import read_manifest  # noqa: E402
from ml.common.stage import RunPaths, run_stage  # noqa: E402
from tests.litsense_core.bundle_build import build_tiny_bundle  # noqa: E402
from tests.ml.nlp.stubs import DIM  # noqa: E402

Config = dict[str, Any]
TS = 1_600_000_000_000


@pytest.fixture
def bundle(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Config, RunPaths]:
    """The whole pipeline through ``serve.export`` on the tiny fixture (``bundle_build``)."""
    paths = build_tiny_bundle(tiny_cfg, monkeypatch, tmp_path)
    store_mod.set_store(None)
    yield tiny_cfg, paths
    store_mod.set_store(None)


def test_vector_index_numpy_path_matches_brute_force() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(size=(40, 6)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    index = VectorIndex(m, use_faiss=False)
    q = m[3]
    ids, scores = index.search(q, 5, exclude=[3])
    brute = np.argsort(-(m @ q), kind="stable")
    assert ids.tolist() == [int(i) for i in brute if i != 3][:5]
    assert np.allclose(scores, (m @ q)[ids])
    allow_ids, _ = index.search(q, 3, allow=[10, 11, 12, 3], exclude=[3])
    assert set(allow_ids.tolist()) <= {10, 11, 12} and len(allow_ids) == 3
    assert index.search(q, 0)[0].size == 0


def test_union_round_robin_and_rank_scores() -> None:
    ids, prov = candidates.union({"cf": [1, 2, 3], "semantic": [2, 4], "popularity": [5, 1]}, 4)
    assert ids == [1, 2, 5, 3] and prov[2] == ["cf", "semantic"] and prov[1] == ["cf", "popularity"]
    assert rank_scores([1, 2, 9], [1, 2, 3, 4]).tolist() == pytest.approx([1.0, 0.75, 0.0])
    buckets = {"cold": [0, 0], "warming": [1, 4], "blending": [5, 19], "warm": [20, 10**6]}
    assert [bucket_for(n, buckets) for n in (0, 0.5, 1, 4.5, 5, 25)] == [
        "cold", "cold", "warming", "warming", "blending", "warm"]  # fmt: skip


def test_bundle_contents_and_manifest(bundle: tuple[Config, RunPaths]) -> None:
    cfg, paths = bundle
    folder = paths.resolve("artifacts:serve_bundle/")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    n_items = manifest["n_items"]
    for name, entry in manifest["files"].items():
        assert (folder / name).stat().st_size == entry["bytes"]
    arrays = np.load(folder / "catalog.npz")
    assert arrays["popularity"].shape == (n_items,) and arrays["is_longtail"].dtype == bool
    assert np.load(folder / "lgcn_item.f16.npy").dtype == np.float16
    assert np.load(folder / "content_item.f16.npy").shape == (n_items, DIM)
    # D63: the Phase 3 files travel with the bundle and leave the absent list
    for name in (
        "item_aspects.npy", "item_mention_rate.npy", "item_pos_rate.npy", "item_neg_rate.npy",
        "item_descriptors.npy", "aspect_means.npy", "aspect_topitems.json",
        "evidence_snippets.jsonl", "snippets_offsets.npy", "spoiler_lr.npz",
        "review_centroid_item.f16.npy",
        "aspect_prototypes.npy", "descriptor_poles.npy", "minilm.onnx", "tokenizer.json",
    ):  # fmt: skip
        assert name in manifest["files"], name
        assert name not in manifest["absent"]
    assert np.load(folder / "item_aspects.npy").shape == (n_items, 12)
    assert np.load(folder / "aspect_prototypes.npy").shape == (12, 8, DIM)
    assert manifest["onnx"]["quantized"] is False and manifest["onnx"]["source"] == "minilm.onnx"
    assert "minilm.int8.onnx" in manifest["absent"] and "proj_item.f16.npy" in manifest["absent"]
    assert manifest["config"]["aspects"]["names"][0] == "plot"
    # D66: the ranker ships with its feature list; corpus readers carry review centroids
    assert "ranker.txt" in manifest["files"] and "ranker.txt" not in manifest["absent"]
    assert json.loads((folder / "feature_list.json").read_text("utf-8")) == list(FEATURES)
    users = np.load(folder / "corpus_users.npz")
    assert users["centroid"].shape == (len(users["user_index"]), DIM)
    # D70: no key in the test config, so synthesis wrote nothing and the bundle keeps null
    # summaries; the llm config slice travels with the bundle (never the key).
    assert manifest["review_summaries"] == 0 and manifest["config"]["llm"]["provider"] == "gemini"
    assert manifest["config"]["llm"]["api_key_env"] == "LITSENSE_TEST_NO_KEY"
    assert "api_key" not in json.dumps(manifest["config"]["llm"]).replace("api_key_env", "")
    with gzip.open(folder / "catalog_text.json.gz", "rt", encoding="utf-8") as f:
        assert set(json.load(f)["review_summary"]) == {None}
    synth_meta = json.loads(
        paths.resolve("artifacts:serve/genai/meta.json").read_text(encoding="utf-8")
    )
    assert synth_meta["skipped"] == "no_api_key" and synth_meta["written"] == 0

    # The val bundle (ranker training) sits at t1 over train-only statistics, every user a
    # corpus reader, with no serve-only files.
    val = paths.resolve("artifacts:eval_val/bundle/")
    val_manifest = json.loads((val / "manifest.json").read_text(encoding="utf-8"))
    t1 = json.loads(paths.resolve("data:splits.json").read_text(encoding="utf-8"))["t1"]["ms"]
    lgcn_val = json.loads(
        paths.resolve("artifacts:eval_val/lightgcn/meta.json").read_text(encoding="utf-8")
    )
    assert val_manifest["cutoff"] == t1 == lgcn_val["cutoff"]
    assert val_manifest["onnx"] is None and "minilm.onnx" in val_manifest["absent"]
    assert (
        "ranker.txt" in val_manifest["absent"] and "aspect_topitems.json" in val_manifest["files"]
    )
    val_users = np.load(val / "corpus_users.npz")
    inter = pd.read_parquet(paths.resolve("data:interactions.parquet"))
    assert len(val_users["user_index"]) == inter.loc[inter["split"] == "train", "u"].nunique()
    assert np.load(val / "catalog.npz")["rating_count"].sum() == (inter["split"] == "train").sum()

    # Ranker training artifacts (s13.4, s13.5, s13.7)
    training = pd.read_parquet(paths.resolve("artifacts:eval_val/ranker/training.parquet"))
    assert list(training.columns[:7]) == ["u", "i", "label", "split", "cf_mode", "n_truth",
                                          "blend_score"]  # fmt: skip
    assert list(training.columns[7:]) == list(FEATURES)
    assert set(training["label"]) == {0, 1} and set(training["split"]) <= {"train", "valid"}
    assert set(training["cf_mode"]) <= {"exact", "pseudo"}
    assert (training["cf_available"][training["cf_mode"] == "exact"] == 1.0).all()
    assert (training["cf_available"][training["cf_mode"] == "pseudo"] == 0.5).all()
    assert training["u"].is_monotonic_increasing  # contiguous groups
    assert training.groupby("u")["label"].max().eq(1).all()  # every kept user has a positive
    assert training[["item_bayes_rating", "item_count_log", "n_sources"]].notna().all().all()
    assert (training[list(f"src_{s}" for s in candidates.SOURCE_ORDER)].sum(axis=1)
            == training["n_sources"]).all()  # fmt: skip
    recall = pd.read_csv(paths.resolve("results:ranker/candidate_recall.csv"))
    assert {"union", "union_after_filters"} <= set(recall["source"])
    assert 0 < recall.loc[recall["source"] == "union", "recall"].iloc[0] <= 1
    ranker_meta = json.loads(
        paths.resolve("artifacts:eval_val/ranker/meta.json").read_text(encoding="utf-8")
    )
    assert ranker_meta["cutoff"] == t1 and ranker_meta["best_iteration"] >= 1
    assert ranker_meta["features"] == list(FEATURES)
    importance = pd.read_csv(paths.resolve("results:ranker/importance.csv"))
    assert set(importance["feature"]) == set(FEATURES)
    assert importance["gain_share"].sum() == pytest.approx(1.0) or importance["gain"].sum() == 0
    vs_blend = pd.read_csv(paths.resolve("results:ranker/ranker_vs_blend.csv"))
    assert {"ranker", "blend", "cf_score_only"} <= set(vs_blend["model"])
    ablation = pd.read_csv(paths.resolve("results:ranker/ablation_val.csv"))
    assert {"M4", "M5", "M6", "M7", "M8", "M8a", "M8b", "blend"} == set(ablation["model"])
    assert (ablation["ndcg@10"].between(0, 1)).all()
    paired = pd.read_csv(paths.resolve("results:ranker/ablation_val_bootstrap.csv"))
    assert {"M4", "blend"} == set(paired["vs"])

    # The s8 protocol on every val user: same users as eval.val, baselines side by side
    full = pd.read_csv(paths.resolve("results:ranker/eval_val_main.csv"))
    val_main = pd.read_csv(paths.resolve("results:val/table_main.csv"))
    assert {"ranker", "ranker_pseudo_cf", "blend", "lightgcn_raw", "popularity"} <= set(
        full["model"]
    )
    for model in ("ranker", "ranker_pseudo_cf", "blend"):
        row = full[full["model"] == model].iloc[0]
        assert row["users"] == val_main["users"].iloc[0]
        assert 0 <= row["ndcg@10"] <= 1 and 0 < row["coverage@5"] <= 1
    for ref in ("lightgcn_raw", "popularity"):  # the stored rows reproduce eval.val's means
        assert full.loc[full["model"] == ref, "ndcg@10"].iloc[0] == pytest.approx(
            val_main.loc[val_main["model"] == ref, "ndcg@10"].iloc[0]
        )
    full_paired = pd.read_csv(paths.resolve("results:ranker/eval_val_bootstrap.csv"))
    assert set(full_paired["model"]) == {"ranker", "ranker_pseudo_cf", "blend", "ranker_home"}
    home = full[full["model"] == "ranker_home"].iloc[0]  # the rail can only add hits
    assert home["hit@10"] >= full.loc[full["model"] == "ranker", "hit@10"].iloc[0] - 1e-9
    assert {"lightgcn_raw", "lightgcn_norm", "popularity", "blend"} <= set(full_paired["vs"])
    summary = json.loads(paths.resolve("results:ranker/eval_val.json").read_text("utf-8"))
    assert summary["cutoff"] == t1 and summary["candidates_per_user"]["ranker"]["mean"] > 0
    assert summary["folds"] == 2 and len(summary["best_iteration_per_fold"]) == 2  # D67 cross-fit
    assert (training["item_age_years"].dropna() >= 0).all()  # D67: no books from the future
    assert pd.read_csv(paths.resolve("results:ranker/eval_val_strata.csv"))["model"].nunique() == 4

    # Filter diagnostic: recall is monotone in the review threshold; reasons add up
    assert run_stage("ranking.filter_diagnostic", cfg) is True
    diag = pd.read_csv(paths.resolve("results:ranker/filter_diagnostic.csv"))
    assert diag["min_reviews_main_feed"].tolist() == [0, 1, 2, 3, 5]
    assert diag["recall_after_filters"].is_monotonic_decreasing
    assert diag.loc[diag["current"], "min_reviews_main_feed"].iloc[0] == 0  # tiny cfg sets 0
    diag_json = json.loads(
        paths.resolve("results:ranker/filter_diagnostic.json").read_text("utf-8")
    )
    assert (
        sum(diag_json["loss_reasons_at_current_filter"].values()) == diag_json["positives_in_union"]
    )
    assert diag_json["union_recall"] >= diag["recall_after_filters"].max() - 1e-9
    serve_meta = json.loads(
        paths.resolve("artifacts:serve/lightgcn/meta.json").read_text(encoding="utf-8")
    )
    val_meta = json.loads(
        paths.resolve("artifacts:eval_val/lightgcn/meta.json").read_text(encoding="utf-8")
    )
    assert serve_meta["mode"] == "serve" and serve_meta["epochs"] == val_meta["best_epoch"]
    assert manifest["cutoff"] == serve_meta["cutoff"]
    assert read_manifest(paths.manifest("serve.export"))["cutoff"] == manifest["cutoff"]
    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    assert {m["id"] for m in metrics["models"]} >= {"popularity", "lightgcn_raw", "projection"}
    with tarfile.open(paths.resolve("artifacts:serve_bundle.tar.gz")) as tar:
        names = {m.name for m in tar.getmembers()}
    assert "serve_bundle/manifest.json" in names and "serve_bundle/minilm.onnx" in names


def test_real_core_recommends_from_the_bundle(
    bundle: tuple[Config, RunPaths], tmp_path: Path
) -> None:
    cfg, paths = bundle
    set_client(None)
    store = core.load_store(str(paths.resolve("artifacts:serve_bundle/")))
    assert core.load_store(str(paths.resolve("artifacts:serve_bundle/"))) is store  # singleton
    cat = store.catalog
    n = cat.n_items
    health = core.health()
    assert health["mock"] is False and health["n_books"] == n and health["ranker"] is True
    assert health["aspects"] and health["snippets"] and health["spoiler"] and health["encoder"]
    assert health["llm"] is False  # the bundle's llm slice names a key env var that is unset
    assert core.book_insights(0).review_summary is None
    assert store.aspect_names[0] == "plot" and store.aspect_tau.shape == (12,)

    # D66: the bundled ranker scores the main feed; the blend is the fallback
    cold = core.recommend(UserState(), k=5)
    assert cold.bucket == "cold" and 1 <= len(cold.items) <= 5 and cold.degraded is False
    assert cold.items[0].reason == "Popular with readers right now"
    assert all(i.rank == r for r, i in enumerate(cold.items, 1))
    assert all(i.evidence.signals["ranked_by"] == "ranker" for i in cold.items)
    assert core.recommend(UserState(), k=5, use_ranker=False).degraded is False
    ranker = store.ranker
    store.ranker = None
    try:
        assert core.recommend(UserState(), k=5).degraded is True
    finally:
        store.ranker = ranker

    # A reader who loved two books: every surfaced item is unseen and reasons are templates.
    liked = [int(i) for i in np.argsort(-cat.rating_count, kind="stable")[:2]]
    state = UserState(
        ratings=[RatingEvent(book_id=b, stars=5.0, ts=TS + k) for k, b in enumerate(liked)],
        reading_list=[ListEvent(book_id=liked[0], status="read", ts=TS)],
        profile_version=3,
    )
    warm = core.recommend(state, k=5, use_ranker=False)
    assert warm.bucket == "warming" and warm.profile_version == 3 and warm.degraded is False
    got = [i.book_id for i in warm.items]
    assert all(i.evidence.signals["ranked_by"] == "blend" for i in warm.items)

    # s13.3 feature table over this reader's candidates: shape, order, NaN only where allowed
    profile = build_profile(state, store)
    sources = candidates.generate(profile, store, store.config["quotas"][profile.bucket])
    ids, prov = candidates.union(sources, int(store.config["candidates_total"]))
    x = feats.assemble(profile, ids, prov, store)
    col = {f: k for k, f in enumerate(FEATURES)}
    assert x.shape == (len(ids), len(FEATURES)) and x.dtype == np.float32
    assert (x[:, col["cf_available"]] == 0.5).all()  # pseudo-CF reader
    assert np.isfinite(x[:, col["cf_score"]]).all() and np.isfinite(x[:, col["semantic_sim"]]).all()
    assert np.isnan(x[:, col["onboarding_sim"]]).all()  # no onboarding text
    assert np.isnan(x[:, col["proj_hybrid_score"]]).all()  # projection absent on v0
    assert (
        np.isfinite(x[:, col["max_item_sim"]]).all() and (x[:, col["max_item_sim"]] <= 1.01).all()
    )
    assert (x[:, col["top5_item_sim"]] <= x[:, col["max_item_sim"]] + 1e-6).all()
    assert np.isfinite(x[:, col["aspect_match"]]).all()
    assert np.allclose(x[:, col["user_activity_log"]], np.log1p(profile.n))
    assert (x[:, col["user_mean_rating"]] == 5.0).all()
    assert set(np.unique(x[:, col["disliked_genre_hit"]])) <= {0.0, 1.0}
    flags = x[:, col["src_cf"] : col["src_cf"] + len(candidates.SOURCE_ORDER)]
    assert (flags.sum(axis=1) == x[:, col["n_sources"]]).all() and flags.max() == 1.0
    for r, i in enumerate(ids):
        flagged = [s for k, s in enumerate(candidates.SOURCE_ORDER) if flags[r, k]]
        assert flagged == [s for s in candidates.SOURCE_ORDER if s in prov[i]]
    with pytest.raises(ValueError, match="cutoff"):
        feats.assemble(profile, ids, prov, store, expected_cutoff=store.cutoff - 1)
    assert feats.assemble(profile, [], {}, store).shape == (0, len(FEATURES))
    assert got and not set(got) & set(liked) and len(set(got)) == len(got)
    assert all(i.evidence.book["book_id"] == i.book_id for i in warm.items)
    assert all(i.sources for i in warm.items)
    assert all(i.evidence.signals["bucket"] == "warming" for i in warm.items)
    assert all(len(i.evidence.item_evidence["aspect_profile"]) == 12 for i in warm.items)

    # Exclusions and filters are honoured; a bad id raises ValueError (s15.3).
    again = core.recommend(state, k=5, exclude=got[:1])
    assert got[0] not in [i.book_id for i in again.items]
    narrow = core.recommend(
        state, k=5, filters=Filters(genres_include=[cat.primary_genre_of(got[0])])
    )
    assert all(
        cat.primary_genre_of(i.book_id) == cat.primary_genre_of(got[0]) for i in narrow.items
    )
    with pytest.raises(ValueError, match="book_id"):
        core.recommend(UserState(ratings=[RatingEvent(book_id=n + 5, stars=4.0, ts=TS)]))

    # Corpus demo user: exact LightGCN vector and the corpus history counts toward the bucket.
    demo = next(iter(store.corpus_users))
    demo_resp = core.recommend(UserState(corpus_user_id=demo), k=3)
    assert demo_resp.items and demo_resp.items[0].evidence.signals["cf_available"] == 1.0
    assert demo_resp.bucket != "cold"
    assert any("cf" in i.sources for i in demo_resp.items)
    assert demo_resp.items[0].reason != "Popular with readers right now"

    # D63: aspects, onboarding text, pinned aspects and review analyses flow into the profile.
    picky = UserState(
        ratings=state.ratings,
        onboarding=OnboardingProfile(positive_aspects=["humor"], descriptors={"darkness": -0.5}),
        onboarding_text="plot characters dark",
        review_analyses=[ReviewAnalysis(book_id=liked[0], ts=TS, aspects={"plot": 0.8})],
    )
    picky.overrides.pinned_aspects = ["characters"]
    aspect_resp = core.recommend(picky, k=5)
    assert aspect_resp.items
    first = aspect_resp.items[0].evidence
    assert first.user_evidence["top_positive_aspects"]  # humor +0.5 and plot from the review
    assert any("aspect" in i.sources or "semantic" in i.sources for i in aspect_resp.items)

    rails = core.home_rails(state)
    assert rails.top_picks and rails.trending and len(rails.because_you_liked) == 2
    assert all(r.anchor_book_id in liked for r in rails.because_you_liked)
    sim = core.similar_books(liked[0], k=4, state=state)
    assert (
        sim and liked[0] not in [i.book_id for i in sim] and sim[0].reason.startswith("Similar to")
    )
    grid = core.onboarding_grid(8)
    assert 1 <= len(grid) <= 8 and len(set(grid)) == len(grid)
    hits = core.semantic_search(cat.title[0].split()[0], k=5)
    assert hits and hits[0].matched in ("title", "author")
    free = core.semantic_search("plot characters dark", k=5)
    assert free and any(h.matched == "semantic" for h in free)
    assert core.resolve_entities(cat.title[1]).kind == "title"
    exp = core.explain(state, got[0])
    assert exp.method == "template" and exp.evidence.book["title"] == cat.title[got[0]]
    reply = core.chat_turn(state, None, f"something like {cat.title[liked[0]]}")
    assert reply.intent == "similar_to" and reply.session.turn == 1
    general = core.chat_turn(state, reply.session, "recommend a dark plot heavy book")
    assert general.intent == "recommend_general"
    assert core.metrics_report()["bundle_version"] == store.version
    insights = core.book_insights(got[0])
    assert len(insights.aspect_bars) == 12 and insights.n_reviews >= 0
    assert all(0 <= b.pos_rate <= 1 for b in insights.aspect_bars)
    analysis = core.analyze_user_review(
        "The plot kept me guessing. It turns out the butler did it."
    )
    assert isinstance(analysis, ReviewAnalysis) and analysis.spoiler_sentences >= 1

    # Phase 6 (D68, D70): with the fake provider the synthesis stage writes summaries, the export
    # carries them, and the package serves them and reports the LLM as available.
    cfg["llm"] |= {"enabled": True, "provider": "fake", "model": "fake", "model_quality": "fake-q",
                   "cache_path": str(tmp_path / "llm.sqlite")}  # fmt: skip
    cfg["genai"]["synthesis"]["top_n"] = 4
    assert run_stage("genai.synthesis", cfg) is True
    synth_meta = json.loads(
        paths.resolve("artifacts:serve/genai/meta.json").read_text(encoding="utf-8")
    )
    assert synth_meta["written"] == 4 and synth_meta["counts"]["llm"] == 4
    assert run_stage("serve.export", cfg) is True
    store_mod.set_store(None)
    set_client(None)
    store = core.load_store(str(paths.resolve("artifacts:serve_bundle/")))
    assert store.manifest["review_summaries"] == 4 and core.health()["llm"] is True
    summarised = [i for i in range(n) if cat.summary_of(i) is None and store.catalog.summary_of(i)]
    assert len(summarised) == 4
    assert core.book_insights(summarised[0]).review_summary == "A canned fake response."
    exp = core.explain(state, got[0])
    assert exp.method == "llm" and exp.text == "A fitting pick for this reader."
    # The canned chat reply lists no items, so the validators keep the template text.
    chat = core.chat_turn(state, None, f"something like {cat.title[liked[0]]}")
    assert chat.intent == "similar_to" and chat.method == "template" and chat.items
    assert core.parse_onboarding("funny mystery novels", []).method == "llm"
    set_client(None)
