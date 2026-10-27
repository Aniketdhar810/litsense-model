"""Item and user aspect profiles by hand (CLAUDE.md s11.8) and the shared litsense_core math."""

import numpy as np
import pandas as pd
import pytest

from litsense_core import aspects as shared
from ml.eval.errors import LeakageError
from ml.nlp import profiles_offline as prof
from ml.nlp.aspect_seeds import ASPECTS

A = {name: k for k, name in enumerate(ASPECTS)}
NA = len(ASPECTS)


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # item 0: review 10 (user 0, 0 votes) says plot +0.8 twice and pacing -0.9;
    #         review 11 (user 1, 9 votes -> weight 1 + ln 10) says plot -0.2.
    # item 1: review 12 (user 1) says plot +0.6. item 2 has a review but no mentions.
    sentences = pd.DataFrame(
        {
            "sent_id": [0, 1, 2, 3, 4, 5],
            "review_row": [10, 10, 10, 11, 12, 13],
            "u": [0, 0, 0, 1, 1, 2],
            "i": [0, 0, 0, 0, 1, 2],
            "ts": [100, 100, 100, 200, 300, 400],
            "rating": [5.0, 5.0, 5.0, 2.0, 4.0, 3.0],
            "helpful_vote": [0, 0, 0, 9, 0, 0],
        }
    )
    polarity = pd.DataFrame(
        {
            "sent_id": [0, 1, 2, 3, 4],
            "aspect": [A["plot"], A["plot"], A["pacing"], A["plot"], A["plot"]],
            "polarity": [0.8, 0.8, -0.9, -0.2, 0.6],
            "from_clause": [False] * 5,
        }
    ).astype({"aspect": "int8", "polarity": "float32"})
    descriptors = pd.DataFrame(
        {
            "sent_id": [0, 1, 2, 3, 4, 5],
            "darkness": [np.nan] * 6,
            "pace": [np.nan, np.nan, -0.4, np.nan, np.nan, np.nan],
            "complexity_level": [np.nan] * 6,
        }
    )
    return sentences, polarity, descriptors


def test_item_profiles_by_hand() -> None:
    sentences, polarity, descriptors = _tables()
    k = 2.0
    out = prof.build_item_profiles(sentences, polarity, descriptors, 3, 1000, k, 0.5, -0.5)
    w_votes = 1 + np.log1p(9)
    raw_plot_0 = (0.8 + 0.8 - 0.2 * w_votes) / (2 + w_votes)
    mean_plot = (0.8 + 0.8 - 0.2 * w_votes + 0.6) / (3 + w_votes)  # weighted over every mention
    assert out.means[A["plot"]] == pytest.approx(mean_plot, abs=1e-6)
    assert out.means[A["pacing"]] == pytest.approx(-0.9, abs=1e-6)
    assert out.n_mentioning[0, A["plot"]] == 2 and out.n_mentioning[0, A["pacing"]] == 1
    assert out.aspects[0, A["plot"]] == pytest.approx(
        (2 * raw_plot_0 + k * mean_plot) / (2 + k), abs=1e-6
    )
    assert out.aspects[1, A["plot"]] == pytest.approx((0.6 + k * mean_plot) / (1 + k), abs=1e-6)
    assert out.aspects[2, A["plot"]] == pytest.approx(mean_plot, abs=1e-6)  # no data: the mean
    assert out.aspects[2, A["humor"]] == 0.0  # an aspect nobody mentioned anywhere
    assert out.n_reviews.tolist() == [2, 1, 1]
    assert out.mention_rate[0, A["plot"]] == 1.0 and out.mention_rate[0, A["pacing"]] == 0.5
    assert out.mention_rate[2].sum() == 0
    assert (
        out.pos_rate[0, A["plot"]] == 0.5 and out.neg_rate[0, A["plot"]] == 0.0
    )  # -0.2 is neutral
    assert out.neg_rate[0, A["pacing"]] == 1.0
    assert out.descriptors[0, 1] == pytest.approx(-0.4 / (1 + k)) and out.descriptors[1].sum() == 0
    assert out.aspects.dtype == np.float32 and out.descriptors.shape == (3, 3)
    with pytest.raises(LeakageError):
        prof.build_item_profiles(sentences, polarity, descriptors, 3, 400, k, 0.5, -0.5)
    with pytest.raises(ValueError, match="id map"):
        prof.build_item_profiles(sentences, polarity, descriptors, 2, 1000, k, 0.5, -0.5)


def test_user_profiles_match_the_shared_online_function() -> None:
    sentences, polarity, _ = _tables()
    per_review = prof.review_aspect_means(polarity.merge(sentences, on="sent_id"))
    assert len(per_review) == 4  # (10, plot), (10, pacing), (11, plot), (12, plot)
    assert per_review.set_index(["review_row", "aspect"]).loc[
        (10, A["plot"]), "polarity"
    ] == pytest.approx(0.8)
    cutoff, half_life, conf_k = 86_400_000 * 1000, 730.0, 3.0
    vec, conf = prof.build_user_profiles(per_review, 3, cutoff, half_life, conf_k)
    assert vec.shape == (3, NA) and conf.shape == (3, NA)
    # user 1: review 11 (plot -0.2 at ts 200) and review 12 (plot +0.6 at ts 300)
    rows = per_review[per_review["u"] == 1]
    pol = np.full((2, NA), np.nan, dtype=np.float32)
    pol[np.arange(2), rows["aspect"].to_numpy()] = rows["polarity"].to_numpy()
    weights = shared.decay(cutoff - rows["ts"].to_numpy(), half_life)
    online_vec, online_conf = shared.explicit_aspect_vec(pol, weights, NA, conf_k)
    assert np.allclose(vec[1], online_vec, atol=1e-6) and np.allclose(conf[1], online_conf)
    assert conf[1, A["plot"]] == pytest.approx(2 / 5) and conf[2].sum() == 0
    assert vec[0, A["pacing"]] == pytest.approx(-0.9, abs=1e-6)
    with pytest.raises(LeakageError):
        prof.build_user_profiles(per_review, 3, 250, half_life, conf_k)


def test_shared_implicit_and_combine() -> None:
    item_aspects = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]], dtype=np.float32)
    means = np.array([0.5, 0.5], dtype=np.float32)
    vec = shared.implicit_aspect_vec(item_aspects, means, np.array([0, 1]), np.array([1.0, -0.5]))
    assert np.allclose(
        vec, (1.0 * (item_aspects[0] - means) - 0.5 * (item_aspects[1] - means)) / 1.5
    )
    assert shared.implicit_aspect_vec(item_aspects, means, np.array([], int), np.array([])) is None
    assert shared.combine(None, None, None, 0.6, n_aspects=2) is None
    explicit = np.array([1.0, 0.0], dtype=np.float32)
    conf = np.array([0.5, 0.0], dtype=np.float32)
    mixed = shared.combine(vec, explicit, conf, 0.6, positive=[1], n_aspects=2)
    assert mixed[0] == pytest.approx(0.3 * 1.0 + 0.7 * vec[0])
    assert mixed[1] == pytest.approx(min(1.0, vec[1] + 0.5))
    only_onboarding = shared.combine(None, None, None, 0.6, negative=[0], n_aspects=2)
    assert only_onboarding.tolist() == [-0.5, 0.0]
    assert shared.shrink(np.array([1.0]), np.array([0.0]), np.array([0.2]), 5.0)[
        0
    ] == pytest.approx(0.2)


def test_topitems_respects_min_reviews_and_order() -> None:
    sentences, polarity, descriptors = _tables()
    out = prof.build_item_profiles(sentences, polarity, descriptors, 3, 1000, 2.0, 0.5, -0.5)
    top = prof.aspect_topitems(out, n=5, min_reviews=1)
    assert top["plot"][0] == 1 and set(top["plot"]) == {0, 1}  # item 1 (+0.6) ranks above item 0
    assert top["pacing"] == [0] and top["humor"] == []
    assert prof.aspect_topitems(out, n=5, min_reviews=2)["plot"] == [0]
