"""Evidence snippet selection (CLAUDE.md s11.9): length, spoiler flag, polarity, ranking, caps."""

import pandas as pd

from ml.nlp import snippets
from ml.nlp.aspect_seeds import ASPECTS

A = {name: k for k, name in enumerate(ASPECTS)}
PARAMS = {"positive": 2, "negative": 1, "min_chars": 40, "max_chars": 220}


def _mentions() -> pd.DataFrame:
    long_ok = "The pacing of this thriller is relentless and I read it in a single evening."
    return pd.DataFrame(
        {
            "i": [0, 0, 0, 0, 0, 0, 0, 1],
            "aspect": [A["pacing"]] * 7 + [A["plot"]],
            "polarity": [0.9, 0.8, 0.95, 0.7, -0.8, -0.9, 0.2, 0.9],
            "text": [
                long_ok,  # kept, 5 votes
                "Fast paced and gripping, a real page-turner from the first chapter onwards.",
                "Great pacing.",  # too short
                "It turns out the pacing picks up only after the captain dies at sea.",  # spoiler
                "The middle third dragged on forever and I nearly gave up on the whole book.",
                "The middle third dragged on forever and I nearly gave up on the whole book.",
                "The pacing was okay I suppose, nothing special but not slow either really.",
                "A clever plot with twists that are fair to the reader and land every time.",
            ],
            "review_row": [1, 2, 3, 4, 5, 6, 7, 8],
            "helpful_vote": [5, 0, 9, 9, 1, 3, 0, 0],
            "spoiler_flag": [False, False, False, True, False, False, False, False],
        }
    ).astype({"aspect": "int8"})


def test_select_snippets_filters_ranks_and_caps() -> None:
    out, counts = snippets.select_snippets(_mentions(), PARAMS, 0.5, -0.5)
    pacing = out["0"]["pacing"]
    assert [s["review_row"] for s in pacing["pos"]] == [1, 2]  # votes first, both above threshold
    assert [s["review_row"] for s in pacing["neg"]] == [6]  # the duplicate text kept once (3 votes)
    assert pacing["pos"][0]["helpful_votes"] == 5 and pacing["pos"][0]["polarity"] == 0.9
    assert out["1"]["plot"]["pos"][0]["review_row"] == 8 and out["1"]["plot"]["neg"] == []
    assert counts == {
        "mentions": 8,
        "dropped_length": 1,
        "dropped_spoiler": 1,
        "dropped_neutral": 1,
        "snippets": 4,
        "items_with_snippets": 2,
    }
    out, _ = snippets.select_snippets(_mentions(), {**PARAMS, "positive": 1}, 0.5, -0.5)
    assert len(out["0"]["pacing"]["pos"]) == 1
