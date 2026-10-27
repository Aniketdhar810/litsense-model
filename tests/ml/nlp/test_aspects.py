"""Aspect detection rule, lexicon, masks and descriptor scores (CLAUDE.md s11.2-11.3)."""

import numpy as np
import pytest

from ml.nlp import aspects
from ml.nlp.aspect_seeds import ASPECTS, DESCRIPTOR_POLES, LEXICON, PROTOTYPES, compile_lexicon
from tests.ml.nlp.stubs import StubEncoder

A = {name: k for k, name in enumerate(ASPECTS)}


def test_taxonomy_matches_config(repo_config: dict) -> None:
    aspects.check_taxonomy(repo_config["aspects"])
    assert len(ASPECTS) == 12 and all(len(PROTOTYPES[a]) == 8 for a in ASPECTS)
    assert all(LEXICON[a] for a in ASPECTS)
    assert tuple(DESCRIPTOR_POLES) == ("darkness", "pace", "complexity_level")
    with pytest.raises(ValueError, match="aspects.names"):
        aspects.check_taxonomy({**repo_config["aspects"], "names": list(reversed(ASPECTS))})


def test_tau_vector_uses_overrides() -> None:
    tau = aspects.tau_vector({"tau_default": 0.45, "tau": {"humor": 0.6}})
    assert tau.shape == (12,) and tau.dtype == np.float32
    assert tau[A["humor"]] == pytest.approx(0.6) and tau[A["plot"]] == pytest.approx(0.45)
    assert np.all(aspects.tau_vector({"tau_default": 0.5, "tau": None}) == 0.5)


def test_mask_roundtrip() -> None:
    rng = np.random.default_rng(0)
    hits = rng.random((50, 12)) < 0.3
    mask = aspects.pack_mask(hits)
    assert mask.dtype == np.uint16 and np.array_equal(aspects.unpack_mask(mask), hits)
    assert aspects.pack_mask(np.zeros((0, 12), dtype=bool)).shape == (0,)
    only_last = np.zeros((1, 12), dtype=bool)
    only_last[0, 11] = True
    assert aspects.pack_mask(only_last)[0] == 2**11


def test_lexicon_matches_whole_words_case_insensitively() -> None:
    texts = [
        "The STORY was great.",
        "A history of the war.",  # 'story' inside 'history' must not match
        "I loved the banter between them.",
        "World-building is immersive; the dialog sparkles.",
    ]
    lex = aspects.lexicon_hits(texts, compile_lexicon())
    assert lex[0, A["plot"]] and not lex[1].any()
    assert lex[2, A["dialogue"]] and not lex[2, A["plot"]]
    assert lex[3, A["world_building"]] and lex[3, A["dialogue"]]


def _toy_detector(dim: int = 4, tau: float = 0.5) -> aspects.AspectDetector:
    proto = np.zeros((12, 8, dim), dtype=np.float32)
    proto[:, :, 0] = 1.0  # every prototype points along axis 0 ...
    proto[A["humor"], 0] = [0, 1, 0, 0]  # ... except humor's first seed, along axis 1
    poles = np.zeros((3, 2, dim), dtype=np.float32)
    poles[:, 0, 2] = 1.0  # plus poles along axis 2, minus poles along axis 3
    poles[:, 1, 3] = 1.0
    return aspects.AspectDetector(proto, poles, np.full(12, tau, dtype=np.float32), dim)


def test_detection_rule_prototype_or_lexicon() -> None:
    det = _toy_detector()
    emb = np.array([[0, 1, 0, 0], [0, 0, 0.6, 0.8], [0, 0, 0.6, 0.8]], dtype=np.float32)
    texts = ["Nothing lexical here at all.", "Nothing lexical here either.", "Great banter!"]
    found = det.detect(emb, texts)
    assert found.sims.shape == (3, 12)
    assert found.hits[0, A["humor"]] and found.hits[0].sum() == 1  # prototype similarity 1.0
    assert not found.hits[1].any()  # below tau, no lexicon
    assert found.hits[2, A["dialogue"]] and found.lexicon[2, A["dialogue"]]  # lexicon only
    assert found.sims[2, A["dialogue"]] < det.tau[A["dialogue"]]
    with pytest.raises(ValueError, match="differ in length"):
        det.detect(emb, texts[:2])
    with pytest.raises(ValueError, match="aspect_prototypes shape"):
        aspects.AspectDetector(det.prototypes[:, :7], det.poles, det.tau, 4)


def test_descriptor_scores_only_where_the_linked_aspect_hits() -> None:
    det = _toy_detector()
    emb = np.array([[0, 0, 0.8, 0.6], [0, 0, 0.6, 0.8]], dtype=np.float32)
    hits = np.zeros((2, 12), dtype=bool)
    hits[0, A["tone_dark"]] = True  # darkness is linked to tone_dark
    hits[1, A["pacing"]] = True  # pace is linked to pacing
    scores = det.descriptor_scores(emb, hits)
    assert scores.shape == (2, 3) and scores.dtype == np.float32
    assert scores[0, 0] == pytest.approx(0.2) and np.isnan(scores[0, 1:]).all()
    assert scores[1, 1] == pytest.approx(-0.2) and np.isnan(scores[1, [0, 2]]).all()


def test_prototype_and_pole_embeddings_are_normalized() -> None:
    encoder = StubEncoder()
    proto = aspects.embed_prototypes(encoder)
    poles = aspects.embed_poles(encoder)
    assert proto.shape == (12, 8, encoder.dim) and poles.shape == (3, 2, encoder.dim)
    assert np.allclose(np.linalg.norm(proto, axis=2), 1, atol=1e-5)
    assert np.allclose(np.linalg.norm(poles, axis=2), 1, atol=1e-5)
    plot_first = encoder.encode([PROTOTYPES["plot"][0]])[0]
    assert np.allclose(proto[A["plot"], 0], plot_first, atol=1e-6)
    det = aspects.AspectDetector(proto, poles, np.full(12, 0.95, dtype=np.float32), encoder.dim)
    found = det.detect(encoder.encode([PROTOTYPES["ending"][2]]), ["I hated the cliffhanger."])
    assert found.hits[0, A["ending"]] and found.sims[0, A["ending"]] == pytest.approx(1.0, abs=1e-5)


def test_detection_frame_columns() -> None:
    det = _toy_detector()
    emb = np.array([[0, 1, 0, 0]], dtype=np.float32)
    frame = aspects.detection_frame(det, np.array([5]), emb, ["It made me laugh."])
    assert list(frame.columns) == aspects.ASPECT_SCHEMA.names
    assert frame["sent_id"].tolist() == [5] and frame["top_sim"].iloc[0] == pytest.approx(1.0)
    hits = aspects.unpack_mask(frame["aspect_mask"].to_numpy())[0]
    assert hits[A["humor"]] and np.isnan(frame["darkness"].iloc[0])


@pytest.mark.slow
def test_real_minilm_detects_writing_style(repo_config: dict) -> None:  # s16.2 NLP
    from ml.nlp import embed

    content = repo_config["content"]
    encoder = embed.load_encoder(content["model_name"], 32, int(content["max_seq_length"]))
    det = aspects.AspectDetector(
        aspects.embed_prototypes(encoder),
        aspects.embed_poles(encoder),
        aspects.tau_vector(repo_config["aspects"]),
        int(repo_config["nlp"]["embedding_dim"]),
    )
    text = "The sentences themselves are just gorgeous to read."
    found = det.detect(encoder.encode([text]), [text])
    assert found.hits[0, A["writing_style"]]
    assert found.sims[0, A["writing_style"]] >= det.tau[A["writing_style"]]  # not lexicon alone
