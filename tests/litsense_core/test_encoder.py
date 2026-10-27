"""Runtime ONNX encoder (CLAUDE.md s10.4) on a hand-built tiny model, and the parity math."""

import json
from pathlib import Path

import numpy as np
import pytest

from litsense_core.encoder import MiniLMEncoder
from ml.nlp import onnx_export

VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "hello": 4, "world": 5, "books": 6}
DIM = 4


def _tiny_model(path: Path) -> np.ndarray:
    """``last_hidden_state = table[input_ids]``: a Gather over a fixed (V, DIM) table."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    table = rng.normal(size=(len(VOCAB), DIM)).astype(np.float32)
    node = helper.make_node("Gather", ["table", "input_ids"], ["last_hidden_state"], axis=0)
    graph = helper.make_graph(
        [node],
        "tiny",
        [
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "sequence"]
            ),
        ],
        [
            helper.make_tensor_value_info(
                "last_hidden_state", TensorProto.FLOAT, ["batch", "sequence", DIM]
            )
        ],
        initializer=[numpy_helper.from_array(table, "table")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return table


def _tiny_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tok.save(str(path))


def test_encoder_mean_pools_with_the_mask_and_normalizes(tmp_path: Path) -> None:
    table = _tiny_model(tmp_path / "m.onnx")
    _tiny_tokenizer(tmp_path / "tok.json")
    enc = MiniLMEncoder(tmp_path / "m.onnx", tmp_path / "tok.json", max_len=8, threads=1)
    out = enc.encode(["hello world", "books"])
    assert out.shape == (2, DIM) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-6)
    expect0 = table[[2, 4, 5, 3]].mean(axis=0)
    expect1 = table[[2, 6, 3]].mean(axis=0)  # padding rows are masked out
    assert np.allclose(out[0], expect0 / np.linalg.norm(expect0), atol=1e-6)
    assert np.allclose(out[1], expect1 / np.linalg.norm(expect1), atol=1e-6)
    assert enc.count_tokens("hello world again") == 3 and enc.truncate("hello world", 1) == "hello"
    assert enc.encode([]).shape == (0, DIM)
    long = enc.encode(["hello " * 20])  # truncated to max_len
    assert long.shape == (1, DIM)


def test_parity_report_and_cosines() -> None:
    ref = np.eye(3, dtype=np.float32)
    good = ref * 2.0  # scale does not matter
    worse = ref.copy()
    worse[0] = [0.98, 0.2, 0.0]
    thresholds = {"fp32_min_cos": 0.9999, "int8_mean_cos": 0.99, "int8_min_cos": 0.98}
    report = onnx_export.parity_report(ref, good, worse, thresholds)
    assert report["fp32_ok"] and report["fp32"]["min_cos"] == pytest.approx(1.0)
    assert not report["int8_ok"] and report["int8"]["min_cos"] < 0.99
    assert json.dumps(report)  # JSON-serialisable (plain bools and floats)
    assert onnx_export.cosines(ref, -ref).tolist() == pytest.approx([-1, -1, -1])


def test_parity_texts_mix_prototypes_and_sample() -> None:
    import pandas as pd

    sentences = pd.DataFrame({"text": [f"review sentence {k}" for k in range(50)]})
    texts = onnx_export.parity_texts(sentences, 120, np.random.default_rng(0))
    assert len(texts) == 120 and texts[0] == "The plot kept me guessing until the very last page."
    assert sum(t.startswith("review sentence") for t in texts) == 24
    assert len(onnx_export.parity_texts(sentences, 10, np.random.default_rng(0))) == 10
