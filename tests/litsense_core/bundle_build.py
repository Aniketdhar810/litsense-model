"""The tiny serving bundle built by the real pipeline, shared by test_bundle.py (per test) and
tests/service (once per session): stubbed MiniLM, spaCy and DistilBERT for the data and NLP
stages, a hand-built ONNX model for ``nlp.onnx``. Needs torch (CI installs the CPU wheel).

Import this module only after ``ml.stages.load_stages()``: stages run in registration order,
and importing ``ml.nlp.onnx_export`` first would register ``nlp.onnx`` before ``eval.val``."""

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from litsense_core.encoder import MiniLMEncoder
from ml.common.stage import RunPaths, run_all, run_stage
from ml.data import clean
from ml.nlp import embed, onnx_export
from ml.nlp import sentences as sentences_module
from ml.nlp import sentiment as sentiment_module
from tests.conftest import rooted_config
from tests.ml.nlp.stubs import DIM, StubEncoder, stub_polarity, stub_splitter

Config = dict[str, Any]
NLP_STAGES = (
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
)
VOCAB = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "plot": 4, "characters": 5, "dark": 6}


def tiny_onnx(folder: Path) -> None:
    """A Gather-only ONNX model with ``DIM``-d rows and a word-level tokenizer, so the ONNX stage
    and the runtime encoder run without MiniLM (same construction as test_encoder)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    table = np.random.default_rng(0).normal(size=(len(VOCAB), DIM)).astype(np.float32)
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
    onnx.save(model, str(folder / "minilm.onnx"))
    shutil.copyfile(folder / "minilm.onnx", folder / "minilm.int8.onnx")
    tok = Tokenizer(WordLevel(VOCAB, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    tok.save(str(folder / "tokenizer.json"))


def build_tiny_bundle(
    tiny_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> RunPaths:
    """Run the pipeline through ``serve.export`` (and the ranker chain) on the tiny fixture.
    ``tiny_cfg`` is adjusted in place; the stubs stay patched until ``monkeypatch`` is undone."""
    monkeypatch.setattr(clean, "load_language_id", lambda path: lambda text: ("en", 0.9))
    monkeypatch.setattr(embed, "load_encoder", lambda *args: StubEncoder())
    monkeypatch.setattr(sentences_module, "load_splitter", lambda *args, **kwargs: stub_splitter)
    monkeypatch.setattr(sentiment_module, "load_polarity_model", lambda *args: stub_polarity)
    tiny_cfg["data"]["kcore"] = {"user": 2, "item": 2}
    tiny_cfg["eval"]["ks"] = [5, 10]
    tiny_cfg["eval"]["bootstrap_resamples"] = 20
    small = {"dim": 8, "batch_size": 64, "max_epochs": 4, "eval_every": 2, "patience": 1,
             "inner_val_min_user": 3}  # fmt: skip
    tiny_cfg["models"]["lightgcn"] |= small
    tiny_cfg["models"]["projection"] |= {"epochs": 1, "steps_per_epoch": 5, "batch_size": 32}
    tiny_cfg["ranking"]["min_reviews_main_feed"] = 0
    # D66: a ranker the tiny data can actually grow trees for
    tiny_cfg["ranking"]["lgbm"] |= {"n_estimators": 30, "min_child_samples": 2,
                                    "early_stopping_rounds": 5}  # fmt: skip
    tiny_cfg["ranking"]["eval_folds"] = 2
    tiny_cfg["nlp"]["embedding_dim"] = DIM
    tiny_cfg["nlp"]["chunk_rows"] = 64
    tiny_cfg["aspects"]["tau_default"] = 0.95
    tiny_cfg["nlp"]["onnx_parity"]["texts"] = 20
    # Phase 6: the pipeline runs with the LLM unavailable (no key), whatever the developer's env.
    tiny_cfg["llm"]["api_key_env"] = "LITSENSE_TEST_NO_KEY"
    monkeypatch.delenv("LITSENSE_TEST_NO_KEY", raising=False)

    ran = run_all(tiny_cfg, until="eval.val")
    assert ran[-1] == "eval.val"
    for name in NLP_STAGES:
        assert run_stage(name, tiny_cfg) is True
    # nlp.onnx: the export writes the tiny model; the parity reference is that same model.
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    tiny_onnx(onnx_dir)

    def fake_export(model_name: str, max_len: int, out_dir: Path) -> dict[str, Any]:
        for name in onnx_export.FILES:
            shutil.copyfile(onnx_dir / name, out_dir / name)
        return {"inputs": ["input_ids", "attention_mask"], "opset": 17, "torch": "stub"}

    tiny_encoder = MiniLMEncoder(onnx_dir / "minilm.onnx", onnx_dir / "tokenizer.json", 32, 1)
    with monkeypatch.context() as m:
        m.setattr(onnx_export, "export_onnx", fake_export)
        m.setattr(embed, "load_encoder", lambda *args: tiny_encoder)
        assert run_stage("nlp.onnx", tiny_cfg) is True
    ran = run_all(tiny_cfg, until="ranking.eval_val")
    assert "serve.export" in ran and "models.lightgcn_serve" in ran
    # D66: the ranker chain runs before the bundle that ships it; the ablation comes after
    for name in ("serve.export_val", "ranking.build_training", "ranking.train"):
        assert ran.index(name) < ran.index("serve.export"), ran
    assert ran[-1] == "ranking.eval_val" and "ranking.ablation_val" in ran
    assert "models.lightgcn_sweep" not in ran and "nlp.validate_aspects" not in ran
    assert ran.index("genai.synthesis") < ran.index("serve.export")  # D70: summaries feed export
    return RunPaths.from_config(tiny_cfg, None)


@contextmanager
def tiny_bundle_session(
    root: Path, repo_config: Config, tiny_dataset: Path
) -> Iterator[tuple[Config, RunPaths]]:
    """``build_tiny_bundle`` outside a function-scoped fixture: its own config copy, file server
    and MonkeyPatch, all undone on exit (tests/service builds the bundle once per session)."""
    from tests.ml.data.conftest import point_at_tiny, start_file_server

    patch = pytest.MonkeyPatch()
    server_state, server = start_file_server(root / "served", patch)
    try:
        cfg = point_at_tiny(rooted_config(repo_config, root), server_state, tiny_dataset)
        yield cfg, build_tiny_bundle(cfg, patch, root)
    finally:
        server.shutdown()
        server.server_close()
        patch.undo()
