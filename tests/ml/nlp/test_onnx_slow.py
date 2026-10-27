"""Real MiniLM export and the s10.3 parity thresholds (needs torch, transformers, the model)."""

from pathlib import Path

import numpy as np
import pytest

from litsense_core.encoder import MiniLMEncoder
from ml.nlp import embed, onnx_export
from ml.nlp.aspect_seeds import PROTOTYPES


@pytest.mark.slow
def test_real_export_meets_parity_thresholds(tmp_path: Path, repo_config: dict) -> None:
    content, thresholds = repo_config["content"], repo_config["nlp"]["onnx_parity"]
    max_len = int(content["max_seq_length"])
    info = onnx_export.export_onnx(content["model_name"], max_len, tmp_path)
    assert "input_ids" in info["inputs"]
    texts = [t for name in PROTOTYPES for t in PROTOTYPES[name]][:60]
    reference = embed.load_encoder(content["model_name"], 32, max_len).encode(texts)
    fp32 = MiniLMEncoder(tmp_path / "minilm.onnx", tmp_path / "tokenizer.json", max_len, 1)
    int8 = MiniLMEncoder(tmp_path / "minilm.int8.onnx", tmp_path / "tokenizer.json", max_len, 1)
    report = onnx_export.parity_report(
        reference, fp32.encode(texts), int8.encode(texts), thresholds
    )
    assert report["fp32_ok"], report
    assert report["int8_ok"], report  # if this fails, ship fp32: serving.onnx_quantized false
    assert fp32.encode(["x"]).shape == (1, int(repo_config["nlp"]["embedding_dim"]))
    assert np.isfinite(onnx_export.latency_ms(int8, runs=3))
