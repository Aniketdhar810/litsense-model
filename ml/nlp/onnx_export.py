"""Stage nlp.onnx (CLAUDE.md s10.3): export MiniLM to ONNX for the serving container.

Exports the transformer (``AutoModel``, output ``last_hidden_state``) with dynamic batch and
sequence axes, saves the fast tokenizer as ``tokenizer.json``, quantizes the weights to int8
(``onnxruntime.quantization.quantize_dynamic``) and checks parity against sentence-transformers
on ``nlp.onnx_parity.texts`` varied sentences: fp32 cosine >= ``fp32_min_cos`` for every text,
int8 mean >= ``int8_mean_cos`` and min >= ``int8_min_cos``. The fp32 check failing is an error;
the int8 check failing is reported (``int8_ok: false``) with the instruction to set
``serving.onnx_quantized: false``, and both files are still written. One short query is timed
with one thread.
"""

import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from litsense_core.encoder import MiniLMEncoder
from ml.common.stage import StageContext, stage
from ml.nlp import embed
from ml.nlp.aspect_seeds import PROTOTYPES

FOLDER = "serve/onnx"
FILES = ("minilm.onnx", "minilm.int8.onnx", "tokenizer.json")
PARITY = "results:nlp/onnx_parity.json"
OPSET = 17
LATENCY_QUERY = "cozy mysteries with a witty detective"
LATENCY_RUNS = 20


def export_onnx(model_name: str, max_seq_length: int, out_dir: Path) -> dict[str, Any]:
    """Write ``minilm.onnx``, ``minilm.int8.onnx`` and ``tokenizer.json`` into ``out_dir``."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval()
    sample = tokenizer(
        ["a sample sentence for the export", "another one"],
        padding=True,
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    input_names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    dynamic = {name: {0: "batch", 1: "sequence"} for name in input_names}
    dynamic["last_hidden_state"] = {0: "batch", 1: "sequence"}
    fp32 = out_dir / "minilm.onnx"

    class HiddenStateOnly(torch.nn.Module):
        """ONNX wants one tensor output: ``last_hidden_state`` only."""

        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: Any) -> Any:
            return self.inner(*args).last_hidden_state

    wrapper = HiddenStateOnly(model).eval()
    for parameter in wrapper.parameters():  # weights become initializers, not traced constants
        parameter.requires_grad_(False)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(sample[n] for n in input_names),
            str(fp32),
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic,
            opset_version=OPSET,
            dynamo=False,
        )
    # per-channel weight scales: per-tensor scales lost too much cosine on MiniLM (D61)
    quantize_dynamic(
        str(fp32), str(out_dir / "minilm.int8.onnx"), weight_type=QuantType.QInt8, per_channel=True
    )
    tokenizer.backend_tokenizer.save(str(out_dir / "tokenizer.json"))
    return {"inputs": input_names, "opset": OPSET, "torch": torch.__version__}


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine between two (n, d) matrices."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return num / np.clip(den, 1e-12, None)


def parity_report(
    reference: np.ndarray,
    fp32: np.ndarray,
    int8: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Cosine statistics and pass/fail against ``nlp.onnx_parity`` thresholds."""
    c32, c8 = cosines(reference, fp32), cosines(reference, int8)
    return {
        "texts": int(len(reference)),
        "fp32": {"min_cos": round(float(c32.min()), 6), "mean_cos": round(float(c32.mean()), 6)},
        "int8": {"min_cos": round(float(c8.min()), 6), "mean_cos": round(float(c8.mean()), 6)},
        "fp32_ok": bool(c32.min() >= float(thresholds["fp32_min_cos"])),
        "int8_ok": bool(
            c8.mean() >= float(thresholds["int8_mean_cos"])
            and c8.min() >= float(thresholds["int8_min_cos"])
        ),
        "thresholds": {k: thresholds[k] for k in ("fp32_min_cos", "int8_mean_cos", "int8_min_cos")},
    }


def latency_ms(
    encoder: MiniLMEncoder, query: str = LATENCY_QUERY, runs: int = LATENCY_RUNS
) -> float:
    """Median wall time of one short query (after one warm-up call)."""
    encoder.encode([query])
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        encoder.encode([query])
        times.append((time.perf_counter() - start) * 1000)
    return round(float(np.median(times)), 2)


def parity_texts(sentences: pd.DataFrame, n: int, rng: np.random.Generator) -> list[str]:
    """The prototype sentences plus a seeded sample of real review sentences, ``n`` in total."""
    texts = [t for name in PROTOTYPES for t in PROTOTYPES[name]]
    pool = sentences["text"].tolist()
    need = max(n - len(texts), 0)
    if pool and need:
        picks = rng.choice(len(pool), size=min(need, len(pool)), replace=False)
        texts += [pool[k] for k in sorted(picks.tolist())]
    return texts[:n] if n < len(texts) else texts


@stage(
    "nlp.onnx",
    inputs=["data:sentences.parquet"],
    outputs=[*(f"artifacts:{FOLDER}/{name}" for name in FILES), PARITY],
    config_keys=["content", "serving.onnx_threads", "nlp.embedding_dim", "nlp.onnx_parity"],
)
def build_onnx(ctx: StageContext) -> None:
    """Export, quantize, save the tokenizer, and check parity and latency."""
    content = ctx.param("content")
    thresholds: Mapping[str, Any] = ctx.param("nlp.onnx_parity")
    dim = int(ctx.param("nlp.embedding_dim"))
    max_len = int(content["max_seq_length"])
    work = ctx.work_dir / "onnx"
    work.mkdir(parents=True, exist_ok=True)
    info = export_onnx(str(content["model_name"]), max_len, work)
    for name in FILES:
        shutil.copyfile(work / name, ctx.out(f"artifacts:{FOLDER}/{name}"))

    sentences = pd.read_parquet(ctx.path("data:sentences.parquet"), columns=["text"])
    texts = parity_texts(sentences, int(thresholds["texts"]), ctx.rng)
    reference = embed.load_encoder(str(content["model_name"]), 64, max_len).encode(texts)
    threads = int(ctx.param("serving.onnx_threads"))
    fp32 = MiniLMEncoder(work / "minilm.onnx", work / "tokenizer.json", max_len, threads)
    int8 = MiniLMEncoder(work / "minilm.int8.onnx", work / "tokenizer.json", max_len, threads)
    v32, v8 = fp32.encode(texts), int8.encode(texts)
    if v32.shape != (len(texts), dim) or reference.shape != (len(texts), dim):
        raise ValueError(
            f"embedding shapes {v32.shape} / {reference.shape} != ({len(texts)}, {dim})"
        )
    report = parity_report(reference, v32, v8, thresholds)
    report |= {
        "model_name": content["model_name"],
        "max_seq_length": max_len,
        "export": info,
        "latency_ms_one_thread": {"fp32": latency_ms(fp32), "int8": latency_ms(int8)},
        "max_latency_ms": float(thresholds["max_latency_ms"]),
        "bytes": {name: (work / name).stat().st_size for name in FILES},
    }
    ctx.out(PARITY).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["fp32_ok"]:
        raise ValueError(f"fp32 ONNX parity failed: {report['fp32']}")
    if not report["int8_ok"]:
        ctx.log.warning(
            "int8 parity failed (%s): ship fp32 and set serving.onnx_quantized: false",
            report["int8"],
        )
    slow = max(report["latency_ms_one_thread"].values())
    if slow > float(thresholds["max_latency_ms"]):
        ctx.log.warning("one-thread latency %.1f ms is above max_latency_ms", slow)
    ctx.count("parity_texts", len(texts))
    ctx.notes["parity"] = {k: report[k] for k in ("fp32", "int8", "fp32_ok", "int8_ok")}
    ctx.notes["latency_ms_one_thread"] = report["latency_ms_one_thread"]
