"""Runtime MiniLM encoder over ONNX Runtime (CLAUDE.md s10.4): mean pooling + L2 normalization,
the same ``encode()`` as ``ml/nlp/embed.py`` so callers can swap implementations. No torch."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class MiniLMEncoder:
    """Tokenizer + ONNX session; ``encode`` returns (n, d) float32 unit vectors."""

    def __init__(
        self,
        onnx_path: str | Path,
        tokenizer_path: str | Path,
        max_len: int = 256,
        threads: int = 1,
    ) -> None:
        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        self.max_len = max_len
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        # One short query at a time: the arena and memory patterns only add resident memory (D64)
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = False
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}
        self.output_name = self.sess.get_outputs()[0].name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalized embeddings (empty input gives shape (0, d))."""
        if not texts:
            dim = self.sess.get_outputs()[0].shape[-1]
            return np.zeros((0, int(dim) if isinstance(dim, int) else 0), dtype=np.float32)
        enc = self.tok.encode_batch(list(texts))
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run([self.output_name], feeds)[0]  # (B, T, d)
        m = mask[..., None].astype(np.float32)
        v = (hidden.astype(np.float32) * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        norms = np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
        return (v / norms).astype(np.float32)

    def count_tokens(self, text: str) -> int:
        """Word pieces without special tokens (same contract as the offline encoder)."""
        return len(self.tok.encode(text, add_special_tokens=False).ids)

    def truncate(self, text: str, max_tokens: int) -> str:
        enc = self.tok.encode(text, add_special_tokens=False)
        return self.tok.decode(enc.ids[: max(max_tokens, 0)])
