"""Offline MiniLM encoder (CLAUDE.md s10.1) backed by sentence-transformers.

``litsense_core/encoder.py`` (ONNX, Week 6) exposes the same ``encode()`` so callers can switch
implementations. Imported lazily: torch is only needed when a stage really embeds text.
"""

from typing import Protocol

import numpy as np


class TextEncoder(Protocol):
    """What content stages need from an encoder."""

    def encode(self, texts: list[str]) -> np.ndarray: ...  # (n, d) float32, L2-normalized

    def count_tokens(self, text: str) -> int: ...  # word pieces, without special tokens

    def truncate(self, text: str, max_tokens: int) -> str: ...


class SentenceEncoder:
    """sentence-transformers model on GPU when available (fp16 there), normalized embeddings."""

    def __init__(self, model_name: str, batch_size: int, max_seq_length: int) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_seq_length
        if device == "cuda":
            self.model.half()
        self.batch_size = batch_size
        self.tokenizer = self.model.tokenizer

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def truncate(self, text: str, max_tokens: int) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"][: max(max_tokens, 0)]
        return self.tokenizer.decode(ids)


def load_encoder(model_name: str, batch_size: int, max_seq_length: int) -> TextEncoder:
    """The offline encoder (tests replace this with a stub)."""
    return SentenceEncoder(model_name, batch_size, max_seq_length)
