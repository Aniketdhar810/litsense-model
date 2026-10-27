"""Stage data.clean (CLAUDE.md s7.6): clean review text and mark which reviews NLP may use.

Every review keeps its row (the rating is still signal); ``text_ok`` only says whether Phase 3 may
use the text, and ``text_drop_reason`` says why not. Rules, in order:

1. text: tags stripped, entities unescaped, NFKC, whitespace collapsed; texts longer than
   ``data.max_review_chars`` are cut at the last sentence end before the limit;
2. ``too_short``: fewer than ``data.min_review_chars`` characters;
3. ``non_english``: fastText ``lid.176`` on the first 500 characters says another language with
   probability >= ``data.lid_min_confidence`` (short English texts are often misread, D35);
4. ``shipping``: a ``data.shipping_patterns`` match and no aspect-lexicon term.

The table is processed in record batches, so the review text is never all in memory.
"""

import html
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ml.common.stage import StageContext, stage
from ml.data.meta import PARQUET_COMPRESSION
from ml.data.reviews import REVIEW_SCHEMA
from ml.nlp.aspect_seeds import has_aspect_term

CLEAN_SCHEMA = pa.schema(
    [
        *REVIEW_SCHEMA,
        pa.field("text_ok", pa.bool_()),
        pa.field("text_drop_reason", pa.string()),  # too_short | non_english | shipping | None
        pa.field("truncated", pa.bool_()),
    ]
)
LID_CHARS = 500
BATCH_ROWS = 100_000
_TAG_RE = re.compile(r"<[^>]+>")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
LanguageId = Callable[[str], tuple[str, float]]  # text -> (ISO code, probability)


def clean_text(text: str | None) -> str:
    """Tags stripped, HTML entities unescaped, NFKC, whitespace collapsed."""
    text = html.unescape(_TAG_RE.sub(" ", text or ""))
    return " ".join(unicodedata.normalize("NFKC", text).split())


def truncate(text: str, limit: int) -> tuple[str, bool]:
    """``text`` cut at the last sentence end within ``limit`` characters (else the last space)."""
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    ends = [m.end() for m in _SENTENCE_END_RE.finditer(head)]
    cut = ends[-1] if ends else head.rfind(" ")
    return head[: cut if cut > 0 else limit].rstrip(), True


def load_language_id(path: Path) -> LanguageId:
    """fastText language ID from ``lid.176.ftz``. One text per call: batch ``predict`` in
    fasttext-predict 0.9.2.4 mis-unpacks its results (D35)."""
    import fasttext

    model = fasttext.load_model(str(path))

    def predict(text: str) -> tuple[str, float]:
        labels, probs = model.predict(text.replace("\n", " "), k=1)
        return labels[0].removeprefix("__label__"), float(probs[0])

    return predict


def compile_patterns(patterns: Sequence[str]) -> re.Pattern[str]:
    """One case-insensitive pattern for ``data.shipping_patterns``."""
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


def drop_reason(
    text: str, params: Mapping[str, Any], language_id: LanguageId, shipping: re.Pattern[str]
) -> str | None:
    """Why NLP must not use this cleaned text, or None."""
    if len(text) < int(params["min_review_chars"]):
        return "too_short"
    lang, prob = language_id(text[:LID_CHARS])
    if lang != "en" and prob >= float(params["lid_min_confidence"]):
        return "non_english"
    if shipping.search(text) and not has_aspect_term(text):
        return "shipping"
    return None


def clean_batch(
    frame: pd.DataFrame,
    params: Mapping[str, Any],
    language_id: LanguageId,
    shipping: re.Pattern[str],
) -> pd.DataFrame:
    """Clean one batch of ``reviews_raw`` rows and add ``text_ok``/``text_drop_reason``."""
    limit = int(params["max_review_chars"])
    cleaned = [truncate(clean_text(t), limit) for t in frame["text"]]
    frame = frame.assign(text=[t for t, _ in cleaned], truncated=[c for _, c in cleaned])
    reasons = [drop_reason(t, params, language_id, shipping) for t in frame["text"]]
    return frame.assign(text_drop_reason=reasons, text_ok=[r is None for r in reasons])


@stage(
    "data.clean",
    inputs=["data:reviews_raw.parquet", "data:lid.176.ftz"],
    outputs=["data:reviews_clean.parquet"],
    config_keys=[
        "data.min_review_chars",
        "data.max_review_chars",
        "data.lid_min_confidence",
        "data.shipping_patterns",
    ],
)
def build_clean(ctx: StageContext) -> None:
    """Clean every review and mark the texts NLP may use."""
    params = {
        k: ctx.param(f"data.{k}")
        for k in ("min_review_chars", "max_review_chars", "lid_min_confidence")
    }
    shipping = compile_patterns(ctx.param("data.shipping_patterns"))
    language_id = load_language_id(ctx.path("data:lid.176.ftz"))
    reasons: Counter[str] = Counter()
    rows = truncated = 0
    source = pq.ParquetFile(ctx.path("data:reviews_raw.parquet"))
    with pq.ParquetWriter(
        ctx.out("data:reviews_clean.parquet"), CLEAN_SCHEMA, compression=PARQUET_COMPRESSION
    ) as out:
        for batch in source.iter_batches(batch_size=BATCH_ROWS):
            frame = clean_batch(batch.to_pandas(), params, language_id, shipping)
            rows += len(frame)
            truncated += int(frame["truncated"].sum())
            reasons.update(frame["text_drop_reason"].dropna())
            frame = frame.reindex(columns=CLEAN_SCHEMA.names)
            out.write_table(pa.Table.from_pandas(frame, schema=CLEAN_SCHEMA, preserve_index=False))
    ctx.count("reviews", rows)
    ctx.count("text_ok", rows - sum(reasons.values()))
    ctx.count("truncated", truncated)
    for reason in ("too_short", "non_english", "shipping"):
        ctx.count(f"dropped_{reason}", reasons[reason])
    ctx.notes["dropped_share"] = {r: round(n / max(rows, 1), 4) for r, n in sorted(reasons.items())}
