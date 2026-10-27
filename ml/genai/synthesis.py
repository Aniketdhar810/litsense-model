"""Offline review synthesis (CLAUDE.md s14.8), stage ``genai.synthesis``.

For the ``genai.synthesis.top_n`` books by review count at the serve cutoff: praised aspects
(``pos_rate`` and ``mention_rate`` above the thresholds), criticised aspects (``neg_rate``), and
the safe snippets, into the ``review_synthesis`` prompt on ``llm.model_quality``. Each summary is
validated (titles: only the book's own; numbers: only those in the prompt; spoilers: the regex
scorer; length) and a stats template replaces any rejected or failed one. Output:
``artifacts/<v>/serve/genai/review_summaries.jsonl`` (one row per processed book) plus
``meta.json``; ``serve.export`` copies the summaries into ``catalog_text.review_summary``.

The stage is resumable (rows accumulate in the work dir, so a run that hits the daily budget
continues the next day) and respects ``rpm_limit`` through the blocking limiter. With
``llm.enabled: false``, no API key, or ``model_quality: SET-ME`` it writes an empty file and
says why in ``meta.json``: the bundle then keeps ``null`` summaries and nothing else changes.
Summaries are never used as a model feature (s1.5).
"""

import gzip
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from litsense_core.llm import LLMClient, LLMConfig
from litsense_core.schemas import ASPECT_LABELS
from litsense_core.spoiler import SpoilerScorer
from litsense_core.validators import TitleMatcher, check_text
from ml.common.io import load_id_map
from ml.common.stage import StageContext, stage

FOLDER = "serve/genai"
OUTPUT = f"artifacts:{FOLDER}/review_summaries.jsonl"
META = f"artifacts:{FOLDER}/meta.json"
ASPECT_DIR = "artifacts:serve/aspects"
SNIPPETS = "artifacts:serve/snippets/evidence_snippets.json.gz"
RATE_FILES = ("item_mention_rate.npy", "item_pos_rate.npy", "item_neg_rate.npy")
WORK_FILE = "summaries.jsonl"
KEEP = (None, "rejected")  # rows worth keeping across runs (an LLM answer, or its rejection)
STOP = ("budget", "quota", "unavailable", "no_api_key", "not_configured", "disabled")
MAX_TRANSIENT = 5  # consecutive timeouts / provider errors before the run gives up
MINUTE_WAIT_S = 65.0  # a per-minute 429: wait out the window and retry the same book
MAX_MINUTE_WAITS = 3


class Summary(BaseModel):
    summary: str


def select_books(n_reviews: np.ndarray, top_n: int) -> list[int]:
    """Item ids of the ``top_n`` books by review count (ties by id), reviews > 0 only."""
    order = np.lexsort((np.arange(len(n_reviews)), -n_reviews))
    return [int(i) for i in order[:top_n] if n_reviews[i] > 0]


def aspect_stats(
    i: int,
    names: list[str],
    mention: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    params: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Praised and criticised aspects of item ``i`` (s14.8 thresholds), strongest first."""
    praised, criticised = [], []
    p, c = params["praised"], params["criticised"]
    for a, name in enumerate(names):
        m = float(mention[i, a])
        if m >= float(p["mention_rate"]) and float(pos[i, a]) >= float(p["pos_rate"]):
            praised.append({"aspect": ASPECT_LABELS.get(name, name),
                            "positive_pct": int(round(100 * float(pos[i, a]))),
                            "mention_pct": int(round(100 * m))})  # fmt: skip
        if m >= float(c["mention_rate"]) and float(neg[i, a]) >= float(c["neg_rate"]):
            criticised.append({"aspect": ASPECT_LABELS.get(name, name),
                               "negative_pct": int(round(100 * float(neg[i, a]))),
                               "mention_pct": int(round(100 * m))})  # fmt: skip
    praised.sort(key=lambda d: -d["positive_pct"])
    criticised.sort(key=lambda d: -d["negative_pct"])
    return praised, criticised


def pick_snippets(entry: Mapping[str, Any], n_pos: int, n_neg: int) -> tuple[list[str], list[str]]:
    """Up to ``n_pos`` positive and ``n_neg`` negative safe snippet texts across aspects, by
    helpful votes."""
    pos, neg = [], []
    for slots in entry.values():
        pos += [(int(s.get("helpful_votes", 0)), s["text"]) for s in slots.get("pos", [])]
        neg += [(int(s.get("helpful_votes", 0)), s["text"]) for s in slots.get("neg", [])]
    pos.sort(key=lambda t: -t[0])
    neg.sort(key=lambda t: -t[0])
    return [t for _, t in pos[:n_pos]], [t for _, t in neg[:n_neg]]


def template_summary(n_reviews: int, praised: list[dict[str, Any]],
                     criticised: list[dict[str, Any]]) -> str:  # fmt: skip
    """The stats-based fallback of s14.8."""
    parts = []
    if praised:
        p = praised[0]
        parts.append(f"Readers most often praise its {p['aspect']} ({p['positive_pct']}% positive)")
    if criticised:
        c = criticised[0]
        parts.append(f"some criticise its {c['aspect']} ({c['negative_pct']}% negative)")
    if not parts:
        return f"Based on {n_reviews} reviews, no single aspect stands out."
    return "; ".join(parts) + "."


def synthesize(
    client: LLMClient,
    variables: Mapping[str, Any],
    matcher: TitleMatcher,
    spoiler: SpoilerScorer | None,
    max_words: int,
) -> dict[str, Any]:
    """One book: the validated LLM summary or the template; the row written to the JSONL."""
    n_reviews = int(variables["n_reviews"])
    praised, criticised = variables["praised"], variables["criticised"]
    fallback = template_summary(n_reviews, praised, criticised)
    result = client.json("review_synthesis", variables, Summary, "synthesis")
    if not result.ok or result.data is None:
        return {"summary": fallback, "method": "template", "rejections": [],
                "fallback_reason": result.fallback_reason, "cached": result.cached}  # fmt: skip
    text = " ".join(str(result.data.summary).split())  # type: ignore[attr-defined]
    row = validate_row(text, variables, matcher, spoiler, max_words)
    row["cached"] = result.cached
    return row


def validate_row(
    raw: str,
    variables: Mapping[str, Any],
    matcher: TitleMatcher,
    spoiler: SpoilerScorer | None,
    max_words: int,
) -> dict[str, Any]:
    """The row for an LLM answer ``raw``: accepted as the summary, or the template with the
    rejection codes. ``raw`` is kept in the row so a later run can re-validate without a call."""
    fallback = template_summary(int(variables["n_reviews"]), variables["praised"],
                                variables["criticised"])  # fmt: skip
    problems = check_text(raw, [str(variables["title"])], dict(variables), max_words, matcher,
                          spoiler)  # fmt: skip
    if problems:
        return {"summary": fallback, "method": "template", "rejections": problems,
                "fallback_reason": "rejected", "raw": raw}  # fmt: skip
    return {"summary": raw, "method": "llm", "rejections": [], "fallback_reason": None, "raw": raw}


def read_summaries(path: Path) -> dict[int, dict[str, Any]]:
    """``{item: row}`` from a JSONL file (missing file -> empty)."""
    out: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                out[int(row["i"])] = row
    return out


@stage(
    "genai.synthesis",
    inputs=[
        "data:books.parquet",
        "data:interactions.parquet",
        "data:id_map.json",
        *[f"{ASPECT_DIR}/{name}" for name in RATE_FILES],
        SNIPPETS,
    ],
    outputs=[OUTPUT, META],
    config_keys=["llm", "genai.synthesis", "aspects.names", "spoiler.regex_patterns",
                 "spoiler.regex_boost", "spoiler.late_position_boost", "spoiler.late_fraction"],
)  # fmt: skip
def build_summaries(ctx: StageContext) -> None:
    """Review summaries for the top books, or an empty file when the LLM is off."""
    params = ctx.param("genai.synthesis")
    llm_raw = dict(ctx.param("llm"))
    llm_raw["timeout_s"] = max(float(llm_raw.get("timeout_s", 10)), float(params["timeout_s"]))
    llm_raw["rpm_limit"] = min(int(llm_raw.get("rpm_limit", 12)), int(params["rpm_limit"]))
    llm_cfg = LLMConfig.from_mapping(llm_raw)
    ids = load_id_map(ctx.path("data:id_map.json"))
    books = pd.read_parquet(ctx.path("data:books.parquet")).sort_values("i").reset_index(drop=True)
    inter = pd.read_parquet(ctx.path("data:interactions.parquet"), columns=["i", "has_text"])
    n_reviews = np.bincount(inter.loc[inter["has_text"], "i"], minlength=ids.n_items)
    names = list(ctx.param("aspects.names"))
    rates = {name: np.load(ctx.path(f"{ASPECT_DIR}/{name}")) for name in RATE_FILES}
    chosen = select_books(n_reviews, int(params["top_n"]))
    out_path = ctx.out(OUTPUT)
    meta: dict[str, Any] = {"books_selected": len(chosen), "model": llm_cfg.model_quality}

    client = LLMClient(llm_cfg, model=llm_cfg.model_quality)
    reason = client.unavailable_reason()
    if reason is not None:
        ctx.log.warning("LLM unavailable (%s): writing no summaries", reason)
        out_path.write_text("", encoding="utf-8")
        meta.update({"skipped": reason, "written": 0})
        ctx.out(META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        ctx.count("summaries", 0)
        return
    client.blocking_limiter = True  # offline: wait for the rpm window instead of failing

    with gzip.open(ctx.path(SNIPPETS), "rt", encoding="utf-8") as f:
        snippets = json.load(f)
    matcher = TitleMatcher([str(t) for t in books["title"].tolist()])
    spoiler = SpoilerScorer.regex_only(
        [str(p) for p in ctx.param("spoiler.regex_patterns")],
        float(ctx.param("spoiler.regex_boost")),
        float(ctx.param("spoiler.late_position_boost")),
        float(ctx.param("spoiler.late_fraction")),
    )
    work = ctx.work_dir / WORK_FILE
    n_pos, n_neg = int(params["snippets"]["positive"]), int(params["snippets"]["negative"])

    def variables_for(i: int) -> dict[str, Any]:
        praised, criticised = aspect_stats(
            i, names, rates["item_mention_rate.npy"], rates["item_pos_rate.npy"],
            rates["item_neg_rate.npy"], params,
        )  # fmt: skip
        pos_snips, neg_snips = pick_snippets(snippets.get(str(i), {}), n_pos, n_neg)
        author = books.at[i, "author"]
        return {
            "title": str(books.at[i, "title"]),
            "author": author if isinstance(author, str) and author.strip() else "unknown",
            "n_reviews": int(n_reviews[i]),
            "praised": praised,
            "criticised": criticised,
            "pos_snippets": pos_snips,
            "neg_snippets": neg_snips,
        }

    # Rows kept from an earlier run: accepted LLM text as is; rejected text re-validated with the
    # current validators (no call); rows without the raw answer or with a transient failure are
    # dropped so this run retries those books.
    done: dict[int, dict[str, Any]] = {}
    revalidated = 0
    for i, r in read_summaries(work).items():
        if r.get("fallback_reason") is None and r.get("method") == "llm":
            done[i] = r
        elif r.get("fallback_reason") == "rejected" and r.get("raw") and i in set(chosen):
            row = validate_row(r["raw"], variables_for(i), matcher, spoiler,
                               int(params["max_words"]))  # fmt: skip
            row["i"], row["cached"] = i, True
            revalidated += row["method"] == "llm"
            done[i] = row
    with work.open("w", encoding="utf-8") as f:
        for i in sorted(done):
            f.write(json.dumps(done[i], ensure_ascii=False) + "\n")
    counts = {"llm": 0, "template": 0, "resumed": len(done), "revalidated_ok": revalidated,
              "stopped": None}  # fmt: skip
    transient = 0
    with work.open("a", encoding="utf-8") as f:
        for i in chosen:
            if i in done:
                continue
            variables = variables_for(i)
            row = synthesize(client, variables, matcher, spoiler, int(params["max_words"]))
            waits = 0
            while row["fallback_reason"] == "rate_limited" and waits < MAX_MINUTE_WAITS:
                waits += 1
                ctx.log.warning("item %d: per-minute 429, waiting %.0f s (%d)", i, MINUTE_WAIT_S,
                                waits)  # fmt: skip
                time.sleep(MINUTE_WAIT_S)
                row = synthesize(client, variables, matcher, spoiler, int(params["max_words"]))
            reason = row["fallback_reason"]
            if reason == "rate_limited":
                reason = "unavailable"  # three waits did not clear it: resume later
                row["fallback_reason"] = reason
            if reason in STOP:
                counts["stopped"] = reason  # quota or config: keep the rows so far, resume later
                ctx.log.warning("stopping at item %d: %s", i, reason)
                break
            if reason not in KEEP:  # timeout / provider error: do not persist, retry next run
                transient += 1
                ctx.log.warning("item %d: %s (transient failure %d)", i, reason, transient)
                if transient >= MAX_TRANSIENT:
                    counts["stopped"] = reason
                    break
                continue
            transient = 0
            row["i"] = i
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()  # every row reaches the work file at once (Drive, resume after a crash)
            done[i] = row
            counts[row["method"]] += 1
            if len(done) % 50 == 0:
                ctx.log.info("summaries so far: %d of %d (llm %d, template %d)", len(done),
                             len(chosen), counts["llm"], counts["template"])  # fmt: skip
    # Drive's FUSE mount can forget an empty staging folder created an hour earlier (D72).
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for i in sorted(done):
            f.write(json.dumps(done[i], ensure_ascii=False) + "\n")
    rejections: dict[str, int] = {}
    for row in done.values():
        for code in row.get("rejections", []):
            key = code.split(":")[0]
            rejections[key] = rejections.get(key, 0) + 1
    meta.update({"written": len(done), "counts": counts, "rejections": rejections,
                 "remaining": len([i for i in chosen if i not in done])})  # fmt: skip
    ctx.out(META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    ctx.count("summaries", len(done))
    ctx.count("summaries_llm", counts["llm"])
    if counts["stopped"]:
        # The run did not finish: the manifest would mark the stage complete with a partial
        # file, so fail loudly and let the next run resume from the work dir.
        raise RuntimeError(
            f"genai.synthesis stopped ({counts['stopped']}) after {len(done)} of {len(chosen)}"
            " books; rerun to resume"
        )
