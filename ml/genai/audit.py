"""Explanation audit and automated rejection rates (CLAUDE.md s14.9).

* ``export``: sample ``n`` explanations for evaluation readers (the bundle's corpus demo
  accounts, each with their own feed), stratified by method (half LLM, half template when the
  LLM is available; all template otherwise), into ``docs/annotation/explain_audit.csv`` with the
  rater columns ``grounded`` (Y/N), ``hallucinated_title`` (Y/N), ``spoiler`` (Y/N) and
  ``useful`` (1-5) left blank. Three raters each fill a copy.
* ``score``: rates per method and Fleiss' kappa per question across the filled copies ->
  ``results/<v>/genai/explain_audit.json``. Targets: >= 90% grounded, 0 hallucinated titles,
  0 spoilers.
* ``rejections``: generate ``n`` LLM explanations and count validator rejections by type (title,
  number, spoiler, length, format, quote), the automated hallucination-rate proxy.

    python -m ml.genai.audit export --bundle artifacts/serve_bundle --n 100
    python -m ml.genai.audit score --files r1.csv r2.csv r3.csv \
        --out results/v0/genai/explain_audit.json
    python -m ml.genai.audit rejections --bundle artifacts/serve_bundle --n 1000 --out ...
"""

import argparse
import contextlib
import csv
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import litsense_core as core
from litsense_core.explain import llm_reason
from litsense_core.schemas import UserState
from litsense_core.store import ArtifactStore

COLUMNS = (
    "id", "method", "reader", "book_id", "book_title", "explanation", "evidence_text",
    "evidence_json", "grounded", "hallucinated_title", "spoiler", "useful",
)  # fmt: skip


def evidence_text(evidence: Mapping[str, Any]) -> str:
    """The evidence in plain words for a rater: what the app knew about the reader and the book
    when the explanation was written (``docs/annotation_guides/explain_audit.md``)."""
    book = evidence.get("book", {})
    user = evidence.get("user_evidence", {})
    item = evidence.get("item_evidence", {})
    lines = [
        f"BOOK: {book.get('title')} by {book.get('author') or 'unknown'}"
        f" ({', '.join(book.get('genres', [])) or 'no genre'}, {book.get('year') or 'year ?'})"
    ]
    liked = [f"{x['title']} ({x['stars']:g} stars)" for x in user.get("similar_liked_items", [])]
    lines.append("READER LIKED: " + (", ".join(liked) if liked else "nothing on record"))
    if user.get("favorite_author"):
        lines.append(f"READER'S FAVOURITE AUTHOR: {user['favorite_author']}")
    pos = [a["label"] for a in user.get("top_positive_aspects", [])]
    if pos:
        lines.append("READER VALUES: " + ", ".join(pos))
    if user.get("preferred_genres"):
        lines.append("READER'S GENRES: " + ", ".join(user["preferred_genres"]))
    rates = item.get("pos_rates", {})
    praised = sorted(rates.items(), key=lambda kv: -kv[1])[:3]
    if praised:
        lines.append("REVIEWERS PRAISE: " + ", ".join(f"{k} ({round(100 * v)}% positive)"
                                                      for k, v in praised))  # fmt: skip
    snippet = item.get("snippet") or {}
    if snippet.get("text"):
        lines.append(f'REVIEW SNIPPET: "{snippet["text"]}"')
    return "\n".join(lines)


QUESTIONS = ("grounded", "hallucinated_title", "spoiler")
REJECTION_TYPES = ("title", "number", "spoiler", "length", "format", "quote", "empty", "privacy")


def demo_states(store: ArtifactStore, limit: int | None = None) -> list[UserState]:
    """One ``UserState`` per corpus demo reader (their history lives in the bundle)."""
    ids = list(store.corpus_users)[:limit]
    return [UserState(corpus_user_id=u) for u in ids]


def collect_explanations(
    store: ArtifactStore, states: Sequence[UserState], n: int, k: int = 5, seed: int = 42
) -> list[dict[str, Any]]:
    """Up to ``n`` (reader, book) explanations: LLM where available, template otherwise, in a
    seeded order over readers; ``method`` says which one each row got."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(states))
    rows: list[dict[str, Any]] = []
    for pos in order:
        state = states[int(pos)]
        feed = core.recommend(state, k=k)
        for item in feed.items:
            if len(rows) >= n:
                return rows
            out = llm_reason(item.evidence, store, state.overrides.disabled_ai)
            rows.append(
                {
                    "reader": state.corpus_user_id or f"state{pos}",
                    "book_id": item.book_id,
                    "book_title": store.catalog.title[item.book_id],
                    "method": "llm" if out.text else "template",
                    "explanation": out.text or item.reason,
                    "rejections": out.rejections,
                    "fallback_reason": out.fallback_reason,
                    "evidence_json": item.evidence.model_dump_json(),
                }
            )
    return rows


def stratify(rows: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Half LLM, half template when both exist (seeded), else whatever there is, up to ``n``."""
    rng = np.random.default_rng(seed)
    by_method = {m: [r for r in rows if r["method"] == m] for m in ("llm", "template")}
    for lst in by_method.values():
        rng.shuffle(lst)  # type: ignore[arg-type]
    if by_method["llm"] and by_method["template"]:
        half = n // 2
        chosen = by_method["llm"][:half] + by_method["template"][: n - half]
        if len(chosen) < n:  # one side was short: top up from the other
            extra = [r for r in rows if r not in chosen]
            chosen += extra[: n - len(chosen)]
    else:
        chosen = (by_method["llm"] or by_method["template"])[:n]
    rng.shuffle(chosen)  # type: ignore[arg-type]
    return chosen


def write_audit_csv(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for k, row in enumerate(rows, 1):
            writer.writerow(
                {
                    "id": f"a{k:03d}",
                    "method": row["method"],
                    "reader": row["reader"],
                    "book_id": row["book_id"],
                    "book_title": row["book_title"],
                    "explanation": row["explanation"],
                    "evidence_text": evidence_text(json.loads(row["evidence_json"])),
                    "evidence_json": row["evidence_json"],
                    "grounded": "",
                    "hallucinated_title": "",
                    "spoiler": "",
                    "useful": "",
                }
            )
            count += 1
    return count


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa for an (subjects, categories) count matrix with a constant number of raters
    per subject; NaN when chance agreement is total."""
    counts = np.asarray(ratings, dtype=float)
    if counts.size == 0:
        return float("nan")
    n = counts.sum(axis=1)
    if not np.all(n == n[0]) or n[0] < 2:
        raise ValueError("every subject needs the same number (>= 2) of ratings")
    raters = float(n[0])
    p_i = ((counts**2).sum(axis=1) - raters) / (raters * (raters - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / counts.sum()
    p_e = float((p_j**2).sum())
    return float("nan") if p_e == 1.0 else (p_bar - p_e) / (1 - p_e)


def _yes(value: str) -> bool | None:
    v = value.strip().lower()
    if v in ("y", "yes", "1", "true"):
        return True
    if v in ("n", "no", "0", "false"):
        return False
    return None


def score_audit(files: Sequence[Path]) -> dict[str, Any]:
    """Rates per method (majority vote over raters) and Fleiss' kappa per yes/no question."""
    raters = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as f:
            raters.append({row["id"]: row for row in csv.DictReader(f)})
    ids = sorted(set.intersection(*(set(r) for r in raters))) if raters else []
    methods = {i: raters[0][i]["method"] for i in ids}
    report: dict[str, Any] = {"files": [str(p) for p in files], "rows": len(ids),
                              "raters": len(raters), "by_method": {}, "kappa": {}}  # fmt: skip
    for q in QUESTIONS:
        counts = []
        for i in ids:
            votes = [_yes(r[i].get(q, "")) for r in raters]
            votes = [v for v in votes if v is not None]
            if len(votes) == len(raters):
                counts.append([votes.count(True), votes.count(False)])
        kappa = fleiss_kappa(np.array(counts)) if len(counts) >= 2 and len(raters) >= 2 else None
        report["kappa"][q] = None if kappa is None or np.isnan(kappa) else round(float(kappa), 3)
    for method in sorted(set(methods.values())):
        rows = [i for i in ids if methods[i] == method]
        entry: dict[str, Any] = {"rows": len(rows)}
        for q in QUESTIONS:
            majority = []
            for i in rows:
                votes = [v for v in (_yes(r[i].get(q, "")) for r in raters) if v is not None]
                if votes:
                    majority.append(sum(votes) * 2 > len(votes))
            entry[f"{q}_rate"] = round(sum(majority) / len(majority), 4) if majority else None
        useful = []
        for i in rows:
            for r in raters:
                with contextlib.suppress(ValueError):  # blank or non-numeric cell
                    useful.append(float(r[i].get("useful", "")))
        entry["useful_mean"] = round(float(np.mean(useful)), 3) if useful else None
        report["by_method"][method] = entry
    return report


def rejection_rates(
    store: ArtifactStore, states: Sequence[UserState], n: int, k: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Validator rejection counts and rates by type over ``n`` LLM explanations."""
    rows = collect_explanations(store, states, n, k, seed)
    counts = dict.fromkeys(REJECTION_TYPES, 0)
    fallbacks: dict[str, int] = {}
    rejected = 0
    for row in rows:
        if row["rejections"]:
            rejected += 1
        for code in row["rejections"]:
            key = code.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
        if row["fallback_reason"] and not row["rejections"]:
            fallbacks[row["fallback_reason"]] = fallbacks.get(row["fallback_reason"], 0) + 1
    total = max(1, len(rows))
    return {
        "explanations": len(rows),
        "llm": sum(r["method"] == "llm" for r in rows),
        "rejected": rejected,
        "rejection_rate": round(rejected / total, 4),
        "by_type": {k: {"count": v, "rate": round(v / total, 4)} for k, v in counts.items()},
        "fallbacks_without_text": fallbacks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="explanation audit (CLAUDE.md s14.9)")
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export", help="sample explanations for the three raters")
    exp.add_argument("--bundle", required=True)
    exp.add_argument("--n", type=int, default=100)
    exp.add_argument("--out", default="docs/annotation/explain_audit.csv")
    exp.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score", help="rates and Fleiss' kappa from the filled copies")
    sc.add_argument("--files", nargs="+", required=True)
    sc.add_argument("--out")
    rej = sub.add_parser("rejections", help="validator rejection rates over n explanations")
    rej.add_argument("--bundle", required=True)
    rej.add_argument("--n", type=int, default=1000)
    rej.add_argument("--out")
    rej.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.command == "score":
        report = score_audit([Path(p) for p in args.files])
    else:
        store = core.load_store(args.bundle)
        from litsense_core.llm.client import configure

        # Offline batch: wait for the rate window (the online limiter never sleeps) and stay
        # under the free tier's 15 requests per minute.
        client = configure({**(store.config.get("llm") or {}), "rpm_limit": 7})
        client.blocking_limiter = True
        states = demo_states(store)
        if args.command == "export":
            rows = collect_explanations(store, states, 3 * args.n, seed=args.seed)
            chosen = stratify(rows, args.n, args.seed)
            count = write_audit_csv(chosen, Path(args.out))
            llm_rows = sum(r["method"] == "llm" for r in chosen)
            print(f"wrote {count} explanations to {args.out}: {llm_rows} llm, {count - llm_rows}"
                  f" template (LLM available: {core.health()['llm']})")  # fmt: skip
            return 0
        report = rejection_rates(store, states, args.n, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
