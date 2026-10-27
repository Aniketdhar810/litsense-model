"""Pipeline CLI (CLAUDE.md s7.0).

    python -m ml.stages list [--sample N]
    python -m ml.stages run <stage> [--sample N] [--force]
    python -m ml.stages run all [--until <stage>] [--sample N] [--force]

``--sample N`` reads only the first N lines of each raw file and writes under
``project.local_root``; without it, stages read ``project.raw_root`` and write under
``project.drive_root`` (the Colab layout of CLAUDE.md s4.5).
"""

import argparse
import importlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from ml.common.config import DEFAULT_CONFIG_PATH, load_config
from ml.common.errors import ConfigError, StageError
from ml.common.stage import REGISTRY, StageRegistry, run_all, run_stage, stage_status

# Modules whose @stage decorators register the pipeline. Each phase appends its own.
STAGE_MODULES: tuple[str, ...] = (
    "ml.data.download",
    "ml.data.meta",
    "ml.data.goodreads",
    "ml.data.canonical",
    "ml.data.reviews",
    "ml.data.clean",
    "ml.data.split",
    "ml.data.eda",
    "ml.nlp.content",
    "ml.models.lightgcn",
    "ml.models.projection",
    "ml.eval.validation",
    "ml.serve.export",  # models.lightgcn_serve + serve.export: after eval.val (D56)
    "ml.nlp.sentences",  # Phase 3 review NLP: after eval.val so `--until eval.val` skips them
    "ml.nlp.aspects",
    "ml.nlp.sentiment",
    "ml.nlp.centroids",
    "ml.nlp.profiles_offline",
    "ml.nlp.spoiler_train",
    "ml.nlp.snippets",
    "ml.nlp.onnx_export",
    "ml.nlp.annotate",
    "ml.genai.synthesis",  # Phase 6 (D70): review summaries, an input of serve.export; it is
    "ml.ranking.build_training",  # registered before the ranking stages so the export keeps its
    "ml.ranking.train_ranker",  # place before ranking.eval_val in `run all --until` order
    "ml.ranking.eval_val",  # the ranker under the s8 protocol on every val user
    "ml.ranking.filter_diagnostic",  # which s13.2 filter drops the found positives
    "ml.models.lightgcn_sweep",  # `run all --until eval.val` never reaches it (D47)
    "ml.nlp.validate_aspects",  # last: needs the labelled CSV in the raw dir; run by name (D68)
)


def load_stages() -> StageRegistry:
    """Import every stage module so its stages register, and return the registry."""
    for module in STAGE_MODULES:
        importlib.import_module(module)
    return REGISTRY


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(prog="python -m ml.stages", description=__doc__.split("\n")[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="config YAML path")
    sub = parser.add_subparsers(dest="command", required=True)
    list_cmd = sub.add_parser("list", help="show every stage and whether it is up to date")
    list_cmd.add_argument("--sample", type=int, metavar="N", help="report on the --sample layout")
    run_cmd = sub.add_parser("run", help="run one stage, or 'all' in dependency order")
    run_cmd.add_argument("target", help="stage name or 'all'")
    run_cmd.add_argument("--sample", type=int, metavar="N", help="first N raw lines; local output")
    run_cmd.add_argument("--force", action="store_true", help="rerun even if up to date")
    run_cmd.add_argument("--until", metavar="STAGE", help="with 'all': stop after this stage")
    args = parser.parse_args(argv)
    if args.sample is not None and args.sample <= 0:
        parser.error("--sample must be a positive number of lines")
    if args.command == "run" and args.until and args.target != "all":
        parser.error("--until only applies to 'run all'")

    try:
        cfg = load_config(args.config)
        registry = load_stages()
        if args.command == "list":
            _print_status(cfg, registry, args.sample)
        elif args.target == "all":
            run_all(cfg, registry=registry, until=args.until, sample=args.sample, force=args.force)
        else:
            run_stage(args.target, cfg, registry=registry, sample=args.sample, force=args.force)
    except (ConfigError, StageError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def _print_status(cfg: Mapping[str, Any], registry: StageRegistry, sample: int | None) -> None:
    stages = registry.ordered()
    if not stages:
        print("no stages registered yet")
        return
    width = max(len(s.name) for s in stages)
    for s in stages:
        print(f"{s.name:<{width}}  {stage_status(s.name, cfg, registry=registry, sample=sample)}")


if __name__ == "__main__":
    raise SystemExit(main())
