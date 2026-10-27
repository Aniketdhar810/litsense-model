# LitSense: ML and GenAI

The machine-learning half of LitSense, a book recommender built on Amazon Reviews 2023 (Books) that learns
what readers valued in a book, not just which books they rated highly. `CLAUDE.md` is the implementation brief
and `docs/LitSense_PRD.md` the product spec.

| Path | What it is |
| --- | --- |
| `ml/` | Offline pipeline (runs on Colab): data, models, NLP, ranking, evaluation. Every step is a stage. |
| `ml/config.yaml` | The only place for hyperparameters, paths, thresholds and model names. |
| `litsense_core/` | Serving package the backend imports (CPU, no torch). `litsense_core.mock` serves the full API today. |
| `tests/` | pytest suite; `tests/fixtures/tiny/` generates a small synthetic raw dataset. |
| `docs/` | `API_CONTRACT.md` (backend boundary), `decisions.md`, `progress.md`. |
| `reference/` | Read-only notebooks from the earlier MovieLens project. |

## Setup

Python 3.11 or 3.12.

```bash
python -m venv .venv
```

Activate it (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), then install what you need:

```bash
pip install -r requirements-core.txt
```

`requirements-core.txt` is enough to use `litsense_core`. For the offline pipeline add the ML stack:

```bash
pip install -r requirements-core.txt -r requirements-ml.txt
```

## Everyday commands

The Makefile wraps these; it needs `make` and a POSIX shell (on Windows, Git Bash plus `mingw32-make`, or just run
the command on the right).

| Target | Command |
| --- | --- |
| `make test` | `python -m pytest -m "not slow"` |
| `make test-all` | `python -m pytest` (includes tests that need MiniLM, spaCy or a GPU) |
| `make lint` | `python -m ruff check .` then `python -m ruff format --check .` |
| `make format` | `python -m ruff format .` then `python -m ruff check --fix .` |
| `make tiny` | `python -m tests.fixtures.tiny.make_tiny --out local/tiny` |
| `make stages` | `python -m ml.stages list` |
| `make install-ci` | core requirements plus the test subset of the ML requirements (what CI installs) |

## The pipeline

Every offline step is a stage declared with `@stage(name, inputs, outputs, config_keys)` (see
`ml/common/stage.py`). Stages skip themselves when nothing they depend on has changed, write outputs atomically,
and finish by writing a manifest that downstream stages and the evaluator trust.

```bash
python -m ml.stages list
```

```bash
python -m ml.stages run all --until eval.val
```

Large sources (`meta_Books`, the reviews, the Goodreads graph) are streamed from their URLs and filtered on
the fly; they are never stored. Add `--sample N` to read only the first N lines of each streamed source from
`local/raw_sample/` (or the URL when no local copy exists) and write under `./local` instead of Google Drive;
add `--force` to rerun a stage that is up to date. Stages arrive phase by phase (CLAUDE.md §17).

```bash
python -m ml.stages run all --until data.meta --sample 50000
```

On Colab, run `notebooks/00_colab_setup.ipynb` (mounts Drive, clones the repo, `make install-colab`, which
keeps Colab's own torch) and then the numbered notebooks, which only call stages.

## For the backend

Build against `litsense_core.mock` now; switch the import to `litsense_core` when the bundle ships. Signatures
and schemas are identical. See `docs/API_CONTRACT.md`.

## Data and citation

Amazon Reviews 2023: Hou et al. (2024), *Bridging Language and Items for Retrieval and Recommendation*,
arXiv:2403.03952. UCSD Goodreads Book Graph: Wan & McAuley, RecSys 2018; Wan, Misra, Nakashole & McAuley,
ACL 2019. Both for research use.
