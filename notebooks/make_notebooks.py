"""Write the thin Colab notebooks (stage calls only; no logic)."""

import json
import sys
from pathlib import Path


def cell(kind: str, text: str, n: int) -> dict:
    lines = text.strip("\n").split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    base = {"cell_type": kind, "id": f"c{n:02d}", "metadata": {}, "source": source}
    if kind == "code":
        base |= {"execution_count": None, "outputs": []}
    return base


def notebook(cells: list[tuple[str, str]]) -> dict:
    return {
        "cells": [cell(kind, text, i) for i, (kind, text) in enumerate(cells)],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = [
    (
        "markdown",
        """
# LitSense: Colab setup

Run once per Colab session, top to bottom. It mounts Google Drive (all pipeline outputs live in
`MyDrive/litsense/`), clones the repository and installs the pinned requirements **except torch**:
Colab's preinstalled torch is used as is (docs/decisions.md D21). No pipeline logic lives here.

While a `!` command runs, Colab shows an empty text box under the cell. It is Colab's input box for
shell commands, not a question: leave it empty and wait for the cell to finish.
""",
    ),
    (
        "code",
        """
from google.colab import drive

drive.mount("/content/drive")
""",
    ),
    (
        "code",
        """
REPO = "https://github.com/Aniketdhar810/LitSense.git"
BRANCH = "main"  # the branch to run; change it to test a branch under review
!rm -rf /content/LitSense
!git clone --depth 1 --branch $BRANCH $REPO /content/LitSense
""",
    ),
    (
        "code",
        """
%cd /content/LitSense
!make install-colab
""",
    ),
    (
        "code",
        """
!python --version
!pip list 2>/dev/null | grep -iE "^(torch|numpy|pandas|pyarrow|requests) "
!python -m ml.stages list
""",
    ),
]

DATA = [
    (
        "markdown",
        """
# Phase 1: data pipeline

## Week 2: 5-core ratings, book metadata, genre-slice decision

Each Colab notebook has its own runtime, so the first four cells repeat `00_colab_setup`
(mount Drive, clone, install without torch). `data.download` keeps the 5-core ratings CSV
(0.14 GB) on Drive; `data.meta` streams `meta_Books` (4.94 GB) from the UCSD server and filters it
on the fly, so nothing large is stored.

**While it runs:** `data.meta` takes a while (the whole 4.94 GB file passes through) and prints a
progress line about once a minute with GB done, speed and time left. Colab shows an empty text box
under a running `!` command; it is Colab's input box, not a question. Leave it empty and do not
stop the cell. If Colab disconnects anyway, run the setup cells and the same cell again: work
saved in the checkpoints on Drive is reused.

Outputs in `MyDrive/litsense/data/v0/`: `ratings_5core.csv.gz`, `books_raw.parquet`,
`category_counts.csv`, `genre_slice_proposal.json`; run manifests in
`MyDrive/litsense/manifests/v0/`.
""",
    ),
    (
        "code",
        """
%cd /content/LitSense
!python -m ml.stages run all --until data.meta < /dev/null
""",
    ),
    (
        "code",
        """
!python -m ml.data.meta
""",
    ),
    (
        "markdown",
        """
**Stop here.** Choose `data.genre_slices` from the report above (`category_counts.csv` has the
full table) and send the choice back; it is recorded in `ml/config.yaml` and `docs/decisions.md`
before any later stage runs.

Optional, once `data.use_goodreads_graph: true` is committed: rerun the setup cells above (to
pull it), then the cell below to build the Goodreads ISBN index (streams about 2.2 GB, keeps only
matching rows).
""",
    ),
    (
        "code",
        """
%cd /content/LitSense
!python -m ml.stages run data.goodreads < /dev/null
""",
    ),
    (
        "markdown",
        """
## Week 3: editions to works (`data.canonical`)

Run the setup cells at the top first, so the clone has the merged Week 3 code. The cell below
checks every earlier stage and reruns only what changed: `data.download` and `data.meta` run once
more because their code changed (about 8 minutes, the `meta_Books` stream again), `data.lid` keeps
fastText's language-ID model (0.9 MB), `data.goodreads` writes its empty tables, and
`data.canonical` merges the 454K editions into works and keeps the works with an edition in a
genre slice.

Outputs: `MyDrive/litsense/data/v0/asin_to_canonical.parquet`, `books_canonical.parquet`,
`lid.176.ftz`; `MyDrive/litsense/results/v0/canonical/canonical_report.json` and
`merge_audit.csv`.
""",
    ),
    (
        "code",
        """
%cd /content/LitSense
!python -m ml.stages run all --until data.canonical < /dev/null
""",
    ),
    (
        "code",
        """
!python -m ml.data.canonical
""",
    ),
    (
        "markdown",
        """
**Stop here.** Review `MyDrive/litsense/results/v0/canonical/merge_audit.csv` (through Google
Drive for desktop, or download `results/v0/` as a zip): for each `audit_id`, write `Y` in
`verdict` if the decision is right (merged editions are one work; not-merged pairs are different
works) and `N` if it is wrong. Send the reviewed file back. Do not run `data.reviews` before the
audit is recorded in `docs/decisions.md`.
""",
    ),
    (
        "markdown",
        """
## Week 3: reviews, cleaning, k-core, split, EDA

Run the setup cells at the top first. The cell below runs `data.reviews` (streams the 6.2 GB review
file, about 20-30 minutes, progress every minute, resumable), `data.clean` (fastText language ID),
`data.kcore_split` (the frozen v0 dataset: `interactions`, `books`, `reviews`, `id_map.json`,
`splits.json`) and `data.eda` (`MyDrive/litsense/results/v0/eda/`). Earlier stages show
"up to date".
""",
    ),
    (
        "code",
        """
%cd /content/LitSense
!python -m ml.stages run all --until data.eda < /dev/null
""",
    ),
    (
        "code",
        """
from google.colab import drive

drive.flush_and_unmount()  # finish uploading to Drive before the session can end (D38)
""",
    ),
]

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
DATA = DATA[:1] + SETUP[1:] + DATA[1:]
EVAL = [
    (
        "markdown",
        """
# Phase 2: evaluation harness, baselines and LightGCN (Weeks 4-5)

Use a GPU runtime (Runtime > Change runtime type > T4 GPU). Run the setup cells, then the cell
below:
`nlp.content` embeds every book's metadata with MiniLM, `models.lightgcn_val` trains LightGCN on
train with early stopping, and `eval.val` fits random, popularity, ItemKNN, ALS and content-only and
evaluates them with both LightGCN variants on validation. Results go to
`MyDrive/litsense/results/v0/val/`, `results/v0/content/` and `results/v0/training/`. The last cell
finishes uploading to Drive.
""",
    ),
    *SETUP[1:],
    (
        "code",
        """
%cd /content/LitSense
!python -m ml.stages run all --until eval.val < /dev/null
""",
    ),
    (
        "code",
        """
from google.colab import drive

drive.flush_and_unmount()  # finish uploading to Drive before the session can end (D38)
""",
    ),
]
NOTEBOOKS = (
    ("00_colab_setup.ipynb", SETUP),
    ("01_data.ipynb", DATA),
    ("02_eval_baselines.ipynb", EVAL),
)
for name, cells in NOTEBOOKS:
    (out / name).write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
    print("wrote", out / name)
