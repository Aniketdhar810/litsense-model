# Progress

## 2026-10-12: Phase 0 (Week 1). Repo skeleton, stage framework, API contract, mock

### Built
- **Skeleton and tooling:** `pyproject.toml` (setuptools; dependencies read from the requirements files; ruff and
  pytest config), `requirements-core.txt` / `requirements-ml.txt` (exact pins, D1–D3), `Makefile`, `.gitignore`,
  `README.md`, GitHub Actions CI (`.github/workflows/ci.yml`: Python 3.11 and 3.12, ruff, `pytest -m "not slow"`).
  Empty `ml/` subpackages for later phases.
- **`ml/config.yaml`:** §5 verbatim except quoted regexes (D4) and `project.local_raw_root` (D5).
- **`ml/common/`:** `config.py` (load, dotted lookup, structural validation), `seed.py`
  (`set_global_seed`), `log.py`, `errors.py`, `manifest.py` (hashes, file descriptions, fingerprints, atomic JSON,
  git SHA), `stage.py` (registry, `@stage`, `StageContext`, fingerprints, skip/rerun, staging plus atomic moves,
  manifests, work dirs for checkpoints, dependency ordering, status).
- **`ml/stages.py`:** CLI with `list` and `run <stage|all> [--sample N] [--force] [--until X]`. No stages
  are registered yet; Phase 1 adds them to `STAGE_MODULES`.
- **`litsense_core/`:** `schemas.py` (all §15.2 models; deviations in D10), `mock.py` (every §15.1 function over
  a synthetic 200-book catalogue, plus the proposed `catalog()`), `__api_version__ = "0.1.0"`.
- **`docs/API_CONTRACT.md`:** functions, schema rules, chat protocol, behaviour rules, budgets, open questions.
- **`tests/fixtures/tiny/make_tiny.py`:** deterministic raw dataset in Amazon and Goodreads formats (50 users,
  80 books, 600 interactions over 3 years) with `truth.json` for every planted case (D13).

### Tests run
- `make install-ci lint test` (via `mingw32-make`, Windows, Python 3.12 venv with the CI package set):
  ruff clean, **100 passed, 1 skipped**. The skipped test is torch seeding (no torch in the venv); checked by hand
  with the system Python and torch 2.8.0 (same seed gives the same draws, a different seed differs).
- Test files: `tests/ml/common/test_stage.py` (22), `test_config.py` (11), `test_manifest.py` (5),
  `test_seed.py` (3), `tests/litsense_core/test_mock.py` (44), `test_schemas.py` (5), `test_imports.py` (3,
  including the no-torch subprocess check), `tests/fixtures/tiny/test_make_tiny.py` (8).
- Dependency check: the full core+ml pin set resolves with binary wheels only for manylinux cp311 and cp312
  (`pip install --dry-run --only-binary=:all:`); the core set installs on Windows cp312.
- Packaging: `pip wheel .` builds `litsense-0.1.0` with the pinned core dependencies and an `ml` extra.

### Week 1 gate
- [x] Phase 0 deliverables built: skeleton, pyproject, requirements, config, stage framework and manifests,
  seed utils, schemas, mock, API contract, tiny fixture generator, Makefile, CI workflow
- [x] Tests for every new module pass locally; lint clean
- [x] Decisions logged (D1–D14 in `docs/decisions.md`)
- [ ] CI green on GitHub: the workflow is written and its exact commands pass locally, but the branch has not been
  pushed (waiting for the user's go-ahead)
- [ ] Mock serves the teammate's skeleton app: teammate's step, using `docs/API_CONTRACT.md` §1
- [ ] User reviewed the gate
- Not applicable yet: `--sample`/Colab stage runs, manifests and registry entries, results and figures (no data
  stages exist until Phase 1).

### Open issues
1. Needs user OK: stage fingerprint uses each stage's import closure instead of the repo-wide git SHA (D6);
   proposed `catalog()` contract addition and the mock's other protocol choices (D12); keeping Colab's preinstalled
   torch (D2).
2. For the teammate: exact core pins (pydantic 2.13.5, httpx 0.28.1, numpy 2.4.6) must fit the backend; Python
   3.11; half stars; whether onboarding picks are also stored as ratings (API_CONTRACT §7).
3. Deferred, by design: the tiny serving bundle (Phase 4, with `serve.export`); `make sample` (Phase 1, with
   `data.download`); `dev/dev_server.py` (optional, Phase 4).
4. Verify in Phase 1 against real files: Goodreads values typed as strings, Amazon timestamps in milliseconds, the
   §6 field lists, and that `fasttext-wheel` predicts correctly under numpy 2.

### Next (Phase 1, Week 2)
`data.download`, `data.meta` with `category_counts.csv`, then stop for the user to choose the genre slice;
`data.goodreads`.

## 2026-10-12: Raw sample data and format inspection

### Built
- `local/raw_sample/` (gitignored): first 200,000 lines of the 5-core ratings, 50,000 of `meta_Books`, 50,000
  of the reviews and 5,000 of the Goodreads spoiler file, streamed with `curl | gunzip | head` (no full
  downloads) and stored gzipped under the original names.
- `ml/config.yaml`: `sources.goodreads_base` (UCSD mirror; the Drive links are dead).
- `docs/decisions.md`: D15 (Goodreads source), D16 (sample files), F1–F12 (format findings).

### Tests run
- `pytest tests/ml/common/test_config.py` and the raw-file-name test after the config change: pass.

### Open issues
1. F1: `meta_Books` includes Kindle editions (about 21% of 5-core interactions in the sample) and audiobooks;
   the user decides before `data.canonical` (Week 3) whether to keep them as editions of their works.
2. The Week-2 plan (download stages, framework fixes, `data.meta`, `data.goodreads`) still awaits approval.

## 2026-10-13: Week 2 (Phase 1) up to the genre-slice gate

### Built
- **Stage framework:** `url:<source>` inputs streamed from their URL; `ctx.source()` returns the URL, or the
  local head in `local/raw_sample/` for `--sample` runs (D18).
- **`ml/data/stream.py`:** line streaming with on-the-fly gunzip, Range-resume after dropped connections,
  truncation and size checks, `PartWriter` output checkpoints.
- **`ml/data/parsing.py`:** HTML cleanup, ISBN classification and conversion, years, pages, authors,
  formats and edition kinds, images, Goodreads series suffixes, fast field reads from raw JSON lines.
- **Stages:** `data.download` (5-core CSV, validated, kept on Drive), `data.meta` (streams `meta_Books`,
  writes `books_raw.parquet`, `category_counts.csv`, `genre_slice_proposal.json`, then STOP),
  `data.goodreads` (off by default; when on, keeps only Goodreads rows matching `books_raw` ISBNs).
- **Config:** `data.use_goodreads_graph: false`, `data.stream`, `data.genre_slice_rule`,
  `project.local_raw_root: ./local/raw_sample`; `gdown` removed.
- **Colab:** `make install-colab` (no torch), `notebooks/00_colab_setup.ipynb`, `notebooks/01_data.ipynb`.
- **Fixture:** Kindle, audiobook and other F-cases planted (D23).
- **Decisions:** D17 (Kindle/audio as editions, user), D18–D23.

### Tests run
- `ruff check .`, `ruff format --check .`: clean (notebooks included).
- `pytest -m "not slow"`: **170 passed, 1 skipped** (torch seeding; no torch in the venv). New:
  `tests/ml/data/` (stream 10, parsing 36, download 5, meta 8, goodreads 2, real-sample 3) and the `url:`
  framework tests (2).
- Local sample runs (real data, laptop):
  - `run all --until data.meta --sample 50000`: 37 s. `data.download` fetched and validated the full
    5-core (9,488,297 ratings, 776,370 users, 495,063 items). `data.meta` read the 50,000-line local head:
    7,707 lines matched the 5-core, 155 dropped by the title blocklist, 417 by `min_pages`, 7,135 kept.
    Proposal on this head: Literature & Fiction, Mystery Thriller & Suspense, Teen & Young Adult (not
    meaningful on a head; the full Colab run decides).
  - `run data.goodreads --sample 50000` with the flag off: 0.1 s, no network, empty tables with schema.
  - Smoke test with the flag on (separate output folder): 77 of the first 50,000 Goodreads books matched
    the 7,135-book head, with series names and work titles joined, in 16 s.

### Gate (Week 2): status
- [x] `data.download`, `data.meta` (with `category_counts.csv`), `data.goodreads` built and tested
- [x] `--sample` runs pass locally
- [x] Decisions logged
- [ ] Full run on Colab (`notebooks/01_data.ipynb`)
- [ ] User chooses `data.genre_slices`; recorded in `ml/config.yaml` and `docs/decisions.md`

### Follow-up (same day)
- `books_raw.parquet` and checkpoint parts written with zstd: 24.5 MB to 17.1 MB on the sample, same
  content (D24). `features` checked against `description` and kept: not a duplicate (D24).
- Branch rebased onto `main` after PR #2 merged.

### Open issues
1. After the slice choice: optionally enable `data.use_goodreads_graph` and run `data.goodreads` on Colab.

## 2026-10-13: First Colab run and fixes

### What happened
- `data.download` ran on Colab in 23 s with the same counts as locally. `data.meta` was interrupted by the
  user while streaming (no progress output; Colab's input box looked like a prompt). Colab runs Python 3.13,
  where the `fasttext-wheel` pin had no wheel, so the pinned install could not complete.

### Fixed (D25)
- Progress lines every minute for long streams; notebook guidance about Colab's input box; `< /dev/null`.
- `fasttext-wheel` replaced by `fasttext-predict==0.9.2.4` (cp311-cp313 wheels, no numpy dependency).
- CI matrix gains Python 3.13.

### Tests run
- `pytest -m "not slow"`: 171 passed, 1 skipped (new: `tests/ml/common/test_log.py`).
- Pin resolution with binary wheels only: cp311, cp312, cp313 all resolve (104 packages).

### Next
- Rerun `notebooks/01_data.ipynb` on Colab. Because `data.meta`'s code changed, its fingerprint changed
  and the interrupted run's checkpoints are discarded: the stage starts the stream from the beginning.

## 2026-10-13: Full Colab run, genre slices, notebook check

### Full Colab run of `notebooks/01_data.ipynb` (branch `main` at `04c3419`)
- The notebook cloned `main`, which lacked the fix `7b3fc9e`, so the pinned install failed again on
  `fasttext-wheel` and the stages ran on Colab's preinstalled pandas 2.2.3 / pyarrow 23.0.1 / numpy 2.1.3
  (Python 3.13.15). The stages do not depend on those versions; the outputs are valid.
- `data.download`: 23.9 s; 9,488,297 ratings, 776,370 users, 495,063 items.
- `data.meta`: 446 s; 4,448,181 lines read; 495,063 matched; dropped 13,408 (title blocklist) and
  27,647 (`min_pages`); 454,008 books in `books_raw`; no duplicate parent_asins. Category table in D26.
- `data.goodreads`: flag off, empty tables, 0.1 s.

### Built
- Fresh branch `claude/litsense-week2-colab-fixes` from `main` with the Colab fix cherry-picked (D27).
- `tests/test_notebooks.py`: CI fails on notebooks with outputs (D27).
- `data.genre_slices` set (D26); `require_genre_slices` test updated.
- `python -m ml.data.meta --find TITLE ...`: title lookup in `books_raw` (category, edition kind, 5-core
  interactions per edition).

### Tests run
- `ruff check .`, `ruff format --check .`: clean. `pytest -m "not slow"`: 175 passed, 1 skipped (torch).

### Gate (Week 2)
- [x] `data.download`, `data.meta`, `data.goodreads` built, tested, run in full on Colab
- [x] Genre slices chosen by the user and recorded (config, D26)
- [x] Title lookups run on Colab and reviewed (2026-10-13, see below): **Week 2 gate closed**

## 2026-10-13: Week 2 gate closed; Week 3 part 1 (`data.canonical`)

### Week 2 gate: title lookups (`python -m ml.data.meta --find`, full Colab run)
- All eight titles are in `books_raw`. Under rule A (D28), every one has a kept work.
- Editions of one work sit in different categories: Project Hail Mary print in Literature & Fiction, audio
  in SF & Fantasy; Mistborn book 1 and the box set in SF & Fantasy, books 2–6 in Literature & Fiction; The
  Midnight Library has editions in SF & Fantasy, Literature & Fiction, no category and Textbooks.
- The most-rated edition is often Kindle: Gone Girl 8,184 interactions, The Silent Patient 2,217, Circe 743.
- The lookup rows became a regression test of the expected work groupings
  (`test_real_titles_from_the_gate_lookup`).
- User answers (Week 3 plan approved): A = merge all editions, keep a work if any edition is in a slice;
  B = results come back through Google Drive for desktop (or a zip of `results/v0/`), only `results/v0/` and
  the registry entry are copied into the repo; C = keep `lid.176.ftz` in `data/`.

### Built
- `ml/data/titles.py`: matching key, display title, box-set and volume detection, series parsing, author key.
- `ml/data/genres.py`: 26 genre labels plus fallback (D29).
- `ml/data/canonical.py`: stage `data.canonical` (D28). Outputs `asin_to_canonical.parquet`,
  `books_canonical.parquet`, `results/v0/canonical/canonical_report.json` and `merge_audit.csv`; CLI
  `python -m ml.data.canonical` prints the report summary.
- Stage `data.lid` (D30); config `canonical.subtitle_noise_patterns`, `canonical.subset_guard_threshold`,
  more edition-noise terms; `notebooks/01_data.ipynb` Week 3 cells.

### Tests run
- `ruff check .`, `ruff format --check .`: clean. `pytest -m "not slow"`: 225 passed, 1 skipped (torch).
  New: titles 33, genres 3, canonical 12 (§16.2 cases, the subset trap, other authors, Kindle page-source
  ISBN, Goodreads works, the gate-lookup titles, the tiny fixture end to end with Goodreads off and on, a
  seeded audit), `data.lid` 2.
- Local sample `run all --until data.canonical --sample 50000`: download 41.6 s, lid 1.2 s (the real model
  passed the magic-number check), meta 15.5 s, goodreads 0.1 s, canonical 2.2 s. On the 7,135-edition
  head: 7,108 works (27 with 2+ editions, all `title` merges), 36 box sets, 2,612 kept works with 69,687
  interactions. All 12 kept multi-edition works in the sample audit are correct on inspection; the head is
  too small to measure precision (the first 50K meta lines rarely hold two editions of a work).
- Scale check (laptop, sample head copied 64 times = 456,640 editions with heavy duplication): prepare
  38 s, merge 103 s, records 39 s. That run produced 3.7M near-miss pairs, so near misses are now kept in a
  bounded reservoir per reason.
- Bug found and fixed: pandas stored missing volume numbers as NaN, and NaN != NaN made the volume guard
  block every merge in frames that mixed known and unknown volumes. Volumes now stay `None`, with a
  regression test.

### Gate (Week 3, part 1): status
- [x] `data.canonical` built and tested; `--sample` run passes locally
- [x] Decisions logged (D28–D30)
- [ ] Full run on Colab (`notebooks/01_data.ipynb`, Week 3 cells)
- [ ] User reviews `results/v0/canonical/merge_audit.csv`; merge precision recorded in D28
- Then: `data.reviews`, `data.clean`, `data.kcore_split`, `data.eda`, registry hash.


## 2026-10-14: `data.canonical` full Colab run and merge audit (first pass)

### Full Colab run (main after PR #6; output pasted by the user)
- Upstream stages were up to date; `data.canonical` took 193.9 s.
- 454,008 editions became 383,360 works (50,383 with 2 or more editions); 2,948 box sets.
- Merge edges: isbn 705, page_source 2,370, title 67,573.
- Kept 131,081 works, 170,864 editions, 4,497,594 interactions. Rule A rescued 4,071 editions and
  111,155 interactions.
- Kept works by primary slice:
  - Literature & Fiction: 77,539 works, 2,202,974 interactions.
  - Mystery, Thriller & Suspense: 42,862 works, 2,079,937 interactions.
  - Science Fiction & Fantasy: 10,680 works, 214,683 interactions.
- Near misses by reason: below_threshold 38,325, subset_guard 35,644, volume 8,784, year 3,448.

### Audit (334 rows)
- My first pass, for the user to confirm: 97/100 merged works correct; 44/50 not-merged pairs correct.
- Wrong merges:
  - G018: shared-ISBN data error (fixed, D31).
  - G054: sequel at `token_sort` exactly 90.
  - G004: a two-in-one "Animal Farm: 1984" edition (debatable).
- All 6 wrong non-merges were blocked by the subset guard (same work, subtitle on one side only).

### Built
- D31: author guard on shared-ISBN merges, plus a regression test. `pytest -m "not slow"`: 226 passed,
  1 skipped (torch).
- `context.md` handoff notes carried over (their commit had missed PR #6).

### Next
- The user returns the edited verdicts; rerun `data.canonical` on Colab; record precision in D28/D31; then
  `data.reviews`.

## 2026-10-14: Subtitle rule (D32), audit recorded

- `title_head` (titles.py), `subtitle_unions` plus blurb readers (canonical.py), `assign_works` (shared
  glue), a `subtitle_merge` audit section and a `subtitle_rule` report block. Config:
  `canonical.subtitle_description_threshold: 70`, `subtitle_description_chars: 500`.
- Tests: Cold Hand in Mine merges; Mistborn vs Mistborn: The Final Empire with different blurbs does not;
  guards for other author, empty description, volume and box set; audit section; `title_head`.
  `pytest -m "not slow"`: 234 passed, 1 skipped (torch).
- Sample run (50K head): subtitle rule: 6 candidates, 2 merged, 3 dissimilar blurbs, 1 blocked by the
  volume or work guard.
- Run-1 audit verdicts (model-drafted, spot-checked by the user) recorded in D31 and committed as
  `results/v0/canonical/merge_audit_run1_reviewed.csv`.
- Next: merge PR #7 once CI is green; the user reruns `data.canonical` on Colab (one cell), and the
  `subtitle_merge` section is reviewed the same way; then `data.reviews`.

## 2026-10-14: Run 2 of `data.canonical`, number guard (D33)

- Colab run 2 (after PR #7): 454,008 editions became 381,949 works; merge edges isbn 513, page_source
  2,370, subtitle 1,603, title 67,573; kept 130,555 works and 4,499,855 interactions; 225 s.
- Subtitle audit: 30/30 correct (model-drafted). Found NYPD Red 2/3/4/6 merged into one work by the title
  rule; fixed with D33 plus a test. `pytest -m "not slow"`: 235 passed, 1 skipped.
- Next: run 3 on Colab; check that the largest kept works are clean; then `data.reviews`.

## 2026-10-14: `data.canonical` run 3 accepted; `data.reviews` built

- Run 3 (D33 number guard): 384,579 works, 130,960 kept, 4,499,455 interactions; largest works clean.
- `ml/data/reviews.py` (stage `data.reviews`, D34) with tests: text hash, the seconds-timestamp guard, the
  two-step dedup, and the tiny fixture end to end. `pytest -m "not slow"`: 239 passed, 1 skipped.
- Local sample: 194 reviews matched (the sample heads of meta and reviews barely overlap, D16).
- Next: `data.clean`, `data.kcore_split`, `data.eda`; then one Colab run of the remaining Week 3 stages.

## 2026-10-14: `data.clean`, `data.kcore_split`, `data.eda`

- Built `ml/data/clean.py` (D35), `ml/nlp/aspect_seeds.py`, `ml/data/kcore.py` + `ml/data/split.py`
  (stage `data.kcore_split`, D36) and `ml/data/eda.py` (stage `data.eda`). `01_data.ipynb` gets one
  cell for all of them.
- Tests: cleaning rules (with a stub language ID), k-core invariants, sizing reproducibility, split
  chronology, one interaction per work with `has_text`, the tiny fixture end to end through `data.eda`,
  and `id_map.json` stable across reruns. `pytest -m "not slow"`: 254 passed, 1 skipped.
- Local sample: `data.clean` on the 194 sample reviews kept 183 as `text_ok`; 11 were too short.
- Next: ONE Colab run of `run all --until data.eda`; then record sizing f, split stats and the slice
  mix, and close the Week 3 gate (registry hash).

## 2026-10-14: Colab `data.reviews` failure, checkpoint fix (D37)

- A first Colab run streamed all reviews and stopped; the rerun failed because Drive had lost
  `part-00089.parquet` while keeping `progress.json`. `PartWriter` now validates parts, rolls back to the
  last good one, and marks finished streams. Tests: lost part, old-format checkpoint, finished stream.
  `pytest -m "not slow"`: 257 passed, 1 skipped.

## 2026-10-14: Week 3 gate

- Colab: every stage through `data.eda` complete. `results/v0/` (canonical audit and report, EDA
  summary, attrition table and figures) is committed; `models/registry.json` holds the v0 file sha256s,
  fingerprint and sizing.
- The notebook's data cell now ends with `drive.flush_and_unmount()` (D38).

### Gate (Week 3)
- [x] All Phase 1 stages run with `--sample` locally and in full on Colab; manifests complete
- [x] Registry updated (`models/registry.json`, v0)
- [x] Tests pass; CI green
- [x] Results committed (`results/v0/canonical`, `results/v0/eda`)
- [x] Decisions logged (D28-D38); merge audit recorded (model-drafted, spot-checked by the user)
- [ ] User reviews the gate, including the SF & Fantasy share after sizing (`summary.json` `slice_mix`)

## 2026-10-14: Week 4 harness and baselines built

- `ml/eval/` (data, metrics, evaluator, strata, bootstrap, report, lock), `ml/models/baselines.py`
  (random, popularity, ItemKNN, ALS, content-only), `ml/nlp/embed.py` + `ml/nlp/content.py` (stage
  `nlp.content`), stage `eval.baselines_val`, `notebooks/02_eval_baselines.ipynb`. Decisions D39 (v1
  open items) and D40.
- Tests: hand-computed metrics, masking, degenerate scores, cutoff mismatch, bootstrap, strata, test
  lock, baseline shapes and cold items, content text builder (no review text), the tiny fixture
  through `eval.baselines_val`. `pytest -m "not slow"`: 273 passed, 1 skipped.
- Next: one Colab run (`run all --until eval.baselines_val`); commit `results/v0/val` and
  `results/v0/content`; the Week 4 gate.

## 2026-10-15: Week 4 gate (validation baselines)

- Colab: `data.clean` and `data.kcore_split` rebuilt the output Drive had lost (D42). `data.eda` stayed
  up to date, so the rebuilt v0 files have the same sha256 as `models/registry.json`. `nlp.content` and
  `eval.baselines_val` complete.
- Committed `results/v0/val/`:
  - `table_main.csv`, `table_strata.csv`, `table_longtail.csv`, `bootstrap.csv`;
  - per-model JSON and per-user rows;
  - two figures.
  Also `results/v0/content/` (anisotropy check, neighbour samples).
- Read from those files: ItemKNN, content-only and ALS all beat popularity on NDCG@10 and Recall@20
  (paired CIs exclude zero). Content-only is the only baseline that scores cold items and the strongest
  on long-tail truth. Popularity recommends no long-tail items. Mean pairwise content cosine is below
  the anisotropy warning. Exact values are in `results/v0/val/` and `results/v0/content/`.

### Gate (Week 4)
- [x] Metric unit tests pass (hand-computed case, masks, degenerate scores, bootstrap, lock)
- [x] Random and popularity (and ItemKNN, ALS, content-only) evaluated end to end on val, with strata and CIs
- [x] Results committed; decisions logged (D39-D42); CI green
- [ ] User reviews the gate

## 2026-10-15: Handoff (state at the end of Week 4)

- Weeks 0-4 done; PRs #1-#18 merged; decisions D1-D42. Week 4 gate criteria met (user review pending).
- Strongest baseline on val: ItemKNN (`results/v0/val/table_main.csv`).
- Open v1 items (D39): slice-aware sizing, `kcore.item` 5 vs 10, catalogue size (10-20K target), and a
  serving-catalogue proposal.
- `notebooks/make_notebooks.py` (the notebook generator) is now in the repo; `context.md` and CLAUDE.md
  §0.5 are updated. Next: Week 5 (LightGCN, projection hybrid and controls, MVP `recommend`, bundle v0).


## 2026-10-15: Week 4 gate approved; Week 5 part 1 (LightGCN)

- The user approved the Week 4 gate.
- Built `ml/models/lightgcn.py` (stage `models.lightgcn_val`), `ml/models/embedding_scorers.py`
  (`lightgcn_raw`, `lightgcn_norm`), and renamed the evaluation stage to `eval.val`, which now also
  evaluates LightGCN and compares every model with both LightGCN variants (D43). CI installs CPU
  torch; `02_eval_baselines.ipynb` runs `--until eval.val` on a GPU runtime.
- Tests (`tests/ml/models/test_lightgcn.py`): the sampler never returns a positive, normalized
  adjacency with zero rows for isolated nodes, the inner split, LightGCN beats random on training
  NDCG, seeded training and early stopping, -inf for zero-degree items in both scorers, artifact
  leakage checks; the tiny fixture runs through `eval.val`. `pytest -m "not slow"`: 280 passed,
  2 skipped (local raw samples absent in this worktree).
- Next: one Colab run (`run all --until eval.val`, GPU); then the Week 5 gate check (LightGCN vs
  ALS on val), the projection hybrid and its controls.

## 2026-10-16: LightGCN on Colab (Week 5, part 1 results)

- Colab T4 run of `run all --until eval.val` (main after PR #20): data stages and `nlp.content` up to
  date; `models.lightgcn_val` 234 s (best inner epoch 275); `eval.val` 23 s, 2,578 evaluated users,
  4,190 items.
- Committed `results/v0/val/` (now with `lightgcn_raw` and `lightgcn_norm`, paired bootstraps
  against both) and `results/v0/training/lightgcn_val.csv`.
- From those files: LightGCN raw beats ALS on NDCG@10 (paired 95% CI excludes zero; Recall@20 CI
  crosses zero) and ties ItemKNN; normalized LightGCN trades all-user accuracy for long-tail accuracy
  and coverage. Details and reasoning in D44.

### Gate (Week 5): status
- [x] LightGCN raw and normalized on val with strata and CIs; LightGCN >= ALS (NDCG@10)
- [ ] Projection hybrid, projection-shuffled and static-fusion controls on val
- [ ] MVP blend `recommend` on real artifacts; bundle export v0
- [ ] User reviews the gate

## 2026-10-16: Projection hybrid and controls built (Week 5, part 2)

- Built `ml/models/projection.py` (stage `models.projection_val`: projection + shuffled control),
  `StaticFusionScorer`, `pca` and `check_artifact` in `ml/models/embedding_scorers.py`; `eval.val`
  now also evaluates `projection`, `projection_shuffled` and `static_fusion_{0.5,0.7,0.9}` (D45).
- Tests (`tests/ml/models/test_projection.py`): same normalization in training and scoring, cold
  items represented by W c only, raw vs normalized items, seeding, loss decreases, cold items scored,
  static fusion (alpha 1 ranks like normalized LightGCN; cold items scored by content; cutoff check);
  the tiny fixture runs through `eval.val` with every model. `pytest -m "not slow"`: 284 passed,
  2 skipped (local raw samples absent).
- Next: one Colab run (`run all --until eval.val`, T4); then the Week 5 projection rows.

## 2026-10-16: Projection results (Week 5, part 2)

- Colab T4 run (main after PR #23): `models.projection_val` 75 s, `eval.val` 35 s; LightGCN reran with
  identical metrics. Committed `results/v0/val/` (projection, shuffled control, static fusion rows) and
  the refreshed `results/v0/training/lightgcn_val.csv`.
- From those files: the projection does not differ from its shuffled control, and is below raw
  LightGCN on all users; static fusion (alpha 0.7) is the strongest fusion row; no fusion recovers
  cold items. Reported as a null result in D46.

### Gate (Week 5): status
- [x] Val results with strata and CIs for random, popularity, ItemKNN, ALS, content-only, LightGCN raw
  and normalized, projection, projection-shuffled, static fusion
- [x] LightGCN >= ALS on val (NDCG@10, D44)
- [ ] MVP blend `recommend` on real artifacts; bundle export v0; recommendations visible in the browser
- [ ] User reviews the gate

## 2026-10-17: LightGCN training budget, sweep stage, paper-style projection (D47)

- Built: `models.lightgcn` config (batch 2048, max_epochs 1500, reg 1e-4); stage
  `models.lightgcn_sweep` with `ml/models/lightgcn_sweep.py` (grid on the inner split ->
  `results/v0/training/lightgcn_sweep.csv`, registered last); projection `normalize_items: false`,
  `w_init: zero`, `w_reg: 1e-4` with backward-compatible defaults; `embedding_health()`,
  `user_norm` and `w_frobenius` in the LightGCN and projection `meta.json`.
- Tests added: zero init reproduces raw LightGCN at epoch 0 and `w_reg` shrinks W; `embedding_health`
  anisotropy flag, cold-item exclusion and exact pairwise mean; sweep grid order.
- Verification: no working Python on the local machine this session, so ruff and
  `tests/ml/models` + `tests/ml/common/test_stage.py` ran in a `python:3.12-slim` container with CPU
  torch 2.14.1 (the full `-m "not slow"` suite needs the ML requirements and was not run here).
- Next (Colab, T4): `run all --until eval.val` (LightGCN, projection and eval.val rerun because
  their config changed), then `run models.lightgcn_sweep`; download `results_v0.zip`, commit, read
  LightGCN raw vs ItemKNN and projection vs shuffled control, log in D48.

## 2026-10-17 (later): D47 results read; dim 128; temporal inner split; projection dropout (D48)

- Colab T4 (user's own account, v0 rebuilt from scratch, byte-identical): three `eval.val` passes
  (dim 64 reg 1e-4; dim 128 reg 1e-4) and the 6-point sweep. Raw LightGCN 0.0203 to 0.0209 vs
  0.0216 before (inside the CI); norm and static fusion up; projection still equals its shuffled
  control with BPR 0.006. Full table and readings in D48.
- Built: `models.lightgcn.dim: 128`; `inner_split: temporal` in `lightgcn.py` (`inner_split`,
  `inner_eval_data` take a `mode`, default `user`), early-stopping metric ndcg@10; projection
  `gcn_dropout: 0.3` (seeded per-step mask on e_gcn, training only); sweep regs widened. Tests
  for the temporal split and the dropout path.
- Verification: ruff and `pytest -m "not slow"` in the `python:3.12-slim` container with CPU torch
  (no local Python this session); see the result line below.
- Next (Colab, one run): `run all --until eval.val`, read raw vs ItemKNN and projection vs
  shuffled; log D49 as the final v0 LightGCN/projection numbers; then `recommend` + `serve.export`.

## 2026-10-17 (evening): D48 run read, results committed, tuning closed (D49)

- Colab T4 run of the D48 config (data stages reran once because of an incomplete Drive flush, see
  D49). Committed `results/v0/val/*` and `results/v0/training/lightgcn_val.csv` from
  `results_v0_d48.zip`; `canonical/`, `eda/`, `content/` outputs were byte-identical and left as is.
- Temporal inner split: early stop at epoch 80 (inner NDCG@10 0.0250), val raw 0.0212, ties ItemKNN
  (paired CI includes 0). Projection with dropout: training BPR 0.235 vs shuffled 0.309, W larger,
  but val ranking unchanged and cold-item NDCG@10 = 0. Content-only remains the only cold-item model.
- D49 closes v0 LightGCN/projection tuning with the final table. Phase 3 review NLP is being built
  in parallel on branch `nlp-phase3` (D50 to D55 there); it merges after its Colab run.
- Next: Week 5 gate items, `recommend` on the real artifacts and `serve.export` (s12, s12.7).

### Gate (Week 5): status
- [x] Val results with strata and CIs for every model incl. projection, shuffled control and fusion
- [x] LightGCN >= ALS on val (NDCG@10): 0.0212 vs 0.0172, paired CI above zero (D49)
- [ ] MVP blend `recommend` on real artifacts; bundle export v0; recommendations visible in the browser
- [ ] User reviews the gate
## 2026-10-17: Phase 3 started, review NLP (branch `nlp-phase3`, parallel session; D50-D55)

Built in a second session alongside the Week 5 work, on the fork `zee10524/LightGcn`, branch
`nlp-phase3` (base `main` at 0509124). Nothing under `ml/models/`, `ml/eval/` or the data stages
was touched; `ml/nlp/aspect_seeds.py` stays byte-identical because `data.clean` imports it (D50).

### Built
- **`nlp.sentences`** (`ml/nlp/sentences.py`, s11.1): spaCy sentencizer over `text_ok` reviews,
  4-80 token filter, `sentences.parquet` (`sent_id` = row index, `pos`/`n_in_review` over the whole
  review) and `sent_emb.f16.npy` (MiniLM via `embed.load_encoder`, float16), checkpointed in parts
  with `ml/nlp/parts.py`.
- **`nlp.aspects`** (`ml/nlp/aspects.py`, s11.2-11.3): `AspectDetector` (max prototype cosine >=
  tau or lexicon), `sentence_aspects.parquet` (uint16 `aspect_mask`, `lexicon_mask`, `top_sim`,
  three descriptor scores), `artifacts/<v>/nlp/aspect_prototypes.npy` (12, 8, 384) and
  `descriptor_poles.npy` (3, 2, 384), `results/<v>/nlp/aspect_checks.json`.
- **`nlp.sentiment`** (`ml/nlp/sentiment.py`, s11.4): DistilBERT-SST2 on aspect-bearing sentences,
  clause splitting with re-detection, `aspect_polarity.parquet` (one row per sentence x aspect),
  `results/<v>/nlp/sentiment_checks.json`.
- **`nlp.centroids_val`, `nlp.centroids_serve`** (`ml/nlp/centroids.py`, s11.6): item and user
  review centroids per cutoff in `artifacts/<v>/{eval_val,serve}/review_centroids/` with
  `meta.json`; `build_centroids` raises `LeakageError` on rows at or after the cutoff.
- **`nlp.annotation_export`** (`ml/nlp/annotate.py`, s11.5 step 1) and
  `docs/annotation_guides/aspects.md` (step 2).
- Shared: `ml/nlp/cutoffs.py` (artifact-set cutoffs, leakage check), `ml/nlp/parts.py`
  (`PartStore`). Config: `nlp.embedding_dim`, `nlp.chunk_rows`, `nlp.sentiment_max_length`,
  `aspects.annotation_sample`. Stages registered after `eval.val` and before the sweep.

### Tests run
- No Python on this laptop this session: ruff and pytest ran in a `python:3.12-slim` container
  with CPU torch and the ML requirements (image built from the two requirements files).
- `ruff check .` and `ruff format --check .`: clean.
- `pytest -m "not slow"`: see the result line at the end of this section. The only failure is
  `tests/ml/common/test_manifest.py::test_git_sha`, which needs a git binary the container lacks
  (environment, not code). New tests: `tests/ml/nlp/` (sentences, parts, cutoffs, aspects,
  sentiment, centroids, annotation export, and the Phase 3 stages end to end on the tiny fixture
  with stubbed MiniLM, spaCy and DistilBERT).
- Result line (container, `pytest -m "not slow" -q`): `1 failed, 311 passed, 2 skipped,
  2 deselected in 104.92s`; the skips are the local raw samples, the failure is `test_git_sha`.
- `pytest -m slow tests/ml/nlp` with the real models on CPU: 2 passed in 190 s. "The sentences
  themselves are just gorgeous to read." hits `writing_style` by prototype similarity (above
  `tau_default` 0.45, not only by lexicon); "Wonderful characters, but the middle third really
  dragged." gives characters > 0 and pacing < 0 through the clause rule with DistilBERT.

### Not done (next, in order)
- Colab run of the five stages plus the annotation export on v0 (one cell handed to the user);
  then commit `results/v0/nlp/*.json` and copy `aspects_300.csv` to `docs/annotation/`.
- `nlp.validate_aspects` steps 4-5 (tau tuning, P/R/F1, kappa) once the labelled CSV exists.
- Spoiler classifier (s11.7; streams `goodreads_reviews_spoiler.json.gz` as a `url:` input),
  `nlp.onnx` + `litsense_core/encoder.py` + parity test (s10.3-10.4), profiles (s11.8), snippets
  (s11.9), `eval_test` centroids (Week 13).
- Open for the user: lexicon changes after validation mean editing `aspect_seeds.py`, which
  reruns `data.clean` (an hour of streaming), or moving the lexicon to a new module first (D50).

### Gate (Week 6): status
- [x] `nlp.sentences`, `nlp.aspects`, `nlp.sentiment` built and tested on the fixture
- [x] Annotation CSV stage and guide written
- [ ] Full run on Colab; annotation CSV delivered to the labellers
- [ ] `nlp.onnx` + runtime encoder + parity tests
- [ ] User reviewed the gate

## 2026-10-17 (night): Week 5 gate build, `recommend` on real artifacts and bundle export v0 (D56)

- Built: `ml/serve/export.py` with stages `models.lightgcn_serve` and `serve.export` (registered
  after `eval.val`); `litsense_core/{store,vector_index,profile,candidates,rank,reasons,recommend}.py`
  implementing every s15.1 function over the bundle, with the documented degradations;
  `dev/dev_server.py`; `tests/litsense_core/test_bundle.py` (pipeline through `serve.export` on the
  tiny fixture, then the real package end to end).
- Verification: ruff clean; `pytest -m "not slow"` in the `python:3.12-slim` container with CPU torch
  (see the result line in the commit message).
- Next (Colab, one cell): `run all --until serve.export` (data stages skipped, `lightgcn_serve` and
  `serve.export` run), then `health()` and `recommend()` against the real bundle in the same cell;
  commit `results/v0/` if anything changed; hand `serve_bundle.tar.gz` (Drive) and the dev server to
  the teammate; recommendations visible in the browser closes the Week 5 gate.

### Gate (Week 5): status
- [x] Val results with strata and CIs for every model incl. projection, shuffled control and fusion
- [x] LightGCN >= ALS on val (NDCG@10): 0.0212 vs 0.0172, paired CI above zero (D49)
- [x] MVP blend `recommend` on real artifacts; bundle export v0 (Colab run 2026-10-17: bundle v0-20261017, 6.4 MB, RSS 155 MB, demo reader with 292 corpus ratings gets five cf-sourced thrillers in 9 ms; cold reader gets trending)
- [ ] Recommendations visible in the browser (teammate swaps the mock for `load_store(bundle)`)
- [ ] User reviews the gate

## 2026-10-18: Phase 3, second block (branch `nlp-phase3b`; D57-D61)

### Built
- **`nlp.profiles_val`, `nlp.profiles_serve`** (`ml/nlp/profiles_offline.py`, s11.8): item aspect
  profiles (shrunk polarity, mention/pos/neg rates, descriptors, aspect means), corpus-user explicit
  vectors + confidence, `user_review_aspects.parquet`, `aspect_topitems.json` (serve).
- **`litsense_core/aspects.py`**: the shared numpy math (`shrink`, `decay`, implicit/explicit
  vectors, `combine`); `profile.py` not edited, wiring described in D58.
- **`nlp.spoiler_train`, `nlp.spoiler_flags`** (`ml/nlp/spoiler_train.py`, s11.7) and
  **`litsense_core/spoiler.py`**: opt-in Goodreads training (`spoiler.train_on_goodreads`), regex-only
  model by default, `sentence_spoiler.parquet` for every Amazon sentence.
- **`nlp.snippets`** (`ml/nlp/snippets.py`, s11.9): `evidence_snippets.json.gz`, spoiler-flagged
  sentences excluded.
- **`nlp.onnx`** (`ml/nlp/onnx_export.py`, s10.3) and **`litsense_core/encoder.py`** (s10.4):
  fp32 + int8 ONNX, `tokenizer.json`, parity and latency report.
- Config: `aspects.explicit_conf_k`, `aspects.topitems`, `aspects.snippets`,
  `spoiler.train_on_goodreads`, `spoiler.late_fraction`, `nlp.onnx_parity`; `onnx` in the CI
  package list.

### Tests run (Docker `python:3.12-slim`, CPU torch; no local Python)
- New: `tests/ml/nlp/test_profiles.py`, `test_snippets.py`, `test_spoiler.py`,
  `test_spoiler_pipeline.py` (training on the fixture's Goodreads spoiler file),
  `tests/litsense_core/test_encoder.py` (hand-built ONNX model), `test_onnx_slow.py` (real export,
  slow); `test_pipeline.py` now runs every Phase 3 stage end to end on the tiny fixture.
- Result line (container, `pytest -m "not slow" -q`, before the final ONNX fix): `1 failed, 327
  passed, 2 skipped, 3 deselected in 206.54s`; the failure is `test_git_sha` (no git binary), the
  skips the local raw samples. The slow ONNX test with the real MiniLM passed afterwards: fp32
  parity exact, per-channel int8 within the s10.3 thresholds (D61).

### Not done
- Real runs on Colab (one cell below in the session report); the Goodreads spoiler training waits
  for the user's yes (`spoiler.train_on_goodreads: true`).
- `nlp.validate_aspects` steps 4-5 (needs the labelled CSV); `eval_test` profiles and centroids
  (Week 13); wiring of `aspect_vec`, snippets, spoiler and encoder into `litsense_core` (main
  session, D58); `serve.export` file list (main session).
## 2026-10-18: Phase 3 wired into the bundle and the serving package (D63)

- Merged `nlp-phase3b` (D57 to D62) into main. `serve.export` now ships the aspect profiles,
  descriptors, aspect top-items, evidence snippets, spoiler model, review centroids, prototypes and
  the fp32 ONNX encoder; `litsense_core` uses them: aspect side of the profile, the `aspect`
  candidate source, `aspect_match` in the blend, template 2 with safe snippets, semantic search,
  book insights, review analysis, onboarding text.
- Verification: ruff clean; `tests/litsense_core` 59 passed and the full `-m "not slow"` suite in the
  `python:3.12-slim` container (result line in the commit message).
- Next (Colab, one cell): `run all --until serve.export` (sentiment, profiles, snippets rerun after
  D62, then the export), then load the bundle and print `health()`, a demo reader's picks with
  reasons, `book_insights` and `analyze_user_review` on one sentence. Then the labelled CSV ->
  `nlp.validate_aspects` closes Phase 3's gate items.- 2026-10-18 (late): first Phase 3 bundle on Colab (v0-20261018, 93.7 MB). RSS 394 MB > 350 budget
  and demo readers without aspect reasons; both fixed in D64 (lazy snippets file, ORT arena off,
  corpus readers' offline aspect vectors in the bundle). Next Colab run re-measures RSS.- 2026-10-18 (night): D65 measured. RSS 229 MB after load, 234 MB after 30 recommends (fresh
  process; budget 350 met). Demo readers get centered aspect reasons (pacing, dialogue, characters
  for thriller readers). Phase 3 serving integration is done; what remains for the Phase 3 gate is
  the labelled CSV -> `nlp.validate_aspects` (tau per aspect, P/R/F1 table) and, optionally, the
  Goodreads-trained spoiler model (`spoiler.train_on_goodreads: true`, 1.3 GB stream, user's call).
- 2026-10-19: CI on main had been red since the nlp-phase3b merge. Cause: `tests/ml/nlp/test_spoiler.py::test_training_helpers` imported scikit-learn through `holdout_metrics` (its AUC), and `make install-ci` installs only the CI subset (no sklearn). Fix: `roc_auc` computed from scipy ranks (same value as sklearn, ties at half credit); sklearn stays confined to `fit_classifier`, which only the opt-in Goodreads training runs. Verified in Docker under the CI subset: 325 passed before the fix with that one failure, spoiler tests pass after.

## 2026-10-19: Phase 5 (Week 9/10). Val bundle, features, LambdaRank ranker, val ablation (D66)

### Built
- `serve.export_val` (val bundle at t1), `litsense_core/features.py` (34 columns of s13.3),
  `ranking.build_training` (training table + candidate recall), `ranking.train` (LambdaRank, importances,
  ranker vs blend on held-out users), `ranking.ablation_val` (M4..M8b with CIs), ranker shipped in the
  serve bundle and used by `recommend` (`degraded` only when it is missing or fails).
- `nlp.profiles_val` writes `aspect_topitems.json`; `corpus_users.npz` carries review centroids.

### Tests run
- Docker `python:3.12-slim`, full ML pins: ruff clean; `tests/litsense_core`, `tests/ml/nlp/test_pipeline.py`,
  `test_profiles.py`, `tests/ml/common/test_stage.py`, `tests/ml/eval` (result line in the commit message).
  The bundle test now runs the whole pipeline through `ranking.ablation_val` on the tiny fixture and checks the
  val bundle, the training table, the ranker artifacts and the ranker path in `recommend`.

### Gate checklist (Week 9)
- [x] Candidate sources (7) + recall analysis stage, `features.py`, training data, LambdaRank training
- [ ] Run on Colab (`run all --until ranking.ablation_val`), commit `results/v0/ranker/`
- [ ] Ranker val NDCG@10 vs blend reported; user decides on any zero-contribution source
- [x] Tests for new modules pass; CI green
- [x] Decisions logged (D66)
- [ ] User reviewed the gate
### Measured on Colab (2026-10-19, CPU runtime, D66 run; `results/v0/ranker/`)
- Candidate recall (2,578 val users): union 0.288, after the s13.2 filters 0.168; per source cf 0.121,
  semantic 0.106, item_item 0.077, author 0.069, popularity 0.058, longtail 0.056, aspect 0.010; unique
  contribution: cf 0.069, popularity 0.033, semantic 0.027, author 0.023, item_item 0.015, aspect 0.003,
  longtail 0.00003. 724 users kept (a positive among the candidates), 1,854 dropped.
- Held-out users (133): ranker NDCG@10 0.140 [0.105, 0.177], blend 0.032 [0.014, 0.052], cf-score-only
  0.051; ranker minus blend +0.108 [0.071, 0.147] (paired). Gate passed on this protocol. These numbers are
  conditional on the candidates holding a positive; a rough all-user figure is 0.140 x 724 / 2,578 = 0.039
  (dropped users at 0), to be measured properly through the evaluator (Week 13 protocol).
- Ablation (133 users): M4 0.1355, M5 0.1181, M6 0.1339, M7 0.1277, M8 0.1399, M8a 0.1277, M8b 0.1214.
  M6 minus M4: NDCG@10 -0.002 [-0.037, 0.030], recall@20 +0.045 [0.003, 0.093]. M8 minus M4: +0.004
  [-0.030, 0.034]. Every ranker row beats the blend with the CI above 0. Aspects move recall@20, not NDCG,
  on a small held-out set.
- Importances (gain): item_age_years 12%, cf_rank_pct 11%, author_affinity 8%, cf_norm_score 7%,
  cf_score 5%; ASPECT group in the top 12 (descriptor_match, aspect_match, aspect_conflict). Author and
  aspect both noted for the report, as s13.5 asks.
- Serving (fresh process, Colab): `health()["ranker"]` true, `recommend` 63.7 ms with the ranker vs 30.5 ms
  blend (budget 80), RSS 358.8 MB after load (budget 350; D65 was 229). Measured in Docker: importing
  lightgbm costs 30 MB alone but 125 MB when pandas and scikit-learn are installed (it imports both when
  present, which Colab has and the backend image does not). Expected backend RSS about 260 MB; to be
  confirmed on the backend.
- Open: the filters drop 42% of the positives the sources found (suspect `min_reviews_main_feed: 3`), the
  aspect and longtail sources contribute almost nothing unique (s13.1 says ask the user before dropping),
  the LightGCN serve stage retrained because it shares a module with the export (split it out), and
  `features.assemble` has two Python loops worth vectorizing for the latency budget.- **Added (2026-10-19, later):** stage `ranking.eval_val`: the ranker (exact and pseudo CF) and the blend on
  every val user under the s8 protocol (candidates scored, every other item below them in popularity
  order), through the same evaluator and tables as `eval.val`, with paired bootstraps against popularity
  and both LightGCN variants from their stored per-user rows. Outputs `results/<v>/ranker/eval_val_*`.
  Not part of `run all --until serve.export`; the Colab cell runs it by name.- **Superseded (D67, 2026-10-19):** the ranker numbers above were measured with two protocol leaks
  (training users scored by their own ranker; val-window releases in the candidates). The rerun with
  cross-fitted scoring and the future-book filter replaces them; see the next entry once measured.
### Measured after D67 (2026-10-19, Colab CPU, cross-fitted; `results/v0/ranker/`, the numbers that count)
| model | NDCG@10 [95% CI] | recall@20 | hit@10 | long-tail NDCG@10 | cold-item NDCG@10 |
|---|---|---|---|---|---|
| ranker (M8, what serves) | 0.0296 [0.0253, 0.0341] | 0.0738 | 0.096 | 0.0208 | 0.0136 |
| ranker, pseudo-CF (new reader) | 0.0282 [0.0241, 0.0325] | 0.0756 | 0.095 | 0.0170 | 0.0111 |
| itemknn (eval.val) | 0.0227 | 0.0541 | | 0.0123 | 0 |
| lightgcn_raw (eval.val) | 0.0212 | 0.0553 | 0.068 | 0.0057 | 0 |
| content (eval.val) | 0.0191 | 0.0463 | | 0.0203 | 0.0230 |
| blend (served before D66) | 0.0115 [0.0090, 0.0141] | 0.0332 | 0.044 | 0.0048 | 0.0007 |

- Paired vs lightgcn_raw on all 2,578 users: ranker +0.0084 NDCG@10 [0.0034, 0.0134], +0.0185 recall@20
  [0.0092, 0.0275]; vs lightgcn_norm +0.0128 [0.0079, 0.0177]; vs blend +0.0182 [0.0136, 0.0225]. The first
  hybrid to beat LightGCN on val with the CI above 0 (PRD G1, by the ranker route; the projection never did).
  The blend was worse than LightGCN alone, so shipping the ranker fixes a regression as well.
- Per sparsity bucket the ranker is 2 to 3x the blend everywhere; best at 11 to 20 fit interactions (0.049).
  Cold items: content-only (0.0230) still beats the ranker (0.0136); long tail: ranker 0.0208 vs content 0.0203.
- Held-out protocol (128 users with a positive among the candidates): ranker 0.1357 vs blend 0.0276, CI above 0.
- Ablation (same 128 users): M4 0.1443, M5 0.1219, M6 0.1380, M7 0.1425, M8 0.1357, M8a 0.1425, M8b 0.1359.
  M6 minus M4 NDCG@10 -0.006 [-0.038, 0.023]; M8 minus M4 -0.009 [-0.045, 0.030]. Every ranker row beats the
  blend (CI above 0); no feature group beyond M4 moves NDCG on this set. The aspect features do not improve
  ranking on v0; their value is the explanations. 128 users give wide intervals: a small gain cannot be excluded,
  but none can be claimed.
- Importances: cf_rank_pct 9.5%, item_age_years 8.4% (same-year recency, available in serving), author_affinity
  6.6% (third: note for the report), item_count_log, cf_norm_score, cf_score, review_centroid_sim,
  descriptor_match, aspect_match, semantic_recent_sim.
- Candidate recall unchanged (sources run before the filter): union 0.288, after filters 0.159 (the future-book
  filter took 0.009). Open: `min_reviews_main_feed: 3` cuts most of the gap; aspect and longtail sources add
  0.0035 and 0.0000 unique recall (user decides, s13.1).- **Filter diagnostic (2026-10-19, `results/v0/ranker/filter_diagnostic.*`):** of the 1,806 val positives the
  sources found, the filters keep 928: series guard drops 422 (23%), the review threshold 290 (16%), the
  next-in-series rule 124 (7%, those go to the `continue_series` rail, which the evaluator does not count), the
  D67 year rule 42. Recall after filters: 0.1686 at `min_reviews_main_feed` 0, 0.1589 at 1, 0.1574 at 3 (current),
  0.1565 at 5. 2,600 of the 6,196 val positives had no text review at t1 (new or cold books), so only 0 moves
  the number and those items carry no aspect bars or snippets. Decision: keep 3 (training = serving). The real
  cost is the series handling (30% of found positives); next: split the guard losses by whether the reader is
  in the series at all, and report a home-screen row (rail + feed) next to the main-feed row.- **Series rules and the home-screen row (2026-10-19, `results/v0/ranker/`):** the 422 series-guard losses split
  into 251 where the reader has no volume of the series in their history (the guard doing its job on a
  catalogue that lost the earlier volumes) and 171 where the reader is in the series but skipped a volume
  (9% of the found positives). Decision: keep the guard as the brief defines it; report the number. New row
  `ranker_home` = `continue_series` rail (the exact next volumes the feed rule sets aside) above the ranked
  feed, what the reader sees on the home screen: NDCG@10 0.0466 [0.0410, 0.0525], hit@10 0.134, vs 0.0296 for
  the feed alone and 0.0212 for LightGCN (paired +0.0254 [0.0192, 0.0312]). The report carries both rows: the
  feed alone measures the model, the home screen measures the product. Phase 5 on v0 closes here; open items
  are vectorizing `features.assemble` (63.7 ms) and the backend RSS check.- **Built (2026-10-19, late): `nlp.validate_aspects`** (s11.5 steps 3 to 5). Reads the labelled
  `aspects_300.csv` from the raw dir (`raw:` input, so label changes rerun it), recomputes per-aspect
  prototype similarities and lexicon hits for those sentences from the stored embeddings, tunes tau per
  aspect on a seeded half (best F1, ties to the larger threshold), reports P/R/F1 per aspect and macro on
  the other half at the tuned tau, the current tau and lexicon-only, and Cohen's kappa per aspect against
  `aspects_300_second.csv` when present. Writes `results/<v>/nlp/aspect_validation.{json,csv}` with the
  proposed `aspects.tau` block (user confirms). Config: `aspects.validation`. Guide process section updated.
  Verified in Docker: ruff clean; unit tests + the NLP pipeline test with a synthetic labelled file, 8 passed.
  Waiting on the labelled CSV (Zeeshan, 300 rows; second labeller, first 100 rows).
## 2026-10-19: Phase 6 GenAI layer (branch `genai-phase6`, D68 to D71)

### Built
- `litsense_core/llm/`: config, sqlite cache + daily budget, rpm limiter, `LLMClient` (`text`, `json`, retries,
  repair retry, privacy guard, events), adapters `gemini` (google-genai), `openai_compat` (stub), `fake`;
  the five v1 prompt files of s14.3.
- `litsense_core/validators.py`, `explain.py` (LLM explanation with template fallback; the feed never calls the LLM),
  `entities.py` (rapidfuzz resolvers; public `resolve_entities` unchanged), `onboarding.py` (LLM parse with
  vocabularies enforced; keyword parse as fallback), `chat.py` (LLM intent parse with the rule parser as
  fallback, anchors, session updates, deltas and filters on the shared ranking path, validated LLM reply,
  chips, clarifications). `health()["llm"]`, `litsense_core.configure_llm(config, on_event)`.
- `ml/genai/synthesis.py` (stage `genai.synthesis`, before `serve.export`; empty when the LLM is off),
  `serve.export` fills `catalog_text.review_summary` and ships the `llm` config slice; `book_insights` serves it.
- `ml/genai/eval_intent.py` (gold-set format, scorer, 70 candidate messages), `ml/genai/audit.py` (export,
  score with Fleiss' kappa, rejection rates), `docs/annotation/intent_example.csv`.
- `dev/dev_server.py`: `/onboarding/parse`, `/insights/{id}`, `/review/analyze`,
  `/entities`, `/author/{id}`; `LITSENSE_LOG_LLM=1` prints LLM events.
- `ml/config.yaml`: `genai.synthesis` block. `ml/stages.py`: `genai.synthesis` registered; `nlp.validate_aspects`
  moved last (it needs the labelled CSV, so `run all --until ...` must not reach it; D68).

### Tests run (Docker `python:3.12-slim`, full ML pins, fake provider only, no network)
- New: `tests/litsense_core/test_llm_client.py`, `test_explain.py`, `test_entities.py`, `test_onboarding.py`,
  `test_chat.py` (three-turn demo with the LLM off and on), `tests/litsense_core/fake_store.py` (a hand-built
  12-book store), `tests/ml/genai/test_synthesis.py`, `test_eval.py`; `test_bundle.py` now also runs
  `genai.synthesis` with the fake provider, re-exports, and checks summaries, `health()["llm"]` and the LLM
  paths on the real package. Result lines are in the commit messages and the session report.

### Real Gemini smoke test (2026-10-19, later; D72)
- Model ids set: `gemini-3.5-flash-lite` for both roles (Flash failed JSON validation, flash-latest is
  quota-limited on the free tier); `timeout_s` 10 (API minimum). On the fake 12-book store with the real
  key: `health()["llm"]` true; `explain`, `parse_onboarding` and the three-turn chat demo all `method="llm"`,
  zero validator rejections, cache hit on the repeated explain; latencies 1.2 s to 2.9 s per call.
- The key stays in the environment only (`GEMINI_API_KEY`); the user rotates it after this session.

### Colab synthesis run and the Phase 6 bundle (2026-10-20, D72)
- `genai.synthesis` over five sessions (free tier, 500 requests per project per day, three projects):
  1,000 of 1,000 books, 990 LLM summaries and about 10 templates after the validator fixes (the first
  114 had 50 rejections from the two false positives; those rows were regenerated). Rate 7 per
  minute; the per-day stop and resume worked as designed.
- `serve.export`: bundle `v0-20261020`, 30 files, 94.0 MB tar, `review_summaries: 1000`; on Drive as
  `MyDrive/litsense/serve_bundle_v0_phase6.zip` and `artifacts/v0/serve_bundle/`. The ranker chain
  reran (code fingerprint) with the same data and seed.
- Checked in a fresh process on Colab with the key set: `health()` ranker true, llm true, RSS 360 MB
  (Colab image; backend expected lower), `recommend` 35 ms, `degraded` false, `ranked_by` ranker.
  `explain` for a demo reader: `method="llm"`, no rejections, quotes the evidence snippet;
  `book_insights().review_summary` filled; `chat_turn("something like <title> but shorter")`:
  `similar_to_with_modifier`, LLM reply, five books under the page limit. The in-kernel check showed
  `ranker: False` only because `make install-colab` swapped numpy under a kernel that had already
  imported it (lightgbm failed to import there); the pipeline and the fresh-process check are right.

### Intent gold set scored (2026-10-20, D73; `results/v0/genai/intent_eval.json`)
Labelled by hand through the plain sheet (69 rows kept of 70). Scored on the real v0 bundle
(second run; the first run's LLM column was void because 55 rows fell back to the rules when the
non-sleeping online limiter ran dry, so the scorer now uses a blocking limiter at 7 per minute):

| parser | intent | anchor | filters | aspect signs | descriptor signs | length | exclude | target | field accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Gemini 3.5 Flash-Lite (62 rows by the LLM, 7 by the rule fallback) | 0.971 | 0.884 | 0.928 | 0.855 | 0.971 | 0.957 | 1.0 | 0.986 | 0.944 |
| rule parser (in-sample, D73) | 0.986 | 0.899 | 0.928 | 0.928 | 1.0 | 0.971 | 1.0 | 1.0 | 0.964 |

Target of s14.9 (field accuracy >= 0.8) met. The LLM's two intent misses: c47 "published before
1980" (recommend_general) and c69 "exclude science fiction" (refine_previous), both gold
filter_by_attribute. Seven rows (c04, c09, c10, c12, c18, c29, c44) got no valid LLM answer in
that run (transient provider failures, no pattern) and were scored through the rule fallback, as
the product would do. The rule parser's number is in-sample and not a generalisation estimate.

### Validator rejection rates on real explanations (2026-10-20; `results/v0/genai/rejection_rates.json`)
200 LLM explanations for the bundle's demo readers (real Gemini, blocking limiter at 7 rpm): 198
answered by the LLM, 2 rejected (1.0%): one foreign title, one quote not in the snippet; zero
spoilers, zero numbers outside the evidence, no fallbacks. The audit export
(`results/v0/genai/explain_audit.csv`) holds 100 rows, 95 LLM and 5 template, for the three raters
when the viva needs the human grounding numbers.

### Waiting on the user
- Three raters for the explanation audit (optional for the website; needed for the report). The
  first project's key was pasted in chat and must be rotated.
- The intent gold set (`docs/annotation/intent_50.csv`, from the `--draft` candidates), then
  `python -m ml.genai.eval_intent`; three raters for `docs/annotation/explain_audit.csv` after `audit export`.
- Follow-ups: `named_authors` into the profile's favourite authors; `openai_compat` adapter if a local model
  is wanted; the Week 12 three-turn demo on the real bundle with the key set.

### Aspect validation, first run (2026-10-20, Colab, labelled by Zeeshan, no second labeller yet)
- 300 sentences, seeded 150/150 split. Macro F1 on the held-out half: 0.391 at the tuned tau, 0.294 at the
  current tau 0.45, 0.293 lexicon only. Per aspect (F1 at tuned tau): characters 0.64, plot 0.62, humor 0.67
  (2 test positives), emotional_impact 0.42, writing_style 0.41, world_building 0.37, dialogue 0.38,
  ending 0.30, pacing 0.29, complexity 0.28, editing_quality 0.21 (3 positives), tone_dark 0.12.
  Seven aspects flagged below 0.4. Target 0.6 not met.
- Not adopted yet: rows 104 to 200 of the labelled file carry 3.7 labels per sentence against 1.0 elsewhere,
  with labels the guide would leave blank ("I love reading books by Lisa Jackson" -> emotional_impact),
  which inflates the positives of emotional_impact (83), complexity (44) and world_building (50) and pulls
  recall down. The labeller re-checks that block, then the stage reruns; tau will be adopted only for
  aspects with 20+ labelled positives, the rest stay at 0.45. Proposed tau from this run, for the record:
  plot 0.31, writing_style 0.30, characters 0.52, world_building 0.36, pacing 0.45, editing_quality 0.47,
  dialogue 0.45, humor 0.61, emotional_impact 0.33, ending 0.80, complexity 0.35, tone_dark 0.44.
## 2026-10-20: HTTP model service (branch `serve-api`, D75 to D81)

### Built
- `service/`: FastAPI app that serves every contract function under `/v1` (one route each, bodies of
  the function arguments, responses = `litsense_core.schemas`), `GET /v1/catalog` and
  `/v1/catalog/{id}` for seeding the backend's books (contract section 7 item 1), `/health`, `/ready`,
  `/metrics`. Background boot (download into a staging folder, `configure_llm` with an event hook,
  `load_store`, warm-up), `X-API-Key` auth, request ids, JSON logs without user content, token-bucket
  rate limit, body limit, gzip, one error envelope (core `ValueError` is 400), LLM hard timeout with
  the template fallback, ranking slots, TTL caches for explain and insights.
- `service/openapi.json` (stale-checked by a test), `service/README.md`, `service/loadtest.py`,
  `service/Dockerfile` (slim, libgomp1, non-root, `/ready` health check), `service/docker-compose.yml`
  (model, bundle host, Node stub), `render.yaml`, `service/.env.example`, `requirements-service.txt`,
  Makefile targets `service`, `service-test`, `service-openapi`, `service-image`, `install-service`;
  `make install-ci` installs the service pins; CI gains a `service-image` job.
- D75: the double-encoded star in `similar_books` reasons fixed (the only core change).
- Tests: `tests/litsense_core/bundle_build.py` holds the tiny-bundle builder (moved from the
  `test_bundle.py` fixture, unchanged) so `tests/service` can build it once per session.

### Tests run (Docker `python:3.12-slim`)
- Full pins plus service pins: ruff check and ruff format --check clean; `pytest -m "not slow"`
  419 passed, 2 skipped (raw samples absent), 4 deselected (`test_git_sha`, slow); `tests/service`
  35 passed.
- Exactly the CI install (`make install-ci` on `python:3.12-slim`): before this branch's fix both
  `test_bundle.py` tests errored there (lightgbm's `LGBMRanker` needs scikit-learn, missing from
  the CI subset since D66; reproduced on main 084f1c0), so GitHub CI on main is most likely red.
  With `scikit-learn` added to `CI_ML_PACKAGES` (D81): ruff clean, 419 passed, 2 skipped,
  4 deselected. GitHub's own run could not be read from here (no `gh`, repo API not public).

### Measured (tiny bundle; numbers and method in D81)
| Setting | Ready after start | Recommend p50 / p95 | Explain (template) p50 / p95 | RSS |
| --- | --- | --- | --- | --- |
| In-process test client, warm, sequential | n/a | core p95 7.6 ms, HTTP p95 13.5 ms | n/a | n/a |
| Image, 1 CPU, 512 MB, one client | 3.4 s | 13.5 / 30.3 ms | 3.9 / 6.0 ms | 156 MB |
| Image, 1 CPU, 512 MB, four clients | 3.4 s | 72.6 / 108.4 ms | 25.2 / 41.3 ms | 156 MB |
| Image, 0.1 CPU, 512 MB (Render free), two clients | 48 s | 202 / 296 ms | 98 / 190 ms | 155 MB |

### Open
- Publish the bundle as the GitHub Release `bundle-v0-20261020` (contract section 1b), then deploy the
  Blueprint and read the real `download_s`, `load_s`, `rss_mb` and feed latency from `/health` and the
  load test. On the free plan's 0.1 CPU the feed will be several times the 80 ms budget (D81).
- The Node backend (sibling session) consumes `service/openapi.json`; it needs the service URL and
  `SERVICE_API_KEY`.
