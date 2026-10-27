# LitSense: session handoff (as of 2026-10-19, Zeeshan's fork)

Read this, then `CLAUDE.md` (the brief; §0.5 lists what overrides it), `docs/progress.md` and
`docs/decisions.md` (D1 to D71). Code: https://github.com/zee10524/LightGcn (owner `zee10524`),
a fork of `Aniketdhar810/LitSense` plus everything below. `main` head: see `git log` (bde7d78 = CI fix, then the D66 commit). Nothing has been
merged back into Aniket's repo yet (open question for Aniket).

## The user and how to work with them

- Zeeshan, B.Tech final year, beginner with ML. Writes Hinglish; prefers short, simple, crisp
  answers with tables, honest numbers, no jargon without a one-line explanation. Asks "kitna time
  lagega" often: give realistic minutes.
- They run every GPU job on free Colab (T4) with their own Google Drive (`MyDrive/litsense`, rebuilt
  from scratch on 2026-10-17, byte-identical to the frozen v0). Give ONE cell per run with comments,
  plus a markdown text cell describing it, and say exactly which output lines to paste back. They
  keep notebook `Untitled14.ipynb` (Colab Recents) as the run log.
- Same-kernel reloads: `import sys; [sys.modules.pop(m) for m in list(sys.modules) if
  m.startswith("litsense_core")]` before importing new code, else the old module is used. RSS must
  be measured in a fresh subprocess, never in a reused kernel (D65).
- Deliverables they asked for live in `C:\Users\zeeshan.ahmad\Desktop\7thSemProject\`:
  `LitSense_LightGCN_Explained.md`, `LitSense_Paper_vs_Implementation_Review.md`,
  `LitSense_Update_2026-10-17.{md,pdf}`, `LitSense_Update_for_Aniket_2026-10-17.pdf`,
  `Colab_Run_Instructions.md`, this file.

## Local environment (unusual, important)

- Windows laptop. This week `python.exe` and Git Bash `bash.exe` disappeared from disk (cause
  unknown, Defender history empty). Do not try to install them. The Bash tool fails; use PowerShell.
  A hook needs Git Bash and once blocked every tool for a while; it works again now.
- Git: `C:\Users\zeeshan.ahmad\AppData\Local\Programs\Git\cmd\git.exe` (full path). Always
  `git -c core.autocrlf=input add/commit` so LF is preserved (the repo is LF; the Edit/Write tools
  write CRLF on Windows). Remote `zee` = the fork; push `main` there.
- Python, ruff, pytest run in Docker: `python:3.12-slim`, repo mounted at `/w`, CPU torch from
  `https://download.pytorch.org/whl/cpu`, then `requirements-core.txt` + `requirements-ml.txt`
  without the torch line (mirrors `make install-colab`). Script: scratchpad `run_tests.sh`.
  Expect `tests/ml/common/test_manifest.py::test_git_sha` to fail in the container (no git); all
  else must pass. Last full run: 327 passed, 2 skipped (raw samples absent).
- Working clone used by this session:
  `%LOCALAPPDATA%\Temp\claude\C--Users-zeeshan-ahmad-Desktop-7thSemProject\7b12c06f-2ccc-442a-90c8-b295031ae892\scratchpad\LitSense`
  (scratchpad dirs can vanish between sessions; if it is gone, clone the fork fresh). A second
  clone `Desktop\7thSemProject\LitSense-nlp` belongs to the NLP session (branch nlp-phase3b, merged).
- Chrome headless makes PDFs: `chrome.exe --headless=new --print-to-pdf=... file:///...html`.
- Parallel session used for Phase 3: "LitSense Phase 3: review NLP with MiniLM (BERT half)"
  (id `local_60dee1dc-5478-4a61-a81e-8876ebc3769b`). It must never push `main`; it works on
  `nlp-phase3*` branches and the main session merges.

## Where the project stands (phases of CLAUDE.md §17)

- **Weeks 0 to 4:** done (data pipeline, frozen v0: 109,454 interactions, 10,987 users, 4,190
  books; eval harness; baselines; content embeddings).
- **Week 5 (Phase 2 + gate):** LightGCN and projection final (D47 to D49); real `recommend` and
  bundle export done and verified on Colab (D56); only "recommendations visible in the browser"
  (teammate) and the user's gate review remain.
- **Phase 3 (review NLP, the BERT half): about 85%.** All stages built and run on v0; everything
  is in the bundle and used by the serving package (D50 to D65). Remaining: the labelled
  `docs/annotation/aspects_300.csv` -> `nlp.validate_aspects` (per-aspect tau, P/R/F1; Week 7 gate
  item), optional Goodreads spoiler training (`spoiler.train_on_goodreads: true`, 1.3 GB stream,
  needs the user's yes), aspect EDA figures, `eval_test` sets (Week 13).
- **Phase 4:** the real `litsense_core` serves every §15.1 function from the bundle; degradations
  left: no LightGBM ranker (blend ranks, `degraded=True` when `use_ranker=True`), no LLM.
- **Phase 5 (ranker, ablation M4..M8): built and measured on v0 (D66, D67, 2026-10-19).** Cross-fitted
  full-catalogue val: ranker NDCG@10 0.0296 vs LightGCN 0.0212 (paired CI above 0), blend 0.0115; aspects
  (M6 vs M4) within CI; home-screen row (rail + feed) 0.0466. Filter and source decisions taken: keep
  min_reviews 3, keep both sources, keep the series guard (docs/progress.md has the diagnostic). Open:
  vectorize features.assemble (63.7 ms), RSS on the backend image.
  `serve.export_val` (val bundle at t1), `litsense_core/features.py` (34 columns),
  `ranking.build_training`, `ranking.train`, `ranking.ablation_val`; the serve bundle ships
  `ranker.txt` + `feature_list.json` and `recommend` uses it (`degraded` only on fallback).
  Next: the Colab cell in `Colab_Phase5_Ranker_Run.md` (`run all --until ranking.ablation_val`),
  then commit `results/v0/ranker/` and read the gate: ranker vs blend NDCG@10 on held-out users
  (`ranker_vs_blend.csv`) and M6 vs M4 (`ablation_val.csv`). Phase 6 (GenAI): built on branch `genai-phase6` (D68 to D71, 2026-10-19): LLM client (gemini via google-genai, fake for tests, SET-ME never calls the network), validators, LLM explain/onboarding/entities/chat with the template paths as fallback, `genai.synthesis` stage feeding `catalog_text.review_summary`, intent and audit scripts. LLM-off mode unchanged. Waiting on the Gemini model ids + `GEMINI_API_KEY`, the intent gold set and the three raters.

## Final v0 numbers (val, 2,578 users; D49)

| model | NDCG@10 | recall@20 | long-tail NDCG@10 | cold-item NDCG@10 |
|---|---|---|---|---|
| itemknn | 0.0227 | 0.0541 | 0.0123 | 0 |
| content (MiniLM) | 0.0191 | 0.0463 | 0.0203 | 0.0230 |
| lightgcn_raw (dim 128) | 0.0212 | 0.0553 | 0.0057 | 0 |
| lightgcn_norm | 0.0168 | 0.0465 | 0.0134 | 0 |
| projection | 0.0207 | 0.0535 | 0.0059 | 0 |
| projection_shuffled | 0.0206 | 0.0575 | 0.0053 | 0 |
| als | 0.0172 | 0.0500 | 0.0043 | 0 |

- LightGCN ties ItemKNN (paired CI includes 0), beats ALS. Tuning closed (D49 stop rule).
- Projection equals its shuffled control; with e_gcn dropout W learns (BPR 0.235 vs 0.309) but
  ranking does not move and cold-item NDCG is 0 (scale mismatch). Content-only owns cold items.
- Per-user oracle (D47): LightGCN and content hit disjoint users (union hit@10 11.6% vs 6.9%);
  oracle NDCG@10 0.0365. That headroom is the Phase 5 ranker's job; PRD G1 is not met by projection.
- Config kept: dim 128, reg 1e-4, batch 2048, `inner_split: temporal`, metric ndcg@10,
  projection `normalize_items: false`, `w_init: zero`, `w_reg: 1e-4`, `gcn_dropout: 0.3`.
- Comparable literature (verify before quoting): LightGCN paper Amazon-Book Recall@20 0.041,
  NDCG@20 0.032; our protocol is stricter (temporal split, cold items kept, 4+ stars relevant).

## Phase 3 numbers (v0, serve set)

552,265 sentences; 294,757 (53%) with an aspect hit (tau 0.45 everywhere; six aspects are > 90%
prototype-only hits, so tau tuning after labelling matters); 797,266 mentions scored; centroids for
4,190 items / 10,769 users; 192,720 snippets; 12,298 regex spoiler flags (2.2%); int8 ONNX parity
failed on real sentences (min cos 0.952), fp32 ships (D62). No precision number exists yet.

## The bundle (D56, D63 to D65)

`artifacts/v0/serve_bundle/` + `serve_bundle.tar.gz` (v0-20261018, 28 files, 93.7 MB; the fp32
ONNX is 90 MB). On Drive also as `serve_bundle_v0_phase3.tar.gz`. Measured in a fresh process:
RSS 229 MB after load, 234 MB after 30 recommends (budget 350); `recommend` 24 ms (budget 80).
Contents: catalog arrays and text, LightGCN serve vectors (float16), content embeddings, item_nn,
trending_top, author/series/genre indices, corpus_users (300 demo readers with exact LightGCN
vectors, counts, offline aspect vectors), metrics.json, item aspect profiles (12) and descriptors
(3), aspect_topitems, `evidence_snippets.jsonl` + offsets (read lazily), spoiler_lr.npz (regex
only), review_centroid_item (float16), aspect_prototypes, descriptor_poles, minilm.onnx (fp32),
tokenizer.json, manifest.json (sha256, shapes, cutoff, config slice, `absent` list = projection,
ranker, feature_list, content.faiss, int8 onnx).
Backend switch: `litsense_core.download_bundle(url, dest)` or extract, `load_store(dir)` once,
then `from litsense_core import recommend, ...` instead of `litsense_core.mock` (API_CONTRACT
changelog row 2026-10-17). `dev/dev_server.py` shows payloads.
**Model service (branch `serve-api`, D75 to D81, 2026-10-20):** `service/` serves every contract function
over HTTP under `/v1` for the Node backend (`X-API-Key`, error envelope, `/ready`, `/metrics`, catalogue
endpoint); `service/README.md`, `service/openapi.json`, `service/Dockerfile`, `render.yaml` (Render free).
Boot it with `BUNDLE_URL` (the GitHub Release asset) or an extracted bundle in `BUNDLE_DIR`, plus
`SERVICE_API_KEY` and optionally `GEMINI_API_KEY`. Measured on the tiny bundle only (D81).

## Key decisions since the original handoff (D47 to D65, short)

- D47 training budget + sweep stage + paper-style projection + embedding diagnostics.
- D48 dim 128, temporal inner split, projection e_gcn dropout. D49 final numbers, tuning closed.
- D50 to D55 Phase 3 block 1 (sentences, aspects, sentiment, centroids, annotation export).
- D56 Week 5 gate: `models.lightgcn_serve`, `serve.export`, real `litsense_core`; corpus readers
  count their corpus history toward the bucket; CF reason line.
- D57 to D62 Phase 3 block 2 (profiles, snippets, spoiler opt-in, ONNX; fp32 ships).
- D63 Phase 3 files into the bundle and the package (aspect source, aspect_match, template 2,
  semantic search, insights, review analysis). D64 lazy snippets + ORT arena off + corpus aspect
  vectors. D65 explicit aspect vectors centered; template 2 on centered values (reader > 0.15,
  item > 0.08).

## Stage order (ml/stages.py STAGE_MODULES)

data.* -> nlp.content -> models.lightgcn_val -> models.projection_val -> eval.val ->
ml.serve.export (models.lightgcn_serve, serve.export) -> nlp.sentences, aspects, sentiment,
centroids_{val,serve}, profiles_{val,serve}, spoiler_train, spoiler_flags, snippets, onnx,
annotation_export -> models.lightgcn_sweep (last). `run all --until eval.val` is unchanged by all
of this; `run all --until serve.export` runs the Phase 3 stages too (they skip when up to date).

## Colab recipe (every run)

```python
from google.colab import drive; drive.mount("/content/drive", force_remount=True)
%cd /content
!rm -rf LitSense && git clone -q --depth 1 https://github.com/zee10524/LightGcn.git LitSense
%cd /content/LitSense
!make install-colab > /dev/null 2>&1; python -m ml.stages run all --until <STAGE> < /dev/null > /content/drive/MyDrive/litsense/<name>.log 2>&1; grep -E "running|complete in|bundle|Traceback|Error" /content/drive/MyDrive/litsense/<name>.log | tail -30
import shutil; shutil.make_archive("/content/drive/MyDrive/litsense/results_v0", "zip", "/content/drive/MyDrive/litsense/results/v0")
drive.flush_and_unmount(); print("saved to Drive")
```

- Drive mount sometimes fails right after an unmount: Runtime > Disconnect and delete runtime, or
  `force_remount=True`. Drive has lost half-written files twice (D42, D62): finished stages then
  rerun once (about 40 min for the data stages). Always end with `flush_and_unmount()`.
- Free T4 budget was enough for everything so far; Colab Pro is not needed yet.

## Open items and who decides

1. Labelling of `docs/annotation/aspects_300.csv` (Zeeshan; a second labeller for kappa), then
   `nlp.validate_aspects`, then the user confirms per-aspect tau.
2. Goodreads spoiler download (user's yes), then `nlp.spoiler_train` reruns and snippets refresh.
3. Merge back into `Aniketdhar810/LitSense` as a PR (asked Aniket; not done).
4. Browser demo on the teammate's backend (closes the Week 5 gate).
5. Phase 5 ranker: built (D66, 2026-10-19); run it on Colab, commit `results/v0/ranker/`, user reads the Week 9/10 gate.
6. v1 rebuild items (D39) before Week 11.
7. CI: red since nlp-phase3b, fixed 2026-10-19 (bde7d78): `holdout_metrics` imported sklearn, which `make install-ci` does not install; AUC now comes from scipy ranks. To reproduce CI locally: install only the CI subset (Makefile `CI_ML_PACKAGES`) in Docker and run `pytest -m 'not slow' --deselect tests/ml/common/test_manifest.py::test_git_sha`.
