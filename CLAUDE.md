# LitSense — Claude Code Implementation Brief (ML + GenAI)

Oct 12, 2026 · @Aniket

## 0. How to use this file (instructions to Claude Code)

You are implementing the ML and GenAI half of LitSense, a book recommender for a B.Tech final-year project. Read this whole file before writing any code. It is the source of truth for the ML/GenAI implementation. If `LitSense_PRD.md` is in the repo, it is the product spec; where the two conflict, this file wins (overrides listed in §1.5).

### 0.1 Scope

- **You own:** everything under `ml/` (offline pipeline, training, evaluation) and `litsense_core/` (the online package the backend imports), plus `tests/`, `notebooks/` (thin wrappers only), `docs/`, `results/`, `ml/config.yaml`.
- **You do not own:** FastAPI routes, the database schema, auth, the Next.js frontend, or deployment. A teammate builds those. Do not create or edit them.
- You may build `dev/dev_server.py`, a tiny FastAPI app that wraps `litsense_core` for local manual testing only.

### 0.2 Working rules

1. Work phase by phase in the order of §17. Do not start a phase until the previous phase's gate passes.
2. Start every session by reading `docs/progress.md` and `docs/decisions.md`. End every session by appending to `docs/progress.md`: what was built, which tests ran, open issues.
3. Record every non-obvious choice in `docs/decisions.md` as: date, decision, reason, alternatives considered. This includes thresholds, the genre slice, and any schema surprise in the data.
4. Never fabricate data, results or metric values. Never hard-code a metric number in code, docs or tests. Results come only from running the pipeline and live in `results/`.
5. Inspect datasets, don't assume them. Before parsing any source, stream its first 5–20 records, print keys and types, and adapt the parser. Field lists in §6 are a starting point, partly from memory.
6. Every stage runs in two environments: locally on a sample (`--sample N`) for development and tests, and on Google Colab (T4 GPU, \~12 GB RAM, \~100 GB local disk) for full runs.
7. Hyperparameters, paths, thresholds and model names live only in `ml/config.yaml`. No magic numbers in code.
8. Write tests with each module (§16). A module is not done until its tests pass.
9. Prefer simple, vectorized numpy/pandas over clever abstractions. Python 3.11, type hints everywhere, docstrings on public functions, `ruff` clean.
10. Pin versions in `requirements-ml.txt` (offline) and `requirements-core.txt` (serving). Keep the serving list short (§4.4).
11. No logic in notebooks. A notebook cell may only call `ml.stages` or public functions.
12. No dead code. Remove experiments that are not wired into a stage.

### 0.3 Ask the user before you

- change the evaluation protocol (§8) or the relevance definition;
- add any dependency to `litsense_core`;
- spend money or exceed the configured LLM daily budget;
- run anything on the test split before the Week-13 final run (§8.8);
- cut, replace or add a component not described here;
- choose the genre slice (§7.2) — propose a default, the user decides;
- accept canonicalization or aspect thresholds after the manual audits (§7.4, §11.5).

### 0.4 Reporting

After each phase gate, write a short summary in `docs/progress.md`: the gate checklist ticked, and any metrics produced, copied from `results/` files (never typed by hand). Then stop and tell the user the gate is ready for review.

### 0.5 Current state and decisions that override this brief (read `context.md` first)

`context.md` (repo root) is the session handoff: where the project stands, what is waiting on the user, and how to work in this environment. `docs/decisions.md` (D1–D46, F1–F12) records every deviation with its reasons. The user-approved deviations that change sections of this file:

- **Stage fingerprints (§7.0):** code part = the stage's module plus its transitive `ml`/`litsense_core` imports, not the repo-wide git SHA (D6). Input identity = the sha256 the producer recorded (D7).
- **Large sources are streamed, never stored (§7.1, §7.5, D18):** `url:<source>` stage inputs; `ml/data/stream.py` (gunzip on the fly, HTTP Range resume, `PartWriter` checkpoints). `data.download` keeps only the 5-core CSV on Drive (D19). `--sample N` reads heads in `local/raw_sample/`.
- **Goodreads graph is optional (§1.5, §7.3, D20):** `data.use_goodreads_graph: false` (default). When on, keep only Goodreads rows whose ISBN-13 matches `books_raw` (incl. Kindle page-source ISBNs). Files come from `sources.goodreads_base` (the Drive links are dead, D15).
- **Kindle and audiobook records stay (F1, D17):** they are editions of the same work; `data.canonical` must merge them with print (ISBN incl. `page_source_isbn13`, else title+author heuristic with box-set/volume guards).
- **Genre slices are set (D26):** Literature & Fiction; Mystery, Thriller & Suspense; Science Fiction & Fantasy.
- **Canonicalization (§7.4, D28–D29):** merge all `books_raw` editions first, keep a work if any edition is in a slice; title merges also need `token_sort_ratio >= canonical.subset_guard_threshold`, the year gap applies only to non-identical titles, volume guard = both known and different or one unknown and the other ≥ 2; audit files live in `results/<v>/canonical/`. `data.lid` keeps `lid.176.ftz` (D30).
- **Canonicalization, later rules (§7.4, D31–D33):** shared-ISBN and page-source merges need the same author (or one missing); "Title" merges with "Title: subtitle" only with the same author, the volume guard, no box set and similar blurbs (`canonical.subtitle_description_*`); fuzzy title merges are refused when the titles' numbers differ ("NYPD Red 2" vs "3").
- **Reviews and cleaning (§7.5–7.6, D34–D35):** dedup = latest per (user, work), then one per (user, text hash); `non_english` only when fastText's other-language probability >= `data.lid_min_confidence`.
- **Interactions (§7.7, D36):** one interaction per (user, work), the latest rating; `has_text` = that review survived cleaning; sizing finds the user fraction by bisection; `splits.json` also holds attrition and the slice mix.
- **Evaluation and baselines (§8–9, D40–D41, D43):** `LeakageError`/`DegenerateScoresError` live in `ml/eval/errors.py` (not `ml/common/errors.py`, to keep data-stage fingerprints stable); scorers record their fit cutoff and the evaluator refuses a mismatch; baselines are refit inside `eval.val` (formerly `eval.baselines_val`) instead of storing artifacts; LightGCN is trained by `models.lightgcn_val` and stored in `artifacts/<v>/eval_val/lightgcn/` (D43). The stage name in §7.0 (`eval.val`) is the one in use; CI installs CPU-only torch.
- **Projection and controls (§9.5, D45):** stage `models.projection_val` trains `projection` and `projection_shuffled` (refuses `shuffled_control: false`); new key `models.projection.n_negs`; projection scorers rank cold items (W c only). Static fusion = alpha * cos(u, e_gcn) + (1 - alpha) * cos(content profile, PCA(c)) as one dot product of concatenated vectors, built in `eval.val`, one row per alpha.
- **Colab and Drive (D37, D38, D42):** checkpoints validate their parts; every Colab run ends with `drive.flush_and_unmount()`; a stage is up to date only if its outputs keep the byte sizes in its manifest.
- **v1 open items (D39):** slice-aware sizing, `kcore.item` 5 vs 10, a 10–20K book catalogue, and a serving-catalogue proposal, all decided before Week 11.
- **Environment:** Colab runs Python 3.13; pins work on 3.11–3.13; `make install-colab` never installs torch (D21); fastText is `fasttext-predict` (D25). Notebooks are committed without outputs; CI enforces it (D27).

## 1. Project context

### 1.1 What LitSense is

LitSense is a web app that recommends books and explains why, built on the Amazon Reviews 2023 Books corpus (McAuley Lab). It learns not only which books a reader rated highly but what they valued in them. It mines aspect-level opinions (plot, prose, characters, pacing…) from review text and combines them with graph collaborative filtering (LightGCN). An LLM makes the system conversational and explainable, but it never chooses which books to recommend.

Worked example of the idea: two readers both rate *Project Hail Mary* 5 stars. Reader A praises the problem-solving science; Reader B praises the friendship. A rating-only model sees them as identical. LitSense's aspect profiles separate them (A: plot/complexity; B: characters/emotional\_impact) and recommends differently.

### 1.2 Research claim and success criteria

Claim: aspect-level opinions from review text add signal beyond collaborative filtering, especially for sparse readers and long-tail books. The test: a LightGBM LambdaRank reranker over \~200 candidates, evaluated under a global temporal split with full-catalogue ranking.

- **Overall:** the full hybrid is not worse than LightGCN on NDCG@10 (paired bootstrap CI), ideally better.
- **Sparse readers** (≤5 training interactions) **and long-tail items:** the hybrid is significantly better than LightGCN (paired bootstrap 95% CI of the difference excludes zero).
- **Long-tail gains are shown against both raw and L2-normalized LightGCN** (see §2.2 for why).
- **Aspect contribution:** ranker with aspect features vs. the same ranker without them (ablation M4 vs M6, §13.7).
- Honest null results are acceptable and are reported as such. Never tune on test to rescue a result.

### 1.3 Product users (why features exist)

| Persona | Needs | Implementation consequence |
| --- | --- | --- |
| Casual reader | Short confident shortlist, one-line reason | Feed must work with <5 ratings; reasons ≤ 1 sentence |
| Power reader (300+ books) | Non-obvious picks, recognition of specific taste | Exposed aspect profile; long-tail candidates; negative preferences |
| Mood reader | Describe a vibe, refine over turns | Multi-turn chat applying deltas to a session state |
| New reader | Useful feed in <60 s | Cover-picker or one-sentence onboarding → content-based feed |
| Admin / examiner | Metrics, ablation, grounding checks | `metrics_report()` JSON consumed by the admin page |

### 1.4 Team split and constraints

- The user (your principal) owns ML + GenAI. A teammate owns backend (FastAPI, Postgres, auth), frontend (Next.js) and deployment.
- **Offline compute:** Google Colab free/limited T4; laptop with 8–16 GB RAM.
- **Online serving:** free-tier CPU container. Plan for **512 MB RAM and 0.1–0.5 vCPU** (Render's free web service is 512 MB / 0.1 CPU and sleeps after 15 idle minutes). Therefore **no PyTorch in serving**; MiniLM runs through ONNX Runtime.
- **LLM:** Gemini API free tier, roughly 15 requests/min and on the order of 1,000–1,500 requests/day on Flash-class models (verify current limits in AI Studio). Free-tier prompts may be used by Google for training, so never send user identifiers.
- **Model churn:** Gemini 2.0 Flash and Flash-Lite were shut down on 1 June 2026. The model string lives only in config. Verify current model IDs on the Gemini API deprecations page before first use.
- **Timeline:** 16 weeks; MVP deployed at Week 8 (§17). Written on 12 Oct 2026.

### 1.5 Overrides to the PRD (this file wins)

| PRD says | Do this instead | Why |
| --- | --- | --- |
| Stream all 29.5M raw Books ratings, then filter | Start from the official Books 5-core rating file (776.4K users, 495.1K items, 9.5M ratings); stream raw files only to fetch metadata and text for surviving items | Removes the riskiest Colab stage |
| Heuristic edition canonicalization only | ISBN join to the UCSD Goodreads Book Graph for work IDs and series; heuristics only as fallback | Edition grouping and series data come almost free |
| Regex spoiler heuristic | Logistic-regression spoiler classifier on MiniLM embeddings, trained on Goodreads sentence-level spoiler labels; regex as an extra signal | Spoiler language is largely book-specific; regex recall is too low |
| Gemini 2.0 Flash | Current Flash / Flash-Lite model, name from config | 2.0 models are shut down |
| HF Spaces as free fallback host | Render free tier (512 MB) | HF now requires a paid plan for Docker/Gradio Spaces |
| DistilBERT online for submitted reviews | Online: MiniLM-ONNX prototype matching + VADER polarity; DistilBERT-SST2 offline only | Memory budget |
| 25 ranker features | 27 features (+ `cf_norm_score`, `proj_hybrid_score`, `descriptor_match`, review-centroid similarity), source flags as 7 binary columns | Controls from prior work; descriptors power chat modifiers |
| G1: beat best baseline by ≥5% NDCG@10 | Criteria in §1.2 | Realistic given prior results and literature |
| 5–6 candidate sources | 7 sources, adding a long-tail source (projection hybrid) | Powers the Hidden Gems rail |
| Content embedding includes `review_summary` | Item content embedding = title, author, genres, description only | Avoids leakage; review semantics live in separate centroids |
| LLM writes chat text including titles | LLM writes per-item reasons keyed by `book_id`; the UI renders titles from the database | Titles can never be hallucinated |
| Incremental online embedding updates (η = 0.15) | Stateless: online profile recomputed from `UserState` on every call | Simpler, no drift, cheap at this scale |
| In scope (stretch) | Out of scope: CLIP covers, SASRec, Kindle linkage, pgvector, LLM reranking in chat | Time |

## 2. Lessons from prior work (HyRec-LBERT-LongTail notebooks)

The team previously built LightGCN + frozen-BERT fusion on MovieLens-1M (paper: "Projection-Based Fusion of Semantic and Collaborative Embeddings for Long-Tail Recommendation"; notebooks `LightGcn_implementation.ipynb`, `MLM_implementation.ipynb`, `Evaluation.ipynb`). Reuse the good parts; never repeat the bugs below. If the user puts these notebooks in `reference/`, read them, but do not copy code without applying these rules.

### 2.1 Reuse

- **LightGCN core:** symmetric normalized adjacency D^-1/2 A D^-1/2 over the bipartite graph; K = 3 propagation layers; final embedding = mean of layers 0..K; BPR loss with several negatives per positive; L2 regularization on layer-0 ("ego") embeddings only, divided by batch size; Adam lr 1e-3; early stopping on a validation split carved from training data. Port into `ml/models/lightgcn.py` (§9.4).
- **Projection hybrid idea:** item embedding = e\_gcn + W\_P · e\_text, with W\_P learned by BPR while LightGCN stays frozen. Keep it as a ranker feature and the long-tail candidate source (§9.5).

### 2.2 Bugs and confounds — never repeat

| # | What happened | Rule now |
| --- | --- | --- |
| 1 | **Normalization confound.** Raw LightGCN item norms grow with popularity. The paper compared raw LightGCN (historical: Recall@20 0.1186, NDCG@20 0.1692, Coverage@20 0.296) against L2-normalized hybrids, yet normalized LightGCN alone (0.0792 / 0.1032 / 0.766) beat the normalized static hybrid. Most of the reported "fusion" effect was normalization. | Every fusion model is compared with **both** raw and normalized LightGCN. Training and evaluation use the same scoring function. Add a shuffled-text control. |
| 2 | ID maps rebuilt in a later notebook from sorted IDs; they matched only because every item appeared in training. | One `id_map.json`, created once (§7.7), loaded everywhere, never rebuilt. |
| 3 | Validation items not masked at test time. | Test mode masks train ∪ val items (§8.1). |
| 4 | Notebook cells run out of order; top-to-bottom execution produced misaligned embeddings. | No logic in notebooks; stages are scripts (§4, §7.0). |
| 5 | `item_final_emb.npy` saved while holding a different model; results labelled "Projection" came from static fusion. | Every artifact has a manifest (model name, config hash, input hashes, git commit). Evaluation prints the manifest next to results. |
| 6 | Training interrupted (KeyboardInterrupt), yet that model was evaluated. | Artifacts are written only on successful completion with `status: complete`; evaluation refuses anything else. |
| 7 | "BERT-only" baseline scored with all-zero user vectors (all scores tie → item-index order). | Content-only baseline uses a real user profile (§9.2). The evaluator raises on rows of tied scores. |
| 8 | Unseeded random projection; config said dim 128 while saved embeddings were 64. | Global seed from config set in numpy, torch, random; dims read from config and asserted against artifact shapes. |
| 9 | Frozen-BERT \[CLS\] vectors nearly identical across items (anisotropy) → \~0 recall on cold items. | Use `sentence-transformers/all-MiniLM-L6-v2` (contrastive, mean-pooled, normalized). Log mean pairwise cosine of item embeddings; warn if > 0.5. |
| 10 | Pseudo-label loss defined but never used. | No dead code. |
| 11 | MinMax-scaled embeddings into \[0, 1\] (all-positive vectors). | Never MinMax embeddings before dot products; center and L2-normalize instead. |
| 12 | Long text truncated at 512 tokens with reviews last, so reviews were dropped. | Put the most important fields first and truncate by tokens; embed descriptions and review text separately. MiniLM's max sequence is 256 word pieces. |
| 13 | Paper text and code disagreed (cold test rows "excluded" in the paper, moved to train in code). | Code is the spec: methods text in the report is generated from config and manifests where possible. |

These historical numbers are context only. Never reuse them as results or targets.

## 3. Non-negotiable engineering principles

1. **The LLM never selects items.** Only the recommender chooses books. The LLM parses intent, writes explanations, and writes replies about items it is handed. Every title-like string it outputs is validated against an allow-list; chat replies are keyed by `book_id` so the UI renders titles from the database.
2. **Leakage-free temporal evaluation.** Every feature, profile, embedding and summary used in evaluation is computed only from data before the relevant cutoff. There are three artifact sets: `eval_val` (cutoff t1, models fit on train), `eval_test` (cutoff t2, models refit on train ∪ val), and `serve` (all data). Every builder takes a `cutoff` argument and raises `LeakageError` if an input row is newer. Manifests record the cutoff; the evaluator asserts it matches the mode.
3. **Test set touched once.** All tuning uses validation. The test run happens once in Week 13 through a guarded stage (§8.8).
4. **Same code offline and online.** Profile building, candidate generation, feature assembly, scoring and post-processing live in `litsense_core` and are imported by `ml`. Never duplicate this logic.
5. **No PyTorch in `litsense_core`.** A test asserts that importing `litsense_core` does not import `torch`.
6. **Idempotent, resumable stages.** Each stage declares inputs and outputs, writes a manifest with a fingerprint, skips work if up to date, and checkpoints long streams.
7. **Graceful degradation.** Every LLM call has a non-LLM fallback. With `llm.enabled: false`, every feature still works (templates, rule-based intent parsing, chip-based chat).
8. **Reproducibility.** One global seed; dataset SHA-256 in `models/registry.json`; split boundaries committed; artifacts published as GitHub Release bundles with manifests.
9. **Spoiler safety.** Any text shown to a user that originates from reviews or the LLM passes the spoiler filter first.
10. **Privacy.** Never send user IDs, emails, usernames or full histories to the LLM. Send only aggregate preferences and book titles. The LLM client asserts this.
11. **Honest reporting.** Results tables are generated from `results/` files. Confidence intervals that cross zero are reported as such.

## 4. System architecture

The system has two halves that share code: an offline pipeline (`ml/`, Colab) that builds artifacts, and a stateless online package (`litsense_core/`, CPU, 512 MB) that the backend imports. Diagrams are ASCII on purpose so they survive Markdown export.

### 4.1 Overview

```text
OFFLINE  (ml/, Google Colab T4)                              ONLINE  (litsense_core/, CPU, <=350 MB RSS)
---------------------------------                            -------------------------------------------
5-core ratings ---+                                           Backend (teammate, FastAPI + Postgres)
meta_Books -------+--> data pipeline (s7) --> interactions      |  builds UserState from its DB
Books reviews ----+                           books, reviews    v
Goodreads --------+                           sentences      litsense_core.recommend(state)
                                              id_map.json      |- ArtifactStore (npy, faiss, lgbm, onnx)
        |                                                      |- online profile (s12.2)
        |--> baselines + LightGCN + projection (s9)           |- 7 candidate sources (s13.1)
        |--> MiniLM content emb + FAISS + ONNX (s10)          |- feature assembly (shared code)
        |--> aspects, sentiment, spoiler, profiles (s11)      |- LightGBM ranker | blend fallback
        |--> candidates + features + LambdaRank (s13)         |- post-process: series, MMR, caps
        |--> evaluation + ablation (s8, s13.7)                 '- template reasons
        '--> serve.export --> GitHub Release bundle ------>  litsense_core.explain / chat_turn
                                                               '- LLM client + validators + fallback
```

### 4.2 Request flow for `recommend()`

```text
UserState --> build_profile --> candidates (7 sources, quotas by interaction bucket)
          --> filters (seen, dismissed, muted, hard filters, series guard)
          --> assemble features (27 cols + 7 source flags)
          --> LightGBM score (or blend fallback)
          --> overrides --> MMR + genre cap + exploration slot --> top-K
          --> evidence + template reason per item --> RecResponse
```

### 4.3 Repository layout

```text
litsense/
|-- ml/
|   |-- config.yaml
|   |-- common/      io.py  manifest.py  seed.py  timeutil.py  log.py  errors.py (LeakageError, DegenerateScoresError)
|   |-- data/        download.py  stream.py  meta.py  goodreads.py  canonical.py  reviews.py
|   |                clean.py  kcore.py  split.py  eda.py
|   |-- eval/        metrics.py  evaluator.py  strata.py  bootstrap.py  report.py  coldstart.py
|   |-- models/      base.py (Scorer protocol)  random_pop.py  itemknn.py  als.py  content.py
|   |                lightgcn.py  projection.py  registry.py
|   |-- nlp/         embed.py  onnx_export.py  sentences.py  aspect_seeds.py  aspects.py
|   |                sentiment.py  spoiler_train.py  profiles_offline.py  snippets.py  annotate.py
|   |-- ranking/     build_training.py  train_ranker.py  ablation.py  candidate_recall.py
|   |-- genai/       synthesis.py  eval_intent.py  audit.py
|   '-- stages.py    # CLI: python -m ml.stages run <stage> [--sample N] [--force]
|-- litsense_core/
|   |-- __init__.py  # public API (s15); __api_version__
|   |-- schemas.py   config.py  store.py  encoder.py  vector_index.py
|   |-- profile.py   candidates.py  features.py  rank.py  postprocess.py  reasons.py
|   |-- spoiler.py   validators.py  entities.py  onboarding.py  intent.py  explain.py  chat.py
|   |-- recommend.py rails.py  insights.py  mock.py
|   '-- llm/         client.py  cache.py  budget.py  adapters/{gemini.py, openai_compat.py, fake.py}
|                    prompts/{onboarding_parse.v1.md, intent_parse.v1.md, explain.v1.md,
|                             chat_reply.v1.md, review_synthesis.v1.md}
|-- dev/dev_server.py         # local FastAPI wrapper, manual testing only
|-- notebooks/                # 00_colab_setup, 01_data, 02_eval_baselines, ... (thin)
|-- tests/                    # mirrors packages; fixtures/tiny/ synthetic data generator
|-- docs/                     # progress.md decisions.md API_CONTRACT.md annotation/ annotation_guides/
|-- results/                  # committed metrics (json/csv) and figures (png)
|-- artifacts/                # gitignored; bundles uploaded to GitHub Releases
|-- requirements-ml.txt  requirements-core.txt  pyproject.toml  Makefile  README.md
```

### 4.4 Dependency rules

- `litsense_core` never imports from `ml`. `ml` may import from `litsense_core` (profile, candidates, features, encoder, spoiler, postprocess) so offline and online share logic.
- `requirements-core.txt` allows only: numpy, faiss-cpu, lightgbm, onnxruntime, tokenizers, rapidfuzz, pydantic (v2), PyYAML, httpx, the LLM SDK (`google-genai`), vaderSentiment. Anything else needs user approval. No pandas, torch, transformers, sentence-transformers or scikit-learn in core.
- `requirements-ml.txt` adds: pandas, pyarrow, torch, sentence-transformers, transformers, scikit-learn, scipy, implicit, spacy, fasttext (language ID), rapidfuzz, matplotlib, onnx, optimum (optional), tqdm, requests, gdown, pytest, ruff.
- Notebooks import only `ml.stages` and public functions.

### 4.5 Storage layout on Colab

```text
/content/raw/                                  # downloads on Colab local disk (NOT Drive; free Drive = 15 GB)
/content/drive/MyDrive/litsense/
    data/<version>/        interactions.parquet  books.parquet  reviews.parquet
                           sentences.parquet  sent_emb.f16.npy  id_map.json  splits.json
    artifacts/<version>/   eval_val/  eval_test/  serve/   (each with manifest.json)
    results/<version>/     val/  test/  eda/
./local/                                       # same tree for --sample runs on the laptop
```

### 4.6 Serving bundle contents

Built by `serve.export` from `artifacts/<v>/serve/`, cast to float16 where noted, tarred with a manifest.

| File | Shape / type | Purpose |
| --- | --- | --- |
| `catalog.npz` | arrays over n\_items: year, pages, series\_id, series\_pos, genre bitmask, primary genre, bayes\_rating, rating\_count, rating\_std, popularity, trending, is\_longtail, n\_reviews, author\_id | Filters and metadata features |
| `catalog_text.json.gz` | titles, authors, genre labels, image URLs, review\_summary | Reasons, entity resolution, insights |
| `lgcn_item.f16.npy`, `lgcn_item_norm.f16.npy` | (n\_items, 64) | CF scoring (raw and normalized) |
| `proj_item.f16.npy` | (n\_items, 64), already normalized | Long-tail source, `proj_hybrid_score` |
| `content_item.f16.npy` + `content.faiss` | (n\_items, 384) | Semantic retrieval and similarity |
| `item_nn.npy` | (n\_items, 50) int32 | Precomputed content neighbours |
| `review_centroid_item.f16.npy` | (n\_items, 384) | Review-embedding similarity |
| `item_aspects.npy`, `item_mention_rate.npy`, `item_pos_rate.npy`, `item_neg_rate.npy` | (n\_items, 12) float32 | Aspect features, charts, reasons |
| `item_descriptors.npy` | (n\_items, 3) | darkness, pace, complexity\_level |
| `aspect_topitems.json` | aspect → item ids | Aspect candidate source |
| `author_index.json`, `series_index.json`, `genre_index.json` | maps | Author source, series guard, genre filters |
| `evidence_snippets.json.gz` | item → aspect → safe snippets | Reasons and insights |
| `trending_top.npy` | (200,) | Popularity source |
| `aspect_prototypes.npy` | (12, 8, 384) | Online review analysis |
| `spoiler_lr.npz` | coef (384,), intercept, threshold | Runtime spoiler check |
| `minilm.onnx` (+ `minilm.int8.onnx`), `tokenizer.json` | ONNX model | Query/onboarding/review embedding |
| `ranker.txt`, `feature_list.json` | LightGBM model | Ranking |
| `corpus_users.npz` | demo user ids → LightGCN vectors (a few hundred users max) | Dense demo accounts |
| `manifest.json` | versions, sha256, shapes, cutoff | Integrity, `health()` |

Budget: embeddings for \~20K items are \~15 MB each at 384-d float16; total bundle well under 200 MB; process RSS target ≤ 350 MB.

## 5. Configuration (`ml/config.yaml`)

Create this file in Week 1. Values marked `# tune` are starting points to adjust on validation only; record changes in `docs/decisions.md`. `litsense_core` reads only the `serving`, `ranking.postprocess`, `ranking.quotas`, `ranking.blend`, `aspects` and `llm` sections (copied into the bundle manifest at export).

```yaml
project:
  seed: 42
  data_version: v0              # v0 ~100K interactions (development), v1 ~250K (final)
  drive_root: /content/drive/MyDrive/litsense
  raw_root: /content/raw
  local_root: ./local           # used by --sample runs

sources:
  ratings_5core: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz
  meta_books:    https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_Books.jsonl.gz
  reviews_books: https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Books.jsonl.gz
  goodreads_page: https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html
  goodreads_files: [goodreads_books.json.gz, goodreads_book_series.json.gz,
                    goodreads_book_works.json.gz, goodreads_reviews_spoiler.json.gz]
  fasttext_lid: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz

data:
  genre_slices: []              # SET AFTER EDA (s7.2); top-level category names under Books
  fiction_allowlist: [Literature & Fiction, Science Fiction & Fantasy,
                      'Mystery, Thriller & Suspense', Romance, Teen & Young Adult]
  title_blocklist: [journal, planner, notebook, coloring book, colouring book, calendar,
                    sticker book, activity book, blank book, workbook, study guide,
                    cliffsnotes, sparknotes, summary of, test prep, puzzle book]
  min_pages: 40
  min_review_chars: 20
  max_review_chars: 5000
  kcore: {user: 5, item: 10}    # tune after EDA
  target_interactions: {v0: 100000, v1: 250000}
  size_tolerance: 0.15
  split_quantiles: [0.85, 0.925]
  relevance_min_rating: 4
  longtail_fraction: 0.8
  sparsity_buckets: [[1, 2], [3, 5], [6, 10], [11, 20], [21, 1000000]]
  shipping_patterns: [arrived damaged, pages (were )?(missing|falling out|torn|blank),
                      wrong (book|edition|item), (print|font) (is|was) (too )?(small|tiny|blurry),
                      binding, cover (was )?(bent|damaged|ripped), kindle (version|formatting),
                      the seller, shipping, package, delivered]

canonical:
  drop_boxsets: true
  fuzzy_threshold: 90           # rapidfuzz token_set_ratio on normalized titles
  max_year_gap: 3
  boxset_patterns: [box(ed)? set, omnibus, complete series, complete collection,
                    collection set, books? \d+ ?(-|to|through) ?\d+, \d+ ?books? set, trilogy set]
  edition_noise: [hardcover, paperback, mass market, large print, unabridged, abridged,
                  illustrated edition, anniversary edition, special edition, deluxe edition,
                  collector's edition, reprint, kindle edition, audio cd, library binding,
                  a novel, a thriller, a mystery, a memoir]

models:
  lightgcn: {dim: 64, layers: 3, lr: 0.001, batch_size: 4096, max_epochs: 300,
             reg: 1.0e-5, n_negs: 5, eval_every: 5, patience: 10,
             inner_val_fraction: 0.1, inner_val_min_user: 6, metric: ndcg@20}
  als: {factors: 64, regularization: 0.01, iterations: 30, alpha: 20}
  itemknn: {k: 100, shrink: 10}
  popularity: {bayes_m: 10, trend_half_life_days: 180}
  projection: {lr: 0.001, epochs: 10, steps_per_epoch: 1000, batch_size: 1024,
               normalize_items: true, shuffled_control: true}
  static_fusion_alphas: [0.5, 0.7, 0.9]   # control only

content:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  max_seq_length: 256
  batch_size: 256
  anisotropy_warn: 0.5

nlp:
  spacy_n_process: 2            # fall back to 1 if Colab hangs
  min_sentence_tokens: 4
  max_sentence_tokens: 80
  sentence_embed_batch: 512
  sentiment_model: distilbert-base-uncased-finetuned-sst-2-english
  sentiment_batch: 256
  clause_split_markers: [' but ', ' although ', ' though ', ' however ', ' yet ', ' whereas ']
  pos_threshold: 0.5
  neg_threshold: -0.5

aspects:
  names: [plot, writing_style, characters, world_building, pacing, editing_quality,
          dialogue, humor, emotional_impact, ending, complexity, tone_dark]
  tau_default: 0.45             # tune per aspect in s11.5
  tau: {}
  shrinkage_k: 5
  user_half_life_days: 730
  user_explicit_weight: 0.6     # vs implicit (liked items' profiles)
  descriptors: [darkness, pace, complexity_level]
  rating_weights: {5: 1.0, 4: 0.5, 3: 0.0, 2: -0.5, 1: -1.0}

spoiler:
  sample_size: 200000
  holdout_fraction: 0.1
  C_grid: [0.1, 1.0, 10.0]
  target_recall: 0.9
  regex_boost: 0.15
  late_position_boost: 0.05     # sentences in last 30% of a review
  regex_patterns: [\bdies\b, \bdied\b, \bkilled\b, turns out, in the end, the ending,
                   twist (is|was), revealed to be, is actually, \bmarries\b, it was all,
                   the killer, the murderer, ends up]

ranking:
  candidates_total: 200
  quotas:                       # by interaction bucket; sums exceed 200 on purpose (dedupe)
    cold:     {cf: 0,  semantic: 110, item_item: 40, popularity: 50, author: 0,  aspect: 20, longtail: 20}
    warming:  {cf: 40, semantic: 80,  item_item: 40, popularity: 35, author: 10, aspect: 20, longtail: 20}
    blending: {cf: 70, semantic: 60,  item_item: 40, popularity: 25, author: 15, aspect: 25, longtail: 20}
    warm:     {cf: 90, semantic: 55,  item_item: 40, popularity: 25, author: 15, aspect: 25, longtail: 20}
  buckets: {cold: [0, 0], warming: [1, 4], blending: [5, 19], warm: [20, 1000000]}
  min_reviews_main_feed: 3
  negatives_per_positive: 30
  ranker_user_split: 0.8
  lgbm: {objective: lambdarank, n_estimators: 500, learning_rate: 0.05, num_leaves: 31,
         min_child_samples: 20, colsample_bytree: 0.8, subsample: 0.8, subsample_freq: 1,
         early_stopping_rounds: 50, eval_at: [10]}
  blend: {alpha_k: 8, w_pop: 0.1, w_aspect: 0.15}
  postprocess: {mmr_lambda: 0.75, mmr_pool: 50, genre_cap: 4, explore_slot: 8}
  overrides: {pinned_aspect_boost: 0.1}

serving:
  bundle_dir: artifacts/serve_bundle
  max_rss_mb: 350
  onnx_quantized: true
  onnx_threads: 1
  verify_hashes_on_load: false

llm:
  enabled: true
  provider: gemini              # gemini | openai_compat | fake
  model: SET-ME                 # current Gemini Flash-Lite id; verify on the deprecations page
  model_quality: SET-ME         # current Gemini Flash id, offline synthesis only
  api_key_env: GEMINI_API_KEY
  temperature: {explain: 0.3, chat: 0.5, parse: 0.0, synthesis: 0.3}
  max_output_tokens: {explain: 120, chat: 400, parse: 400, synthesis: 200}
  timeout_s: 8
  max_retries: 2
  rpm_limit: 12                 # stay under the free-tier limit
  daily_request_budget: 1200
  cache_path: .cache/llm.sqlite
  cache_ttl_hours: 168
  explain_max_words: 60
  chat_max_words: 120

eval:
  ks: [5, 10, 20, 50]
  primary: ndcg@10
  bootstrap_resamples: 1000
  cold_user_sim: {n_users: 500, min_relevant: 8, onboarding_picks: 3}
```

Note: patterns are regular expressions stored as YAML strings; quote any entry containing `:` or starting with a special character, and compile them once with `re.IGNORECASE`.

## 6. Data sources, download URLs and schemas

All schemas below must be verified against the first records of each file (rule 0.2.5). Fields marked (verify) are from memory.

### 6.1 Amazon Reviews 2023 — McAuley Lab

Cite: Hou, Li, He, Yan, Chen, McAuley (2024), "Bridging Language and Items for Retrieval and Recommendation", arXiv:2403.03952. Research use. Raw category stats for Books: 10.3M users, 4.4M items, 29.5M ratings.

| File | URL (also in config) | Content | Use |
| --- | --- | --- | --- |
| Books 5-core ratings | `.../benchmark/5core/rating_only/Books.csv.gz` | CSV `user_id,parent_asin,rating,timestamp`; 776.4K users, 495.1K items, 9.5M ratings; de-duplicated | Interaction backbone |
| meta\_Books | `.../raw/meta_categories/meta_Books.jsonl.gz` | One JSON object per item | Metadata for 5-core items |
| Books reviews | `.../raw/review_categories/Books.jsonl.gz` | One JSON object per review (29.5M) | Review text for surviving interactions |
| Optional: 5-core absolute-timestamp split | `.../benchmark/5core/timestamp_w_his/Books.{train,valid,test}.csv.gz` | Official split | Optional secondary benchmark only |

URL prefix: `https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023`.

**Review record fields:** `rating` (float 1–5), `title`, `text`, `images`, `asin`, `parent_asin`, `user_id`, `timestamp` (documented as unix time, but values are milliseconds — assert the range 8e11–1.8e12), `verified_purchase` (bool), `helpful_vote` (int). Always join on `parent_asin`, never `asin`.

**Meta record fields:**

| Field | Type | Notes for books |
| --- | --- | --- |
| `parent_asin` | str | Item key; matches the 5-core file. For print books it often equals the ISBN-10 (measure the rate in EDA) |
| `title` | str | Often contains format/edition noise |
| `subtitle` | str (verify) | Sometimes format + date, e.g. "Hardcover – January 1, 2004" |
| `author` | dict or null | `{name, avatar, about}`; may be a JSON string in some copies |
| `store` | str | e.g. "Peter Ackroyd (Author)" — fallback author source |
| `categories` | list\[str\] | Path, e.g. \["Books", "Literature & Fiction", "History & Criticism"\]; can be empty |
| `details` | dict or JSON string | Book keys seen: `Publisher` ("Name; Edition (Month D, YYYY)"), `Language`, `Hardcover`/`Paperback`/`Mass Market Paperback` ("N pages"), `ISBN 10`, `ISBN 13` ("978-…"), `Item Weight`, `Dimensions` |
| `description` | list\[str\] | Join with spaces; strip HTML |
| `features` | list\[str\] | Fallback description |
| `images` | list\[dict\] | Keys `thumb`, `large`, `hi_res`, `variant` ("MAIN" first) |
| `average_rating`, `rating_number` | float, int | Amazon-page aggregates (do not use as features: they include future ratings) |
| `price`, `videos`, `bought_together`, `main_category` | — | Ignore |

### 6.2 UCSD Goodreads Book Graph — Wan & McAuley

Page: `https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/UCSD%20Book%20Graph.html`. The complete book graph covers 2,360,655 books (1,521,962 works, 400,390 series, 829,529 authors), collected late 2017, so books published after 2017 will not match. Download links on the page may point to Google Drive; if direct download fails, use `gdown` with the file id from the page, or ask the user to upload the files to Drive.

Cite: Wan & McAuley, RecSys 2018 ("Item Recommendation on Monotonic Behavior Chains"); Wan, Misra, Nakashole & McAuley, ACL 2019 ("Fine-Grained Spoiler Detection from Large-Scale Review Corpora").

| File | Expected fields (verify) | Use |
| --- | --- | --- |
| `goodreads_books.json.gz` | `book_id`, `work_id`, `isbn`, `isbn13`, `asin`, `kindle_asin`, `title`, `title_without_series`, `series` (list of series ids), `authors` (list of {author\_id, role}), `num_pages`, `publication_year`, `popular_shelves`, `description`, `language_code` | ISBN → work and series |
| `goodreads_book_series.json.gz` | `series_id`, `title`, `numbered`, `primary_work_count` | Series names |
| `goodreads_book_works.json.gz` | `work_id`, `best_book_id`, `original_title`, `original_publication_year` | Canonical title and year |
| `goodreads_reviews_spoiler.json.gz` | \~1.38M reviews; `review_sentences` = list of \[label, sentence\]; `has_spoiler`, `book_id`, `rating`, `user_id`, `review_id` | Spoiler classifier training |

Series position is not a separate field: parse it from the title suffix, e.g. "The Way of Kings (The Stormlight Archive, #1)" → ("The Stormlight Archive", 1.0). Positions can be fractional (#2.5 novellas).

### 6.3 Pretrained models and tools

| Item | Use | Where |
| --- | --- | --- |
| `sentence-transformers/all-MiniLM-L6-v2` (384-d) | Item, sentence, query embeddings; spoiler features | Offline (torch) and online (ONNX) |
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentence polarity | Offline only |
| spaCy `blank("en")` + `sentencizer` | Sentence splitting | Offline |
| fastText `lid.176.ftz` | Language ID | Offline |
| `vaderSentiment` | Polarity for online review analysis | Online |
| `rapidfuzz` | Fuzzy titles/authors | Both |
| LightGBM | LambdaRank | Both (predict online) |
| `implicit` | ALS baseline | Offline |
| Gemini via `google-genai` | LLM roles (§14) | Online + offline synthesis |

## 7. Phase 1 — Data pipeline (`ml/data/`)

Phase 1 turns the raw files into a frozen dataset: `interactions.parquet`, `books.parquet`, `reviews.parquet`, `id_map.json`, `splits.json`, plus EDA. v0 targets \~100K interactions for development; v1 (\~250K) is rebuilt in Week 11 with the same code.

### 7.0 Stage framework (build first)

`ml/common/manifest.py` + `ml/stages.py`.

```python
@stage(name="data.meta", inputs=["raw:ratings_5core", "raw:meta_books"],
       outputs=["data:books_raw.parquet", "data:category_counts.csv"],
       config_keys=["data.title_blocklist", "data.min_pages"])
def build_meta(ctx: StageContext) -> None: ...
```

- **Fingerprint** = sha256 of (selected config values, input manifests' fingerprints or raw file size+mtime, git short SHA of `ml/` + `litsense_core/`). If every output exists with a `manifest.json` whose fingerprint matches and `status == "complete"`, skip and log "up to date".
- **Writes** go to `<path>.tmp` and are renamed atomically. The manifest is written last: `{stage, fingerprint, created_at, git_sha, cutoff (if any), files: {name: {sha256, rows|shape, dtype}}, status: "complete", notes}`.
- **CLI:** `python -m ml.stages list`; `python -m ml.stages run data.meta [--sample 50000] [--force]`; `python -m ml.stages run all --until eval.val`. `--sample N` reads only the first N lines of each raw file and writes under `project.local_root`.
- **Logging:** each stage logs row counts in and out; these feed the attrition table (§7.8).
- **Seeds:** `ml/common/seed.py: set_global_seed(seed)` sets `random`, `numpy`, `torch` (if importable), and `PYTHONHASHSEED`; every stage calls it.

| Stage | Inputs | Outputs |
| --- | --- | --- |
| `data.download` | config URLs | `/content/raw/*` |
| `data.meta` | 5-core CSV, meta\_Books | `books_raw.parquet`, `category_counts.csv` |
| `data.goodreads` | Goodreads books + series + works | `gr_isbn_index.parquet`, `gr_series.parquet` |
| `data.canonical` | `books_raw`, `gr_isbn_index`, `data.genre_slices` | `asin_to_canonical.parquet`, `books_canonical.parquet`, `canonical_report.json`, `merge_audit.csv` |
| `data.reviews` | reviews jsonl, 5-core CSV, `asin_to_canonical` | `reviews_raw.parquet` |
| `data.clean` | `reviews_raw` | `reviews_clean.parquet` |
| `data.kcore_split` | `reviews_clean`, 5-core CSV, `books_canonical` | `interactions.parquet`, `books.parquet`, `reviews.parquet`, `id_map.json`, `splits.json` |
| `data.eda` | all of the above | `results/<v>/eda/*` |

### 7.1 `data.download`

- Stream-download with `requests` (chunked, `Range` header to resume), verify the gzip stream end-to-end once, record size + sha256 in a raw manifest.
- Goodreads: try direct links; on failure print the exact manual steps (which files, where to put them) and exit non-zero.
- Download fastText `lid.176.ftz`.

### 7.2 `data.meta`

1. Load the 5-core CSV (`dtype={'user_id': str, 'parent_asin': str, 'rating': 'float32', 'timestamp': 'int64'}`) and build `keep = set(parent_asin)`.
2. Stream `meta_Books.jsonl.gz` with `gzip.open(..., 'rt')` line by line. Skip records whose `parent_asin` is not in `keep`.
3. Parse each kept record with `parse_meta` (below).
4. Drop non-narrative items: title contains a `title_blocklist` term (case-insensitive, word-boundary), or `pages < min_pages` when pages is known. Count drops by reason.
5. Write `books_raw.parquet` and `category_counts.csv`: for each `categories[1]` (top level under Books) and each `categories[1:3]` pair, the number of items and the number of 5-core interactions.
6. **Stop and ask the user** to choose `data.genre_slices` from `category_counts.csv`. Propose a default: the 2–3 entries from `fiction_allowlist` with the most interactions whose combined interaction count is at least 4× the v1 target (k-core and canonicalization will remove a lot). Write the choice to config and decisions.md.

```python
DATE_RE  = re.compile(r"\(([A-Z][a-z]+ \d{1,2}, \d{4})\)")
PAGES_RE = re.compile(r"(\d+)\s+pages", re.I)
STORE_RE = re.compile(r"^(.*?)\s*\((Author|Editor|Illustrator)", re.I)
FORMAT_KEYS = ("Hardcover", "Paperback", "Mass Market Paperback", "Board book",
               "Library Binding", "Spiral-bound", "Audio CD")

def _maybe_json(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except json.JSONDecodeError: return {}
    return x or {}

def parse_meta(r: dict) -> dict | None:
    title = clean_html(r.get("title") or "").strip()
    if not title:
        return None
    d = _maybe_json(r.get("details"))
    a = _maybe_json(r.get("author"))
    author = (a.get("name") if isinstance(a, dict) else None) or None
    if not author and r.get("store"):
        m = STORE_RE.match(r["store"]); author = m.group(1).strip() if m else None
    pages = next((int(m.group(1)) for v in d.values() if isinstance(v, str)
                  for m in [PAGES_RE.search(v)] if m), None)
    fmt = next((k for k in FORMAT_KEYS if k in d), None)
    m = DATE_RE.search(str(d.get("Publisher", "")))
    year = parse_year(m.group(1)) if m else parse_year(d.get("Publication date"))
    isbn10 = norm_isbn(d.get("ISBN 10")) or (r["parent_asin"] if ISBN10_RE.match(r["parent_asin"]) else None)
    images = r.get("images") or []
    image = next((i.get("large") for i in images if i.get("variant") == "MAIN" and i.get("large")),
                 next((i.get("large") for i in images if i.get("large")), None))
    return {"parent_asin": r["parent_asin"], "title": title, "author": author,
            "categories": r.get("categories") or [],
            "description": clean_html(" ".join(r.get("description") or r.get("features") or [])),
            "isbn10": isbn10, "isbn13": norm_isbn(d.get("ISBN 13")), "pages": pages,
            "format": fmt, "year": year, "language": d.get("Language"), "image": image}
```

`norm_isbn` keeps digits and a trailing X; `ISBN10_RE = ^\d{9}[\dX]$`. Note `images` may also arrive as a dict of lists in some copies; handle both.

### 7.3 `data.goodreads`

1. Stream `goodreads_books.json.gz`; for every record with `isbn` or `isbn13`, emit `{isbn10, isbn13, gr_book_id, work_id, series_ids, title, title_without_series, num_pages, publication_year, ratings_count}`.
2. Parse series position from `title` with `SERIES_RE = \(([^()]+?),?\s*#(\d+(?:\.\d+)?)\)\s*$` → `series_name`, `series_pos` (float). Keep the first series id when several exist.
3. Stream the series file → `series_id → title, numbered`. Join names.
4. Write `gr_isbn_index.parquet` (keyed by isbn13 and isbn10) and `gr_series.parquet`. 2.4M rows fit in memory.

### 7.4 `data.canonical`

Goal: map every kept `parent_asin` to a `canonical_id` (one work), never merging different works, different volumes, or a box set with a volume.

1. **Genre slice:** keep items whose `categories[1]` is in `data.genre_slices`.
2. **`normalize_title(t)`:** NFKC, lowercase; remove bracketed/parenthetical segments that contain edition noise or format words; remove series suffixes like "(book 3 of …)" and "(the x series, #3)" (store the series info first); remove a subtitle after `:` only when it matches a noise phrase ("a novel", "a thriller", "special edition"…); strip punctuation; collapse spaces; drop a leading "the ", "a ", "an ".
3. **`is_boxset(t)`:** any `canonical.boxset_patterns` match. With `drop_boxsets: true`, remove them from the catalogue (log the count; their interactions disappear).
4. **`volume_number(t)`:** from "book 3", "#3", "vol. 3", "volume iii", "part ii", "3 of"; roman numerals I–XX. Used only as a merge guard.
5. **Goodreads join:** look up isbn13, then isbn10. Matched items get `canonical_id = "gr_" + work_id` plus `series_id`, `series_name`, `series_pos`.
6. **Heuristic fallback** for unmatched items: block by (first token of normalized title, author surname lowercased). Within a block, merge two items when `fuzz.token_set_ratio(norm_a, norm_b) >= fuzzy_threshold` AND same normalized author AND equal volume numbers (or both None) AND neither is a box set AND (year gap ≤ `max_year_gap` or a year is missing). Use union-find. An unmatched item may also join a Goodreads work in the same block by the same rule. Unmerged groups get `canonical_id = "am_" + min(parent_asin)`.
7. **Canonical record:** title = the member title with the most 5-core interactions, cleaned of format noise; author = most common; description = longest; categories = union; year = min; pages = median; image = from the most-rated member; isbn13 = from the most-rated member; `member_asins` list; series fields; `gr_matched`; `n_members`.
8. **Genre labels:** map category paths to \~25 fixed labels with an ordered keyword table in `ml/data/genres.py` (e.g. "Fantasy" → Fantasy, "Science Fiction" → Science Fiction, "Thriller" → Thriller, "Mystery" → Mystery, "Romance" → Romance, "Historical" → Historical Fiction, "Horror" → Horror, "Literary" → Literary Fiction, "Young Adult|Teen" → Young Adult …). `primary_genre` = most specific match; unmatched → "Fiction (general)".
9. **Audit outputs:** `canonical_report.json` (Goodreads match rate, members-per-canonical histogram, box sets dropped, unmatched count) and `merge_audit.csv`: 100 random multi-member groups and 50 near-threshold non-merges (score 80–89) with titles, authors, years, scores. **Ask the user to review it** and record merge precision in decisions.md before moving on.

### 7.5 `data.reviews`

1. Build the set of 5-core `(user_id, parent_asin)` pairs restricted to items in `asin_to_canonical`.
2. Stream `Books.jsonl.gz` (29.5M lines; a long run). Keep a record only if its `(user_id, parent_asin)` is in the 5-core set. Keep `user_id, parent_asin, canonical_id, rating, timestamp, title, text, helpful_vote, verified_purchase`.
3. Checkpoint every 1M input lines into `reviews_part_XXXX.parquet` plus a progress file with the line offset; on restart, skip finished parts.
4. Concatenate. Deduplicate per `(user_id, canonical_id)` keeping the latest timestamp (this collapses the same review posted on hardcover and paperback). Then drop rows whose `(user_id, sha1(normalized text))` already appeared (cross-edition copy-paste).
5. 5-core pairs with no review record found stay as interactions with empty text; log how many.

### 7.6 `data.clean`

- Strip HTML tags, `html.unescape`, NFKC, collapse whitespace.
- fastText language ID on the first 500 characters; non-English → `text_ok = False`.
- Length: shorter than `min_review_chars` → `text_ok = False`; longer than `max_review_chars` → truncate at the last sentence boundary before the limit.
- Shipping/format complaints: a `shipping_patterns` match AND no aspect-lexicon hit (from `ml/nlp/aspect_seeds.py`) → `text_ok = False`.
- The interaction is always kept (the rating is still signal); `text_ok` only controls NLP use. Report fractions removed by each rule.

### 7.7 `data.kcore_split`

```python
def kcore(df: pd.DataFrame, ku: int, ki: int) -> pd.DataFrame:
    while True:
        n = len(df)
        df = df[df.groupby("user_id")["canonical_id"].transform("size") >= ku]
        df = df[df.groupby("canonical_id")["user_id"].transform("size") >= ki]
        if len(df) == n:
            return df.reset_index(drop=True)
```

1. Build interactions `(user_id, canonical_id, rating, ts, has_text)`; apply `kcore`. Assert the invariants afterwards.
2. **Sizing:** if interactions exceed `target × (1 + size_tolerance)`, sample users (seeded) with fraction f = target / current, re-run k-core, and adjust f until within tolerance. Record f in decisions.md.
3. **Split:** `t1, t2 = df.ts.quantile(split_quantiles)`; train = ts < t1, val = t1 ≤ ts < t2, test = ts ≥ t2. Save `splits.json` with t1, t2 (ms and ISO dates), counts per split, users/items per split, share of val/test users with zero train interactions, share of val/test items with zero train interactions.
4. **IDs:** `id_map.json = {"users": sorted user_id strings, "items": sorted canonical_id strings}`; list position = integer id. Created once here; every later stage loads it via `ml.common.io.load_id_map()`; nothing else may build ids.
5. Write:

| File | Columns |
| --- | --- |
| `interactions.parquet` | `u` int32, `i` int32, `user_id`, `canonical_id`, `rating` float32, `ts` int64 (ms), `split` (train/val/test), `has_text` bool |
| `books.parquet` | `i`, `canonical_id`, `title`, `author`, `author_key`, `description`, `genres` list, `primary_genre`, `categories_raw`, `year`, `pages`, `image`, `isbn13`, `member_asins`, `series_id`, `series_name`, `series_pos`, `gr_matched`, `n_members` |
| `reviews.parquet` | `review_row` int64, `u`, `i`, `rating`, `ts`, `title`, `text`, `helpful_vote`, `verified`, `text_ok` |

`author_key` = lowercase author with initials normalized ("J.K. Rowling" and "JK Rowling" → "jk rowling").

### 7.8 `data.eda`

Write PNGs (150 dpi) and `results/<v>/eda/summary.json`:

1. Attrition table: users/items/interactions after each stage (5-core → metadata found → non-narrative removed → genre slice → box sets removed → canonical merge → reviews found → k-core → sizing).
2. Interactions per user and per item (log-log); long-tail curve (cumulative share of interactions vs share of items).
3. Rating distribution overall and per split (expect heavy 5-star skew; report the share of ratings ≥ 4).
4. Review length distribution; `text_ok` fractions by reason.
5. Interactions over time with t1/t2 marked.
6. Genre distribution; members-per-canonical histogram; Goodreads match rate; share of `parent_asin` equal to ISBN-10.
7. Author frequency (authors with ≥ 2 books) — decides how useful the author source is.
8. Cold shares: val/test users with 0 train interactions; val/test items with 0 train interactions.

Aspect figures come later (§11).

**Gate (end of Week 3):** v0 manifests complete; dataset sha256 in `models/registry.json`; EDA committed; merge audit reviewed by the user; genre slice and k-core values recorded in decisions.md.

## 8. Phase 2 — Evaluation harness (`ml/eval/`)

Build the harness before any model; every claim in the report flows through it. Every model is evaluated by the same `evaluate()` function.

### 8.1 Protocol

| Mode | Fit data | Profiles/features cutoff | Masked items per user | Ground truth per user |
| --- | --- | --- | --- | --- |
| `val` | train | t1 | the user's train items | val interactions with rating ≥ 4 |
| `test` | train ∪ val | t2 | the user's train ∪ val items | test interactions with rating ≥ 4 |

- **Evaluated users (main table):** ≥ 1 relevant ground-truth item and ≥ 1 fit-data interaction. Users with zero fit interactions are evaluated separately (§8.5).
- **Catalogue:** every item in `books.parquet` (full-catalogue ranking; no sampled negatives). CF-only models cannot score items with zero fit-data degree: they get score −inf and show up in the cold-item stratum. Content-capable models can score them.
- **Graph edges** for CF use all fit-window interactions regardless of rating (implicit). Relevance for evaluation uses rating ≥ `relevance_min_rating`. Record this in decisions.md.
- **K** ∈ {5, 10, 20, 50}. Primary metrics: NDCG@10 and Recall@20.

### 8.2 `metrics.py`

Per user, given a ranked list `top` (item ids, length ≥ K) and a truth set:

- `precision@k = hits@k / k`
- `recall@k = hits@k / |truth|`
- `hit@k = 1 if hits@k > 0`
- `ndcg@k`: binary gains, DCG = Σ 1/log2(rank + 1) over hits (rank starts at 1), IDCG over min(|truth|, k) ideal positions
- `mrr@k = 1/rank of the first hit within k, else 0`

Aggregate = mean over users. Beyond-accuracy, computed over all evaluated users' top-K lists: `coverage@k` (distinct recommended items / catalogue size; state the denominator), `gini@k` of recommendation counts, `novelty@k` (mean −log2 of each item's share of fit interactions), `ild@k` (mean pairwise 1 − cosine of content embeddings), `tail_share@k` (fraction of recommended items in the long-tail set).

### 8.3 `evaluator.py`

```python
class Scorer(Protocol):
    name: str
    def fit(self, data: FitData) -> None: ...
    def score(self, users: np.ndarray) -> np.ndarray: ...   # (len(users), n_items) float32

def evaluate(scorer: Scorer, mode: Literal["val", "test"], ks=(5, 10, 20, 50),
             batch: int = 1024, strata: bool = True) -> EvalResult:
    data = load_eval_data(mode)              # fit interactions, masks, truth, tail set, cutoff
    assert scorer.manifest["cutoff"] == data.cutoff and scorer.manifest["status"] == "complete"
    K, rows = max(ks), []
    for b in range(0, len(data.users), batch):
        ub = data.users[b:b + batch]
        S = scorer.score(ub).astype(np.float32, copy=True)
        if np.any(np.nanmax(S, 1) == np.nanmin(S, 1)):
            raise DegenerateScoresError(scorer.name)        # all-tie rows (zero user vector bug)
        for r, u in enumerate(ub):
            S[r, data.mask[u]] = -np.inf
        top = np.argpartition(-S, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(S, top, 1), axis=1, kind="stable")
        top = np.take_along_axis(top, order, 1)
        rows += [per_user_metrics(u, top[r], data.truth[u], ks) for r, u in enumerate(ub)]
    return aggregate(rows, scorer, data)
```

- `data.mask[u]` and `data.truth[u]` are int arrays; assert they are disjoint.
- Candidate-based pipelines (the ranker) return scores for candidates and −inf elsewhere; ties at −inf are acceptable (exempt them from the degenerate check by checking only finite values). Document: non-candidates rank after candidates.
- Save per-user rows to `results/<v>/<mode>/<scorer>.users.parquet` (for bootstrap and strata) and aggregates to `<scorer>.json` together with the scorer's manifest.

### 8.4 `strata.py`

- **User sparsity:** bucket by fit-interaction count using `data.sparsity_buckets`.
- **Long tail:** bottom `longtail_fraction` of items by fit-window interaction count (zero-count items included). Long-tail evaluation restricts each user's truth to tail items; users without tail truth are dropped; candidates and scoring are unchanged.
- **Cold items:** truth items with zero fit interactions.
- Report every metric per stratum, with user counts.

### 8.5 `coldstart.py` — simulated new readers

1. Pick `cold_user_sim.n_users` users (seeded) with ≥ `min_relevant` relevant interactions overall.
2. Remove all their interactions from the fit data and refit the models used (a dedicated fit run; cache it).
3. Give each user their `onboarding_picks` earliest liked items as onboarding picks; truth = their remaining relevant interactions.
4. Evaluate the cold path of `litsense_core` (onboarding profile only, bucket `cold`) against popularity and content-only.

### 8.6 `bootstrap.py`

- Paired bootstrap over users, `bootstrap_resamples` resamples, seeded. For models A and B on the same users: mean of (A − B), 95% percentile CI, and the share of resamples ≤ 0.
- Single-model 95% CIs for every reported mean.

### 8.7 `report.py`

Merge `<scorer>.json` files into `results/<v>/<mode>/table_main.csv`, `table_strata.csv`, `table_longtail.csv`, `table_coldstart.csv`, `bootstrap.csv`. Figures: grouped bars (NDCG@10, Recall@20 with CIs), NDCG@10 by sparsity bucket (line per model), coverage vs NDCG scatter. Also write `results/<v>/metrics.json` in the admin schema (§15.6).

### 8.8 Test-run lock

`python -m ml.stages run eval.test --confirm` is the only path to test mode. It writes `results/<v>/test/LOCKED` with a timestamp and the git SHA. If `LOCKED` exists it refuses to run unless `--i-know-this-reuses-test` is passed, and then it appends a warning to decisions.md. Ask the user before ever passing that flag.

**Gate (end of Week 4):** metric unit tests pass (§16); random and popularity evaluate end to end in val mode with strata and CIs.

## 9. Phase 2 — Baselines, LightGCN, projection hybrid (`ml/models/`)

Every model implements the `Scorer` protocol (§8.3), is fit once per mode (`val`: train; `test`: train ∪ val), and writes artifacts to `artifacts/<v>/eval_<mode>/<model>/` with a manifest whose `status` becomes `complete` only after success.

### 9.1 Random and popularity (`random_pop.py`)

- **Random:** seeded uniform scores (a sanity floor).
- **Popularity:** score = fit-window interaction count. Also compute, for later features and rails: Bayesian mean rating `(v·R + m·C) / (v + m)` with m = `bayes_m` and C = global mean; and trending = Σ exp(−Δdays / half-life) over fit interactions, Δ measured from the cutoff.

### 9.2 Content-only (`content.py`)

Needs item content embeddings (§10.1); build those early in Week 4. User profile = rating-weighted mean of content vectors of the user's fit-window items rated ≥ 4 (all fit items if none), L2-normalized. Score = profile · item vector. Never use a zero user vector (§2.2 #7).

### 9.3 ItemKNN and ALS

- **ItemKNN (`itemknn.py`):** binary user–item CSR matrix; item–item cosine with shrinkage `sim = dot / (norm_i·norm_j + shrink)`; keep top-`k` neighbours per item (sparse); score = user row × similarity matrix.
- **ALS (`als.py`):** `implicit.als.AlternatingLeastSquares` on a confidence matrix 1 + α·binary. The `implicit` API changed across versions (user×item vs item×user input to `fit`); check the installed version and assert output shapes.

### 9.4 LightGCN (`lightgcn.py`)

Port the model from the old notebook (§2.1) with these changes:

- **Graph:** fit-window interactions as implicit edges; adjacency built with scipy, symmetric-normalized, converted with `torch.sparse_coo_tensor(indices, values, shape).coalesce()` (not the deprecated `torch.sparse.FloatTensor`). Zero-degree nodes get zero rows (no divide-by-zero).
- **Model:** user and item `nn.Embedding`, Xavier-uniform init; `forward` propagates K = `layers` times with `torch.sparse.mm` and returns the mean over layers 0..K. Dim, layers, lr, reg, negatives all from config.
- **Loss:** BPR with `n_negs` negatives per positive (mean over negatives) + reg on ego embeddings of the batch / batch size.
- **Vectorized negative sampling** (no per-sample Python loops):

```python
def sample_negatives(users, n_items, pos_keys, n_negs, rng, rounds=10):
    # pos_keys: sorted int64 array of u * n_items + i for every fit positive
    neg = rng.integers(0, n_items, size=(len(users), n_negs))
    for _ in range(rounds):
        keys = users[:, None].astype(np.int64) * n_items + neg
        idx = np.clip(np.searchsorted(pos_keys, keys), 0, len(pos_keys) - 1)
        bad = pos_keys[idx] == keys
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg
```

- **Early stopping:** carve an inner validation set from the fit window (last `inner_val_fraction` of each user's fit interactions, users with ≥ `inner_val_min_user`). Train on the rest; every `eval_every` epochs compute inner NDCG@20 with the shared evaluator code; stop after `patience` evaluations without improvement. Then **refit on the full fit window for the best epoch count** and save that model. Log loss, metric and time per evaluation.
- **Outputs:** `lgcn_user.npy` (n\_users, dim) float32, `lgcn_item.npy` (n\_items, dim) float32, `lgcn_item_norm.npy` (row-L2-normalized), `item_degree.npy`. Zero-degree items are flagged.
- **Two scorers:** `LightGCNRaw` (u · i) and `LightGCNNorm` (u · i/‖i‖). Zero-degree items score −inf in both.
- **Sanity test:** on the tiny fixture, training NDCG must beat random after a few epochs.

### 9.5 Projection hybrid and controls (`projection.py`)

- **Frozen inputs:** LightGCN user/item embeddings (§9.4) and item content embeddings c\_i (384-d MiniLM, §10.1).
- **Model:** learn W ∈ R^(dim×384) with no bias. Item vector e'\_i = e\_gcn\_i + W c\_i, where e\_gcn\_i is set to **zero for zero-degree items** (so cold items are represented purely by W c\_i — their only route into CF space).
- **Scorer consistency:** if `normalize_items` is true, both the BPR training score and evaluation use u · e'\_i / ‖e'\_i‖. Never train with one scorer and evaluate with another (§2.2 #1).
- **Training:** BPR over a precomputed array of fit positive pairs; each step samples a batch of pair indices and negatives with `sample_negatives`; Adam; `epochs × steps_per_epoch` from config; seeded. Save `W.npy` and `proj_item.npy` (the normalized e'\_i matrix) only after completion.
- **Controls that must be run and reported:**
  - `ProjectionShuffled`: identical training, but rows of c permuted across items (seeded). If it matches the real projection, the gain is not semantic.
  - `LightGCNNorm` (§9.4).
  - `StaticFusion(α)`: α·normalize(e\_gcn) + (1−α)·normalize(W\_pca c) for α in `static_fusion_alphas`, where W\_pca is a 384→64 PCA fit on content vectors; tuned on val only.
- **Report rows** (val now, test in Week 13): Raw LightGCN, Normalized LightGCN, Projection, Projection-shuffled, best Static fusion — all-item, long-tail and cold-item strata, with paired bootstrap against both LightGCN variants.

**Gate (end of Week 5):** val results exist for random, popularity, ItemKNN, ALS, content-only, LightGCN raw and normalized, projection, projection-shuffled, with strata and CIs. LightGCN should beat ALS on val; if it does not, investigate (tuning, graph construction, evaluation) before moving on and log the outcome.

## 10. Phase 3 — Content embeddings, FAISS, ONNX (`ml/nlp/embed.py`, `ml/nlp/onnx_export.py`, `litsense_core/encoder.py`)

One model, `all-MiniLM-L6-v2`, embeds items, review sentences, onboarding text, chat queries and spoiler-classifier inputs. Offline it runs in PyTorch on the T4; online it runs through ONNX Runtime with identical outputs.

### 10.1 Item content embeddings (stage `nlp.content`)

- **Text per book:** `"{title}. By {author}. Genres: {genre labels}. {description}"`. Build the header first, then fill the remaining token budget (256 word pieces) with the description, truncating by tokens (use the model's tokenizer to count), so title, author and genres are never cut.
- **No review text and no LLM summaries** in item content embeddings (leakage, §1.5). Review semantics live in separate review centroids (§11.6). Because metadata does not depend on time, one content matrix serves all three artifact sets.
- Encode with `SentenceTransformer(model).encode(texts, batch_size=256, normalize_embeddings=True, convert_to_numpy=True)`, fp16 autocast on GPU. Output `content_item.npy` (n\_items, 384) float32.
- **Sanity checks** (logged to `results/<v>/content_checks.json`): mean cosine over 2,000 random item pairs (warn if > `anisotropy_warn`); `content_nn_samples.txt` with 20 random books and their 5 nearest titles for a human look.
- **Neighbours:** `item_nn.npy` (n\_items, 50) int32, top-50 content neighbours excluding self (used by the item-item candidate source).

### 10.2 Vector index (`litsense_core/vector_index.py`)

`VectorIndex(matrix)` wraps FAISS `IndexFlatIP` over L2-normalized float32 vectors, and falls back to `matrix @ q` with `argpartition` when faiss is not importable (tests run without faiss). Methods: `search(q, k, exclude=None, allow=None)` returning (ids, scores). Save the FAISS index as `content.faiss`. At \~20K × 384, exact search costs a few milliseconds.

### 10.3 ONNX export (stage `nlp.onnx`)

- Export the transformer with dynamic batch and sequence axes, either `optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 --task feature-extraction <dir>` or `torch.onnx.export` on the underlying `AutoModel` (output: `last_hidden_state`). Save `tokenizer.json` (HF fast tokenizer).
- Quantize with `onnxruntime.quantization.quantize_dynamic(..., weight_type=QuantType.QInt8)` → `minilm.int8.onnx`.
- **Parity test** on 200 varied texts: fp32 ONNX vs sentence-transformers cosine ≥ 0.9999 for every text; int8 mean cosine ≥ 0.99 and min ≥ 0.98. If int8 fails, ship fp32 and set `serving.onnx_quantized: false`.
- **Latency check:** one short query with one thread should take under 50 ms on a laptop CPU; log the number.

### 10.4 Runtime encoder (`litsense_core/encoder.py`)

```python
import numpy as np, onnxruntime as ort
from tokenizers import Tokenizer

class MiniLMEncoder:
    def __init__(self, onnx_path: str, tokenizer_path: str, max_len: int = 256, threads: int = 1):
        self.tok = Tokenizer.from_file(tokenizer_path)
        self.tok.enable_truncation(max_length=max_len)
        self.tok.enable_padding(pad_id=0, pad_token="[PAD]")
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str]) -> np.ndarray:
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feeds)[0]                 # (B, T, 384)
        m = mask[..., None].astype(np.float32)
        v = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
```

The offline `ml/nlp/embed.py` exposes the same `encode()` signature backed by sentence-transformers, so callers can switch implementations in tests.

## 11. Phase 3 — Review NLP (`ml/nlp/`)

This turns reviews into: a sentence table with embeddings, per-sentence aspect hits and polarity, spoiler flags, item and user aspect profiles, three bipolar descriptors per item, review centroids, and spoiler-safe evidence snippets. Everything derived from review text is built three times, once per cutoff (`eval_val` < t1, `eval_test` < t2, `serve` = all), by functions that take `cutoff` and raise `LeakageError` on newer rows.

### 11.1 Sentence table (stage `nlp.sentences`)

- Input: `reviews.parquet` rows with `text_ok`. Split with spaCy `blank("en")` + `sentencizer` via `nlp.pipe(texts, batch_size=1000, n_process=nlp.spacy_n_process)`.
- Keep sentences with 4–80 whitespace tokens. Columns: `sent_id` int64, `review_row`, `u`, `i`, `ts`, `rating`, `helpful_vote`, `pos` (index in review), `n_in_review`, `text`.
- Embed every sentence with MiniLM (fp16 on GPU, batch 512) into a float16 memmap `sent_emb.f16.npy` (n\_sent, 384). Checkpoint every 200K sentences. Expect a few million sentences at v1.
- The sentence table is built once over all reviews; cutoffs are applied by filtering on `ts` when aggregating.

### 11.2 Aspect taxonomy, lexicons, prototypes (`aspect_seeds.py`)

The order below is the vector index everywhere (0–11). Lexicon terms match with word boundaries, case-insensitive; prototypes are embedded once and stored as `aspect_prototypes.npy` (12, 8, 384).

| # | Aspect | UI label | Lexicon (abbreviated; extend during validation) |
| --- | --- | --- | --- |
| 0 | plot | plot | plot, story, storyline, narrative, twists?, subplots?, premise, mystery |
| 1 | writing\_style | prose | prose, writing, written, style, voice, sentences, language, lyrical, beautifully written |
| 2 | characters | characters | characters?, protagonist, heroine, hero, villain, main character, cast, relatable, likable, character development |
| 3 | world\_building | world-building | world-?building, world, setting, universe, lore, magic system, atmosphere, immersive |
| 4 | pacing | pacing | pacing, pace, slow, slow-burn, dragged, drags, page-turner, fast-paced, couldn't put it down, tedious, rushed |
| 5 | editing\_quality | editing | typos?, editing, edited, proofreading, grammar, grammatical errors, formatting, translation, translated |
| 6 | dialogue | dialogue | dialogue, dialog, conversations?, banter, exchanges |
| 7 | humor | humor | funny, humou?r, hilarious, witty, laugh(ed)?, comedic, jokes, sarcastic |
| 8 | emotional\_impact | emotional punch | moving, cried, tears, emotional, heartbreaking, heartwarming, touching, gut-wrenching |
| 9 | ending | ending | ending, conclusion, finale, resolution, cliffhanger, epilogue, last chapters? |
| 10 | complexity | depth | confusing, complex, complicated, convoluted, dense, hard to follow, layered, intricate, easy read |
| 11 | tone\_dark | tone | dark, bleak, grim, disturbing, brutal, violent, gritty, depressing, harrowing, grimdark, cozy |

```python
PROTOTYPES = {
 "plot": ["The plot kept me guessing until the very last page.", "The storyline was predictable and full of holes.",
          "What a clever, intricate story.", "The premise was great but the story went nowhere.",
          "The twists were genuinely surprising.", "Too many subplots that never connected.",
          "The mystery at the heart of the book was well constructed.", "Nothing much happens in this story."],
 "writing_style": ["The prose is gorgeous and lyrical.", "The writing was clunky and amateurish.",
          "Beautifully written, every sentence is a pleasure to read.", "The author's voice is distinctive and sharp.",
          "The writing style felt flat and repetitive.", "Her sentences are precise and elegant.",
          "It reads like a first draft.", "The language is rich without being overwrought."],
 "characters": ["I fell in love with the characters.", "The main character was annoying and hard to root for.",
          "The characters felt flat and one-dimensional.", "Great character development throughout the series.",
          "The protagonist is complex and believable.", "I couldn't connect with any of the characters.",
          "The villain was genuinely menacing.", "The friendship between the two leads was wonderful."],
 "world_building": ["The world-building is detailed and immersive.", "The magic system was original and well thought out.",
          "The setting came alive on the page.", "I never understood how this world worked.",
          "The atmosphere of the small town was perfectly captured.", "The lore was rich and layered.",
          "The world felt thin and underdeveloped.", "Vivid descriptions made me feel I was there."],
 "pacing": ["The pacing was perfect, I couldn't put it down.", "The middle dragged on forever.",
          "A real page-turner from start to finish.", "It took two hundred pages before anything happened.",
          "The ending felt rushed compared to the slow start.", "The book moves at a breakneck pace.",
          "It was a slow burn but worth it.", "I found it tedious and kept skimming."],
 "editing_quality": ["The book is full of typos and grammatical errors.", "It desperately needed a good editor.",
          "The translation reads smoothly.", "The formatting was a mess with missing words.",
          "Well edited and cleanly produced.", "So many spelling mistakes it pulled me out of the story.",
          "The translation felt awkward and stilted.", "Clearly no one proofread this."],
 "dialogue": ["The dialogue is sharp and witty.", "The conversations felt stilted and unnatural.",
          "I loved the banter between the two leads.", "Nobody talks like that in real life.",
          "The dialogue crackles with energy.", "Too much of the book is long expository conversation.",
          "Their exchanges were funny and believable.", "The dialogue was cheesy."],
 "humor": ["This book made me laugh out loud.", "The humor is dry and clever.", "The jokes fell flat for me.",
          "Hilarious from start to finish.", "A witty, sarcastic narrator.", "It tried too hard to be funny.",
          "There's a lovely thread of humor throughout.", "I chuckled on almost every page."],
 "emotional_impact": ["I cried my eyes out.", "A deeply moving and emotional read.", "It left me feeling nothing at all.",
          "Heartwarming and hopeful.", "This book broke my heart.", "It stayed with me for days after I finished.",
          "Emotionally manipulative and shallow.", "A touching story about family and loss."],
 "ending": ["The ending was perfect.", "The ending felt rushed and unsatisfying.", "I hated the cliffhanger.",
          "The conclusion tied everything together beautifully.", "The last few chapters were a letdown.",
          "What a satisfying finale to the series.", "The ending came out of nowhere.", "The epilogue was the cherry on top."],
 "complexity": ["It was confusing and hard to follow.", "A complex, layered novel that rewards attention.",
          "Too many characters to keep track of.", "An easy, light read.", "The plot was convoluted.",
          "Dense but worth the effort.", "I had to reread passages to understand what was happening.",
          "Intricate without ever being confusing."],
 "tone_dark": ["This is a dark and disturbing book.", "Very violent and graphic in places.", "It's grim and bleak throughout.",
          "A cozy, gentle story.", "Too depressing for me.", "Gritty and unflinching.", "Lighthearted and uplifting.",
          "Harrowing but powerful."],
}

# Bipolar descriptors: intensity, not sentiment. Computed only on sentences hitting the linked aspect.
DESCRIPTOR_POLES = {
 "darkness": {"aspect": "tone_dark",
     "plus":  ["This is a dark and disturbing book.", "Grim, bleak and violent.", "Gritty and unflinching.", "Harrowing and brutal at times."],
     "minus": ["A cozy, gentle story.", "Lighthearted and uplifting.", "Sweet and comforting.", "A feel-good read."]},
 "pace": {"aspect": "pacing",
     "plus":  ["A fast-paced page-turner.", "Action from the first page.", "It moves at a breakneck pace.", "I read it in one sitting."],
     "minus": ["A slow, meditative read.", "It's a slow burn.", "Very little happens for long stretches.", "It takes its time."]},
 "complexity_level": {"aspect": "complexity",
     "plus":  ["Dense and complex.", "Layered and intricate, it demands attention.", "Hard to follow at times.", "A challenging literary read."],
     "minus": ["An easy, light read.", "Simple and straightforward.", "A quick, breezy read.", "Perfect beach read."]},
}
```

Why descriptors: aspect polarity says whether reviewers *liked* the darkness or pacing; descriptors say *how dark / fast / dense* the book is. Chat modifiers ("cozier", "faster", "lighter") and onboarding ("not into grimdark") act on descriptors.

### 11.3 Aspect detection (`aspects.py`, stage `nlp.aspects`)

```python
proto = load_prototypes()                                      # (12, 8, 384), normalized
for chunk in iter_chunks(sent_emb, 200_000):
    sim = np.einsum("nd,apd->nap", chunk.astype(np.float32), proto).max(axis=2)   # (n, 12) max over seeds
    lex = lexicon_hits(texts_of(chunk))                        # (n, 12) bool, compiled regex per aspect
    hit = (sim >= tau_vector) | lex                            # tau per aspect from config
    store(bitmask=pack(hit), top_sim=sim.max(1))
```

Store per sentence: a uint16 bitmask of aspect hits and the max similarity. Descriptor score per sentence (only where the linked aspect hits): cos(mean plus-pole) − cos(mean minus-pole).

### 11.4 Sentence polarity (`sentiment.py`, stage `nlp.sentiment`)

- Run DistilBERT-SST2 (`AutoModelForSequenceClassification`, fp16, batch 256, max\_length 128) only on sentences with ≥ 1 aspect hit. Polarity = p\_pos − p\_neg ∈ \[−1, 1\].
- **Clause splitting:** when a sentence hits ≥ 2 aspects, split on `clause_split_markers`, re-run detection on each clause, and score each clause separately. "Wonderful characters, but the middle third really dragged." must give characters > 0 and pacing < 0.
- Mention statistics use `pos_threshold` / `neg_threshold` (polarity > 0.5 positive, < −0.5 negative).

### 11.5 Validation (`annotate.py`, stage `nlp.validate_aspects`)

1. Export `docs/annotation/aspects_300.csv`: 300 random `text_ok` sentences (seeded), 200 with ≥ 1 predicted hit and 100 without. Columns: `sent_id, text`, one blank column per aspect (values: +, −, 0 for mentioned-neutral, blank for absent).
2. Write `docs/annotation_guides/aspects.md`: definition of each aspect, two examples each, edge cases (ending vs emotional\_impact, pacing vs plot).
3. The user and teammate label; 100 sentences are labelled by both for Cohen's κ per aspect.
4. The stage reads the labelled CSV, tunes τ per aspect on 150 sentences (maximize F1), and reports precision/recall/F1 per aspect and macro on the other 150 → `results/<v>/aspect_validation.json`. It prints the proposed `aspects.tau` values; the user confirms before they go into config.
5. Target macro-F1 ≥ 0.6. Any aspect below 0.4 F1 is reported; ask the user whether to expand its lexicon, merge it, or drop it.

### 11.6 Review centroids (stage `nlp.centroids`, per cutoff)

- `review_centroid_item[i]` = normalized mean of sentence embeddings from reviews of item i with rating ≥ 4 and ts < cutoff; zero vector if none.
- `review_centroid_user[u]` = normalized mean over the user's own review sentences (rating ≥ 4, ts < cutoff). Offline only (corpus users).

### 11.7 Spoiler classifier (`spoiler_train.py` offline, `litsense_core/spoiler.py` runtime)

1. Load `goodreads_reviews_spoiler.json.gz`; verify that `review_sentences` is a list of \[label, sentence\].
2. Expand to sentence rows with `book_id`. Sample `spoiler.sample_size` sentences, balanced 50/50 between spoiler and non-spoiler. Hold out `holdout_fraction` **by book\_id** (group split) so books do not leak.
3. Embed with the same MiniLM model. Train `LogisticRegression(class_weight="balanced", max_iter=2000)` over `C_grid`, choosing C by held-out AUC.
4. Pick the probability threshold as the largest value that still gives spoiler recall ≥ `target_recall` on the held-out set. Over-flagging only costs snippets. Report precision, recall, F1, AUC → `results/<v>/spoiler_metrics.json`.
5. Final score = classifier probability + `regex_boost` if any `spoiler.regex_patterns` matches + `late_position_boost` if the sentence sits in the last 30% of its review (spoilers cluster late). Flag if score ≥ threshold.
6. Apply to every Amazon sentence: `spoiler_p`, `spoiler_flag`. Report the flagged fraction.
7. Save `spoiler_lr.npz` (coef (384,), intercept, threshold, regex list, boosts). `litsense_core/spoiler.py` reimplements scoring in numpy over ONNX embeddings and is used on LLM outputs and snippets at runtime.

### 11.8 Profiles (`profiles_offline.py` + shared `litsense_core/profile.py`)

**Item aspect profile (per cutoff):**

- For item i and aspect a: mentioning sentences M\_ia, weight w = 1 + log1p(helpful\_vote).
- raw\_ia = Σ w·polarity / Σ w; n\_ia = number of distinct reviews mentioning a.
- shrunk\_ia = (n\_ia·raw\_ia + k·μ\_a) / (n\_ia + k), with μ\_a the global mean polarity of aspect a and k = `shrinkage_k`.
- mention\_rate\_ia = n\_ia / n\_reviews\_i; pos\_rate\_ia, neg\_rate\_ia = shares of mentioning reviews whose mean polarity is > 0.5 / < −0.5.
- descriptors\_i (darkness, pace, complexity\_level) = shrunk mean of sentence descriptor scores (shrink toward 0), clipped to \[−1, 1\].

**User profile** — one shared function `build_profile(events, store, review_aspects=None, onboarding=None, overrides=None, now=cutoff)` in `litsense_core/profile.py`, used offline for corpus users and online for app users:

- Rating weight r\_w from `aspects.rating_weights` (5 → 1.0 … 1 → −1.0); time decay d = 0.5^(age\_days / `user_half_life_days`).
- Implicit aspect vector = Σ r\_w·d·(shrunk\_i − μ) / Σ |r\_w|·d over rated items.
- Explicit aspect vector (if the user's own review sentences are available) = decay-weighted mean polarity per aspect; confidence\_a = n\_a / (n\_a + 3).
- aspect\_vec = `user_explicit_weight` · explicit + (1 − w) · implicit where explicit exists, else implicit. Onboarding positive/negative aspects add ±0.5; pinned aspects from overrides add +0.5. Clip to \[−1, 1\].
- descriptor\_pref = Σ r\_w·d·descriptors\_i / Σ |r\_w|·d (+ onboarding descriptors).
- favorite\_authors: authors with ≥ 1 rating ≥ 4; affinity = Σ r\_w·d over their books, scaled to \[0, 1\]; keep top 20.
- preferred/disliked genres: same scheme over genre labels; disliked if affinity < −0.3; plus onboarding and muted genres.
- series\_progress: series\_id → max series\_pos among items the user rated or marked read.
- recent variants (last 10 interactions) of the semantic and aspect vectors.

**Outputs per cutoff:** `item_aspects.npy`, `item_mention_rate.npy`, `item_pos_rate.npy`, `item_neg_rate.npy` (n\_items, 12); `item_descriptors.npy` (n\_items, 3); for corpus users offline only: `user_aspects.npy`, `user_aspect_conf.npy`, `user_review_aspects.parquet` (per-user explicit aspect polarities, consumed by ranker training).

### 11.9 Evidence snippets (`snippets.py`, serve cutoff only)

Per item and aspect: up to 3 positive and 2 negative sentences with `spoiler_flag == False`, 40–220 characters, ranked by helpful votes then |polarity|. Store text, review\_row and helpful votes in `evidence_snippets.json.gz`. These are the only review sentences the product ever displays.

**Gate (end of Week 7):** aspect validation table with the user-confirmed τ; spoiler metrics; all per-cutoff artifacts with manifests; aspect EDA figures (mention frequency per aspect; mean polarity by genre).

## 12. Phase 4 — Serving package `litsense_core` and the MVP recommender

By Week 8 `litsense_core` must serve real recommendations with template reasons, using a blend ranker; the LightGBM ranker plugs in later through the same interface. The package is stateless: the backend passes a `UserState` in and persists whatever comes back.

### 12.1 `store.py` — ArtifactStore

- `load_store(bundle_dir) -> ArtifactStore`, a process-wide singleton. Reads `manifest.json`, optionally verifies sha256s, memory-maps large `.npy` files (`np.load(..., mmap_mode="r")`), and casts small arrays to float32 once. Scoring casts fp16 slices to float32 before dot products.
- Holds: catalog arrays and text, all embedding matrices, aspects, descriptors, indices (author, series, genre, aspect top-items), evidence snippets, `VectorIndex`, the LightGBM `Booster` + `feature_list`, the ONNX encoder, the spoiler scorer, `corpus_users` (demo users with exact LightGCN vectors), and the serving config slice.
- `health()` returns bundle version, cutoff, data version, RSS in MB, whether the ranker and LLM are available.
- Fails loudly at load if shapes disagree with the manifest or `feature_list` disagrees with `features.FEATURES`.

### 12.2 `profile.py` — online profile from `UserState`

`build_profile(state, store) -> Profile`, recomputed on every call (no incremental state). Uses the shared function from §11.8 plus:

- `n` = number of ratings + 0.5 × reading-list adds; bucket from `ranking.buckets`.
- `cf_vec`: exact LightGCN vector if `state.corpus_user_id` is in `corpus_users` (`cf_available = 1`); else the r\_w-weighted mean of `lgcn_item` over items rated ≥ 4 plus reading-list items at weight 0.5 (`cf_available = 0.5`, "pseudo"); else none (`0`).
- `semantic_vec`: weighted mean of content vectors of liked items; when onboarding text exists, blend 0.6 × picks + 0.4 × encoded text. `onboarding_vec` is kept separately for the `onboarding_sim` feature.
- `semantic_recent`, `aspect_vec`, `aspect_conf`, `aspect_recent`, `descriptor_pref`, `favorite_authors`, `preferred_genres`, `disliked_genres`, `series_progress` as in §11.8; explicit aspects come from `state.review_analyses` (reviews the user submitted in the app).
- `seen` = rated ∪ reading list with status read/dropped ∪ dismissed.

### 12.3 `rank.py` — blend ranker (MVP, and permanent fallback)

For each candidate, per-source rank scores r = 1 − rank / len(source list) (0 if absent). With α(n) = n / (n + `alpha_k`):

```text
score = α(n)·r_cf + (1 − α(n))·r_semantic + w_pop·r_popularity + w_aspect·aspect_match
```

Used when `use_ranker=False`, when the ranker is missing, or if ranker scoring raises (then `degraded = True`).

### 12.4 Cold-start paths

| Situation | Behaviour |
| --- | --- |
| No ratings, no onboarding | Trending in preferred genres (or global trending); reason "Popular with readers right now" |
| Onboarding only | Bucket `cold`: semantic from picks and/or text, pseudo-CF from picks, popularity filtered by onboarding genres, aspect source from onboarding aspects |
| 1–4 ratings | Bucket `warming` (pseudo-CF enters with low weight) |
| Picked/liked a book with a known series position | If the next volume is in the catalogue and unseen, return it in `continue_series`, never in the main feed |

### 12.5 `reasons.py` — evidence and templates

`build_evidence(profile, book_id, store, signals) -> Evidence` collects facts only (schema in §15). Reason chips are chosen in this priority order; the first whose condition holds wins:

| # | Condition | Template |
| --- | --- | --- |
| 1 | Author in favorites | "Because you loved {author}'s {liked\_title}" |
| 2 | Some aspect a with user aspect > 0.3 and item shrunk aspect > 0.3 | "Readers praise its {label} ({pos\_pct}% of mentions) — something you value" |
| 3 | max item-item similarity ≥ 0.6 to a liked book | "Similar to {liked\_title}, which you rated {stars}★" |
| 4 | Onboarding theme matches the description embedding | "Matches what you described: {theme}" |
| 5 | Came from the long-tail source | "A lesser-known pick that fits your taste" |
| 6 | Fallback | "Popular with {genre} readers right now" |

The optional snippet is a stored safe snippet (§11.9) for the chosen aspect. Every number in a template comes from evidence fields.

### 12.6 `recommend.py` — flow

```text
recommend(state, k, filters, exclude, use_ranker):
  profile = build_profile(state, store)
  cands   = candidates.generate(profile, store, quotas[bucket])        # s13.1 (MVP: cf, semantic, item_item, popularity)
  cands   = filters.apply(cands, profile, filters, exclude)           # s13.2
  X       = features.assemble(profile, cands, store)                  # s13.3 (MVP: subset used by blend)
  scores  = ranker.score(X) if use_ranker and store.ranker else blend(cands, profile)
  scores  = apply_overrides(scores, profile)
  top     = postprocess(cands, scores, profile, store, k)             # series guard, MMR, genre cap, explore slot
  return RecResponse(items=[RecItem(..., reason=template_reason(...), evidence=...)],
                     continue_series=..., model_version=store.version, latency_ms=...)
```

Other public functions built in this phase: `home_rails` (§15), `similar_books` (content neighbours re-ranked by the user's blend score if a state is given), `more_by_author`, `semantic_search` (encode query → `VectorIndex` → filters), `onboarding_grid` (60 popular, high-rating, genre-diverse books: top items per primary genre by Bayesian rating with ≥ 50 ratings, round-robin), `book_insights`, `analyze_user_review` (sentence split by regex, ONNX embed, prototype match with τ, VADER compound as polarity, spoiler count).

### 12.7 Bundle export (stage `serve.export`)

Collect `artifacts/<v>/serve/*`, cast embeddings to float16, build the indices in §4.6, copy the ONNX model and tokenizer, write `manifest.json` (versions, cutoff = max ts, sha256, shapes, config slice), and tar to `artifacts/serve_bundle_<v>_<date>.tar.gz`. Print the bundle size and the RSS measured by loading it in a subprocess. Provide `litsense_core.download_bundle(url, dest)` for the backend to fetch it from a GitHub Release at boot.

### 12.8 `dev/dev_server.py`

A minimal FastAPI app exposing `/recommend`, `/rails`, `/explain`, `/chat`, `/search`, `/health` that builds a `UserState` from a JSON body. For local testing and for the teammate to see example payloads. Not deployed.

**Gate (end of Week 8):** the PRD §37.4 MVP acceptance tests pass on the deployed URL, from a phone, with the LLM disabled. Measured RSS ≤ `serving.max_rss_mb`.

## 13. Phase 5 — Candidates, features, LambdaRank, post-processing, ablation

The final recommender retrieves \~200 candidates from seven sources and reranks them with LightGBM LambdaRank over 27 features plus 7 source flags. Candidate generation and features live in `litsense_core` and are imported by `ml/ranking/`.

### 13.1 `candidates.py` — seven sources

| Source | Method | Notes |
| --- | --- | --- |
| `cf` | top by `cf_vec · lgcn_item` (raw) with `argpartition` | Skipped if no cf\_vec |
| `semantic` | `VectorIndex.search(semantic_vec)`; split the quota with `onboarding_vec` when present | Main cold-start source |
| `item_item` | for each of the last 3 liked items (or chat anchors), its `item_nn` rows; take the top 15 each | Precomputed, no runtime search |
| `popularity` | `trending_top` filtered to preferred genres (all genres if none) | Safety net |
| `author` | all items by favorite authors (or chat anchor authors), ordered by affinity then popularity | Skipped if no favorites |
| `aspect` | `aspect_topitems` lists for the user's top-2 positive aspects, intersected with preferred genres when possible | Explanation-friendly |
| `longtail` | top by `cf_vec · proj_item` restricted to `is_longtail`; with no cf\_vec, top by `semantic_vec · content` restricted to tail | Hidden Gems |

Quotas come from `ranking.quotas[bucket]`. Union with provenance flags, dedupe, then cap at `candidates_total` by round-robin over sources in quota order. `aspect_topitems`: for each aspect, the top 300 items by shrunk aspect value among items with ≥ 5 mentioning reviews.

**Offline diagnostic (`ml/ranking/candidate_recall.py`):** on val, report recall@200 of the union and of each source, plus each source's unique contribution (positives found only by that source). If a source contributes almost nothing, report it and ask the user before dropping it.

### 13.2 Filters before ranking

Remove: `profile.seen`; dismissed; session exclusions; hard filters (`min_year`, `max_year`, `min_pages`, `max_pages`, `genres_include`, `genres_exclude`); muted genres; items with fewer than `min_reviews_main_feed` reviews (except in the longtail source and search). **Series guard:** an item with `series_pos = p > 1` passes only if `series_progress[series_id] ≥ p − 1`; the exact next volume goes to `continue_series`, not the main feed. Items without series data always pass.

### 13.3 `features.py` — the feature table

All features are computed as vectorized numpy over the candidate array (target < 15 ms for 200 candidates). Missing values are `np.nan` (LightGBM handles them). The column order is `features.FEATURES` and is saved as `feature_list.json` with the ranker.

| # | Name | Definition | Group |
| --- | --- | --- | --- |
| 1 | cf\_score | cf\_vec · lgcn\_item\_i (NaN if no cf\_vec) | CF |
| 2 | cf\_norm\_score | cf\_vec · lgcn\_item\_norm\_i | CF |
| 3 | cf\_rank\_pct | percentile of cf\_score among candidates | CF |
| 4 | cf\_available | 0 none, 0.5 pseudo, 1 exact | CF |
| 5 | proj\_hybrid\_score | cf\_vec · proj\_item\_i | PROJ |
| 6 | semantic\_sim | cos(semantic\_vec, content\_i) | CONTENT |
| 7 | semantic\_recent\_sim | cos(semantic\_recent, content\_i) | CONTENT |
| 8 | onboarding\_sim | cos(onboarding\_vec, content\_i); NaN if none | CONTENT |
| 9 | max\_item\_sim | max cos(content\_i, content\_j) over liked j | CONTENT |
| 10 | top5\_item\_sim | mean of the 5 largest such similarities | CONTENT |
| 11 | aspect\_match | confidence-weighted cosine between aspect\_vec and item shrunk aspects | ASPECT |
| 12 | aspect\_match\_recent | same with the recent aspect vector | ASPECT |
| 13 | aspect\_conflict | Σ\_a conf\_a · max(0, −aspect\_vec\_a · item\_a) | ASPECT |
| 14 | descriptor\_match | − L2 distance between descriptor\_pref and item descriptors | ASPECT |
| 15 | user\_aspect\_confidence | mean of conf\_a | ASPECT |
| 16 | review\_centroid\_sim | cos(user review centroid, item review centroid); NaN if either missing | REVIEW\_EMB |
| 17 | genre\_overlap | Σ preferred-genre weights over the item's genres | META |
| 18 | disliked\_genre\_hit | 1 if any item genre is disliked or muted | META |
| 19 | decade\_match | share of the user's liked items from the item's decade | META |
| 20 | author\_affinity | affinity of the item's author if a favorite, else 0 | AUTHOR |
| 21 | item\_bayes\_rating | Bayesian mean rating (cutoff) | META |
| 22 | item\_count\_log | log1p(fit-window interactions) | META |
| 23 | item\_rating\_std | rating std (cutoff) | META |
| 24 | item\_age\_years | cutoff year − publication year; NaN if unknown | META |
| 25 | user\_activity\_log | log1p(n) | USER |
| 26 | user\_mean\_rating | mean of the user's ratings | USER |
| 27 | n\_sources | number of sources that produced the item | SOURCE |
| 28–34 | src\_cf … src\_longtail | one binary flag per source | SOURCE |

`assemble(profile, cands, store)` asserts `store.cutoff` equals the cutoff the caller expects (leakage guard for offline use).

### 13.4 Ranker training data (`ml/ranking/build_training.py`)

1. Load the `eval_val` store (models fit on train, profiles at t1).
2. Users: ≥ 1 val interaction and ≥ 1 train interaction. Include sparse users (1–4 train interactions) on purpose; they matter for the strata.
3. For each user: build a `UserState` from train interactions (ratings with timestamps); attach their train-window explicit aspect polarities from `user_review_aspects.parquet` as `review_analyses`; then profile → candidates → filters → features, all via `litsense_core`.
4. Label = 1 if the candidate is a val interaction with rating ≥ 4; else 0 (val interactions rated ≤ 3 are hard negatives).
5. Log candidate recall (share of val positives present among candidates). Drop users with no positive among candidates.
6. Keep every positive and sample up to `negatives_per_positive` negatives per positive per user (seeded, uniform among that user's negatives).
7. Split users `ranker_user_split` / rest for early stopping (seeded). Save Parquet with `group_id`, sorted so each group is contiguous.

### 13.5 Training (`train_ranker.py`)

```python
import lightgbm as lgb
ranker = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, learning_rate=0.05,
                        num_leaves=31, min_child_samples=20, colsample_bytree=0.8,
                        subsample=0.8, subsample_freq=1, random_state=seed)
ranker.fit(X_tr, y_tr, group=g_tr, eval_set=[(X_va, y_va)], eval_group=[g_va],
           eval_at=[10], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(25)])
ranker.booster_.save_model("ranker.txt")
```

All parameters come from `ranking.lgbm`. Save `feature_list.json`, gain importances (CSV + bar chart) and the best iteration. Assert group sizes sum to the row count. If `author_affinity` or the ASPECT group ranks in the top 8 by gain, note it for the report.

### 13.6 Scoring and `postprocess.py`

1. `booster.predict(X)`.
2. Overrides: muted genre → score set to −inf; pinned aspect a → + `pinned_aspect_boost` × item\_a.
3. Series guard re-check (race safety).
4. **MMR** over the top `mmr_pool` items with content embeddings: repeatedly pick argmax of λ·ŝ − (1 − λ)·max\_sim\_to\_selected, where ŝ is the min-max-normalized score and λ = `mmr_lambda`.
5. **Genre cap:** at most `genre_cap` items with the same primary genre in any 10.
6. **Exploration slot:** position `explore_slot` gets the best longtail-source item not already listed (if any).
7. Return top K with scores, sources, evidence and template reasons.

### 13.7 Test-mode pipeline and ablation (`ablation.py`)

**Final protocol (Week 13, once):** refit LightGCN, ALS, ItemKNN, popularity and projection on train ∪ val (`eval_test` artifacts, profiles at t2); for test users build states from train ∪ val; generate candidates and features with the `eval_test` store; score with the ranker trained in §13.5 on val-window labels (the standard two-window setup; state it in the report); evaluate through the evaluator in test mode (non-candidates rank last).

Each ranker row below is retrained with a feature-group mask on the same candidates and labels:

| Id | Model | Features / notes |
| --- | --- | --- |
| M0 | Popularity | — |
| M1 | ALS (matrix factorization) | — |
| M2 | LightGCN, raw | — |
| M2n | LightGCN, normalized | — |
| M2p | Projection hybrid | — |
| M2s | Projection, shuffled text | control |
| M3 | Content-only | — |
| M4 | Ranker: CF + META + USER + SOURCE | does reranking alone help? |
| M5 | M4 + CONTENT + REVIEW\_EMB | dense text signal |
| M6 | M4 + ASPECT | **thesis row** (M6 vs M4, and M6 vs M2) |
| M7 | M4 + CONTENT + REVIEW\_EMB + ASPECT | all text |
| M8 | M7 + AUTHOR + PROJ + post-processing | full LitSense |
| M8a | M8 − AUTHOR | isolates author signal |
| M8b | M8 − ASPECT | aspect contribution in the full model |

For every row report: all-item metrics; sparsity strata; long-tail metrics (with paired bootstrap against both M2 and M2n); cold items; coverage, novelty, ILD; paired bootstrap against M2 and M4. Outputs: `results/<v>/test/ablation_main.csv`, `ablation_strata.csv`, `ablation_longtail.csv`, `bootstrap.csv`, plus figures (grouped bars with CIs; NDCG@10 by sparsity bucket).

**Supplementary ablations (val only, only if time allows):** remove one candidate source at a time; LightGCN layers {1, 2, 3, 4} × dims {32, 64, 128}; MMR λ ∈ {0.5, 0.75, 0.9, 1.0}; aspect τ ± 0.05; random vs temporal split (shows how much a random split inflates scores).

**Gate (end of Week 10):** the ablation on val shows M2 vs M4 vs M6 with CIs; the ranker beats the blend ranker on val NDCG@10 (if not, report it and keep the blend as default).

## 14. Phase 6 — GenAI layer

The LLM interprets and narrates; the recommender decides. Five bounded roles, each with a validator and a non-LLM fallback. The main feed makes zero LLM calls.

### 14.1 Roles

| Role | Trigger | Input | Output | Fallback |
| --- | --- | --- | --- | --- |
| Onboarding parse | User types a taste sentence | The sentence | `OnboardingProfile` JSON | Embed the raw text only |
| Intent parse | Chat message or natural-language search | Message + session summary | `Intent` JSON | Rule-based parser (§14.7) |
| Explanation | "Why this?" | Evidence object | ≤ 60-word text | Template reason |
| Chat reply | Each chat turn with results | Request + results with evidence | `ChatReplyLLM` JSON keyed by `book_id` | Template intro + template reasons |
| Review synthesis | Offline batch | Aspect stats + safe snippets | 2–3 sentences | Stats-based template |

### 14.2 `llm/client.py`

- `LLMClient(config, on_event=None)`; adapter by `llm.provider`: `gemini` (`google-genai` SDK), `openai_compat` (any OpenAI-compatible endpoint, including a local Ollama server), `fake` (tests: scripted outputs, can raise 429/timeouts).
- `json(prompt_id, variables, schema, role) -> BaseModel | None` and `text(prompt_id, variables, role) -> str | None`.
- Prompts load from `llm/prompts/<id>.md` (front matter: `id`, `version`, `role`), rendered with `string.Template` (`$var`, so JSON braces need no escaping).
- Use the provider's JSON mode / response schema when available; strip code fences; validate with Pydantic; one repair retry that sends the validation error back; otherwise `None`.
- Timeout `timeout_s`; up to `max_retries` retries with exponential backoff + jitter on 429/5xx; token-bucket limiter at `rpm_limit`; daily counter (file-backed, or a callback the backend provides) — over `daily_request_budget` returns `None`.
- Cache: sqlite keyed by sha256(prompt\_id | version | model | canonical JSON of variables), TTL `cache_ttl_hours`.
- `llm.enabled: false` or `state.overrides.disabled_ai` → always `None`.
- Privacy guard: raise if variables contain keys or values that look like user ids, emails or usernames.
- Every call emits an event `{prompt_id, model, latency_ms, cached, ok, fallback_reason, tokens}` to `on_event` so the backend can log it.

```python
def json(self, prompt_id, variables, schema, role):
    if not self.enabled: return None
    self._assert_no_pii(variables)
    key = self._key(prompt_id, variables)
    if (hit := self.cache.get(key)) is not None:
        return schema.model_validate_json(hit)
    prompt = self.prompts[prompt_id].render(variables)
    for attempt in range(self.max_retries + 1):
        if not self.budget.take(): return None
        try:
            raw = self.adapter.generate(prompt, json_schema=schema.model_json_schema(),
                                        temperature=self.temp[role], max_tokens=self.max_tokens[role],
                                        timeout=self.timeout)
            obj = schema.model_validate_json(strip_fences(raw))
            self.cache.set(key, obj.model_dump_json()); return obj
        except RateLimited:
            time.sleep(min(8, 2 ** attempt) + random.random())
        except ValidationError as e:
            prompt = prompt.with_repair(str(e))
        except (ProviderError, TimeoutError):
            break
    return None
```

### 14.3 Prompt files (version v1)

Write these files verbatim, then iterate only with new versions (`*.v2.md`), never by editing v1 after results exist.

**`onboarding_parse.v1.md`**

```markdown
---
id: onboarding_parse
version: 1
role: parse
---
You convert a reader's description of their reading taste into structured preferences for a book recommender.
Use ONLY these vocabularies. Anything that does not fit goes into "themes".
GENRES: $genres
ASPECTS: plot, writing_style, characters, world_building, pacing, editing_quality, dialogue, humor, emotional_impact, ending, complexity, tone_dark
DESCRIPTORS (each a number in [-1, 1]): darkness (-1 cozy ... +1 dark), pace (-1 slow ... +1 fast), complexity_level (-1 easy ... +1 dense)
Rules:
- Never invent book titles or authors. If the reader names books or authors, copy them exactly into named_titles / named_authors.
- Include an aspect only if the text clearly signals it. Include a descriptor only if the text clearly signals it.
- themes: at most 6 short phrases in the reader's own terms (e.g. "found family", "morally grey").
- Output a single JSON object matching the schema. No other text.

Example
Reader: "I love slow-burn fantasy with morally grey characters and found-family dynamics. Not into grimdark or love triangles."
JSON: {"liked_genres": ["Fantasy"], "disliked_genres": [], "positive_aspects": ["characters", "world_building"], "negative_aspects": [], "descriptors": {"darkness": -0.4, "pace": -0.5}, "themes": ["slow burn", "morally grey characters", "found family", "no love triangles"], "named_titles": [], "named_authors": []}

Example
Reader: "Cozy mysteries like the ones by Richard Osman, funny and not gory."
JSON: {"liked_genres": ["Mystery"], "disliked_genres": ["Horror"], "positive_aspects": ["humor", "plot"], "negative_aspects": [], "descriptors": {"darkness": -0.8}, "themes": ["cozy mystery", "not gory"], "named_titles": [], "named_authors": ["Richard Osman"]}

Reader: """$text"""
JSON:
```

**`intent_parse.v1.md`**

```markdown
---
id: intent_parse
version: 1
role: parse
---
You interpret one message sent to a book-recommendation assistant. You do not recommend books.
Return one JSON object matching the schema. No other text.
Intents: recommend_general, similar_to, similar_to_with_modifier, more_by_author, refine_previous, explain_recommendation, filter_by_attribute, mixed_opinion, out_of_scope.
Field guide:
- anchor_titles / anchor_authors: titles or authors the user names, copied exactly as written. Never add ones the user did not name.
- "that one", "the second one", "it": set target_index (1-based) into LAST_SHOWN, or anchor_from_session=true.
- aspect_deltas: aspect -> number in [-1, 1] for what the user wants more (+) or less (-) of. Aspects: $aspects
- descriptor_deltas: darkness, pace, complexity_level in [-1, 1]. "cozier"/"lighter"/"less bleak" -> darkness -0.6; "darker" -> +0.6; "faster"/"more gripping" -> pace +0.6; "slow burn" -> pace -0.5; "easier" -> complexity_level -0.5; "more literary" -> complexity_level +0.3 and aspect writing_style +0.5; "funnier" -> aspect humor +0.5.
- relative_length: "shorter" | "longer" | null (the system turns this into page limits).
- filters: min_year, max_year, genres_include, genres_exclude (from GENRES: $genres). "recent" -> min_year $recent_year; "classic" -> max_year 1980.
- exclude_previous: true when the user rejects what was shown ("something else", "not these").
- free_text_query: a short description of the vibe for semantic search, or null.
- needs_clarification: a short question if the request is ambiguous, else null.
SESSION: $session_summary
LAST_SHOWN: $last_shown

Examples
Message: "something like Project Hail Mary but shorter"
JSON: {"intent": "similar_to_with_modifier", "anchor_titles": ["Project Hail Mary"], "anchor_authors": [], "target_index": null, "anchor_from_session": false, "aspect_deltas": {}, "descriptor_deltas": {}, "relative_length": "shorter", "filters": {}, "exclude_previous": false, "free_text_query": null, "needs_clarification": null}
Message: "more by that author"
JSON: {"intent": "more_by_author", "anchor_titles": [], "anchor_authors": [], "target_index": null, "anchor_from_session": true, "aspect_deltas": {}, "descriptor_deltas": {}, "relative_length": null, "filters": {}, "exclude_previous": false, "free_text_query": null, "needs_clarification": null}
Message: "a bit cozier, and nothing older than 2010"
JSON: {"intent": "refine_previous", "anchor_titles": [], "anchor_authors": [], "target_index": null, "anchor_from_session": false, "aspect_deltas": {}, "descriptor_deltas": {"darkness": -0.6}, "relative_length": null, "filters": {"min_year": 2010}, "exclude_previous": false, "free_text_query": null, "needs_clarification": null}
Message: "why did you suggest the second one?"
JSON: {"intent": "explain_recommendation", "anchor_titles": [], "anchor_authors": [], "target_index": 2, "anchor_from_session": false, "aspect_deltas": {}, "descriptor_deltas": {}, "relative_length": null, "filters": {}, "exclude_previous": false, "free_text_query": null, "needs_clarification": null}
Message: "I loved the characters in Circe but the pacing was slow"
JSON: {"intent": "mixed_opinion", "anchor_titles": ["Circe"], "anchor_authors": [], "target_index": null, "anchor_from_session": false, "aspect_deltas": {"characters": 0.6, "pacing": 0.4}, "descriptor_deltas": {"pace": 0.4}, "relative_length": null, "filters": {}, "exclude_previous": false, "free_text_query": null, "needs_clarification": null}
Message: "what's the weather tomorrow"
JSON: {"intent": "out_of_scope", "anchor_titles": [], "anchor_authors": [], "target_index": null, "anchor_from_session": false, "aspect_deltas": {}, "descriptor_deltas": {}, "relative_length": null, "filters": {}, "exclude_previous": false, "free_text_query": null, "needs_clarification": null}

Message: """$message"""
JSON:
```

**`explain.v1.md`**

```markdown
---
id: explain
version: 1
role: explain
---
Write one short, friendly explanation (at most 45 words, 1-2 sentences) of why this book suits this reader.
Rules:
- Use ONLY facts in EVIDENCE. Do not attribute any preference to the reader that is not in EVIDENCE.user_evidence.
- Mention no book other than EVIDENCE.book.title and titles listed in EVIDENCE.user_evidence.similar_liked_items.
- Never reveal plot events, twists, character fates or the ending. Describe the book only at the level of its back-cover premise.
- Quote at most one snippet, verbatim, only from EVIDENCE.item_evidence.snippet, in double quotes.
- Use a number only if it appears in EVIDENCE.
- If the evidence is thin, write a shorter, vaguer sentence. No markdown, no emoji, no lists.

EVIDENCE:
$evidence_json

Explanation:
```

**`chat_reply.v1.md`**

```markdown
---
id: chat_reply
version: 1
role: chat
---
You are LitSense's reading assistant. The recommender has ALREADY chosen the books in RESULTS.
You must not add, remove or reorder books, and you must not recommend anything that is not in RESULTS.
Return one JSON object: {"intro": string, "items": [{"book_id": int, "reason": string}], "followup": string}.
- intro: one sentence acknowledging the request (max 25 words). No book titles.
- items: one entry per RESULTS book, same order, same book_id. reason: max 30 words, grounded only in that book's evidence. Do not write the book's title; the app displays it.
- followup: one short suggestion for refining further (max 15 words), based on ACTIVE_FILTERS or common refinements.
- Never reveal plot twists, character fates or endings. No markdown, no emoji.
- If RESULTS is empty: intro says nothing matched, items is [], followup suggests relaxing one active filter.
REQUEST: $request
ACTIVE_FILTERS: $active_filters
RESULTS: $results_json
JSON:
```

**`review_synthesis.v1.md`**

```markdown
---
id: review_synthesis
version: 1
role: synthesis
---
Summarize what reviewers think of this book in 2-3 sentences (max 70 words).
Use ONLY the statistics and snippets given. Mention no other book. Do not reveal plot twists, character fates or the ending; describe the book only at the level of its back-cover premise. Use numbers only if they appear below. Return JSON: {"summary": string}.
TITLE: $title
AUTHOR: $author
REVIEWS ANALYSED: $n_reviews
PRAISED: $praised
CRITICISED: $criticised
POSITIVE SNIPPETS: $pos_snippets
NEGATIVE SNIPPETS: $neg_snippets
JSON:
```

### 14.4 `onboarding.py`

`parse_onboarding(text, picks) -> OnboardingResult`:

1. If LLM available: `json("onboarding_parse", ...)`; drop any genre or aspect not in the vocabularies; clip descriptors to \[−1, 1\].
2. Resolve `named_titles` and `named_authors` with `entities.py`; resolved titles join the picks, resolved authors join favorite authors.
3. Always encode the raw text with the ONNX encoder (this keeps onboarding useful when parsing fails).
4. Return the structured profile plus `display_chips` ("Likes: Fantasy", "Values: characters", "Prefers: cozy") so the UI can show it back for correction.

### 14.5 `entities.py`

- At load, build indices: normalized title → item ids (sorted by popularity); `author_key` → item ids; a list of display names for fuzzy search.
- `resolve_title(q, author_hint=None)`: exact normalized match first; else `rapidfuzz.process.extract(q, titles, scorer=fuzz.WRatio, limit=5)`. Accept the top hit if score ≥ 88 and it beats the runner-up by ≥ 5 (or both map to the same item). Same title by several authors: use the author hint, else the most popular. Otherwise return `Ambiguous(candidates)`.
- `resolve_author(q)`: normalized exact, else `fuzz.token_sort_ratio` ≥ 85 with the same margin rule.
- Session references ("that author", "the second one") resolve from `session.last_shown` and `session.anchor_ids`.
- Log every resolution (query, match, score) in the reply's debug field.

### 14.6 `explain.py` and `validators.py`

```text
explain(state, book_id):
  profile  = build_profile(state, store)
  evidence = reasons.build_evidence(profile, book_id, store)
  if llm available:
      text = llm.text("explain", {"evidence_json": evidence.model_dump_json()}, role="explain")
      problems = validators.check(text, allowed_titles=evidence.allowed_titles(), evidence=evidence,
                                  max_words=llm.explain_max_words)
      if text and not problems: return Explanation(text, method="llm", evidence, rejections=[])
  return Explanation(reasons.template(evidence), method="template", evidence, rejections=problems)
```

Validators (all deterministic):

- **Titles:** at load, compile one case-insensitive regex alternation over normalized catalogue titles that have ≥ 2 words or ≥ 7 characters (skip titles that are common phrases). Any match in the output that is not in `allowed_titles` → reject.
- **Numbers:** every number or percentage in the output must appear in the evidence JSON (after rounding) → else reject.
- **Spoilers:** split the output into sentences and score each with the runtime spoiler classifier → any flag rejects.
- **Length/format:** word limit, no markdown lists, no URLs.
- **Snippet quotes:** quoted text must be a substring of the evidence snippet.

Rejections are returned so the backend can log rejection rates (a hallucination-rate proxy for the report).

### 14.7 `chat.py` — one turn

`SessionState`: `session_id`, `turn`, `aspect_deltas`, `descriptor_deltas`, `filters`, `excluded_ids`, `anchor_ids`, `anchor_authors`, `last_shown` (book ids in order), `last_request`, `history` (last 6 messages, truncated to 300 chars each). The backend persists it and sends it back next turn.

```text
chat_turn(state, session, message):
  1. intent = llm.json("intent_parse", {...}) or rule_parse(message, session)
  2. if intent.intent == out_of_scope: return template redirect (no recommend call)
  3. resolve anchors (titles/authors/target_index); ambiguous -> return clarification with up to 3 options
  4. update session:
       similar_to / similar_to_with_modifier -> anchors replaced, deltas reset then modifier applied
       refine_previous                       -> keep anchors, add deltas (clip to [-1, 1]), merge filters
       more_by_author                        -> anchor_authors = resolved author (or author of target)
       mixed_opinion                         -> anchors from titles, deltas from the stated likes/dislikes
       filter_by_attribute                   -> merge filters only
       exclude_previous                      -> excluded_ids += last_shown
  5. explain_recommendation -> explain(state, last_shown[target_index-1]); return
  6. modifiers -> recommend(state, k=5, filters, exclude, overrides):
       aspect_vec += 0.5 * aspect_deltas; descriptor_pref += descriptor_deltas
       item_item source uses anchor_ids; author source uses anchor_authors
       free_text_query -> semantic_vec = normalize(0.5*semantic_vec + 0.5*encode(query))
       relative_length -> shorter: max_pages = 0.8 * anchor pages (300 if unknown); longer: min_pages = 1.2 * anchor pages
  7. reply = llm.json("chat_reply", {request, active_filters, results_json}) validated:
       book_ids must equal the result ids in order; reasons pass title/number/spoiler validators
     else template intro + template reasons
  8. return ChatReply(intro, items, followup, chips=active modifiers as removable chips, session=updated)
```

**Rule-based fallback parser** (used when the LLM is off or fails): regexes for "like \<title>", "similar to \<title>", "by \<author>", "more \<author>", "why", "something else"; modifier keywords map to deltas exactly as in the intent prompt's field guide; the remaining text becomes `free_text_query`.

**Chip removal:** the backend sends the session with one modifier removed; the core recomputes from scratch.

### 14.8 Offline review synthesis (`ml/genai/synthesis.py`, stage `genai.synthesis`)

- Books: top 1,000 by review count (serve cutoff). Praised aspects: pos\_rate ≥ 0.6 and mention\_rate ≥ 0.1; criticised: neg\_rate ≥ 0.25 and mention\_rate ≥ 0.1. Snippets: 3 positive + 2 negative safe snippets.
- Uses `llm.model_quality`; resumable JSONL keyed by book; respects `rpm_limit` and the daily budget (may span days).
- Validate (titles: only the target; numbers; spoilers); on failure write a stats template ("Readers most often praise its characters (74% positive); some find the pacing slow.").
- Output goes into `catalog_text.json.gz` as `review_summary`. Never used as a model feature.

### 14.9 GenAI evaluation (`ml/genai/eval_intent.py`, `ml/genai/audit.py`)

- **Intent set:** draft \~70 candidate queries covering every intent (you may use the LLM to draft them); the user selects and edits 50 and writes the gold JSON by hand. Save `docs/annotation/intent_50.csv`. LLM-written gold labels are not allowed.
- **Intent metrics:** intent accuracy; anchor resolution accuracy (after `entities.py`); filter exact match; sign accuracy of aspect/descriptor deltas; overall field accuracy (target ≥ 0.8).
- **Explanation audit:** sample 100 explanations for evaluation users (stratified by method), export `docs/annotation/explain_audit.csv` for 3 raters: grounded (Y/N), hallucinated title (Y/N), spoiler (Y/N), useful (1–5). Compute rates and Fleiss' κ. Targets: ≥ 90% grounded, 0 hallucinated titles, 0 spoilers.
- **Automated rates:** generate 1,000 explanations and report validator rejection rates by type (title, number, spoiler, length).

**Gate (end of Week 12):** a three-turn chat demo works ("like Project Hail Mary but shorter" → "more by that author" → "why the second one?"), and the same flow works with `llm.enabled: false`.

## 15. API contract with the backend

This is the boundary with the teammate. Copy it into `docs/API_CONTRACT.md` in Week 1, ship `litsense_core/mock.py` with the same signatures, and freeze the schemas by Week 6. Any later change bumps `litsense_core.__api_version__` and is announced in the contract file.

### 15.1 Public functions

```python
def load_store(bundle_dir: str) -> ArtifactStore
def download_bundle(url: str, dest: str) -> str
def health() -> dict
def recommend(state: UserState, k: int = 10, filters: Filters | None = None,
              exclude: list[int] | None = None, use_ranker: bool = True) -> RecResponse
def home_rails(state: UserState) -> HomeRails
def similar_books(book_id: int, k: int = 10, state: UserState | None = None) -> list[RecItem]
def more_by_author(book_id: int, k: int = 10, state: UserState | None = None) -> list[RecItem]
def semantic_search(query: str, k: int = 20, filters: Filters | None = None) -> list[SearchHit]
def resolve_entities(query: str) -> EntityMatch          # for author-mode search
def onboarding_grid(n: int = 60) -> list[int]
def parse_onboarding(text: str, picks: list[int]) -> OnboardingResult
def explain(state: UserState, book_id: int) -> Explanation
def chat_turn(state: UserState, session: SessionState | None, message: str) -> ChatReply
def book_insights(book_id: int) -> BookInsights
def analyze_user_review(text: str) -> ReviewAnalysis
def metrics_report() -> dict                               # contents of results metrics.json
```

`book_id` everywhere is the integer item index from `id_map.json` (the backend stores it as `books.model_item_idx`).

### 15.2 Schemas (`litsense_core/schemas.py`, Pydantic v2)

```python
class RatingEvent(BaseModel):  book_id: int; stars: float; ts: int            # ms
class ListEvent(BaseModel):    book_id: int; status: Literal["want_to_read", "currently_reading", "read", "dropped"]; ts: int
class Overrides(BaseModel):    muted_genres: list[str] = []; pinned_aspects: list[str] = []; disabled_ai: bool = False

class OnboardingProfile(BaseModel):
    liked_genres: list[str] = []; disliked_genres: list[str] = []
    positive_aspects: list[str] = []; negative_aspects: list[str] = []
    descriptors: dict[str, float] = {}; themes: list[str] = []
    named_titles: list[str] = []; named_authors: list[str] = []

class ReviewAnalysis(BaseModel):
    book_id: int | None = None; ts: int | None = None
    aspects: dict[str, float] = {}        # aspect -> polarity in [-1, 1]
    spoiler_sentences: int = 0

class UserState(BaseModel):
    user_id: int | None = None            # never sent to the LLM
    corpus_user_id: str | None = None     # demo accounts seeded from the corpus
    ratings: list[RatingEvent] = []
    reading_list: list[ListEvent] = []
    dismissed: list[int] = []
    viewed: list[int] = []                # recent detail-page views, newest first (max 50)
    onboarding: OnboardingProfile | None = None
    onboarding_picks: list[int] = []
    onboarding_text: str | None = None
    review_analyses: list[ReviewAnalysis] = []
    overrides: Overrides = Overrides()
    profile_version: int = 0

class Filters(BaseModel):
    min_year: int | None = None; max_year: int | None = None
    min_pages: int | None = None; max_pages: int | None = None
    genres_include: list[str] = []; genres_exclude: list[str] = []

class AspectStrength(BaseModel): aspect: str; label: str; strength: float; n: int
class LikedItem(BaseModel):      book_id: int; title: str; stars: float
class Snippet(BaseModel):        text: str; review_ref: int; helpful_votes: int; aspect: str

class Evidence(BaseModel):
    book: dict                            # book_id, title, author, genres
    user_evidence: dict                   # top_positive_aspects, top_negative_aspects, similar_liked_items, favorite_author
    item_evidence: dict                   # aspect_profile, mention_rates, pos_rates, snippet
    signals: dict                         # sources, cf_rank, semantic_rank, aspect_match, bucket
    def allowed_titles(self) -> set[str]: ...

class RecItem(BaseModel):
    book_id: int; rank: int; score: float; sources: list[str]
    reason: str; reason_method: Literal["template", "llm"] = "template"
    evidence: Evidence

class RecResponse(BaseModel):
    items: list[RecItem]; continue_series: list[RecItem] = []
    model_version: str; profile_version: int; bucket: str
    degraded: bool = False; latency_ms: float

class HomeRails(BaseModel):
    top_picks: list[RecItem]
    because_you_liked: list[dict]         # [{anchor_book_id, items: list[RecItem]}] x2
    trending: list[RecItem]
    more_by_authors: list[RecItem] = []   # empty if no favorite author has 2+ books
    continue_exploring: list[RecItem] = []# from state.viewed; empty if none
    hidden_gems: list[RecItem] = []       # longtail source; empty until n >= 5
    from_reading_list: list[RecItem] = []
    continue_series: list[RecItem] = []

class SearchHit(BaseModel):    book_id: int; score: float; matched: Literal["semantic", "title", "author"]
class EntityMatch(BaseModel):  kind: Literal["title", "author", "none", "ambiguous"]; book_ids: list[int] = []; author: str | None = None; options: list[str] = []
class OnboardingResult(BaseModel): profile: OnboardingProfile; resolved_picks: list[int]; display_chips: list[str]; method: Literal["llm", "embed_only"]
class Explanation(BaseModel):  book_id: int; text: str; method: Literal["llm", "template"]; evidence: Evidence; rejections: list[str] = []

class SessionState(BaseModel):
    session_id: str; turn: int = 0
    aspect_deltas: dict[str, float] = {}; descriptor_deltas: dict[str, float] = {}
    filters: Filters = Filters(); excluded_ids: list[int] = []
    anchor_ids: list[int] = []; anchor_authors: list[str] = []
    last_shown: list[int] = []; last_request: str | None = None
    history: list[dict] = []

class ChatItem(BaseModel):     book_id: int; reason: str; evidence: Evidence
class Chip(BaseModel):         key: str; label: str          # e.g. key="descriptor:darkness", label="cozier"
class Clarification(BaseModel): question: str; options: list[dict]   # [{label, value}]
class ChatReply(BaseModel):
    intro: str; items: list[ChatItem] = []; followup: str | None = None
    chips: list[Chip] = []; clarification: Clarification | None = None
    session: SessionState; method: Literal["llm", "template"]; intent: str

class AspectBar(BaseModel):    aspect: str; label: str; pos_rate: float; neg_rate: float; mention_rate: float
class BookInsights(BaseModel): book_id: int; aspect_bars: list[AspectBar]; review_summary: str | None; snippets: list[Snippet]; n_reviews: int
```

### 15.3 Behaviour rules

- **No raises for model trouble.** If a component fails (ranker, LLM, FAISS), return a degraded but valid result with `degraded = True` and log. Raise `ValueError` only for invalid input (unknown `book_id`, malformed state).
- **Thread safety:** the store is read-only after load; the LLM client is thread-safe (locks around the cache and budget).
- **Caching** of responses is the backend's job, keyed by `(user_id, profile_version)`; the core is stateless.
- **Titles in chat:** the backend renders titles and covers from `book_id`; LLM text never supplies titles.
- **Logging:** the backend may pass `on_event` to receive LLM events and recommendation timing.

### 15.4 Performance budgets (CPU, warm, one thread)

| Call | Budget |
| --- | --- |
| `load_store` | ≤ 20 s, RSS ≤ 350 MB after load |
| `recommend` (ranker path, 200 candidates) | p95 ≤ 80 ms |
| `home_rails` | p95 ≤ 250 ms |
| `semantic_search` | p95 ≤ 60 ms |
| `explain`, template | ≤ 10 ms |
| `explain`, LLM | ≤ 4 s typical, hard timeout 8 s then template |
| `chat_turn` | ≤ 6 s typical (two LLM calls) |

### 15.5 Week-1 mock

`litsense_core.mock` implements every function above with the real schemas over a 200-book fixture catalogue (popular items, template reasons, a canned chat). The teammate builds against it from Week 1.

### 15.6 Admin metrics JSON (`metrics_report()`)

```json
{
  "data_version": "v1", "generated_at": "ISO-8601", "bundle_version": "...",
  "dataset": {"users": 0, "items": 0, "interactions": 0, "attrition": [], "splits": {}},
  "models": [{"id": "M2", "name": "LightGCN raw", "mode": "test",
              "metrics": {"ndcg@10": 0.0, "recall@20": 0.0, "precision@10": 0.0, "mrr@10": 0.0, "coverage@10": 0.0},
              "ci95": {"ndcg@10": [0.0, 0.0]}}],
  "ablation": [{"id": "M6", "vs": "M4", "metric": "ndcg@10", "diff": 0.0, "ci95": [0.0, 0.0]}],
  "strata": [{"model": "M8", "bucket": "1-2", "ndcg@10": 0.0, "users": 0}],
  "longtail": [], "coldstart": [],
  "genai": {"intent_field_accuracy": 0.0, "rejection_rates": {}, "audit": {}}
}
```

The zeros are placeholders for the shape only; real values are filled by `ml/eval/report.py`.

## 16. Testing requirements

Run with `make test` (pytest). Tests needing MiniLM, spaCy or a GPU are marked `@pytest.mark.slow`; CI (GitHub Actions on every push) runs `ruff` and `pytest -m "not slow"`. A module is done only when its tests pass.

### 16.1 Fixtures

- `tests/fixtures/tiny/make_tiny.py` generates a synthetic dataset deterministically: 50 users, 80 books (including a 3-book series, a box set, two editions of one work, a book with no author), 600 interactions spread over 3 years, and reviews built from template sentences with known aspects and polarities (some with spoiler sentences). It also builds a tiny serving bundle with random but correctly shaped artifacts and a 2-tree LightGBM model.
- `make sample` writes the first 2,000 lines of each real raw file to `local/raw_sample/` (gitignored) for parser tests against real formats.
- `FakeAdapter` for the LLM: scripted responses per prompt id, and modes that raise 429, time out, or return invalid JSON.

### 16.2 Mandatory tests

**Data**

- `parse_meta` handles `details`/`author` as dict and as JSON string, missing author with a `store` fallback, ISBN with dashes, missing pages.
- `normalize_title("Dune (Mass Market Paperback, 40th Anniversary Edition)") == "dune"`.
- Series volumes never merge: "Mistborn: The Final Empire" vs "Mistborn: The Well of Ascension".
- Box sets never merge with members: "The Mistborn Trilogy Box Set" vs "Mistborn: The Final Empire"; the box set is flagged.
- Editions merge: hardcover and paperback listings with the same title and author.
- Series parse: "The Way of Kings (The Stormlight Archive, #1)" → ("The Stormlight Archive", 1.0); "#2.5" → 2.5.
- k-core invariants hold after `kcore`.
- Split chronology: max(train.ts) < t1 ≤ min(val.ts), max(val.ts) < t2 ≤ min(test.ts).
- `id_map.json` is stable across reruns, and every stage loads the same map.
- Cross-edition duplicate reviews are removed.
- Timestamps are milliseconds (range assertion).

**Evaluation**

- Hand-computed metrics: truth {1, 3}, ranked \[3, 5, 1, 7\], k = 3 → precision 2/3, recall 1.0, hit 1, MRR 1.0, NDCG = (1 + 1/log2 4) / (1 + 1/log2 3) = 1.5 / 1.6309 ≈ 0.9197. Truth {9}, ranked \[1, 2, 3\] → all zeros.
- Masks are applied and ground truth is never masked.
- An all-zero user vector raises `DegenerateScoresError`.
- Bootstrap of identical models gives a CI containing 0.
- `eval.test` refuses to run twice without the override flag.

**Leakage**

- Injecting a review newer than the cutoff into a profile or feature build raises `LeakageError`.
- Evaluator refuses artifacts whose manifest cutoff does not match the mode.
- Item content embeddings contain no review text (check the text builder).

**Models**

- LightGCN on the tiny fixture beats random on training NDCG after a few epochs.
- Zero-degree items score −inf in CF scorers.
- Projection uses the same normalization flag in training and scoring; zero-degree items get e\_gcn = 0.
- Negative sampler never returns a positive.

**NLP (slow)**

- "The sentences themselves are just gorgeous to read." → `writing_style` detected.
- "Wonderful characters, but the middle third really dragged." → characters > 0 and pacing < 0.
- "It turns out the narrator was dead the whole time." → spoiler flagged.
- "A young girl discovers a hidden world beneath the city." → not flagged.
- ONNX parity thresholds from §10.3.

**Serving**

- Importing `litsense_core` in a fresh subprocess does not import `torch` (check `sys.modules`).
- `recommend` returns valid schemas, k items, no seen or dismissed items, and respects filters.
- Series guard: a user who read book 1 never gets book 3 in the main feed; book 2 appears in `continue_series`.
- A brand-new user with onboarding picks only gets a non-empty feed; one with nothing gets trending.
- Ranker missing → blend path, `degraded` stays False; ranker raising → blend path, `degraded` True.
- Feature column order equals `feature_list.json` of the bundled ranker.
- Latency check (warn-only in CI): `recommend` on the fixture under budget.

**GenAI (FakeAdapter)**

- With `llm.enabled: false`, every public function works and reports `method="template"`.
- Title validator rejects an output naming a catalogue title outside the allowed set.
- Number validator rejects a percentage not in the evidence.
- A spoiler-flagged LLM explanation falls back to the template.
- A chat reply with an extra or reordered `book_id` is rejected and replaced by the template.
- Onboarding drops genres and aspects outside the vocabularies.
- Entity resolution: "project hail mary" → the book; "andy weir" → author; a deliberately ambiguous title → clarification.
- Privacy guard raises if a prompt variable contains an email or user id.
- 429 responses trigger backoff and, after retries, a clean fallback.
- Rule-based intent parser handles "like X but shorter", "more by that author", "something else".

### 16.3 Manual checks (logged in progress.md)

- Merge audit (§7.4), aspect annotation (§11.5), intent gold set and explanation audit (§14.9).
- RSS and latency of the real bundle measured on a 512 MB container or with a memory limit (`docker run -m 512m`).

## 17. Build order, milestones and gates

Build in this order; each week ends at a gate the user reviews. Kept as a table (not a drawing) so it survives export for Claude Code.

| Week | Build (Claude Code) | User / teammate actions | Gate |
| --- | --- | --- | --- |
| 1 | Repo skeleton, `pyproject`, requirements, `config.yaml`, stage framework + manifests, seed utils, `schemas.py`, `mock.py`, `docs/API_CONTRACT.md`, tiny fixture generator, CI | Teammate wires the backend to `litsense_core.mock` | Mock serves the teammate's skeleton app; CI green |
| 2 | `data.download`, `data.meta`, `category_counts.csv`, `data.goodreads` | User picks genre slices | Slices recorded in config and decisions.md |
| 3 | `data.canonical` + audit files, `data.reviews`, `data.clean`, `data.kcore_split`, `data.eda` | User reviews `merge_audit.csv` | v0 frozen (\~100K); EDA committed; registry hash |
| 4 | Metrics + tests, evaluator, strata, bootstrap, report; random, popularity, ItemKNN, ALS; `nlp.content` early; content-only baseline | — | Baseline table on val with CIs |
| 5 | LightGCN port (raw + normalized), projection + shuffled + static controls; MVP blend `recommend` with real artifacts; bundle export v0 | Teammate swaps mock for real bundle | LightGCN ≥ ALS on val (or a logged investigation); recommendations visible in the browser |
| 6 | `nlp.onnx` + runtime encoder + parity tests; `nlp.sentences`; `nlp.aspects`; `nlp.sentiment`; annotation CSV + guide | User + teammate start labelling 300 sentences | Annotation files delivered; schemas frozen |
| 7 | `nlp.validate_aspects`, spoiler classifier, centroids, profiles (3 cutoffs), snippets, template reasons, ArtifactStore hardening | User confirms τ values | Aspect F1 table; spoiler P/R; RSS ≤ 350 MB |
| 8 | Cold-start paths, series guard, `home_rails`, `similar_books`, `more_by_author`, `semantic_search`, `onboarding_grid`, `book_insights`; bug fixes | Teammate deploys | **MVP:** PRD §37.4 tests pass on the deployed URL, from a phone, LLM off |
| 9 | Candidate sources (all 7) + recall analysis, `features.py`, ranker training data, LambdaRank training | User decides on any zero-contribution source | Ranker val NDCG@10 vs blend reported |
| 10 | Val ablation M4–M8; LLM client + adapters + cache + budget; explain pipeline + validators | User sets the Gemini model id and API key | Ablation v1 (val); explanations with working fallback |
| 11 | Rebuild everything at v1 (\~250K) end to end; intent parser + rule fallback; entities; onboarding parse | User drafts the intent gold set | v1 frozen; all artifacts regenerated; bundle v1 |
| 12 | `chat_turn`, `analyze_user_review`, review synthesis batch, `metrics_report`; GenAI eval scripts | Teammate builds chat UI on real responses | Three-turn chat demo works, with and without the LLM |
| 13 | `eval.test --confirm` (once), test-mode ablation, cold-start simulation, explanation audit export, all figures | Three raters do the audit | Results locked (`LOCKED` file) |
| 14 | Performance and memory tuning, demo accounts (`corpus_users`), load-test support | Teammate runs the load test | Budgets in §15.4 met |
| 15–16 | Report support: method write-ups generated from config and manifests, tables and figures from `results/`; final bundle tag | Report writing, demo rehearsal | `v1.0` tagged; reproducible from a fresh clone |

### 17.1 Gate checklist template (copy into `docs/progress.md` each week)

- [ ] All stages for this phase run with `--sample` locally and in full on Colab
- [ ] Manifests complete; registry updated
- [ ] Tests for new modules pass; CI green
- [ ] Results files written and committed; figures regenerated
- [ ] Decisions logged
- [ ] User reviewed the gate

### 17.2 If the schedule slips, cut in this order

1. Review synthesis batch (show aspect bars only).
2. Supplementary ablations.
3. The aspect candidate source.
4. Online `analyze_user_review` (users can still rate).
5. LLM chat replies (templates still work; keep LLM intent parsing).
6. Cold-start simulation (keep the sparsity strata).

**Never cut:** the temporal protocol and leakage tests, M2 vs M4 vs M6 with CIs, both LightGCN variants in long-tail claims, the spoiler filter, the title validator, the LLM-off mode, and the Week-8 MVP.

## 18. Pitfalls checklist and definitions of done

### 18.1 Pitfalls (re-read before every gate)

| Area | Pitfall | Guard |
| --- | --- | --- |
| Data | Timestamps are milliseconds despite "unix time" in the docs | Range assertion at load |
| Data | Joining on `asin` instead of `parent_asin` | Use `parent_asin` everywhere |
| Data | `details` / `author` as JSON strings in some copies | `_maybe_json` helper |
| Data | Using Amazon `average_rating` / `rating_number` as features (they include future ratings) | Compute all stats from the fit window |
| Data | Goodreads covers books only up to 2017 | Heuristic fallback; report match rate by publication year |
| Data | Box sets merged with volumes, or sequels merged | Guards + merge audit |
| Eval | Not refitting on train ∪ val before test | `eval.test` stage refits explicitly |
| Eval | Serve artifacts used in evaluation | Manifest cutoff assertion in the evaluator |
| Eval | Comparing a normalized hybrid with raw LightGCN only | Both LightGCN variants in every long-tail claim |
| Eval | Python loops with full argsort per user | Batched `argpartition` |
| Eval | Coverage without a stated denominator | Always report the catalogue size used |
| Leakage | Review summaries or LLM text in features or content embeddings | Forbidden; test on the text builder |
| Leakage | User aspect profiles built from val/test reviews | Cutoff argument + `LeakageError` |
| Models | `implicit` API differences across versions | Assert shapes after fit |
| Models | LightGBM groups not contiguous or not summing to rows | Assertion before fit |
| NLP | spaCy `n_process > 1` hanging on Colab | Fall back to 1 process |
| NLP | FAISS `IndexFlatIP` fed un-normalized or float16 vectors | Normalize and cast to float32 before indexing |
| NLP | SST-2 overconfidence on neutral sentences | Use thresholds for pos/neg rates; keep continuous polarity for means |
| Serving | fp16 dot products losing precision | Cast slices to float32 before scoring |
| Serving | torch sneaking into `litsense_core` through an import | Subprocess import test |
| Serving | Memory creeping over 512 MB | Measure RSS at every bundle export |
| GenAI | Hard-coded or deprecated model ids | Config only; `health()` reports LLM status |
| GenAI | JSON wrapped in code fences or prose | JSON mode + `strip_fences` + one repair retry |
| GenAI | Fuzzy matching resolving to the wrong book | Thresholds + margin rule + clarification |
| GenAI | Sending user identifiers to the provider | PII guard in the client |
| Ops | Colab disconnects mid-stream | Checkpoints; outputs on Drive; raw files on local disk |
| Ops | Free Drive quota (15 GB) exceeded | Never store raw dumps on Drive |

### 18.2 Definitions of done

- **Function / module:** type hints, docstrings, config-driven parameters, tests passing, entry in `docs/progress.md`.
- **Stage:** idempotent, writes a complete manifest, runs with `--sample` locally and in full on Colab, logs row counts.
- **Phase:** its gate in §17 is met, results are committed, decisions are logged, and the user has reviewed it.
- **ML/GenAI side of the project:** v1 results locked after a single test run; every function in §15 implemented within the budgets in §15.4; the LLM-off mode works end to end; `README.md` explains how to reproduce everything from a fresh clone (`make sample && make all-sample` locally; the numbered notebooks on Colab); `results/` contains every table and figure the report needs.

### 18.3 Figures and tables the report needs (all generated from `results/`)

1. Attrition table and dataset statistics (users, items, interactions, sparsity, split sizes).
2. Long-tail curve, rating distribution, members-per-canonical histogram, Goodreads match rate.
3. Aspect validation table (P/R/F1 per aspect, κ) and spoiler classifier metrics.
4. Aspect mention frequency and mean polarity by genre.
5. Baseline table (val and test) with CIs.
6. Projection controls table: raw vs normalized LightGCN vs projection vs shuffled.
7. Ablation table M0–M8b with paired bootstrap; NDCG@10 by sparsity bucket chart; long-tail table; coverage/novelty/ILD.
8. Candidate recall by source; ranker feature importance.
9. Cold-start simulation results.
10. Intent-parser accuracy, explanation audit (grounding, hallucination, spoiler, usefulness, κ), validator rejection rates.
11. Serving latency and memory measurements.
