# LitSense core API contract (`litsense_core`)

| | |
| --- | --- |
| API version | `litsense_core.__api_version__ = "0.1.0"` |
| Status | Draft. Schemas freeze in Week 6; after that every change bumps the version and gets a changelog entry below. |
| Source of truth | `litsense_core/schemas.py` (Pydantic v2). This document explains it; if they disagree, the code wins and this file gets fixed. |
| Owners | ML/GenAI: core package. Backend: everything that calls it. |

This is the boundary between the ML package and the FastAPI backend (CLAUDE.md §15). The backend builds a
`UserState` from its database on every call, calls a core function, renders the result and persists anything
returned (for example a chat `SessionState`). The core is stateless and never touches the database.

## 1. Using it now: the mock

Until the artifact bundle ships (Week 5), `litsense_core.mock` serves every function below with the same
signatures and schemas, over a deterministic synthetic catalogue of 200 books (`book_id` 0–199).

```python
from litsense_core import mock as core          # Week 1-4
# from litsense_core import <same names>        # Week 5+: same signatures, real models

feed = core.recommend(core_state, k=10)          # -> RecResponse
```

What the mock does and does not do:

- Feed: popularity plus genre and author affinity. Rating a book visibly changes the feed. Seen, dismissed,
  `exclude`, filters, muted genres and the series guard all behave as specified, so backend tests written
  against the mock stay valid.
- Reasons: the §12.5 templates. Explanations, chat and onboarding always report `method: "template"` /
  `"embed_only"`; the mock never calls an LLM.
- Chat: a few regular expressions ("like X but shorter", "more by that author", "why the second one?",
  "something else", "cozier", "nothing older than 2010"). Good enough to build the chat UI and session
  handling.
- Titles, authors, review snippets, aspect bars and review summaries are invented. Summaries and snippets start
  with "Mock". `metrics_report()` returns the admin JSON shape with every metric `null`.
- `load_store` and `download_bundle` do nothing (no network, no files); `health()` reports `"mock": true`.

## 1b. Booting the real package from the bundle

The bundle is one folder (`serve_bundle/`, 30 files, about 94 MB at v0) that `serve.export` writes on Colab and
also tars as `serve_bundle.tar.gz`. It is published as a GitHub Release asset so the backend can fetch it at
boot without Drive.

**Publishing (ML owner, once per bundle version):**

1. Download `MyDrive/litsense/artifacts/v0/serve_bundle.tar.gz` from Drive (the tar, not the zip: the tar keeps
   the `serve_bundle/` prefix that `download_bundle` expects).
2. On github.com/Aniketdhar810/litsense-model: Releases, "Draft a new release", tag `bundle-v0-20261020` (the
   `bundle_version` in the bundle's `manifest.json`), attach the tar as the only asset, publish.
3. The asset URL is then
   `https://github.com/Aniketdhar810/litsense-model/releases/download/bundle-v0-20261020/serve_bundle.tar.gz`.

**Booting (backend):**

```python
import os
import litsense_core as core

url = os.environ["LITSENSE_BUNDLE_URL"]            # the release asset above
folder = core.download_bundle(url, "/var/litsense")  # downloads + extracts once; returns .../serve_bundle
core.load_store(folder)                             # a few seconds; RSS expected 230 to 260 MB on the core image
assert core.health()["ranker"]
```

- `download_bundle` skips the download when `/var/litsense/serve_bundle/manifest.json` already exists, so a
  restart is instant if the disk persists. On a host with an ephemeral disk (Render free tier) it downloads at
  every cold start; the 94 MB fetch takes a few seconds.
- Environment: `pip install -r requirements-core.txt` only (no torch, pandas or scikit-learn: they add 100 MB
  of RSS through lightgbm's optional imports). Python 3.11 or 3.12. **lightgbm needs the OpenMP runtime
  `libgomp.so.1`**: a minimal Docker image (`python:3.12-slim`) lacks it and the ranker silently falls back to
  the blend. Add `RUN apt-get update && apt-get install -y --no-install-recommends libgomp1` to the Dockerfile
  (Render's native Python runtime already has it). Verify with `health()["ranker"] == True`; when it is false,
  `health()["ranker_error"]` names the cause.
- `GEMINI_API_KEY` in the environment turns the LLM on (`health()["llm"]` true). Without it every function still
  works and reports `method: "template"`.
- A manually extracted bundle works too: unzip or untar anywhere and pass that folder to `load_store` (either the
  folder that holds `manifest.json` or its parent).
- `dev/dev_server.py` is the reference: `LITSENSE_BUNDLE=<folder> uvicorn dev.dev_server:app` after
  `pip install fastapi uvicorn` on top of the core requirements.
- **Over HTTP:** the model service in `service/` does all of the above at boot (download, `configure_llm`,
  `load_store`, warm-up) and serves every function under `/v1`; a Node backend calls it instead of importing
  the package. See `service/README.md` and the 2026-10-20 changelog row.

## 2. Public functions (CLAUDE.md §15.1)

`book_id` everywhere is the integer item index from `id_map.json`; the backend stores it as
`books.model_item_idx`.

| Function | Returns | Notes | Budget (§15.4: CPU, warm, one thread) |
| --- | --- | --- | --- |
| `load_store(bundle_dir: str)` | `ArtifactStore` | Call once at boot (process-wide singleton). | ≤ 20 s, RSS ≤ 350 MB |
| `download_bundle(url: str, dest: str)` | `str` (bundle dir) | Fetches the GitHub Release bundle at boot. | n/a |
| `health()` | `dict` | Bundle version, cutoff, data version, `rss_mb`, `ranker`, `llm`. | not set |
| `recommend(state, k=10, filters=None, exclude=None, use_ranker=True)` | `RecResponse` | Main feed. Never calls the LLM. | p95 ≤ 80 ms (ranker path, 200 candidates) |
| `home_rails(state)` | `HomeRails` | Every home rail in one call. | p95 ≤ 250 ms |
| `similar_books(book_id, k=10, state=None)` | `list[RecItem]` | With a state: hides seen books and personalizes order. | not set |
| `more_by_author(book_id, k=10, state=None)` | `list[RecItem]` | Series order first. | not set |
| `semantic_search(query, k=20, filters=None)` | `list[SearchHit]` | `matched` says whether title, author or meaning matched. | p95 ≤ 60 ms |
| `resolve_entities(query)` | `EntityMatch` | For author-mode search; `ambiguous` returns `options`. | not set |
| `onboarding_grid(n=60)` | `list[int]` | Popular, well-rated, genre-diverse books for the cover picker. | not set |
| `parse_onboarding(text, picks)` | `OnboardingResult` | Show `display_chips` back so the reader can correct them. | not set |
| `explain(state, book_id)` | `Explanation` | LLM text when available and valid, else the template. | template ≤ 10 ms; LLM ≤ 4 s typical, 8 s hard timeout |
| `chat_turn(state, session, message)` | `ChatReply` | See §4. | ≤ 6 s typical (two LLM calls) |
| `book_insights(book_id)` | `BookInsights` | Aspect bars, review summary, spoiler-safe snippets. | not set |
| `analyze_user_review(text)` | `ReviewAnalysis` | Store the result and send it back in `UserState.review_analyses`. | not set |
| `metrics_report()` | `dict` | Admin page JSON (§6). | not set |
| `catalog()` **(proposed)** | `list[CatalogBook]` | Not in CLAUDE.md §15.1; see §7. | not set |

## 3. Key schema rules

The full models are in `litsense_core/schemas.py`. Rules that matter to the backend:

- **Timestamps are epoch milliseconds** (`RatingEvent.ts`, `ListEvent.ts`, `ReviewAnalysis.ts`). Values outside
  `1e11–1e13` are rejected, which catches seconds sent by mistake.
- **Stars are 1–5** (half stars allowed). `book_id` is ≥ 0. An unknown `book_id` anywhere in the state raises
  `ValueError`.
- **`UserState.viewed`**: recent detail-page views, newest first, at most 50. **`onboarding_picks`**: books the
  reader said they enjoyed; feeds treat them as already read and never recommend them back.
- **Seen books** (never in feeds): rated ∪ reading list with status `read` or `dropped` ∪ `dismissed` ∪
  onboarding picks. `want_to_read` and `currently_reading` books can still appear.
- **Interaction bucket** (`RecResponse.bucket`): `n` = ratings + 0.5 × reading-list entries. The reader is in the
  last bucket whose lower bound is ≤ `n`: `cold` 0, `warming` 1, `blending` 5, `warm` 20. So `n = 4.5` is
  `warming`.
- **`RecItem.reason`** is one sentence, safe to show as is. **`Evidence`** holds the facts behind it; its dicts
  contain plain JSON values. `Evidence.allowed_titles()` lists the only titles generated text may mention.
- **Titles in chat**: render titles and covers from `book_id` using your own `books` table. Chat `reason`
  strings never contain the book's own title.
- **`HomeRails.because_you_liked`** is a list of `BecauseYouLiked{anchor_book_id, items}` (up to 2).
  `hidden_gems` stays empty until `n ≥ 5`; `more_by_authors` stays empty unless a favourite author has 2+ books;
  `continue_exploring` needs `viewed`; `from_reading_list` needs `want_to_read` entries.
- **Series guard**: a later volume never appears in a feed until the reader has read the one before it. The exact
  next volume goes to `continue_series` (in `RecResponse` and `HomeRails`), never the main feed.

## 4. Chat protocol

1. First turn: `chat_turn(state, None, message)`. The core creates a `SessionState` with a new `session_id`.
2. Persist `reply.session` exactly as returned (it is JSON; `SessionState.model_validate_json` reads it back)
   and pass it to the next turn. `history` keeps the last 6 messages, each cut to 300 characters.
3. Render `reply.intro`, then each `reply.items[i]` (title and cover from `book_id`, plus `reason`), then
   `reply.followup`. `reply.intent` says what the core understood.
4. **Chips**: `reply.chips` lists the active modifiers (`key`, `label`). To remove one, delete the matching field
   from the session and call `chat_turn(state, session, "")`. An empty message re-runs the session as it stands.
   Key formats: `anchor:book:<id>`, `anchor:author`, `descriptor:<name>`, `aspect:<name>`,
   `filter:<field>` and `filter:genres_include:<genre>` / `filter:genres_exclude:<genre>`.
5. **Clarification**: when `reply.clarification` is set, `items` is empty. Show the question and the options; when
   the reader picks one, send its `value` as the next message.
6. "Why …?" questions return `intent: "explain_recommendation"` with the explained book as the only item and leave
   `session.last_shown` unchanged.

## 5. Behaviour rules (CLAUDE.md §15.3)

- **No raises for model trouble.** If the ranker, the LLM or FAISS fails, the core returns a valid but degraded
  result (`degraded: true`) and logs it. `ValueError` means invalid input only.
- **Thread safety**: the store is read-only after load; the LLM client locks its cache and budget.
- **Caching** is the backend's job, keyed by `(user_id, profile_version)`; the core echoes `profile_version`.
- **LLM off**: with `llm.enabled: false` or `state.overrides.disabled_ai`, every function still works and reports
  `method: "template"`.
- **Privacy**: `user_id` and `corpus_user_id` never leave the core; nothing identifying is sent to the LLM.
- **Logging**: the backend may pass an `on_event` callback to `litsense_core.configure_llm(config, on_event)`
  to receive one event per LLM call (prompt id, latency, cached, ok, fallback reason; never the prompt or key).

## 6. Admin metrics JSON (`metrics_report()`)

Shape from CLAUDE.md §15.6: `data_version`, `generated_at`, `bundle_version`, `dataset{users, items,
interactions, attrition, splits}`, `models[{id, name, mode, metrics{ndcg@10, recall@20, precision@10, mrr@10,
coverage@10}, ci95}]`, `ablation[{id, vs, metric, diff, ci95}]`, `strata[]`, `longtail[]`, `coldstart[]`,
`genai{intent_field_accuracy, rejection_rates, audit}`. Real values come only from `results/` files produced by
the pipeline. The mock adds `"is_mock": true` and leaves every metric `null`. Show `null` as "—", never as 0.

## 7. Open questions for the ML owner and backend owner

1. **`catalog()` (proposed addition).** Nothing in §15.1 gives the backend the catalogue it must seed into
   `books` (title, author, genres, year, pages, description, image, series, `book_id`). The mock offers
   `catalog() -> list[CatalogBook]`; the real core would read the same from the bundle. Accept, or agree on another
   export (for example a CSV in the bundle)? **Answered over HTTP (2026-10-20):** the model service serves
   `GET /v1/catalog` (paged `CatalogBook` rows from the loaded bundle) and `GET /v1/catalog/{book_id}`; see the
   changelog.
2. **Onboarding picks as read books.** The mock never recommends a picked book back. Should the backend also store
   picks as ratings, or only as `onboarding_picks`?
3. **Half stars.** The schema accepts any value in 1–5. If the UI only allows whole stars, the backend should round
   before sending.
4. **Pinned versions.** The core pins `pydantic==2.13.5`, `httpx==0.28.1` and `numpy==2.4.6` (see
   `requirements-core.txt`). It runs inside the backend process, so the backend must accept these versions.
5. **Python 3.11** is the serving target (CI tests 3.11 and 3.12).

## 8. Changelog

| Version | Date | Change |
| --- | --- | --- |
| 0.1.0 | 2026-10-12 | First contract and mock. Differences from CLAUDE.md §15.2, all with identical JSON: `because_you_liked` items are `BecauseYouLiked`, clarification options are `ClarificationOption{label, value}`, history entries are `ChatMessage{role, text}`. Added input validation (ms timestamps, 1–5 stars, `book_id ≥ 0`), `Bucket`/`Intent` literals, `ASPECT_LABELS`, and the proposed `CatalogBook`/`catalog()`. |
| 0.1.0 | 2026-10-17 | Real package available (decisions.md D56). Call `litsense_core.load_store(bundle_dir)` once at boot (or `download_bundle(url, dest)` first), then import the functions from `litsense_core` instead of `litsense_core.mock`; signatures and schemas unchanged. v0 degradations: `recommend` reports `degraded=True` when `use_ranker=True` (no LightGBM ranker in the bundle yet); `semantic_search` matches titles and authors only (no free-text embedding yet); `book_insights` has no aspect bars or snippets; `analyze_user_review` returns an empty analysis; `chat_turn` is rule-based. `health()["absent"]` lists bundle files not shipped yet. Bundle: `artifacts/v0/serve_bundle.tar.gz` (a few MB); `dev/dev_server.py` shows example payloads. |
| 0.1.0 | 2026-10-19 | GenAI layer (decisions.md D68 to D71). No signature or schema change. `recommend()` never calls the LLM: `RecItem.reason` stays the template (`reason_method="template"`); LLM text comes from `explain()` per item on demand and from `chat_turn`. `explain`, `chat_turn` and `parse_onboarding` now return `method="llm"` when the LLM is configured, available and its text passed the validators, else `"template"` / `"embed_only"` exactly as before; `Explanation.rejections` lists validator codes (`title:<t>`, `number:<n>`, `spoiler`, `length`, `format`, `quote`). `degraded` is never set by the LLM (a template answer is a valid result). New `litsense_core.configure_llm(config=None, on_event=None)`: call it before `load_store` to pass your own `llm` mapping and an event callback (`{prompt_id, model, latency_ms, cached, ok, fallback_reason, attempts}`); otherwise the bundle's `llm` config slice is used. The API key is read only from the environment variable named by `llm.api_key_env` (`GEMINI_API_KEY`). `health()["llm"]` is true only when enabled, the model id is set and the key is present. `BookInsights.review_summary` is filled from the bundle when the offline synthesis ran (null otherwise). Chat: an empty message re-runs the session as it stands (chip removal); `reply.chips` and `reply.clarification` are populated as section 4 describes. |
| 0.1.0 | 2026-10-19 | Ranker shipped (decisions.md D66). The bundle now carries `ranker.txt` + `feature_list.json`; `recommend(use_ranker=True)` scores the 200 candidates with LightGBM LambdaRank and returns `degraded=False`; `degraded=True` now means the ranker was missing or failed and the blend ranked instead. New read-only field `evidence.signals.ranked_by` (`ranker` or `blend`). `health()["ranker"]` is true. No signature or schema change; the bundle gains two small files (< 1 MB). |
| 0.1.0 | 2026-10-20 | HTTP model service (`service/`, decisions.md D75 to D81; `service/README.md`, OpenAPI in `service/openapi.json`). No core signature or schema change. **Mapping:** each public function is `/v1/<name in kebab case>`; functions taking a state, filters or lists are POST with a JSON body of their arguments (`/v1/recommend` takes `{state, k, filters, exclude, use_ranker}`, `/v1/chat-turn` `{state, session, message}`, `/v1/explain` `{state, book_id}`), id-only reads are GET (`/v1/onboarding-grid?n=`, `/v1/book-insights/{book_id}`, `/v1/metrics-report`); responses are the schemas of section 3 unchanged; unknown body fields are rejected. `health()` is `GET /health` (plus `ready`, boot timings, process facts); `GET /ready` is 503 until the bundle is loaded. **Auth:** header `X-API-Key: <SERVICE_API_KEY>` on every `/v1` call and `/metrics`. Optional `X-Request-Id` (echoed) and `X-Client-Id` (per-reader pseudonym for the rate limit). **Errors:** `{"error": {"code", "message", "request_id", "details"?}}`; a core `ValueError` (unknown `book_id`, bad `k`) is 400 `bad_request`, schema problems 400 `validation_error`, then 401, 404, 413, 429 (`Retry-After`), 503 `not_ready` or `overloaded`, 500 `internal`. LLM routes past the service's hard timeout (12 s) answer with the template and send `X-LitSense-Fallback: timeout`. **Catalogue (section 7 item 1):** `GET /v1/catalog?offset&limit` (limit at most 1000) returns `{bundle_version, total, offset, limit, items: CatalogBook[]}` in `book_id` order and `GET /v1/catalog/{book_id}` one `CatalogBook`; reseed when `bundle_version` changes. Fix: `similar_books` reasons end in `★` again (a double-encoded character, D75). |
