# LitSense model service

`litsense_core` over HTTP for the Node backend. Every `/v1` route is one public function of
[`docs/API_CONTRACT.md`](../docs/API_CONTRACT.md) with the same request fields and the same response
schema (the `litsense_core.schemas` classes, unchanged). The service adds auth, request ids, JSON
logs, Prometheus metrics, a rate limit, readiness and a catalogue endpoint for seeding the
backend's `books` table. Design decisions: `docs/decisions.md` D75 to D81.

The OpenAPI document the backend builds against is [`service/openapi.json`](openapi.json)
(regenerate with `python -m service.openapi`; a test fails when it is stale). A running service
also serves it at `/openapi.json` and the interactive docs at `/docs`.

## Run it

**Docker compose (recommended locally).** Model service, a static host for the bundle tar and a
stub of the Node backend:

```bash
cp service/.env.example service/.env
```

Set `SERVICE_API_KEY` in `service/.env`, put `serve_bundle.tar.gz` (from Drive, or the GitHub
Release) in `./local/`, then:

```bash
docker compose -f service/docker-compose.yml up --build
```

The model answers on <http://localhost:8000> once `/ready` says so; the stub on
<http://localhost:3000/api/feed?liked=12> calls `/v1/recommend` and `/v1/catalog/{id}` the way the
backend should. The bundle lands in the `bundle` volume, so restarts skip the download.

**Plain Python** (3.11 or 3.12, Linux or macOS; lightgbm needs `libgomp1` on slim Linux images):

```bash
pip install -r requirements-core.txt -r requirements-service.txt
```

```bash
BUNDLE_DIR=./local/serve_bundle SERVICE_API_KEY=dev-key python -m service
```

`BUNDLE_DIR` may hold an extracted bundle (`manifest.json` in it or in its `serve_bundle/`);
otherwise `BUNDLE_URL` is downloaded into it. `make service`, `make service-test`,
`make service-openapi` and `make service-image` wrap the same commands.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `SERVICE_API_KEY` | none (required) | Shared secret; the backend sends it as `X-API-Key`. Boot refuses to start without it unless `ALLOW_NO_AUTH=true` (development only). |
| `BUNDLE_URL` | none | `serve_bundle.tar.gz` to download when `BUNDLE_DIR` has no bundle (the GitHub Release asset, contract section 1b). |
| `BUNDLE_DIR` | `/var/litsense` | Where the bundle lives or is extracted; also holds the LLM cache `llm.sqlite`. |
| `GEMINI_API_KEY` | none | Turns the LLM on (`health()["llm"]`). Read by the core from the environment, never by the service. Without it every route answers with templates. |
| `LOG_LEVEL` | `INFO` | JSON logs on stdout. |
| `ONNX_THREADS` | bundle value (1) | Query-encoder threads; leave at 1 on 0.1 CPU. A different value rebuilds the encoder at boot. |
| `PORT` | `8000` | Render and Railway set it. |
| `CORE_THREADS` | `4` | Worker threads for core calls (the event loop stays free for probes). |
| `RECOMMEND_CONCURRENCY` | `2` | Ranking calls (`recommend`, `home-rails`) allowed at once. |
| `QUEUE_TIMEOUT_S` | `10` | How long a ranking call waits for a slot before 503 `overloaded`. |
| `RATE_LIMIT_RPS`, `RATE_LIMIT_BURST` | `20`, `40` | Token bucket per client; `RATE_LIMIT_RPS=0` turns it off. |
| `LLM_TIMEOUT_S` | `12` | Hard timeout of the LLM routes; past it the template answers. |
| `CACHE_TTL_S`, `CACHE_MAX_ENTRIES` | `300`, `1024` | Response cache for `explain` and `book-insights`. |
| `MAX_BODY_BYTES` | `1000000` | Larger bodies get 413. |
| `CORS_ORIGINS` | empty (off) | Comma-separated origins; only needed if a browser calls the service directly. |
| `LLM_CACHE_PATH`, `LLM_PROVIDER` | `<BUNDLE_DIR>/llm.sqlite`, bundle value | Overrides for tests and demos (`LLM_PROVIDER=fake`). |
| `BOOT_RETRIES`, `BOOT_BACKOFF_S`, `GRACEFUL_SHUTDOWN_S`, `FORWARDED_ALLOW_IPS` | `3`, `5`, `20`, `*` | Boot retries with backoff, drain time on SIGTERM, proxies trusted for the client address. |

## Endpoints

All `/v1` routes need `X-API-Key`. Bodies are JSON; `state` is a contract `UserState` and may be
omitted (an empty reader). Unknown fields in a body are a 400.

| Method and path | Contract function | Body or query | Notes |
| --- | --- | --- | --- |
| `POST /v1/recommend` | `recommend` | `state, k=10, filters, exclude, use_ranker=true` | p95 budget 80 ms; never calls the LLM. |
| `POST /v1/home-rails` | `home_rails` | `state` | Every home rail in one call. |
| `POST /v1/similar-books` | `similar_books` | `book_id, k=10, state` | |
| `POST /v1/more-by-author` | `more_by_author` | `book_id, k=10, state` | |
| `POST /v1/semantic-search` | `semantic_search` | `query, k=20, filters` | |
| `POST /v1/resolve-entities` | `resolve_entities` | `query` | |
| `GET /v1/onboarding-grid` | `onboarding_grid` | `?n=60` | List of `book_id`. |
| `POST /v1/parse-onboarding` | `parse_onboarding` | `text, picks` | LLM route. |
| `POST /v1/explain` | `explain` | `state, book_id` | LLM route; cached. |
| `POST /v1/chat-turn` | `chat_turn` | `state, session, message` | LLM route; persist `session` from the reply. |
| `GET /v1/book-insights/{book_id}` | `book_insights` | | Cached. |
| `POST /v1/analyze-user-review` | `analyze_user_review` | `text` | |
| `GET /v1/metrics-report` | `metrics_report` | | Admin JSON. |
| `GET /v1/catalog` | (contract section 7 item 1) | `?offset=0&limit=100` (max 1000) | `{bundle_version, total, offset, limit, items: CatalogBook[]}` in `book_id` order. |
| `GET /v1/catalog/{book_id}` | | | One `CatalogBook`; 404 if out of range. |
| `GET /health` | `health` | | Always 200: core health plus `ready`, `boot` timings, process facts. |
| `GET /ready` | | | 200 once loaded and warm, else 503. Use it as the platform health check. |
| `GET /metrics` | | | Prometheus text; needs `X-API-Key`. |

**Headers.** Send `X-Request-Id` (letters, digits, `._:-`, up to 64) and the response echoes it,
or a generated one; it is in every log line of that request. Send `X-Client-Id` with a per-reader
pseudonym (for example a hash of the user id, never an email) so the rate limit applies per
reader instead of to the whole backend. LLM routes answering after the timeout send
`X-LitSense-Fallback: timeout`; cached answers send `X-LitSense-Cache: hit`.

**Errors.** Every non-2xx answer has the same shape:

```json
{"error": {"code": "bad_request", "message": "unknown book_id 99999", "request_id": "7c1e..."}}
```

| Status | `code` | When |
| --- | --- | --- |
| 400 | `bad_request` | The core raised `ValueError` (unknown `book_id`, bad `k`); never a 500. |
| 400 | `validation_error` | The body or query does not match the schema; `details` lists `loc`, `msg`, `type` (never the submitted values). |
| 401 | `unauthorized` | Missing or wrong `X-API-Key`. |
| 404, 405 | `not_found`, `method_not_allowed` | Unknown route, catalogue id out of range, wrong method. |
| 413 | `payload_too_large` | Body over `MAX_BODY_BYTES`. |
| 429 | `rate_limited` | Token bucket empty; wait `Retry-After` seconds. |
| 503 | `not_ready` | Still loading (or the boot failed; see `/health`). `Retry-After: 5`. |
| 503 | `overloaded` | No ranking slot within `QUEUE_TIMEOUT_S`. `Retry-After: 1`. |
| 500 | `internal` | A bug; quote `request_id`. |

## curl examples

```bash
curl -s localhost:8000/ready
```

```bash
curl -s -H "X-API-Key: $SERVICE_API_KEY" -H "Content-Type: application/json" -d '{"state": {"ratings": [{"book_id": 12, "stars": 5, "ts": 1760000000000}]}, "k": 5}' localhost:8000/v1/recommend
```

```bash
curl -s -H "X-API-Key: $SERVICE_API_KEY" -H "Content-Type: application/json" -d '{"state": {}, "book_id": 12}' localhost:8000/v1/explain
```

```bash
curl -s -H "X-API-Key: $SERVICE_API_KEY" -H "Content-Type: application/json" -d '{"state": {}, "session": null, "message": "something like Circe but shorter"}' localhost:8000/v1/chat-turn
```

```bash
curl -s -H "X-API-Key: $SERVICE_API_KEY" "localhost:8000/v1/catalog?offset=0&limit=500"
```

```bash
curl -s -H "X-API-Key: $SERVICE_API_KEY" localhost:8000/metrics
```

## Deploy

**Render (free).** The repo root has a Blueprint, [`render.yaml`](../render.yaml): Docker runtime,
`service/Dockerfile`, health check `/ready`, region Singapore, `SERVICE_API_KEY` generated by
Render, `GEMINI_API_KEY` asked for at creation, `BUNDLE_URL` pointing at the GitHub Release.

1. Publish the bundle as a GitHub Release asset (contract section 1b) so `BUNDLE_URL` resolves.
2. Render dashboard: New, Blueprint, pick `Aniketdhar810/litsense-model` and the branch that has `service/`.
3. Enter `GEMINI_API_KEY` when asked (or leave it empty for template-only answers).
4. After the first deploy, copy `SERVICE_API_KEY` from the service's Environment tab into the
   Node backend's environment, with the service URL (`https://litsense-model.onrender.com` or
   whatever Render assigns).
5. Check `https://<service>/health`: `ready`, `ranker` and `llm` should be true.

Free plan facts (Render docs, checked 2026-10-20): 512 MB RAM and 0.1 CPU; spins down after
15 minutes without inbound traffic and takes about a minute to spin up (longer here, see below);
750 free instance hours per workspace per month; the disk is ephemeral and free services cannot
attach a persistent disk, so **every cold start downloads the 94 MB bundle again**. Render may
restart a free service at any time. An uptime pinger hitting `/health` every 10 minutes keeps it
awake within the 750 hours.

**Railway.** Railway builds a root `Dockerfile` by default; set the service variable
`RAILWAY_DOCKERFILE_PATH=service/Dockerfile`, add the same variables as above (Railway sets
`PORT`), and set the health check path to `/ready`. A volume mounted at `/var/litsense` keeps the
bundle across restarts. Plans (Railway docs, checked 2026-10-20): Free is $0 with $1 of usage a
month and at most 0.5 GB RAM per service; the Trial is a one-time $5 grant with up to 1 GB; Hobby
is $5 a month. The 512 MB Free cap is the same squeeze as Render's.

**Boot time.** The port opens as soon as Python has imported the app; the bundle then downloads,
loads and warms in a background thread while `/ready` answers 503. Measured numbers are in the
next section; on Render add the platform's own spin-up of about a minute.

## Concurrency, memory and limits

- One uvicorn process (no workers: the store would be loaded once per worker and 512 MB holds
  one). Core calls run on `CORE_THREADS` threads; the store is read-only after load (memory-mapped
  arrays, FAISS flat search, the ONNX session and LightGBM predict all allow concurrent reads) and
  the core's LLM client locks its own cache and budget.
- `RECOMMEND_CONCURRENCY` caps the CPU-heavy ranking calls; on 0.1 CPU more parallelism only adds
  latency. Chat is not capped (it mostly waits on Gemini).
- `OMP_NUM_THREADS=1` in the image keeps LightGBM from starting one thread per host core.
- The rate limiter and the response caches live in this process. With more than one instance,
  move the token buckets to Redis (a Lua token bucket or INCR plus EXPIRE behind the same
  `take(key)` call) and the caches to Redis or drop them.
- Logs are JSON lines: one access line per request (method, route template, status, duration,
  request id) and one `llm_call` line per LLM call (prompt id, model, latency, cached, ok,
  fallback reason). Bodies, `UserState`, chat messages, reviews and prompts are never logged.

## Measured

Docker on a laptop with the tiny test bundle (52 books; the real 94 MB bundle was not available
locally), 2026-10-20. Details and method in `docs/decisions.md` D81.

| Setting | Ready after start | Recommend p50 / p95 | Explain (template) p50 / p95 | RSS |
| --- | --- | --- | --- | --- |
| 1 CPU, 512 MB, one client | 3.4 s | 13.5 / 30.3 ms | 3.9 / 6.0 ms | 156 MB |
| 1 CPU, 512 MB, four clients | 3.4 s | 72.6 / 108.4 ms | 25.2 / 41.3 ms | 156 MB |
| 0.1 CPU, 512 MB (Render free), two clients | 48 s | 202 / 296 ms | 98 / 190 ms | 155 MB |

At 0.1 CPU the 48 s are 34 s of Python imports, 12 s of loading and the rest download and
warm-up. With the real bundle expect a longer load, RSS near 230 to 260 MB and roughly 0.4 to
0.7 s per feed request on the free plan; `/health` reports the real `download_s`, `load_s`,
`warm_s` and `rss_mb` once it runs there.

## Load test

`service/loadtest.py` drives a running service with `recommend` and `explain` (cache-busting
`profile_version`) and prints p50, p95, max, throughput and RSS:

```bash
python -m service.loadtest --url http://127.0.0.1:8000 --key "$SERVICE_API_KEY" --requests 200 --concurrency 4
```

## Tests

```bash
python -m pytest tests/service -m "not slow"
```

`tests/service` boots the app over a hand-built 12-book store with the fake LLM provider (every
route, the error envelope, auth, readiness, request ids, rate limit, body limit, catalogue paging,
LLM timeouts, caches, metrics, logs) and, when torch is installed (CI has it), over the real tiny
pipeline bundle served from a local HTTP server: download, load, ranker on, every route, and
recommend's p95 under the 80 ms budget.
