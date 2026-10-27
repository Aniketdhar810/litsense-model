"""Small load test against a running service: p50/p95/max of recommend and explain plus RSS.

    python -m service.loadtest --url http://127.0.0.1:8000 --key "$SERVICE_API_KEY" \
        --requests 200 --concurrency 4

Readers are built from the onboarding grid (two 5-star ratings each); explain uses a fresh
``profile_version`` per request so the response cache never answers. Prints a table and, with
``--json``, one JSON object.
"""

import argparse
import asyncio
import json
import time
from typing import Any

import httpx
import numpy as np

TS = 1_760_000_000_000


def summary(name: str, ms: list[float], errors: int, wall_s: float) -> dict[str, Any]:
    arr = np.asarray(ms) if ms else np.zeros(1)
    return {
        "endpoint": name,
        "requests": len(ms) + errors,
        "errors": errors,
        "p50_ms": round(float(np.percentile(arr, 50)), 1),
        "p95_ms": round(float(np.percentile(arr, 95)), 1),
        "max_ms": round(float(arr.max()), 1),
        "throughput_rps": round(len(ms) / wall_s, 1) if wall_s > 0 else None,
    }


async def run(
    client: httpx.AsyncClient, name: str, bodies: list[tuple[str, dict[str, Any]]], concurrency: int
) -> dict[str, Any]:
    queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
    for item in bodies:
        queue.put_nowait(item)
    latencies: list[float] = []
    errors = 0

    async def worker() -> None:
        nonlocal errors
        while not queue.empty():
            path, body = queue.get_nowait()
            started = time.perf_counter()
            r = await client.post(path, json=body)
            if r.status_code == 200:
                latencies.append((time.perf_counter() - started) * 1000)
            else:
                errors += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return summary(name, latencies, errors, time.perf_counter() - started)


async def main_async(args: argparse.Namespace) -> list[dict[str, Any]]:
    headers = {"X-API-Key": args.key} if args.key else {}
    async with httpx.AsyncClient(base_url=args.url, headers=headers, timeout=60) as client:
        grid = (await client.get("/v1/onboarding-grid", params={"n": 40})).raise_for_status().json()
        readers = [
            {"ratings": [{"book_id": grid[i % len(grid)], "stars": 5, "ts": TS},
                         {"book_id": grid[(i + 7) % len(grid)], "stars": 5, "ts": TS + 1}]}
            for i in range(20)
        ]  # fmt: skip
        n = args.requests
        warm = [("/v1/recommend", {"state": readers[i % 20], "k": 10}) for i in range(10)]
        await run(client, "warmup", warm, 1)
        rec = [("/v1/recommend", {"state": readers[i % 20] if i % 2 else {}, "k": 10})
               for i in range(n)]  # fmt: skip
        first = (await client.post("/v1/recommend", json={"k": 10})).json()["items"]
        books = [it["book_id"] for it in first] or grid
        exp = [("/v1/explain", {"state": {**readers[i % 20], "profile_version": i},
                                "book_id": books[i % len(books)]}) for i in range(n)]  # fmt: skip
        results = [
            await run(client, "recommend", rec, args.concurrency),
            await run(client, "explain", exp, args.concurrency),
        ]
        health = (await client.get("/health")).json()
        for r in results:
            r.update(rss_mb=health.get("rss_mb"), bundle_version=health.get("bundle_version"),
                     llm=health.get("llm"), concurrency=args.concurrency)  # fmt: skip
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default="")
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = asyncio.run(main_async(args))
    if args.json:
        print(json.dumps(results))
        return
    cols = ("endpoint", "requests", "errors", "p50_ms", "p95_ms", "max_ms", "throughput_rps")
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))
    print(f"rss_mb after the run: {results[0]['rss_mb']} (bundle {results[0]['bundle_version']})")


if __name__ == "__main__":
    main()
