// Stub of the Node backend for docker compose (service/README.md). It calls the model service
// the way the real backend should: X-API-Key, a fresh X-Request-Id per call, X-Client-Id per
// reader, and the error envelope passed through. No dependencies (Node 22 fetch).
const http = require("node:http");
const crypto = require("node:crypto");

const MODEL_URL = process.env.MODEL_URL || "http://model:8000";
const KEY = process.env.SERVICE_API_KEY || "";

async function model(path, body) {
  const requestId = crypto.randomUUID();
  const res = await fetch(MODEL_URL + path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": KEY,
      "X-Request-Id": requestId,
      "X-Client-Id": "stub-reader-1",
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  return { status: res.status, requestId: res.headers.get("x-request-id"), json: await res.json() };
}

async function feed(url) {
  const liked = url.searchParams.get("liked");
  const state = liked
    ? { ratings: [{ book_id: Number(liked), stars: 5, ts: Date.now() }], profile_version: 1 }
    : {};
  const rec = await model("/v1/recommend", { state, k: 5 });
  if (rec.status !== 200) return rec;
  const items = await Promise.all(
    rec.json.items.map(async (item) => {
      const book = await model(`/v1/catalog/${item.book_id}`);
      return { book_id: item.book_id, title: book.json.title, reason: item.reason };
    }),
  );
  return { status: 200, requestId: rec.requestId, json: { bucket: rec.json.bucket, items } };
}

http
  .createServer(async (req, res) => {
    const url = new URL(req.url, "http://stub");
    try {
      let out;
      if (url.pathname === "/api/feed") out = await feed(url);
      else if (url.pathname === "/api/ready") out = await model("/ready");
      else out = { status: 404, json: { error: "try /api/feed?liked=<book_id> or /api/ready" } };
      res.writeHead(out.status, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ request_id: out.requestId, ...out.json }, null, 2));
    } catch (err) {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: String(err) }));
    }
  })
  .listen(3000, () => console.log("node stub on :3000, model at " + MODEL_URL));
