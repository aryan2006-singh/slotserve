"""Load generator. Produces the numbers that go in BENCHMARKS.md.

Two modes, and the difference matters:

  --mode closed   N clients, each sending the next request as soon as the last
                  one returns. Measures peak throughput. This is what most
                  people report, and it flatters the server.

  --mode open     Poisson arrivals at a fixed rate, independent of how fast the
                  server responds. Measures what users actually experience,
                  because real traffic does not wait for you. If the server
                  cannot keep up, queueing delay shows up in TTFT — which is
                  exactly the signal you want.

Report open-loop numbers. Report closed-loop too if you like, but label them.

Usage:
    python bench/load_test.py --url http://localhost:8000 \
        --mode open --rate 8 --num-requests 200 \
        --prompt-tokens 512 --max-tokens 128 \
        --label "fp16 + continuous batching" --out bench/results/

The output JSON is consumed by bench/report.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field

import aiohttp


@dataclass
class RequestResult:
    ok: bool
    ttft: float | None
    e2e: float
    output_tokens: int
    prompt_tokens: int
    error: str | None = None
    inter_token_latencies: list[float] = field(default_factory=list)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def synthetic_prompt(num_tokens: int, rng: random.Random) -> str:
    """Random words, not lorem ipsum.

    A repeated prompt would let any sane server hit a prefix cache and report a
    prefill time that has nothing to do with real traffic.
    """
    words = [
        "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 9)))
        for _ in range(num_tokens)
    ]
    return " ".join(words)


async def one_request(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> RequestResult:
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    start = time.perf_counter()
    ttft = None
    tokens = 0
    last = start
    gaps: list[float] = []
    try:
        async with session.post(f"{url}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                return RequestResult(False, None, time.perf_counter() - start, 0, 0,
                                     error=f"HTTP {resp.status}: {body[:200]}")
            async for raw in resp.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                text = chunk["choices"][0].get("text", "")
                if not text:
                    continue
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - start
                else:
                    gaps.append(now - last)
                last = now
                tokens += 1
    except Exception as exc:  # noqa: BLE001
        return RequestResult(False, ttft, time.perf_counter() - start, tokens, 0,
                             error=repr(exc))
    return RequestResult(True, ttft, time.perf_counter() - start, tokens, 0,
                         inter_token_latencies=gaps)


async def run_closed(args, prompts) -> list[RequestResult]:
    results: list[RequestResult] = []
    semaphore = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession() as session:
        async def worker(prompt):
            async with semaphore:
                return await one_request(session, args.url, prompt,
                                         args.max_tokens, args.temperature)

        results = await asyncio.gather(*[worker(p) for p in prompts])
    return list(results)


async def run_open(args, prompts) -> list[RequestResult]:
    """Poisson arrivals: inter-arrival times ~ Exponential(1/rate)."""
    rng = random.Random(args.seed)
    tasks = []
    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        for i, prompt in enumerate(prompts):
            tasks.append(
                asyncio.create_task(
                    one_request(session, args.url, prompt, args.max_tokens, args.temperature)
                )
            )
            if i < len(prompts) - 1:
                await asyncio.sleep(rng.expovariate(args.rate))
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    args._wall = wall
    return list(results)


def summarize(args, results: list[RequestResult], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    ttfts = [r.ttft for r in ok if r.ttft is not None]
    e2es = [r.e2e for r in ok]
    total_tokens = sum(r.output_tokens for r in ok)
    itls = [gap for r in ok for gap in r.inter_token_latencies]

    return {
        "label": args.label,
        "mode": args.mode,
        "rate": args.rate if args.mode == "open" else None,
        "concurrency": args.concurrency if args.mode == "closed" else None,
        "num_requests": len(results),
        "failed": failed,
        "prompt_tokens_target": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "wall_seconds": round(wall, 3),
        "output_tokens": total_tokens,
        "throughput_tok_per_s": round(total_tokens / wall, 2) if wall else 0.0,
        "requests_per_s": round(len(ok) / wall, 3) if wall else 0.0,
        "ttft_ms": {
            "p50": round(percentile(ttfts, 0.50) * 1000, 1),
            "p95": round(percentile(ttfts, 0.95) * 1000, 1),
            "p99": round(percentile(ttfts, 0.99) * 1000, 1),
            "mean": round(statistics.fmean(ttfts) * 1000, 1) if ttfts else None,
        },
        "itl_ms": {
            "p50": round(percentile(itls, 0.50) * 1000, 2),
            "p95": round(percentile(itls, 0.95) * 1000, 2),
        },
        "e2e_ms": {
            "p50": round(percentile(e2es, 0.50) * 1000, 1),
            "p95": round(percentile(e2es, 0.95) * 1000, 1),
            "p99": round(percentile(e2es, 0.99) * 1000, 1),
        },
        "errors": [r.error for r in results if not r.ok][:5],
    }


async def fetch_server_stats(url: str) -> dict:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/stats") as resp:
                return await resp.json()
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--mode", choices=["open", "closed"], default="open")
    parser.add_argument("--rate", type=float, default=4.0, help="requests/sec (open mode)")
    parser.add_argument("--concurrency", type=int, default=16, help="clients (closed mode)")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument("--out", default="bench/results")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    prompts = [synthetic_prompt(args.prompt_tokens, rng) for _ in range(args.num_requests)]

    if args.warmup:
        warm = argparse.Namespace(**vars(args))
        warm.mode = "closed"
        warm.concurrency = min(4, args.warmup)
        print(f"warmup: {args.warmup} requests...")
        asyncio.run(run_closed(warm, prompts[: args.warmup]))

    print(f"running: mode={args.mode} n={args.num_requests} label={args.label!r}")
    start = time.perf_counter()
    if args.mode == "open":
        results = asyncio.run(run_open(args, prompts))
    else:
        results = asyncio.run(run_closed(args, prompts))
    wall = time.perf_counter() - start

    summary = summarize(args, results, wall)
    summary["server_stats"] = asyncio.run(fetch_server_stats(args.url))

    os.makedirs(args.out, exist_ok=True)
    slug = args.label.lower().replace(" ", "-").replace("/", "-")
    path = os.path.join(args.out, f"{slug}-{args.mode}-{int(time.time())}.json")
    with open(path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps({k: v for k, v in summary.items() if k != "server_stats"}, indent=2))
    print(f"\nwrote {path}")
    if summary["failed"]:
        print(f"WARNING: {summary['failed']} requests failed — do not report this run")


if __name__ == "__main__":
    main()
