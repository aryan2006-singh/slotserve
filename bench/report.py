"""Turn bench/results/*.json into the BENCHMARKS.md table.

The point of generating the table rather than typing it: a reader can rerun
`make bench` and get your numbers back. A hand-typed table is a claim; a
generated one is evidence.

    python bench/report.py --results bench/results --out BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
from datetime import datetime, timezone

COLUMNS = [
    ("label", "Config"),
    ("mode", "Mode"),
    ("load", "Load"),
    ("ttft_p50", "TTFT p50 (ms)"),
    ("ttft_p95", "TTFT p95 (ms)"),
    ("itl_p50", "ITL p50 (ms)"),
    ("e2e_p95", "E2E p95 (ms)"),
    ("throughput", "Output tok/s"),
    ("kv_mb", "KV cache (MB)"),
]


def row_from(summary: dict) -> dict:
    stats = summary.get("server_stats") or {}
    load = (
        f"{summary['rate']} req/s"
        if summary.get("mode") == "open"
        else f"{summary.get('concurrency')} clients"
    )
    return {
        "label": summary["label"],
        "mode": summary["mode"],
        "load": load,
        "ttft_p50": summary["ttft_ms"]["p50"],
        "ttft_p95": summary["ttft_ms"]["p95"],
        "itl_p50": summary["itl_ms"]["p50"],
        "e2e_p95": summary["e2e_ms"]["p95"],
        "throughput": summary["throughput_tok_per_s"],
        "kv_mb": stats.get("kv_cache_mb", "—"),
    }


def hardware_block() -> str:
    lines = [
        f"- Host: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
    ]
    try:
        import torch

        lines.append(f"- torch: `{torch.__version__}`")
        if torch.cuda.is_available():
            lines.append(f"- GPU: `{torch.cuda.get_device_name(0)}`")
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            lines.append(f"- VRAM: `{total:.1f} GiB`")
        else:
            lines.append("- GPU: none (CPU run — throughput numbers are not comparable to GPU)")
    except Exception:  # noqa: BLE001
        pass
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        if commit:
            lines.append(f"- Commit: `{commit}`")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="bench/results")
    parser.add_argument("--out", default="BENCHMARKS.md")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.results, "*.json")))
    if not files:
        raise SystemExit(f"no result files in {args.results}; run bench/load_test.py first")

    rows = []
    for path in files:
        with open(path) as handle:
            summary = json.load(handle)
        if summary.get("failed"):
            print(f"skipping {os.path.basename(path)}: {summary['failed']} failed requests")
            continue
        rows.append(row_from(summary))

    header = "| " + " | ".join(label for _, label in COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(COLUMNS)) + "|"
    body = [
        "| " + " | ".join(str(row[key]) for key, _ in COLUMNS) + " |" for row in rows
    ]

    content = f"""# Benchmarks

Generated {datetime.now(timezone.utc).isoformat(timespec="seconds")} by `bench/report.py`.
Regenerate with `make bench`.

## Environment

{hardware_block()}

## Results

{header}
{divider}
{os.linesep.join(body)}

**How to read this**

- **TTFT** — time to first token. Dominated by prefill and by queueing delay when
  the server is saturated. This is the number users feel first.
- **ITL** — inter-token latency during streaming. Dominated by decode step time,
  which grows with batch size. The tradeoff continuous batching makes explicit:
  larger batches raise throughput and raise ITL.
- **E2E p95** — what a slow user actually waits.
- **Output tok/s** — aggregate across all concurrent requests, not per request.

Open-loop rows are the honest ones: arrivals are Poisson and do not slow down
when the server does, so an overloaded server shows up as exploding TTFT rather
than as a quietly lower request rate.
"""
    with open(args.out, "w") as handle:
        handle.write(content)
    print(f"wrote {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()
