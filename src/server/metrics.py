"""Prometheus metrics without the prometheus_client dependency.

Deliberately hand-rolled histograms so the exposition format is visible and the
bucket boundaries are chosen for LLM serving (TTFT in the tens of milliseconds to
tens of seconds) rather than the library defaults, which are tuned for web
request latency and put almost everything in the last bucket.
"""

from __future__ import annotations

import threading

TTFT_BUCKETS = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
E2E_BUCKETS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]


class Histogram:
    def __init__(self, buckets: list[float]) -> None:
        self.buckets = buckets
        self.counts = [0] * (len(buckets) + 1)
        self.total = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.total += value
        self.count += 1
        for i, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def render(self, name: str) -> str:
        lines = [
            f"# TYPE {name} histogram",
            f"# HELP {name} seconds",
        ]
        cumulative = 0
        for edge, count in zip(self.buckets, self.counts):
            cumulative += count
            lines.append(f'{name}_bucket{{le="{edge}"}} {cumulative}')
        cumulative += self.counts[-1]
        lines.append(f'{name}_bucket{{le="+Inf"}} {cumulative}')
        lines.append(f"{name}_sum {self.total:.6f}")
        lines.append(f"{name}_count {self.count}")
        return "\n".join(lines)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.ttft = Histogram(TTFT_BUCKETS)
        self.e2e = Histogram(E2E_BUCKETS)
        self.requests_total = 0
        self.prompt_tokens_total = 0
        self.output_tokens_total = 0

    def observe(
        self,
        ttft: float | None,
        e2e: float,
        prompt_tokens: int,
        output_tokens: int,
    ) -> None:
        with self._lock:
            self.requests_total += 1
            self.prompt_tokens_total += prompt_tokens
            self.output_tokens_total += output_tokens
            if ttft is not None:
                self.ttft.observe(ttft)
            self.e2e.observe(e2e)

    def render(self, engine_stats: dict) -> str:
        with self._lock:
            parts = [
                "# TYPE slotserve_requests_total counter",
                f"slotserve_requests_total {self.requests_total}",
                "# TYPE slotserve_prompt_tokens_total counter",
                f"slotserve_prompt_tokens_total {self.prompt_tokens_total}",
                "# TYPE slotserve_output_tokens_total counter",
                f"slotserve_output_tokens_total {self.output_tokens_total}",
                self.ttft.render("slotserve_ttft_seconds"),
                self.e2e.render("slotserve_e2e_seconds"),
            ]
        for key in (
            "running",
            "waiting",
            "slot_utilization",
            "tokens_per_second",
            "kv_cache_mb",
            "max_batch_seen",
        ):
            if key in engine_stats:
                parts.append(f"# TYPE slotserve_{key} gauge")
                parts.append(f"slotserve_{key} {engine_stats[key]}")
        return "\n".join(parts) + "\n"
