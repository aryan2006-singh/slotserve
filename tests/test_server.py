"""End-to-end HTTP tests against the tiny model — no downloads, no GPU.

Covers the layer the engine tests can't: the thread bridge between the blocking
engine loop and asyncio, streaming SSE framing, and concurrent requests actually
sharing a batch.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine.config import EngineConfig  # noqa: E402
from tests.test_equivalence import VOCAB_SIZE, tiny_model  # noqa: E402


class StubTokenizer:
    """Character-level stand-in so tests need no tokenizer download."""

    eos_token_id = None
    pad_token_id = 0
    chat_template = None

    def __call__(self, text: str, add_special_tokens: bool = True):
        ids = [(ord(c) % (VOCAB_SIZE - 1)) + 1 for c in text][:64] or [1]
        return type("Encoding", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(97 + (i % 26)) for i in ids)


@pytest.fixture
def client(monkeypatch):
    from src.server import app as app_module

    model = tiny_model(seed=11)
    monkeypatch.setattr(
        app_module,
        "load_model_and_tokenizer",
        lambda config: (model, StubTokenizer()),
    )
    app_module.STATE["config"] = EngineConfig(
        max_slots=4, max_seq_len=256, device="cpu", dtype="float32"
    )
    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_completion_non_streaming(client):
    response = client.post(
        "/v1/completions",
        json={"prompt": "hello world", "max_tokens": 12, "temperature": 0.0},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["usage"]["completion_tokens"] == 12
    assert body["choices"][0]["finish_reason"] == "length"
    assert len(body["choices"][0]["text"]) > 0


def test_completion_streaming(client):
    with client.stream(
        "POST",
        "/v1/completions",
        json={"prompt": "streaming test", "max_tokens": 10, "stream": True},
    ) as response:
        assert response.status_code == 200
        chunks = []
        saw_done = False
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(json.loads(payload))

    assert saw_done
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    text = "".join(c["choices"][0]["text"] for c in chunks)
    assert len(text) > 0


def test_concurrent_requests_share_a_batch(client):
    """Fire several requests at once; the engine should report a batch > 1."""
    import concurrent.futures

    def send(i: int):
        return client.post(
            "/v1/completions",
            json={"prompt": f"prompt number {i}", "max_tokens": 40 + i * 10},
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(send, range(4)))

    assert all(r.status_code == 200 for r in responses)
    for i, response in enumerate(responses):
        assert response.json()["usage"]["completion_tokens"] == 40 + i * 10

    stats = client.get("/stats").json()
    assert stats["max_batch_seen"] > 1, "requests never overlapped; batching not exercised"


def test_metrics_endpoint(client):
    client.post("/v1/completions", json={"prompt": "metrics", "max_tokens": 5})
    body = client.get("/metrics").text
    assert "slotserve_requests_total" in body
    assert "slotserve_ttft_seconds_bucket" in body
    assert "slotserve_e2e_seconds_count 1" in body


def test_max_tokens_validation(client):
    response = client.post("/v1/completions", json={"prompt": "x", "max_tokens": 0})
    assert response.status_code in (400, 422, 500)


def test_stats_shape(client):
    client.post("/v1/completions", json={"prompt": "stats", "max_tokens": 4})
    stats = client.get("/stats").json()
    for key in ("running", "waiting", "slot_utilization", "kv_cache_mb", "tokens_per_second"):
        assert key in stats
    assert stats["running"] == 0
    assert torch.is_tensor(torch.tensor(0))  # sanity: torch imported fine


def test_stats_reports_every_slot(client):
    stats = client.get("/stats").json()
    assert stats["max_slots"] == 4
    assert stats["max_seq_len"] == 256
    assert stats["slots"] == [None, None, None, None]


def test_slot_snapshot_tracks_occupancy():
    from src.engine.request import Request
    from src.engine.sampling import SamplingParams
    from src.engine.scheduler import Scheduler

    sched = Scheduler(max_slots=3, max_seq_len=64, max_prefills_per_step=3)
    reqs = [Request(f"r{i}", [1, 2, 3], SamplingParams(max_tokens=10)) for i in range(2)]
    for r in reqs:
        sched.add(r)
    sched.admit()
    reqs[0].output_token_ids.extend([4, 5])

    snap = sched.slot_snapshot()
    assert len(snap) == 3
    by_id = {s["request_id"]: s for s in snap if s}
    assert set(by_id) == {"r0", "r1"}
    assert by_id["r0"]["generated"] == 2 and by_id["r0"]["seq_len"] == 5
    assert by_id["r0"]["max_tokens"] == 10
    assert snap.count(None) == 1

    reqs[0].status = reqs[0].status.FINISHED
    sched.retire_finished()
    assert [s["request_id"] for s in sched.slot_snapshot() if s] == ["r1"]
