"""Randomised stress test for the scheduler.

`test_equivalence.py` checks hand-picked shapes. This fuzzes the dimensions that
interact badly — slot count, prompt length spread, output length spread, arrival
timing — and asserts the same invariant: greedy output is independent of what
else was in the batch.

Run more rounds before you trust a benchmark number:
    pytest tests/test_fuzz.py -q --rounds 50
"""

from __future__ import annotations

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import SamplingParams  # noqa: E402
from tests.test_equivalence import (  # noqa: E402
    VOCAB_SIZE,
    build_engine,
    reference_greedy,
    tiny_model,
)


def pytest_addoption(parser):  # pragma: no cover - registered via conftest
    parser.addoption("--rounds", action="store", default=8, type=int)


@pytest.mark.parametrize("round_id", range(8))
def test_random_workload(round_id: int):
    rng = random.Random(round_id)
    torch.manual_seed(round_id)

    model = tiny_model(seed=round_id % 4, num_kv_heads=rng.choice([1, 2, 4]))
    num_requests = rng.randint(3, 10)
    max_slots = rng.randint(1, 6)

    prompts = [
        torch.randint(1, VOCAB_SIZE, (rng.randint(1, 25),)).tolist()
        for _ in range(num_requests)
    ]
    max_tokens = [rng.randint(1, 25) for _ in range(num_requests)]
    # How many engine steps to run before submitting each request.
    delays = [rng.randint(0, 6) for _ in range(num_requests)]

    expected = [reference_greedy(model, p, n) for p, n in zip(prompts, max_tokens)]

    engine = build_engine(model, max_slots=max_slots,
                         max_prefills_per_step=rng.choice([1, 1, 2]))
    pending = list(zip(delays, prompts, max_tokens))
    submitted: list = [None] * num_requests
    order = sorted(range(num_requests), key=lambda i: delays[i])

    step = 0
    idx = 0
    while idx < num_requests or engine.has_work:
        while idx < num_requests and pending[order[idx]][0] <= step:
            i = order[idx]
            submitted[i] = engine.add_request(
                prompts[i], SamplingParams(max_tokens=max_tokens[i], temperature=0.0)
            )
            idx += 1
        engine.step()
        step += 1
        assert step < 5000, "engine failed to make progress"

    for i, (req, want) in enumerate(zip(submitted, expected)):
        assert req.output_token_ids == want, (
            f"round {round_id} request {i}: slots={max_slots} "
            f"prompt_len={len(prompts[i])} max_tokens={max_tokens[i]}"
        )

    assert engine.scheduler.num_running == 0
    assert len(engine.scheduler.free_slots) == max_slots
