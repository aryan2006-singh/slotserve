"""The test that matters: continuous batching must not change the output.

A continuous-batching engine is only useful if it is *invisible*. Given greedy
decoding, a request must produce byte-identical tokens whether it ran alone or
shared a batch with seven other requests of wildly different lengths that joined
and left at arbitrary steps.

Almost every hand-rolled batching bug shows up here:
  * off-by-one in `write_positions` (row reads its own future token),
  * a mask that lets row 2 attend to row 5's cache,
  * position_ids taken from the batch instead of the row,
  * a slot reused before its previous occupant was evicted.

The reference is HF `generate()` run one prompt at a time, which is slow but
unambiguous.

Runs on CPU with a small randomly-initialised Llama, so it needs no downloads
and no GPU. `pytest tests/ -v`
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import EngineConfig, LLMEngine, SamplingParams  # noqa: E402

VOCAB_SIZE = 512
MAX_SEQ_LEN = 128


def tiny_model(seed: int = 0, num_kv_heads: int = 2) -> LlamaForCausalLM:
    """A real Llama, just small: GQA, RoPE, RMSNorm, SwiGLU — all the moving parts."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=num_kv_heads,  # GQA: kv heads != q heads
        max_position_embeddings=MAX_SEQ_LEN,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config)
    model.eval()
    # No EOS: we want every request to run to max_tokens so lengths stay staggered.
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


def reference_greedy(model, prompt: list[int], max_new_tokens: int) -> list[int]:
    input_ids = torch.tensor([prompt], dtype=torch.long)
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return out[0, len(prompt) :].tolist()


def build_engine(model, max_slots: int, max_prefills_per_step: int = 1) -> LLMEngine:
    cfg = EngineConfig(
        max_slots=max_slots,
        max_seq_len=MAX_SEQ_LEN,
        max_prefills_per_step=max_prefills_per_step,
        device="cpu",
        dtype="float32",
    )
    return LLMEngine(model, cfg, eos_token_ids=[])


# --------------------------------------------------------------------------- #


def test_single_request_matches_hf():
    model = tiny_model()
    prompt = [7, 11, 42, 3, 99]
    expected = reference_greedy(model, prompt, 16)

    engine = build_engine(model, max_slots=4)
    req = engine.add_request(prompt, SamplingParams(max_tokens=16, temperature=0.0))
    engine.run_to_completion()

    assert req.output_token_ids == expected
    assert req.finish_reason.value == "length"


@pytest.mark.parametrize("max_slots", [2, 4, 8])
def test_staggered_batch_matches_sequential(max_slots: int):
    """8 requests, 8 different prompt lengths, 8 different output lengths.

    With max_slots < 8 the engine is forced to evict and re-admit mid-flight,
    which is exactly the path where slot reuse bugs live.
    """
    model = tiny_model(seed=1)
    torch.manual_seed(123)

    prompts = [
        torch.randint(1, VOCAB_SIZE, (length,)).tolist()
        for length in [3, 17, 5, 32, 8, 1, 24, 11]
    ]
    max_tokens = [12, 4, 20, 7, 15, 30, 2, 9]

    expected = [reference_greedy(model, p, n) for p, n in zip(prompts, max_tokens)]

    engine = build_engine(model, max_slots=max_slots)
    reqs = [
        engine.add_request(p, SamplingParams(max_tokens=n, temperature=0.0))
        for p, n in zip(prompts, max_tokens)
    ]
    engine.run_to_completion()

    for i, (req, want) in enumerate(zip(reqs, expected)):
        assert req.output_token_ids == want, (
            f"request {i} (prompt_len={len(prompts[i])}, max_tokens={max_tokens[i]}) "
            f"diverged at token {_first_divergence(req.output_token_ids, want)}"
        )


def test_requests_arriving_mid_generation():
    """New requests joining a batch that is already decoding must not disturb it."""
    model = tiny_model(seed=2)
    torch.manual_seed(7)

    early_prompt = torch.randint(1, VOCAB_SIZE, (20,)).tolist()
    late_prompts = [torch.randint(1, VOCAB_SIZE, (n,)).tolist() for n in (4, 13, 6)]

    expected_early = reference_greedy(model, early_prompt, 40)
    expected_late = [reference_greedy(model, p, 10) for p in late_prompts]

    engine = build_engine(model, max_slots=8)
    early = engine.add_request(early_prompt, SamplingParams(max_tokens=40, temperature=0.0))

    # Let the long request get well into its generation before anyone else shows up.
    for _ in range(12):
        engine.step()

    late = [
        engine.add_request(p, SamplingParams(max_tokens=10, temperature=0.0))
        for p in late_prompts
    ]
    engine.run_to_completion()

    assert early.output_token_ids == expected_early
    for req, want in zip(late, expected_late):
        assert req.output_token_ids == want


def test_multi_head_kv_grouping():
    """num_key_value_heads == num_attention_heads (MHA) as well as GQA."""
    model = tiny_model(seed=3, num_kv_heads=4)
    prompt = [5, 6, 7, 8]
    expected = reference_greedy(model, prompt, 12)

    engine = build_engine(model, max_slots=2)
    req = engine.add_request(prompt, SamplingParams(max_tokens=12, temperature=0.0))
    engine.run_to_completion()
    assert req.output_token_ids == expected


def test_slots_are_released():
    model = tiny_model(seed=4)
    engine = build_engine(model, max_slots=2)
    for _ in range(6):
        engine.add_request([1, 2, 3], SamplingParams(max_tokens=3, temperature=0.0))
    engine.run_to_completion()
    assert engine.scheduler.num_running == 0
    assert engine.scheduler.num_waiting == 0
    assert len(engine.scheduler.free_slots) == 2


def test_stop_token_ends_request():
    model = tiny_model(seed=5)
    prompt = [1, 2, 3]
    expected = reference_greedy(model, prompt, 8)
    stop_id = expected[3]

    engine = build_engine(model, max_slots=1)
    req = engine.add_request(
        prompt, SamplingParams(max_tokens=8, temperature=0.0, stop_token_ids=[stop_id])
    )
    engine.run_to_completion()

    assert req.finish_reason.value == "stop"
    assert req.output_token_ids == expected[:3]


def test_context_limit_is_enforced():
    model = tiny_model(seed=6)
    engine = build_engine(model, max_slots=1)
    req = engine.add_request(
        list(range(1, 60)), SamplingParams(max_tokens=1000, temperature=0.0)
    )
    engine.run_to_completion()
    assert req.finish_reason.value == "context"
    assert req.seq_len <= MAX_SEQ_LEN


def test_prompt_longer_than_context_is_rejected():
    model = tiny_model(seed=7)
    engine = build_engine(model, max_slots=1)
    req = engine.add_request(list(range(MAX_SEQ_LEN + 5)), SamplingParams(max_tokens=4))
    engine.run_to_completion()
    assert req.finish_reason.value == "context"
    assert req.output_token_ids == []


def test_sampling_respects_per_row_temperature():
    """Rows with temperature 0 stay deterministic even when batched with hot rows."""
    from src.engine.sampling import sample

    torch.manual_seed(0)
    logits = torch.randn(4, 64)
    temps = torch.tensor([0.0, 1.5, 0.0, 0.9])
    top_p = torch.ones(4)
    top_k = torch.zeros(4, dtype=torch.long)

    greedy = logits.argmax(-1)
    for _ in range(10):
        out = sample(logits, temps, top_p, top_k)
        assert out[0] == greedy[0]
        assert out[2] == greedy[2]


def test_top_k_never_empties_a_row():
    from src.engine.sampling import sample

    torch.manual_seed(0)
    logits = torch.randn(3, 128)
    out = sample(
        logits,
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.01, 1.0, 0.5]),  # aggressive top_p
        torch.tensor([1, 0, 5]),
    )
    assert out.shape == (3,)
    assert (out >= 0).all() and (out < 128).all()


def _first_divergence(a: list[int], b: list[int]) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))
