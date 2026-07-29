"""Head-to-head: sequential vs. static batching vs. continuous batching.

No HTTP server, no GPU required. This is the benchmark to run first, because it
isolates the scheduling win from every other variable — same model, same
process, same prompts, same greedy decoding, same machine.

The workload deliberately has a heavy-tailed output length distribution (most
requests short, a few very long). That is what real traffic looks like, and it
is the exact condition under which static batching collapses: the whole batch is
held hostage by its longest member.

    python bench/offline_compare.py --model HuggingFaceTB/SmolLM2-135M-Instruct
    python bench/offline_compare.py --tiny     # random weights, no download

Output is written to bench/results/offline-compare.json.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.engine import EngineConfig, LLMEngine, SamplingParams  # noqa: E402


def make_workload(num_requests: int, seed: int) -> list[tuple[int, int]]:
    """(prompt_len, max_tokens) pairs with a long tail on output length."""
    rng = random.Random(seed)
    workload = []
    for _ in range(num_requests):
        prompt_len = rng.randint(16, 128)
        # 80% short, 20% long — the tail is the whole point.
        max_tokens = rng.randint(8, 32) if rng.random() < 0.8 else rng.randint(128, 256)
        workload.append((prompt_len, max_tokens))
    return workload


def build_tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=688,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval()
    model.generation_config.eos_token_id = None
    model.generation_config.pad_token_id = 0
    return model


# ---------------------------------------------------------------- strategies
@torch.inference_mode()
def run_sequential(model, prompts, max_tokens_list) -> float:
    start = time.perf_counter()
    for prompt, max_tokens in zip(prompts, max_tokens_list):
        model.generate(
            torch.tensor([prompt], device=model.device),
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    return time.perf_counter() - start


@torch.inference_mode()
def run_static_batched(model, prompts, max_tokens_list, batch_size: int, pad_id: int) -> float:
    """Left-padded fixed batches, every member generating max(batch) tokens.

    This is what you get from a naive `model.generate` server. Note the two
    sources of waste: padding on the prompt side, and short requests continuing
    to burn compute until the longest member of their batch finishes.
    """
    start = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        budget = max(max_tokens_list[i : i + batch_size])
        width = max(len(p) for p in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long,
                               device=model.device)
        attention_mask = torch.zeros((len(chunk), width), dtype=torch.long,
                                     device=model.device)
        for row, prompt in enumerate(chunk):
            input_ids[row, width - len(prompt) :] = torch.tensor(prompt, device=model.device)
            attention_mask[row, width - len(prompt) :] = 1
        model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=budget,
            do_sample=False,
            use_cache=True,
        )
    return time.perf_counter() - start


def run_continuous(model, prompts, max_tokens_list, max_slots: int, max_seq_len: int):
    config = EngineConfig(
        max_slots=max_slots,
        max_seq_len=max_seq_len,
        device=str(model.device),
        dtype=str(next(model.parameters()).dtype).replace("torch.", ""),
    )
    engine = LLMEngine(model, config, eos_token_ids=[])
    for prompt, max_tokens in zip(prompts, max_tokens_list):
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, temperature=0.0))
    start = time.perf_counter()
    finished = engine.run_to_completion()
    elapsed = time.perf_counter() - start
    latencies = [r.e2e_latency() for r in finished if r.e2e_latency() is not None]
    return elapsed, latencies, engine.stats()


# --------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--tiny", action="store_true", help="random weights, no download")
    parser.add_argument("--num-requests", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8, help="static batching width")
    parser.add_argument("--max-slots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-sequential", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--out", default="bench/results/offline-compare.json")
    args = parser.parse_args()

    if args.tiny or args.model is None:
        model = build_tiny_model()
        vocab_size, pad_id = model.config.vocab_size, 0
        model_name = "tiny-llama-random-6L-256d"
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model).eval()
        model.set_attn_implementation("eager")
        vocab_size = model.config.vocab_size
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        model_name = args.model

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    workload = make_workload(args.num_requests, args.seed)
    rng = random.Random(args.seed)
    prompts = [
        [rng.randint(1, vocab_size - 1) for _ in range(plen)] for plen, _ in workload
    ]
    max_tokens_list = [n for _, n in workload]
    max_seq_len = max(p + n for p, n in workload) + 8
    total_output = sum(max_tokens_list)

    print(f"model={model_name} device={device}")
    print(
        f"workload: {args.num_requests} requests, "
        f"prompt {min(p for p,_ in workload)}-{max(p for p,_ in workload)} tokens, "
        f"output {min(max_tokens_list)}-{max(max_tokens_list)} tokens, "
        f"{total_output} output tokens total\n"
    )

    results = {
        "model": model_name,
        "device": device,
        "num_requests": args.num_requests,
        "total_output_tokens": total_output,
        "strategies": {},
    }

    if not args.skip_sequential:
        print("1/3 sequential (one request at a time)...")
        elapsed = run_sequential(model, prompts, max_tokens_list)
        results["strategies"]["sequential"] = {
            "wall_s": round(elapsed, 3),
            "tok_per_s": round(total_output / elapsed, 1),
        }
        print(f"     {elapsed:.2f}s  {total_output / elapsed:.1f} tok/s\n")

    print(f"2/3 static batching (batch={args.batch_size})...")
    elapsed = run_static_batched(model, prompts, max_tokens_list, args.batch_size, pad_id)
    results["strategies"]["static_batching"] = {
        "wall_s": round(elapsed, 3),
        "tok_per_s": round(total_output / elapsed, 1),
        "batch_size": args.batch_size,
    }
    print(f"     {elapsed:.2f}s  {total_output / elapsed:.1f} tok/s\n")

    print(f"3/3 continuous batching (slots={args.max_slots})...")
    elapsed, latencies, stats = run_continuous(
        model, prompts, max_tokens_list, args.max_slots, max_seq_len
    )
    results["strategies"]["continuous_batching"] = {
        "wall_s": round(elapsed, 3),
        "tok_per_s": round(total_output / elapsed, 1),
        "max_slots": args.max_slots,
        "mean_request_latency_s": round(statistics.fmean(latencies), 3) if latencies else None,
        "engine_stats": stats,
    }
    print(f"     {elapsed:.2f}s  {total_output / elapsed:.1f} tok/s\n")

    base = results["strategies"].get("static_batching", {}).get("tok_per_s")
    cont = results["strategies"]["continuous_batching"]["tok_per_s"]
    if base:
        results["speedup_vs_static"] = round(cont / base, 2)
        print(f"continuous / static  = {cont / base:.2f}x")
    if "sequential" in results["strategies"]:
        seq = results["strategies"]["sequential"]["tok_per_s"]
        results["speedup_vs_sequential"] = round(cont / seq, 2)
        print(f"continuous / sequential = {cont / seq:.2f}x")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
