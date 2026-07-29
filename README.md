# slotserve

An LLM inference server with **token-level continuous batching**, built on a
slot-addressed KV cache instead of HuggingFace's `DynamicCache`.

> **Headline (fill in with your own run):** _N.NNx_ higher output throughput than
> static batching on a heavy-tailed workload, with byte-identical greedy output —
> verified against HuggingFace `generate()` by a 27-test suite.
>
> Reference run on 4 CPU threads, 24 requests, 6-layer model:
> **238.8 → 651.5 tok/s (2.73x)**. Reproduce with `make compare-tiny`.

```
┌──────────┐   HTTP    ┌───────────────┐  thread   ┌──────────────────────┐
│ clients  │──────────▶│ FastAPI       │  bridge   │  engine loop         │
│          │◀─ SSE ────│ OpenAI-compat │◀─────────▶│  (single thread)     │
└──────────┘           └───────────────┘           └──────────┬───────────┘
                                                              │
                              ┌───────────────────────────────▼─────────────┐
                              │ step():                                     │
                              │  1. admit + prefill  ≤1 request             │
                              │  2. decode ALL running requests, 1 token    │
                              │  3. release finished slots                  │
                              └───────────────────────────────┬─────────────┘
                                                              │
                        ┌─────────────────────────────────────▼───────────┐
                        │ SlotKVCache  [slots, kv_heads, max_len, dim]     │
                        │  slot 0 ████████░░░░░░░░  req A, len 8           │
                        │  slot 1 ████████████████  req B, len 16          │
                        │  slot 2 ██░░░░░░░░░░░░░░  req C, len 2  (joined  │
                        │  slot 3 ░░░░░░░░░░░░░░░░  free      mid-flight)  │
                        └──────────────────────────────────────────────────┘
```

---

## The problem it solves

With **static batching**, a batch forms once and every member waits for the
slowest member. Queue a 20-token request behind a 2000-token request and the
short one pays the long one's latency. Worse, as members finish, their rows keep
consuming compute on padding until the whole batch retires.

With **continuous batching**, the batch is re-formed every single decode step. A
request that finishes releases its cache slot immediately; a queued request takes
that slot on the next step. Throughput stops being a function of the unluckiest
request in the batch.

The data structure is what makes this possible. `DynamicCache` grows by
`torch.cat` each step and assumes uniform sequence length, so a request cannot
join or leave without reindexing everyone else. `SlotKVCache` preallocates
`[max_slots, num_kv_heads, max_seq_len, head_dim]` per layer and gives each
request a fixed physical slot, addressed by index.

---

## Quickstart (no GPU required)

```bash
git clone <your-repo> && cd slotserve
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU wheels
pip install -r requirements.txt

make test          # 27 tests, ~10s, CPU, no model download
make compare-tiny  # static vs continuous batching, random-weight model
```

Then serve a real (small) model:

```bash
make serve MODEL=HuggingFaceTB/SmolLM2-135M-Instruct SLOTS=16

curl localhost:8000/v1/completions -H 'content-type: application/json' -d '{
  "prompt": "The three hardest problems in LLM serving are",
  "max_tokens": 64, "temperature": 0.7, "stream": true
}'
```

Because the API is OpenAI-compatible, you can point the official client — or any
benchmark harness written for vLLM — straight at it. That is what makes the
comparison in `BENCHMARKS.md` meaningful.

---

## Correctness comes first

A batching engine is only useful if it is **invisible**: under greedy decoding, a
request must emit identical tokens whether it ran alone or shared a batch with
seven others that joined and left at arbitrary steps.

```
tests/test_equivalence.py   staggered lengths, mid-flight arrivals, slot reuse,
                            GQA and MHA, stop tokens, context limits
tests/test_fuzz.py          randomised slot counts / lengths / arrival times
tests/test_server.py        SSE framing, thread bridge, concurrent batching
```

The reference is HF `generate()` run one prompt at a time. **This suite already
earned its keep**: it caught an off-by-one in `write_positions` where each decode
step wrote its K/V one position too far, leaving a zeroed cache slot that the row
then attended to. The single-request test passed anyway. Only the staggered
batch test failed. That is precisely the class of bug that silently degrades
output quality in a server that looks like it works.

```bash
pytest tests/ -v
```

---

## Design decisions (and their costs)

**One prefill per step.** Prefill is compute-bound and long; running several
back-to-back starves decoding requests and shows up as spiky inter-token latency.
Capping at one bounds the jitter. The cost is lower prefill throughput under a
burst of arrivals. vLLM resolves this tension with *chunked prefill* — splitting a
long prompt across steps so it interleaves with decode. That's the next thing to
build here.

**No preemption.** An admitted request holds its slot until it finishes. Simple,
predictable, and it means slots cannot be oversubscribed. Real servers preempt —
either swapping a victim's KV cache to host memory or dropping and recomputing
it — which lets you admit more requests than you have slots for.

**Padded reads instead of paged attention.** Reading a window with
`keys[slot_ids, :, :kv_length, :]` is a *gather*: it materialises a copy every
layer, every step, padded to the longest sequence in the batch. With one 2000-token
sequence and 31 short ones, we read 32×2000 positions to use a few hundred.
PagedAttention avoids this entirely with a custom CUDA kernel that reads
non-contiguous blocks in place. **This is the single biggest reason this server
loses to vLLM at high batch sizes**, and quantifying that gap is the most useful
thing in the benchmark table.

**Explicit 4D attention masks.** Slots have different fill levels, so the mask
must hide each row's unwritten positions — and the previous occupant's data still
sitting in the slot. Handing the model a precomputed additive mask is slower than
letting it build a 2D one, but it's the only version I could verify bit-for-bit.

---

## Benchmarking honestly

`bench/load_test.py` has two modes and the difference matters:

| Mode | Arrivals | Measures | Flatters the server? |
|---|---|---|---|
| `closed` | N clients, each waits for its own response | peak throughput | **yes** |
| `open` | Poisson at fixed rate, independent of server | what users feel | no |

**Report open-loop numbers.** In closed-loop, an overloaded server just gets fewer
requests, so latency looks fine and the failure is invisible. In open-loop,
arrivals keep coming and queueing delay lands in TTFT, which is where it belongs.

Prompts are randomly generated per request, not repeated — a repeated prompt would
let any sane server hit a prefix cache and report a prefill time unrelated to real
traffic.

```bash
make serve MODEL=... &
make bench LABEL="fp16 continuous"     # sweeps rates, regenerates BENCHMARKS.md
```

Metrics reported: TTFT (p50/p95/p99), inter-token latency, end-to-end p95,
aggregate output tok/s, and KV cache bytes. Prometheus exposition at `/metrics`
with buckets chosen for LLM serving rather than web-request defaults.

---

## No GPU? Here's the plan

Everything above runs on CPU, including the full test suite and the
static-vs-continuous comparison. What CPU **cannot** give you is a credible
absolute throughput number or any quantization result (`bitsandbytes` is
CUDA-only, and the loader refuses to pretend otherwise).

Do it in two stages:

**Stage 1 — free, on your laptop.** Build and verify everything. Run
`make compare-tiny` for the scheduling speedup. This is a real, defensible result:
it isolates the algorithmic win from the hardware.

**Stage 2 — a few hours of rented GPU, once.** When the code is finished and
tested, rent a single mid-tier GPU (L4 / A10G / RTX 4090 on RunPod, Vast.ai, or
Lambda; generally under ~$1/hr — check live rates) and run the benchmark sweep
plus the quantization comparison in one sitting. Two or three focused hours is
enough for the whole table. Free Colab and Kaggle T4 sessions also work for the
smaller models if you'd rather spend nothing.

Debug on CPU, benchmark on GPU. Renting time to fix a bug you could have caught
locally is how the budget disappears.

---

## Roadmap — the phases that make this a flagship project

Phase 1 and 2 are done. Each remaining phase is a self-contained increment with
its own measurable result.

- [x] **Phase 1** — naive baseline, one request at a time, measured
- [x] **Phase 2** — slot KV cache + token-level continuous batching, correctness-verified
- [ ] **Phase 3** — quantization sweep: fp16 → int8 → int4 (AWQ/GPTQ). Plot the
      quality/latency/memory frontier. Quality on a fixed 500-prompt eval set, not vibes.
- [ ] **Phase 4** — speculative decoding with a small draft model. Report token
      acceptance rate and end-to-end speedup, and find the regime where it *loses*.
- [ ] **Phase 5** — the honest comparison: benchmark against vLLM and TensorRT-LLM
      on identical hardware and load. You will lose. Write up precisely why.
      That analysis is the most senior-sounding artifact in the whole project.

Also worth building, in rough order of value: chunked prefill, prefix caching for
shared system prompts, preemption with swap-or-recompute, and CUDA graphs for the
decode step.

---

## Layout

```
src/engine/
  kv_cache.py     SlotKVCache / SlotLayer + 4D mask construction  ← the core
  engine.py       the step() loop: admit, prefill, decode, retire
  scheduler.py    admission control and slot allocation
  sampling.py     per-row temperature / top-k / top-p
  model_loader.py loading + quantization configs
  config.py       EngineConfig, incl. KV cache size estimation
src/server/
  app.py          FastAPI, OpenAI-compatible, SSE streaming
  async_engine.py thread bridge between the blocking loop and asyncio
  metrics.py      Prometheus histograms with serving-appropriate buckets
bench/
  offline_compare.py  sequential vs static vs continuous (no server, no GPU)
  load_test.py        open/closed-loop HTTP load generator
  report.py           results/*.json → BENCHMARKS.md
tests/
```

## License

MIT
