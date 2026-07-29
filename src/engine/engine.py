"""The engine loop: continuous batching over a slot-addressed KV cache.

One `step()` does, in order:

  1. Admit up to `max_prefills_per_step` waiting requests and prefill them.
  2. Run a single decode forward pass over *every* request already generating,
     regardless of how long each has been running or how long its prompt was.
  3. Release the slots of anything that finished.

Step 2 is the whole point. In static batching, a batch is formed once and every
member waits for the slowest member to finish before any new request starts.
With a 20-token request queued behind a 2000-token request, the short one pays
the long one's latency. Here a finished request frees its slot immediately and a
waiting request takes it on the very next step.
"""

from __future__ import annotations

import time
import uuid

import torch

from .config import EngineConfig
from .kv_cache import SlotKVCache, build_decode_mask, build_prefill_mask
from .request import FinishReason, Request, RequestStatus
from .sampling import SamplingParams, sample
from .scheduler import Scheduler


class StepStats:
    __slots__ = ("num_prefill", "num_decode", "batch_size", "duration", "tokens_out")

    def __init__(self) -> None:
        self.num_prefill = 0
        self.num_decode = 0
        self.batch_size = 0
        self.duration = 0.0
        self.tokens_out = 0


class LLMEngine:
    def __init__(
        self,
        model,
        config: EngineConfig,
        eos_token_ids: list[int] | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.config = config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        text_config = getattr(model.config, "text_config", model.config)
        self.num_layers = text_config.num_hidden_layers
        self.eos_token_ids = set(eos_token_ids or [])
        gen_eos = getattr(model.generation_config, "eos_token_id", None)
        if gen_eos is not None:
            self.eos_token_ids.update(gen_eos if isinstance(gen_eos, list) else [gen_eos])

        self.cache = SlotKVCache(
            num_layers=self.num_layers,
            max_slots=config.max_slots,
            max_seq_len=config.max_seq_len,
        )
        self.scheduler = Scheduler(
            max_slots=config.max_slots,
            max_seq_len=config.max_seq_len,
            max_prefills_per_step=config.max_prefills_per_step,
        )

        self.generator: torch.Generator | None = None
        if config.seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(config.seed)

        # Cumulative counters, read by /stats and the Prometheus exporter.
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_steps = 0
        self.total_prefill_time = 0.0
        self.total_decode_time = 0.0

    # --------------------------------------------------------------- public
    def add_request(
        self,
        prompt_token_ids: list[int],
        params: SamplingParams,
        request_id: str | None = None,
    ) -> Request:
        req = Request(
            request_id=request_id or uuid.uuid4().hex,
            prompt_token_ids=list(prompt_token_ids),
            params=params,
        )
        self.scheduler.add(req)
        return req

    def abort(self, request_id: str) -> bool:
        return self.scheduler.abort(request_id)

    @property
    def has_work(self) -> bool:
        return self.scheduler.has_work

    @torch.inference_mode()
    def step(self) -> tuple[list[Request], StepStats]:
        """Advance every in-flight request by one token. Returns finished requests."""
        stats = StepStats()
        started = time.perf_counter()

        for req in self.scheduler.admit():
            t0 = time.perf_counter()
            self._prefill(req)
            self.total_prefill_time += time.perf_counter() - t0
            stats.num_prefill += 1

        decoding = [
            r
            for r in self.scheduler.running
            if r.status == RequestStatus.RUNNING and not r.is_finished
        ]
        if decoding:
            t0 = time.perf_counter()
            self._decode(decoding)
            self.total_decode_time += time.perf_counter() - t0
            stats.num_decode = len(decoding)

        stats.batch_size = self.scheduler.num_running
        stats.tokens_out = stats.num_prefill + stats.num_decode
        self.total_output_tokens += stats.tokens_out
        self.total_steps += 1

        finished = self.scheduler.retire_finished()
        stats.duration = time.perf_counter() - started
        return finished, stats

    def run_to_completion(self, max_steps: int = 1_000_000) -> list[Request]:
        """Drain the queue. Convenience wrapper for offline batch inference."""
        done: list[Request] = []
        steps = 0
        while self.has_work and steps < max_steps:
            finished, _ = self.step()
            done.extend(finished)
            steps += 1
        return done

    # -------------------------------------------------------------- internals
    def _prefill(self, req: Request) -> None:
        prompt_len = req.prompt_len
        slot_ids = torch.tensor([req.slot], dtype=torch.long, device=self.device)
        write_positions = torch.zeros(1, dtype=torch.long, device=self.device)
        self.cache.set_batch(slot_ids, write_positions, kv_length=prompt_len)

        input_ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.device)
        position_ids = torch.arange(prompt_len, device=self.device).unsqueeze(0)
        mask = build_prefill_mask(prompt_len, prompt_len, self.dtype, self.device)

        out = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=torch.arange(prompt_len, device=self.device),
            use_cache=True,
        )
        next_token = self._sample_rows(out.logits[:, -1, :], [req])[0]

        req.schedule_time = time.monotonic()
        req.status = RequestStatus.RUNNING
        self.total_prompt_tokens += prompt_len
        self._accept_token(req, int(next_token))

    def _decode(self, reqs: list[Request]) -> None:
        # `seq_len` counts prompt + already-generated tokens, so the token we are
        # about to feed lives at absolute position seq_len - 1. That is also where
        # its K/V must land. Getting this index wrong is the classic continuous
        # batching bug: the row attends to an uninitialised cache position.
        positions = torch.tensor(
            [r.seq_len - 1 for r in reqs], dtype=torch.long, device=self.device
        )
        kv_length = int(positions.max().item()) + 1

        slot_ids = torch.tensor([r.slot for r in reqs], dtype=torch.long, device=self.device)
        self.cache.set_batch(slot_ids, positions, kv_length=kv_length)

        input_ids = torch.tensor(
            [[r.last_token()] for r in reqs], dtype=torch.long, device=self.device
        )
        position_ids = positions.unsqueeze(1)
        # +1: each row may attend to the token it is writing this very step.
        mask = build_decode_mask(positions + 1, kv_length, self.dtype, self.device)

        out = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            cache_position=torch.tensor([kv_length - 1], device=self.device),
            use_cache=True,
        )
        tokens = self._sample_rows(out.logits[:, -1, :], reqs)
        for req, token in zip(reqs, tokens.tolist()):
            self._accept_token(req, int(token))

    def _sample_rows(self, logits: torch.Tensor, reqs: list[Request]) -> torch.Tensor:
        device = logits.device
        temperature = torch.tensor(
            [r.params.temperature for r in reqs], dtype=torch.float32, device=device
        )
        top_p = torch.tensor([r.params.top_p for r in reqs], dtype=torch.float32, device=device)
        top_k = torch.tensor([r.params.top_k for r in reqs], dtype=torch.long, device=device)
        return sample(logits, temperature, top_p, top_k, generator=self.generator)

    def _accept_token(self, req: Request, token: int) -> None:
        now = time.monotonic()
        if req.first_token_time is None:
            req.first_token_time = now

        is_stop = (token in self.eos_token_ids or token in req.params.stop_token_ids) and (
            not req.params.ignore_eos
        )
        if is_stop:
            # EOS is a control token: record it as the reason, not as output.
            self._finish(req, FinishReason.STOP, now)
            return

        req.output_token_ids.append(token)
        req.new_token_ids.append(token)

        if req.num_generated >= req.params.max_tokens:
            self._finish(req, FinishReason.LENGTH, now)
        elif req.seq_len >= self.config.max_seq_len:
            self._finish(req, FinishReason.CONTEXT, now)

    @staticmethod
    def _finish(req: Request, reason: FinishReason, now: float) -> None:
        req.status = RequestStatus.FINISHED
        req.finish_reason = reason
        req.finish_time = now

    # ------------------------------------------------------------------ info
    def stats(self) -> dict:
        return {
            "running": self.scheduler.num_running,
            "waiting": self.scheduler.num_waiting,
            "slot_utilization": round(self.scheduler.slot_utilization(), 4),
            "total_steps": self.total_steps,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "prefill_time_s": round(self.total_prefill_time, 3),
            "decode_time_s": round(self.total_decode_time, 3),
            "kv_cache_mb": round(self.cache.bytes_allocated() / 2**20, 1),
        }
