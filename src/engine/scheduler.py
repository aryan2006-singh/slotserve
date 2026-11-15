"""Admission control and slot allocation.

The scheduling policy is deliberately simple and deliberately explicit, because
the policy *is* the interesting part of a serving system:

  * FCFS admission from a waiting queue.
  * A request is admitted only if (a) a physical cache slot is free and (b) its
    prompt fits in `max_seq_len`.
  * At most `max_prefills_per_step` prefills run per engine step. Prefill is
    compute-bound and long; letting several run back to back starves the
    decoding requests and shows up as a spiky TPOT. One per step bounds that
    jitter. This is the same tension vLLM resolves with chunked prefill.
  * No preemption. A request that has been admitted holds its slot until it
    finishes. Adding preemption (swap or recompute) is the natural next step and
    is what lets you oversubscribe slots.
"""

from __future__ import annotations

from collections import deque

from .request import FinishReason, Request, RequestStatus


class Scheduler:
    def __init__(
        self,
        max_slots: int,
        max_seq_len: int,
        max_prefills_per_step: int = 1,
    ) -> None:
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.max_prefills_per_step = max_prefills_per_step

        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.free_slots: list[int] = list(reversed(range(max_slots)))

        self.num_preempted = 0
        self.num_rejected = 0

    # ------------------------------------------------------------------ queue
    def add(self, request: Request) -> None:
        if request.prompt_len >= self.max_seq_len:
            request.status = RequestStatus.FINISHED
            request.finish_reason = FinishReason.CONTEXT
            self.num_rejected += 1
            return
        self.waiting.append(request)

    def abort(self, request_id: str) -> bool:
        for req in list(self.waiting):
            if req.request_id == request_id:
                self.waiting.remove(req)
                req.status = RequestStatus.ABORTED
                req.finish_reason = FinishReason.ABORT
                return True
        for req in self.running:
            if req.request_id == request_id:
                req.status = RequestStatus.ABORTED
                req.finish_reason = FinishReason.ABORT
                return True
        return False

    # -------------------------------------------------------------- admission
    def admit(self) -> list[Request]:
        """Move up to `max_prefills_per_step` requests from waiting to running."""
        admitted: list[Request] = []
        while (
            self.waiting
            and self.free_slots
            and len(admitted) < self.max_prefills_per_step
        ):
            req = self.waiting.popleft()
            if req.status == RequestStatus.ABORTED:
                continue
            req.slot = self.free_slots.pop()
            req.status = RequestStatus.PREFILL
            self.running.append(req)
            admitted.append(req)
        return admitted

    # ---------------------------------------------------------------- retire
    def retire_finished(self) -> list[Request]:
        """Release slots held by finished requests and return them."""
        finished = [r for r in self.running if r.is_finished]
        if not finished:
            return []
        for req in finished:
            if req.slot is not None:
                self.free_slots.append(req.slot)
                req.slot = None
        self.running = [r for r in self.running if not r.is_finished]
        return finished

    # ----------------------------------------------------------------- state
    @property
    def num_running(self) -> int:
        return len(self.running)

    @property
    def num_waiting(self) -> int:
        return len(self.waiting)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def slot_utilization(self) -> float:
        return 1.0 - len(self.free_slots) / self.max_slots

    def slot_snapshot(self) -> list[dict | None]:
        """One entry per physical slot: who holds it and how full it is.

        Read from the server thread while the engine thread mutates state, so
        copy the list first and tolerate a request finishing mid-read. The
        result is for display only; it can be one step stale.
        """
        slots: list[dict | None] = [None] * self.max_slots
        for req in list(self.running):
            slot = req.slot
            if slot is None or not 0 <= slot < self.max_slots:
                continue
            slots[slot] = {
                "request_id": req.request_id,
                "status": req.status.value,
                "prompt_tokens": req.prompt_len,
                "generated": req.num_generated,
                "max_tokens": req.params.max_tokens,
                "seq_len": req.seq_len,
            }
        return slots
