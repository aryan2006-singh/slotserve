"""Per-request state carried through the scheduler and engine."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from .sampling import SamplingParams


class RequestStatus(str, enum.Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


class FinishReason(str, enum.Enum):
    STOP = "stop"  # hit EOS or a stop token
    LENGTH = "length"  # hit max_tokens
    CONTEXT = "context"  # ran out of cache slots for this sequence
    ABORT = "abort"  # client disconnected


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams

    status: RequestStatus = RequestStatus.WAITING
    slot: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: FinishReason | None = None

    # Timing, all monotonic seconds.
    arrival_time: float = field(default_factory=time.monotonic)
    schedule_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None

    # Filled by the engine so the async layer can stream without locking.
    new_token_ids: list[int] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def seq_len(self) -> int:
        """Total tokens currently resident in this request's cache slot."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_generated(self) -> int:
        return len(self.output_token_ids)

    @property
    def is_finished(self) -> bool:
        return self.status in (RequestStatus.FINISHED, RequestStatus.ABORTED)

    def last_token(self) -> int:
        return self.output_token_ids[-1] if self.output_token_ids else self.prompt_token_ids[-1]

    # ------------------------------------------------------------- metrics
    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def tpot(self) -> float | None:
        """Mean time per output token, excluding the first."""
        if self.finish_time is None or self.first_token_time is None:
            return None
        if self.num_generated < 2:
            return None
        return (self.finish_time - self.first_token_time) / (self.num_generated - 1)

    def e2e_latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time
