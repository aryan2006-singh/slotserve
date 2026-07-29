"""Bridge between the synchronous engine loop and asyncio.

The engine is a tight `while True: step()` loop that must never block on I/O, and
PyTorch releases the GIL inside its kernels. So the engine runs on a dedicated
thread and communicates with the asyncio side through per-request queues.

Two rules keep this correct:
  * Only the engine thread ever touches engine/scheduler/cache state.
  * Tokens cross the boundary via `loop.call_soon_threadsafe`.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass

from src.engine.engine import LLMEngine
from src.engine.request import Request
from src.engine.sampling import SamplingParams


@dataclass
class StreamChunk:
    token_id: int | None
    finished: bool
    finish_reason: str | None = None


class AsyncLLMEngine:
    def __init__(self, engine: LLMEngine, idle_sleep: float = 0.0005) -> None:
        self.engine = engine
        self.idle_sleep = idle_sleep

        self._loop: asyncio.AbstractEventLoop | None = None
        self._submit_queue: queue.Queue = queue.Queue()
        self._streams: dict[str, asyncio.Queue] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # Rolling throughput window for /stats.
        self._window_start = time.monotonic()
        self._window_tokens = 0
        self.tokens_per_second = 0.0
        self.max_batch_seen = 0

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._run, name="engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    # --------------------------------------------------------------- public
    async def generate(
        self,
        prompt_token_ids: list[int],
        params: SamplingParams,
        request_id: str,
    ):
        """Async generator yielding `StreamChunk`s until the request finishes."""
        stream: asyncio.Queue = asyncio.Queue()
        self._streams[request_id] = stream
        self._submit_queue.put((request_id, prompt_token_ids, params))
        try:
            while True:
                chunk: StreamChunk = await stream.get()
                yield chunk
                if chunk.finished:
                    return
        except asyncio.CancelledError:
            self._submit_queue.put(("__abort__", request_id, None))
            raise
        finally:
            self._streams.pop(request_id, None)

    def stats(self) -> dict:
        base = self.engine.stats()
        base.update(
            {
                "tokens_per_second": round(self.tokens_per_second, 2),
                "max_batch_seen": self.max_batch_seen,
            }
        )
        return base

    # -------------------------------------------------------- engine thread
    def _run(self) -> None:
        engine = self.engine
        while not self._stop.is_set():
            self._drain_submissions()

            if not engine.has_work:
                time.sleep(self.idle_sleep)
                continue

            finished, step_stats = engine.step()
            self.max_batch_seen = max(self.max_batch_seen, step_stats.batch_size)
            self._window_tokens += step_stats.tokens_out
            self._tick_throughput()

            # Emit newly generated tokens, then terminal chunks.
            for req in engine.scheduler.running:
                self._flush_tokens(req)
            for req in finished:
                self._flush_tokens(req)
                self._emit(
                    req.request_id,
                    StreamChunk(
                        token_id=None,
                        finished=True,
                        finish_reason=req.finish_reason.value if req.finish_reason else None,
                    ),
                )

    def _drain_submissions(self) -> None:
        while True:
            try:
                item = self._submit_queue.get_nowait()
            except queue.Empty:
                return
            if item[0] == "__abort__":
                self.engine.abort(item[1])
                continue
            request_id, prompt_token_ids, params = item
            self.engine.add_request(prompt_token_ids, params, request_id=request_id)

    def _flush_tokens(self, req: Request) -> None:
        if not req.new_token_ids:
            return
        tokens, req.new_token_ids = req.new_token_ids, []
        for token in tokens:
            self._emit(req.request_id, StreamChunk(token_id=token, finished=False))

    def _emit(self, request_id: str, chunk: StreamChunk) -> None:
        stream = self._streams.get(request_id)
        if stream is None or self._loop is None:
            return
        self._loop.call_soon_threadsafe(stream.put_nowait, chunk)

    def _tick_throughput(self, window: float = 2.0) -> None:
        now = time.monotonic()
        elapsed = now - self._window_start
        if elapsed >= window:
            self.tokens_per_second = self._window_tokens / elapsed
            self._window_start = now
            self._window_tokens = 0
