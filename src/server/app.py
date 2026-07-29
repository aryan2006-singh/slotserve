"""OpenAI-compatible HTTP server.

Endpoints:
    POST /v1/completions        (stream=true|false)
    POST /v1/chat/completions   (stream=true|false)
    GET  /health
    GET  /stats
    GET  /metrics               Prometheus exposition format

Being OpenAI-compatible is not cosmetic: it means you can point existing
benchmarking tools and the official `openai` client at your server and compare it
to vLLM without changing the client. Anything else and the comparison is suspect.

Run:  python -m src.server.app --model HuggingFaceTB/SmolLM2-135M-Instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request as HTTPRequest
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.engine.config import EngineConfig
from src.engine.engine import LLMEngine
from src.engine.model_loader import load_model_and_tokenizer, model_memory_footprint
from src.engine.sampling import SamplingParams
from src.server.async_engine import AsyncLLMEngine
from src.server.metrics import Metrics

log = logging.getLogger(__name__)

STATE: dict = {}


# --------------------------------------------------------------------- schema
class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    stop_token_ids: list[int] = Field(default_factory=list)

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_token_ids=self.stop_token_ids,
        )


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=5.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )


# ----------------------------------------------------------------- lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    config: EngineConfig = STATE["config"]
    model, tokenizer = load_model_and_tokenizer(config)
    engine = LLMEngine(model, config, eos_token_ids=_eos_ids(tokenizer))

    text_config = getattr(model.config, "text_config", model.config)
    head_dim = getattr(
        text_config,
        "head_dim",
        text_config.hidden_size // text_config.num_attention_heads,
    )
    reserved = config.kv_cache_bytes(
        text_config.num_hidden_layers,
        getattr(text_config, "num_key_value_heads", text_config.num_attention_heads),
        head_dim,
    )
    log.info(
        "kv cache reserved: %.1f MB (%d slots x %d tokens) | weights: %s",
        reserved / 2**20,
        config.max_slots,
        config.max_seq_len,
        model_memory_footprint(model),
    )

    async_engine = AsyncLLMEngine(engine)
    async_engine.start()

    STATE["tokenizer"] = tokenizer
    STATE["engine"] = async_engine
    STATE["metrics"] = Metrics()
    STATE["started_at"] = time.time()
    try:
        yield
    finally:
        async_engine.stop()


app = FastAPI(title="slotserve", version="0.1.0", lifespan=lifespan)


@app.exception_handler(ValueError)
async def value_error_handler(request: HTTPRequest, exc: ValueError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


def _eos_ids(tokenizer) -> list[int]:
    eos = tokenizer.eos_token_id
    if eos is None:
        return []
    return eos if isinstance(eos, list) else [eos]


# ----------------------------------------------------------------- endpoints
@app.get("/health")
async def health():
    return {"status": "ok", "uptime_s": round(time.time() - STATE["started_at"], 1)}


@app.get("/stats")
async def stats():
    return STATE["engine"].stats()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return STATE["metrics"].render(STATE["engine"].stats())


@app.post("/v1/completions")
async def completions(body: CompletionRequest, http_request: HTTPRequest):
    tokenizer = STATE["tokenizer"]
    prompt_ids = tokenizer(body.prompt, add_special_tokens=True).input_ids
    return await _run(prompt_ids, body.to_sampling_params(), body.stream, "text_completion")


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest, http_request: HTTPRequest):
    tokenizer = STATE["tokenizer"]
    messages = [m.model_dump() for m in body.messages]
    if tokenizer.chat_template is None:
        raise HTTPException(400, "this model has no chat template; use /v1/completions")
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    # Recent tokenizers return a BatchEncoding (dict-like) instead of a bare
    # list of ids; unwrap it so downstream code always gets plain ints.
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids.input_ids
    return await _run(prompt_ids, body.to_sampling_params(), body.stream, "chat.completion")


async def _run(prompt_ids, params: SamplingParams, stream: bool, kind: str):
    engine: AsyncLLMEngine = STATE["engine"]
    tokenizer = STATE["tokenizer"]
    metrics: Metrics = STATE["metrics"]
    request_id = f"cmpl-{uuid.uuid4().hex}"
    arrival = time.monotonic()

    if stream:
        return StreamingResponse(
            _stream_response(engine, tokenizer, metrics, prompt_ids, params,
                             request_id, kind, arrival),
            media_type="text/event-stream",
        )

    token_ids: list[int] = []
    finish_reason = None
    first_token_at = None
    async for chunk in engine.generate(prompt_ids, params, request_id):
        if chunk.token_id is not None:
            if first_token_at is None:
                first_token_at = time.monotonic()
            token_ids.append(chunk.token_id)
        if chunk.finished:
            finish_reason = chunk.finish_reason

    metrics.observe(
        ttft=(first_token_at - arrival) if first_token_at else None,
        e2e=time.monotonic() - arrival,
        prompt_tokens=len(prompt_ids),
        output_tokens=len(token_ids),
    )
    return JSONResponse(
        {
            "id": request_id,
            "object": kind,
            "created": int(time.time()),
            "choices": [
                {
                    "index": 0,
                    "text": tokenizer.decode(token_ids, skip_special_tokens=True),
                    "message": {
                        "role": "assistant",
                        "content": tokenizer.decode(token_ids, skip_special_tokens=True),
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        }
    )


async def _stream_response(
    engine, tokenizer, metrics, prompt_ids, params, request_id, kind, arrival
):
    first_token_at = None
    count = 0
    # Incremental detokenization: decoding token-by-token breaks multi-byte
    # characters and BPE word boundaries, so keep a buffer and emit the delta.
    buffer: list[int] = []
    emitted = ""
    async for chunk in engine.generate(prompt_ids, params, request_id):
        if chunk.token_id is not None:
            if first_token_at is None:
                first_token_at = time.monotonic()
            count += 1
            buffer.append(chunk.token_id)
            text = tokenizer.decode(buffer, skip_special_tokens=True)
            delta = text[len(emitted) :]
            if not delta:
                continue
            emitted = text
            payload = {
                "id": request_id,
                "object": kind,
                "created": int(time.time()),
                "choices": [
                    {"index": 0, "text": delta, "delta": {"content": delta},
                     "finish_reason": None}
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        if chunk.finished:
            payload = {
                "id": request_id,
                "object": kind,
                "choices": [
                    {"index": 0, "text": "", "delta": {},
                     "finish_reason": chunk.finish_reason}
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

    metrics.observe(
        ttft=(first_token_at - arrival) if first_token_at else None,
        e2e=time.monotonic() - arrival,
        prompt_tokens=len(prompt_ids),
        output_tokens=count,
    )


# ------------------------------------------------------------------ frontend
# Mounted last so it only catches requests that don't match an API route above
# (Starlette matches routes in registration order).
app.mount(
    "/",
    StaticFiles(directory=str(Path(__file__).parent / "static"), html=True),
    name="static",
)


# ---------------------------------------------------------------------- main
def build_config_from_args(argv=None) -> EngineConfig:
    parser = argparse.ArgumentParser(description="slotserve")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--quantization", default="none", choices=["none", "int8", "nf4", "fp4"]
    )
    parser.add_argument("--max-slots", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-prefills-per-step", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)
    return EngineConfig(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        max_slots=args.max_slots,
        max_seq_len=args.max_seq_len,
        max_prefills_per_step=args.max_prefills_per_step,
        host=args.host,
        port=args.port,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
    )


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = build_config_from_args()
    STATE["config"] = config
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    main()
