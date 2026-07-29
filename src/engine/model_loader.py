"""Model loading, including the quantization sweep.

`bitsandbytes` (int8 / nf4 / fp4) is CUDA-only. On CPU the loader falls back to
fp32 and says so loudly rather than silently producing a number you might put in
a benchmark table.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import EngineConfig

log = logging.getLogger(__name__)


def load_model_and_tokenizer(config: EngineConfig):
    device = config.resolve_device()
    dtype = config.resolve_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "dtype": dtype,
        "trust_remote_code": config.trust_remote_code,
    }

    quantization = config.quantization
    if quantization != "none":
        if device.type != "cuda":
            log.warning(
                "quantization=%s requires CUDA; bitsandbytes has no CPU kernels. "
                "Falling back to unquantized %s. Do not report this row as a "
                "quantized benchmark result.",
                quantization,
                dtype,
            )
            quantization = "none"
        else:
            kwargs["quantization_config"] = _quant_config(quantization, dtype)

    model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs)
    if quantization == "none":
        model = model.to(device)
    model.eval()

    # Eager attention: our 4D masks are explicit and eager is the only backend
    # guaranteed to honour them bit-for-bit. Switch to sdpa once you have the
    # equivalence test passing with it too (it usually does; verify, don't assume).
    if hasattr(model.config, "_attn_implementation"):
        model.set_attn_implementation("eager")

    log.info(
        "loaded %s | device=%s dtype=%s quantization=%s params=%.2fB",
        config.model,
        device,
        dtype,
        quantization,
        sum(p.numel() for p in model.parameters()) / 1e9,
    )
    return model, tokenizer


def _quant_config(quantization: str, compute_dtype: torch.dtype):
    from transformers import BitsAndBytesConfig

    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization in ("nf4", "fp4"):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quantization,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"unknown quantization: {quantization}")


def model_memory_footprint(model) -> dict:
    """Actual bytes held by weights, split by dtype. Useful for the report table."""
    by_dtype: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.dtype)
        by_dtype[key] = by_dtype.get(key, 0) + param.numel() * param.element_size()
    for buf in model.buffers():
        key = str(buf.dtype)
        by_dtype[key] = by_dtype.get(key, 0) + buf.numel() * buf.element_size()
    total = sum(by_dtype.values())
    return {
        "total_mb": round(total / 2**20, 1),
        "by_dtype_mb": {k: round(v / 2**20, 1) for k, v in by_dtype.items()},
    }
