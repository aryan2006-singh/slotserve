"""Engine configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

Quantization = Literal["none", "int8", "nf4", "fp4"]


@dataclass
class EngineConfig:
    model: str = "meta-llama/Llama-3.2-1B-Instruct"
    dtype: str = "auto"  # "auto" | "float16" | "bfloat16" | "float32"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    quantization: Quantization = "none"

    # Cache sizing. max_slots is the maximum number of concurrent sequences;
    # max_seq_len is prompt + generated tokens for any one sequence.
    max_slots: int = 32
    max_seq_len: int = 2048

    # Scheduling.
    max_prefills_per_step: int = 1

    # Serving.
    host: str = "0.0.0.0"
    port: int = 8000

    seed: int | None = None
    trust_remote_code: bool = False

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolve_dtype(self) -> torch.dtype:
        if self.dtype != "auto":
            return getattr(torch, self.dtype)
        device = self.resolve_device()
        if device.type != "cuda":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def kv_cache_bytes(self, num_layers: int, num_kv_heads: int, head_dim: int) -> int:
        """Bytes the KV cache will reserve. Print this before you OOM, not after."""
        element_size = torch.empty((), dtype=self.resolve_dtype()).element_size()
        per_tensor = self.max_slots * num_kv_heads * self.max_seq_len * head_dim
        return 2 * num_layers * per_tensor * element_size
