"""Batched sampling with per-request parameters.

Every row in the batch belongs to a different request, so temperature / top-p /
top-k must be applied per-row rather than per-batch. This is the part people
usually get wrong when they hand-roll a server: they apply one temperature to
the whole batch, which silently changes the output distribution for every
request that asked for something else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0  # 0.0 => greedy
    top_p: float = 1.0
    top_k: int = 0  # 0 => disabled
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")


@torch.no_grad()
def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    top_k: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row.

    Args:
        logits:      [B, V] raw logits (any float dtype; upcast internally).
        temperature: [B] float, <=0 means greedy for that row.
        top_p:       [B] float in (0, 1].
        top_k:       [B] long, 0 means disabled.

    Returns:
        [B] long tensor of sampled token ids.
    """
    logits = logits.float()
    _, vocab_size = logits.shape

    greedy_tokens = logits.argmax(dim=-1)
    is_greedy = temperature <= 0
    if bool(is_greedy.all()):
        return greedy_tokens

    scaled = logits / temperature.clamp(min=1e-5).unsqueeze(1)

    # --- top-k: mask everything ranked at or beyond each row's k ----------
    if bool((top_k > 0).any()):
        k_max = int(top_k.clamp(max=vocab_size).max().item())
        kth_values = scaled.topk(k_max, dim=-1).values  # [B, k_max], descending
        # Rows with top_k == 0 (disabled) index the last column, so nothing is cut.
        k_index = torch.where(
            top_k > 0, top_k.clamp(1, k_max), torch.full_like(top_k, k_max)
        ) - 1
        threshold = kth_values.gather(1, k_index.unsqueeze(1))
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    # --- top-p (nucleus) ---------------------------------------------------
    if bool((top_p < 1.0).any()):
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # Keep tokens whose *preceding* cumulative mass is below p, so the token
        # that crosses the threshold is always kept and no row can be emptied.
        drop = (cumulative - probs) > top_p.unsqueeze(1)
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        scaled = torch.full_like(scaled, float("-inf")).scatter(1, sorted_idx, sorted_logits)

    probs = scaled.softmax(dim=-1)
    sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
    return torch.where(is_greedy, greedy_tokens, sampled)
