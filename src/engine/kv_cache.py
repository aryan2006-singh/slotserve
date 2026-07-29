"""Slot-based KV cache — the data structure that makes continuous batching possible.

Why not just use HF's DynamicCache?
-----------------------------------
`DynamicCache` grows by `torch.cat` on every decode step and assumes every row of
the batch has the same length. That means:

  * every step reallocates the whole cache (O(total_tokens) memory traffic per step);
  * a request cannot leave the batch without reindexing everyone else's cache;
  * a request cannot join a batch that is already mid-generation.

Continuous batching needs the opposite: a fixed pool of physical slots that
requests are assigned to and released from, independently of each other.

Layout, per layer:  [max_slots, num_kv_heads, max_seq_len, head_dim]

The engine sets three fields before each forward pass:

  slot_ids        [B] long  — physical slot for each row of the current batch
  write_positions [B] long  — absolute position where this step's K/V is written
  kv_length       int       — how many positions of the cache the model may read

Because slots have different fill levels, the engine must also supply a 4D
additive attention mask that hides positions beyond each row's own length.
See `build_decode_mask` below.

Honest limitation
-----------------
Reading a window with `keys[slot_ids, :, :kv_length, :]` is a *gather*, which
materialises a copy every layer, every step. vLLM avoids this with PagedAttention:
a custom CUDA kernel reads directly from non-contiguous blocks, so nothing is
copied and the cache can be block-sparse instead of padded to `kv_length`. That
gather, plus the padding waste, is the main reason this server loses to vLLM at
high batch sizes — see BENCHMARKS.md.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class SlotLayer(CacheLayerMixin):
    """One transformer layer's worth of slot-addressed KV storage."""

    is_compileable = False
    is_sliding = False
    supports_early_init = True

    def __init__(self, max_slots: int, max_seq_len: int) -> None:
        super().__init__()
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len
        self.max_cache_len = max_seq_len
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.is_initialized = False
        # Set by the owning SlotKVCache before every forward pass.
        self.slot_ids: torch.Tensor | None = None
        self.write_positions: torch.Tensor | None = None
        self.kv_length: int = 0

    # ------------------------------------------------------------------ setup
    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.num_heads = key_states.shape[1]
        self.k_head_dim = key_states.shape[-1]
        self.v_head_dim = value_states.shape[-1]
        self.keys = torch.zeros(
            (self.max_slots, self.num_heads, self.max_seq_len, self.k_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.values = torch.zeros(
            (self.max_slots, self.num_heads, self.max_seq_len, self.v_head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.is_initialized = True

    # ----------------------------------------------------------------- update
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        assert self.slot_ids is not None and self.write_positions is not None
        q_len = key_states.shape[-2]

        if q_len == 1:
            # Decode: one token per row, each row writing to its own position.
            # Advanced indexing on dims 0 and 2 with a slice between them yields
            # shape [B, num_heads, head_dim] — matching key_states[:, :, 0, :].
            self.keys[self.slot_ids, :, self.write_positions, :] = key_states[:, :, 0, :].to(
                self.keys.dtype
            )
            self.values[self.slot_ids, :, self.write_positions, :] = value_states[:, :, 0, :].to(
                self.values.dtype
            )
        else:
            # Prefill: one sequence at a time, contiguous write.
            if self.slot_ids.numel() != 1:
                raise RuntimeError(
                    "Multi-sequence prefill is not supported; the scheduler prefills "
                    "one request per step. See README 'Design decisions'."
                )
            slot = int(self.slot_ids[0])
            start = int(self.write_positions[0])
            self.keys[slot, :, start : start + q_len, :] = key_states[0].to(self.keys.dtype)
            self.values[slot, :, start : start + q_len, :] = value_states[0].to(self.values.dtype)

        # Return the readable window for this micro-batch.
        return (
            self.keys[self.slot_ids, :, : self.kv_length, :],
            self.values[self.slot_ids, :, : self.kv_length, :],
        )

    # ------------------------------------------------------- interface plumbing
    def get_seq_length(self, cache_position: torch.Tensor | None = None) -> int:
        return self.kv_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # (kv_length, kv_offset). We always hand the model an explicit 4D mask,
        # so these values are only used for sanity checks upstream.
        return self.kv_length, 0

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def get_max_length(self) -> int:
        return self.max_seq_len

    def reset(self) -> None:
        if self.keys is not None:
            self.keys.zero_()
            self.values.zero_()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError("Beam search is not supported.")

    def offload(self) -> None:  # pragma: no cover
        pass

    def prefetch(self) -> None:  # pragma: no cover
        pass


class SlotKVCache(Cache):
    """A `Cache` whose layers are all `SlotLayer`s addressed by physical slot id."""

    def __init__(
        self,
        num_layers: int,
        max_slots: int,
        max_seq_len: int,
    ) -> None:
        super().__init__(layers=[SlotLayer(max_slots, max_seq_len) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.max_slots = max_slots
        self.max_seq_len = max_seq_len

    def set_batch(
        self,
        slot_ids: torch.Tensor,
        write_positions: torch.Tensor,
        kv_length: int,
    ) -> None:
        """Point every layer at the slots participating in the next forward pass."""
        for layer in self.layers:
            layer.slot_ids = slot_ids
            layer.write_positions = write_positions
            layer.kv_length = kv_length

    def bytes_allocated(self) -> int:
        total = 0
        for layer in self.layers:
            if layer.keys is not None:
                total += layer.keys.numel() * layer.keys.element_size()
                total += layer.values.numel() * layer.values.element_size()
        return total

    # `Cache.get_seq_length` defers to layer 0, which is what we want.


def build_decode_mask(
    seq_lengths: torch.Tensor,
    kv_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive 4D mask for a decode step: [B, 1, 1, kv_length].

    Row b may attend to cache positions [0, seq_lengths[b]) — that is, its own
    prompt plus everything it has generated, including the token written this
    step. Everything past that is another request's data or uninitialised
    memory, and must be masked to -inf.
    """
    positions = torch.arange(kv_length, device=device).unsqueeze(0)  # [1, kv_length]
    allowed = positions < seq_lengths.unsqueeze(1)  # [B, kv_length]
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[:, None, None, :]


def build_prefill_mask(
    prompt_length: int,
    kv_length: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive 4D causal mask for a single-sequence prefill: [1, 1, q, kv_length]."""
    min_value = torch.finfo(dtype).min
    mask = torch.full((prompt_length, kv_length), min_value, dtype=dtype, device=device)
    # Query i (absolute position i, since prefill starts at 0) may see keys 0..i.
    causal = torch.arange(kv_length, device=device).unsqueeze(0) <= torch.arange(
        prompt_length, device=device
    ).unsqueeze(1)
    mask.masked_fill_(causal, 0.0)
    return mask[None, None, :, :]
