"""Tensor-only BF16 cache boundary around already reconstructed source modules.

Inputs are ids [B,Q], explicit absolute positions [B,Q], an additive BF16 mask
[B,1,Q,L+Q], then (K0,V0,...,Kn,Vn), each [B,Hkv,L,D]. Outputs are source logits
and the same ordered cache pairs at length L+Q. K is stored after the source's
RoPE and before GQA expansion. Every call delegates attention to the existing
patched source, which dynamically quantizes the complete concatenated K/V.

This wrapper does not load a model, change its policy, quantize weights, or
claim generated execution. Each fixed prefill/decode shape needs its own source
capture, extraction, independent reference, compilation, and cache seam proof.
"""
from __future__ import annotations

import torch
from torch import nn


def empty_cache(model, batch_size=1):
    """Allocate zero-length BF16 K/V carriers, without touching source state."""
    config = model.config
    weight = model.model.embed_tokens.weight
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    shape = (batch_size, config.num_key_value_heads, 0, head_dim)
    return tuple(torch.empty(shape, dtype=torch.bfloat16, device=weight.device)
                 for _ in range(2 * config.num_hidden_layers))


def phase_positions_and_mask(input_ids, past_length):
    """Prepare an unpadded, contiguous causal phase outside the captured graph."""
    if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or type(past_length) is not int or past_length < 0:
        raise ValueError("phase needs int64 [B,Q] ids and a nonnegative integer past length")
    batch, query = input_ids.shape
    if batch < 1 or query < 1:
        raise ValueError("empty batches or query phases are unsupported")
    positions = torch.arange(past_length, past_length + query, device=input_ids.device)[None].expand(batch, -1)
    keys = torch.arange(past_length + query, device=input_ids.device)
    allowed = keys[None, None, None, :] <= positions[:, None, :, None]
    mask = torch.where(allowed, 0.0, torch.finfo(torch.bfloat16).min).to(torch.bfloat16)
    return positions, mask


def validate_phase(input_ids, position_ids, attention_mask, cache_tensors):
    """Eager admission for the initial unpadded append-only phase contract.

    Do this before export/execution. Data-dependent position and mask checks
    intentionally stay outside the captured numerical graph.
    """
    if not cache_tensors or len(cache_tensors) % 2:
        raise ValueError("cache inputs require an ordered K/V pair for every layer")
    first = cache_tensors[0]
    if first.ndim != 4:
        raise ValueError("cache must be [B,Hkv,L,D]")
    expected_positions, expected_mask = phase_positions_and_mask(input_ids, first.shape[2])
    if position_ids.dtype != torch.int64 or not torch.equal(position_ids, expected_positions):
        raise ValueError("positions must append immediately after the consumed cache")
    if attention_mask.dtype != torch.bfloat16 or not torch.equal(attention_mask, expected_mask):
        raise ValueError("mask must be the exact unpadded BF16 causal append mask")
    if any(value.ndim != 4 or value.dtype != torch.bfloat16 or value.shape != first.shape
           or value.device != input_ids.device for value in cache_tensors):
        raise ValueError("all layers require matching BF16 cache lengths, geometry, and device")


def validate_prefill_phase(input_ids, position_ids, attention_mask):
    """Admit an unpadded prefill without creating an empty tensor boundary."""
    positions, mask = phase_positions_and_mask(input_ids, 0)
    if position_ids.dtype != torch.int64 or not torch.equal(position_ids, positions):
        raise ValueError("prefill positions must begin at zero")
    if attention_mask.dtype != torch.bfloat16 or not torch.equal(attention_mask, mask):
        raise ValueError("prefill mask must be the exact BF16 causal mask")


class _AppendTensorCache:
    """Ephemeral source update hook; external state consists only of tensors."""

    def __init__(self, pairs):
        self.pairs = list(pairs)
        self.updated = [False] * len(pairs)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx < 0 or layer_idx >= len(self.pairs) or self.updated[layer_idx]:
            raise ValueError("source layer must append its cache exactly once")
        old_key, old_value = self.pairs[layer_idx]
        if key_states.dtype != torch.bfloat16 or value_states.dtype != torch.bfloat16:
            raise ValueError("source cache update must retain BF16 K/V")
        if old_key is None and old_value is None:
            key, value = key_states, value_states
        elif old_key is None or old_value is None:
            raise ValueError("initial source K/V must both be absent")
        else:
            key = torch.cat((old_key, key_states), dim=-2)
            value = torch.cat((old_value, value_states), dim=-2)
        # Cache outputs are serialized by the source exporter. Preserve their
        # raw BF16 words while materializing layout inside the source layer.
        key, value = key.contiguous(), value.contiguous()
        self.pairs[layer_idx] = (key, value)
        self.updated[layer_idx] = True
        return key, value

    def get_seq_length(self, layer_idx=0):
        key = self.pairs[layer_idx][0]
        return 0 if key is None else key.shape[-2]

    def tensors(self):
        if not all(self.updated):
            raise ValueError("source omitted a layer cache update")
        return tuple(value for pair in self.pairs for value in pair)


class TensorCacheLogits(nn.Module):
    """Pinned Llama source decoder with explicit masks, positions, and BF16 K/V.

    Construct only from the authenticated raw loader's already patched model.
    Keeping the model under the same ``model`` attribute as CausalLogits retains
    source module names for native-FP8 site discovery. No persistent Python cache
    survives a call, and no source module is replaced or modified.
    """

    def __init__(self, model):
        super().__init__()
        if model.config.model_type != "llama":
            raise ValueError("tensor cache source currently admits only the pinned Llama architecture")
        if model.training:
            raise ValueError("source model must already be in evaluation mode")
        if any(getattr(layer.self_attn, "_atlas_ref_attn", {}).get("variant") != "text"
               for layer in model.model.layers):
            raise ValueError("source attention must already carry the original reference policy")
        self.model = model
        self.layer_count = model.config.num_hidden_layers
        self.kv_heads = model.config.num_key_value_heads
        self.head_dim = getattr(model.config, "head_dim", None) or model.config.hidden_size // model.config.num_attention_heads

    def forward(self, input_ids, position_ids, attention_mask, *cache_tensors):
        if len(cache_tensors) != 2 * self.layer_count:
            raise ValueError("one BF16 K/V input pair per source layer is required")
        if input_ids.ndim != 2:
            raise ValueError("ids must be a two-dimensional tensor")
        batch = input_ids.shape[0]
        past = cache_tensors[0].shape[2]
        for value in cache_tensors:
            if value.dtype != torch.bfloat16 or value.shape != (batch, self.kv_heads, past, self.head_dim):
                raise ValueError("cache requires BF16 post-RoPE, pre-GQA source head geometry")
        cache = _AppendTensorCache(tuple(zip(cache_tensors[::2], cache_tensors[1::2])))
        return self._source_forward(input_ids, position_ids, attention_mask, cache, past)

    def _source_forward(self, input_ids, position_ids, attention_mask, cache, past):
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64 or position_ids.dtype != torch.int64:
            raise ValueError("ids and absolute positions must be int64 tensors")
        batch, query = input_ids.shape
        if position_ids.shape != input_ids.shape or attention_mask.shape != (batch, 1, query, past + query):
            raise ValueError("phase positions or mask geometry differs from the cache transition")
        if attention_mask.dtype != torch.bfloat16:
            raise ValueError("source additive mask must retain BF16")
        hidden = self.model.model.embed_tokens(input_ids)
        position_embeddings = self.model.model.rotary_emb(hidden, position_ids)
        for layer in self.model.model.layers:
            hidden = layer(hidden, attention_mask=attention_mask, position_ids=position_ids,
                           past_key_values=cache, position_embeddings=position_embeddings,
                           use_cache=True)
            if isinstance(hidden, tuple):
                raise ValueError("source decoder-layer return ABI differs from the pinned tensor ABI")
        logits = self.model.lm_head(self.model.model.norm(hidden))
        return (logits, *cache.tensors())


class TensorPrefillLogits(TensorCacheLogits):
    """Three positive-shape inputs; initial K/V exists only as Python None.

    The first source update returns its post-RoPE K/V directly. No zero-sized
    tensor, cache placeholder, or empty-prefix concatenation enters the graph.
    """

    def forward(self, input_ids, position_ids, attention_mask):
        cache = _AppendTensorCache(tuple((None, None) for _ in range(self.layer_count)))
        return self._source_forward(input_ids, position_ids, attention_mask, cache, 0)
