"""Independent small causal decoder for iteration, not TinyLlama or its checkpoint.

An optional adjacent ``profile.json`` (schema ``merlin.iteration_workload_profile.v1``,
``workload_id: causal_decoder``) is a selected capture input. Without one, the model and its
inputs are exactly the original 8-token, 32-wide, 4-head development workload. Each optional key
keeps its default when absent:

* ``seq`` (8, at most 16), ``hidden`` (32, at most 64), ``heads`` (4, dividing ``hidden``) and
  ``ffn`` (64, at most 256) size the model; widths stay small so the workload remains certifiable;
* ``kv_heads`` (``heads``): fewer key/value heads than query heads, each shared by a group of
  ``heads // kv_heads`` query heads;
* ``rope`` (false): rotary position embedding of queries and keys (rotate-half form);
* ``decode_step`` (false): a single-token step whose key/value cache tensors of ``cache_len``
  (default ``seq - 1``, at most 15) positions are explicit extra inputs; the step attends to the
  cache plus its own position and returns that position's logits.
"""

import hashlib
import json
from pathlib import Path

import torch

PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"
REQUIRED_KEYS = frozenset({"schema", "workload_id"})
DEFAULTS = {
    "seq": 8,
    "hidden": 32,
    "heads": 4,
    "kv_heads": None,
    "ffn": 64,
    "rope": False,
    "decode_step": False,
    "cache_len": None,
}
OPTIONAL_KEYS = frozenset(DEFAULTS)
VOCABULARY = 64


class CausalDecoder(torch.nn.Module):
    def __init__(self, seq=8, hidden=32, heads=4, kv_heads=None, ffn=64, rope=False, cache_len=None):
        super().__init__()
        kv_heads = heads if kv_heads is None else kv_heads
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, hidden // heads
        self.length = seq if cache_len is None else 1
        self.rope = rope
        kv_width = kv_heads * self.head_dim
        self.split_sizes = (hidden, kv_width, kv_width)
        self.embedding = torch.nn.Embedding(VOCABULARY, hidden)
        self.norm = torch.nn.RMSNorm(hidden)
        self.qkv = torch.nn.Linear(hidden, hidden + 2 * kv_width, bias=False)
        self.projection = torch.nn.Linear(hidden, hidden, bias=False)
        self.gate = torch.nn.Linear(hidden, ffn, bias=False)
        self.up = torch.nn.Linear(hidden, ffn, bias=False)
        self.down = torch.nn.Linear(ffn, hidden, bias=False)
        self.head = torch.nn.Linear(hidden, VOCABULARY, bias=False)
        if cache_len is None:
            mask = torch.full((seq, seq), float("-inf")).triu(diagonal=1)
            self.register_buffer("mask", mask)
        if rope:
            start = 0 if cache_len is None else cache_len
            positions = torch.arange(start, start + self.length, dtype=torch.float32)
            frequencies = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
            angles = torch.outer(positions, frequencies).repeat(1, 2)
            self.register_buffer("rope_cos", angles.cos())
            self.register_buffer("rope_sin", angles.sin())

    def _rotate(self, states):
        half = self.head_dim // 2
        rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
        return states * self.rope_cos + rotated * self.rope_sin

    def _share(self, states):
        if self.kv_heads == self.heads:
            return states
        _, _, length, width = states.shape
        group = self.heads // self.kv_heads
        expanded = states[:, :, None].expand(1, self.kv_heads, group, length, width)
        return expanded.reshape(1, self.heads, length, width)

    def _decode(self, tokens, cache):
        hidden = self.embedding(tokens)
        if self.kv_heads == self.heads:
            query, key, value = self.qkv(self.norm(hidden)).chunk(3, dim=-1)
        else:
            query, key, value = self.qkv(self.norm(hidden)).split(self.split_sizes, dim=-1)
        length = self.length
        query = query.reshape(1, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.reshape(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(1, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.rope:
            query, key = self._rotate(query), self._rotate(key)
        if cache is not None:
            key = torch.cat((cache[0], key), dim=2)
            value = torch.cat((cache[1], value), dim=2)
        key, value = self._share(key), self._share(value)
        scores = (query @ key.transpose(-2, -1)) / (self.head_dim**0.5)
        if cache is None:
            scores = scores + self.mask
        attention = torch.softmax(scores, dim=-1) @ value
        hidden = hidden + self.projection(attention.transpose(1, 2).reshape(1, length, self.hidden))
        normalized = self.norm(hidden)
        hidden = hidden + self.down(torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized))
        return self.head(hidden)

    def forward(self, tokens):
        return self._decode(tokens, None)


class CachedDecoderStep(CausalDecoder):
    """One token attending to explicit key/value cache inputs of shape (1, kv_heads, cache_len, head_dim)."""

    def forward(self, token, key_cache, value_cache):
        return self._decode(token, (key_cache, value_cache))


def validate_profile(profile):
    """The complete profile with defaults filled in, or ``ValueError`` for any unsupported field."""
    if (
        not isinstance(profile, dict)
        or not REQUIRED_KEYS <= set(profile)
        or set(profile) - REQUIRED_KEYS - OPTIONAL_KEYS
    ):
        raise ValueError("causal decoder profile has an unsupported shape")
    if profile["schema"] != PROFILE_SCHEMA or profile["workload_id"] != "causal_decoder":
        raise ValueError("causal decoder profile has an unsupported identity")
    if any(value is None for value in profile.values()):
        raise ValueError("causal decoder profile has a null field")
    for key in ("rope", "decode_step"):
        if key in profile and type(profile[key]) is not bool:
            raise ValueError(f"causal decoder profile has an invalid {key}")
    selected = {**DEFAULTS, **{key: value for key, value in profile.items() if key not in REQUIRED_KEYS}}
    if selected["kv_heads"] is None:
        selected["kv_heads"] = selected["heads"]
    if selected["cache_len"] is None and selected["decode_step"]:
        selected["cache_len"] = selected["seq"] - 1
    for key, maximum in (("seq", 16), ("hidden", 64), ("heads", 64), ("kv_heads", 64), ("ffn", 256)):
        if type(selected[key]) is not int or not 1 <= selected[key] <= maximum:
            raise ValueError(f"causal decoder profile has an invalid {key}")
    if selected["hidden"] % selected["heads"] or selected["heads"] % selected["kv_heads"]:
        raise ValueError("causal decoder profile needs heads dividing hidden and kv_heads dividing heads")
    if selected["rope"] and (selected["hidden"] // selected["heads"]) % 2:
        raise ValueError("causal decoder profile needs an even head width for rotary embedding")
    if not selected["decode_step"] and "cache_len" in profile:
        raise ValueError("causal decoder profile names a cache length without a decode step")
    if selected["decode_step"] and (type(selected["cache_len"]) is not int or not 1 <= selected["cache_len"] <= 15):
        raise ValueError("causal decoder profile has an invalid cache_len")
    return selected


def _selected_profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return validate_profile({"schema": PROFILE_SCHEMA, "workload_id": "causal_decoder"}), None
    if path.is_symlink() or not path.is_file():
        raise ValueError("causal decoder profile must be an adjacent ordinary file")
    raw = path.read_bytes()
    return validate_profile(json.loads(raw)), hashlib.sha256(raw).hexdigest()


def _cache(shape, multiplier, offset):
    values = torch.arange(torch.Size(shape).numel(), dtype=torch.int64).reshape(shape)
    return (((values * multiplier + offset) % 31) - 15).to(torch.float32) / 32


def get_model_and_inputs():
    profile, profile_sha256 = _selected_profile()
    decode = profile["decode_step"]
    model = (CachedDecoderStep if decode else CausalDecoder)(
        profile["seq"],
        profile["hidden"],
        profile["heads"],
        profile["kv_heads"],
        profile["ffn"],
        profile["rope"],
        profile["cache_len"] if decode else None,
    ).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    if profile_sha256 is not None:
        model.session_provenance = {
            "workload_id": "causal_decoder",
            "workload_role": "iteration",
            "input_source": "seeded_synthetic",
            "profile_sha256": profile_sha256,
        }
    if decode:
        cache_shape = (1, profile["kv_heads"], profile["cache_len"], profile["hidden"] // profile["heads"])
        token = torch.tensor([[profile["cache_len"] % VOCABULARY]], dtype=torch.int64)
        return model, (token, _cache(cache_shape, 29, 7), _cache(cache_shape, 17, 3))
    return model, (torch.arange(profile["seq"], dtype=torch.int64).reshape(1, profile["seq"]),)
