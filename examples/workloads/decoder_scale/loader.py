"""Independent performance-scale decoder block (Phase 2 form source; not a held-out model).

Grouped-query attention with rotary embeddings, RMSNorm and a gated MLP at widths chosen to sit in
the multi-tile / multi-block regime of a small accelerator. An optional adjacent ``profile.json``
selects the widths; without one the defaults below apply.
"""

import json
from pathlib import Path

import torch

_DEFAULT = {"hidden": 832, "heads": 16, "kv_heads": 4, "ffn": 2176, "layers": 2, "seq": 96, "vocab": 4096}


def _profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return dict(_DEFAULT)
    doc = json.loads(path.read_bytes())
    if set(doc) != set(_DEFAULT) or any(type(v) is not int or v < 1 for v in doc.values()):
        raise ValueError("decoder profile must declare exactly the default keys as positive integers")
    if doc["hidden"] % doc["heads"] or doc["heads"] % doc["kv_heads"] or (doc["hidden"] // doc["heads"]) % 2:
        raise ValueError("decoder profile widths are inconsistent")
    return doc


def _rotate(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Block(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.heads, self.kv_heads, self.dim = p["heads"], p["kv_heads"], p["hidden"] // p["heads"]
        self.norm1 = torch.nn.RMSNorm(p["hidden"])
        self.norm2 = torch.nn.RMSNorm(p["hidden"])
        self.q = torch.nn.Linear(p["hidden"], p["hidden"], bias=False)
        self.k = torch.nn.Linear(p["hidden"], self.kv_heads * self.dim, bias=False)
        self.v = torch.nn.Linear(p["hidden"], self.kv_heads * self.dim, bias=False)
        self.o = torch.nn.Linear(p["hidden"], p["hidden"], bias=False)
        self.gate = torch.nn.Linear(p["hidden"], p["ffn"], bias=False)
        self.up = torch.nn.Linear(p["hidden"], p["ffn"], bias=False)
        self.down = torch.nn.Linear(p["ffn"], p["hidden"], bias=False)

    def forward(self, x, cos, sin, mask):
        b, s, _ = x.shape
        h = self.norm1(x)
        q = self.q(h).reshape(b, s, self.heads, self.dim).transpose(1, 2)
        k = self.k(h).reshape(b, s, self.kv_heads, self.dim).transpose(1, 2)
        v = self.v(h).reshape(b, s, self.kv_heads, self.dim).transpose(1, 2)
        q, k = q * cos + _rotate(q) * sin, k * cos + _rotate(k) * sin
        group = self.heads // self.kv_heads
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        scores = (q @ k.transpose(-2, -1)) / (self.dim**0.5) + mask
        attn = torch.softmax(scores, dim=-1) @ v
        x = x + self.o(attn.transpose(1, 2).reshape(b, s, -1))
        h = self.norm2(x)
        return x + self.down(torch.nn.functional.silu(self.gate(h)) * self.up(h))


class DecoderScale(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.embed = torch.nn.Embedding(p["vocab"], p["hidden"])
        self.blocks = torch.nn.ModuleList(Block(p) for _ in range(p["layers"]))
        self.norm = torch.nn.RMSNorm(p["hidden"])
        self.head = torch.nn.Linear(p["hidden"], p["vocab"], bias=False)
        dim = p["hidden"] // p["heads"]
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        angles = torch.outer(torch.arange(p["seq"], dtype=torch.float32), inv)
        angles = torch.cat((angles, angles), dim=-1)
        self.register_buffer("cos", angles.cos()[None, None])
        self.register_buffer("sin", angles.sin()[None, None])
        self.register_buffer("mask", torch.full((p["seq"], p["seq"]), float("-inf")).triu(diagonal=1))

    def forward(self, tokens):
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, self.cos, self.sin, self.mask)
        return self.head(self.norm(x))


def _init(model):
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    return model


def get_model_and_inputs():
    p = _profile()
    model = _init(DecoderScale(p).eval())
    tokens = (torch.arange(p["seq"], dtype=torch.int64) * 7 % p["vocab"]).reshape(1, p["seq"])
    return model, (tokens,)
