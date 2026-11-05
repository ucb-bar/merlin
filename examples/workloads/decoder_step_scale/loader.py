"""Independent performance-scale single-token decoder step (Phase 2 form source; not a held-out model).

One new token attends over an explicit key/value cache passed as inputs, so every projection is a
one-row contraction that streams its whole weight. Optional adjacent ``profile.json`` as for the
prefill sibling, plus ``cache`` (the cached length).
"""

import json
from pathlib import Path

import torch

_DEFAULT = {"hidden": 896, "heads": 16, "kv_heads": 8, "ffn": 2432, "layers": 2, "cache": 95, "vocab": 4096}


def _profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return dict(_DEFAULT)
    doc = json.loads(path.read_bytes())
    if set(doc) != set(_DEFAULT) or any(type(v) is not int or v < 1 for v in doc.values()):
        raise ValueError("decoder-step profile must declare exactly the default keys as positive integers")
    if doc["hidden"] % doc["heads"] or doc["heads"] % doc["kv_heads"] or (doc["hidden"] // doc["heads"]) % 2:
        raise ValueError("decoder-step profile widths are inconsistent")
    return doc


def _rotate(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Step(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.heads, self.kv_heads, self.dim = p["heads"], p["kv_heads"], p["hidden"] // p["heads"]
        self.embed = torch.nn.Embedding(p["vocab"], p["hidden"])
        self.layers = torch.nn.ModuleList()
        for _ in range(p["layers"]):
            layer = torch.nn.Module()
            layer.norm1, layer.norm2 = torch.nn.RMSNorm(p["hidden"]), torch.nn.RMSNorm(p["hidden"])
            layer.q = torch.nn.Linear(p["hidden"], p["hidden"], bias=False)
            layer.k = torch.nn.Linear(p["hidden"], self.kv_heads * self.dim, bias=False)
            layer.v = torch.nn.Linear(p["hidden"], self.kv_heads * self.dim, bias=False)
            layer.o = torch.nn.Linear(p["hidden"], p["hidden"], bias=False)
            layer.gate = torch.nn.Linear(p["hidden"], p["ffn"], bias=False)
            layer.up = torch.nn.Linear(p["hidden"], p["ffn"], bias=False)
            layer.down = torch.nn.Linear(p["ffn"], p["hidden"], bias=False)
            self.layers.append(layer)
        self.norm = torch.nn.RMSNorm(p["hidden"])
        self.head = torch.nn.Linear(p["hidden"], p["vocab"], bias=False)
        inv = 1.0 / (10000 ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        angle = torch.cat((p["cache"] * inv, p["cache"] * inv))
        self.register_buffer("cos", angle.cos()[None, None, None])
        self.register_buffer("sin", angle.sin()[None, None, None])

    def forward(self, token, keys, values):
        x = self.embed(token)
        group = self.heads // self.kv_heads
        for index, layer in enumerate(self.layers):
            h = layer.norm1(x)
            q = layer.q(h).reshape(1, 1, self.heads, self.dim).transpose(1, 2)
            k = layer.k(h).reshape(1, 1, self.kv_heads, self.dim).transpose(1, 2)
            v = layer.v(h).reshape(1, 1, self.kv_heads, self.dim).transpose(1, 2)
            q, k = q * self.cos + _rotate(q) * self.sin, k * self.cos + _rotate(k) * self.sin
            k = torch.cat((keys[index], k), dim=2).repeat_interleave(group, dim=1)
            v = torch.cat((values[index], v), dim=2).repeat_interleave(group, dim=1)
            attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.dim**0.5), dim=-1) @ v
            x = x + layer.o(attn.transpose(1, 2).reshape(1, 1, -1))
            h = layer.norm2(x)
            x = x + layer.down(torch.nn.functional.silu(layer.gate(h)) * layer.up(h))
        return self.head(self.norm(x))


def get_model_and_inputs():
    p = _profile()
    model = Step(p).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    dim = p["hidden"] // p["heads"]
    shape = (p["layers"], 1, p["kv_heads"], p["cache"], dim)
    count = int(torch.tensor(shape).prod())
    keys = (torch.arange(count, dtype=torch.float32).reshape(shape) % 17 - 8) / 16
    values = (torch.arange(count, dtype=torch.float32).reshape(shape) % 13 - 6) / 16
    return model, (torch.tensor([[11]], dtype=torch.int64), keys, values)
