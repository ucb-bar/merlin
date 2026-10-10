"""Independent performance-scale vision encoder (Phase 2 form source; not a held-out model).

Strided patch embedding, LayerNorm, multi-head self-attention and a GELU MLP over a few hundred
patch tokens. Optional adjacent ``profile.json`` selects the widths.
"""

import json
from pathlib import Path

import torch

_DEFAULT = {"image": 136, "patch": 8, "hidden": 384, "heads": 8, "mlp": 1536, "layers": 2}


def _profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return dict(_DEFAULT)
    doc = json.loads(path.read_bytes())
    if set(doc) != set(_DEFAULT) or any(type(v) is not int or v < 1 for v in doc.values()):
        raise ValueError("vision profile must declare exactly the default keys as positive integers")
    if doc["image"] % doc["patch"] or doc["hidden"] % doc["heads"]:
        raise ValueError("vision profile widths are inconsistent")
    return doc


class Layer(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.heads, self.dim = p["heads"], p["hidden"] // p["heads"]
        self.norm1, self.norm2 = torch.nn.LayerNorm(p["hidden"]), torch.nn.LayerNorm(p["hidden"])
        self.qkv = torch.nn.Linear(p["hidden"], 3 * p["hidden"])
        self.proj = torch.nn.Linear(p["hidden"], p["hidden"])
        self.fc1 = torch.nn.Linear(p["hidden"], p["mlp"])
        self.fc2 = torch.nn.Linear(p["mlp"], p["hidden"])

    def forward(self, x):
        b, s, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, s, 3, self.heads, self.dim).permute(2, 0, 3, 1, 4)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.dim**0.5), dim=-1) @ v
        x = x + self.proj(attn.transpose(1, 2).reshape(b, s, d))
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm2(x)), approximate="tanh"))


class VitScale(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, p["hidden"], p["patch"], stride=p["patch"])
        tokens = (p["image"] // p["patch"]) ** 2
        self.position = torch.nn.Parameter(torch.zeros(1, tokens, p["hidden"]))
        self.layers = torch.nn.ModuleList(Layer(p) for _ in range(p["layers"]))
        self.norm = torch.nn.LayerNorm(p["hidden"])

    def forward(self, image):
        x = self.patch(image).flatten(2).transpose(1, 2) + self.position
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def get_model_and_inputs():
    p = _profile()
    model = VitScale(p).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    elements = 3 * p["image"] * p["image"]
    image = torch.arange(elements, dtype=torch.float32).reshape(1, 3, p["image"], p["image"]) / elements
    return model, (image,)
