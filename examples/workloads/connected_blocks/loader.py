"""Independent complete connected blocks; normal Phase 0 selects their numeric recipe.

An adjacent profile.json selects one bounded public geometry. These are synthetic
compiler-development sources, not checkpoint models, target kernels or device grants.
All returned values, including updated unexpanded KV state and repeated results, are observable.
"""

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

PROFILE_SCHEMA = "merlin.iteration_workload_profile.v1"
WORKLOAD_ID = "connected_blocks"
DEFAULTS = {
    "conv_pool": dict(channels=3, out_channels=16, height=9, width=11, conv_kernel=3,
                      conv_stride=1, conv_padding=1, pool_kernel=3, pool_stride=2, pool_padding=1),
    "residual": dict(channels=8, width=4, out_channels=16, side=8, stride=2, blocks=1),
    "attention": dict(tokens=2, hidden=16, heads=2, kv_heads=1, prefix=3, ffn=32),
    "repeated": dict(steps=2, rows=3, hidden=8, ffn=12, output=8),
}
MAX_ELEMENTS = 16_000_000


def validate_profile(profile):
    """Refuse unknown fields, inconsistent heads and excessive source/reference storage."""
    if not isinstance(profile, dict) or profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError("connected blocks need the selected profile schema")
    if (profile.get("workload_id") != WORKLOAD_ID or not isinstance(profile.get("kind"), str)
            or profile["kind"] not in DEFAULTS):
        raise ValueError("connected blocks need an independent source kind")
    kind = profile["kind"]
    identity = {"schema", "workload_id", "kind"}
    if set(profile) - identity - set(DEFAULTS[kind]):
        raise ValueError("connected block profile has unknown fields")
    p = {**DEFAULTS[kind], **profile}
    for key in DEFAULTS[kind]:
        low = 0 if key in {"prefix", "conv_padding", "pool_padding"} else 1
        if type(p[key]) is not int or not low <= p[key] <= 8192:
            raise ValueError(f"connected block profile has invalid {key}")
    # A finite construction screen, not an allocation, placement or traffic proof.
    if kind == "conv_pool":
        if p["pool_padding"] > p["pool_kernel"] // 2:
            raise ValueError("pool source padding exceeds half its window")
        conv = tuple((p[axis] + 2*p["conv_padding"] - p["conv_kernel"]) // p["conv_stride"] + 1
                     for axis in ("height", "width"))
        pooled = tuple((extent + 2*p["pool_padding"] - p["pool_kernel"]) // p["pool_stride"] + 1
                       for extent in conv)
        if min(*conv, *pooled) <= 0:
            raise ValueError("conv/pool source geometry has no output elements")
        elements = (p["channels"]*p["height"]*p["width"]
                    + p["channels"]*p["out_channels"]*p["conv_kernel"]**2
                    + p["out_channels"]*(math.prod(conv) + math.prod(pooled)))
    elif kind == "residual":
        if p["stride"] not in (1, 2) or p["blocks"] > 8:
            raise ValueError("residual source needs stride 1/2 and at most eight blocks")
        elements = (p["channels"] + p["out_channels"] + p["width"]) * p["side"]**2
        elements += p["blocks"] * (p["channels"] * p["width"] + 9 * p["width"]**2
                                   + p["width"] * p["out_channels"] + p["out_channels"]**2)
    elif kind == "attention":
        if (p["heads"] > 32 or p["hidden"] % p["heads"] or p["heads"] % p["kv_heads"]
                or (p["hidden"] // p["heads"]) % 2):
            raise ValueError("attention needs grouped heads and an even rotary head width")
        elements = (p["heads"] * p["tokens"] * (p["prefix"] + p["tokens"])
                    + 4 * p["hidden"]**2 + 3 * p["hidden"] * p["ffn"]
                    + 2 * p["hidden"] * (p["prefix"] + p["tokens"]))
    else:
        if p["steps"] > 8:
            raise ValueError("repeated source needs at most eight complete calls")
        elements = (p["hidden"] * p["ffn"] + p["ffn"] * p["output"]
                    + p["steps"] * p["rows"] * (p["hidden"] + p["ffn"] + p["output"]))
    if elements > MAX_ELEMENTS:
        raise ValueError("connected source exceeds its finite logical element budget")
    return p


def values(shape, offset=0):
    count = math.prod(shape)
    return (((torch.arange(count, dtype=torch.int64) + offset) % 31 - 15).float() / 64).reshape(shape)


class ConvPool(nn.Module):
    """One connected original graph; the selected capture recipe owns quantization."""
    def __init__(self, p):
        super().__init__()
        self.convolution = nn.Conv2d(p["channels"], p["out_channels"], p["conv_kernel"],
                                     stride=p["conv_stride"], padding=p["conv_padding"], bias=False)
        self.pool = nn.MaxPool2d(p["pool_kernel"], stride=p["pool_stride"], padding=p["pool_padding"])

    def forward(self, image):
        return (self.pool(torch.relu(self.convolution(image))),)


class Bottleneck(nn.Module):
    def __init__(self, channels, width, output, stride):
        super().__init__()
        self.reduce = nn.Conv2d(channels, width, 1, bias=False)
        self.spatial = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.expand = nn.Conv2d(width, output, 1, bias=False)
        self.norm1, self.norm2, self.norm3 = (nn.BatchNorm2d(c) for c in (width, width, output))
        self.shortcut = (nn.Identity() if channels == output and stride == 1 else nn.Sequential(
            nn.Conv2d(channels, output, 1, stride=stride, bias=False), nn.BatchNorm2d(output)))

    def forward(self, x):
        branch = torch.relu(self.norm1(self.reduce(x)))
        branch = torch.relu(self.norm2(self.spatial(branch)))
        return torch.relu(self.norm3(self.expand(branch)) + self.shortcut(x))


class ResidualStages(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.blocks = nn.Sequential(Bottleneck(p["channels"], p["width"], p["out_channels"], p["stride"]),
                                    *(Bottleneck(p["out_channels"], p["width"], p["out_channels"], 1)
                                      for _ in range(p["blocks"] - 1)))

    def forward(self, image):
        return (self.blocks(image),)


class AttentionBlock(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.heads, self.kv_heads = p["heads"], p["kv_heads"]
        self.width, self.tokens = p["hidden"] // p["heads"], p["tokens"]
        kv_width = self.width * self.kv_heads
        self.norm1, self.norm2 = nn.RMSNorm(p["hidden"]), nn.RMSNorm(p["hidden"])
        self.query = nn.Linear(p["hidden"], p["hidden"], bias=False)
        self.key = nn.Linear(p["hidden"], kv_width, bias=False)
        self.value = nn.Linear(p["hidden"], kv_width, bias=False)
        self.projection = nn.Linear(p["hidden"], p["hidden"], bias=False)
        self.gate, self.up = (nn.Linear(p["hidden"], p["ffn"], bias=False) for _ in range(2))
        self.down = nn.Linear(p["ffn"], p["hidden"], bias=False)
        positions = torch.arange(p["prefix"], p["prefix"] + self.tokens, dtype=torch.float32)
        frequency = 10000.0 ** (-torch.arange(0, self.width, 2, dtype=torch.float32) / self.width)
        angle = torch.outer(positions, frequency).repeat(1, 2)
        self.register_buffer("cosine", angle.cos())
        self.register_buffer("sine", angle.sin())
        allowed = torch.arange(p["prefix"] + self.tokens)[None, :] <= positions[:, None]
        self.register_buffer("mask", torch.where(allowed, 0.0, float("-inf")))

    def rotate(self, x):
        half = self.width // 2
        return x * self.cosine + torch.cat((-x[..., half:], x[..., :half]), dim=-1) * self.sine

    def share(self, x):
        batch, heads, length, width = x.shape
        return x[:, :, None].expand(batch, heads, self.heads // heads, length, width).reshape(
            batch, self.heads, length, width)

    def forward(self, hidden, prior_keys, prior_values):
        normalized = self.norm1(hidden)
        q = self.query(normalized).reshape(1, self.tokens, self.heads, self.width).transpose(1, 2)
        k = self.key(normalized).reshape(1, self.tokens, self.kv_heads, self.width).transpose(1, 2)
        v = self.value(normalized).reshape(1, self.tokens, self.kv_heads, self.width).transpose(1, 2)
        keys = torch.cat((prior_keys, self.rotate(k)), dim=2)
        vals = torch.cat((prior_values, v), dim=2)
        scores = (self.rotate(q) @ self.share(keys).transpose(-2, -1)) / math.sqrt(self.width)
        probabilities = torch.softmax(scores + self.mask, dim=-1)
        context = (probabilities @ self.share(vals)).transpose(1, 2).reshape(hidden.shape)
        hidden = hidden + self.projection(context)
        normalized = self.norm2(hidden)
        output = hidden + self.down(torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized))
        return output, keys, vals


class RepeatedInference(nn.Module):
    def forward(self, up_weight, down_weight, *activations):
        return tuple(torch.relu(x @ up_weight) @ down_weight for x in activations)


def build(profile):
    """Construct deterministic original semantics; no quantizer or device placement is implied."""
    p = validate_profile(profile)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        if p["kind"] == "conv_pool":
            model = ConvPool(p)
            inputs = (values((1, p["channels"], p["height"], p["width"])),)
        elif p["kind"] == "residual":
            model = ResidualStages(p)
            inputs = (values((1, p["channels"], p["side"], p["side"])),)
        elif p["kind"] == "attention":
            model = AttentionBlock(p)
            cache = (1, p["kv_heads"], p["prefix"], p["hidden"] // p["heads"])
            inputs = (values((1, p["tokens"], p["hidden"])), values(cache, 3), values(cache, 9))
        else:
            model = RepeatedInference()
            inputs = (values((p["hidden"], p["ffn"]), 5), values((p["ffn"], p["output"]), 11),
                      *(values((p["rows"], p["hidden"]), 3 * step) for step in range(p["steps"])))
        with torch.no_grad():
            for ordinal, module in enumerate(model.modules()):
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    module.weight.copy_(values(module.weight.shape, ordinal) / math.sqrt(module.weight.shape[1]))
    return model.eval(), inputs


def get_model_and_inputs():
    path = Path(__file__).with_name("profile.json")
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("connected profile must be an adjacent ordinary file")
    raw = path.read_bytes() if path.exists() else None
    profile = (json.loads(raw) if raw is not None else
               dict(schema=PROFILE_SCHEMA, workload_id=WORKLOAD_ID, kind="residual"))
    model, inputs = build(profile)
    model.session_provenance = dict(workload_id=WORKLOAD_ID, workload_role="iteration",
                                    source_role="handauthored_compiler_test", input_source="seeded_synthetic",
                                    family=profile["kind"],
                                    **({"profile_sha256": hashlib.sha256(raw).hexdigest()}
                                       if raw is not None else {"default_profile": True}))
    return model, inputs
