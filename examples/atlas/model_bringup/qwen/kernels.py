"""Pure PyTorch subgraphs of DeepSeek-R1-Distill-Qwen-1.5B.

Full-size modules preserve Qwen2 arithmetic. AttentionTile is a head-local
32-token window with all 128 head channels, decomposed into 32-channel slices.
It accepts an additive mask (including a causal mask) in the full-size path;
the compiler experiment initially exercises an unmasked decode/cache window.
No Atlas or Merlin imports are used here.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RADIANCE_PR = "https://github.com/ucb-bar/radiance-kernels/pull/1"
RADIANCE_REVISION = "b60401712885ad0bbbc5d8946726f39f8d43ecb6"


@dataclass(frozen=True)
class Config:
    hidden: int = 1536
    intermediate: int = 8960
    query_heads: int = 12
    kv_heads: int = 2
    head_dim: int = 128
    layers: int = 28
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0


CONFIG = Config()


class RMSNormQKV(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.eps = config.rms_eps
        self.norm = nn.Parameter(torch.ones(config.hidden))
        self.q = nn.Linear(
            config.hidden, config.query_heads * config.head_dim, bias=True
        )
        self.k = nn.Linear(config.hidden, config.kv_heads * config.head_dim, bias=True)
        self.v = nn.Linear(config.hidden, config.kv_heads * config.head_dim, bias=True)

    def forward(self, x):
        h = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.norm
        return self.q(h), self.k(h), self.v(h)


class SwiGLUResidual(nn.Module):
    def __init__(self, config=CONFIG):
        super().__init__()
        self.eps = config.rms_eps
        self.norm = nn.Parameter(torch.ones(config.hidden))
        self.gate = nn.Linear(config.hidden, config.intermediate, bias=False)
        self.up = nn.Linear(config.hidden, config.intermediate, bias=False)
        self.down = nn.Linear(config.intermediate, config.hidden, bias=False)

    def forward(self, x):
        h = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + self.eps) * self.norm
        return x + self.down(F.silu(self.gate(h)) * self.up(h))


class RotaryEmbedding(nn.Module):
    def forward(self, q, k, cos, sin):
        def rotate(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin


class GroupedQueryAttention(nn.Module):
    """Inputs [batch, head, token, channel]; caller supplies the causal/cache mask."""

    def __init__(self, config=CONFIG):
        super().__init__()
        self.groups = config.query_heads // config.kv_heads
        self.scale = config.head_dim**-0.5

    def forward(self, q, k, v, mask):
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        scores = q @ k.transpose(-1, -2) * self.scale + mask
        return scores.softmax(dim=-1) @ v


class AttentionTile(nn.Module):
    """One 32x32 score window, d=128, four output channel tiles.

    Q/K arrive after RoPE. A complete long-context attention requires an
    online-softmax merge across windows; this module does not claim that merge.
    Each v slice is transposed on input to use linear's [out, in] convention.
    """

    def forward(self, q0, q1, q2, q3, k0, k1, k2, k3, v0, v1, v2, v3):
        scores = (
            F.linear(q0, k0) + F.linear(q1, k1) + F.linear(q2, k2) + F.linear(q3, k3)
        ) * (128**-0.5)
        probability = torch.softmax(scores, dim=-1)
        return (
            F.linear(probability, v0),
            F.linear(probability, v1),
            F.linear(probability, v2),
            F.linear(probability, v3),
        )


class Projection(nn.Module):
    """One full reduction-width projection, used with Merlin's ordered K tiling."""

    def forward(self, x, weight):
        return F.linear(x, weight)


def synthetic_tensor(shape, *, generator=None, signed=False):
    # Keep magnitudes normal in FP8 and nonlinear inputs small. The selected
    # input policy rejects negative zero after flushing tiny negative values.
    values = torch.rand(shape, generator=generator) * 0.125 + 0.03125
    if signed:
        values *= torch.where(torch.rand(shape, generator=generator) < 0.5, -1, 1)
    return values


def attention_inputs(seed=0, *, signed=False):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        synthetic_tensor((32, 32), generator=generator, signed=signed)
        for _ in range(12)
    )


def get_model_and_inputs():
    return AttentionTile().eval(), attention_inputs()


class SwiGLUTile(nn.Module):
    """A 32-channel FFN partition: gate/up, SiLU product, down projection.

    Full-model FFN evaluation needs K-accumulation before SiLU and sums of down
    projection partitions. This is the fused interior scheduling experiment.
    """

    def forward(self, x, gate_weight, up_weight, down_weight):
        gate = F.linear(x, gate_weight)
        up = F.linear(x, up_weight)
        return F.linear((gate * torch.sigmoid(gate)) * up, down_weight)
