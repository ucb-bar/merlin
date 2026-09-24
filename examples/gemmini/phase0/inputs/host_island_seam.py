"""Authored int8 contraction → floating LayerNorm → int8 contraction seam.

This is a target-specific Phase 0 input, not a generated capsule or core Merlin
policy. The two contractions already have int8 operands; applying a TorchAO PT2E
recipe again would change the program under test. The recipe declares
``capture_quantization: already_materialized`` and the capture verifies those
contractions survive lowering. The fixed seed makes the golden reproducible.
"""

from __future__ import annotations

import torch
from torch import nn

TILE = 16
M = TILE
C = 2 * TILE
AMP = 3


def _ternary(g, *shape):
    """Bound each weight to [-1, 1], so |sum| <= C * AMP = 96 < 127."""
    return torch.randint(0, 3, shape, generator=g, dtype=torch.int8) - 1


def _q(x, gain: float):
    return (torch.tanh(x * gain) * float(AMP)).to(torch.int8)


class HostIslandSeam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(20260830)
        self.register_buffer("w_in", _ternary(g, C, C))
        self.ln = nn.LayerNorm(C)
        self.register_buffer("w_out", _ternary(g, C, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = torch.matmul(x, self.w_in)
        host = self.ln(acc.to(torch.float32))
        return torch.matmul(_q(host, 1.1), self.w_out)


def get_model_and_inputs():
    g = torch.Generator().manual_seed(11)
    x = torch.randint(-AMP, AMP + 1, (M, C), generator=g, dtype=torch.int8)
    return HostIslandSeam().eval(), (x,)
