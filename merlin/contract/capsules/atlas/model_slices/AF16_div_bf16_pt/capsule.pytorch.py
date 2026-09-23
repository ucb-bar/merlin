"""Auto-generated capsule loader (div, bf16). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = 12435
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0

class Model(nn.Module):
    def forward(self, a, b):
        return a / b
def get_model_and_inputs():
    a = _r(32, 32)
    b = _r(32, 32)
    # keep the divisor away from zero: a near-zero divisor makes the quotient unbounded, so the
    # golden would be dominated by a handful of huge elements and an absolute tolerance would say
    # nothing about the rest of the tile.
    b = torch.where(b.abs() < 0.25, torch.full_like(b, 0.5), b)
    return Model(), (a, b)
