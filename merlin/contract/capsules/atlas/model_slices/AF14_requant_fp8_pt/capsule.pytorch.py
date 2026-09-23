"""Auto-generated capsule loader (requant, bf16). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = 19029
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0

class Model(nn.Module):
    def forward(self, x):
        return x.to(torch.float8_e4m3fn).to(x.dtype)
def get_model_and_inputs():
    # A WIDE range on purpose. fp8_e4m3 keeps 3 mantissa bits, so its rounding error scales with the
    # binade: on [-1, 1) it is a few hundredths, which any absolute tolerance a float corpus declares
    # would swallow -- a capsule whose golden a plain COPY of the input also satisfies tests nothing.
    # At this scale the discarded mantissa bits are worth whole units and the round trip is graded.
    return Model(), (_r(32, 32) * 64.0,)
