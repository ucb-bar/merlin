"""Auto-generated capsule loader (gemma_4norm, bf16). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = 28204
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0

def _rms(x, g, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * g
class Model(nn.Module):
    def forward(self, x, g1, g2):
        return _rms(_rms(x, g1, 1e-05), g2, 1e-05)
def get_model_and_inputs():
    return Model(), (_r(16, 16), _r(1, 16) * 0.5 + 1.0, _r(1, 16) * 0.5 + 1.0)
