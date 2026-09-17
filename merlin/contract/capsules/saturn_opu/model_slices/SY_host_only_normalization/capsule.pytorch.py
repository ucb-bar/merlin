"""Auto-generated capsule loader (layernorm, f32). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = 38052
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0

class Model(nn.Module):
    def forward(self, x, w, b):
        return torch.nn.functional.layer_norm(x, (32,), w, b, 1.52587890625e-05)
def get_model_and_inputs():
    return Model(), (_r(16, 32), _r(32) * 0.5 + 1.0, _r(32) * 0.1)
