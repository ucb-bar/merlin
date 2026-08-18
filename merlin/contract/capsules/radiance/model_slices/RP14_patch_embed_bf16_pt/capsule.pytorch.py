"""Auto-generated capsule loader (patch_embed, bf16). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = 28324
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0

class Model(nn.Module):
    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w, stride=2)
def get_model_and_inputs():
    return Model(), (_r(1, 3, 8, 8), _r(16, 3, 2, 2))
