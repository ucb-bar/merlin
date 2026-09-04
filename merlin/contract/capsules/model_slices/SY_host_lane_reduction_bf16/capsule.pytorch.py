"""Auto-generated capsule loader (reduce_sum, bf16). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = 36734
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0

class Model(nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)
def get_model_and_inputs():
    return Model(), (_r(16, 32),)
