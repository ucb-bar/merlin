"""A GEMM whose right operand is TRANSPOSED: C[M,N] = A[M,K] . B[N,K]^T, both operands activations.

    GEMMNT_M=64 GEMMNT_N=96 GEMMNT_K=80      (defaults)

Written as ``torch.einsum("mk,nk->mn")`` so the contraction reads B along its rows: the layout a
transpose-b GEMM, a similarity matrix or an attention score has. Both operands are computed (a linear
of the input each), so neither is a weight a build could pre-transpose. Weights and inputs are
deterministic. Use with a chip example's `op` stage, e.g. `examples/kodiak/run.sh op gemm_nt 64 96 80`.
"""

import os

import torch


def _dim(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name}={value} must be positive")
    return value


class GemmNT(torch.nn.Module):
    def __init__(self, m: int, n: int, k: int) -> None:
        super().__init__()
        self.m, self.n = m, n
        self.a = torch.nn.Linear(k, k, bias=False)
        self.b = torch.nn.Linear(k, k, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a(x[: self.m])
        b = self.b(x[self.m : self.m + self.n])
        return torch.einsum("mk,nk->mn", a, b)


def get_model_and_inputs():
    m, n, k = _dim("GEMMNT_M", 64), _dim("GEMMNT_N", 96), _dim("GEMMNT_K", 80)
    torch.manual_seed(0)
    model = GemmNT(m, n, k).eval()
    inputs = (torch.randn(m + n, k),)
    return model, inputs
