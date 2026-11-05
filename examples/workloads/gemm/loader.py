"""One GEMM, C[M,N] = A[M,K] @ W[K,N]: the smallest model that exercises a matrix-unit kernel.

Shapes come from the environment so one loader covers every test point:

    GEMM_M=64 GEMM_N=64 GEMM_K=64      (defaults; one 64x64 OPU tile on Kodiak)
    GEMM_M=96 GEMM_N=80 GEMM_K=200     (partial tiles in M and N, a K that is not a multiple of 8)

Captured as a single `linalg.matmul` after int8 quantization. Weights and inputs are deterministic, so
the capture and its golden are reproducible. Use with a chip example's `op` stage, e.g.
`examples/kodiak/run.sh op gemm 64 64 64`, or capture it yourself with `--loader`.
"""

import os

import torch


def _dim(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name}={value} must be positive")
    return value


def get_model_and_inputs():
    m, n, k = _dim("GEMM_M", 64), _dim("GEMM_N", 64), _dim("GEMM_K", 64)
    torch.manual_seed(0)
    model = torch.nn.Linear(k, n, bias=False).eval()
    inputs = (torch.randn(m, k),)
    return model, inputs
