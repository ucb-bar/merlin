"""One 2-D convolution: lowered as im2col + a matmul, so it reaches a matrix unit like a GEMM does.

    CONV_CIN=16 CONV_COUT=64 CONV_HW=16 CONV_K=3     (defaults)

The matmul it becomes is M = output pixels ((HW-K+1)^2 = 196), N = COUT, K = CIN*K*K. Weights and
inputs are deterministic. Use with a chip example's `op` stage, e.g. `examples/kodiak/run.sh op conv2d`.
"""

import os

import torch


def _dim(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name}={value} must be positive")
    return value


def get_model_and_inputs():
    cin, cout, hw, k = _dim("CONV_CIN", 16), _dim("CONV_COUT", 64), _dim("CONV_HW", 16), _dim("CONV_K", 3)
    torch.manual_seed(0)
    model = torch.nn.Conv2d(cin, cout, k, bias=False).eval()
    inputs = (torch.randn(1, cin, hw, hw),)
    return model, inputs
