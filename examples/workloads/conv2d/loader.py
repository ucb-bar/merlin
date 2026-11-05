"""One 2-D convolution: lowered as im2col + a matmul, so it reaches a matrix unit like a GEMM does.

    CONV_CIN=16 CONV_COUT=64 CONV_HW=16 CONV_K=3     (defaults)
    CONV_STRIDE=1 CONV_DILATION=1                    (optional; default 1)

The matmul it becomes is M = COUT, N = output pixels (((HW - DILATION*(K-1) - 1) // STRIDE + 1)^2, 196 at
the defaults), K = CIN*K*K. A stride or dilation changes only the window the im2col gather reads, so a
strided conv is the cheapest way to put a small pixel count (a small N) in front of the routing. Weights
and inputs are deterministic. Use with a chip example's `op` stage, e.g. `examples/kodiak/run.sh op
conv2d`, or `CONV_STRIDE=2 examples/kodiak/run.sh op conv2d 32 64 15 3`.
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
    stride, dilation = _dim("CONV_STRIDE", 1), _dim("CONV_DILATION", 1)
    torch.manual_seed(0)
    model = torch.nn.Conv2d(cin, cout, k, stride=stride, dilation=dilation, bias=False).eval()
    inputs = (torch.randn(1, cin, hw, hw),)
    return model, inputs
