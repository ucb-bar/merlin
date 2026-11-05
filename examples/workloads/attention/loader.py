"""Multi-head self-attention: the two activation x activation contractions Q.K^T and P.V, plus the projections.

    ATTN_S=64 ATTN_D=128 ATTN_H=4      (defaults: sequence, model width, heads)

The scores ``Q.K^T`` contract over the head dim with K TRANSPOSED, and ``P.V`` contracts over the
sequence; both are batched over heads and both operands are activations, so neither is a weight GEMM.
They are written as rank-4 ``torch.matmul`` on (batch, head, seq, head_dim), the form most attention
code uses, so whatever layout the frontend gives them is what the routing has to handle. Weights and
inputs are deterministic. Use with a chip example's `op` stage, e.g. `examples/kodiak/run.sh op attention`.
"""

import math
import os

import torch


def _dim(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value < 1:
        raise ValueError(f"{name}={value} must be positive")
    return value


class Attention(torch.nn.Module):
    def __init__(self, d: int, h: int) -> None:
        super().__init__()
        if d % h:
            raise ValueError(f"ATTN_D={d} is not a multiple of ATTN_H={h}")
        self.h, self.dh = h, d // h
        self.qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.out = torch.nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (t.reshape(b, s, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        p = torch.softmax(scores, dim=-1)
        y = torch.matmul(p, v).transpose(1, 2).reshape(b, s, d)
        return self.out(y)


def get_model_and_inputs():
    s, d, h = _dim("ATTN_S", 64), _dim("ATTN_D", 128), _dim("ATTN_H", 4)
    torch.manual_seed(0)
    model = Attention(d, h).eval()
    inputs = (torch.randn(1, s, d),)
    return model, inputs
