"""Quantized workloads: int8 and fp8 matmul, weight-quantized linear/MLP."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from workloads.base import Workload


class _Int8Linear(nn.Module):
    def __init__(self, inf, outf):
        super().__init__()
        self.weight_i8 = nn.Parameter(
            torch.randint(-128, 127, (outf, inf), dtype=torch.int8),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.randn(outf).abs() * 0.01, requires_grad=False
        )

    def forward(self, x):
        w = self.weight_i8.to(x.dtype) * self.scale.unsqueeze(1)
        return x @ w.T


@dataclass
class Int8Matmul(Workload):
    name: str = "int8_matmul"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "N": 128, "K": 128, "desc": "128"},
            {"M": 512, "N": 512, "K": 512, "desc": "512"},
            {"M": 1024, "N": 1024, "K": 1024, "desc": "1024"},
            {"M": 2048, "N": 2048, "K": 2048, "desc": "2048"},
            {"M": 4096, "N": 4096, "K": 4096, "desc": "4096"},
        ]
    )

    def torch_module(self, size):
        M, N, K = size["M"], size["N"], size["K"]

        class I8MM(nn.Module):
            def forward(self, a, b):
                return torch.matmul(a.to(torch.int32), b.to(torch.int32))

        return I8MM(), [
            torch.randint(-128, 127, (M, K), dtype=torch.int8),
            torch.randint(-128, 127, (K, N), dtype=torch.int8),
        ]

    def input_specs(self, size):
        M, N, K = size["M"], size["N"], size["K"]
        return [f"{M}x{K}xi8", f"{K}x{N}xi8"]

    def mlir_source(self, size):
        M, N, K = size["M"], size["N"], size["K"]
        return f"""\
func.func @main(%A: tensor<{M}x{K}xi8>, %B: tensor<{K}x{N}xi8>) -> tensor<{M}x{N}xi32> {{
  %c0 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<{M}x{N}xi32>
  %fill = linalg.fill ins(%c0 : i32) outs(%empty : tensor<{M}x{N}xi32>) -> tensor<{M}x{N}xi32>
  %result = linalg.matmul ins(%A, %B : tensor<{M}x{K}xi8>, tensor<{K}x{N}xi8>)
                          outs(%fill : tensor<{M}x{N}xi32>) -> tensor<{M}x{N}xi32>
  return %result : tensor<{M}x{N}xi32>
}}
"""


@dataclass
class Int8WeightLinear(Workload):
    """Linear layer with int8 weights, fp32 activations (dequant on the fly)."""

    name: str = "int8_weight_linear"
    level: int = 2
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "N": 256, "K": 512, "desc": "128x256x512"},
            {"M": 512, "N": 1024, "K": 1024, "desc": "512x1024x1024"},
            {"M": 1024, "N": 2048, "K": 2048, "desc": "1024x2048x2048"},
        ]
    )

    def torch_module(self, size):
        M, K, N = size["M"], size["K"], size["N"]
        return _Int8Linear(K, N), [torch.randn(M, K)]

    def input_specs(self, size):
        M, K = size["M"], size["K"]
        return [f"{M}x{K}xf32"]


@dataclass
class Int8MLP(Workload):
    """2-layer MLP with int8-quantized weights."""

    name: str = "int8_mlp"
    level: int = 3
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"B": 32, "I": 512, "H": 1024, "O": 256, "desc": "32x512x1024x256"},
            {"B": 128, "I": 1024, "H": 2048, "O": 512, "desc": "128x1024x2048x512"},
        ]
    )

    def torch_module(self, size):
        B, I, H, O = size["B"], size["I"], size["H"], size["O"]

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = _Int8Linear(I, H)
                self.fc2 = _Int8Linear(H, O)

            def forward(self, x):
                return self.fc2(torch.relu(self.fc1(x)))

        return MLP(), [torch.randn(B, I)]

    def input_specs(self, size):
        B, I = size["B"], size["I"]
        return [f"{B}x{I}xf32"]


ALL_QUANTIZED: list[Workload] = [
    Int8Matmul(),
    Int8WeightLinear(),
    Int8MLP(),
]
