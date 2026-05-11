"""Level 1 primitive workloads. All defined as PyTorch modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from workloads.base import Workload


@dataclass
class ElementwiseAdd(Workload):
    name: str = "elementwise_add"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 32, "N": 32, "desc": "1K"},
            {"M": 256, "N": 256, "desc": "64K"},
            {"M": 1000, "N": 1000, "desc": "1M"},
            {"M": 4096, "N": 4096, "desc": "16M"},
        ]
    )

    def torch_module(self, size):
        M, N = size["M"], size["N"]

        class Add(nn.Module):
            def forward(self, a, b):
                return a + b

        return Add(), [torch.randn(M, N), torch.randn(M, N)]

    def input_specs(self, size):
        M, N = size["M"], size["N"]
        return [f"{M}x{N}xf32", f"{M}x{N}xf32"]


@dataclass
class ElementwiseGelu(Workload):
    name: str = "elementwise_gelu"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"N": 1024, "desc": "1K"},
            {"N": 16384, "desc": "16K"},
            {"N": 1048576, "desc": "1M"},
        ]
    )

    def torch_module(self, size):
        N = size["N"]

        class Gelu(nn.Module):
            def forward(self, x):
                return torch.nn.functional.gelu(x)

        return Gelu(), [torch.randn(N)]

    def input_specs(self, size):
        return [f"{size['N']}xf32"]


@dataclass
class ReduceSum(Workload):
    name: str = "reduce_sum"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"N": 1024, "desc": "1K"},
            {"N": 16384, "desc": "16K"},
            {"N": 1048576, "desc": "1M"},
        ]
    )

    def torch_module(self, size):
        N = size["N"]

        class Sum(nn.Module):
            def forward(self, x):
                return x.sum().unsqueeze(0)

        return Sum(), [torch.randn(N)]

    def input_specs(self, size):
        return [f"{size['N']}xf32"]


@dataclass
class Softmax(Workload):
    name: str = "softmax"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 64, "N": 64, "desc": "64x64"},
            {"M": 256, "N": 256, "desc": "256x256"},
            {"M": 1024, "N": 1024, "desc": "1Kx1K"},
        ]
    )

    def torch_module(self, size):
        M, N = size["M"], size["N"]

        class Sfmx(nn.Module):
            def forward(self, x):
                return torch.nn.functional.softmax(x, dim=-1)

        return Sfmx(), [torch.randn(M, N)]

    def input_specs(self, size):
        return [f"{size['M']}x{size['N']}xf32"]


@dataclass
class MatmulF32(Workload):
    name: str = "matmul_f32"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "N": 128, "K": 128, "desc": "128"},
            {"M": 256, "N": 256, "K": 256, "desc": "256"},
            {"M": 512, "N": 512, "K": 512, "desc": "512"},
            {"M": 1024, "N": 1024, "K": 1024, "desc": "1024"},
            {"M": 2048, "N": 2048, "K": 2048, "desc": "2048"},
            {"M": 4096, "N": 4096, "K": 4096, "desc": "4096"},
        ]
    )

    def torch_module(self, size):
        M, N, K = size["M"], size["N"], size["K"]

        class MM(nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b)

        return MM(), [torch.randn(M, K), torch.randn(K, N)]

    def input_specs(self, size):
        M, N, K = size["M"], size["N"], size["K"]
        return [f"{M}x{K}xf32", f"{K}x{N}xf32"]


@dataclass
class MatmulF16(Workload):
    name: str = "matmul_f16"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "N": 128, "K": 128, "desc": "128"},
            {"M": 512, "N": 512, "K": 512, "desc": "512"},
            {"M": 2048, "N": 2048, "K": 2048, "desc": "2048"},
            {"M": 4096, "N": 4096, "K": 4096, "desc": "4096"},
        ]
    )

    def torch_module(self, size):
        M, N, K = size["M"], size["N"], size["K"]

        class MM(nn.Module):
            def forward(self, a, b):
                return torch.matmul(a, b)

        return MM(), [
            torch.randn(M, K, dtype=torch.float16),
            torch.randn(K, N, dtype=torch.float16),
        ]

    def input_specs(self, size):
        M, N, K = size["M"], size["N"], size["K"]
        return [f"{M}x{K}xf16", f"{K}x{N}xf16"]


@dataclass
class Conv2d(Workload):
    name: str = "conv2d"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"B": 1, "C": 64, "H": 56, "W": 56, "F": 64, "KH": 3, "KW": 3, "desc": "56x56_k3"},
            {"B": 1, "C": 128, "H": 28, "W": 28, "F": 128, "KH": 3, "KW": 3, "desc": "28x28_k3"},
            {"B": 1, "C": 256, "H": 14, "W": 14, "F": 256, "KH": 3, "KW": 3, "desc": "14x14_k3"},
        ]
    )

    def torch_module(self, size):
        B, C, H, W = size["B"], size["C"], size["H"], size["W"]
        F, KH, KW = size["F"], size["KH"], size["KW"]

        conv = nn.Conv2d(C, F, (KH, KW), bias=False)
        return conv, [torch.randn(B, C, H, W)]

    def input_specs(self, size):
        B, C, H, W = size["B"], size["C"], size["H"], size["W"]
        return [f"{B}x{C}x{H}x{W}xf32"]


@dataclass
class Transpose(Workload):
    name: str = "transpose"
    level: int = 1
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 1024, "N": 1024, "desc": "1Kx1K"},
            {"M": 4096, "N": 4096, "desc": "4Kx4K"},
        ]
    )

    def torch_module(self, size):
        M, N = size["M"], size["N"]

        class T(nn.Module):
            def forward(self, x):
                return x.T.contiguous()

        return T(), [torch.randn(M, N)]

    def input_specs(self, size):
        return [f"{size['M']}x{size['N']}xf32"]


ALL_PRIMITIVES: list[Workload] = [
    ElementwiseAdd(),
    ElementwiseGelu(),
    ReduceSum(),
    Softmax(),
    MatmulF32(),
    MatmulF16(),
    Conv2d(),
    Transpose(),
]
