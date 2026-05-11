"""Level 2 composite workloads. All defined as PyTorch modules."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from workloads.base import Workload


@dataclass
class LinearBiasRelu(Workload):
    name: str = "linear_bias_relu"
    level: int = 2
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "N": 256, "K": 512, "desc": "128x256x512"},
            {"M": 512, "N": 1024, "K": 1024, "desc": "512x1024x1024"},
        ]
    )

    def torch_module(self, size):
        M, N, K = size["M"], size["N"], size["K"]

        class LBR(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(K, N)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.linear(x))

        return LBR(), [torch.randn(M, K)]

    def input_specs(self, size):
        M, K = size["M"], size["K"]
        return [f"{M}x{K}xf32"]


@dataclass
class MLP2Layer(Workload):
    name: str = "mlp_2layer"
    level: int = 2
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "H": 512, "O": 256, "I": 256, "desc": "128_256_512_256"},
            {"M": 512, "H": 2048, "O": 1024, "I": 1024, "desc": "512_1024_2048_1024"},
        ]
    )

    def torch_module(self, size):
        M, I, H, O = size["M"], size["I"], size["H"], size["O"]

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(I, H)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(H, O)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        return MLP(), [torch.randn(M, I)]

    def input_specs(self, size):
        M, I = size["M"], size["I"]
        return [f"{M}x{I}xf32"]


@dataclass
class LayerNorm(Workload):
    name: str = "layernorm"
    level: int = 2
    sizes: list[dict] = field(
        default_factory=lambda: [
            {"M": 128, "N": 512, "desc": "128x512"},
            {"M": 512, "N": 1024, "desc": "512x1024"},
        ]
    )

    def torch_module(self, size):
        M, N = size["M"], size["N"]

        class LN(nn.Module):
            def __init__(self):
                super().__init__()
                self.ln = nn.LayerNorm(N)

            def forward(self, x):
                return self.ln(x)

        return LN(), [torch.randn(M, N)]

    def input_specs(self, size):
        return [f"{size['M']}x{size['N']}xf32"]


ALL_COMPOSITES: list[Workload] = [
    LinearBiasRelu(),
    MLP2Layer(),
    LayerNorm(),
]
