"""L8_rmsnorm_f32 — torch source for the capsule (authored for model2MLIR).

row-wise RMS norm: y = x / sqrt(mean(x^2) + 1e-5). Lowering this through model2MLIR's FXImporter backend yields standard-dialect
linalg-on-tensors MLIR equivalent to capsule.interface.mlir (0 opaque ops; same signature).

    m2m convert capsule.torch.py --backend fx_importer --out lowered.mlir

model2MLIR canonicalises (named linalg ops, unfused elementwise) where capsule.interface.mlir uses
fused generics; the two are numerically identical. capsule.interface.mlir remains the authoritative
contract — this file is the torch provenance for it.
"""
from __future__ import annotations

import torch
from torch import nn


class Capsule(nn.Module):
    def forward(self, a):
        _ms = (a * a).mean(dim=1, keepdim=True)
        out = a / torch.sqrt(_ms + 1e-5)
        return out


def get_model_and_inputs():
    torch.manual_seed(0)
    a = torch.randn(16, 64, dtype=torch.float32)
    return Capsule().eval(), (a,)
