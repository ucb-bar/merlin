"""C8_mlp_ffn_f32 — torch source for the capsule (authored for model2MLIR).

FFN: gelu(X@W1)@W2. Lowering this through model2MLIR's FXImporter backend yields standard-dialect
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
    def forward(self, a, w, w2):
        _h = torch.matmul(a, w)
        _x3 = _h * _h * _h
        _g = 0.5 * _h * (1.0 + torch.tanh(0.797884583 * (_h + 0.044715 * _x3)))
        out = torch.matmul(_g, w2)
        return out


def get_model_and_inputs():
    torch.manual_seed(0)
    a = torch.randn(16, 16, dtype=torch.float32)
    w = torch.randn(16, 32, dtype=torch.float32)
    w2 = torch.randn(32, 16, dtype=torch.float32)
    return Capsule().eval(), (a, w, w2)
