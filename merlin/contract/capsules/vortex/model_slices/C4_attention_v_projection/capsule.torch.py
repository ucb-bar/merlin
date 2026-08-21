"""C4_attention_v_projection — torch source for the capsule (authored for model2MLIR).

linear (i8 x i8 -> i32). Lowering this through model2MLIR's FXImporter backend yields standard-dialect
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
    def forward(self, a, w):
        out = torch.matmul(a.to(torch.int32), w.to(torch.int32))
        return out


def get_model_and_inputs():
    torch.manual_seed(0)
    a = torch.randint(-8, 8, (16, 64,), dtype=torch.int8)
    w = torch.randint(-8, 8, (64, 16,), dtype=torch.int8)
    return Capsule().eval(), (a, w)
