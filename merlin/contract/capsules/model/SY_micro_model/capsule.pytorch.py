"""DERIVED micro model -- regenerate with merlin.targetgen.micro_model.emit_pytorch.

Composition: A->H->A
Layer inventory, in composition order:
#   0. accelerator contraction      -> matmul
#   1. host        normalization    -> rmsnorm
#   2. accelerator elementwise_map  -> gelu
#   3. accelerator movement         -> movement
#   4. accelerator reduction        -> reduce_sum

Every layer is here because the target's capability manifest admits its family (accelerator) or
because a real capture contains a family the manifest does not admit (host). The order is the
interleave that puts host layers in the INTERIOR, so the model exercises a round trip rather
than a prefix.
"""
import torch
import torch.nn as nn

E = 32


class MicroModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = nn.Parameter(torch.randn(E, E) * 0.05)

    def forward(self, x):
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w0
        # host: normalization (observed spelling 'generic')
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        # accelerator: elementwise_map (observed spelling 'generic')
        x = torch.nn.functional.gelu(x)
        # accelerator: movement (observed spelling 'transpose')
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        # accelerator: reduction (observed spelling 'reduce')
        x = x - x.sum(-1, keepdim=True) / E
        return x


def get_model_and_inputs():
    torch.manual_seed(0)
    return MicroModel().eval(), (torch.randn(E, E),)
