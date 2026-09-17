"""DERIVED micro model -- regenerate with merlin.targetgen.micro_model.emit_pytorch.

Composition: A->A
Layer inventory, in composition order:
#   0. accelerator attention        -> attention_full
#   1. accelerator attention        -> attention_full
#   2. accelerator attention        -> attention_full
#   3. accelerator attention        -> attention_full
#   4. accelerator attention        -> attention_full
#   5. accelerator attention        -> attention_full
#   6. accelerator contraction      -> matmul
#   7. accelerator contraction      -> matmul
#   8. accelerator contraction      -> matmul
#   9. accelerator contraction      -> matmul
#   10. accelerator contraction      -> matmul
#   11. accelerator contraction      -> matmul
#   12. accelerator contraction      -> matmul
#   13. accelerator elementwise_map  -> gelu
#   14. accelerator elementwise_map  -> gelu
#   15. accelerator elementwise_map  -> gelu
#   16. accelerator movement         -> movement
#   17. accelerator movement         -> movement
#   18. accelerator movement         -> movement
#   19. accelerator normalization    -> rmsnorm
#   20. accelerator normalization    -> rmsnorm
#   21. accelerator normalization    -> rmsnorm
#   22. accelerator reduction        -> reduce_sum
#   23. accelerator reduction        -> reduce_sum
#   24. accelerator reduction        -> reduce_sum
#   25. accelerator softmax          -> softmax
#   26. accelerator softmax          -> softmax
#   27. accelerator softmax          -> softmax

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
        self.qkv0 = nn.Parameter(torch.randn(3, E, E) * 0.05)
        self.qkv1 = nn.Parameter(torch.randn(3, E, E) * 0.05)
        self.qkv2 = nn.Parameter(torch.randn(3, E, E) * 0.05)
        self.qkv3 = nn.Parameter(torch.randn(3, E, E) * 0.05)
        self.qkv4 = nn.Parameter(torch.randn(3, E, E) * 0.05)
        self.qkv5 = nn.Parameter(torch.randn(3, E, E) * 0.05)
        self.w6 = nn.Parameter(torch.randn(E, E) * 0.05)
        self.w7 = nn.Parameter(torch.randn(E, E) * 0.05)
        self.w8 = nn.Parameter(torch.randn(E, E) * 0.05)
        self.w9 = nn.Parameter(torch.randn(E, E) * 0.05)
        self.w10 = nn.Parameter(torch.randn(E, E) * 0.05)
        self.w11 = nn.Parameter(torch.randn(E, E) * 0.05)
        self.w12 = nn.Parameter(torch.randn(E, E) * 0.05)

    def forward(self, x):
        # accelerator: attention (observed spelling None)
        x = x + torch.nn.functional.scaled_dot_product_attention(x @ self.qkv0[0], x @ self.qkv0[1], x @ self.qkv0[2])
        # accelerator: attention (observed spelling None)
        x = x + torch.nn.functional.scaled_dot_product_attention(x @ self.qkv1[0], x @ self.qkv1[1], x @ self.qkv1[2])
        # accelerator: attention (observed spelling None)
        x = x + torch.nn.functional.scaled_dot_product_attention(x @ self.qkv2[0], x @ self.qkv2[1], x @ self.qkv2[2])
        # accelerator: attention (observed spelling None)
        x = x + torch.nn.functional.scaled_dot_product_attention(x @ self.qkv3[0], x @ self.qkv3[1], x @ self.qkv3[2])
        # accelerator: attention (observed spelling None)
        x = x + torch.nn.functional.scaled_dot_product_attention(x @ self.qkv4[0], x @ self.qkv4[1], x @ self.qkv4[2])
        # accelerator: attention (observed spelling None)
        x = x + torch.nn.functional.scaled_dot_product_attention(x @ self.qkv5[0], x @ self.qkv5[1], x @ self.qkv5[2])
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w6
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w7
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w8
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w9
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w10
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w11
        # accelerator: contraction (observed spelling 'matmul')
        x = x @ self.w12
        # accelerator: elementwise_map (observed spelling 'generic')
        x = torch.nn.functional.gelu(x)
        # accelerator: elementwise_map (observed spelling 'generic')
        x = torch.nn.functional.gelu(x)
        # accelerator: elementwise_map (observed spelling 'generic')
        x = torch.nn.functional.gelu(x)
        # accelerator: movement (observed spelling 'transpose')
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        # accelerator: movement (observed spelling 'transpose')
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        # accelerator: movement (observed spelling 'transpose')
        x = x.transpose(-1, -2).contiguous().transpose(-1, -2)
        # accelerator: normalization (observed spelling 'generic')
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        # accelerator: normalization (observed spelling 'generic')
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        # accelerator: normalization (observed spelling 'generic')
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        # accelerator: reduction (observed spelling 'reduce')
        x = x - x.sum(-1, keepdim=True) / E
        # accelerator: reduction (observed spelling 'reduce')
        x = x - x.sum(-1, keepdim=True) / E
        # accelerator: reduction (observed spelling 'reduce')
        x = x - x.sum(-1, keepdim=True) / E
        # accelerator: softmax (observed spelling None)
        x = torch.softmax(x, dim=-1)
        # accelerator: softmax (observed spelling None)
        x = torch.softmax(x, dim=-1)
        # accelerator: softmax (observed spelling None)
        x = torch.softmax(x, dim=-1)
        return x


def get_model_and_inputs():
    torch.manual_seed(0)
    return MicroModel().eval(), (torch.randn(E, E),)
