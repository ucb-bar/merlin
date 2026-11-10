"""Deterministic public operation loaders and their ordered input vocabulary."""

from __future__ import annotations

from merlin.targetgen import component_sources

_PREAMBLE = '''"""Auto-generated capsule loader ({op}, {dtype}). Defines the op in PyTorch; model2MLIR
lowers it to linalg and the host torch-eager result is the reference golden. Deterministic inputs."""
import math
import torch
from torch import nn

SEED = {seed}
torch.manual_seed(SEED)
_G = torch.Generator().manual_seed(SEED)


def _r(*shape):
    # distinct, asymmetric, order-sensitive values in [-1, 1) (a wrong row stride / transpose changes output)
    return (torch.rand(*shape, generator=_G) - 0.5) * 2.0
'''


_OP_BODIES: dict[str, str] = {
    "transpose": """
class Model(nn.Module):
    def forward(self, x):
        return x.transpose(-2, -1)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "matmul": """
class Model(nn.Module):
    def forward(self, a, w):
        return a @ w
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}))
""",
    "linear": """
class Model(nn.Module):
    def forward(self, a, w):
        return a @ w
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}))
""",
    "attention_qk": """
class Model(nn.Module):
    def forward(self, q, k):
        return q @ k.transpose(-2, -1)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({N}, {K}))
""",
    "attention_full": """
class Model(nn.Module):
    def forward(self, q, k, v):
        s = (q @ k.transpose(-2, -1)) / math.sqrt({K})
        if {causal}:
            t = q.shape[-2]
            neg = torch.finfo(s.dtype).min
            mask = torch.triu(torch.full((t, k.shape[-2]), neg, dtype=s.dtype), 1)
            s = s + mask
        return s.softmax(-1) @ v
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({N}, {K}), _r({N}, {Dv}))
""",
    "rmsnorm": """
class Model(nn.Module):
    def forward(self, x, g):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + {eps}) * g
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r(1, {K}) * 0.5 + 1.0)
""",
    "softmax": """
class Model(nn.Module):
    def forward(self, x):
        return x.softmax(-1)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "layernorm": """
class Model(nn.Module):
    def forward(self, x, w, b):
        return torch.nn.functional.layer_norm(x, ({K},), w, b, {eps})
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}) * 0.5 + 1.0, _r({K}) * 0.1)
""",
    "geglu": """
class Model(nn.Module):
    def forward(self, x, wg, wu):
        return torch.nn.functional.silu(x @ wg) * (x @ wu)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({K}, {N}))
""",
    "gelu": """
class Model(nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x, approximate="tanh")
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "silu": """
class Model(nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "add": """
class Model(nn.Module):
    def forward(self, a, b):
        return a + b
def get_model_and_inputs():
    return Model(), (_r({shape_args}), _r({shape_args}))
""",
    "mul": """
class Model(nn.Module):
    def forward(self, a, b):
        return a * b
def get_model_and_inputs():
    return Model(), (_r({shape_args}), _r({shape_args}))
""",
    # The next four are spelled as model2MLIR names the operations real captures carry (`prov.op`), so a
    # host-lane probe can be built from an operation the captured host work actually consists of.
    "layer_norm": """
class Model(nn.Module):
    def forward(self, x, w, b):
        return torch.nn.functional.layer_norm(x, ({K},), w, b, {eps})
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}) * 0.5 + 1.0, _r({K}) * 0.1)
""",
    "reduce_mean": """
class Model(nn.Module):
    def forward(self, x):
        return x.mean(-1, keepdim=True)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "batch_matmul": """
class Model(nn.Module):
    def forward(self, a, w):
        return torch.bmm(a, w)
def get_model_and_inputs():
    return Model(), (_r({B}, {M}, {K}), _r({B}, {K}, {N}))
""",
    "permute": """
class Model(nn.Module):
    def forward(self, x):
        return x.permute({perm_args})
def get_model_and_inputs():
    return Model(), (_r({shape_args}),)
""",
    "dtype_cast": """
class Model(nn.Module):
    def forward(self, x):
        return x.to(dtype=torch.{native_output_torch_dtype})
def get_model_and_inputs():
    return Model(), (_r({shape_args}),)
""",
    "reduce_sum": """
class Model(nn.Module):
    def forward(self, x):
        return x.sum(-1, keepdim=True)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "rope": """
def _rope(x, half):
    freq = 1.0 / (10000 ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(x.shape[-2], dtype=torch.float32)
    ang = pos[:, None] * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1); sin = torch.cat([ang.sin(), ang.sin()], -1)
    x1, x2 = x[..., :half], x[..., half:]
    return x * cos + torch.cat([-x2, x1], -1) * sin
class Model(nn.Module):
    def forward(self, x):
        return _rope(x, {K} // 2)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    # --- elementwise / composite ops from real model graphs (all FUNCTIONAL, operands as inputs) ---
    "bias_add": """
class Model(nn.Module):
    def forward(self, x, b):
        return x + b
def get_model_and_inputs():
    return Model(), (_r({M}, {N}), _r({N}))
""",
    "fused_matmul_bias": """
class Model(nn.Module):
    def forward(self, x, w, b):
        return x @ w + b
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({N}))
""",
    "k_chain": """
class Model(nn.Module):
    def forward(self, a, w1, w2):
        return (a @ w1) @ w2
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r({K}, {N}), _r({N}, {N}))
""",
    "logit_softcap": """
class Model(nn.Module):
    def forward(self, x):
        c = {cap}
        return c * torch.tanh(x / c)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "embed_scale": """
class Model(nn.Module):
    def forward(self, x):
        return x * ({K} ** 0.5)
def get_model_and_inputs():
    return Model(), (_r({M}, {K}),)
""",
    "gemma_4norm": """
def _rms(x, g, eps):
    v = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(v + eps) * g
class Model(nn.Module):
    def forward(self, x, g1, g2):
        return _rms(_rms(x, g1, {eps}), g2, {eps})
def get_model_and_inputs():
    return Model(), (_r({M}, {K}), _r(1, {K}) * 0.5 + 1.0, _r(1, {K}) * 0.5 + 1.0)
""",
    "gemv_batched": """
class Model(nn.Module):
    def forward(self, a, v):
        return torch.bmm(a, v)
def get_model_and_inputs():
    return Model(), (_r({B}, {M}, {K}), _r({B}, {K}, 1))
""",
    "patch_embed": """
class Model(nn.Module):
    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w, stride={P})
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}), _r({N}, {Cin}, {P}, {P}))
""",
    # --- conv/pool family that runs on the vector lanes (fused, host-eager golden) ---
    "depthwise_conv2d": """
class Model(nn.Module):
    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w, stride=1, padding={P} // 2, groups={Cin})
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}), _r({Cin}, 1, {P}, {P}))
""",
    # pooling decomposed into reshape+reduce so it lowers to pure linalg (torch-mlir leaves the fused
    # aten pool ops opaque); kernel == stride == P, exact same numerics as F.{max,avg}_pool2d.
    "maxpool2d": """
class Model(nn.Module):
    def forward(self, x):
        return x.reshape(1, {Cin}, {Himg} // {P}, {P}, {Wimg} // {P}, {P}).amax((3, 5))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}),)
""",
    "avgpool2d": """
class Model(nn.Module):
    def forward(self, x):
        return x.reshape(1, {Cin}, {Himg} // {P}, {P}, {Wimg} // {P}, {P}).mean((3, 5))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}),)
""",
    "global_average": """
class Model(nn.Module):
    def forward(self, x):
        return x.mean((-2, -1))
def get_model_and_inputs():
    return Model(), (_r(1, {Cin}, {Himg}, {Wimg}),)
""",
}


# Weight quantization needs a module parameter; input-input matmul supplies no
# weight for TorchAO to transform. This template preserves that distinction.
_PARAMETRIC_LINEAR = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Linear({K}, {N}, bias=False)
        with torch.no_grad():
            self.f.weight.copy_(_r({N}, {K}))
    def forward(self, a):
        return self.f(a)
def get_model_and_inputs():
    return Model().eval(), (_r({M}, {K}),)
"""


# This slice starts at captured i8 operands and ends at the i32 accumulator.
# Quantizing it again would change the computation selected by the model.
_INT_MATMUL = """
class Model(nn.Module):
    def forward(self, a, w):
        return torch._int_mm(a, w)
def get_model_and_inputs():
    a = torch.randint(-8, 8, ({M}, {K}), dtype=torch.int8, generator=_G)
    w = torch.randint(-8, 8, ({K}, {N}), dtype=torch.int8, generator=_G)
    return Model().eval(), (a, w)
"""


_OP_INPUT_NAMES = {
    "matmul": ["A0", "W"],
    "linear": ["A0", "W"],
    "attention_qk": ["Q", "K"],
    "rmsnorm": ["X", "G"],
}


_FUSED_OP_INPUT_NAMES = {
    "transpose": ["X"],
    "attention_full": ["Q", "K", "V"],
    "softmax": ["X"],
    "layernorm": ["X", "W", "B"],
    "geglu": ["X", "WG", "WU"],
    "rope": ["X"],
    "gelu": ["X"],
    "silu": ["X"],
    "add": ["A", "B"],
    "mul": ["A", "B"],
    "reduce_sum": ["X"],
    "layer_norm": ["X", "W", "B"],
    "reduce_mean": ["X"],
    "batch_matmul": ["A0", "W"],
    "permute": ["X"],
    "dtype_cast": ["X"],
    # composite ops from real model graphs (linalg-as-interface, positional args)
    "bias_add": ["X", "B"],
    "fused_matmul_bias": ["X", "W", "B"],
    "k_chain": ["A0", "W", "W2"],
    "logit_softcap": ["X"],
    "embed_scale": ["X"],
    "gemma_4norm": ["X", "G1", "G2"],
    "gemv_batched": ["A0", "V"],
    "patch_embed": ["X", "W"],
    "depthwise_conv2d": ["X", "W"],
    "maxpool2d": ["X"],
    "avgpool2d": ["X"],
    "global_average": ["X"],
}


_OP_BODIES.update(component_sources.BODIES)


_FUSED_OP_INPUT_NAMES.update(component_sources.INPUT_NAMES)


def supported_ops() -> list[str]:
    return sorted(_OP_BODIES)


def build_loader_src(spec: dict) -> str:
    """Render the selected builtin template through shared typed input validation."""
    from merlin.targetgen.capsule_builtin_source import render_builtin_source

    return render_builtin_source(
        spec,
        preamble=_PREAMBLE,
        integer_matmul=_INT_MATMUL,
        parametric_linear=_PARAMETRIC_LINEAR,
        bodies=_OP_BODIES,
        input_names=_OP_INPUT_NAMES | _FUSED_OP_INPUT_NAMES,
    )
