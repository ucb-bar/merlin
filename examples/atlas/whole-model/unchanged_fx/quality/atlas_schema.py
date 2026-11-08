"""Schema-only registration for loading FX. Eager reference bodies are forbidden."""
import torch

_LIBRARIES = []


def register():
    if hasattr(torch.ops.atlas, "qlinear"):
        return
    lib = torch.library.Library("atlas", "DEF")
    schemas = [
        'quantize_act(Tensor x, str mode, str reduce, str rule, int? static_exp, str code_dtype, str tag="", int rows=32) -> (Tensor, Tensor)',
        'qlinear(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format) -> Tensor',
        'qconv2d(Tensor x_codes, Tensor x_exp, Tensor w_codes, Tensor w_exp, Tensor? bias, str out_dtype, str weight_format, int batch, int out_h, int out_w) -> Tensor',
        'qk_matmul(Tensor q_codes, Tensor q_exp, Tensor k_codes, Tensor k_exp) -> Tensor',
        'pv_matmul(Tensor p_codes, Tensor p_exp, Tensor vt_codes, Tensor v_exp, int pop_shift=0) -> Tensor',
    ]
    for schema in schemas:
        lib.define(schema)
    _LIBRARIES.append(lib)
