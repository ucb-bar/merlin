"""The spatial convolution-backward capsule: the three gradients of a padded 3x3 stride-1 convolution, one program.

Forward: ``Y[L, N] = Xcol[L, K'] @ W[K', N]`` with ``Xcol`` the im2col of an NCHW image (``K' = C*kh*kw`` in
(C, kh, kw) order, ``L = Ho*Wo`` blocks). The gradients of ``aten.convolution_backward`` are

* ``grad_weight = Xcol^T @ dY``      ``[K', N]``  a contraction of the permuted im2col with the output gradient;
* ``grad_input_col = W @ dY^T``      ``[K', L]``  a contraction (the transposed convolution before its fold), read out
                                                  at int8 (the quantized backward requantizes between operations);
* ``grad_input = fold(grad_input_col)``  ``[1, C, H, W]``  the overlap-add of every block's taps onto the image
  (``col2im``: ``window_sum`` mode ``fold``, an exact i32 sum with zero-padding taps dropped);
* ``grad_bias = sum_l dY``           ``[1, N]``   the standalone integer sum down the outer axis (``window_sum`` reduce).

Written as ONE interface program of declared commands (permute movement, two contractions, a reshape, two window sums).
Every region is a form the spec already declares (contraction row, reduction_standalone's fold and sum, the movement
permutation and copy); nothing is lowered here. The golden composes the existing integer semantics
(``Tensor.matmul``, ``_apply_epilogue``, ``integer_ops.move`` and ``integer_ops.window_sum``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .corpus_spec import CorpusBinding

#: The canonical case: a 4-channel 4x4 image, 3x3 kernel, stride 1, padding 1 -> 16 blocks; 16 output channels.
CASE: dict[str, Any] = {"C": 4, "H": 4, "W": 4, "kernel": [3, 3], "stride": [1, 1], "padding": [1, 1], "N": 16}
#: The int8 readout of the transposed-convolution contraction: a power-of-two gain (1/2048), as the readout rows state.
GRAD_INPUT_COL_SCALE = 2.0**-11


def geometry(entry: dict) -> dict[str, int]:
    case = {**CASE, **{k: entry[k] for k in CASE if k in entry}}
    kh, kw = case["kernel"]
    (sh, sw), (ph, pw) = case["stride"], case["padding"]
    ho = (case["H"] + 2 * ph - kh) // sh + 1
    wo = (case["W"] + 2 * pw - kw) // sw + 1
    return {**case, "L": ho * wo, "K": case["C"] * kh * kw}


def build_conv_backward_spatial(entry: dict, binding: CorpusBinding) -> tuple[dict, str]:
    from .corpus_spec import _iface_prelude, _mlir_attr, _numeric_policy, _tensor_type

    g = geometry(entry)
    L, K, N, C, H, W = g["L"], g["K"], g["N"], g["C"], g["H"], g["W"]
    idt, midt = binding.cap_dtype(binding.operand_dtype), binding.mlir_dtype(binding.operand_dtype)
    adt, madt = binding.cap_dtype(binding.accum_dtype), binding.mlir_dtype(binding.accum_dtype)
    x, dy, w = "Xcol", "dY", "W"
    outs = ["grad_weight", "grad_input", "grad_bias"]
    aten = "aten.convolution_backward.default"
    fold = {"kernel": g["kernel"], "stride": g["stride"], "padding": g["padding"], "dilation": [1, 1]}
    attrs = {
        "xcol": x, "dy": dy, "weight": w, "outs": outs, "out": outs[0], "geometry": g, "fold": fold,
        "grad_input_col_scale": GRAD_INPUT_COL_SCALE, "output_dtype": adt, "semantic": "convolution_backward",
    }  # fmt: skip
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": x, "role": "input", "shape": [L, K], "dtype": idt},
            {"name": dy, "role": "input", "shape": [L, N], "dtype": idt},
            {"name": w, "role": "weight", "shape": [K, N], "dtype": idt},
        ],
        "operation": {"op": "conv_backward_spatial", "attributes": attrs},
        "numeric_policy": _numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=adt),
            "modes": {},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }

    def ty(shape: list[int], dtype: str) -> str:
        return _tensor_type(shape, dtype)

    def window(name, src, src_shape, out_shape, mode, fields):
        body = ", ".join(
            [f'name = "{name}"', f'mode = "{mode}"', *fields, 'layout = "nchw"', "readout_scale = 1.0 : f32"]
            + [f'output_dtype = "{adt}"', f'prov.aten = "{aten}"']
        )
        return (
            f"  %{name} = merlin_iface.window_sum %{src} {{{body}}} : "
            f"({ty(src_shape, midt)}) -> {ty(out_shape, madt)}"
        )

    lines = _iface_prelude(binding.target, entry.get("comment", ""))
    lines += [
        f'  %{x} = merlin_iface.tensor {{name = "{x}", role = "input"}} : {ty([L, K], midt)}',
        f'  %{dy} = merlin_iface.tensor {{name = "{dy}", role = "input"}} : {ty([L, N], midt)}',
        f'  %{w} = merlin_iface.tensor {{name = "{w}", role = "weight"}} : {ty([K, N], midt)}',
        # grad_weight = Xcol^T @ dY
        f'  %xt = merlin_iface.movement %{x} {{name = "xt", form = "permute", perm = [1, 0], output_dtype = "{idt}"}} '
        f": ({ty([L, K], midt)}) -> {ty([K, L], midt)}",
        f'  %{dy}_res = merlin_iface.resident_pack %{dy} {{layout = "packed_rhs"}} '
        f": ({ty([L, N], midt)}) -> !merlin_iface.resident",
        f"  %acc_w = merlin_iface.matmul %xt, %{dy}_res : ({ty([K, L], midt)}, !merlin_iface.resident) "
        f"-> !merlin_iface.acc<{madt}>",
        f'  %{outs[0]} = merlin_iface.commit %acc_w {{name = "{outs[0]}", epilogue = [], output_dtype = "{adt}", '
        f'prov.aten = "{aten}"}} : (!merlin_iface.acc<{madt}>) -> {ty([K, N], madt)}',
        # grad_input_col = W @ dY^T, read out at int8
        f'  %dyt = merlin_iface.movement %{dy} {{name = "dyt", form = "permute", perm = [1, 0], output_dtype = "{idt}"}} '
        f": ({ty([L, N], midt)}) -> {ty([N, L], midt)}",
        f'  %dyt_res = merlin_iface.resident_pack %dyt {{layout = "packed_rhs"}} '
        f": ({ty([N, L], midt)}) -> !merlin_iface.resident",
        f"  %acc_i = merlin_iface.matmul %{w}, %dyt_res : ({ty([K, N], midt)}, !merlin_iface.resident) "
        f"-> !merlin_iface.acc<{madt}>",
        f'  %gic = merlin_iface.commit %acc_i {{name = "gic", epilogue = ["acc_scale"], output_dtype = "{idt}", '
        f'acc_scale = {GRAD_INPUT_COL_SCALE} : f32, prov.aten = "{aten}"}} : (!merlin_iface.acc<{madt}>) -> '
        f"{ty([K, L], midt)}",
        # grad_input = fold(grad_input_col)
        f'  %gic3 = merlin_iface.movement %gic {{name = "gic3", form = "reshape", output_dtype = "{idt}"}} '
        f": ({ty([K, L], midt)}) -> {ty([1, K, L], midt)}",
        window(
            outs[1], "gic3", [1, K, L], [1, C, H, W], "fold",
            [f"{k} = {_mlir_attr(v)}" for k, v in fold.items()],
        ),  # fmt: skip
        # grad_bias = column sum of dY
        window(
            outs[2], dy, [L, N], [1, N], "reduce", ["dims = [0]", "average = false"],
        ),  # fmt: skip
        "}",
    ]
    return cap, "\n".join(lines) + "\n"
