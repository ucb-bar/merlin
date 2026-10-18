"""One fusion pass, many SYNTHETIC output paths: the declared sequence alone decides each grouping.

Every capability set here is fabricated -- no target, no facts file, no names -- so a grouping that
differs between two of them can only come from the declared path and the exactness proof.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from fake_quant_layer import Oracle as _Oracle

from merlin.common import mlir_query as mq
from merlin.targetgen import output_path as OP
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import output_fusion as OF

_ID2 = "affine_map<(d0, d1) -> (d0, d1)>"
_COL = "affine_map<(d0, d1) -> (d1)>"
_MUL = "    ^bb0(%a: f32, %b: f32, %o: f32):\n      %c = arith.mulf %a, %b : f32\n      linalg.yield %c : f32"
_ADD = "    ^bb0(%a: f32, %b: f32, %o: f32):\n      %c = arith.addf %a, %b : f32\n      linalg.yield %c : f32"
_MATMUL = (
    "    %mm = linalg.matmul ins(%x, %w : tensor<4x8xi8>, tensor<8x8xi8>) outs(%init : tensor<4x8xi32>) "
    "-> tensor<4x8xi32>"
)


def _generic(result, ins, types, maps, body, out_type="tensor<4x8xf32>"):
    return (
        f"    %e_{result} = tensor.empty() : {out_type}\n"
        f"    %{result} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = "
        f'["parallel", "parallel"]}} ins({ins} : {types}) outs(%e_{result} : {out_type}) {{\n{body}\n'
        f"    }} -> {out_type}\n"
    )


def _dense(values):
    return "dense<[" + ", ".join(f"{float(v):.6e}" for v in values) + "]>"


def _module(*, weight_scale=(0.5,), bias=(0, 64, -64, 128, 8, -8, 0, 256), relu=True, residual=False,
            pool=False, quant=(-128, 127), out_scale=8.0, weight_step=1):  # fmt: skip
    """``x`` (int8) @ ``w`` (int8) -> i32 -> f32 -> *0.25 -> *weight_scale -> +bias -> [+residual] -> [relu]
    -> [row-pair max pool] -> quantize. The accumulator unit is 0.125 when ``weight_scale`` is 0.5."""
    scales = weight_scale * 8 if len(weight_scale) == 1 else weight_scale
    unit = [0.25 * s for s in scales]
    weights = (
        "dense<"
        + "["
        + ", ".join("[" + ", ".join(str(((i * 3 + j) % 5 - 2) * weight_step) for j in range(8)) + "]" for i in range(8))
        + "]>"
    )
    lines = [
        "builtin.module {",
        "  func.func @forward(%x: tensor<4x8xi8>, %rq: tensor<4x8xi8>) -> tensor<"
        + ("2" if pool else "4")
        + "x8xi8> {",
        f"    %w = arith.constant {weights} : tensor<8x8xi8>",
        "    %zero = arith.constant 0 : i32",
        "    %e0 = tensor.empty() : tensor<4x8xi32>",
        "    %init = linalg.fill ins(%zero : i32) outs(%e0 : tensor<4x8xi32>) -> tensor<4x8xi32>",
        _MATMUL,
        _generic("f", "%mm", "tensor<4x8xi32>", [_ID2, _ID2],
                 "    ^bb0(%a: i32, %o: f32):\n      %c = arith.sitofp %a : i32 to f32\n      linalg.yield %c : f32"),
        "    %sin = arith.constant 2.500000e-01 : f32",
        "    %s0 = tensor.splat %sin : tensor<4x8xf32>",
        _generic("m1", "%f, %s0", "tensor<4x8xf32>, tensor<4x8xf32>", [_ID2, _ID2, _ID2],
                 _MUL),
        f"    %sw = arith.constant {_dense(scales)} : tensor<8xf32>",
        _generic("m2", "%m1, %sw", "tensor<4x8xf32>, tensor<8xf32>", [_ID2, _COL, _ID2],
                 _MUL),
        f"    %bias = arith.constant {_dense([b * u for b, u in zip(bias, unit)])} : tensor<8xf32>",
        _generic("ba", "%m2, %bias", "tensor<4x8xf32>, tensor<8xf32>", [_ID2, _COL, _ID2],
                 _ADD),
    ]  # fmt: skip
    last = "%ba"
    if residual:
        lines += [
            "    %rs = arith.constant dense<1.250000e-01> : tensor<f32>",
            "    %rz = arith.constant dense<0> : tensor<i64>",
            '    %rd = "quant_ext.dequantize_per_tensor"(%rq, %rs, %rz) <{quant_min = -128 : i64, quant_max = 127 : '
            "i64}> : (tensor<4x8xi8>, tensor<f32>, tensor<i64>) -> tensor<4x8xf32>",
            _generic("sum", "%ba, %rd", "tensor<4x8xf32>, tensor<4x8xf32>", [_ID2, _ID2, _ID2],
                     _ADD),
        ]  # fmt: skip
        last = "%sum"
    if relu:
        lines.append(
            _generic("r", last, "tensor<4x8xf32>", [_ID2, _ID2],
                     "    ^bb0(%a: f32, %o: f32):\n      %z = arith.constant 0.000000e+00 : f32\n"
                     "      %c = arith.maximumf %a, %z : f32\n      linalg.yield %c : f32")
        )  # fmt: skip
        last = "%r"
    rows = 4
    if pool:
        lines += [
            "    %ninf = arith.constant 0xFF800000 : f32",
            "    %pe = tensor.empty() : tensor<2x8xf32>",
            "    %pi = linalg.fill ins(%ninf : f32) outs(%pe : tensor<2x8xf32>) -> tensor<2x8xf32>",
            "    %win = tensor.empty() : tensor<2xf32>",
            "    %p = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0 * 2 + d2, d1)>, "
            "affine_map<(d0, d1, d2) -> (d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = "
            f'["parallel", "parallel", "reduction"]}} ins({last}, %win : tensor<4x8xf32>, tensor<2xf32>) '
            "outs(%pi : tensor<2x8xf32>) {\n    ^bb0(%a: f32, %w2: f32, %o: f32):\n"
            "      %c = arith.maximumf %o, %a : f32\n      linalg.yield %c : f32\n    } -> tensor<2x8xf32>",
        ]
        last, rows = "%p", 2
    lines += [
        f"    %qs = arith.constant dense<{out_scale:.6e}> : tensor<f32>",
        "    %qz = arith.constant dense<0> : tensor<i64>",
        f'    %q = "quant_ext.quantize_per_tensor"({last}, %qs, %qz) <{{quant_min = {quant[0]} : i64, quant_max = '
        f'{quant[1]} : i64, output_dtype = "int8"}}> : (tensor<{rows}x8xf32>, tensor<f32>, tensor<i64>) -> '
        f"tensor<{rows}x8xi8>",
        f"    func.return %q : tensor<{rows}x8xi8>",
        "  }",
        "}",
    ]
    return "\n".join(lines)


_BASE = {
    "schema": OP.SCHEMA,
    "accumulator": {"dtype": "i32"},
    "accumulator_stages": [
        {"kind": "preload", "dtype": "i32", "range": [-(1 << 31), (1 << 31) - 1], "rounding": "none"},
        {"kind": "accumulate", "dtype": "i32", "operand_scale": "none"},
    ],
    "readout": [
        {"kind": "activation", "function": "relu", "domain": "accumulator"},
        {"kind": "convert", "to": "f32", "rounding": "half_even"},
        {
            "kind": "scale",
            "dtype": "f32",
            "rounding": "half_even",
            "granularity": "per_command",
            "columns_per_scale": 4,
        },
        {"kind": "requant", "to": "i8", "rounding": "half_even", "clamp": [-128, 127]},
        {"kind": "pool", "reductions": ["max"], "padding": "unknown"},
    ],
    "exits": {"accumulator": True, "requant": True},
}


def _path(edit=None):
    path = copy.deepcopy(_BASE)
    if edit:
        edit(path)
    return OP.validate(path)


def _drop(kind):
    def edit(path):
        path["readout"] = [s for s in path["readout"] if s["kind"] != kind]
        path["accumulator_stages"] = [s for s in path["accumulator_stages"] if s["kind"] != kind]

    return edit


def _fused(text, path):
    module = mq.parse(text)
    first = CG.form_groups(module, "synthetic", oracle=_Oracle())
    decided = OF.decisions(first, module, path, OF.Values())
    groups = CG.form_groups(module, "synthetic", oracle=_Oracle(), fusion=decided)
    (device,) = [g for g in groups if g.placement != CG.HOST]
    return device, [g for g in groups if g.placement == CG.HOST]


def _codes(device):
    return [(r["cut"], r["code"]) for r in device.fusion["refusals"]]


@pytest.mark.parametrize("path", [_path(), _path(_drop("activation")), _path(_drop("preload"))])
def test_capture_bound_source_formation_uses_the_same_exact_cut(path, monkeypatch, tmp_path):
    from functools import partial

    from merlin.llvmlower.source_compute_groups import form_exact_groups, form_source_groups

    expected, _ = _fused(_module(), path)
    module = mq.parse(_module())
    monkeypatch.setattr(OF, "declared_path", lambda target: path)
    groups, _ = form_exact_groups(
        module, "synthetic", tmp_path, partial(form_source_groups, oracle=_Oracle())
    )
    (actual,) = [group for group in groups if group.placement != CG.HOST]
    assert actual.fusion == expected.fusion
    assert actual.stages == expected.stages


def test_exact_fusion_statement_refusal_never_falls_back_to_bare_contraction(monkeypatch, tmp_path):
    from merlin.llvmlower.device_offload import rewrite_groups_to_device
    from merlin.xdsl_dialects.lowering import group_command as GC

    monkeypatch.setattr(OF, "declared_path", lambda target: _path())
    original = GC.program

    def no_requant_form(group, **kwargs):
        if CG.QUANTIZE in group.stages:
            raise CG.NoCapsuleForm("selected requant form is unavailable")
        return original(group, **kwargs)

    monkeypatch.setattr(GC, "program", no_requant_form)
    module = mq.parse(_module())
    result = rewrite_groups_to_device(
        module, "synthetic", select=lambda _: True, oracle=_Oracle(), capture=tmp_path
    )
    assert result.moved == 0
    assert result.skipped == (("group0", "no device form: selected requant form is unavailable"),)
    assert any(mq.op_name(op) == "quant_ext.quantize_per_tensor" for op in module.walk())


def test_the_full_path_takes_the_whole_chain_with_the_bias_preloaded():
    device, host = _fused(_module(), _path())
    assert device.fusion["cut"] == "requant"
    assert device.fusion["stages"] == ["convert", "scale", "scale", "bias", "relu", "quantize"]
    assert device.fusion["proof"]["offsets"] == [0, 64, -64, 128, 8, -8, 0, 256]
    assert device.fusion["proof"]["commands"] == 2 and device.fusion["proof"]["scale_reconfigurations"] == 0
    assert not [g for g in host if CG._computes(g.stages)]
    assert _codes(device) == [("pool", OF.UNDECLARED)]  # nothing to pool in this chain


def test_no_activation_keeps_the_readout_on_the_host_and_preloads_the_bias():
    device, host = _fused(_module(), _path(_drop("activation")))
    assert device.fusion["cut"] == "accumulator" and device.fusion["stages"] == ["bias"]
    assert ("requant", OF.UNDECLARED) in _codes(device)
    assert "rectifier" in device.fusion["refusals"][1]["reason"]
    assert any(CG.RELU in g.stages and CG.QUANTIZE in g.stages for g in host)


def test_an_activation_after_the_scale_is_the_same_exact_readout():
    def after(path):
        path["readout"] = [path["readout"][1], path["readout"][2], {**path["readout"][0], "domain": "scaled"},
                           *path["readout"][3:]]  # fmt: skip

    device, _host = _fused(_module(), _path(after))
    assert device.fusion["cut"] == "requant"


def test_a_per_channel_scale_is_taken_only_where_the_readout_holds_one():
    per_channel = (0.5, 0.25, 0.5, 0.125, 0.5, 0.25, 0.5, 0.125)
    text = _module(weight_scale=per_channel, bias=(0,) * 8)

    def channel(path):
        path["readout"][2]["granularity"] = "per_channel"

    device, _host = _fused(text, _path(channel))
    assert device.fusion["cut"] == "requant"
    device, _host = _fused(text, _path())
    assert device.fusion["cut"] == "accumulator" and device.fusion["stages"] == ["bias"]
    assert ("requant", OF.GRANULARITY) in _codes(device)


@pytest.mark.parametrize(("columns", "cut"), [(4, "requant"), (8, "accumulator")])
def test_a_different_mesh_size_changes_which_scales_one_command_holds(columns, cut):
    blocks = (0.5,) * 4 + (0.25,) * 4  # uniform within each group of four columns

    def width(path):
        path["readout"][2]["columns_per_scale"] = columns

    device, _host = _fused(_module(weight_scale=blocks), _path(width))
    assert device.fusion["cut"] == cut
    if cut == "requant":
        assert device.fusion["proof"]["commands"] == 2 and device.fusion["proof"]["scale_reconfigurations"] == 1
    else:
        assert ("requant", OF.GRANULARITY) in _codes(device)


def test_without_a_preload_a_nonzero_bias_cannot_be_absorbed_at_all():
    device, _host = _fused(_module(), _path(_drop("preload")))
    assert device.fusion["cut"] == "bare"
    assert ("requant", OF.INEXACT) in _codes(device) and ("accumulator", OF.UNDECLARED) in _codes(device)
    zero_bias, _host = _fused(_module(bias=(0,) * 8), _path(_drop("preload")))
    assert zero_bias.fusion["cut"] == "requant"


def test_a_residual_is_accumulated_only_where_the_path_accumulates():
    text = _module(residual=True, quant=(-8, 7))
    device, _host = _fused(text, _path())
    assert device.fusion["cut"] == "requant" and "residual" in device.fusion["stages"]
    device, host = _fused(text, _path(_drop("accumulate")))
    assert device.fusion["cut"] == "accumulator" and device.fusion["stages"] == ["bias"]
    assert device.fusion["proof"]["residual_on_host"]["code"] == OF.UNDECLARED
    assert any(CG.RESIDUAL_ADD in g.stages for g in host)


def test_a_residual_off_the_accumulator_scale_is_refused():
    text = _module(residual=True, quant=(-8, 7), weight_scale=(0.25,))  # unit 0.0625, residual 0.125
    device, _host = _fused(text, _path())
    assert all(code != OF.GRANULARITY for _cut, code in _codes(device))
    assert ("requant", OF.INEXACT) in _codes(device)
    assert "accumulator's scale" in next(r["reason"] for r in device.fusion["refusals"] if r["cut"] == "requant")


def test_pooling_is_taken_only_where_the_path_pools():
    device, _host = _fused(_module(pool=True), _path())
    assert device.fusion["cut"] == "pool" and "pool" in device.fusion["stages"]
    device, host = _fused(_module(pool=True), _path(_drop("pool")))
    assert device.fusion["cut"] == "accumulator"
    assert ("pool", OF.UNDECLARED) in _codes(device) and ("requant", OF.UNDECLARED) in _codes(device)


def test_a_different_scale_format_is_proved_in_that_format():
    def half(path):
        path["readout"][1]["to"] = "f16"
        path["readout"][2]["dtype"] = "f16"

    # Weights large enough that the accumulator passes 2048, past which f16 cannot hold every integer.
    device, _host = _fused(_module(bias=(0,) * 8, weight_step=15), _path(half))
    assert ("requant", OF.INEXACT) in _codes(device)
    exact, _host = _fused(_module(bias=(0,) * 8, weight_step=15), _path())
    assert exact.fusion["cut"] == "requant"  # the same chain is exact in f32
    assert device.fusion["cut"] == "accumulator" and device.fusion["stages"] == ["bias"]

    def unmodeled(path):
        path["readout"][2]["dtype"] = "bf16"

    device, _host = _fused(_module(), _path(unmodeled))
    assert ("requant", OF.UNMODELED) in _codes(device)


def test_an_accepted_offset_reproduces_every_output_by_brute_force():
    text = _module()
    module = mq.parse(text)
    first = CG.form_groups(module, "synthetic", oracle=_Oracle())
    root = next(g.root for g in first if g.placement != CG.HOST)
    values = OF.Values()
    chain = OF.source_chain(root, values)
    from merlin.targetgen import model_coverage as MC

    decision = OF.decide(chain, _path(), values, MC._contraction_extents(module))
    offsets = np.asarray(decision.proof["offsets"], dtype=np.int64)
    scale = np.asarray(decision.proof["device_scale"])
    lo, hi = decision.proof["accumulator_bound"]
    for x in range(lo, hi + 1, 7):
        xs = np.full(8, x, dtype=np.int64)
        source = OF._evaluate(chain.stages, np.float32, xs)
        device = OF._device(_path()["readout"][:-1], scale, xs + offsets, relu=True)
        assert (source == device).all(), x


def test_a_float_bias_off_the_accumulator_grid_is_refused_not_rounded():
    text = _module().replace(_dense([b * 0.125 for b in (0, 64, -64, 128, 8, -8, 0, 256)]), _dense([0.03] * 8))
    device, _host = _fused(text, _path())
    assert ("requant", OF.INEXACT) in _codes(device) and ("accumulator", OF.INEXACT) in _codes(device)
    assert device.fusion["cut"] == "bare"
