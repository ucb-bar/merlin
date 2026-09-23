"""A closed group's integer arithmetic is read from the group and agrees with the captured graph."""

from __future__ import annotations

import numpy as np
import pytest
from fake_quant_layer import Oracle as _Oracle  # noqa: E402
from fake_quant_layer import module as _module

from merlin.common import mlir_query as mq
from merlin.targetgen import readout_facet as RF
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_numerics as GN


def _closed_group(**kwargs) -> CG.Group:
    (group,) = CG.form_groups(
        mq.parse(_module(weight_dequantize="per_tensor", **kwargs)), "synthetic", oracle=_Oracle()
    )
    return group


def test_the_scales_the_bias_rule_and_the_multiplier_come_from_the_group() -> None:
    numerics = GN.numerics_of(_closed_group())
    assert (numerics.input.value, numerics.weight.value, numerics.output.value) == (0.5, 0.5, 0.5)
    assert numerics.multiplier == pytest.approx(0.5)  # 0.5 * 0.5 / 0.5
    # %b is the fifth model argument; its accumulator-domain value is b / (s_x * s_w).
    assert (numerics.bias_arg_index, numerics.bias_divisor) == (4, 0.25)
    assert (numerics.activation, numerics.clamp, numerics.granularity) == ("relu", (-128, 127), "tensor")
    assert GN.stages(numerics) == ("bias_i32", "scale_f32", "round_to_nearest_even", "clamp", "relu")


def test_the_integer_group_computes_what_the_captured_graph_computes() -> None:
    numerics = GN.numerics_of(_closed_group())
    rng = np.random.default_rng(7)
    accumulator = rng.integers(-40000, 40000, size=(64, 16))
    bias = rng.normal(0.0, 200.0, size=16)
    integer = GN.reference_integer(numerics, accumulator, GN.prepack_bias(numerics, bias))
    floating = GN.reference_fake_quant(numerics, accumulator, bias)
    # They differ only by the bias rounded into accumulator units and one f32 product: at most one
    # output code, and only on values that sit on a rounding boundary.
    difference = np.abs(integer - floating)
    assert difference.max() <= 1 and (difference == 0).mean() > 0.9
    assert (integer >= 0).all() and integer.max() == 127  # the relu and the saturation both bite


def test_the_derived_capability_admits_a_per_tensor_group() -> None:
    abi = {
        "schema": RF.SCALAR_ABI_SCHEMA,
        "accumulator_dtype": "i32",
        "output_dtype": "i8",
        "scale_dtype": "f32",
        "clamp_min": -128,
        "clamp_max": 127,
        "provenance": {},
    }
    facts = {
        "facts": {
            "datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}],
            "interfaces": [
                {
                    "name": "register_bundle_layouts",
                    "unresolved": {},
                    "bundles": {
                        "StoreConfig": {
                            "width": 64,
                            "fields": {
                                "out_scale": {
                                    "offset": 32,
                                    "width": None,
                                    "width_param": "scale_bits",
                                    "slot_width": 32,
                                }
                            },
                        }
                    },
                }
            ],
        }
    }
    facet = RF.derive(
        "t",
        facts=facts,
        unit={"name": "u"},
        scalar_abi=abi,
        readouts=[{"selector": "i8", "applies": ["acc_scale", "relu", "bias_add"]}],
    )
    capability = RF.epilogue_capability(facet, name="t:u")
    numerics = GN.numerics_of(_closed_group())
    admitted = GN.candidate(numerics, "t:u", bias=(3,))
    assert admitted.ordered_stages in capability.ordered_stage_templates
    assert admitted.scale_granularity in capability.scale_granularities
    assert admitted.saturation in capability.saturations
    assert admitted.activation in capability.activations


def test_a_group_that_is_not_closed_or_whose_scale_is_stored_is_refused_with_the_reason() -> None:
    open_group = next(
        g
        for g in CG.form_groups(mq.parse(_module(weight_dequantize="per_channel")), "synthetic", oracle=_Oracle())
        if g.placement != CG.HOST
    )
    with pytest.raises(GN.GroupNumericsError, match="not closed"):
        GN.numerics_of(open_group)
    (closed_per_channel,) = CG.form_groups(
        mq.parse(_module(weight_dequantize="per_channel")), "synthetic", oracle=_Oracle(holds=("tensor", "column"))
    )
    numerics = GN.numerics_of(closed_per_channel)
    assert numerics.weight.value is None and numerics.weight.arg_index == 2  # %ws, a stored scale
    assert numerics.multiplier is None
    with pytest.raises(GN.GroupNumericsError, match="not a compile-time number"):
        GN.candidate(numerics, "t:u")


def test_a_zero_point_of_zero_is_a_number_not_a_missing_value() -> None:
    # An integer attribute holding zero is falsy. Reading it with `a or b` reported every symmetric
    # quantizer's zero point as absent, so no real capture's numerics could be read.
    text = """
builtin.module {
  func.func @f(%x: tensor<4xi8>) -> tensor<4xf32> {
    %s = arith.constant 2.500000e-01 : f32
    %st = tensor.splat %s : tensor<f32>
    %z = arith.constant 0 : i64
    %zt = tensor.splat %z : tensor<i64>
    %d = "quant_ext.dequantize_per_tensor"(%x, %st, %zt) <{quant_min = -128 : i64, quant_max = 127 : i64}> : (tensor<4xi8>, tensor<f32>, tensor<i64>) -> tensor<4xf32>
    func.return %d : tensor<4xf32>
  }
}"""
    dequantize = next(op for op in mq.parse(text).walk() if mq.op_name(op) == "quant_ext.dequantize_per_tensor")
    source = GN._scale_source(dequantize)
    assert (source.value, source.zero_point) == (0.25, 0)


def test_the_group_reference_and_the_capsule_golden_engine_agree_to_the_bit() -> None:
    # Two implementations written apart: the capsule golden's pure-Python readout (what every
    # grade compares a backend with) and this module's numpy reference (what a closed group says
    # it computes). A capsule generated from a group is only a fair demand if they are one
    # function, ties and large accumulators included, where a float32 product is inexact.
    import numpy as np

    from merlin.runtime.tensor import Tensor

    rng = np.random.default_rng(11)
    columns = 16
    accumulator = np.concatenate(
        [
            rng.integers(-60000, 60000, size=20 * columns),
            rng.integers(-(2**30), 2**30, size=4 * columns),  # beyond 2**24: the f32 conversion rounds
            np.array([0, 1, -1, 3, -3, 5, -5, 144, -144, 433, -433, 2**24 + 1, -(2**24) - 1, 0, 0, 0]),
        ]
    ).reshape(-1, columns)
    bias = rng.integers(-40000, 40000, size=columns)
    bias[:8] = 0  # leave the hand-picked accumulators of the last row as they are
    rows = accumulator.shape[0]

    # A real layer's multiplier, and one half: the multiplier under which odd sums are exact ties.
    for multiplier in (0.0034612307785188024, 0.5):
        numerics = GN.GroupNumerics(
            input=GN.ScaleSource(value=1.0),
            weight=GN.ScaleSource(value=multiplier),
            output=GN.ScaleSource(value=1.0),
            multiplier=multiplier,
            bias_arg_index=0,
            bias_divisor=1.0,
            activation="relu",
            clamp=(-128, 127),
            granularity="tensor",
        )
        mine = GN.reference_integer(numerics, accumulator, bias)
        golden = (
            Tensor((rows, columns), [int(v) for v in accumulator.flatten()], "i32")
            .add_bias(Tensor((columns,), [int(v) for v in bias], "i32"))
            .requant_acc_scale(multiplier)
            .relu()
            .to_i8()
        )
        assert [int(v) for v in mine.flatten()] == list(golden.data)
        # Both the activation and the saturation did something, or the agreement proves little.
        assert 0 in golden.data and 127 in golden.data

    # The mutation, on the ties: rounding half away from zero is a different function there.
    scaled = np.float32(0.5) * (accumulator + bias).astype(np.float32)
    away = np.clip(np.sign(scaled) * np.floor(np.abs(scaled) + np.float32(0.5)), -128, 127)
    assert not np.array_equal(np.maximum(away, 0).astype(np.int64), mine)
