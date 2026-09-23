"""A closed group is restated as the device program it asks for, or refused with the reason."""

from __future__ import annotations

import pytest
from fake_quant_layer import Oracle as _Oracle
from fake_quant_layer import module as _linear
from im2col_conv_layer import module as _conv

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_command as GC

#: Which model arguments the fixtures' weights manifests would call stored: the weight and the bias.
_LINEAR_WEIGHTS, _CONV_WEIGHTS = {1, 2, 3, 4}, {1, 2}


def _closed(text: str) -> CG.Group:
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=_Oracle())
    (group,) = [g for g in groups if g.placement != CG.HOST]
    return group


def test_a_linear_layer_is_already_in_device_form() -> None:
    stated = GC.program(_closed(_linear(weight_dequantize="per_tensor")), weight_args=_LINEAR_WEIGHTS)
    assert (stated.stored_operand, stated.transposed) == (1, False)
    entry = stated.entry
    assert (entry["op"], entry["M"], entry["K"], entry["N"]) == ("matmul", 4, 8, 16)
    # The readout's order, whatever order the capture applied them in; the multiplier is the group's.
    assert entry["epilogue"] == ["bias_add", "acc_scale", "relu"]
    assert entry["acc_scale"] == pytest.approx(0.5 * 0.5 / 0.5)


def test_a_gathered_convolution_is_restated_as_the_units_own_convolution() -> None:
    stated = GC.program(
        _closed(_conv(channels=2, out_channels=4, image=4, taps=3, stride=1, pad=1)), weight_args=_CONV_WEIGHTS
    )
    assert stated.transposed and stated.stored_operand == 0 and stated.column_order == GC.CAPTURE_COLUMN_ORDER
    entry = stated.entry
    assert entry["op"] == "conv2d"
    assert (entry["ci"], entry["N"], entry["Himg"], entry["Wimg"]) == (2, 4, 4, 4)
    assert (entry["kh"], entry["kw"], entry["stride"], entry["padding"]) == (3, 3, [1, 1], [1, 1, 1, 1])


def test_stride_and_padding_are_read_from_the_gather_and_the_slice() -> None:
    entry = GC.program(_closed(_conv(image=8, taps=3, stride=2, pad=1)), weight_args=_CONV_WEIGHTS).entry
    assert (entry["stride"], entry["padding"], entry["Himg"]) == ([2, 2], [1, 1, 1, 1], 8)


def test_a_one_tap_unit_stride_window_is_a_contraction_over_positions() -> None:
    entry = GC.program(
        _closed(_conv(channels=2, out_channels=4, image=4, taps=1, stride=1, pad=0)), weight_args=_CONV_WEIGHTS
    ).entry
    # Device form: positions are the rows, the stored tensor's outputs the columns.
    assert (entry["op"], entry["M"], entry["K"], entry["N"]) == ("matmul", 16, 2, 4)


def test_a_bias_along_the_activations_axis_is_refused() -> None:
    # The mutation: the same layer with its bias along a spatial axis. A per-output readout
    # cannot apply it, and restating it as one would silently change the arithmetic.
    with pytest.raises(CG.NoCapsuleForm, match="bias"):
        GC.program(_closed(_conv(image=4, taps=3, pad=1, bias_axis=2)), weight_args=_CONV_WEIGHTS)


def test_a_first_layer_needs_the_manifest_to_say_which_operand_is_stored() -> None:
    group = _closed(_conv())
    # Both operands are model arguments here. With nothing saying which is stored it is refused...
    with pytest.raises(CG.NoCapsuleForm, match="which one is stored"):
        GC.program(group)
    # ...and the manifest's answer is taken, not a guess about shapes.
    assert GC.program(group, weight_args={1}).stored_arg == 1


def test_an_operand_sum_is_stated_as_a_residual_add_with_its_computed_bound() -> None:
    from fake_quant_layer import Oracle, residual_module

    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",),
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    oracle = Oracle(readout=RF.TargetReadout((facet,)))
    (group,) = CG.form_groups(mq.parse(residual_module(lhs_scale=1.5, rhs_scale=0.5)), "synthetic", oracle=oracle)
    stated = GC.program(group)
    assert stated.stored_operand is None and stated.stored_arg is None
    assert {k: stated.entry[k] for k in ("op", "M", "N", "lhs_scale", "rhs_scale", "bound_lsb", "epilogue")} == {
        "op": "residual_add",
        "M": 4,
        "N": 16,
        "lhs_scale": 1.5,
        "rhs_scale": 0.5,
        "bound_lsb": 2,
        "epilogue": ["relu"],
    }
    assert "readout scales by it" in stated.notes[0]
    # Multipliers are numbers one program runs with: two sums that differ only there are one demand.
    other = CG.form_groups(mq.parse(residual_module(lhs_scale=1.25, rhs_scale=0.5)), "synthetic", oracle=oracle)
    assert len(CG.demand([group, *other])["entries"]) == 1
