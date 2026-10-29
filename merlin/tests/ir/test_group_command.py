"""A closed group is restated as the device program it asks for, or refused with the reason."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_demand_keeps_group_count_and_records_every_batch_multiplicity(monkeypatch) -> None:
    from merlin.xdsl_dialects.lowering import group_command as command

    groups = [SimpleNamespace(index=i, placement="device", root=object()) for i in (0, 1, 2)]
    shapes = {0: (2,), 1: (5,), 2: ()}
    monkeypatch.setattr(
        command,
        "program",
        lambda group, **_kwargs: command.GroupProgram(
            entry={"op": "matmul", "M": 3, "K": 4, "N": 6},
            stored_operand=1,
            transposed=False,
            batch_shape=shapes[group.index],
        ),
    )
    (row,) = CG.demand(groups)["entries"]
    assert (row["count"], row["slice_instances"]) == (3, 8)
    assert row["group_batch_shapes"] == [
        {"group": 0, "batch_shape": [2], "slices": 2},
        {"group": 1, "batch_shape": [5], "slices": 5},
        {"group": 2, "batch_shape": [], "slices": 1},
    ]
    assert (row["M"], row["K"], row["N"]) == (3, 4, 6)


def test_a_window_mean_is_oriented_as_the_program_holds_it_and_nothing_else_is() -> None:
    """``x[features, window] @ ones[window, 1]`` is asked for as ``ones[1, window] @ x[window, features]``
    with the activation stationary; a stored weight or a many-column contraction is left as stated."""
    from merlin.xdsl_dialects.lowering import group_command as GC

    mean = {"op": "matmul", "M": 5, "K": 3, "N": 1}
    oriented = GC.device_orientation(mean, {"stored_operand": None})
    assert (oriented["M"], oriented["K"], oriented["N"]) == (1, 3, 5) and GC.stationary_is_activation(oriented)
    assert GC.device_orientation(mean, {"stored_operand": 1}) == mean
    assert GC.device_orientation({**mean, "N": 4}, {"stored_operand": None}) == {**mean, "N": 4}
    assert GC.device_orientation(mean, None) == mean


# ------------------------------------------------------------------ a convolution the capture gathered on the host


def _host_gathered(text: str) -> CG.Group:
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=_Oracle())
    (group,) = [g for g in groups if g.placement != CG.HOST and g.root is not None]
    return group


@pytest.mark.parametrize(
    ("channels", "side", "features", "kernel", "stride", "pad"),
    [(3, 9, 8, 3, 1, 1), (4, 9, 8, 3, 2, 1), (5, 6, 7, 1, 1, 0), (5, 7, 6, 1, 2, 0)],
    ids=["3x3_s1_pad1", "3x3_s2_pad1", "1x1_s1", "1x1_s2"],
)
def test_a_host_gathered_convolution_is_stated_as_the_convolution_the_route_passes(
    channels, side, features, kernel, stride, pad
) -> None:
    """The patch matrix a capture builds on the host (strided slices of the padded NCHW image, NHWC per
    tap, concatenated) is the host's lowering, not the layer: the statement names the convolution over
    the image before the gather -- window, stride, padding, the [tap_h, tap_w, channel] weight rows and
    the fused readout -- exactly as the whole-model route reads it back."""
    from test_device_shim_logical_abi import _conv_layer_module

    text, (ho, wo) = _conv_layer_module(channels, side, side, features, kernel, stride, pad)
    stated = GC.program(_host_gathered(text), weight_args={1, 2})
    entry = stated.entry
    assert entry["op"] == "conv2d" and "M" not in entry and "K" not in entry
    assert (entry["ci"], entry["Himg"], entry["Wimg"], entry["N"]) == (channels, side, side, features)
    assert (entry["kh"], entry["kw"], entry["stride"], entry["padding"]) == (kernel, kernel, [stride] * 2, [pad] * 4)
    assert stated.column_order == GC.HOST_GATHER_COLUMN_ORDER == ("tap_h", "tap_w", "channel")
    assert GC.device_output_shape(entry) == [ho * wo, features]
    assert entry["epilogue"] == ["bias_add", "acc_scale", "relu"]


def test_a_host_gather_the_statement_cannot_read_back_is_refused_not_stated_as_a_contraction() -> None:
    from test_device_shim_logical_abi import _conv_layer_module

    text, _ = _conv_layer_module(3, 9, 9, 8, 3, 1, 1)
    line = next(row for row in text.splitlines() if '"tensor.concat"' in row)
    head, rest = line.split('"tensor.concat"(', 1)
    pieces, tail = rest.split(")", 1)
    swapped = ", ".join(reversed(pieces.split(", ")))  # taps concatenated in reverse window order
    with pytest.raises(CG.NoCapsuleForm, match="row-major"):
        GC.program(_host_gathered(text.replace(line, f'{head}"tensor.concat"({swapped}){tail}')), weight_args={1, 2})
