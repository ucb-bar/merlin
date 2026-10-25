"""A closed group's bias is folded into the accumulator's domain once, offline, from the weights."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from fake_quant_layer import Oracle, module  # noqa: E402

from merlin.common import mlir_query as mq
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import group_prepack as GP


def _weights(tmp_path: Path, bias: np.ndarray) -> tuple[Path, Path]:
    payload = bias.astype("<f4").tobytes()
    header = json.dumps(
        {"layer.bias": {"dtype": "F32", "shape": [int(bias.size)], "data_offsets": [0, len(payload)]}}
    ).encode("utf-8")
    weights = tmp_path / "weights.safetensors"
    weights.write_bytes(struct.pack("<Q", len(header)) + header + payload)
    manifest = tmp_path / "manifest.json"
    # %b is the fifth argument of the fixture's @forward; the first is the model input.
    manifest.write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}, "4": {"kind": "param", "weight": "layer.bias"}}),
        encoding="utf-8",
    )
    return manifest, weights


def _groups(**kwargs):
    return CG.form_groups(mq.parse(module(**kwargs)), "synthetic", oracle=Oracle())


def test_the_bias_is_folded_by_the_product_of_the_operand_scales(tmp_path: Path) -> None:
    bias = np.array([0.25, -0.5, 1.0, 0.124, -0.126] + [0.0] * 11)
    manifest, weights = _weights(tmp_path, bias)
    result = GP.prepack(_groups(weight_dequantize="per_tensor"), manifest, weights)
    (row,) = result["record"]["groups"]
    folded = result["arrays"][row["bias"]["array"]]
    # s_x * s_w = 0.25, so b_q = roundeven(b / 0.25); 0.124/0.25 and -0.126/0.25 round to 0 and -1.
    assert folded[:5].tolist() == [1, -2, 4, 0, -1] and folded.dtype == np.int32
    assert row["multiplier_f32_bits"] == struct.unpack("<I", struct.pack("<f", 0.5))[0]
    assert (row["bias"]["stored_tensor"], row["bias"]["elements"]) == ("layer.bias", 16)
    written = GP.write(result, tmp_path / "out")
    assert json.loads(written.read_text())["folded_bias_elements"] == 16
    assert np.load(tmp_path / "out/prepack.npz")[row["bias"]["array"]].tolist() == folded.tolist()


def test_a_bias_that_overflows_the_accumulator_is_refused_not_wrapped(tmp_path: Path) -> None:
    manifest, weights = _weights(tmp_path, np.full(16, 1.0e9))
    result = GP.prepack(_groups(weight_dequantize="per_tensor"), manifest, weights)
    assert result["record"]["groups"] == [] and not result["arrays"]
    assert "outside the 32-bit accumulator" in result["record"]["skipped"][0]["reason"]


def test_a_group_that_is_not_closed_needs_nothing_folded(tmp_path: Path) -> None:
    manifest, weights = _weights(tmp_path, np.zeros(16))
    result = GP.prepack(_groups(weight_dequantize="per_channel"), manifest, weights)
    assert result["record"]["groups"] == [] and result["record"]["skipped"] == []


def test_a_sum_and_a_mean_are_stated_as_the_register_values_their_programs_issue(tmp_path: Path) -> None:
    from fake_quant_layer import residual_then_mean_module

    from merlin.targetgen import readout_facet as RF

    facet = RF.ReadoutFacet(
        target="synthetic",
        unit="unit0",
        accumulator_kind="addressable",
        scale_granularities=("tensor",),
        operand_sum={"operands": 2, "operand_dtype": "i8", "operand_rounding": "half_even", "operand_saturates": True},
    )
    text = residual_then_mean_module().replace("dense<0.5> : tensor<f32>", "dense<1.5> : tensor<f32>", 1)
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=Oracle(readout=RF.TargetReadout((facet,))))
    manifest, weights = _weights(tmp_path, np.zeros(16))
    record = GP.prepack(groups, manifest, weights)["record"]
    assert record["skipped"] == []
    summed, pooled = record["groups"]
    # 1.5 cannot go through a saturating load: both loads are divided by it and the readout carries it.
    assert summed["operand_sum"]["load_multipliers"] == [1.0, float(np.float32(1 / 6))]
    assert summed["operand_sum"]["readout_multiplier_f32_bits"] == struct.unpack("<I", struct.pack("<f", 1.5))[0]
    assert (summed["operand_sum"]["bound_lsb"], summed["operand_sum"]["activation"]) == (2, "relu")
    assert pooled["window_mean"]["multiplier"] == 1.0 / (16.0 * 0.5) and pooled["window_mean"]["window"] == 16


def test_the_stored_weight_is_laid_out_the_way_the_device_program_holds_it() -> None:
    # The capture multiplies W[Cout, K] by host-gathered patches in [channel, tap_h, tap_w] order. The
    # device program holds W[K, Cout] in the command's [tap_h, tap_w, channel] packing and forms the
    # patches itself. The layout is right when the ABI's OWN convolution, fed the laid-out weight,
    # reproduces a convolution computed directly from the stored tensor.
    import numpy as np
    from fake_quant_layer import Oracle
    from im2col_conv_layer import module as conv_module

    from merlin.common import mlir_query as mq
    from merlin.runtime.commandbuffer import conv_im2col
    from merlin.runtime.tensor import Tensor
    from merlin.xdsl_dialects.lowering import group_command

    ci, co, image, taps, stride, pad = 3, 4, 6, 3, 2, 1
    text = conv_module(channels=ci, out_channels=co, image=image, taps=taps, stride=stride, pad=pad)
    groups = CG.form_groups(mq.parse(text), "synthetic", oracle=Oracle())
    (group,) = [g for g in groups if g.placement != CG.HOST]
    stated = group_command.program(group, weight_args={1, 2})

    rng = np.random.default_rng(7)
    stored = rng.integers(-8, 9, size=(co, ci, taps, taps))
    x = rng.integers(-8, 9, size=(1, ci, image, image))
    device = GP.device_weight(group, stated, stored)
    assert device.shape == (taps * taps * ci, co)

    padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = (padded.shape[2] - taps) // stride + 1
    direct = np.zeros((out * out, co), dtype=np.int64)
    for oh in range(out):
        for ow in range(out):
            window = padded[0, :, oh * stride : oh * stride + taps, ow * stride : ow * stride + taps]
            direct[oh * out + ow] = np.tensordot(stored, window, axes=([1, 2, 3], [0, 1, 2]))

    nhwc = np.transpose(x, (0, 2, 3, 1))
    ifm = Tensor(tuple(nhwc.shape), [int(v) for v in nhwc.flatten()], "i8")
    cols = conv_im2col(
        ifm, kh=taps, kw=taps, ci=ci, stride=(stride, stride), padding=(pad, pad, pad, pad), dilation=(1, 1)
    )
    columns = np.array(cols.data, dtype=np.int64).reshape(cols.shape)
    assert np.array_equal(columns @ device, direct)
    # The mutation: the capture's own column order, fed to the device convolution, is wrong.
    assert not np.array_equal(columns @ stored.reshape(co, -1).T, direct)


# --------------------------------------------- the shape the DEVICE writes, derived once and shared


def test_the_device_output_shape_is_positions_by_features_not_the_captures_tensor() -> None:
    """A device program does not write the capture's tensor. A contraction commits `[M, N]`; a
    convolution commits one row per output POSITION -- so `[1, 64, 56, 56]` in the capture is
    `[3136, 64]` on the device, and those are the same elements only under a stated reshape."""
    from merlin.xdsl_dialects.lowering import group_command as GC

    assert GC.device_output_shape({"op": "matmul", "M": 3136, "N": 64}) == [3136, 64]
    assert GC.device_output_shape({"op": "residual_add", "M": 14336, "N": 56}) == [14336, 56]
    # 224 with a 7-tap stride-2 window and 3 of padding -> 112, then a 3x3 stride-2 pool with 1 -> 56.
    conv = {
        "op": "conv2d",
        "N": 64,
        "ci": 3,
        "Himg": 224,
        "Wimg": 224,
        "kh": 7,
        "kw": 7,
        "stride": [2, 2],
        "padding": [3, 3, 3, 3],
        "epilogue": ["bias_add", "acc_scale", "relu", "maxpool"],
        "pool_size": [3, 3],
        "pool_stride": [2, 2],
        "pool_padding": [1, 1, 1, 1],
    }
    assert GC.device_output_shape(conv) == [3136, 64]
    without_pool = {**conv, "epilogue": ["bias_add", "acc_scale", "relu"]}
    assert GC.device_output_shape(without_pool) == [112 * 112, 64]


def test_an_entry_that_states_no_output_is_refused_rather_than_sized_by_invention() -> None:
    """Sizing a buffer by arithmetic nobody performed is a wrong allocation that nothing would
    attribute back to here."""
    import pytest

    from merlin.xdsl_dialects.lowering import group_command as GC

    with pytest.raises(GC.NoDeviceShape, match="no M/N"):
        GC.device_output_shape({"op": "matmul", "N": 64})
    with pytest.raises(GC.NoDeviceShape, match="states no window"):
        GC.device_output_shape(
            {"op": "conv2d", "N": 64, "Himg": 8, "Wimg": 8, "kh": 3, "kw": 3, "stride": [1, 1], "epilogue": ["maxpool"]}
        )


def test_a_group_that_leaves_as_the_accumulator_still_states_its_device_weight():
    """The guard skipped any group without a QUANTIZE stage, conflating "has a quantized readout"
    with "has a stored operand the unit reads in its own layout". The second is a property of the
    TARGET. A model's final classifier commits i32 -- no multiplier, no clamp, no folded bias --
    and its weight is permuted like any other; consumers asking for it were told no row existed and
    declared the capture's [out, in] where the contraction reads [in, out].
    """
    import inspect

    from merlin.xdsl_dialects.lowering import group_prepack as GP

    source = inspect.getsource(GP.prepack)
    assert "if group.placement == CG.HOST:" in source
    assert "if group.placement == CG.HOST or CG.QUANTIZE not in group.stages:" not in source
    assert '"leaves_as": "accumulator"' in source, "such a row must say what it is"
