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
    assert summed["operand_sum"]["load_multipliers"] == [1.0, 0.25 / 1.5]
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
