"""Lowering a whole Voyager program layer by layer (baselines.voyager_schedule.lower_model).

The fixtures are torchvision residual blocks (BasicBlocks and a Bottleneck, each with a 1x1 stride-2
downsample) compiled by the pinned Voyager compiler. Two oracles:

* exact: random int8 parameters through the lowered layers -- accelerator schedules on the numpy
  executor, the bridge's host ops on their reference semantics -- against a direct integer
  re-implementation of the block with the readout arithmetic the target's store path defines;
* independent: Voyager's OWN int8 parameters and input through the lowered program against the output
  Voyager's bufferized graph computed in bf16. The two differ only by the readout's rounding (C1/C5),
  so they must agree closely -- and stop agreeing when the residual or the bias is dropped.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from merlin.baselines.voyager_ir import UnsupportedConstruct, load_model, replay
from merlin.baselines.voyager_schedule import (
    DEQUANTIZE_ACC,
    REQUANTIZE,
    AccMvin,
    Geometry,
    HostOp,
    Mvout,
    Schedule,
    execute_model,
    lower_model,
)
from merlin.common.paths import merlin_dir

FIXTURES = merlin_dir() / "tests" / "data" / "voyager_ir"
# The geometry the fixtures were compiled against (their manifest): 16x16 array, 256 KiB scratchpad of
# 16-byte rows, and an accumulator of 1024 rows.
GEOMETRY = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=1024)
BASIC = "resblock_basic_14x14x32x64"
SPLIT_K = "resblock_splitk_8x8x512x64"
BOTTLENECK = "resblock_bottleneck_14x14x1024x256"


def _load(name: str):
    root = FIXTURES / name
    trace = replay(load_model(root / "model.json"))
    scales = json.loads((root / "scales.json").read_text())
    workload = json.loads((root / "manifest.json").read_text())["workload"]
    return trace, scales, workload


def _random_tensors(trace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    tensors = {}
    for box in list(trace.inputs) + list(trace.parameters):
        lo, hi = (-(2**16), 2**16) if box.dtype == "int32" else (-128, 128)
        tensors[box.node] = rng.integers(lo, hi, size=box.shape, dtype=np.int64)
    return tensors


def _conv(x: np.ndarray, w: np.ndarray, stride: int, pad: int) -> np.ndarray:
    """Direct NHWC x HWIO convolution in int64."""
    k = w.shape[0]
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oh = (padded.shape[1] - k) // stride + 1
    ow = (padded.shape[2] - k) // stride + 1
    out = np.zeros((x.shape[0], oh, ow, w.shape[3]), dtype=np.int64)
    for fy in range(k):
        for fx in range(k):
            window = padded[:, fy : fy + stride * oh : stride, fx : fx + stride * ow : stride]
            out += np.einsum("nhwc,co->nhwo", window, w[fy, fx])
    return out


def _scale(values: np.ndarray, scale: float, lo: int, hi: int) -> np.ndarray:
    """Integer -> fp32, fp32 multiply, round to nearest even, saturate: the store path's scale unit."""
    y = np.rint(values.astype(np.float32) * np.float32(scale))
    return np.clip(y, lo, hi).astype(np.int64)


def _bf16(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def _reference(tensors: dict, layers: list, workload: dict) -> np.ndarray:
    """The BasicBlock, recomputed directly: relu(conv2(relu(conv1 x)) + downsample x)."""
    readout = {layer.name: layer.readout for layer in layers if layer.kind == "conv"}
    host = {layer.program.target: layer.program for layer in layers if layer.kind == "host"}
    s, t = workload["stride"], tensors
    x = t["x_preprocess"]
    r1, r2 = readout["conv1"], readout["conv2"]
    y1 = _conv(x, t["conv1_weight"], s, 1) + t["conv1_bias"]
    y1 = _scale(np.maximum(y1, 0) if r1["relu"] else y1, r1["scale"], -128, 127)
    y2 = _conv(y1, t["conv2_weight"], 1, 1) + t["conv2_bias"]
    y2 = _scale(np.maximum(y2, 0) if r2["relu"] else y2, r2["scale"], -128, 127)
    acc = _conv(x, t["downsample_0_weight"], s, 0) + t["downsample_0_bias"]
    acc = acc + _scale(y2, host[REQUANTIZE].attrs["scale"], -(2**31), 2**31 - 1)
    dq = host[DEQUANTIZE_ACC].attrs
    out = acc.astype(np.float32) * np.float32(dq["scale"])
    out = _bf16(np.maximum(out, np.float32(0)) if dq["relu"] else out)
    return np.transpose(out, (0, 3, 1, 2))


def _bottleneck_reference(tensors: dict, layers: list, workload: dict) -> np.ndarray:
    """The Bottleneck, recomputed directly: relu(conv3(conv2(conv1 x)) + downsample x), stride on
    the 3x3 conv (torchvision)."""
    readout = {layer.name: layer.readout for layer in layers if layer.kind == "conv"}
    host = {layer.program.target: layer.program for layer in layers if layer.kind == "host"}
    s, t = workload["stride"], tensors

    def quantized(values: np.ndarray, r: dict) -> np.ndarray:
        return _scale(np.maximum(values, 0) if r["relu"] else values, r["scale"], -128, 127)

    y1 = quantized(_conv(t["x_preprocess"], t["conv1_weight"], 1, 0) + t["conv1_bias"], readout["conv1"])
    y2 = quantized(_conv(y1, t["conv2_weight"], s, 1) + t["conv2_bias"], readout["conv2"])
    y3 = quantized(_conv(y2, t["conv3_weight"], 1, 0) + t["conv3_bias"], readout["conv3"])
    acc = _conv(t["x_preprocess"], t["downsample_0_weight"], s, 0) + t["downsample_0_bias"]
    acc = acc + _scale(y3, host[REQUANTIZE].attrs["scale"], -(2**31), 2**31 - 1)
    dq = host[DEQUANTIZE_ACC].attrs
    out = acc.astype(np.float32) * np.float32(dq["scale"])
    out = _bf16(np.maximum(out, np.float32(0)) if dq["relu"] else out)
    return np.transpose(out, (0, 3, 1, 2))


def test_a_standalone_residual_add_is_renamed_onto_its_k_split_exactly() -> None:
    # The downsample is a K split whose last part still writes the partial; Voyager then adds the
    # residual in a separate op. On the accumulator that op is a scaled accumulate before the store.
    trace, scales, workload = _load(BOTTLENECK)
    layers = lower_model(trace, GEOMETRY, scales)
    down = next(layer for layer in layers if layer.name == "downsample_0" and layer.kind == "conv")
    assert any("standalone residual add" in note for note in down.program.notes)
    assert any("C2" in note for note in down.program.notes)
    tensors = _random_tensors(trace)
    expected = _bottleneck_reference(dict(tensors), layers, workload)
    assert np.array_equal(execute_model(layers, dict(tensors), trace)["permute_default_1"], expected)
    broken = _without(layers, lambda op: not (isinstance(op, AccMvin) and op.role == "residual"))
    assert not np.array_equal(execute_model(broken, dict(tensors), trace)["permute_default_1"], expected)


def _without(layers: list, keep) -> list:
    return [
        replace(layer, program=replace(layer.program, ops=[op for op in layer.program.ops if keep(op)]))
        if isinstance(layer.program, Schedule)
        else layer
        for layer in layers
    ]


def test_segmentation_is_one_layer_per_voyager_loop() -> None:
    trace, _, _ = _load(BASIC)
    assert len(trace.layers) == 3
    ends = [first for _, first, _ in trace.layers] + [len(trace.events)]
    assert ends == sorted(ends) and ends[0] == 0
    assert all(end > first for _, first, end in trace.layers)


def test_the_block_lowers_to_three_schedules_and_the_bridge_host_ops() -> None:
    trace, scales, _ = _load(BASIC)
    layers = lower_model(trace, GEOMETRY, scales)
    shape = [(layer.name, layer.kind if layer.kind != "host" else layer.program.target) for layer in layers]
    assert shape == [
        ("conv1", "conv"),
        ("conv2", "conv"),
        ("downsample_0", REQUANTIZE),
        ("downsample_0", "conv"),
        ("downsample_0", DEQUANTIZE_ACC),
        ("downsample_0", "aten::permute"),
    ]
    # conv1/conv2 quantize on the store path; the residual layer's output is unquantized, so it is
    # read out raw and dequantized (with the relu) on the host.
    assert [layers[i].readout["out_dtype"] for i in (0, 1, 3)] == ["int8", "int8", "int32"]
    assert layers[0].readout["relu"] and not layers[1].readout["relu"]
    residual = [op for op in layers[3].program.ops if isinstance(op, AccMvin) and op.role == "residual"]
    assert residual and all(op.accumulate and op.scale == 1.0 for op in residual)


@pytest.mark.parametrize("name", [BASIC, SPLIT_K])
def test_the_lowered_block_is_exact_and_the_oracle_can_fail(name: str) -> None:
    trace, scales, workload = _load(name)
    layers = lower_model(trace, GEOMETRY, scales)
    tensors = _random_tensors(trace)
    expected = _reference(dict(tensors), layers, workload)
    got = execute_model(layers, dict(tensors), trace)["permute_default_1"]
    assert np.array_equal(got, expected)
    for broken in (
        _without(layers, lambda op: not (isinstance(op, AccMvin) and op.role == "residual")),
        _without(layers, lambda op: not (isinstance(op, AccMvin) and op.role == "bias")),
        [
            replace(
                layer,
                program=replace(
                    layer.program,
                    ops=[
                        replace(op, relu=not op.relu) if isinstance(op, Mvout) and op.out_dtype == "int8" else op
                        for op in layer.program.ops
                    ],
                ),
            )
            if isinstance(layer.program, Schedule)
            else layer
            for layer in layers
        ],
    ):
        assert not np.array_equal(execute_model(broken, dict(tensors), trace)["permute_default_1"], expected)


def test_the_split_k_block_accumulates_its_k_parts_in_the_accumulator() -> None:
    trace, scales, _ = _load(SPLIT_K)
    layers = lower_model(trace, GEOMETRY, scales)
    conv1 = layers[0].program
    assert any("C2" in note for note in conv1.notes)
    assert layers[0].readout == {"scale": layers[0].readout["scale"], "relu": True, "out_dtype": "int8"}


def test_a_target_whose_accumulator_loads_scale_needs_no_host_requantization() -> None:
    trace, scales, workload = _load(BASIC)
    scaled = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=1024, scaled_acc_loads=True)
    layers = lower_model(trace, scaled, scales)
    assert REQUANTIZE not in [layer.program.target for layer in layers if layer.kind == "host"]
    residual = [
        op
        for layer in layers
        if isinstance(layer.program, Schedule)
        for op in layer.program.ops
        if isinstance(op, AccMvin) and op.role == "residual"
    ]
    assert residual and {op.scale for op in residual} != {1.0}
    host_path = lower_model(trace, GEOMETRY, scales)
    tensors = _random_tensors(trace)
    expected = execute_model(host_path, dict(tensors), trace)["permute_default_1"]
    assert np.array_equal(execute_model(layers, dict(tensors), trace)["permute_default_1"], expected)


def test_voyagers_own_parameters_reproduce_voyagers_own_output() -> None:
    trace, scales, _ = _load(BASIC)
    data = dict(np.load(FIXTURES / BASIC / "data.npz"))
    voyager = data.pop("voyager_output")
    layers = lower_model(trace, GEOMETRY, scales)

    def agreement(candidate: list) -> tuple[float, float]:
        tensors = {k: v.astype(np.int64) for k, v in data.items()}
        got = execute_model(candidate, tensors, trace)["permute_default_1"]
        cosine = float((got * voyager).sum() / np.sqrt((got * got).sum() * (voyager * voyager).sum()))
        return cosine, float(np.abs(got - voyager).max() / np.abs(voyager).max())

    # Measured 0.99997 and 1.0% on this fixture; the thresholds leave room for rounding only.
    cosine, worst = agreement(layers)
    assert cosine >= 0.9999 and worst <= 0.02
    for broken in (
        _without(layers, lambda op: not (isinstance(op, AccMvin) and op.role == "residual")),
        _without(layers, lambda op: not (isinstance(op, AccMvin) and op.role == "bias")),
    ):
        cosine, worst = agreement(broken)
        assert cosine < 0.99 and worst > 0.1
    assert all(isinstance(layer.program, (Schedule, HostOp)) for layer in layers)


def test_an_output_tile_larger_than_the_accumulator_runs_in_passes_exactly() -> None:
    # Voyager keeps its output tile in the scratchpad, so a tile can exceed the accumulator (16 of
    # ResNet-50's layers do on a 1024-row one). Split at the outermost output loops, each pass
    # accumulates and is stored on its own (C7); a reduction loop outermost refuses.
    trace, scales, _ = _load(BASIC)
    tensors = _random_tensors(trace)
    whole = execute_model(lower_model(trace, GEOMETRY, scales), dict(tensors), trace)
    small = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=64)
    layers = lower_model(trace, small, scales)
    in_passes = [
        i
        for i, layer in enumerate(layers)
        if isinstance(layer.program, Schedule) and any("passes" in note for note in layer.program.notes)
    ]
    assert in_passes
    got = execute_model(layers, dict(tensors), trace)
    assert np.array_equal(got["permute_default_1"], whole["permute_default_1"])
    i = in_passes[0]
    ops = layers[i].program.ops
    last = max(k for k, op in enumerate(ops) if isinstance(op, Mvout))
    broken = list(layers)
    broken[i] = replace(layers[i], program=replace(layers[i].program, ops=ops[:last] + ops[last + 1 :]))
    out = layers[i].program.dram_nodes["out"]
    assert not np.array_equal(execute_model(broken, dict(tensors), trace)[out], whole[out])
    tiny = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=32)
    with pytest.raises(UnsupportedConstruct, match="reduction"):
        lower_model(trace, tiny, scales)
