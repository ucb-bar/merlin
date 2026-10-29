"""Synthetic source semantics and actual compact host-control emission."""

import json
import operator
from dataclasses import asdict, replace

import numpy as np
import pytest
from xdsl.dialects import llvm
from xdsl.dialects.builtin import ModuleOp, StringAttr, UnregisteredOp

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.llvm_program import FnBuilderControlAdapter
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.spatial_producer import (
    PackedSpatialProducerPlan,
    SpatialProducerPlan,
    emit_packed_spatial_producer,
    read_spatial_plan_record,
)
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.llvmlower.host_builder import FnBuilder
from merlin.llvmlower.requantization import quantized


class Integers:
    def op(self, name, a, b):
        return {"add": operator.add, "mul": operator.mul, "udiv": operator.floordiv}[name](a, b)

    def loop(self, start, stop, step, body):
        for i in range(start, stop, step):
            body(i)


def plan():
    window = Window2D((2, 6, 7, 3), (3, 3), padding=(1, 1, 1, 1))
    producer = CountedConvolutionPlan(window, 5, 4, 4, 4, 3, 32, 30, 0, 8, 3, True)
    readouts = tuple(
        SpatialReadoutChain(RoundedReadout((q,), True), RoundedReadout((0.75,)), None) for q in (0.25, 0.5)
    )
    source = SpatialProducerPlan(
        producer,
        SpatialMaxReadout((6, 7), (3, 3), (2, 2), (1, 1, 1, 1), -128),
        readouts,
        3,
        readout_mode="staged_readouts",
        intermediate=AddressWindow("old_channel_stripe", 0, 84),
    )
    return PackedSpatialProducerPlan(source, 8, AddressWindow("caller_owned_packed_stripe", 0, 384), 64)


def test_packed_full_k_and_all_channel_q1_equal_original_source_with_tails_and_bias_wrap():
    selected = plan()
    p, s = selected.source.producer, selected.source.spatial
    data = (np.arange(np.prod(p.window.input_shape)) % 17 - 8).astype(np.int8).reshape(p.window.input_shape)
    weight = (np.arange(selected.reduction * 5) % 13 - 6).astype(np.int8).reshape(selected.reduction, 5)
    bias = np.array([-(1 << 31), (1 << 31) - 1, -19, 0, 31], dtype=np.int32)
    b, h, w, _ = data.shape
    raw = np.zeros((b, h, w, 5), dtype=np.int32)
    q1 = np.zeros_like(raw, dtype=np.int8)
    for image in range(b):
        for y in range(h):
            for x in range(w):
                patch = np.zeros((3, 3, 3), dtype=np.int8)
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y - 1 + ky, x - 1 + kx
                        if 0 <= iy < h and 0 <= ix < w:
                            patch[ky, kx] = data[image, iy, ix]
                raw[image, y, x] = patch.ravel().astype(np.int32) @ weight.astype(np.int32) + bias
                for n in range(5):
                    q = selected.source.readouts[n // 4].producer
                    q1[image, y, x, n] = quantized(int(raw[image, y, x, n]), q.scales, q.relu)
    oh, ow = s.output_hw
    expected = np.full((b * oh * ow, 5), -128, dtype=np.int8)
    for image in range(b):
        for y in range(oh):
            for x in range(ow):
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y * 2 - 1 + ky, x * 2 - 1 + kx
                        if 0 <= iy < h and 0 <= ix < w:
                            for n in range(5):
                                expected[(image * oh + y) * ow + x, n] = max(
                                    expected[(image * oh + y) * ow + x, n],
                                    quantized(int(q1[image, iy, ix, n]), (0.75,)),
                                )

    scratch = np.zeros(selected.workspace.size, dtype=np.int8)
    acc = np.zeros((p.accumulator_rows, p.column_tile), dtype=np.int32)
    output = np.zeros_like(expected)
    writes = np.zeros_like(expected, dtype=int)
    pending = None
    events = []

    def lower(kind, f):
        nonlocal pending
        events.append(kind)
        if kind == "pack_input_panel":
            assert pending is None
            pending = f
            width = (f["output_pixels"] - 1) * 2 + 3
            panel = scratch[f["panel_offset"] : f["panel_offset"] + f["rows"] * selected.reduction].reshape(-1, 27)
            for row in range(f["rows"]):
                ly, lx = divmod(f["first_halo_row"] + row, width)
                y = min(max(f["output_y"] * 2 - 1 + ly, 0), h - 1)
                x = min(max(f["output_x"] * 2 - 1 + lx, 0), w - 1)
                panel[row].fill(0)
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y - 1 + ky, x - 1 + kx
                        if 0 <= iy < h and 0 <= ix < w:
                            panel[row, (ky * 3 + kx) * 3 : (ky * 3 + kx + 1) * 3] = data[f["image"], iy, ix]
        elif kind == "produce_packed_panel":
            assert f == pending
            panel = scratch[f["panel_offset"] : f["panel_offset"] + f["rows"] * 27].reshape(-1, 27)
            result = panel.astype(np.int32) @ weight.astype(np.int32) + bias
            target = scratch[f["stripe_offset"] : f["stripe_offset"] + f["rows"] * 5].reshape(-1, 5)
            for row in range(f["rows"]):
                for n in range(5):
                    q = selected.source.readouts[n // 4].producer
                    target[row, n] = quantized(int(result[row, n]), q.scales, q.relu)
            pending = None
        elif kind == "reload_intermediate":
            assert pending is None
            for row in range(f["rows"]):
                start = f["scratch_offset"] + row * f["scratch_row_stride"]
                acc[f["accumulator_row"] + row, : f["columns"]] = scratch[start : start + f["columns"]]
        elif kind == "store_spatial_max":
            for x in range(f["output_pixels"]):
                for n in range(f["columns"]):
                    values = [
                        quantized(int(acc[p.accumulator_row + ky * f["local_width"] + x * 2 + kx, n]), (0.75,))
                        for ky in range(3)
                        for kx in range(3)
                    ]
                    output[f["output_row"] + x, f["output_column"] + n] = max(values)
                    writes[f["output_row"] + x, f["output_column"] + n] += 1
        else:
            raise AssertionError(kind)

    receipt = emit_packed_spatial_producer(selected, Integers(), lower, synchronous_completion=True)
    np.testing.assert_array_equal(output, expected)
    assert np.all(writes == 1) and pending is None
    assert receipt["private_workspace_bytes"] == 344 < b * h * w * 5
    assert events.count("pack_input_panel") == events.count("produce_packed_panel")


def test_actual_packed_driver_emits_compact_llvm_loops_and_explicit_callback_order():
    builder = FnBuilder([])
    events = []

    def lower(kind, fields):
        events.append(kind)
        builder.add(
            UnregisteredOp.with_name("test." + kind).create(attributes={"fields": StringAttr(str(sorted(fields)))})
        )

    receipt = emit_packed_spatial_producer(plan(), FnBuilderControlAdapter(builder), lower, synchronous_completion=True)
    module = ModuleOp([llvm.FuncOp("packed", llvm.LLVMFunctionType([], llvm.LLVMVoidType()), body=builder.finish())])
    module.verify()
    assert sum(op.name == "llvm.cond_br" for op in module.walk()) >= 6
    assert events[:2] == ["pack_input_panel", "produce_packed_panel"]
    assert receipt["original_plan_sha256"] == plan().source.identity


def test_packed_plan_refuses_missing_observers_workspace_and_original_q1_boundary():
    selected = plan()
    with pytest.raises(ValueError, match="workspace capacity"):
        replace(selected, workspace=replace(selected.workspace, size=343))
    with pytest.raises(ValueError, match="external observers"):
        replace(selected.source, external_observers=("original_intermediate_output",))
    with pytest.raises(ValueError, match="staged"):
        q = RoundedReadout((0.75,))
        replace(
            selected,
            source=replace(
                selected.source,
                readout_mode="single_readout",
                intermediate=None,
                readouts=(SpatialReadoutChain(None, q, q),) * 2,
            ),
        )


def test_prospective_plan_decoder_requires_explicit_owner_kind_and_exact_frozen_identity():
    packed = plan()
    record = json.loads(json.dumps(asdict(packed)))
    assert read_spatial_plan_record(record, kind="staged_packed", expected_identity=packed.identity) == packed
    old = json.loads(json.dumps(asdict(packed.source)))
    assert (
        read_spatial_plan_record(old, kind="staged_gather", expected_identity=packed.source.identity) == packed.source
    )
    with pytest.raises(ValueError, match="fields"):
        read_spatial_plan_record(record, kind="staged_gather", expected_identity=packed.identity)
    with pytest.raises(ValueError, match="identity"):
        read_spatial_plan_record(record, kind="staged_packed", expected_identity=packed.source.identity)
    record["source"]["readouts"][0]["producer"]["scales"] = [0.125]
    with pytest.raises(ValueError, match="identity"):
        read_spatial_plan_record(record, kind="staged_packed", expected_identity=packed.identity)
    record["unreviewed_effect"] = "noescape"
    with pytest.raises(ValueError, match="fields"):
        read_spatial_plan_record(record, kind="staged_packed", expected_identity=packed.identity)
