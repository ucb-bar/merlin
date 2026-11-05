"""Source conv Q1 and pool Q2 remain distinct through two circular lifetimes."""

import operator
from dataclasses import replace

import numpy as np
import pytest

from merlin.compile.scheduling.block_schedule import SpatialMaxReadout
from merlin.compile.scheduling.program_limits import AddressWindow
from merlin.compile.scheduling.rolling_spatial import (
    RollingPackedSpatialPlan,
    emit_rolling_spatial_producer,
    propose_rolling_spatial_choices,
)
from merlin.compile.scheduling.rolling_window import emit_rolling_window_lookup
from merlin.compile.scheduling.spatial_producer import PackedSpatialProducerPlan, SpatialProducerPlan
from merlin.compile.scheduling.spatial_readout import RoundedReadout, SpatialReadoutChain
from merlin.compile.scheduling.window_transfer import CountedConvolutionPlan, Window2D
from merlin.llvmlower.requantization import quantized


class Integers:
    def op(self, name, a, b, dtype="i64"):
        if name in ("udiv", "urem"):
            return (a % (1 << 64)) // b if name == "udiv" else (a % (1 << 64)) % b
        return {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "and": operator.and_}[name](a, b)

    def cmp(self, name, a, b):
        return {"eq": operator.eq, "slt": operator.lt, "ult": lambda a, b: a % (1 << 64) < b % (1 << 64)}[name](a, b)

    def select(self, condition, yes, no):
        return yes if condition else no

    def branch(self, condition, yes, no):
        (yes if condition else no)()

    def loop(self, low, high, step, body):
        for i in range(low, high, step):
            body(i)


def make_plan(height, width, pool):
    window = Window2D((2, height, width, 3), (3, 3), padding=(1, 1, 1, 1))
    producer = CountedConvolutionPlan(window, 5, 4, min(4, width), 4, 3, 32, 256, 0, 8, 3, True)
    chains = tuple(SpatialReadoutChain(RoundedReadout((q,), True), RoundedReadout((0.75,)), None) for q in (0.25, 0.5))
    source = SpatialProducerPlan(
        producer,
        pool,
        chains,
        min(3, pool.output_hw[1]),
        readout_mode="staged_readouts",
        intermediate=AddressWindow("old", 0, 4096),
    )
    base = PackedSpatialProducerPlan(source, 4, AddressWindow("owned", 0, 8192), 8)
    selected = RollingPackedSpatialPlan(base)
    return RollingPackedSpatialPlan(replace(base, workspace=replace(base.workspace, size=selected.required_bytes)))


@pytest.mark.parametrize(
    "height,width,pool_kernel,pool_stride,padding",
    [
        (6, 7, (3, 3), (2, 2), (1, 1, 1, 1)),
        (1, 3, (3, 3), (2, 2), (1, 1, 1, 1)),
        (5, 6, (3, 3), (1, 2), (1, 1, 2, 1)),
        (6, 7, (4, 2), (2, 1), (1, 0, 2, 0)),
    ],
)
def test_ring_input_and_q1_keep_original_quantization_and_all_output_writes(
    height, width, pool_kernel, pool_stride, padding
):
    pool = SpatialMaxReadout((height, width), pool_kernel, pool_stride, padding, -128)
    plan = make_plan(height, width, pool)
    packed, source = plan.packed, plan.packed.source
    p = source.producer
    values = (np.arange(2 * height * width * 3) % 17 - 8).astype(np.int8).reshape(2, height, width, 3)
    weights = (np.arange(27 * 5) % 11 - 5).astype(np.int8).reshape(27, 5)
    bias = np.array([-(1 << 31), (1 << 31) - 1, -19, 0, 31], dtype=np.int32)
    q1 = np.zeros((2, height, width, 5), dtype=np.int8)
    for b in range(2):
        for y in range(height):
            for x in range(width):
                patch = np.zeros((3, 3, 3), dtype=np.int8)
                for ky in range(3):
                    for kx in range(3):
                        iy, ix = y - 1 + ky, x - 1 + kx
                        if 0 <= iy < height and 0 <= ix < width:
                            patch[ky, kx] = values[b, iy, ix]
                raw = patch.ravel().astype(np.int32) @ weights.astype(np.int32) + bias
                for n in range(5):
                    q1[b, y, x, n] = quantized(int(raw[n]), source.readouts[n // 4].producer.scales, True)
    oh, ow = pool.output_hw
    expected = np.full((2 * oh * ow, 5), -128, dtype=np.int8)
    for b in range(2):
        for y in range(oh):
            for x in range(ow):
                for ky in range(pool_kernel[0]):
                    for kx in range(pool_kernel[1]):
                        iy, ix = y * pool_stride[0] - padding[0] + ky, x * pool_stride[1] - padding[1] + kx
                        if 0 <= iy < height and 0 <= ix < width:
                            for n in range(5):
                                expected[(b * oh + y) * ow + x, n] = max(
                                    expected[(b * oh + y) * ow + x, n], quantized(int(q1[b, iy, ix, n]), (0.75,))
                                )
    scratch = np.zeros(packed.workspace.size, dtype=np.int8)
    data = values.ravel()
    acc = np.zeros((p.accumulator_rows, p.column_tile), dtype=np.int32)
    output, writes = np.zeros_like(expected), np.zeros_like(expected, dtype=int)
    pending, busy, state = None, False, None
    row_stages = 0
    integers = Integers()

    def lower(kind, f):
        nonlocal pending, busy, state, row_stages
        if kind == "begin_transport":
            assert not busy and pending is None and state is None
            state = object()
            return state
        if kind == "stage_row":
            assert f["transfer"] is state and not busy
            src, dst, count = f["source_byte"], f["cache_byte"], f["row_bytes"]
            scratch[dst : dst + count] = data[src : src + count] if f["valid"] else 0
            row_stages += 1
        elif kind == "await_transport":
            assert f["transfer"] is state
            busy = True
            return state
        elif kind == "pack_cached_input_panel":
            assert busy and pending is None
            pending = f
            panel = scratch[f["panel_offset"] : f["panel_offset"] + f["rows"] * 27].reshape(-1, 27)
            for r in range(f["rows"]):
                x = min(max(f["output_x"] * pool_stride[1] - padding[1] + f["first_halo_column"] + r, 0), width - 1)
                for ky in range(3):
                    for kx in range(3):
                        for c in range(3):
                            lookup = emit_rolling_window_lookup(plan.rolling, integers, f["producer_y"], x, ky, kx, c)
                            panel[r, (ky * 3 + kx) * 3 + c] = (
                                scratch[lookup["cache_byte"]] if lookup["predicate"] else 0
                            )
        elif kind == "produce_packed_panel":
            assert busy and pending == f
            panel = scratch[f["panel_offset"] : f["panel_offset"] + f["rows"] * 27].reshape(-1, 27)
            raw = panel.astype(np.int32) @ weights.astype(np.int32) + bias
            target = scratch[f["stripe_offset"] : f["stripe_offset"] + f["rows"] * 5].reshape(-1, 5)
            for r in range(f["rows"]):
                for n in range(5):
                    target[r, n] = quantized(int(raw[r, n]), source.readouts[n // 4].producer.scales, True)
            pending = None
        elif kind == "reload_intermediate":
            assert busy and pending is None
            for r in range(f["rows"]):
                first = f["scratch_offset"] + r * f["scratch_row_stride"]
                acc[f["accumulator_row"] + r, : f["columns"]] = scratch[first : first + f["columns"]]
        elif kind == "store_spatial_max":
            for x in range(f["output_pixels"]):
                for n in range(f["columns"]):
                    output[f["output_row"] + x, f["output_column"] + n] = max(
                        quantized(
                            int(acc[p.accumulator_row + ky * f["local_width"] + x * pool_stride[1] + kx, n]), (0.75,)
                        )
                        for ky in range(pool_kernel[0])
                        for kx in range(pool_kernel[1])
                    )
                    writes[f["output_row"] + x, f["output_column"] + n] += 1
        elif kind == "consumer_issued":
            assert busy and pending is None
            return state
        elif kind == "complete":
            assert f["consumer"] is state
            busy, state = False, None
        else:
            raise AssertionError(kind)

    receipt = emit_rolling_spatial_producer(plan, integers, lower, synchronous_completion=True)
    np.testing.assert_array_equal(output, expected)
    assert np.all(writes == 1) and not busy and pending is state is None
    stripes = (ow + source.output_tile_width - 1) // source.output_tile_width
    assert row_stages == stripes * 2 * (height + 2)
    assert receipt["private_workspace_bytes"] == packed.workspace.size
    assert receipt["full_intermediate_image_bytes"] == 0
    with pytest.raises(ValueError, match="capacity"):
        RollingPackedSpatialPlan(replace(packed, workspace=replace(packed.workspace, size=plan.required_bytes - 1)))


def test_capacity_choices_shrink_stripe_and_costs_match_actual_counted_consumer_commands():
    source = make_plan(6, 7, SpatialMaxReadout((6, 7), (3, 3), (2, 2), (1, 1, 1, 1), -128)).packed.source
    choices = propose_rolling_spatial_choices(
        source, AddressWindow("bounded", 16, 180), panel_rows_limit=16, output_columns_limit=3, alignment=8
    )
    assert len(choices) == 1
    selected = choices[0]
    packed = selected.plan.packed
    assert packed.source.output_tile_width == 1 < source.output_tile_width
    assert packed.source.producer == source.producer and packed.source.readouts == source.readouts
    assert selected.plan.required_bytes <= 180 and packed.workspace.size == selected.plan.required_bytes
    p = source.producer
    counts = dict(invocations=0, macs=0, primitive=0, staged=0)

    def lower(kind, fields):
        if kind in ("begin_transport", "await_transport", "consumer_issued"):
            return object()
        if kind == "produce_packed_panel":
            rows, k, n = fields["rows"], fields["reduction"], fields["columns"]
            counts["invocations"] += 1
            counts["macs"] += rows * k * n
            rounded = [(v + p.primitive - 1) // p.primitive * p.primitive for v in (rows, k, n)]
            counts["primitive"] += rounded[0] * rounded[1] * rounded[2]
        elif kind == "stage_row":
            counts["staged"] += fields["row_bytes"]

    emit_rolling_spatial_producer(selected.plan, Integers(), lower, synchronous_completion=True)
    assert counts == dict(
        invocations=selected.dense_invocations,
        macs=selected.logical_macs,
        primitive=selected.primitive_mac_extent,
        staged=selected.staged_input_bytes,
    )
    with pytest.raises(ValueError, match="no complete"):
        propose_rolling_spatial_choices(
            source, AddressWindow("too_small", 0, 16), panel_rows_limit=16, output_columns_limit=3, alignment=8
        )
