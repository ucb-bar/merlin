"""Lowering a replayed Voyager GEMM or convolution schedule onto a weight-stationary array
(baselines.voyager_schedule).

The arithmetic is proven by executing the lowered abstract schedule with numpy against ``lhs @ weight``
(or a direct NHWC convolution plus bias) on random int8 data -- exact, since a K split accumulates in
the integer accumulator (concession C2). The structure is pinned against Voyager's own trace: one block
load per block of every tile Voyager loads, the loads in Voyager's order, one weight preload per
resident-block change.
"""
from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from merlin.baselines.voyager_ir import Copy, FusedCompute, UnsupportedConstruct, load_model, replay
from merlin.baselines.voyager_schedule import (AccMvin, Compute, Geometry, Mvin, Mvout, Preload,
                                               execute, lower_conv, lower_gemm)
from merlin.common.paths import merlin_dir

FIXTURES = merlin_dir() / "tests" / "data" / "voyager_ir"
# The geometry the fixtures were compiled against (their manifest): 16x16 array, 256 KiB scratchpad of
# 16-byte rows, and an accumulator of 1024 rows.
GEOMETRY = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=1024)


def _trace(name: str):
    return replay(load_model(FIXTURES / name / "model.json"))


def test_the_split_k_schedule_computes_the_exact_integer_product() -> None:
    schedule = lower_gemm(_trace("split_k_256x512x256"), GEOMETRY)
    assert schedule.shapes == {"lhs": (256, 512), "weight": (512, 256), "out": (256, 256)}
    rng = np.random.default_rng(0)
    lhs = rng.integers(-128, 128, size=(256, 512), dtype=np.int64)
    weight = rng.integers(-128, 128, size=(512, 256), dtype=np.int64)
    assert np.array_equal(execute(schedule, lhs, weight), (lhs @ weight).astype(np.int32))
    assert any("C2" in note for note in schedule.notes)


def test_every_voyager_load_becomes_exactly_its_blocks_in_order() -> None:
    trace = _trace("split_k_256x512x256")
    schedule = lower_gemm(trace, GEOMETRY)
    d = GEOMETRY.dim
    expected = [(("lhs" if c.src.box.node == "x_preprocess" else "weight"),
                 (c.sizes[0] // d) * (c.sizes[1] // d))
                for c in trace.of(Copy) if c.is_load]
    got, run = [], None
    for op in (o for o in schedule.ops if isinstance(o, Mvin)):
        if run and run[0] == op.role and run[1] < run[2]:
            run[1] += 1
        else:
            n = expected[len(got)][1]
            run = [op.role, 1, n]
            got.append((op.role, n))
    assert got == expected
    assert schedule.count(Mvin) == sum(n for _, n in expected)
    stores = [c for c in trace.of(Copy) if c.is_store]
    assert schedule.count(Mvout) == sum((c.sizes[0] // d) * (c.sizes[1] // d) for c in stores)


def test_weights_are_preloaded_once_per_resident_block_and_every_compute_is_preceded() -> None:
    trace = _trace("split_k_256x512x256")
    schedule = lower_gemm(trace, GEOMETRY)
    preloads = [op for op in schedule.ops if isinstance(op, Preload)]
    computes = [op for op in schedule.ops if isinstance(op, Compute)]
    assert len(preloads) == len(computes)
    # Derived from Voyager's own mapping, not from the tile: the nest (outermost first) is
    # L2 OC4 > OX2 > IC16, then L1 OC2 > OX32. Every point outside the 32-row stream selects a new
    # (IC, OC) block -- IC sits INSIDE the L2 row loop, so each of the tile's 16x8 weight blocks is made
    # resident once per row half: 4*2*16*2 = 256 weight changes, each streaming 32 rows as two
    # 16-row computes, per commit.
    commits = trace.of(FusedCompute)
    points = chunks = 0
    for comp in commits:
        nest = [b for level in reversed(comp.tiling) for b in reversed(level)]
        (_, stream), outer = nest[-1], nest[:-1]
        n = 1
        for _, bound in outer:
            n *= bound
        points += n
        chunks += n * -(-stream // GEOMETRY.dim)
    assert points == 256 * len(commits)
    assert sum(op.weight_row is not None for op in preloads) == points
    assert sum(c.fresh_weights for c in computes) == points
    assert len(computes) == chunks == 512 * len(commits)


def test_a_gemm_bias_is_loaded_into_the_accumulator_exactly() -> None:
    schedule = lower_gemm(_trace("lin64"), GEOMETRY)
    assert schedule.shapes["bias"] == (1, 64)
    rng = np.random.default_rng(1)
    lhs = rng.integers(-128, 128, size=schedule.shapes["lhs"], dtype=np.int64)
    weight = rng.integers(-128, 128, size=schedule.shapes["weight"], dtype=np.int64)
    bias = rng.integers(-2**20, 2**20, size=schedule.shapes["bias"], dtype=np.int64)
    expected = (lhs @ weight + bias).astype(np.int32)
    assert np.array_equal(execute(schedule, lhs, weight, bias=bias), expected)
    no_bias = replace(schedule, ops=[op for op in schedule.ops if not isinstance(op, AccMvin)])
    assert not np.array_equal(execute(no_bias, lhs, weight, bias=bias), expected)


def test_an_output_tile_the_accumulator_cannot_hold_is_refused() -> None:
    small = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=16)
    with pytest.raises(UnsupportedConstruct, match="larger than the accumulator"):
        lower_gemm(_trace("split_k_256x512x256"), small)


# Conv fixture -> the rows every compute streams, fixed by Voyager's own mapping on this geometry: the
# 3x3 layers keep an innermost OX extent of 4 (stride 1) or 2 (stride 2) inside a 30- or 29-pixel
# tile row, so each weight residency's pixels come as 4- or 2-pixel runs; the 1x1 layer's tile is 4
# pixels wide, so its OY4 x OX4 stream is one contiguous 16-row run.
CONV_FIXTURES = {"conv3x3_s1_28x28x64x64": 4, "conv3x3_s2_28x28x64x128": 2,
                 "conv1x1_28x28x64x256": 16}


def _conv_case(name: str):
    """(lhs, weight, bias) as the schedule's 2-D views, and the exact int32 convolution plus bias."""
    workload = json.loads((FIXTURES / name / "manifest.json").read_text())["workload"]
    k, cin, cout = workload["k"], workload["Cin"], workload["Cout"]
    stride, pad = workload["stride"], workload.get("padding", k // 2)  # the exporter's default
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=(1, workload["H"], workload["W"], cin), dtype=np.int64)
    w = rng.integers(-128, 128, size=(k, k, cin, cout), dtype=np.int64)
    bias = rng.integers(-2**20, 2**20, size=cout, dtype=np.int64)
    padded = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    oh = (padded.shape[1] - k) // stride + 1
    ow = (padded.shape[2] - k) // stride + 1
    out = np.zeros((1, oh, ow, cout), dtype=np.int64)
    for fy in range(k):
        for fx in range(k):
            window = padded[:, fy:fy + stride * oh:stride, fx:fx + stride * ow:stride]
            out += np.einsum("nhwc,co->nhwo", window, w[fy, fx])
    operands = (x.reshape(-1, cin), w.reshape(-1, cout), (bias[None, :]))
    return operands, (out + bias).reshape(-1, cout).astype(np.int32)


@pytest.mark.parametrize("name", sorted(CONV_FIXTURES))
def test_a_conv_schedule_computes_the_exact_integer_convolution(name: str) -> None:
    schedule = lower_conv(_trace(name), GEOMETRY)
    (lhs, weight, bias), expected = _conv_case(name)
    assert np.array_equal(execute(schedule, lhs, weight, bias=bias), expected)


def test_the_conv_oracle_fails_when_the_bias_or_the_phase_split_is_broken() -> None:
    name = "conv3x3_s2_28x28x64x128"
    schedule = lower_conv(_trace(name), GEOMETRY)
    (lhs, weight, bias), expected = _conv_case(name)
    no_bias = replace(schedule, ops=[op for op in schedule.ops if not isinstance(op, AccMvin)])
    unit_step = replace(schedule, ops=[replace(op, row_step=1)
                                       if isinstance(op, Mvin) and op.role == "lhs" else op
                                       for op in schedule.ops])
    assert any(isinstance(op, AccMvin) for op in schedule.ops)
    assert {op.row_step for op in schedule.ops if isinstance(op, Mvin) and op.role == "lhs"} == {2}
    for broken in (no_bias, unit_step):
        assert not np.array_equal(execute(broken, lhs, weight, bias=bias), expected)


@pytest.mark.parametrize("name, rows", sorted(CONV_FIXTURES.items()))
def test_conv_weights_change_once_per_voyager_residency_and_runs_follow_its_stream(
        name: str, rows: int) -> None:
    trace = _trace(name)
    schedule = lower_conv(trace, GEOMETRY)
    computes = [op for op in schedule.ops if isinstance(op, Compute)]
    preloads = [op for op in schedule.ops if isinstance(op, Preload)]
    # Every point of the loops outside the innermost pixel loops selects a new weight block.
    points = 0
    for comp in trace.of(FusedCompute):
        nest = [b for level in reversed(comp.tiling) for b in reversed(level)]
        while nest and nest[-1][0] in ("LOOP_OY", "LOOP_OX"):
            nest.pop()
        n = 1
        for _, bound in nest:
            n *= bound
        points += n
    assert len(preloads) == len(computes)
    assert sum(op.weight_row is not None for op in preloads) == points
    assert sum(c.fresh_weights for c in computes) == points
    assert {c.rows for c in computes} == {rows}


def test_an_output_ring_deeper_than_the_accumulator_is_made_shallower_exactly() -> None:
    # Two 392-row output tiles do not fit 512 accumulator rows: one region, and each tile's store is
    # issued before the next tile first writes it (C7). A tile larger than the accumulator refuses.
    name = "conv3x3_s1_28x28x64x64"
    small = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=512)
    schedule = lower_conv(_trace(name), small)
    (lhs, weight, bias), expected = _conv_case(name)
    assert np.array_equal(execute(schedule, lhs, weight, bias=bias), expected)
    assert any("C7" in note for note in schedule.notes)
    tiny = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=256)
    with pytest.raises(UnsupportedConstruct, match="larger than the accumulator"):
        lower_conv(_trace(name), tiny)


def test_a_gemm_trace_is_not_lowered_as_a_convolution() -> None:
    with pytest.raises(UnsupportedConstruct, match="not a conv"):
        lower_conv(_trace("split_k_256x512x256"), GEOMETRY)
