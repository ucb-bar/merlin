"""Lowering a replayed Voyager GEMM schedule onto a weight-stationary array (baselines.voyager_schedule).

The arithmetic is proven by executing the lowered abstract schedule with numpy against ``lhs @ weight``
on random int8 data -- exact, since a K split now accumulates in the integer accumulator (concession
C2). The structure is pinned against Voyager's own trace: one block load per block of every tile
Voyager loads, the loads in Voyager's order, one weight preload per resident-block change.
"""
from __future__ import annotations

import numpy as np
import pytest

from merlin.baselines.voyager_ir import Copy, FusedCompute, UnsupportedConstruct, load_model, replay
from merlin.baselines.voyager_schedule import (Compute, Geometry, Mvin, Mvout, Preload, execute,
                                               lower_gemm)
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


def test_a_bias_operand_is_refused_rather_than_dropped() -> None:
    with pytest.raises(UnsupportedConstruct, match="bias"):
        lower_gemm(_trace("lin64"), GEOMETRY)


def test_an_output_tile_the_accumulator_cannot_hold_is_refused() -> None:
    small = Geometry(dim=16, spad_rows=16384, spad_row_bytes=16, acc_rows=256)
    with pytest.raises(UnsupportedConstruct, match="accumulator"):
        lower_gemm(_trace("split_k_256x512x256"), small)
