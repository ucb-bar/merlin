"""Output residency: the accumulator bound, which the capacity obligation used to ignore.

``capacity_fit`` modelled only the OPERAND store -- the weight tile plus the activation tile. But two
stores bound a contraction and they bind different dimensions: operands grow as K*(M+N), the OUTPUT
grows as M*N, and the accumulator holding that output is a separate, much smaller SRAM. A wide-output
layer therefore sits comfortably inside the scratchpad while overrunning the accumulator.

Measured on the whole-model capsule: four routed layers -- (345,32)@(32,256) and (96,64)@(64,512) --
had operands well inside a 256 KiB scratchpad, so nothing tiled them, and each aborted the simulator
with ``vector::_M_range_check: __n (which is 1024) >= this->size() (which is 1024)``: the 1024
accumulator rows (65536 B / (16 elems * 4 B)) addressed one past the end. Every tile that DID run
passed, so the cost was four UNMEASURABLE layers, not four wrong ones -- the capsule failed for want
of a measurement. This is the same class of bug the operand-store term already documents, one store
over.

The bound is DERIVED (mlc's discovered ``accumulator_bytes``), never a literal, so a target that
declares no accumulator stays undecidable rather than being assumed to fit.
"""
from __future__ import annotations

import pytest

from merlin.compile_cli import _accumulator_capacity_elems, _capacity_fit_tile, capacity_fit


def _acc(target="gemmini", dt="i32"):
    a = _accumulator_capacity_elems(target, dt)
    if not a:
        pytest.skip("no derivable accumulator capacity in this checkout (mlc unavailable)")
    return a


def test_the_accumulator_bound_is_derived_not_declared():
    """It comes out of the target's own RTL-discovered capacities, in elements of the accum dtype."""
    assert _acc() > 0
    # narrower accumulation words => more of them fit in the same SRAM
    assert _accumulator_capacity_elems("gemmini", "i32") < _accumulator_capacity_elems("gemmini", "i8")


def test_an_undecidable_target_stays_undecidable():
    assert _accumulator_capacity_elems("definitely_not_a_target", "i32") is None
    v = capacity_fit("definitely_not_a_target", 345, 32, 256, "int8", 16, "i32")
    assert v["holds"] is None, "unknown capacity must never be assumed to hold"


@pytest.mark.parametrize("m,k,n", [(345, 32, 256), (96, 64, 512)])
def test_the_layers_that_aborted_the_simulator_are_now_caught(m, k, n):
    """Operands fit; the OUTPUT does not. Before this term the verdict was an unqualified `holds`."""
    v = capacity_fit("gemmini", m, k, n, "int8", 16, "i32")
    assert v["operands_hold"] is True, "these layers' operands were never the problem"
    assert v["output_holds"] is False, f"output {v['output_elems']} vs acc {v['accumulator_capacity_elems']}"
    assert v["holds"] is False, "a layer whose output cannot be resident does not satisfy capacity_fit"


@pytest.mark.parametrize("m,k,n", [(345, 32, 256), (96, 64, 512)])
def test_the_tiler_produces_an_output_tile_that_fits(m, k, n):
    acc = _acc()
    cap = capacity_fit("gemmini", m, k, n, "int8", 16, "i32")["capacity_elems"]
    mt, kt, nt, n_tiles = _capacity_fit_tile(m, k, n, 16, cap, acc)
    assert mt * nt <= acc, f"tile output {mt}x{nt} still overruns the accumulator"
    assert kt * nt + mt * kt <= cap, "the operand bound must still hold"
    assert n_tiles > 1 and all(x % 16 == 0 or x == v for x, v in ((mt, m), (kt, k), (nt, n)))


def test_row_dim_splits_only_when_the_output_forces_it():
    """M is the last dimension left when N is already at the tile edge -- K does not appear in M*N.
    Without an accumulator bound M stays whole, so callers that pass none keep the old extents."""
    D, cap = 16, 262144
    # N already at the edge, output far too large => M must give
    mt, _, nt, _ = _capacity_fit_tile(4096, 16, 16, D, cap, 1024)
    assert mt * nt <= 1024 and mt < 4096
    # no accumulator bound => unchanged behaviour
    assert _capacity_fit_tile(345, 32, 256, D, cap) == (345, 32, 256, 1)


def test_a_layer_that_fits_both_stores_is_not_tiled():
    acc = _acc()
    v = capacity_fit("gemmini", 16, 16, 16, "int8", 16, "i32")
    assert v["holds"] is True
    assert _capacity_fit_tile(16, 16, 16, 16, v["capacity_elems"], acc) == (16, 16, 16, 1)
