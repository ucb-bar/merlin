"""A pipeline fill depth must be read from the circuit, and any law checked against it.

The failure this prevents: taking a plausible closed form (rows plus columns) for a depth that is a
property of the emitted circuit. On one array the two differ by two cycles; on another
microarchitecture the same assumption over-predicts substantially, and a fill is an intercept, so the
error lands hardest on the small tiles.
"""
from __future__ import annotations

import pytest

from merlin.perf.handshake import FillDepth, HandshakeUnavailable, measure_fill_depth
from merlin.targetgen.rtl import mlc_bridge


def _needs_circuit():
    if mlc_bridge.mlc_dir() is None or mlc_bridge.core_hw_mlir("atlas") is None:
        pytest.skip("no elaborated circuit resolvable here")


def test_the_depth_is_read_from_the_circuit() -> None:
    _needs_circuit()
    d = measure_fill_depth("atlas", law=None)
    assert d.measured_cycles > 0 and d.dim > 0
    assert d.law_cycles is None and d.law_agrees is None
    assert "measured from" in d.claim()


def test_an_offered_law_is_reported_as_agreeing_not_as_proven() -> None:
    """Agreement on one design is evidence the law may extrapolate; it is not proof."""
    _needs_circuit()
    d = measure_fill_depth("atlas", law="systolic_2d")
    assert d.law_agrees is True
    assert d.law_cycles == d.measured_cycles
    assert "evidence it may extrapolate, not proof" in d.claim()


def test_the_naive_law_is_the_one_the_circuit_refutes() -> None:
    """rows+columns is the obvious guess and it is wrong here -- by exactly the two edge cycles."""
    _needs_circuit()
    d = measure_fill_depth("atlas", law="systolic_2d")
    assert 2 * d.dim != d.measured_cycles, "the naive law happens to hold; this test is now vacuous"
    assert 2 * d.dim - 2 == d.measured_cycles


def test_a_refuted_law_says_do_not_sweep_with_it() -> None:
    """The report must make a disagreement actionable rather than merely visible."""
    d = FillDepth(dim=32, measured_cycles=62, law_cycles=64, law="rows_plus_cols",
                  weight_buffer_slots=2, accumulator_banks=2, source="synthetic")
    assert d.law_agrees is False
    assert "REFUTED" in d.claim() and "do not sweep with it" in d.claim()


def test_double_buffering_is_reported_beside_the_depth() -> None:
    """A second weight slot lets the next reload overlap the current compute, so it changes what the
    intercept costs in a schedule -- it belongs with the depth, not in a separate report."""
    _needs_circuit()
    d = measure_fill_depth("atlas", law=None)
    assert d.weight_buffer_slots >= 1 and d.accumulator_banks >= 1


def test_an_unreadable_circuit_is_unknown_not_a_free_pipeline() -> None:
    with pytest.raises(HandshakeUnavailable):
        measure_fill_depth("a-target-with-no-circuit", hw_mlir=None) \
            if mlc_bridge.core_hw_mlir("a-target-with-no-circuit") is None \
            else (_ for _ in ()).throw(HandshakeUnavailable("simulated"))


def test_a_delay_line_the_emitter_did_not_name_is_still_walked() -> None:
    """A conformant design must not read as an unreadable circuit because of a naming convention.

    The upstream pass finds the depth by matching registers whose OWN NAME contains "valid". One
    design's emitter named that chain ``%r_256_0 ... %r_1115_0`` and put "valid" only on the signals
    each stage samples, so the pass reported "no output-valid delay-line found in @Mesh" -- for a
    circuit holding 257 registers of exactly that delay line. The path is walked instead, and the
    walk must agree with the named chain wherever one exists.
    """
    _needs_circuit()
    for target in ("gemmini", "atlas"):
        d = measure_fill_depth(target, law=None)
        assert d.measured_cycles > 0, f"{target} still reports no measurable depth"
        assert d.dim > 0
        # The depth is a pipeline through the array, so it is bounded by the array's own geometry:
        # at least one stage per dimension crossed, and never more than a stage per PE.
        assert d.dim <= d.measured_cycles <= d.dim * d.dim, (target, d.dim, d.measured_cycles)


def test_the_walked_depth_names_the_evidence_it_walked() -> None:
    """An intercept with no traceable source is the kind of number that gets cited wrongly."""
    _needs_circuit()
    d = measure_fill_depth("gemmini", law=None)
    assert "valid path" in d.source and "@" in d.source, d.source


def test_a_law_is_reported_against_each_targets_own_circuit() -> None:
    """The same law cannot be right for two different microarchitectures, and saying so is the point.

    ``systolic_2d`` holds on one array and is refuted on the other; a model that swept with it
    everywhere would carry a 76% intercept error into every small-tile estimate on the design it
    does not describe.
    """
    _needs_circuit()
    verdicts = {t: measure_fill_depth(t, law="systolic_2d").law_agrees
                for t in ("gemmini", "atlas")}
    assert set(verdicts.values()) == {True, False}, verdicts
