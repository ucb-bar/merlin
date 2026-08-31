"""The eta falsifier: it has to be able to FIRE, and it has to refuse when it cannot.

On a hardware-interlocked, command-driven accelerator every reordering is correct by construction, so
a capsule whose falsifier is bit-exactness passes every candidate schedule and learns nothing
(``docs/design/performance_levers_per_archetype.md``). These tests pin the replacement falsifier --
eta, the realised fraction of available overlap -- and they are written around the two properties
that make it worth having:

  * ``test_the_falsifier_fires_*``: a candidate that is BIT-EXACT and did not raise overlap is
    REJECTED. If this test ever passes trivially the falsifier is inert and the capsule is decorative.
  * ``test_*_is_undeterminable_*``: the cases where it must decline to judge -- a vector that could
    not have shown overlap, an unread unit, an unbound column, two runs on different engines or
    different work. Each of those has a matching real failure in this tree, where an unmeasurable
    thing was reported as a measured zero.

The third state is asserted as a distinct value everywhere, never as "not rose".
"""
from __future__ import annotations

from merlin.perf.falsifier import (
    ACCEPT, DID_NOT_RISE, ENGINE_AXIS, KIND_AXIS, REJECT, ROSE, UNDETERMINABLE, EtaObservation,
    ab_decision, compare_eta, eta_from_occupancy, eta_from_timing_block,
)

# Two decoupled engines with a nested third, declared by the producer. No target, no unit name from
# any real device: the binding is a parameter and these are placeholders for it.
BIND = {"mover.busy": "E_move", "mover.load": "E_move", "arith.busy": "E_arith"}
KINDS = {"mover.busy": "data_movement", "mover.load": "data_movement", "arith.busy": "compute"}


def _vec(mover: str, arith: str) -> dict:
    """An occupancy vector written as two bit strings, one character per cycle."""
    return {"mover.busy": [c == "1" for c in mover], "arith.busy": [c == "1" for c in arith]}


# Three schedules of IDENTICAL work: each engine is busy the same 8 cycles in all three, so the only
# thing that differs is the sequence. That is exactly the candidate an interlocked machine makes
# indistinguishable on correctness -- and the reason eta has to be the discriminator.
#
# Serial: movement, then compute. Nothing runs together.
SERIAL = _vec("1111111100000000", "0000000011111111")
# Resequenced so 4 cycles of movement run under compute.
OVERLAPPED = _vec("1111111100000000", "0000111111110000")
# Shuffled into four blocks -- a genuinely different stream that buys no overlap at all.
RESHUFFLED = _vec("1111000011110000", "0000111100001111")


def _obs(label, vec, **kw):
    return eta_from_occupancy(label, vec, unit_of=BIND, kinds=KINDS, work="w", **kw)


# -------------------------------------------------------------------------------------------------
# 1. eta itself: a ratio with a stated denominator, or nothing at all
# -------------------------------------------------------------------------------------------------
def test_eta_is_realised_over_available_overlap():
    o = _obs("serial", SERIAL)
    assert (o.realised_cycles, o.available_cycles) == (0, 8)
    assert o.eta == 0.0
    assert o.axis == ENGINE_AXIS
    assert set(o.engines) == {"E_move", "E_arith"}

    o2 = _obs("overlapped", OVERLAPPED)
    # The arith unit starts at cycle 4 while the mover still has 4 cycles to run, so 4 of the 8
    # available overlap cycles are realised. Both engines are busy 8 cycles in both schedules.
    assert (o2.realised_cycles, o2.available_cycles) == (4, 8)
    assert o2.eta == 0.5
    assert o.busy == o2.busy == _obs("reshuffled", RESHUFFLED).busy   # identical work, resequenced


def test_a_denominator_of_zero_is_none_not_zero():
    """eta is 0/0 when no pair had any overlappable time. Reporting 0.0 would say this schedule
    realises none of its overlap, about a schedule that had none available -- so the ratio itself
    refuses, independently of the observability gates that normally catch this first."""
    o = EtaObservation("hand-built", realised_cycles=0, available_cycles=0, engines=("a", "b"))
    assert o.eta is None and not o.measured

    solo = eta_from_occupancy("solo", _vec("1111000011110000", "0000000000000000"),
                              unit_of=BIND, kinds=KINDS)
    assert solo.eta is None and solo.realised_cycles is None
    assert "live column" in solo.detail          # the earlier, more specific refusal


# -------------------------------------------------------------------------------------------------
# 2. THE FALSIFIER FIRES
# -------------------------------------------------------------------------------------------------
def test_the_falsifier_fires_on_a_bit_exact_reordering_that_bought_nothing():
    """The whole point. The candidate is a real, different schedule; the hardware's reservation
    station makes it bit-exact; and it is REJECTED because eta did not move."""
    base, cand = _obs("serial", SERIAL), _obs("reshuffled", RESHUFFLED)
    assert base.eta == 0.0 and cand.eta == 0.0            # both measured, both zero -- comparable

    verdict = compare_eta(base, cand)
    assert verdict.state == DID_NOT_RISE
    assert verdict.state != UNDETERMINABLE                # the two must never be collapsed
    assert verdict.delta == 0.0 and not verdict.fell

    decision = ab_decision(base, cand, bit_exact=True, invariants_held=True)
    assert decision.state == REJECT and not decision.accepted
    assert "did not move" in decision.reason and "bought anything" in decision.reason


def test_the_falsifier_fires_when_eta_fell():
    base, cand = _obs("overlapped", OVERLAPPED), _obs("serial", SERIAL)
    verdict = compare_eta(base, cand)
    assert verdict.state == DID_NOT_RISE and verdict.fell and verdict.delta == -0.5
    assert ab_decision(base, cand, bit_exact=True, invariants_held=True).state == REJECT


def test_a_reordering_that_raised_overlap_is_accepted():
    base, cand = _obs("serial", SERIAL), _obs("overlapped", OVERLAPPED)
    verdict = compare_eta(base, cand)
    assert verdict.state == ROSE and verdict.delta == 0.5
    assert ab_decision(base, cand, bit_exact=True, invariants_held=True).state == ACCEPT


def test_bit_exactness_alone_never_accepts():
    """The archetype's trap, stated as a test: on this machine bit_exact is True for every candidate,
    so a gate that stopped there would accept all three of these."""
    base = _obs("serial", SERIAL)
    states = {name: ab_decision(base, _obs(name, v), bit_exact=True, invariants_held=True).state
              for name, v in (("serial", SERIAL), ("reshuffled", RESHUFFLED),
                              ("overlapped", OVERLAPPED))}
    assert states == {"serial": REJECT, "reshuffled": REJECT, "overlapped": ACCEPT}


# -------------------------------------------------------------------------------------------------
# 3. and it refuses to judge when it cannot -- one case per real failure
# -------------------------------------------------------------------------------------------------
def test_a_vector_that_could_not_show_overlap_is_undeterminable_not_zero():
    """A joint vector with fewer than two live columns reports zero ARITHMETICALLY, and that zero is
    indistinguishable from a machine that genuinely serialises."""
    one = eta_from_occupancy("one-column", {"mover.busy": [True, True, False, False]},
                             unit_of=BIND, kinds=KINDS)
    assert one.eta is None
    v = compare_eta(one, _obs("overlapped", OVERLAPPED))
    assert v.state == UNDETERMINABLE
    assert ab_decision(one, _obs("o", OVERLAPPED), bit_exact=True,
                       invariants_held=True).state == UNDETERMINABLE


def test_two_live_columns_inside_one_engine_cannot_show_engine_overlap():
    """One step stricter than joint_counts' own flag: both columns are live, so column-level overlap
    IS observable, but they were DECLARED to the same engine and no engine pair could have been seen
    running together. Reporting that zero as an eta is the column-level version of the same mistake."""
    hot = {"mover.busy": [True, True, False, False], "mover.load": [False, True, True, False]}
    o = eta_from_occupancy("one-engine", hot, unit_of=BIND, kinds=KINDS)
    assert o.eta is None
    assert "one engine" in o.detail


def test_an_unbound_busy_column_is_undeterminable():
    """Its cycles are real (so it cannot be dropped) and nobody said which engine they belong to (so
    it cannot be attributed). Which engines ran together is genuinely unknown."""
    hot = dict(SERIAL)
    hot["mystery"] = [True] * 8 + [False] * 8
    o = eta_from_occupancy("unbound", hot, unit_of=BIND, kinds=KINDS)
    assert o.eta is None and "mystery" in o.detail


def test_an_unread_unit_refuses_the_reading_rather_than_shrinking_it():
    """A unit with no top-level busy port reads as permanently idle; including one such unit moved a
    measured kernel's idle fraction from 89.9% to 39.2%. UNKNOWN, never idle."""
    o = eta_from_occupancy("partial", OVERLAPPED, unit_of=BIND, kinds=KINDS,
                           unmeasured=("E_vector",))
    assert o.eta is None and "E_vector" in o.detail


def test_different_engine_sets_are_undeterminable_never_scored_zero():
    """An engine present on one side only is unmeasured there, not zero -- scoring it zero reports
    moving work off an engine as speeding that engine up."""
    third = dict(OVERLAPPED)
    third["extra.busy"] = [False] * 10 + [True] * 6
    cand = eta_from_occupancy("three", third, unit_of={**BIND, "extra.busy": "E_other"},
                              kinds={**KINDS, "extra.busy": "compute"}, work="w")
    v = compare_eta(_obs("serial", SERIAL), cand)
    assert v.state == UNDETERMINABLE and "E_other" in v.reason


def test_different_work_is_undeterminable_because_eta_is_a_ratio():
    a = _obs("serial", SERIAL)
    b = eta_from_occupancy("half", OVERLAPPED, unit_of=BIND, kinds=KINDS, work="half-the-tiles")
    v = compare_eta(a, b)
    assert v.state == UNDETERMINABLE and "different work" in v.reason


def test_two_axes_are_two_instruments():
    a = _obs("serial", SERIAL)
    b = _obs("overlapped", OVERLAPPED)
    kind_axis = eta_from_timing_block("blocky", _RECORD)
    assert kind_axis.axis == KIND_AXIS and a.axis == ENGINE_AXIS
    assert compare_eta(a, kind_axis).state == UNDETERMINABLE
    assert compare_eta(a, b).state == ROSE            # same axis still compares


# -------------------------------------------------------------------------------------------------
# 4. the occupancy guards this must not regress
# -------------------------------------------------------------------------------------------------
def test_a_sub_signal_does_not_manufacture_self_overlap():
    """A unit's busy signal counted beside the sub-signal it contains reports the unit overlapping
    with itself -- 204 fabricated cycles on one measured design. Both columns are declared to the
    same engine, so containment folds them."""
    hot = dict(SERIAL)
    hot["mover.load"] = [c and i < 4 for i, c in enumerate(hot["mover.busy"])]
    o = eta_from_occupancy("nested-signal", hot, unit_of=BIND, kinds=KINDS, work="w")
    assert o.realised_cycles == 0                     # the fold happened; no self-overlap counted
    assert o.busy["E_move"] == 8


def test_a_nested_engine_is_structure_and_is_not_folded_away():
    """The one thing containment cannot decide. An accelerator embedded in the cluster that drives it
    nests exactly like a sub-signal -- and its concurrency with the host IS the measurement. The
    engines are DECLARED apart, so the inner one survives and the overlap is counted."""
    hot = {"outer.busy": [True] * 10 + [False] * 6, "inner.busy": [False] * 2 + [True] * 6 + [False] * 8}
    o = eta_from_occupancy("nested-engine", hot,
                           unit_of={"outer.busy": "E_cluster", "inner.busy": "E_array"},
                           kinds={"outer.busy": "compute", "inner.busy": "compute"}, work="w")
    assert o.realised_cycles == 6                     # not 0 -- the inner engine was not deleted
    assert set(o.engines) == {"E_cluster", "E_array"}


# -------------------------------------------------------------------------------------------------
# 5. the timing-block path (the shape the capsule runner already emits)
# -------------------------------------------------------------------------------------------------
def _record(*, partitioned: bool, overlap: int, unmeasured=(), alias=0, mover=8, arith=10):
    return {
        "timing_observations": [
            {"quantity": "busy_cycles.mover.in_program", "value": mover, "kind": "data_movement"},
            {"quantity": "busy_cycles.arith.in_program", "value": arith, "kind": "compute"},
            {"quantity": "overlap_cycles.across_kinds", "value": overlap},
            {"quantity": "sampled_cycles.dbg_tap", "value": 16},
        ],
        "timing_capability": {"unmeasured_units": list(unmeasured), "partitioned": partitioned,
                              "alias_collisions": alias},
    }


_RECORD = _record(partitioned=False, overlap=2)


def test_a_timing_block_yields_the_same_ratio_on_the_kind_axis():
    o = eta_from_timing_block("base", _RECORD, work="w")
    assert (o.realised_cycles, o.available_cycles) == (2, 8) and o.eta == 0.25


def test_a_partitioned_producer_licenses_no_reading():
    """A partition charges every cycle to exactly one owner, so it reports zero overlap whether or
    not the hardware overlaps."""
    o = eta_from_timing_block("part", _record(partitioned=True, overlap=6))
    assert o.eta is None and "partition" in o.detail


def test_a_block_naming_an_unread_unit_refuses():
    o = eta_from_timing_block("gap", _record(partitioned=False, overlap=6, unmeasured=("vec",)))
    assert o.eta is None and "vec" in o.detail


def test_an_unstated_alias_count_refuses_rather_than_reading_zero():
    """A limit found in our own harness is evidence about the harness -- but an unstated collision
    count means nobody can say whether these cycles are about the program submitted."""
    rec = _record(partitioned=False, overlap=6)
    del rec["timing_capability"]["alias_collisions"]
    o = eta_from_timing_block("noalias", rec)
    assert o.eta is None and "collided" in o.detail


def test_the_block_path_fires_too():
    base = eta_from_timing_block("base", _record(partitioned=False, overlap=2), work="w")
    cand = eta_from_timing_block("cand", _record(partitioned=False, overlap=2), work="w")
    assert ab_decision(base, cand, bit_exact=True, invariants_held=True).state == REJECT


# -------------------------------------------------------------------------------------------------
# 6. the A/B gate's other two inputs, and the tri-state that runs through all of them
# -------------------------------------------------------------------------------------------------
def test_a_candidate_that_changed_the_answer_is_rejected():
    d = ab_decision(_obs("s", SERIAL), _obs("o", OVERLAPPED), bit_exact=False, invariants_held=True)
    assert d.state == REJECT and "did not reproduce" in d.reason


def test_an_unchecked_correctness_answer_does_not_promote():
    d = ab_decision(_obs("s", SERIAL), _obs("o", OVERLAPPED), bit_exact=None, invariants_held=True)
    assert d.state == UNDETERMINABLE


def test_unchecked_phase_f_invariants_do_not_promote():
    """No default of True. A Phase-P candidate whose functional invariants nobody re-proved is
    undeterminable, and defaulting them to held is how an unmeasured thing becomes a measured pass."""
    d = ab_decision(_obs("s", SERIAL), _obs("o", OVERLAPPED), bit_exact=True)
    assert d.state == UNDETERMINABLE and "not checked" in d.reason
    assert ab_decision(_obs("s", SERIAL), _obs("o", OVERLAPPED), bit_exact=True,
                       invariants_held=False).state == REJECT


def test_the_three_states_are_three_distinct_values():
    assert len({ROSE, DID_NOT_RISE, UNDETERMINABLE}) == 3
    assert len({ACCEPT, REJECT, UNDETERMINABLE}) == 3


def test_a_tolerance_is_never_invented_for_the_caller():
    """The default is 0.0 because eta is a ratio of integer cycle counts off a deterministic trace.
    An instrument that samples has a noise floor, and supplying it is the instrument's job."""
    base, cand = _obs("s", SERIAL), _obs("o", OVERLAPPED)
    assert compare_eta(base, cand).state == ROSE
    assert compare_eta(base, cand, tolerance=0.75).state == DID_NOT_RISE
