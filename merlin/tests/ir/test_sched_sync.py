"""Synchronisation: what the IR can say about it, and what each machine makes of the same kernel.

The point this file exists to pin is that ONE kernel gets DIFFERENT verdicts on different machines, with
no instruction changed. Where hardware tracks dependencies, a missing wait is not a defect and no
correctness check can refute a reordering. Where the compiler separates hazards, the identical omission
returns wrong data at full speed. A checker that asked one question would be wrong on half the corpus,
and wrong in the dangerous direction on the half it passed.
"""

from __future__ import annotations

import pytest

from merlin.sched.check.sync import check_sync
from merlin.sched.ir import Kernel, TensorArg, call, loop
from merlin.sched.ir.kernel import check_structure
from merlin.sched.mach import Latency, Machine, Sync, Unit, Unknown
from merlin.sched.primitives import Cursor, NotApplicable, PrimitiveError, pipeline, place, proof_of

pytestmark = pytest.mark.target("gemmini", "atlas", "muon")

ARG = (TensorArg("a", (8,), "i8", "read"),)
UNITS = (Unit(name="dma", kind="dma", queue="q"), Unit(name="mesh", kind="systolic", queue="q"))
COSTS = (
    #: A bulk transfer whose duration is data-dependent: a consumer polls for it.
    Latency(instr="ld", unit="dma", issue=1, result=33, completion="polled", source="test"),
    #: An instruction whose result is visible to the next one: nothing to wait for.
    Latency(instr="mm", unit="mesh", issue=1, result=None, completion="immediate", source="test"),
)


def _machine(hazards: str | None) -> Machine:
    unknowns = () if hazards else (Unknown("hazard_resolution", "not declared by this target"),)
    return Machine(
        target="t",
        hazard_resolution=hazards,
        units=UNITS,
        latencies=COSTS,
        delay_instruction="DELAY",
        unknowns=unknowns,
    )


def _placed() -> Kernel:
    k = Kernel("k", ARG, (call("ld"), call("mm")))
    k = place(k, Cursor(index=0), unit="dma", machine=_machine("explicit"))
    return place(k, Cursor(index=1), unit="mesh", machine=_machine("explicit"))


# -- what the IR can now say ---------------------------------------------------------------------


def test_a_synchronous_kernel_prints_and_digests_exactly_as_before():
    """The asynchrony decorates a call; it does not replace it.

    If adding these fields changed the text of a kernel that uses none of them, every recorded schedule
    digest in the repo would move and every measurement keyed on one would be invalidated.
    """
    plain = Kernel("k", ARG, (call("mm", n=1),))
    assert plain.text() == "kernel k(a: i8[8] read)\n  mm(n=1)\n"


def test_the_text_form_carries_where_it_runs_and_what_it_waits_for():
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("mm", unit="mesh", awaits="t0")))
    assert "t0 = ld() on dma" in k.text()
    assert "mm() on mesh after t0" in k.text()
    assert check_structure(k) == []


@pytest.mark.parametrize(
    "body,expect",
    [
        ((call("mm", awaits="t9"),), "which no earlier call produces"),
        ((call("ld", produces="t0"), call("ld2", produces="t0")), "already a live token"),
    ],
)
def test_token_structure_is_checked_without_a_machine(body, expect):
    """Target-free: a token awaited before it is produced is wrong on any hardware."""
    problems = check_structure(Kernel("k", ARG, body))
    assert any(expect in p for p in problems), problems


def test_whether_an_unawaited_token_is_a_defect_is_NOT_asked_here():
    """It depends on the hardware, so the target-free checker must not answer it.

    Answering it here would make the structural check wrong on one archetype -- and silently, since it
    would simply pass or fail everything of that shape.
    """
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("mm", unit="mesh")))
    assert check_structure(k) == [], "the target-free check took a position that belongs to the machine"


# -- the archetype split -------------------------------------------------------------------------


def test_the_same_kernel_gets_opposite_verdicts_on_the_two_archetypes():
    """THE test. One kernel, no instruction changed, two machines, two answers."""
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("mm", unit="mesh")))

    interlocked = check_sync(k, _machine("interlocked"))
    assert interlocked.ok, "hardware that tracks dependencies does not need the compiler's wait"
    assert "cannot refute a reordering" in interlocked.reason

    explicit = check_sync(k, _machine("explicit"))
    assert not explicit.ok, "a machine where the compiler separates hazards accepted a missing wait"
    assert any("nothing awaits" in p for p in explicit.problems)
    assert any("wrong answer at full speed" in p for p in explicit.problems)


def test_a_machine_with_no_hazard_model_answers_neither():
    """Undecidable is a third state, and its empty problem list does NOT mean nothing is wrong.

    Reading `problems == []` as clean is exactly the mistake: `decidable` is the field that says whether
    the question was asked at all.
    """
    report = check_sync(Kernel("k", ARG, (call("ld", unit="dma", produces="t0"),)), _machine(None))
    assert not report.decidable and not report.ok
    assert report.problems == ()
    assert "cannot be answered" in report.reason
    assert "accept every illegal schedule" in report.reason


def test_a_call_placed_on_a_unit_the_machine_lacks_is_wrong_on_both_archetypes():
    k = Kernel("k", ARG, (call("ld", unit="nonexistent"),))
    for hazards in ("interlocked", "explicit"):
        report = check_sync(k, _machine(hazards))
        assert any("does not have" in p for p in report.problems), hazards


def test_the_checker_reports_how_much_it_looked_at():
    """A report with no problems over zero calls is not a clean kernel; it is an empty one."""
    report = check_sync(Kernel("k", ARG, (loop("i", 4, call("mm", unit="mesh")),)), _machine("explicit"))
    assert report.checked == 1, "calls inside loops were not examined"
    assert check_sync(Kernel("k", ARG, ()), _machine("explicit")).checked == 0


# -- the primitives ------------------------------------------------------------------------------


@pytest.mark.parametrize("primitive", (place, pipeline), ids=lambda p: p.__name__)
def test_each_declares_its_obligation(primitive):
    assert proof_of(primitive) == "bounded_check"


def test_place_binds_a_call_to_a_unit_and_refuses_one_the_machine_lacks():
    machine = _machine("explicit")
    k = place(Kernel("k", ARG, (call("ld"),)), Cursor(index=0), unit="dma", machine=machine)
    assert "on dma" in k.text()
    with pytest.raises(PrimitiveError, match="no unit named"):
        place(k, Cursor(index=0), unit="nope", machine=machine)


def test_pipeline_records_the_dependence_on_both_ends_or_neither():
    """A producer with a token nothing awaits is the state the explicit-machine check rejects.

    Splitting this into two primitives would make that state reachable one call at a time, so the
    kernel between the two calls would be one the checker calls wrong.
    """
    machine = _machine("explicit")
    before = _placed()
    assert check_sync(before, machine).ok
    after = pipeline(before, Cursor(index=0), Cursor(index=1), token="t0", machine=machine)
    assert "t0 = ld() on dma" in after.text() and "after t0" in after.text()
    assert check_sync(after, machine).ok, "the edge this inserted is one the machine rejects"
    assert after.digest() != before.digest()


def test_pipeline_refuses_an_instruction_with_nothing_to_wait_for():
    """An instruction whose result is visible to the next one has no completion.

    Inventing a token for it would record a dependence the hardware never has -- a constraint on every
    later reordering, bought for nothing.
    """
    machine = _machine("explicit")
    k = _placed()
    # mm completes immediately; put it first so the ordering check is not what refuses.
    swapped = Kernel("k", ARG, (k.body[1], k.body[0]))
    with pytest.raises(NotApplicable, match="completes immediately"):
        pipeline(swapped, Cursor(index=0), Cursor(index=1), token="t0", machine=machine)


def test_pipeline_refuses_when_the_machine_declares_no_cost():
    """A token invented here would ASSERT the producer is asynchronous, which nothing derived."""
    machine = Machine(target="t", hazard_resolution="explicit", units=UNITS)
    with pytest.raises(NotApplicable, match="declares no cost"):
        pipeline(_placed(), Cursor(index=0), Cursor(index=1), token="t0", machine=machine)


@pytest.mark.parametrize(
    "producer,consumer,expect",
    [
        (1, 0, "does not follow"),
        (0, 0, "does not follow"),
    ],
)
def test_pipeline_refuses_a_consumer_that_does_not_follow_its_producer(producer, consumer, expect):
    with pytest.raises(NotApplicable, match=expect):
        pipeline(_placed(), Cursor(index=producer), Cursor(index=consumer), token="t", machine=_machine("explicit"))


def test_pipeline_refuses_to_reuse_a_live_token():
    machine = _machine("explicit")
    once = pipeline(_placed(), Cursor(index=0), Cursor(index=1), token="t0", machine=machine)
    with pytest.raises(PrimitiveError, match="already produced"):
        pipeline(once, Cursor(index=0), Cursor(index=1), token="t0", machine=machine)


def test_an_unplaced_producer_has_nothing_to_overlap_with():
    k = Kernel("k", ARG, (call("ld"), call("mm")))
    with pytest.raises(NotApplicable, match="not placed on a unit"):
        pipeline(k, Cursor(index=0), Cursor(index=1), token="t0", machine=_machine("explicit"))


# -- distinct sync kinds --------------------------------------------------------------------------

SYNCS = (
    Sync(instr="drain", orders="completion", source="test"),
    Sync(instr="ready", orders="issue", source="test"),
    Sync(instr="fence_s", orders="visibility", scope="smem", source="test"),
    Sync(instr="barrier", orders="arrival", scope="warps", source="test"),
    Sync(instr="drain_to", orders="completion", depth=2, source="test"),
)


def _with_syncs(hazards: str) -> Machine:
    return Machine(
        target="t",
        hazard_resolution=hazards,
        units=UNITS,
        latencies=COSTS,
        delay_instruction="DELAY",
        syncs=SYNCS,
    )


def test_only_a_completion_wait_discharges_a_dependence():
    """The distinction one `fence` concept cannot express, and therefore cannot check."""
    assert [s.instr for s in SYNCS if s.discharges_dependence] == ["drain", "drain_to"]
    for instr in ("ready", "fence_s", "barrier"):
        declared = next(s for s in SYNCS if s.instr == instr)
        assert not declared.discharges_dependence, f"{instr} would be accepted as establishing a result"


@pytest.mark.parametrize("hazards", ("interlocked", "explicit"))
@pytest.mark.parametrize(
    "waiter,ok", [("drain", True), ("drain_to", True), ("ready", False), ("fence_s", False), ("barrier", False)]
)
def test_the_wrong_kind_of_wait_is_caught_on_both_archetypes(hazards, waiter, ok):
    """Wrong on BOTH, so it is checked before the archetype split.

    Hardware that tracks its own dependencies still cannot know that the compiler intended a
    backpressure poll to mean completion.
    """
    machine = _with_syncs(hazards)
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call(waiter, awaits="t0"), call("mm", unit="mesh")))
    report = check_sync(k, machine)
    wrong_kind = [p for p in report.problems if "not completion" in p]
    assert (not wrong_kind) == ok, f"{waiter} on {hazards}: {report.problems}"


def test_a_machine_declaring_no_sync_instructions_makes_no_claim():
    """An empty declaration is a statement about the DECLARATION, not about the hardware.

    Reporting every wait as suspect where a target has declared nothing would bury the real findings,
    and inventing a default fence would assert a primitive the target may not have.
    """
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("ready", awaits="t0")))
    assert not [p for p in check_sync(k, _machine("explicit")).problems if "not completion" in p]


def test_a_machine_declaring_no_sync_instructions_says_the_question_went_unasked():
    """ "No claim" has to be visible in the report, or it reads as a clean bill of health.

    Measured 2026-09-18: NO target populates `Machine.syncs` -- `mach.derive` never sets the field --
    so this class of check returns immediately on every one of them. An empty problem list then says
    "nothing wrong with the waits" when the truth is "nobody declared what a wait orders". The gap is
    not that the check is wrong; it is that its silence was indistinguishable from a pass.
    """
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("ready", awaits="t0")))
    report = check_sync(k, _machine("explicit"))
    assert report.unchecked, "a whole class of check was skipped and the report did not say so"
    assert "Machine.syncs is empty" in report.unchecked[0]


def test_a_machine_that_declares_its_sync_instructions_has_nothing_unchecked():
    """The control: without it the test above would pass for a report that always cries unchecked."""
    k = Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("ready", awaits="t0")))
    assert check_sync(k, _with_syncs("explicit")).unchecked == ()


def test_no_target_in_the_tree_declares_a_sync_instruction():
    """The measurement behind the two tests above, pinned so it is re-taken rather than remembered.

    When a target does declare its synchronisation instructions this goes red, and the right response
    is to delete it -- the gap it records will have closed.
    """
    from merlin.sched.mach.derive import derive

    declared = {t: len(derive(t).syncs) for t in ("gemmini", "atlas", "muon", "radiance", "toy_npu")}
    assert set(declared.values()) == {0}, (
        f"a target now declares synchronisation instructions ({declared}); the sync check is no longer "
        "inert there, so this test and the gap it records should go"
    )


def test_a_partial_drain_declares_the_depth_it_drains_to():
    """A drain to depth n is not a full drain, and a model that could not say so would let one stand
    in for the other."""
    partial = next(s for s in SYNCS if s.instr == "drain_to")
    assert partial.depth == 2 and partial.discharges_dependence
    assert next(s for s in SYNCS if s.instr == "drain").depth is None


def test_the_sync_vocabulary_is_closed_and_the_discharging_set_is_one():
    """Widening `DISCHARGES_DEPENDENCE` is the single edit that would make the check above vacuous."""
    from merlin.sched.mach import DISCHARGES_DEPENDENCE, SYNC_ORDERS

    assert set(DISCHARGES_DEPENDENCE) == {"completion"}
    assert set(SYNC_ORDERS) >= {"completion", "issue", "visibility", "arrival"}
    with pytest.raises(Exception):
        Sync(instr="x", orders="fence")
