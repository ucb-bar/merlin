"""Pricing a schedule, and refusing to price one that cannot be priced.

The failure this file exists against is not a wrong number. It is a number of the wrong KIND: a partial
sum over the instructions that happen to have declared costs, reported as a cost. That ranks a schedule
built from instructions nobody has measured as the cheapest one available, which is exactly backwards,
and it does so silently because a number invites comparison.
"""

from __future__ import annotations

import dataclasses

import pytest

from merlin.sched.cost import KINDS, Bound, price
from merlin.sched.ir import Kernel, TensorArg, call, loop
from merlin.sched.mach import Latency, Machine, Unit
from merlin.sched.primitives import Cursor, NotApplicable, pipeline, place

pytestmark = pytest.mark.target("gemmini", "atlas", "muon")

ARG = (TensorArg("a", (8,), "i8", "read"),)
UNITS = (Unit(name="dma", kind="dma", queue="q"), Unit(name="mesh", kind="systolic", queue="q"))
EXACT = (
    Latency(instr="ld", unit="dma", issue=2, result=30, completion="polled", source="test"),
    Latency(instr="mm", unit="mesh", issue=3, result=10, completion="polled", source="test"),
)


def _machine(latencies=EXACT, units=UNITS) -> Machine:
    return Machine(target="t", hazard_resolution="explicit", units=units, latencies=latencies, delay_instruction="D")


def _pipelined() -> Kernel:
    return Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("mm", unit="mesh", awaits="t0")))


# -- the three kinds -----------------------------------------------------------------------------


def test_a_fully_declared_schedule_prices_exactly():
    bound = price(_pipelined(), _machine())
    assert bound.kind == "exact" and bound.cycles == 33
    assert not bound.missing and not bound.floors


def test_a_floor_measured_on_an_idle_machine_can_only_bound_from_below():
    """Contention can make a schedule slower and never faster, so a floor is not a prediction."""
    contended = tuple(dataclasses.replace(lat, contended=True) for lat in EXACT)
    bound = price(_pipelined(), _machine(contended))
    assert bound.kind == "lower_bound" and bound.cycles == 33
    assert set(bound.floors) == {("ld", "dma"), ("mm", "mesh")}
    assert "not a prediction" in bound.notes[0]


def test_an_underived_cost_makes_the_whole_schedule_undecidable():
    """THE test. A partial sum would be smaller than the truth by an unknown amount.

    The alternative -- skip what is not declared -- ranks a schedule of unmeasured instructions as the
    cheapest available, and does it silently.
    """
    bound = price(_pipelined(), _machine(EXACT[:1]))
    assert bound.kind == "undecidable"
    assert bound.cycles is None, "a partial sum was reported as a cost"
    assert ("mm", "mesh") in bound.missing
    assert "unbounded" in bound.notes[0]


def test_an_unplaced_call_is_undecidable_not_free():
    """A call on no unit has no cost on any unit; treating it as zero is the same error."""
    bound = price(Kernel("k", ARG, (call("ld"),)), _machine())
    assert bound.kind == "undecidable" and bound.cycles is None
    assert ("ld", "<unplaced>") in bound.missing


def test_a_bound_cannot_be_constructed_inconsistently():
    """The invariant that keeps the kind honest: a decidable cost has a number, an undecidable one
    does not. Without it the dataclass would allow `undecidable` with a number attached, which is the
    partial sum wearing a label."""
    with pytest.raises(ValueError, match="a decidable cost has a number"):
        Bound(kind="exact", cycles=None)
    with pytest.raises(ValueError, match="a decidable cost has a number"):
        Bound(kind="undecidable", cycles=10)
    with pytest.raises(ValueError, match="not in"):
        Bound(kind="probably_fine", cycles=1)


def test_the_kinds_are_ordered_weakest_first_and_only_like_compares_with_like():
    assert KINDS.index("undecidable") < KINDS.index("lower_bound") < KINDS.index("exact")
    assert price(_pipelined(), _machine()).comparable_to == "exact"


# -- what the model actually models ---------------------------------------------------------------


def test_issue_is_serial_per_queue_and_execution_is_not():
    """Units behind one in-order command port take each other's issue slots, and still overlap.

    That split is why a unit separates where it ISSUES from what it EXECUTES on; a model that summed
    execution would price the decoupled-controller overlap -- the entire lever on one target -- at zero.
    """
    one_queue = price(_pipelined(), _machine()).cycles
    two_queues = price(
        _pipelined(),
        _machine(units=(Unit(name="dma", kind="dma", queue="qa"), Unit(name="mesh", kind="systolic", queue="qb"))),
    ).cycles
    assert one_queue == two_queues == 33, "the token edge, not the queue, is what orders these two"

    # Two independent calls on one queue serialise their issue; on two queues they do not.
    independent = Kernel("k", ARG, (call("ld", unit="dma"), call("mm", unit="mesh")))
    serial = price(independent, _machine()).cycles
    parallel = price(
        independent,
        _machine(units=(Unit(name="dma", kind="dma", queue="qa"), Unit(name="mesh", kind="systolic", queue="qb"))),
    ).cycles
    assert serial == 5 and parallel == 3, f"issue sharing is not modelled: {serial} vs {parallel}"


def test_a_consumer_cannot_issue_before_the_result_it_awaits():
    """The token edge is what the price is built on, so removing it must make the schedule cheaper."""
    with_edge = price(_pipelined(), _machine()).cycles
    without = price(
        Kernel("k", ARG, (call("ld", unit="dma", produces="t0"), call("mm", unit="mesh"))), _machine()
    ).cycles
    assert without < with_edge, "the awaited result did not delay the consumer"


def test_a_loop_is_priced_once_per_iteration():
    body = Kernel("k", ARG, (loop("i", 4, call("mm", unit="mesh")),))
    assert price(body, _machine()).cycles == 4 * 3


def test_an_empty_kernel_is_exactly_zero_and_says_so():
    """Distinct from undecidable: nothing to do is a cost, not an absence of one."""
    bound = price(Kernel("k", ARG, ()), _machine())
    assert bound.kind == "exact" and bound.cycles == 0 and bound.notes


# -- contention is load-bearing in the primitive ---------------------------------------------------


def test_pipeline_refuses_an_overlap_the_machine_says_is_impossible():
    """Two units that declare one `executes_on` are one resource whatever their names.

    Asking the machine rather than comparing names is the point: the edge would record a dependence and
    buy no overlap, costing every later reordering a constraint for nothing.
    """
    shared = (
        Unit(name="dma", kind="dma", queue="q", executes_on="core"),
        Unit(name="mesh", kind="systolic", queue="q", executes_on="core"),
    )
    machine = _machine(units=shared)
    k = place(
        place(Kernel("k", ARG, (call("ld"), call("mm"))), Cursor(index=0), unit="dma", machine=machine),
        Cursor(index=1),
        unit="mesh",
        machine=machine,
    )
    with pytest.raises(NotApplicable, match="cannot be in flight at once"):
        pipeline(k, Cursor(index=0), Cursor(index=1), token="t0", machine=machine)


def test_pipeline_still_allows_an_overlap_the_machine_permits():
    """The control: without it the test above passes for a primitive that refuses everything."""
    machine = _machine()
    k = place(
        place(Kernel("k", ARG, (call("ld"), call("mm"))), Cursor(index=0), unit="dma", machine=machine),
        Cursor(index=1),
        unit="mesh",
        machine=machine,
    )
    assert "after t0" in pipeline(k, Cursor(index=0), Cursor(index=1), token="t0", machine=machine).text()
