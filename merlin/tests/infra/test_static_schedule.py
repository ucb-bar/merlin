"""Source-order scheduling across synthetic pipelines and explicit issue machines."""

import random
from dataclasses import replace

import pytest
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, i64

from merlin.compile.scheduling.memory_effects import EncodedAccess
from merlin.compile.scheduling.static_schedule import (
    materialize_timing_plan,
    schedule_fixed_block,
    verify_timed_emission,
)
from merlin.compile.scheduling.timed_schedule import ResourceUse, TimedAccess, TimedEffects, verify_timing_plan
from merlin.sched.ir import Call, Const, Kernel
from merlin.xdsl_dialects import _common, schedule


class Machine:
    def __init__(self, latency=4, occupancy=1, capacity=1, gap=1):
        self.latency, self.occupancy, self.capacity, self.gap = latency, occupancy, capacity, gap

    def project(self, call):
        if call.instr == "idle":
            return TimedEffects("idle", call.arg("cycles").value)
        if call.instr not in ("matrix", "vector"):
            raise ValueError("unknown instruction")
        a, b, out = (call.arg(n).value for n in ("a", "b", "out"))
        return TimedEffects(
            call.instr,
            self.latency,
            (
                TimedAccess(EncodedAccess("registers", a, 1), 0, 1),
                TimedAccess(EncodedAccess("registers", b, 1), 0, 1),
                TimedAccess(EncodedAccess("registers", out, 1, True), self.latency - 1, self.latency),
            ),
            (ResourceUse(call.instr, 0, self.occupancy),),
        )

    def advance(self, call):
        return call.arg("cycles").value if call.instr == "idle" else self.gap

    @property
    def options(self):
        return dict(
            project=self.project,
            capacities={"matrix": self.capacity, "vector": self.capacity},
            issue_gap=self.advance,
            model_identity="test-machine",
            clock="core",
        )


def op(kind, a, b, out):
    return Call(kind, tuple((n, Const(v)) for n, v in zip(("a", "b", "out"), (a, b, out), strict=True)))


def delay(cycles):
    return (Call("idle", (("cycles", Const(cycles)),)),)


def times(kernel, plan, machine):
    options = machine.options.copy()
    options.pop("issue_gap")
    return verify_timing_plan(kernel, plan, **options)


def execute(emission, machine):
    """Independent tick-level arithmetic simulator; no timing verifier calls."""
    values = list(range(32))
    pending, tick = [], 0
    for call in emission.kernel.body:
        # Writes become visible at their completion, then this cycle may issue.
        for end, address, value in sorted(pending):
            if end <= tick:
                values[address] = value
        pending = [v for v in pending if v[0] > tick]
        if call.instr != "idle":
            a, b, out = (call.arg(n).value for n in ("a", "b", "out"))
            result = values[a] * values[b] if call.instr == "matrix" else values[a] + values[b]
            pending.append((tick + machine.latency, out, result))
        tick += machine.advance(call)
    for end, address, value in sorted(pending):
        assert end <= tick, "returned with outstanding work"
        values[address] = value
    return values


@pytest.mark.parametrize("machine", [Machine(), Machine(7, 7), Machine(6, 2, 2, 0)])
def test_two_engines_pipeline_and_multiple_issue_execute(machine):
    kernel = Kernel(
        "compute",
        (),
        (op("matrix", 1, 2, 10), op("vector", 3, 4, 11), op("matrix", 10, 11, 12), op("vector", 12, 2, 13)),
    )
    serial = schedule_fixed_block(kernel, mode="serial", **machine.options)
    plan = schedule_fixed_block(kernel, **machine.options)
    module = _common.roundtrip(ModuleOp([plan]), schedule.get_dialect())
    plan = module.body.block.first_op
    assert max(times(kernel, plan, machine)[1]) <= max(times(kernel, serial, machine)[1])
    emission = materialize_timing_plan(kernel, plan, delay=delay, **machine.options)
    assert execute(emission, machine)[10:14] == [2, 7, 14, 16]
    assert any(type(o) is schedule.DrainOp for o in plan.body.block.ops)
    assert any(type(o) is schedule.OverlapOp for o in plan.body.block.ops)


def test_long_latency_is_not_a_per_cycle_search():
    machine = Machine(10**12, 10**12)
    kernel = Kernel("long", (), (op("matrix", 1, 2, 10), op("matrix", 3, 4, 11)))
    plan = schedule_fixed_block(kernel, **machine.options)
    assert times(kernel, plan, machine)[0] == (0, 10**12)
    assert len(materialize_timing_plan(kernel, plan, delay=delay, **machine.options).kernel.body) == 4


@pytest.mark.parametrize("change", ["short_delay", "side_effect", "dropped", "changed", "tail", "stale"])
def test_emission_mutations_are_rejected(change):
    machine = Machine()
    kernel = Kernel("mutation", (), (op("matrix", 1, 2, 10), op("vector", 10, 2, 11)))
    plan = schedule_fixed_block(kernel, **machine.options)
    if change in ("short_delay", "side_effect"):
        bad = (lambda n: delay(n - 1)) if change == "short_delay" else (lambda n: (op("vector", 1, 2, 3),))
        with pytest.raises(ValueError):
            materialize_timing_plan(kernel, plan, delay=bad, **machine.options)
        return
    emitted = materialize_timing_plan(kernel, plan, delay=delay, **machine.options)
    if change == "dropped":
        emitted = replace(emitted, source_positions=emitted.source_positions[:-1])
    elif change == "changed":
        emitted = replace(
            emitted, kernel=replace(emitted.kernel, body=(op("matrix", 3, 4, 10), *emitted.kernel.body[1:]))
        )
    elif change == "tail":
        emitted = replace(emitted, kernel=replace(emitted.kernel, body=emitted.kernel.body[:-1]))
    else:
        kernel = replace(kernel, body=(op("matrix", 0, 1, 10), *kernel.body[1:]))
    with pytest.raises(ValueError):
        verify_timed_emission(kernel, plan, emitted, **machine.options)


def test_unknowns_self_conflict_and_early_drain_refuse():
    machine = Machine()
    kernel = Kernel("k", (), (op("matrix", 1, 2, 10),))
    for override in (
        dict(project=lambda c: None),
        dict(capacities={}),
        dict(issue_gap=lambda c: None),
        dict(capacities={"matrix": 0}),
        dict(mode="guess"),
    ):
        with pytest.raises(ValueError):
            schedule_fixed_block(kernel, **{**machine.options, **override})
    plan = schedule_fixed_block(kernel, **machine.options)
    next(o for o in plan.body.block.ops if type(o) is schedule.DrainOp).properties["cycle"] = IntegerAttr(0, i64)
    with pytest.raises(ValueError, match="drain"):
        times(kernel, plan, machine)
    effect = replace(machine.project(kernel.body[0]), resources=(ResourceUse("matrix", 0, 1, 2),))
    with pytest.raises(ValueError, match="alone"):
        schedule_fixed_block(kernel, **{**machine.options, "project": lambda c: effect})


def test_delayed_reads_allow_safe_lifetime_overlap():
    kernel = Kernel("delayed", (), (Call("writer", ()), Call("reader", ())))
    effects = {
        "writer": TimedEffects("writer", 8, (TimedAccess(EncodedAccess("r", 0, 1, True), 7, 8),)),
        "reader": TimedEffects("reader", 6, (TimedAccess(EncodedAccess("r", 0, 1), 4, 5),)),
    }
    options = dict(
        project=lambda c: effects[c.instr], capacities={}, issue_gap=lambda c: 1, model_identity="delayed", clock="core"
    )
    plan = schedule_fixed_block(kernel, **options)
    options.pop("issue_gap")
    assert verify_timing_plan(kernel, plan, **options) == ((0, 4), (8, 10))


def test_earliest_source_order_schedule_matches_small_exhaustive_oracle():
    rng = random.Random(19)
    for _ in range(40):
        machine = Machine(rng.randrange(2, 8), 1, rng.randrange(1, 3), rng.randrange(2))
        calls = tuple(
            op(rng.choice(("matrix", "vector")), rng.randrange(4), rng.randrange(4), rng.randrange(4)) for _ in range(5)
        )
        kernel = Kernel("random", (), calls)
        plan = schedule_fixed_block(kernel, **machine.options)
        starts = times(kernel, plan, machine)[0]
        chosen = []
        # Independent discrete cycle occupancy and dependence computation.
        for i, call in enumerate(calls):
            earliest = chosen[-1] + machine.gap if i else 0
            for candidate in range(earliest, 100):
                good = True
                a, b, out = (call.arg(n).value for n in ("a", "b", "out"))
                active = sum(c.instr == call.instr and t == candidate for c, t in zip(calls[:i], chosen, strict=True))
                if active >= machine.capacity:
                    good = False
                for earlier, t in zip(calls[:i], chosen, strict=True):
                    x, y, dst = (earlier.arg(n).value for n in ("a", "b", "out"))
                    if dst in (a, b) and candidate < t + machine.latency:
                        good = False
                    if dst == out and candidate < t + 1:
                        good = False
                    if out in (x, y) and candidate + machine.latency - 1 < t + 1:
                        good = False
                if good:
                    chosen.append(candidate)
                    break
            else:
                pytest.fail("no independent oracle schedule")
        assert starts == tuple(chosen)


def test_resource_intervals_match_independent_tick_occupancy():
    rng = random.Random(43)
    for _ in range(60):
        capacity = rng.randrange(1, 4)
        effects = []
        for i in range(5):
            begin = rng.randrange(4)
            duration = rng.randrange(1, 5)
            effects.append(
                TimedEffects(
                    str(i),
                    begin + duration,
                    resources=(ResourceUse("port", begin, begin + duration, rng.randrange(1, capacity + 1)),),
                )
            )
        calls = tuple(Call(str(i), ()) for i in range(5))
        kernel = Kernel("occupancy", (), calls)

        def project(c):
            return effects[int(c.instr)]

        options = dict(
            project=project, capacities={"port": capacity}, issue_gap=lambda c: 1, model_identity="ports", clock="core"
        )
        plan = schedule_fixed_block(kernel, **options)
        expected, occupied = [], {}
        for effect in effects:
            use = effect.resources[0]
            start = expected[-1] + 1 if expected else 0
            while any(occupied.get(t, 0) + use.amount > capacity for t in range(start + use.begin, start + use.end)):
                start += 1
            expected.append(start)
            for t in range(start + use.begin, start + use.end):
                occupied[t] = occupied.get(t, 0) + use.amount
        options.pop("issue_gap")
        assert verify_timing_plan(kernel, plan, **options)[0] == tuple(expected)


def test_padding_with_hidden_effects_refuses_even_when_cycle_count_matches():
    machine = Machine()
    kernel = Kernel("padding", (), (op("matrix", 1, 2, 10),))
    plan = schedule_fixed_block(kernel, **machine.options)

    def bad_project(call):
        effect = machine.project(call)
        return (
            replace(effect, accesses=(TimedAccess(EncodedAccess("hidden", 0, 1, True), 0, 1),))
            if call.instr == "idle"
            else effect
        )

    with pytest.raises(ValueError, match="delay has effects"):
        materialize_timing_plan(kernel, plan, delay=delay, **{**machine.options, "project": bad_project})


def test_missing_repeated_and_wrong_clock_drains_refuse():
    from xdsl.dialects.builtin import StringAttr

    machine = Machine()
    kernel = Kernel("boundary", (), (op("matrix", 1, 2, 10),))
    for mutation in ("missing", "repeated", "clock"):
        plan = schedule_fixed_block(kernel, **machine.options)
        drain = next(o for o in plan.body.block.ops if type(o) is schedule.DrainOp)
        if mutation == "missing":
            plan.body.block.erase_op(drain)
        elif mutation == "repeated":
            plan.body.block.add_op(drain.clone())
        else:
            drain.properties["clock"] = StringAttr("unrelated-clock")
        with pytest.raises(ValueError, match="drain|clock"):
            materialize_timing_plan(kernel, plan, delay=delay, **machine.options)
