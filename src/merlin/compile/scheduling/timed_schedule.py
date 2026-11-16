"""Check finite timed decisions over the existing Kernel/Call representation.

Targets project actual operands to complete byte effects and resource intervals.
All intervals are half open in a selected clock. Unknown timing is not zero.
The checker preserves source memory dependences, including WAR/WAW, and counts
physical port demand even between readers. It neither derives target facts nor
establishes their independent qualification. Loops/DMA with unknown completion
retain their separate event-based paths.
"""

from collections import defaultdict
from dataclasses import dataclass

from merlin.sched.ir import Call, Kernel
from merlin.sched.ir.kernel import check_structure

from .memory_effects import EncodedAccess


def _interval(begin, end):
    if type(begin) is not int or type(end) is not int or not 0 <= begin < end:
        raise ValueError("timed effects need nonempty nonnegative integer intervals")


@dataclass(frozen=True)
class TimedAccess:
    access: EncodedAccess
    begin: int
    end: int

    def __post_init__(self):
        if type(self.access) is not EncodedAccess or type(self.access.writes) is not bool:
            raise ValueError("timed access needs an exact decoded byte access")
        self.access.__post_init__()
        _interval(self.begin, self.end)


@dataclass(frozen=True)
class ResourceUse:
    resource: str
    begin: int
    end: int
    amount: int = 1

    def __post_init__(self):
        if not isinstance(self.resource, str) or not self.resource or type(self.amount) is not int or self.amount <= 0:
            raise ValueError("resource use requires a named resource and positive demand")
        _interval(self.begin, self.end)


@dataclass(frozen=True)
class TimedEffects:
    semantic_key: str
    complete: int
    accesses: tuple[TimedAccess, ...] = ()
    resources: tuple[ResourceUse, ...] = ()

    def __post_init__(self):
        if (
            not isinstance(self.semantic_key, str)
            or not self.semantic_key
            or type(self.complete) is not int
            or self.complete <= 0
            or type(self.accesses) is not tuple
            or type(self.resources) is not tuple
            or any(type(a) is not TimedAccess for a in self.accesses)
            or any(type(r) is not ResourceUse for r in self.resources)
            or any(a.end > self.complete for a in (*self.accesses, *self.resources))
        ):
            raise ValueError("timing projection requires complete known effects and completion")


def _effects(kernel, project, max_calls):
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or not callable(project)
        or type(max_calls) is not int
        or not 0 < len(kernel.body) <= max_calls
        or any(
            type(c) is not Call
            or c.produces is not None
            or c.awaits
            or c.unit is not None
            or c.stages
            or c.sets
            or c.assumes
            for c in kernel.body
        )
    ):
        raise ValueError("timed checking requires a bounded plain straight-line kernel")
    result = tuple(project(call) for call in kernel.body)
    if any(type(effect) is not TimedEffects for effect in result):
        raise ValueError("unknown timing or unprojected instruction")
    return result


def _resource_conflict(uses, capacities):
    """Shared interval-capacity accounting; unknown resources are never free."""
    events = defaultdict(lambda: defaultdict(int))
    for resource, begin, end, amount in uses:
        if resource not in capacities:
            raise ValueError(f"unknown capacity for {resource}")
        events[resource][begin] += amount
        events[resource][end] -= amount
    for resource, changes in events.items():
        live = 0
        for tick in sorted(changes):
            live += changes[tick]
            if live > capacities[resource]:
                return resource, tick
    return None


def verify_timed_block(kernel, issue_cycles, *, project, capacities, max_calls=1024):
    """Check current actual effects, source dependences and shared resource use.

    Calls may interleave or reorder only where their projected effects permit.
    Targets must include issue ports, hidden state and asynchronous operand
    lifetimes; the absence of a decoded effect is not evidence of independence.
    """
    effects = _effects(kernel, project, max_calls)
    if (
        type(issue_cycles) is not tuple
        or len(issue_cycles) != len(effects)
        or any(type(t) is not int or t < 0 for t in issue_cycles)
        or type(capacities) is not dict
        or any(type(k) is not str or not k or type(v) is not int or v <= 0 for k, v in capacities.items())
    ):
        raise ValueError("timed checking requires one nonnegative issue cycle per call and explicit capacities")
    for i, earlier in enumerate(effects):
        for j in range(i + 1, len(effects)):
            for first in earlier.accesses:
                a = first.access
                for second in effects[j].accesses:
                    b = second.access
                    if (
                        a.space == b.space
                        and a.start < b.start + b.size
                        and b.start < a.start + a.size
                        and (a.writes or b.writes)
                        and issue_cycles[i] + first.end > issue_cycles[j] + second.begin
                    ):
                        raise ValueError(f"timed source memory dependence violated between calls {i} and {j}")
    conflict = _resource_conflict(
        (
            (use.resource, cycle + use.begin, cycle + use.end, use.amount)
            for cycle, effect in zip(issue_cycles, effects, strict=True)
            for use in effect.resources
        ),
        capacities,
    )
    if conflict is not None:
        resource, tick = conflict
        raise ValueError(f"resource {resource} over capacity at cycle {tick}")
    return tuple(cycle + effect.complete for cycle, effect in zip(issue_cycles, effects))


def _verify_plan(kernel, plan, *, model_identity, clock, check):
    from xdsl.dialects.builtin import StringAttr

    from merlin.xdsl_dialects.schedule import BindOp, DrainOp, OverlapOp, PlanOp, TimeAnnotationOp

    if type(plan) is not PlanOp:
        raise ValueError("timing decisions require schedule.plan")
    plan.verify()
    if (
        not model_identity
        or not clock
        or len(plan.body.blocks) != 1
        or plan.attributes
        != {
            "kernel_sha256": StringAttr(kernel.digest()),
            "timing_model": StringAttr(model_identity),
        }
    ):
        raise ValueError("timing decisions differ from the current kernel or selected model")
    handles, issues, completions, overlap = {}, {}, {}, {}
    drain = None
    targets = {f"call:{index}": index for index in range(len(kernel.body))}
    for operation in plan.body.block.ops:
        if operation.attributes:
            raise ValueError("unknown timing decision attributes")
        if type(operation) is BindOp:
            if operation.target.data not in targets or targets[operation.target.data] in handles.values():
                raise ValueError("timing binding is unknown or repeated")
            handles[operation.handle] = targets[operation.target.data]
        elif type(operation) is TimeAnnotationOp:
            if operation.target not in handles or operation.clock.data != clock:
                raise ValueError("time annotation has an unbound target or different clock")
            target = handles[operation.target]
            annotations = issues if operation.event.data == "issue" else completions
            if target in annotations:
                raise ValueError("repeated timing annotation")
            annotations[target] = operation.cycle.value.data
        elif type(operation) is DrainOp:
            if drain is not None or operation.clock.data != clock:
                raise ValueError("repeated drain or different clock")
            drain = operation.cycle.value.data
        elif type(operation) is OverlapOp:
            if operation.first not in handles or operation.second not in handles:
                raise ValueError("overlap has an unbound target")
            pair = tuple(sorted((handles[operation.first], handles[operation.second])))
            if pair in overlap:
                raise ValueError("repeated overlap decision")
            overlap[pair] = operation.mode.data
        else:
            raise ValueError("unconsumed timing decision")
    if set(issues) != set(range(len(kernel.body))):
        raise ValueError("timing plan must assign every actual instruction an issue cycle")
    starts = tuple(issues[i] for i in range(len(kernel.body)))
    ends = check(starts)
    if drain is not None and drain < max(ends):
        raise ValueError("drain precedes outstanding completion")
    if any(ends[i] != value for i, value in completions.items()):
        raise ValueError("completion annotation differs from projected effects")
    for (i, j), mode in overlap.items():
        intersects = starts[i] < ends[j] and starts[j] < ends[i]
        if intersects != (mode == "require"):
            raise ValueError("selected overlap decision is not realized")
    return starts, ends


def verify_timing_plan(kernel, plan, *, model_identity, clock, project, capacities, max_calls=1024):
    """Consume typed decisions with complete memory and capacity projections."""
    return _verify_plan(
        kernel,
        plan,
        model_identity=model_identity,
        clock=clock,
        check=lambda starts: verify_timed_block(
            kernel, starts, project=project, capacities=capacities, max_calls=max_calls
        ),
    )


def verify_issue_timing_plan(kernel, plan, *, model_identity, clock, completion, distance, max_calls=1024):
    """Check actual instruction pair distances and fixed-latency annotations.

    This deliberately establishes only issue spacing. The target must separately
    check resource reservations, event-driven DMA completion and control flow.
    Both callbacks consume current Call operands, never candidate-supplied facts.
    ``distance`` must cover every ordered pair, including zero-distance pairs.
    """
    if (
        type(kernel) is not Kernel
        or check_structure(kernel)
        or type(max_calls) is not int
        or not 0 < len(kernel.body) <= max_calls
        or any(
            type(c) is not Call
            or c.produces is not None
            or c.awaits
            or c.unit is not None
            or c.stages
            or c.sets
            or c.assumes
            for c in kernel.body
        )
    ):
        raise ValueError("issue timing requires a bounded plain straight-line kernel")

    def check(starts):
        ages = tuple(completion(call) for call in kernel.body)
        if any(type(age) is not int or age <= 0 for age in ages):
            raise ValueError("unknown fixed-latency completion")
        for j, later in enumerate(kernel.body):
            for i, earlier in enumerate(kernel.body[:j]):
                minimum = distance(earlier, later)
                if type(minimum) is not int or minimum < 0:
                    raise ValueError("unknown or invalid pair issue distance")
                if starts[j] - starts[i] < minimum:
                    raise ValueError(f"pair issue distance violated between calls {i} and {j}")
        return tuple(start + age for start, age in zip(starts, ages))

    return _verify_plan(kernel, plan, model_identity=model_identity, clock=clock, check=check)


def make_timing_plan(kernel, starts, *, model_identity, clock, overlaps=(), completions=None, drain=None):
    """Build typed, kernel-bound decisions; callers still run a selected checker."""
    from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64
    from xdsl.ir import Block, Region

    from merlin.xdsl_dialects.schedule import BindOp, DrainOp, HandleType, OverlapOp, PlanOp, TimeAnnotationOp

    if len(starts) != len(kernel.body) or any(type(t) is not int or t < 0 for t in starts):
        raise ValueError("one nonnegative issue time per instruction required")
    operations, handles = [], []
    for index, start in enumerate(starts):
        bind = BindOp(result_types=[HandleType()], properties={"target": StringAttr(f"call:{index}")})
        handles.append(bind.handle)
        operations.extend(
            (
                bind,
                TimeAnnotationOp(
                    operands=[bind.handle],
                    properties={
                        "event": StringAttr("issue"),
                        "clock": StringAttr(clock),
                        "cycle": IntegerAttr(start, i64),
                    },
                ),
            )
        )
    if completions is not None:
        if len(completions) != len(starts) or any(type(t) is not int or t < 0 for t in completions):
            raise ValueError("one nonnegative completion cycle per instruction required")
        for handle, end in zip(handles, completions, strict=True):
            operations.append(
                TimeAnnotationOp(
                    operands=[handle],
                    properties={
                        "event": StringAttr("complete"),
                        "clock": StringAttr(clock),
                        "cycle": IntegerAttr(end, i64),
                    },
                )
            )
    if drain is not None:
        if type(drain) is not int or drain < 0:
            raise ValueError("nonnegative drain cycle required")
        operations.append(DrainOp(properties={"clock": StringAttr(clock), "cycle": IntegerAttr(drain, i64)}))
    for first, second, mode in overlaps:
        if any(type(i) is not int or not 0 <= i < len(handles) for i in (first, second)):
            raise ValueError("overlap index outside kernel")
        operations.append(OverlapOp(operands=[handles[first], handles[second]], properties={"mode": StringAttr(mode)}))
    result = PlanOp(
        properties={"sym_name": StringAttr("timed")},
        attributes={"kernel_sha256": StringAttr(kernel.digest()), "timing_model": StringAttr(model_identity)},
        regions=[Region(Block(operations))],
    )
    result.verify()
    return result
