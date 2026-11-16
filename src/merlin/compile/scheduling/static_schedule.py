"""Construct and realize drained fixed-latency schedules from target effects.

Source issue order is retained; independent lifetimes may overlap. Targets supply
all timing, resource facts and delay encodings. Event waits and control transfers
are boundaries, never ordinary instructions with guessed completion times.
"""

from dataclasses import dataclass, replace

from merlin.sched.ir import Call, Kernel
from merlin.xdsl_dialects.schedule import DrainOp

from .timed_schedule import _effects, _resource_conflict, make_timing_plan, verify_timed_block, verify_timing_plan


def _advances(kernel, issue_gap):
    gaps = tuple(issue_gap(call) for call in kernel.body)
    if any(type(gap) is not int or gap < 0 for gap in gaps):
        raise ValueError("unknown or negative natural issue advance")
    return gaps


def _uses(effect, start):
    return tuple((r.resource, start + r.begin, start + r.end, r.amount) for r in effect.resources)


def schedule_fixed_block(
    kernel, *, project, capacities, issue_gap, model_identity, clock, mode="overlap", max_calls=1024
):
    """Return verified issue/completion/overlap decisions and an explicit drain.

    The caller proves drained entry and supplies complete physical effects.
    ``serial`` completes each instruction before the next. ``overlap`` retains
    issue order but places independent work at resource-feasible times.
    """
    if mode not in ("serial", "overlap"):
        raise ValueError("select serial or overlap scheduling")
    effects = _effects(kernel, project, max_calls)
    gaps = _advances(kernel, issue_gap)
    if type(capacities) is not dict or any(
        type(k) is not str or not k or type(v) is not int or v <= 0 for k, v in capacities.items()
    ):
        raise ValueError("explicit positive resource capacities required")
    starts, reserved = [], []
    for i, effect in enumerate(effects):
        if _resource_conflict(_uses(effect, 0), capacities) is not None:
            raise ValueError("instruction alone exceeds resource capacity")
        lower = starts[-1] + gaps[i - 1] if i else 0
        if mode == "serial" and i:
            lower = max(lower, starts[-1] + effects[i - 1].complete)
        for j, earlier in enumerate(effects[:i]):
            for a in earlier.accesses:
                for b in effect.accesses:
                    x, y = a.access, b.access
                    if (
                        x.space == y.space
                        and x.start < y.start + y.size
                        and y.start < x.start + x.size
                        and (x.writes or y.writes)
                    ):
                        lower = max(lower, starts[j] + a.end - b.begin)
        # Feasibility can improve only when an earlier reservation ends at the
        # beginning of one of the new reservations. No per-cycle search bound.
        candidates = {lower}
        for resource, _begin, end, _amount in reserved:
            for use in effect.resources:
                if resource == use.resource and end - use.begin >= lower:
                    candidates.add(end - use.begin)
        start = next(
            (t for t in sorted(candidates) if _resource_conflict((*reserved, *_uses(effect, t)), capacities) is None),
            None,
        )
        if start is None:
            raise ValueError("no resource-feasible source-order schedule")
        starts.append(start)
        reserved.extend(_uses(effect, start))
    ends = tuple(t + effect.complete for t, effect in zip(starts, effects, strict=True))
    overlaps = tuple((i, i + 1, "require" if ends[i] > starts[i + 1] else "forbid") for i in range(len(starts) - 1))
    drain = max(*ends, starts[-1] + gaps[-1])
    plan = make_timing_plan(
        kernel, starts, model_identity=model_identity, clock=clock, overlaps=overlaps, completions=ends, drain=drain
    )
    verify_timing_plan(
        kernel,
        plan,
        model_identity=model_identity,
        clock=clock,
        project=project,
        capacities=capacities,
        max_calls=max_calls,
    )
    return plan


@dataclass(frozen=True)
class TimedEmission:
    """Actual target calls and positions of the unchanged source instructions."""

    kernel: Kernel
    source_positions: tuple[int, ...]


def verify_timed_emission(
    source, plan, emission, *, project, capacities, issue_gap, model_identity, clock, max_calls=8192
):
    """Recompute issues from actual calls; padding cannot certify its own delay.

    Final binary decoding must recover this stream before a hardware verdict.
    This function establishes correspondence to calls, not unseen binary bytes.
    """
    starts, _ends = verify_timing_plan(
        source,
        plan,
        model_identity=model_identity,
        clock=clock,
        project=project,
        capacities=capacities,
        max_calls=max_calls,
    )
    drains = [op.cycle.value.data for op in plan.body.block.ops if type(op) is DrainOp]
    if len(drains) != 1:
        raise ValueError("materialization requires one explicit drain")
    if type(emission) is not TimedEmission or type(emission.kernel) is not Kernel:
        raise ValueError("typed emitted kernel required")
    actual = emission.kernel
    positions = emission.source_positions
    if (
        actual.name != source.name
        or actual.args != source.args
        or actual.attrs != source.attrs
        or type(positions) is not tuple
        or len(positions) != len(source.body)
        or any(type(i) is not int or not 0 <= i < len(actual.body) for i in positions)
        or tuple(sorted(set(positions))) != positions
    ):
        raise ValueError("emission source correspondence differs")
    effects = _effects(actual, project, max_calls)
    gaps = _advances(actual, issue_gap)
    ticks, tick = [], 0
    selected = set(positions)
    for i, (effect, gap) in enumerate(zip(effects, gaps, strict=True)):
        ticks.append(tick)
        if i not in selected and (gap <= 0 or effect.accesses or effect.complete > gap):
            raise ValueError("inserted delay has effects or outlives its issue advance")
        tick += gap
    if tuple(actual.body[i] for i in positions) != source.body or tuple(ticks[i] for i in positions) != starts:
        raise ValueError("emitted calls or issue cycles differ from selected plan")
    if tick < drains[0]:
        raise ValueError("emitted tail does not realize drain")
    verify_timed_block(actual, tuple(ticks), project=project, capacities=capacities, max_calls=max_calls)
    return tuple(ticks), tick


def materialize_timing_plan(
    kernel, plan, *, delay, project, capacities, issue_gap, model_identity, clock, max_calls=8192
):
    """Fill gaps with target-selected calls, preserving every source instruction.

    ``delay(cycles)`` returns a bounded tuple of actual target instructions. Its
    natural advances must exactly cover the requested gap. No silent rounding.
    """
    starts, _ends = verify_timing_plan(
        kernel,
        plan,
        model_identity=model_identity,
        clock=clock,
        project=project,
        capacities=capacities,
        max_calls=max_calls,
    )
    drains = [op.cycle.value.data for op in plan.body.block.ops if type(op) is DrainOp]
    if len(drains) != 1:
        raise ValueError("materialization requires one explicit drain")
    calls, positions, tick = [], [], 0

    def pad(until):
        nonlocal tick
        if until < tick:
            raise ValueError("plan violates natural issue advance")
        if until == tick:
            return
        extra = delay(until - tick)
        if type(extra) is not tuple or not extra or any(type(c) is not Call for c in extra):
            raise ValueError("delay must return a nonempty tuple of target calls")
        if len(calls) + len(extra) > max_calls:
            raise ValueError("emission exceeds call budget")
        advances = _advances(replace(kernel, body=extra), issue_gap)
        if any(gap <= 0 for gap in advances) or sum(advances) != until - tick:
            raise ValueError("delay does not realize requested cycles")
        calls.extend(extra)
        tick = until

    gaps = _advances(kernel, issue_gap)
    for call, start, gap in zip(kernel.body, starts, gaps, strict=True):
        pad(start)
        positions.append(len(calls))
        calls.append(call)
        tick += gap
    pad(drains[0])
    emission = TimedEmission(replace(kernel, body=tuple(calls)), tuple(positions))
    verify_timed_emission(
        kernel,
        plan,
        emission,
        project=project,
        capacities=capacities,
        issue_gap=issue_gap,
        model_identity=model_identity,
        clock=clock,
        max_calls=max_calls,
    )
    return emission
