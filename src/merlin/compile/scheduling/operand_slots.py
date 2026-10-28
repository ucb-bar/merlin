"""Consume dependency-derived operand storage in a bounded counted pipeline.

Operands group existing private backing objects into immutable generations.
Actual decoded initialization, reads and completion commands determine their
live intervals; the ordinary temporary allocator assigns physical addresses.
Only a verified cyclic affine rotation is reconstructed as counted phase loops. No
latency, asynchronous overlap permission or target completion is inferred here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.sched.ir import Call, Const, Expr, Kernel, Loop, Var, check_structure, select
from merlin.sched.ir.integer_domain import check_integer_domain
from merlin.sched.primitives import Cursor, _subst_call, divide_loop

from .temporary_storage import (
    StorageOptimizationUnavailable,
    TemporaryObject,
    _problem,
    allocate_temporary_storage,
    verify_temporary_storage,
)


@dataclass(frozen=True)
class PipelineOperand:
    """A source operand's invocation-private, immutable backing generations.

    These are existing TemporaryObjects, not requested buffer depths. One
    generation can serve every iteration; distinct advancing generations may
    share physical storage only after their actual asynchronous users finish.
    """

    name: str
    generations: tuple[TemporaryObject, ...]

    def __post_init__(self):
        if (
            type(self.name) is not str
            or not self.name
            or type(self.generations) is not tuple
            or not self.generations
            or any(type(obj) is not TemporaryObject for obj in self.generations)
        ):
            raise ValueError("pipeline operands require named existing private generations")


@dataclass(frozen=True)
class PipelineSlotResult:
    kernel: Kernel
    report: dict


def _shape(kernel, max_calls, *, source):
    """Check the entire static shape and dynamic budget before closing calls."""
    if type(max_calls) is not int or max_calls <= 0:
        raise ValueError("pipeline expansion requires an explicit positive call budget")
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("pipeline slots require a structurally valid kernel")
    if kernel.attrs:
        raise StorageOptimizationUnavailable("pipeline slots cannot interpret kernel attributes")
    count, loops = 0, []

    def visit(body, multiplier, depth):
        nonlocal count
        for stmt in body:
            if type(stmt) is Loop:
                if depth >= (1 if source else 2) or stmt.iter_args or stmt.yields or stmt.results:
                    raise StorageOptimizationUnavailable(
                        "pipeline slots do not transport nested or SSA-carried asynchronous boundaries"
                    )
                loops.append(stmt)
                visit(stmt.body, multiplier * stmt.extent, depth + 1)
            elif type(stmt) is Call:
                if (
                    stmt.unit
                    or stmt.produces
                    or stmt.awaits
                    or stmt.stages
                    or stmt.sets
                    or stmt.assumes
                    or any(not isinstance(value, Expr) for _name, value in stmt.args)
                ):
                    raise StorageOptimizationUnavailable(
                        "pipeline slots require integer commands with actual decoded completion operands"
                    )
                count += multiplier
                if count > max_calls:
                    raise StorageOptimizationUnavailable("pipeline effect expansion exceeds its call budget")
            else:
                raise StorageOptimizationUnavailable("pipeline slots encountered an unknown statement")

    visit(kernel.body, 1, 0)
    if source and len(loops) != 1:
        raise StorageOptimizationUnavailable("pipeline slots support exactly one source counted loop")
    check_integer_domain(kernel, 64)
    return count


def _close(kernel):
    """Called only after the complete static shape and dynamic budget check."""
    calls = []

    def visit(body, bindings):
        for stmt in body:
            if type(stmt) is Loop:
                for index in range(stmt.extent):
                    visit(stmt.body, (*bindings, (stmt.var, index)))
            else:
                for name, value in bindings:
                    stmt = _subst_call(stmt, name, Const(value))
                calls.append(stmt)

    visit(kernel.body, ())
    return replace(kernel, body=tuple(calls))


def _objects(operands):
    if (
        type(operands) is not tuple
        or not operands
        or any(type(operand) is not PipelineOperand for operand in operands)
        or len({operand.name for operand in operands}) != len(operands)
    ):
        raise ValueError("pipeline operands require unique explicit roles")
    objects = tuple(obj for operand in operands for obj in operand.generations)
    if len({obj.name for obj in objects}) != len(objects):
        raise ValueError("each backing generation must belong to exactly one operand")
    return objects


def _requirements(original, operands, banks, adapter, model):
    objects = _objects(operands)
    bank_map, effects, bindings, intervals, *_rest = _problem(
        original,
        objects,
        banks,
        adapter,
        model["external_windows"],
        model["private_ownership"],
        model["max_calls"],
        model["max_objects"],
    )
    for index, (effect, row) in enumerate(zip(effects, bindings, strict=True)):
        for access, binding in zip(effect.accesses, row, strict=True):
            if binding is not None and access.writes and index != intervals[binding[0]][0]:
                raise StorageOptimizationUnavailable("an operand generation is modified after initialization")
    result = {}
    for operand in operands:
        if len({bank_map[obj.bank].backing for obj in operand.generations}) != 1:
            raise ValueError("one operand's generations must use the same physical backing")
        names = {obj.name for obj in operand.generations}
        # Closed issue intervals are intentional: objects born in a completing
        # command cannot alias its still-live inputs. No service-cycle estimate.
        events = {}
        for name in names:
            first, last = intervals[name]
            events[first] = events.get(first, 0) + 1
            events[last + 1] = events.get(last + 1, 0) - 1
        active = peak = 0
        for index in sorted(events):
            active += events[index]
            peak = max(peak, active)
        result[operand.name] = {
            "generations": [obj.name for obj in operand.generations],
            "required_live_generations": peak,
            "lifetime_calls": {obj.name: intervals[obj.name] for obj in operand.generations},
        }
    return result


def _cyclic_affine(values):
    """One affine progression modulo equally spaced allocated slot bases."""
    bases = sorted(set(values))
    if len(bases) == 1:
        return 0
    stride = bases[1] - bases[0]
    if any(right - left != stride for left, right in zip(bases, bases[1:])):
        return None
    span = stride * len(bases)
    step = values[1] - values[0]
    if any(value != bases[0] + (values[0] - bases[0] + index * step) % span for index, value in enumerate(values)):
        return None
    return step


def _phase_address(values, phase):
    step = _cyclic_affine(values)
    assert step is not None
    result = Const(values[0]) + Var(phase) * step
    # The small index language has no modulo. The checked, bounded phase loop
    # supplies its finite domain; exact wrap corrections remain ordinary selects.
    for index, value in enumerate(values):
        if value != values[0] + index * step:
            result = select(phase, index, value, result)
    return result


def _rotation(kernel, flat_source, flat_actual, max_period):
    """Find a repeated affine phase, then reuse the ordinary loop divider.

    Only relocated operands change. All original indices/tail predicates go
    through divide_loop's checked quotient/remainder substitution. A bounded
    final partial period is peeled using the exact already-checked calls.
    Irregular assignments keep the caller's original route.
    """
    position, loop = next((i, stmt) for i, stmt in enumerate(kernel.body) if type(stmt) is Loop)
    prefix, width, extent = position, len(loop.body), loop.extent
    changed = {}
    for column, call in enumerate(loop.body):
        for name, _value in call.args:
            before = [flat_source.body[prefix + i * width + column].arg(name) for i in range(extent)]
            after = [flat_actual.body[prefix + i * width + column].arg(name) for i in range(extent)]
            if after != before:
                if any(type(value) is not Const for value in after):
                    raise StorageOptimizationUnavailable("relocated pipeline operands must close to integer addresses")
                changed[column, name] = [value.value for value in after]

    def matches(period, whole):
        return all(
            all(values[i] == values[i % period] for i in range(whole)) and _cyclic_affine(values[:period]) is not None
            for values in changed.values()
        )

    selected = None
    for peel in (0, 1):
        for period in range(1, min(max_period, extent) + 1):
            whole = (extent - peel) // period * period
            if whole and (period == 1 or whole >= 2 * period) and matches(period, whole):
                selected = period, whole
                break
        if selected is not None:
            break
    if selected is None:
        raise StorageOptimizationUnavailable("allocation has no supported repeated affine operand rotation")
    period, whole = selected
    shortened = replace(
        kernel, body=(*kernel.body[:position], replace(loop, extent=whole), *kernel.body[position + 1 :])
    )
    if period == 1:
        divided = shortened
        phase = None
        body = loop.body
    else:
        divided = divide_loop(shortened, Cursor((loop.var,)), period)
        outer = divided.body[position]
        phase = outer.body[0].var
        body = outer.body[0].body
    rewritten = []
    for column, call in enumerate(body):
        args = []
        for name, value in call.args:
            values = changed.get((column, name))
            if values is not None:
                value = Const(values[0])
                if period > 1:
                    value = _phase_address(values[:period], phase)
            args.append((name, value))
        rewritten.append(replace(call, args=tuple(args)))
    if period == 1:
        counted = replace(divided.body[position], body=tuple(rewritten))
    else:
        outer = divided.body[position]
        counted = replace(outer, body=(replace(outer.body[0], body=tuple(rewritten)),))
    end = prefix + extent * width
    actual = replace(
        kernel,
        body=(
            *flat_actual.body[:prefix],
            counted,
            *flat_actual.body[prefix + whole * width : end],
            *flat_actual.body[end:],
        ),
    )
    return actual, {"phase_extent": period, "peeled_iterations": extent - whole}


def verify_pipeline_operand_slots(
    original,
    realized,
    operands,
    banks,
    adapter,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
) -> dict:
    """Independently recover live generations/addresses from emitted calls."""
    _shape(original, max_calls, source=True)
    _shape(realized, max_calls, source=False)
    source, actual = _close(original), _close(realized)
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    requirements = _requirements(source, operands, banks, adapter, model)
    storage = verify_temporary_storage(source, actual, _objects(operands), banks, adapter, **model)
    for operand in operands:
        addresses = {obj.name: storage["addresses"][obj.name] for obj in operand.generations}
        requirements[operand.name]["addresses"] = addresses
        # Distinct bases can exceed the peak-live lower bound with different
        # extents/alignment or shared-bank fragmentation; do not call it optimal.
        requirements[operand.name]["assigned_distinct_bases"] = len(set(addresses.values()))
    return {
        "schema": "merlin.pipeline_operand_slots.v1",
        "source_sha256": original.digest(),
        "realized_sha256": realized.digest(),
        "operands": requirements,
        "storage": storage,
        "command_order": "preserved_with_actual_completion_commands",
        "machine_emission": "not_established",
        "execution_qualification": "not_established",
        "timing": "UNKNOWN",
    }


def allocate_pipeline_operand_slots(
    kernel,
    operands,
    banks,
    adapter,
    *,
    reservations=(),
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_period=8,
    timeout_ms=5000,
) -> PipelineSlotResult | None:
    """Assign existing immutable generations, consume a checked counted rotation.

    Input is a prologue, exactly one nonnested loop of integer commands, and an
    epilogue. Async dependencies use actual encoded launch/wait operands decoded
    by StorageAdapter; SSA-carried/nested async boundaries currently refuse.
    The caller supplies an already legal issue order and qualified target laws.
    This transform neither moves commands nor creates overlap authorization.
    Unknown/irregular effects diagnose conservatively; infeasible storage gives
    None. Expansion is checked before closing any dynamic command. Numerical
    source qualification and actual native/target execution remain independent.
    """
    _shape(kernel, max_calls, source=True)
    if type(max_period) is not int or not 1 <= max_period <= 64:
        raise ValueError("pipeline rotation requires a positive bounded phase-search limit")
    model = dict(
        reservations=reservations,
        external_windows=external_windows,
        private_ownership=private_ownership,
        max_calls=max_calls,
        max_objects=max_objects,
    )
    flat = _close(kernel)
    _requirements(flat, operands, banks, adapter, model)
    allocated = allocate_temporary_storage(
        flat, _objects(operands), banks, adapter, timeout_ms=timeout_ms, eliminate_copies=False, **model
    )
    if allocated is None:
        return None
    actual, rotation = _rotation(kernel, flat, allocated.kernel, max_period)
    _shape(actual, max_calls, source=False)
    if _close(actual) != allocated.kernel:
        raise ValueError("counted rotation changed the allocated command program")
    report = verify_pipeline_operand_slots(kernel, actual, operands, banks, adapter, **model)
    return PipelineSlotResult(actual, {**report, "rotation": rotation})
