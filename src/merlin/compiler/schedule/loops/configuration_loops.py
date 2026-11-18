"""Move and eliminate persistent setup in intact counted loop programs.

Only an ordered pure-write prefix moves, immediately before a nonempty loop.
Every remaining call must have known ordered effects and preserve those fields.
Identical ordered writes preserve established field values; conflicting writes
and explicit clobbers still block motion. Within a body, pure setters commute
only across independently decoded effects that do not touch their fields.
Redundant setters require entry facts preserved by the complete original body.
Each verifier independently matches actual output and its complete scope;
no instruction name, geometry, latency or initial register value is assumed.
"""

from __future__ import annotations

import struct
from dataclasses import fields, replace

from merlin.compiler.kernel_ir.ir import (
    Call,
    Const,
    Expr,
    Kernel,
    Loop,
    Ptr,
    Scaled,
    Select,
    Stage,
    Sum,
    Var,
    check_structure,
    free_vars,
)
from merlin.compiler.schedule.loops.configuration import (
    ConfigurationAdapter,
    ConfigurationEffects,
    ConfigurationResult,
    ConfigurationUnavailable,
    _advance,
    _decode_effects,
)

# ---- configuration_loops -----------------------------------------------------------------------


def _bounded(kernel, max_calls, max_fields, max_depth):
    # Check before recursive IR validation, hashing or effect projection.
    if type(kernel) is not Kernel or any(type(n) is not int or n <= 0 for n in (max_calls, max_fields, max_depth)):
        raise ValueError("configuration hoisting requires a typed kernel and positive explicit static budgets")
    pending, nodes, values, calls = [(item, 0) for item in kernel.body], 0, 0, []
    while pending:
        statement, depth = pending.pop()
        nodes += 1
        if depth > min(max_depth, 64) or nodes > max_calls:
            raise ConfigurationUnavailable("configuration loop source/nesting exceeds its explicit budget")
        if type(statement) is Loop:
            pending.extend((item, depth + 1) for item in statement.body)
            continue
        if type(statement) is not Call:
            raise ValueError("unknown configuration schedule statement")
        calls.append(statement)
        operands = [(statement, 0)]
        while operands:
            value, level = operands.pop()
            values += 1
            if level > 64 or values > max_calls * 1024:
                raise ConfigurationUnavailable("configuration expression verification budget exceeded")
            if type(value) is tuple:
                if len(value) > max(64, max_fields):
                    raise ConfigurationUnavailable("configuration operand/field budget exceeded")
                operands.extend((item, level + 1) for item in value)
            elif type(value) in (Call, Const, Var, Sum, Scaled, Select, Ptr, Stage):
                operands.extend((getattr(value, field.name), level + 1) for field in fields(value))
            elif isinstance(value, Expr):
                raise ConfigurationUnavailable("unknown configuration index expression")
    return tuple(reversed(calls))


def _same_call(left, right):
    pending = [(left, right)]
    while pending:
        a, b = pending.pop()
        if type(a) is not type(b):
            return False
        if type(a) is float:
            if struct.pack(">d", a) != struct.pack(">d", b):
                return False
        elif type(a) is tuple:
            if len(a) != len(b):
                return False
            pending.extend(zip(a, b))
        elif type(a) in (Call, Const, Var, Sum, Scaled, Select, Ptr, Stage):
            pending.extend((getattr(a, field.name), getattr(b, field.name)) for field in fields(a))
        elif a != b:
            return False
    return True


def _effects(kernel, adapter, max_calls, max_fields, max_depth):
    calls = _bounded(kernel, max_calls, max_fields, max_depth)
    if check_structure(kernel) or type(adapter) is not ConfigurationAdapter:
        raise ValueError("configuration hoisting requires checked loop IR and an explicit qualified effect adapter")
    effects: dict[int, ConfigurationEffects | None] = {}
    for statement, effect in _decode_effects(calls, adapter, max_fields, counted=True):
        if (
            effect is not None
            and effect.pure_write
            and any(
                (
                    statement.unit,
                    statement.produces,
                    statement.awaits,
                    statement.stages,
                    statement.sets,
                    statement.assumes,
                )
            )
        ):
            raise ValueError("pure configuration prefix cannot carry execution or state annotations")
        effects[id(statement)] = effect
    return effects


def _pure(statement, effects):
    return (
        type(statement) is Call
        and (effect := effects[id(statement)]) is not None
        and effect.ordered
        and effect.pure_write
        and all(isinstance(expr, Expr) for _, expr in statement.args)
    )


def _invariants(body, established, effects):
    """Keep only entry facts preserved by the complete original loop body.

    Inspect each effect once for all fields, including nested and zero-trip
    loops. A later matching setter cannot restore an invalidated entry fact.
    """
    retained = dict(established)
    for statement in body:
        if type(statement) is Loop:
            retained = _invariants(statement.body, retained, effects)
        else:
            effect = effects[id(statement)]
            if effect is None or not effect.ordered:
                return {}
            for key in effect.clobbers:
                retained.pop(key, None)
            for key, value in effect.writes:
                if key in retained and value != retained[key]:
                    del retained[key]
    return retained


def _movable(loop, prefix, effects):
    if loop.extent <= 0 or not prefix or any(not _pure(call, effects) for call in prefix):
        return False
    if any(loop.var in free_vars(expr) for call in prefix for _, expr in call.args):
        return False
    established = {}
    for call in prefix:
        established.update(effects[id(call)].writes)
    return _invariants(loop.body[len(prefix) :], established, effects) == established


def verify_loop_configuration_rewrite(
    original,
    actual,
    adapter,
    *,
    max_calls=4096,
    max_fields=256,
    max_depth=64,
):
    """Check actual loop prefixes, field invariance and every retained instruction.

    This accepts only prefix motion, with identical loop extents, operands,
    annotations and non-configuration call order. It does not replay the
    optimizer or accept its selected moves as evidence.
    """
    effects = _effects(original, adapter, max_calls, max_fields, max_depth)
    actual_effects = _effects(actual, adapter, max_calls, max_fields, max_depth)
    if (original.name, original.args, original.attrs) != (actual.name, actual.args, actual.attrs):
        raise ValueError("configuration hoisting changed the invocation boundary")
    moved, dynamic_saved = 0, 0

    def compare(before, after, multiplicity=1):
        nonlocal moved, dynamic_saved
        cursor = 0
        for statement in before:
            if cursor >= len(after):
                raise ValueError("configuration hoisting dropped an original instruction or loop")
            if type(statement) is Call:
                if (
                    not _same_call(after[cursor], statement)
                    or effects[id(statement)] != actual_effects[id(after[cursor])]
                ):
                    raise ValueError("configuration hoisting changed a retained instruction")
                cursor += 1
                continue
            prefix = []
            while cursor < len(after) and type(after[cursor]) is Call:
                prefix.append(after[cursor])
                cursor += 1
            if cursor >= len(after) or type(after[cursor]) is not Loop:
                raise ValueError("configuration prefix has no matching original loop")
            loop = after[cursor]
            if (loop.var, loop.extent) != (statement.var, statement.extent):
                raise ValueError("configuration hoisting changed a loop domain")
            if prefix:
                if any(not _same_call(a, b) for a, b in zip(prefix, statement.body)) or not _movable(
                    statement, statement.body[: len(prefix)], effects
                ):
                    raise ValueError("actual configuration prefix is not loop invariant or crosses unknown state")
                if any(effects[id(old)] != actual_effects[id(new)] for old, new in zip(statement.body, prefix)):
                    raise ValueError("actual moved configuration effects differ from the original")
                moved += len(prefix)
                dynamic_saved += multiplicity * (statement.extent - 1) * len(prefix)
            compare(statement.body[len(prefix) :], loop.body, multiplicity * statement.extent)
            cursor += 1
        if cursor != len(after):
            raise ValueError("configuration hoisting introduced extra instructions")

    compare(original.body, actual.body)
    return {
        "hoisted_static_calls": moved,
        "removed_dynamic_configuration_calls": dynamic_saved,
        "scope": "qualified persistent state prefix motion; no device timing or execution qualification",
    }


def hoist_loop_configuration(kernel, adapter, *, enabled=True, max_calls=4096, max_fields=256, max_depth=64):
    """Move complete legal setup prefixes; unknown effects retain their original scope."""
    if type(enabled) is not bool:
        raise ValueError("configuration hoisting requires an explicit Boolean selection")
    _bounded(kernel, max_calls, max_fields, max_depth)
    identity = kernel.digest()
    effects = _effects(kernel, adapter, max_calls, max_fields, max_depth)

    def rewrite(body):
        result = []
        for statement in body:
            if type(statement) is Call:
                result.append(statement)
                continue
            prefix = []
            if enabled:
                for call in statement.body:
                    if not _pure(call, effects):
                        break
                    prefix.append(call)
            if _movable(statement, tuple(prefix), effects):
                result.extend(prefix)
                result.append(replace(statement, body=rewrite(statement.body[len(prefix) :])))
            else:
                result.append(replace(statement, body=rewrite(statement.body)))
        return tuple(result)

    candidate = replace(kernel, body=rewrite(kernel.body))
    report = verify_loop_configuration_rewrite(
        kernel, candidate, adapter, max_calls=max_calls, max_fields=max_fields, max_depth=max_depth
    )
    if kernel.digest() != identity:
        raise ValueError("configuration adapter mutated the original schedule")
    return ConfigurationResult(
        candidate,
        {
            **report,
            "source_sha256": identity,
            "kernel_sha256": candidate.digest(),
            "adapter": adapter.record(),
            "enabled": enabled,
        },
    )


# ---- configuration_loop_cleanup ----------------------------------------------------------------


def verify_counted_configuration_rewrite(original, actual, adapter, *, max_calls=4096, max_fields=256, max_depth=64):
    """Reconstruct each omission and loop invariant from actual source effects."""
    before = _effects(original, adapter, max_calls, max_fields, max_depth)
    after = _effects(actual, adapter, max_calls, max_fields, max_depth)
    if replace(original, body=actual.body) != actual:
        raise ValueError("counted configuration rewrite changed the kernel boundary")
    removed, saved = 0, 0

    def compare(source, output, entry, multiplicity):
        nonlocal removed, saved
        left, right, cursor = dict(entry), dict(entry), 0
        for statement in source:
            if type(statement) is Loop:
                if cursor >= len(output) or type(output[cursor]) is not Loop:
                    raise ValueError("configuration rewrite removed an original loop")
                candidate = output[cursor]
                if replace(statement, body=candidate.body) != candidate:
                    raise ValueError("configuration rewrite changed a loop domain or metadata")
                facts = _invariants(statement.body, left, before)
                exit_state = compare(statement.body, candidate.body, facts, multiplicity * statement.extent)
                if statement.extent:
                    left, right = dict(exit_state), dict(exit_state)
                cursor += 1
                continue
            effect = before[id(statement)]
            if cursor < len(output) and type(output[cursor]) is Call and _same_call(statement, output[cursor]):
                changed = after[id(output[cursor])]
                if effect != changed:
                    raise ValueError("configuration decoder changed retained call effects")
                _advance(left, effect)
                _advance(right, changed)
                cursor += 1
            else:
                if effect is None or not effect.pure_write or not effect.ordered:
                    raise ValueError("configuration rewrite omitted a nonconfiguration effect")
                if any(key not in right or right[key] != value for key, value in effect.writes):
                    raise ValueError("omitted counted setter lacks independently established state")
                _advance(left, effect)
                removed += 1
                saved += multiplicity
            if left != right:
                raise ValueError("counted rewrite changed state observed by retained calls")
        if cursor != len(output):
            raise ValueError("counted configuration rewrite introduced or changed calls")
        return right

    compare(original.body, actual.body, {}, 1)
    return {
        "schema": "merlin.counted_configuration_rewrite.v1",
        "removed_static_calls": removed,
        "removed_dynamic_calls": saved,
        "source_sha256": original.digest(),
        "kernel_sha256": actual.digest(),
        "adapter": adapter.record(),
        "scope": "source-preserved loop invariants and exact retained effects; no target timing qualification",
    }


def eliminate_counted_configuration(kernel, adapter, *, enabled=True, max_calls=4096, max_fields=256, max_depth=64):
    """Keep loops intact; refuse unproven cross-iteration assumptions."""
    if type(enabled) is not bool:
        raise ValueError("counted configuration cleanup requires an explicit Boolean")
    source = kernel.digest()
    effects = _effects(kernel, adapter, max_calls, max_fields, max_depth)

    def rewrite(body, entry):
        live, out = dict(entry), []
        for statement in body:
            if type(statement) is Loop:
                facts = _invariants(statement.body, live, effects)
                nested, final = rewrite(statement.body, facts)
                out.append(replace(statement, body=nested))
                if statement.extent:
                    live = final
                continue
            effect = effects[id(statement)]
            redundant = (
                effect is not None
                and effect.pure_write
                and all(key in live and live[key] == value for key, value in effect.writes)
            )
            if not enabled or not redundant:
                out.append(statement)
            _advance(live, effect)
        return tuple(out), live

    body, _ = rewrite(kernel.body, {})
    actual = replace(kernel, body=body)
    report = verify_counted_configuration_rewrite(
        kernel, actual, adapter, max_calls=max_calls, max_fields=max_fields, max_depth=max_depth
    )
    if kernel.digest() != source:
        raise ValueError("configuration adapter mutated its original counted source")
    return ConfigurationResult(actual, {**report, "enabled": enabled})


# ---- configuration_motion ----------------------------------------------------------------------


def _crosses(setter, earlier, effects):
    if type(earlier) is not Call or any(
        (earlier.unit, earlier.produces, earlier.awaits, earlier.stages, earlier.sets, earlier.assumes)
    ):
        return False
    state = effects[id(earlier)]
    if state is None or not state.ordered or state.pure_write:
        # Keep setter order stable, even when two assignments are disjoint.
        return False
    written = {key for key, _ in effects[id(setter)].writes}
    touched = {key for key, _ in (*state.reads, *state.writes)} | set(state.clobbers)
    return not written.intersection(touched)


def verify_configuration_motion(original, actual, adapter, *, max_calls=4096, max_fields=256, max_depth=64):
    """Independently match every retained call and each crossed original effect."""
    before = _effects(original, adapter, max_calls, max_fields, max_depth)
    after = _effects(actual, adapter, max_calls, max_fields, max_depth)
    if replace(original, body=actual.body) != actual:
        raise ValueError("configuration motion changed the original invocation boundary")
    moved, crossings = 0, 0

    def compare(source, output):
        nonlocal moved, crossings
        pending = list(source)
        for statement in output:
            if type(statement) is Loop:
                if not pending or type(pending[0]) is not Loop:
                    raise ValueError("configuration motion crossed an original loop")
                original_loop = pending.pop(0)
                if replace(original_loop, body=statement.body) != statement:
                    raise ValueError("configuration motion changed an original loop domain")
                compare(original_loop.body, statement.body)
                continue
            match = next(
                (i for i, call in enumerate(pending) if type(call) is Call and _same_call(call, statement)), None
            )
            if match is None:
                raise ValueError("configuration motion introduced or changed an original call")
            selected = pending[match]
            if before[id(selected)] != after[id(statement)]:
                raise ValueError("configuration motion changed decoded original effects")
            if match:
                if not _pure(selected, before) or not all(
                    _crosses(selected, earlier, before) for earlier in pending[:match]
                ):
                    raise ValueError("configuration motion crossed a state dependency or unknown effect")
                moved += 1
                crossings += match
            pending.pop(match)
        if pending:
            raise ValueError("configuration motion dropped original calls or loops")

    compare(original.body, actual.body)
    return {
        "schema": "merlin.configuration_motion.v1",
        "moved_static_calls": moved,
        "crossed_static_effects": crossings,
        "source_sha256": original.digest(),
        "kernel_sha256": actual.digest(),
        "adapter": adapter.record(),
        "scope": "qualified functional state commutation; no deletion, completion, target or timing grant",
    }


def promote_configuration_writes(kernel, adapter, *, enabled=True, max_calls=4096, max_fields=256, max_depth=64):
    """Move source-constant pure setters before state-independent data calls.

    This exposes a complete prefix for ``hoist_loop_configuration`` without
    adding a second hoister or assuming first-iteration state. Unknown effects,
    readers/writers/clobbers of a setter's fields, metadata and loops stop it.
    """
    if type(enabled) is not bool:
        raise ValueError("configuration motion requires an explicit Boolean")
    effects = _effects(kernel, adapter, max_calls, max_fields, max_depth)
    identity = kernel.digest()

    def body(statements):
        result = []
        for statement in statements:
            if type(statement) is Loop:
                result.append(replace(statement, body=body(statement.body)))
                continue
            position = len(result)
            if enabled and _pure(statement, effects):
                while position and _crosses(statement, result[position - 1], effects):
                    position -= 1
            result.insert(position, statement)
        return tuple(result)

    actual = replace(kernel, body=body(kernel.body))
    report = verify_configuration_motion(
        kernel, actual, adapter, max_calls=max_calls, max_fields=max_fields, max_depth=max_depth
    )
    if kernel.digest() != identity:
        raise ValueError("configuration motion adapter changed its original source")
    return ConfigurationResult(actual, {**report, "enabled": enabled})
