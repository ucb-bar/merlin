"""Hoist loop-invariant persistent setup using qualified actual call effects.

Only an ordered pure-write prefix moves, immediately before a nonempty loop.
Every remaining call must have known ordered effects and preserve those fields.
Identical ordered writes preserve established field values; conflicting writes
and explicit clobbers still block motion. The verifier independently matches
actual moved calls and their complete scope;
no instruction name, geometry, latency or initial register value is assumed.
"""

from __future__ import annotations

import struct
from dataclasses import fields, replace

from merlin.sched.ir import Call, Kernel, Loop, Ptr, Stage
from merlin.sched.ir.expr import Const, Expr, Scaled, Select, Sum, Var, free_vars
from merlin.sched.ir.kernel import check_structure

from .configuration import ConfigurationAdapter, ConfigurationEffects, ConfigurationResult, ConfigurationUnavailable


def _bounded(kernel, max_calls, max_fields, max_depth):
    # Check before recursive IR validation, hashing or effect projection.
    if type(kernel) is not Kernel or any(type(n) is not int or n <= 0 for n in (max_calls, max_fields, max_depth)):
        raise ValueError("configuration hoisting requires a typed kernel and positive explicit static budgets")
    pending, nodes, values = [(item, 0) for item in kernel.body], 0, 0
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
    _bounded(kernel, max_calls, max_fields, max_depth)
    if check_structure(kernel) or type(adapter) is not ConfigurationAdapter:
        raise ValueError("configuration hoisting requires checked loop IR and an explicit qualified effect adapter")
    effects, fields, nodes = {}, set(), 0

    def walk(body, depth):
        nonlocal nodes
        if depth > max_depth:
            raise ConfigurationUnavailable("configuration loop nesting exceeds its explicit budget")
        for statement in body:
            nodes += 1
            if nodes > max_calls:
                raise ConfigurationUnavailable("configuration loop source exceeds its static budget")
            if type(statement) is Loop:
                walk(statement.body, depth + 1)
                continue
            if type(statement) is not Call:
                raise ValueError("unknown configuration schedule statement")
            effect = adapter.project(statement)
            if effect is not None and type(effect) is not ConfigurationEffects:
                raise ValueError("configuration adapter returned untyped effects")
            if effect is not None:
                fields.update(key for key, _ in (*effect.reads, *effect.writes))
                fields.update(effect.clobbers)
                if len(fields) > max_fields:
                    raise ConfigurationUnavailable("configuration loop field budget exceeded")
                if effect.pure_write and any(
                    (
                        statement.unit,
                        statement.produces,
                        statement.awaits,
                        statement.stages,
                        statement.sets,
                        statement.assumes,
                    )
                ):
                    raise ValueError("pure configuration prefix cannot carry execution or state annotations")
            effects[id(statement)] = effect

    walk(kernel.body, 0)
    return effects


def _pure(statement, effects):
    return (
        type(statement) is Call
        and (effect := effects[id(statement)]) is not None
        and effect.ordered
        and effect.pure_write
        and all(isinstance(expr, Expr) for _, expr in statement.args)
    )


def _preserves(body, established, effects):
    keys = set(established)
    for statement in body:
        if type(statement) is Loop:
            if not _preserves(statement.body, established, effects):
                return False
        else:
            effect = effects[id(statement)]
            if (
                effect is None
                or not effect.ordered
                or keys.intersection(effect.clobbers)
                or any(key in established and value != established[key] for key, value in effect.writes)
            ):
                return False
    return True


def _movable(loop, prefix, effects):
    if loop.extent <= 0 or not prefix or any(not _pure(call, effects) for call in prefix):
        return False
    if any(loop.var in free_vars(expr) for call in prefix for _, expr in call.args):
        return False
    established = {}
    for call in prefix:
        established.update(effects[id(call)].writes)
    return _preserves(loop.body[len(prefix) :], established, effects)


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
