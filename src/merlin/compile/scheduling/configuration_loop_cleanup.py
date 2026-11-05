"""Remove redundant persistent-state setters in intact counted loop programs.

Loop entry facts are usable inside every iteration only if the complete source
body preserves them. Nothing is dynamically unrolled, and retained calls keep
all original operands, completion metadata and loop scopes.
"""
from __future__ import annotations

from dataclasses import replace

from merlin.sched.ir import Call, Loop
from .configuration import ConfigurationResult, _advance
from .configuration_loops import _effects, _preserves, _same_call


def _invariants(body, live, effects):
    return {key: value for key, value in live.items() if _preserves(body, {key: value}, effects)}


def verify_counted_configuration_rewrite(
    original, actual, adapter, *, max_calls=4096, max_fields=256, max_depth=64
):
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
    return {"schema": "merlin.counted_configuration_rewrite.v1", "removed_static_calls": removed,
            "removed_dynamic_calls": saved, "source_sha256": original.digest(),
            "kernel_sha256": actual.digest(), "adapter": adapter.record(),
            "scope": "source-preserved loop invariants and exact retained effects; no target timing qualification"}


def eliminate_counted_configuration(
    kernel, adapter, *, enabled=True, max_calls=4096, max_fields=256, max_depth=64
):
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
            redundant = (effect is not None and effect.pure_write
                         and all(key in live and live[key] == value for key, value in effect.writes))
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
