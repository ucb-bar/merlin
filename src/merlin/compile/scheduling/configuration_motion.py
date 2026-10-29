"""Expose late pure setup to existing configuration cleanup and prefix hoisting.

Motion is within one counted body only. Actual qualified state reads, writes,
clobbers and unknown effects are barriers; loops and execution/state metadata
are never crossed. No instruction is deleted and no target law is supplied.
"""
from dataclasses import replace

from merlin.compile.scheduling.configuration import ConfigurationResult
from merlin.compile.scheduling.configuration_loops import _effects, _pure, _same_call
from merlin.sched.ir import Call, Loop


def _crosses(setter, earlier, effects):
    if type(earlier) is not Call or any((earlier.unit, earlier.produces, earlier.awaits,
                                       earlier.stages, earlier.sets, earlier.assumes)):
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
            match = next((i for i, call in enumerate(pending)
                          if type(call) is Call and _same_call(call, statement)), None)
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
        "schema": "merlin.configuration_motion.v1", "moved_static_calls": moved,
        "crossed_static_effects": crossings, "source_sha256": original.digest(),
        "kernel_sha256": actual.digest(), "adapter": adapter.record(),
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
