"""Remove redundant setup using independently decoded persistent state effects.

This extends the schedule IR's sets/assumes vocabulary with the missing
execution contract: actual operand decoding, clobbers, ordering and a qualified
pure idempotent write. An annotation or matching instruction name alone never
authorizes deleting an instruction. Unknown effects retain the instruction and
forget all known state. Initial state is unknown for each invocation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from merlin.sched.ir import Call, Kernel
from merlin.sched.ir.kernel import check_structure


class ConfigurationUnavailable(ValueError):
    """The selected configuration family cannot establish a legal rewrite."""


@dataclass(frozen=True)
class ConfigurationEffects:
    """Actual integer-valued configuration fields, in instruction order.

    ordered requires independently qualified visibility in program order.
    pure_write means ONLY persistent, idempotent register writes: no enqueue,
    pulse, acknowledgement, memory, status, synchronization or other effect.
    Reads precede clobbers and writes. Missing ordering forgets all state.
    """

    reads: tuple[tuple[str, int], ...] = ()
    writes: tuple[tuple[str, int], ...] = ()
    clobbers: tuple[str, ...] = ()
    ordered: bool = False
    pure_write: bool = False

    def __post_init__(self):
        for pairs in (self.reads, self.writes):
            if (
                type(pairs) is not tuple
                or any(
                    type(pair) is not tuple
                    or len(pair) != 2
                    or not isinstance(pair[0], str)
                    or not pair[0]
                    or type(pair[1]) is not int
                    for pair in pairs
                )
                or len({key for key, _ in pairs}) != len(pairs)
            ):
                raise ValueError("configuration effects require unique typed integer fields")
        if (
            type(self.clobbers) is not tuple
            or any(not isinstance(key, str) or not key for key in self.clobbers)
            or len(set(self.clobbers)) != len(self.clobbers)
            or type(self.ordered) is not bool
            or type(self.pure_write) is not bool
            or self.pure_write
            and (not self.writes or self.reads or self.clobbers or not self.ordered)
        ):
            raise ValueError("pure configuration updates require ordered writes and no other state effects")


@dataclass(frozen=True)
class ConfigurationAdapter:
    project: Callable[[Call], ConfigurationEffects | None]
    configuration_identity: str
    qualification_identity: str

    def __post_init__(self):
        if not callable(self.project) or any(
            not isinstance(value, str) or not value
            for value in (self.configuration_identity, self.qualification_identity)
        ):
            raise ValueError("configuration cleanup requires actual operand decoding and qualified state semantics")

    def record(self):
        return dict(
            configuration_identity=self.configuration_identity, qualification_identity=self.qualification_identity
        )


@dataclass(frozen=True)
class ConfigurationResult:
    kernel: Kernel
    report: dict


def _effects(kernel, adapter, max_calls, max_fields):
    if type(kernel) is not Kernel or check_structure(kernel):
        raise ValueError("configuration cleanup requires structurally valid schedule IR")
    if type(adapter) is not ConfigurationAdapter or any(type(n) is not int or n <= 0 for n in (max_calls, max_fields)):
        raise ValueError("configuration cleanup requires a typed adapter and bounded work")
    if len(kernel.body) > max_calls or any(type(call) is not Call for call in kernel.body):
        raise ConfigurationUnavailable(
            "configuration cleanup supports bounded straight-line calls; retain loop lowering"
        )
    effects, fields = [], set()
    for call in kernel.body:
        effect = adapter.project(call)
        if effect is not None and type(effect) is not ConfigurationEffects:
            raise ValueError("configuration decoder returned untyped effects")
        if effect is not None:
            fields.update(key for key, _ in (*effect.reads, *effect.writes))
            fields.update(effect.clobbers)
            if len(fields) > max_fields:
                raise ConfigurationUnavailable("configuration field budget exceeded; retain original setup")
            if effect.pure_write and (call.produces or call.awaits or call.stages):
                raise ValueError("pure configuration update carries execution or memory dependencies")
        effects.append(effect)
    return tuple(effects)


def _advance(live, effect):
    if effect is None or not effect.ordered:
        live.clear()
        return
    for key, required in effect.reads:
        if key in live and live[key] != required:
            raise ConfigurationUnavailable(
                f"configuration use requires {key}={required}, but current state is {live[key]}"
            )
    for key in effect.clobbers:
        live.pop(key, None)
    live.update(effect.writes)


def _check_pure_configuration_storage(calls, source_effects, adapter):
    """Cross-check pure setup against the independent storage-effect decoder."""
    for call, effect in zip(calls, source_effects, strict=True):
        config = adapter.project(call)
        if (
            config is not None
            and config.pure_write
            and (
                effect.accesses
                or effect.completion != "synchronous"
                or effect.exact_copy
                or effect.elementwise is not None
            )
        ):
            raise ValueError("pure configuration claim disagrees with decoded execution or memory effects")


def eliminate_redundant_configuration(kernel, adapter, *, enabled=True, max_calls=4096, max_fields=256):
    """Keep the first write and drop only fully redundant pure later writes.

    No dead-store motion, partial instruction rewrite, loop unrolling or
    cross-invocation state assumption. Unknown ordering/effects are barriers.
    """
    if type(enabled) is not bool:
        raise ValueError("configuration elimination needs an explicit boolean control")
    effects = _effects(kernel, adapter, max_calls, max_fields)
    live, retained = {}, []
    for call, effect in zip(kernel.body, effects, strict=True):
        redundant = (
            effect is not None
            and effect.pure_write
            and all(key in live and live[key] == value for key, value in effect.writes)
        )
        if not enabled or not redundant:
            retained.append(call)
        _advance(live, effect)
    rewritten = replace(kernel, body=tuple(retained))
    report = verify_configuration_rewrite(kernel, rewritten, adapter, max_calls=max_calls, max_fields=max_fields)
    report["enabled"] = enabled
    return ConfigurationResult(rewritten, report)


def verify_configuration_rewrite(original, modified, adapter, *, max_calls=4096, max_fields=256):
    """Reconstruct legal omissions from actual output, without a selected plan."""
    before = _effects(original, adapter, max_calls, max_fields)
    after = _effects(modified, adapter, max_calls, max_fields)
    if replace(original, body=modified.body) != modified:
        raise ValueError("configuration conformance: kernel interface or metadata changed")
    source_state, output_state, cursor, removed = {}, {}, 0, 0
    for call, effect in zip(original.body, before, strict=True):
        if cursor < len(modified.body) and call == modified.body[cursor]:
            actual = after[cursor]
            if actual != effect:
                raise ValueError("configuration conformance: decoder changed effects for a retained instruction")
            _advance(source_state, effect)
            _advance(output_state, actual)
            cursor += 1
        else:
            # A separate induction over actual retained state proves every
            # omitted whole instruction has no effect, including final state.
            if effect is None or not effect.pure_write or not effect.ordered:
                raise ValueError(
                    "configuration conformance: omitted instruction has unknown or nonconfiguration effects"
                )
            if any(key not in output_state or output_state[key] != value for key, value in effect.writes):
                raise ValueError("configuration conformance: omitted write changes live state")
            _advance(source_state, effect)
            removed += 1
        if source_state != output_state:
            raise ValueError("configuration conformance: retained calls observe changed persistent state")
    if cursor != len(modified.body):
        raise ValueError("configuration conformance: inserted or changed instructions in actual output")
    return {
        "schema": "merlin.configuration_rewrite.v1",
        "source_sha256": original.digest(),
        "output_sha256": modified.digest(),
        "source_calls": len(before),
        "actual_calls": len(after),
        "removed_calls": removed,
        "barriers": sum(effect is None or not effect.ordered for effect in before),
        "adapter": adapter.record(),
        "execution_qualification": "not_established",
        "timing": "UNKNOWN",
    }


def verify_configuration_state(kernel, adapter, *, max_calls=4096, max_fields=256):
    """Require actual uses to have established values after every later rewrite.

    Fusion, allocation and inserted spills can change configuration effects.
    Unknown instructions clear state; a later required read needs a new setter.
    This check does not assume that an exact memory copy preserves registers.
    """
    effects = _effects(kernel, adapter, max_calls, max_fields)
    live = {}
    reads = 0
    for effect in effects:
        if effect is not None:
            for key, value in effect.reads:
                if key not in live or live[key] != value:
                    raise ConfigurationUnavailable(f"configuration use lacks an established {key}={value}")
                reads += 1
        _advance(live, effect)
    return {"schedule_sha256": kernel.digest(), "checked_reads": reads, "known_final_state": dict(sorted(live.items()))}
