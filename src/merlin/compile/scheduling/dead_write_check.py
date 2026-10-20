"""Check omitted payload writes under an original qualified deletion law.

This is an already-placed, invocation-private storage proof. It supplies no
instruction, fault, observation, ownership, completion or target qualification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from merlin.sched.ir import Kernel
from merlin.semantic_compiler.allocation_types import StorageBank

from .program_limits import AddressWindow
from .temporary_storage import (
    StorageAdapter,
    StorageOptimizationUnavailable,
    TemporaryObject,
    _completion,
    _problem,
    _project,
)


@dataclass(frozen=True)
class WriteDeletionLaw:
    """Original qualifier covers each exact (instruction, semantic_key) form.

    Its premises cover the form's full admitted address/count/selector domain;
    instruction spelling, exact-copy metadata and matching IDs are not proof.
    """

    configuration_identity: str
    numerical_policy_identity: str
    adapter_qualification_identity: str
    qualification_identity: str
    removable_forms: tuple[tuple[str, str], ...]
    no_faults: bool = False
    no_extra_observations: bool = False
    write_removal_preserves_other_effects: bool = False

    def __post_init__(self):
        if (
            any(
                type(value) is not str or not value.strip()
                for value in (
                    self.configuration_identity,
                    self.numerical_policy_identity,
                    self.adapter_qualification_identity,
                    self.qualification_identity,
                )
            )
            or (
                type(self.removable_forms) is not tuple
                or not 0 < len(self.removable_forms) <= 4096
                or any(
                    type(form) is not tuple
                    or len(form) != 2
                    or any(type(name) is not str or not name.strip() for name in form)
                    for form in self.removable_forms
                )
                or len(set(self.removable_forms)) != len(self.removable_forms)
            )
            or any(
                type(value) is not bool
                for value in (
                    self.no_faults,
                    self.no_extra_observations,
                    self.write_removal_preserves_other_effects,
                )
            )
        ):
            raise ValueError(
                "write deletion requires exact instruction forms, original identities and explicit premises"
            )

    def record(self):
        self.__post_init__()
        return asdict(self)


def _snapshot(kernels, objects, banks, adapter, windows, law, max_calls, max_objects, max_deletions, max_scan_steps):
    if any(type(n) is not int or n <= 0 for n in (max_calls, max_objects, max_scan_steps)) or (
        type(max_deletions) is not int or max_deletions < 0
    ):
        raise ValueError("write deletion requires positive work bounds and a nonnegative deletion count")
    if type(adapter) is not StorageAdapter or any(type(k) is not Kernel for k in kernels):
        raise ValueError("write deletion requires original kernels and a typed storage adapter")
    adapter.__post_init__()
    if law is not None and type(law) is not WriteDeletionLaw:
        raise ValueError("write deletion requires an original typed deletion law")
    for values, cls in ((objects, TemporaryObject), (banks, StorageBank), (windows, AddressWindow)):
        if type(values) is not tuple or any(type(v) is not cls for v in values):
            raise ValueError("write deletion requires immutable typed storage and external windows")
        if len(values) > max_objects:
            raise StorageOptimizationUnavailable("write deletion storage/window budget exceeded")
        for value in values:
            value.__post_init__()
    if any(len(k.body) > max_calls for k in kernels):
        raise StorageOptimizationUnavailable("write deletion call budget exceeded")
    callbacks = tuple(
        (f, getattr(getattr(f, "__func__", f), "__code__", None)) for f in (adapter.project, adapter.rewrite)
    )
    return (
        tuple(k.digest() for k in kernels),
        tuple(asdict(o) for o in objects),
        tuple(asdict(b) for b in banks),
        tuple(asdict(w) for w in windows),
        law.record() if law is not None else None,
        (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity),
        callbacks,
    )


def _unchanged(before, after):
    if before[:-1] != after[:-1] or any(
        a is not b or ac is not bc for (a, ac), (b, bc) in zip(before[-1], after[-1], strict=True)
    ):
        raise ValueError("original write-deletion source, law, storage or selected adapter changed during a callback")


def _admitted(law, adapter):
    return (
        law is not None
        and law.no_faults
        and law.no_extra_observations
        and law.write_removal_preserves_other_effects
        and (law.configuration_identity, law.numerical_policy_identity, law.adapter_qualification_identity)
        == (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity)
    )


def _charge(work, amount=1):
    work[0] -= amount
    if work[0] < 0:
        raise StorageOptimizationUnavailable("write deletion scan budget exhausted")


def _effects(kernel, adapter, bank_map, work, max_calls):
    effects = _project(kernel, adapter, max_calls)
    for effect in effects:
        effect.__post_init__()
        _charge(work, 1 + len(effect.accesses))
        if effect.completion != "synchronous":
            raise StorageOptimizationUnavailable("write deletion does not admit asynchronous or wait instructions")
        for access in effect.accesses:
            access.__post_init__()
            if access.space in bank_map and not (
                bank_map[access.space].start <= access.start
                and access.start + access.size <= bank_map[access.space].capacity
            ):
                raise StorageOptimizationUnavailable("write deletion needs actual bounded physical addresses")
    _completion(effects, bank_map)
    return effects


def _source(kernel, objects, banks, adapter, windows, private_ownership, work, max_calls, max_objects):
    bank_map = {b.name: b for b in banks}
    effects = _effects(kernel, adapter, bank_map, work, max_calls)
    _charge(work, sum(len(e.accesses) for e in effects) * max(1, len(objects)))
    # Reuse the original ownership/initialization/alias admission without
    # reinvoking a provider between this projection and the local analysis.
    projected = iter(effects)
    original = replace(adapter, project=lambda call: next(projected))
    checked = _problem(kernel, objects, banks, original, windows, private_ownership, max_calls, max_objects)
    return bank_map, effects, checked[2]


def _write(call, effect, bindings, law):
    if (call.instr, effect.semantic_key) not in law.removable_forms:
        return None
    if not effect.exact_copy and (not effect.accesses or any(not a.writes for a in effect.accesses)):
        return None
    writes = tuple(
        (access, binding) for access, binding in zip(effect.accesses, bindings, strict=True) if access.writes
    )
    # One complete instruction payload, never a partial instruction rewrite.
    return writes[0][0] if len(writes) == 1 and writes[0][1] is not None else None


def _key(access, banks):
    return ("private", banks[access.space].backing) if access.space in banks else ("external", access.space)


def _overlaps(interval, access, banks):
    key, start, end = interval
    return key == _key(access, banks) and start < access.start + access.size and access.start < end


def _subtract(intervals, access, banks, work):
    result = []
    for key, start, end in intervals:
        _charge(work)
        if key != _key(access, banks) or end <= access.start or access.start + access.size <= start:
            result.append((key, start, end))
        else:
            if start < access.start:
                result.append((key, start, access.start))
            if access.start + access.size < end:
                result.append((key, access.start + access.size, end))
    return result


def verify_dead_writes(
    original,
    realized,
    objects,
    banks,
    adapter,
    *,
    deletion_law=None,
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_deletions=128,
    max_scan_steps=100000,
):
    """Read actual omissions and prove every changed byte killed before reading.

    All surviving calls must be unchanged and ordered. Reads precede writes
    within each operation, including implicit/RMW reads. Alias views share the
    original physical backing coordinates. Final private differences also refuse.
    """
    args = (
        objects,
        banks,
        adapter,
        external_windows,
        deletion_law,
        max_calls,
        max_objects,
        max_deletions,
        max_scan_steps,
    )
    before = _snapshot((original, realized), *args)
    try:
        if original == realized:
            return {"status": "unchanged", "deleted_calls": 0}
        if not _admitted(deletion_law, adapter):
            raise StorageOptimizationUnavailable("write deletion lacks its original fault/observation/delete law")
        if replace(realized, body=original.body) != original or len(realized.body) >= len(original.body):
            raise ValueError("write deletion changed the kernel boundary or added unrelated work")
        work = [max_scan_steps]
        bank_map, source, bindings = _source(
            original, objects, banks, adapter, external_windows, private_ownership, work, max_calls, max_objects
        )
        actual = _effects(realized, adapter, bank_map, work, max_calls)
        # Match survivors from the end: repeated identical initializations must
        # not force the first occurrence to survive. This derives correspondence
        # only from actual calls, and the following replay still proves every
        # omitted occurrence safe independently of the proposal's indices.
        positions, source_index = [], len(original.body) - 1
        for call in reversed(realized.body):
            while source_index >= 0 and original.body[source_index] != call:
                _charge(work)
                source_index -= 1
            if source_index < 0:
                raise ValueError("actual program changes, reorders or inserts original instructions")
            positions.append(source_index)
            source_index -= 1
        positions.reverse()
        cursor, removed, differences = 0, [], []
        for index, (call, effect) in enumerate(zip(original.body, source, strict=True)):
            _charge(work)
            if cursor < len(positions) and index == positions[cursor]:
                if effect != actual[cursor]:
                    raise ValueError("unchanged instruction has inconsistent actual effects")
                if differences and not effect.accesses:
                    raise ValueError("omitted payload crosses an opaque observation/configuration barrier")
                for access in effect.accesses:
                    if not access.writes:
                        for interval in differences:
                            _charge(work)
                            if _overlaps(interval, access, bank_map):
                                raise ValueError("omitted payload is read before complete overwrite")
                for access in effect.accesses:
                    if access.writes:
                        differences = _subtract(differences, access, bank_map, work)
                cursor += 1
            else:
                write = _write(call, effect, bindings[index], deletion_law)
                if write is None:
                    raise ValueError("omitted instruction lacks its qualified private payload deletion law")
                removed.append(index)
                if len(removed) > max_deletions:
                    raise ValueError("write deletion exceeds its original deletion budget")
                differences.append((_key(write, bank_map), write.start, write.start + write.size))
        if cursor != len(realized.body):
            raise ValueError("actual program changes, reorders or inserts original instructions")
        if differences:
            raise ValueError("omitted payload is not completely overwritten by surviving actual writes")
        return {
            "schema": "merlin.dead_payload_writes.v1",
            "source_sha256": original.digest(),
            "realized_sha256": realized.digest(),
            "deletion_law": deletion_law.record(),
            "deleted_calls": len(removed),
            "omitted_source_indices": removed,
            "static_commands_before": len(source),
            "static_commands_after": len(actual),
            "original_requested_read_bytes": sum(a.size for e in source for a in e.accesses if not a.writes),
            "original_requested_write_bytes": sum(a.size for e in source for a in e.accesses if a.writes),
            "requested_read_bytes": sum(a.size for e in actual for a in e.accesses if not a.writes),
            "requested_write_bytes": sum(a.size for e in actual for a in e.accesses if a.writes),
            "scan_steps": max_scan_steps - work[0],
            "machine_emission": "not_established",
            "execution_qualification": "not_established",
            "timing": "UNKNOWN",
        }
    finally:
        _unchanged(before, _snapshot((original, realized), *args))
