"""Bounded dead private-payload proposals; original delete permission required."""

from dataclasses import dataclass, replace

from merlin.sched.ir import Kernel

from .dead_write_check import (
    WriteDeletionLaw,
    _admitted,
    _charge,
    _key,
    _overlaps,
    _snapshot,
    _source,
    _subtract,
    _unchanged,
    _write,
    verify_dead_writes,
)
from .temporary_storage import StorageOptimizationUnavailable


@dataclass(frozen=True)
class DeadWriteElimination:
    kernel: Kernel
    report: dict


def eliminate_dead_writes(
    kernel,
    objects,
    banks,
    adapter,
    *,
    deletion_law: WriteDeletionLaw | None = None,
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_deletions=128,
    max_scan_steps=100000,
) -> DeadWriteElimination | None:
    """Remove only complete admitted payloads killed before any observation.

    The candidate never changes addresses, order, ownership or allocation.
    Unknown/async effects and unavailable proof retain the original program.
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
    try:
        before = _snapshot((kernel,), *args)
    except StorageOptimizationUnavailable:
        return None
    try:
        if not _admitted(deletion_law, adapter) or not max_deletions:
            return None
        work = [max_scan_steps]
        bank_map, effects, bindings = _source(
            kernel, objects, banks, adapter, external_windows, private_ownership, work, max_calls, max_objects
        )
        omitted = []
        for index, (call, effect) in enumerate(zip(kernel.body, effects, strict=True)):
            _charge(work)
            write = _write(call, effect, bindings[index], deletion_law)
            if write is None:
                continue
            pending = [(_key(write, bank_map), write.start, write.start + write.size)]
            for later in effects[index + 1 :]:
                _charge(work)
                if not later.accesses:
                    break
                read = False
                for access in later.accesses:
                    if not access.writes:
                        for interval in pending:
                            _charge(work)
                            read |= _overlaps(interval, access, bank_map)
                if read:
                    break
                for access in later.accesses:
                    if access.writes:
                        pending = _subtract(pending, access, bank_map, work)
                if not pending:
                    omitted.append(index)
                    break
            if len(omitted) == max_deletions:
                break
        if not omitted:
            return None
        selected = set(omitted)
        candidate = replace(kernel, body=tuple(call for i, call in enumerate(kernel.body) if i not in selected))
        return DeadWriteElimination(
            candidate,
            verify_dead_writes(
                kernel,
                candidate,
                objects,
                banks,
                adapter,
                deletion_law=deletion_law,
                external_windows=external_windows,
                private_ownership=private_ownership,
                max_calls=max_calls,
                max_objects=max_objects,
                max_deletions=max_deletions,
                max_scan_steps=max_scan_steps,
            ),
        )
    except StorageOptimizationUnavailable:
        return None
    finally:
        _unchanged(before, _snapshot((kernel,), *args))
