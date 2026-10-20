"""Bounded adjacent synchronous copy proposals under an original merge law.

No target opcode, request limit, fault behavior or observational equivalence is
inferred. Final encoded-program admission remains the original caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from merlin.sched.ir import Kernel

from .copy_coalescing_check import (
    CopyMergeLaw,
    _admitted,
    _physical_copy,
    _physical_windows,
    _snapshot,
    _unchanged,
    verify_coalesced_copies,
)
from .temporary_storage import StorageOptimizationUnavailable, _problem, _project


@dataclass(frozen=True)
class CopyCoalescing:
    kernel: Kernel
    report: dict


def coalesce_private_copies(
    kernel,
    objects,
    banks,
    adapter,
    *,
    merge_law: CopyMergeLaw | None = None,
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_merges=128,
) -> CopyCoalescing | None:
    """Combine adjacent compatible requests; missing information retains source.

    Existing initialization/ownership admission is intentionally unchanged.
    Virtual-only addresses, differently qualified forms, partial initialization
    across objects and unmodeled external aliases remain outside this family.
    """
    arguments = (objects, banks, adapter, external_windows, merge_law, max_calls, max_objects, max_merges)
    before = _snapshot((kernel,), *arguments)
    try:
        if not _admitted(merge_law, adapter) or max_merges == 0:
            return None
        current, merged = kernel, 0
        while merged < max_merges:
            bank_map, effects, *_ = _problem(
                current, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
            )
            if any(effect.completion != "synchronous" for effect in effects):
                return None
            if not _physical_windows(effects, bank_map):
                return None
            changed = False
            for index, (left, right) in enumerate(zip(effects, effects[1:])):
                if (
                    current.body[index].instr != current.body[index + 1].instr
                    or not _physical_copy(left, bank_map, merge_law)
                    or not _physical_copy(right, bank_map, merge_law)
                    or replace(right, accesses=left.accesses) != left
                ):
                    continue
                source, destination = left.accesses
                next_source, next_destination = right.accesses
                if (
                    source.space != next_source.space
                    or destination.space != next_destination.space
                    or next_source.start != source.start + source.size
                    or next_destination.start != destination.start + destination.size
                ):
                    continue
                accesses = tuple(replace(a, size=a.size + b.size) for a, b in zip(left.accesses, right.accesses))
                expected = replace(left, accesses=accesses)
                if not _physical_copy(expected, bank_map, merge_law):
                    continue
                call = adapter.rewrite(current.body[index], accesses)
                _unchanged(before, _snapshot((kernel,), *arguments))
                actual = _project(replace(current, body=(call,)), adapter, max_calls)[0]
                if actual != expected:
                    if actual.accesses == expected.accesses:
                        raise StorageOptimizationUnavailable(
                            "rewritten copy form changes its declared nonaddress semantics"
                        )
                    raise ValueError("copy adapter did not realize the requested merged byte effects")
                proposal = replace(current, body=(*current.body[:index], call, *current.body[index + 2 :]))
                # The independent checker reads the actual proposed calls and
                # original source, not this scan's indices or expected effects.
                verify_coalesced_copies(
                    kernel,
                    proposal,
                    objects,
                    banks,
                    adapter,
                    merge_law=merge_law,
                    external_windows=external_windows,
                    private_ownership=private_ownership,
                    max_calls=max_calls,
                    max_objects=max_objects,
                    max_merges=max_merges,
                )
                current, merged, changed = proposal, merged + 1, True
                break
            if not changed:
                break
        if not merged:
            return None
        return CopyCoalescing(
            current,
            verify_coalesced_copies(
                kernel,
                current,
                objects,
                banks,
                adapter,
                merge_law=merge_law,
                external_windows=external_windows,
                private_ownership=private_ownership,
                max_calls=max_calls,
                max_objects=max_objects,
                max_merges=max_merges,
            ),
        )
    except StorageOptimizationUnavailable:
        return None
    finally:
        _unchanged(before, _snapshot((kernel,), *arguments))
