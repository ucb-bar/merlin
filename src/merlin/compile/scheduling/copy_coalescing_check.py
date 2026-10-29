"""Check actual adjacent-copy effects under a separate original merge law.

Exact-copy effects alone do not authorize combining requests. The caller must
independently qualify the count/alignment envelope and absence of faults or
other program-visible observations. This first family retains existing private
initialization rules and never infers a target's DMA semantics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import lcm

from merlin.sched.ir import Kernel
from merlin.semantic_compiler.allocation_types import StorageBank

from .program_limits import AddressWindow
from .temporary_storage import StorageAdapter, StorageOptimizationUnavailable, TemporaryObject, _overlap, _problem


@dataclass(frozen=True)
class CopyMergeLaw:
    configuration_identity: str
    numerical_policy_identity: str
    adapter_qualification_identity: str
    qualification_identity: str
    max_request_bytes: int
    source_alignment: int = 1
    destination_alignment: int = 1
    size_multiple: int = 1
    no_faults: bool = False
    no_extra_observations: bool = False

    def __post_init__(self):
        if any(
            type(value) is not str or not value.strip()
            for value in (
                self.configuration_identity,
                self.numerical_policy_identity,
                self.adapter_qualification_identity,
                self.qualification_identity,
            )
        ) or any(
            type(value) is not int or value <= 0
            for value in (
                self.max_request_bytes,
                self.source_alignment,
                self.destination_alignment,
                self.size_multiple,
            )
        ):
            raise ValueError("copy merging requires original identities and positive explicit request bounds")
        if type(self.no_faults) is not bool or type(self.no_extra_observations) is not bool:
            raise ValueError("copy merging fault and observation premises must be explicit booleans")

    def record(self):
        self.__post_init__()
        return asdict(self)


def _snapshot(kernels, objects, banks, adapter, external_windows, law, max_calls, max_objects, max_merges):
    if any(type(n) is not int or n <= 0 for n in (max_calls, max_objects)) or (
        type(max_merges) is not int or max_merges < 0
    ):
        raise ValueError("copy merging requires bounded calls, objects and nonnegative merge count")
    if type(adapter) is not StorageAdapter or any(type(k) is not Kernel for k in kernels):
        raise ValueError("copy merging requires original kernels and a typed storage adapter")
    adapter.__post_init__()
    if law is not None and type(law) is not CopyMergeLaw:
        raise ValueError("copy merging requires an original typed merge law")
    for values, cls in ((objects, TemporaryObject), (banks, StorageBank), (external_windows, AddressWindow)):
        if type(values) is not tuple or any(type(value) is not cls for value in values):
            raise ValueError("copy merging requires immutable typed objects, banks and external windows")
        for value in values:
            value.__post_init__()
    return (
        tuple(k.digest() for k in kernels),
        law.record() if law is not None else None,
        tuple(asdict(o) for o in objects),
        tuple(asdict(b) for b in banks),
        tuple(asdict(w) for w in external_windows),
        (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity),
        adapter.project,
        adapter.rewrite,
    )


def _unchanged(before, after):
    if before[:-2] != after[:-2] or before[-2] is not after[-2] or before[-1] is not after[-1]:
        raise ValueError("original copy-merging source, bounds, law or selected adapter changed during a callback")


def _admitted(law, adapter):
    if law is None:
        return False
    law.__post_init__()
    return (
        law.no_faults
        and law.no_extra_observations
        and (
            law.configuration_identity,
            law.numerical_policy_identity,
            law.adapter_qualification_identity,
        )
        == (adapter.configuration_identity, adapter.numerical_policy_identity, adapter.qualification_identity)
    )


def _physical_copy(effect, banks, law):
    effect.__post_init__()
    for access in effect.accesses:
        access.__post_init__()
    if not effect.exact_copy or effect.completion != "synchronous":
        return False
    source, destination = effect.accesses
    # External names have no cross-name alias model. At least one endpoint must
    # be a private bank, independently disjoint from external windows.
    if source.space not in banks and destination.space not in banks:
        return False
    if _overlap(source, destination, banks) or source.size > law.max_request_bytes:
        return False
    for access, alignment in zip((source, destination), (law.source_alignment, law.destination_alignment), strict=True):
        if access.space in banks:
            bank = banks[access.space]
            if access.start < bank.start or access.start + access.size > bank.capacity:
                return False
            alignment = lcm(alignment, bank.alignment)
        if access.start % alignment or access.size % law.size_multiple:
            return False
    return True


def _physical_windows(effects, banks):
    return all(
        access.space not in banks
        or banks[access.space].start <= access.start
        and access.start + access.size <= banks[access.space].capacity
        for effect in effects
        for access in effect.accesses
    )


def verify_coalesced_copies(
    original,
    realized,
    objects,
    banks,
    adapter,
    *,
    merge_law=None,
    external_windows=(),
    private_ownership=False,
    max_calls=4096,
    max_objects=256,
    max_merges=128,
):
    """Reconstruct contiguous groups from actual effects, never a proposed plan.

    Unchanged calls remain byte-for-byte IR equal and in their original order.
    Every changed instruction must cover two or more immediately consecutive
    original exact copies, including all source/destination bytes and semantics.
    """
    arguments = (objects, banks, adapter, external_windows, merge_law, max_calls, max_objects, max_merges)
    before = _snapshot((original, realized), *arguments)
    try:
        if original == realized:
            return {"status": "unchanged", "merged_calls": 0}
        if not _admitted(merge_law, adapter):
            raise StorageOptimizationUnavailable("copy merging lacks its original fault/observation merge law")
        if replace(realized, body=original.body) != original or len(realized.body) >= len(original.body):
            raise ValueError("copy merging changed the kernel boundary or added unrelated instructions")
        source_banks, source, *_ = _problem(
            original, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
        )
        _, actual, *_ = _problem(
            realized, objects, banks, adapter, external_windows, private_ownership, max_calls, max_objects
        )
        if any(effect.completion != "synchronous" for effect in (*source, *actual)):
            raise StorageOptimizationUnavailable("copy merging does not admit asynchronous or wait instructions")
        if not _physical_windows(source, source_banks) or not _physical_windows(actual, source_banks):
            raise StorageOptimizationUnavailable(
                "copy merging requires actual bounded addresses, not virtual placement"
            )
        index, removed, groups = 0, 0, []
        for output_index, (call, effect) in enumerate(zip(realized.body, actual, strict=True)):
            if index >= len(original.body):
                raise ValueError("copy merging emitted an extra instruction")
            if call == original.body[index]:
                if effect != source[index]:
                    raise ValueError("unchanged copy instruction decoded inconsistently")
                index += 1
                continue
            if not _physical_copy(effect, source_banks, merge_law):
                raise ValueError("realized merged copy violates physical, alias or request bounds")
            first, count = index, 0
            read, write = effect.accesses
            while count < read.size and index < len(source):
                old = source[index]
                if (
                    original.body[index].instr != call.instr
                    or not _physical_copy(old, source_banks, merge_law)
                    or replace(old, accesses=effect.accesses) != effect
                ):
                    raise ValueError("merged copy changed completion, configuration or instruction semantics")
                old_read, old_write = old.accesses
                if (
                    old_read.space != read.space
                    or old_write.space != write.space
                    or old_read.start != read.start + count
                    or old_write.start != write.start + count
                ):
                    raise ValueError("merged copy skipped, reordered or changed original byte coordinates")
                count += old_read.size
                index += 1
            if count != read.size or index - first < 2:
                raise ValueError("merged instruction does not exactly cover an adjacent original copy group")
            removed += index - first - 1
            if removed > max_merges:
                raise ValueError("copy merging exceeds its original merge budget")
            groups.append({"original_start": first, "original_end": index, "actual_call": output_index, "bytes": count})
        if index != len(source):
            raise ValueError("copy merging dropped original instructions or trailing observations")
        return {
            "schema": "merlin.copy_coalescing.v1",
            "source_sha256": original.digest(),
            "realized_sha256": realized.digest(),
            "merge_law": merge_law.record(),
            "merged_calls": removed,
            "groups": groups,
            "static_commands_before": len(source),
            "static_commands_after": len(actual),
            "requested_read_bytes": sum(a.size for e in actual for a in e.accesses if not a.writes),
            "requested_write_bytes": sum(a.size for e in actual for a in e.accesses if a.writes),
            "machine_emission": "not_established",
            "execution_qualification": "not_established",
            "timing": "UNKNOWN",
        }
    finally:
        _unchanged(before, _snapshot((original, realized), *arguments))
