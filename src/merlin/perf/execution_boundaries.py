"""Summarize provider-decoded call and stack boundaries without pricing them.

Providers own instruction decoding, ABI facts and the exact execution census.
This module checks complete, nonoverlapping extents, retains unknown counts,
and supplies an unpriced training-domain check. It has no execution chronology,
interprocedural dependency, peak stack usage or physical memory interpretation.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_sha256


@dataclass(frozen=True)
class FunctionExtent:
    identity: str
    start: int
    end: int
    entry: int
    frame_bytes: int | None


@dataclass(frozen=True)
class BoundaryInstruction:
    address: int
    width_bytes: int
    executions: int | None
    call: Literal["none", "direct", "indirect", "unknown"]
    callee: str | None
    stack_read_bytes: int | None
    stack_write_bytes: int | None


def _integer(value: int | None, label: str, *, unknown: bool = False) -> None:
    if value is None and unknown:
        return
    if type(value) is not int or value < 0:
        raise ValueError(label + " must be a nonnegative integer" + (" or UNKNOWN" if unknown else ""))


def _sum(values: Sequence[int | None]) -> int | None:
    return None if None in values else sum(values)


def _weighted(count: int | None, amount: int | None) -> int | None:
    if count == 0 or amount == 0:
        return 0
    return None if count is None or amount is None else count * amount


def summarize_boundaries(
    functions: Sequence[FunctionExtent],
    instructions: Sequence[BoundaryInstruction],
    *,
    program_sha256: str,
    census_sha256: str,
    provider_sha256: str,
) -> dict:
    """Require one decoded record per instruction in every selected function.

    Zero counts and zero stack widths are explicit provider facts. Missing or
    undecoded boundary facts use None/unknown, never an implicit zero. Symbols
    bind instances only. Extents cannot overlap or hide omitted instructions.
    The caller must independently verify the supplied artifact digests.
    """
    if not functions or not all(is_sha256(value) for value in (program_sha256, census_sha256, provider_sha256)):
        raise ValueError("selected functions and exact program/census/provider digests required")
    extents = {}
    for function in functions:
        if not isinstance(function, FunctionExtent) or not isinstance(function.identity, str) or not function.identity:
            raise ValueError("typed function extent with nonempty identity required")
        if function.identity in extents:
            raise ValueError("duplicate function identity")
        for label in ("start", "end", "entry"):
            _integer(getattr(function, label), label)
        _integer(function.frame_bytes, "frame bytes", unknown=True)
        if not function.start <= function.entry < function.end:
            raise ValueError("invalid function extent/entry")
        extents[function.identity] = function
    ordered = sorted(functions, key=lambda item: item.start)
    if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
        raise ValueError("function extents overlap")
    by_address = {}
    for instruction in instructions:
        if not isinstance(instruction, BoundaryInstruction):
            raise ValueError("typed provider instruction required")
        for label in ("address", "width_bytes"):
            _integer(getattr(instruction, label), label)
        if not instruction.width_bytes or instruction.address in by_address:
            raise ValueError("zero-width or duplicate instruction")
        _integer(instruction.executions, "execution count", unknown=True)
        for label in ("stack_read_bytes", "stack_write_bytes"):
            _integer(getattr(instruction, label), label, unknown=True)
        if instruction.call not in ("none", "direct", "indirect", "unknown"):
            raise ValueError("unknown call classification spelling")
        if instruction.callee is not None and (
            not isinstance(instruction.callee, str) or instruction.call != "direct" or instruction.callee not in extents
        ):
            raise ValueError("callee requires a direct call to a selected function")
        by_address[instruction.address] = instruction
    rows, consumed = {}, set()
    addresses = sorted(by_address)
    for function in ordered:
        body = [
            by_address[address]
            for address in addresses[bisect_left(addresses, function.start) : bisect_left(addresses, function.end)]
        ]
        cursor = function.start
        for item in body:
            if item.address != cursor or item.address + item.width_bytes > function.end:
                raise ValueError("instruction coverage has a gap, overlap or split extent")
            consumed.add(item.address)
            cursor += item.width_bytes
        if cursor != function.end or function.entry not in by_address:
            raise ValueError("incomplete function coverage or entry splits an instruction")
        entries = by_address[function.entry].executions
        reads = [
            _weighted(item.executions, None if item.stack_read_bytes is None else int(item.stack_read_bytes > 0))
            for item in body
        ]
        writes = [
            _weighted(item.executions, None if item.stack_write_bytes is None else int(item.stack_write_bytes > 0))
            for item in body
        ]
        repeated = []
        for item in body:
            if item.executions == 0 or item.stack_read_bytes == item.stack_write_bytes == 0:
                repeated.append(0)
            elif (
                item.executions is None
                or entries is None
                or item.stack_read_bytes is None
                or item.stack_write_bytes is None
            ):
                repeated.append(None)
            else:
                repeated.append(int(item.executions > entries))
        calls = {}
        for kind in ("direct", "indirect"):
            calls[kind] = _sum(
                [
                    item.executions
                    if item.call == kind
                    else _weighted(item.executions, None)
                    if item.call == "unknown"
                    else 0
                    for item in body
                ]
            )
        rows[function.identity] = dict(
            start=function.start,
            end=function.end,
            entry=function.entry,
            entries=entries,
            frame_bytes=function.frame_bytes,
            frame_entry_bytes=_weighted(entries, function.frame_bytes),
            executed_instructions=_sum([item.executions for item in body]),
            direct_calls=calls["direct"],
            indirect_calls=calls["indirect"],
            unresolved_direct_calls=_sum(
                [
                    item.executions if item.call == "direct" else _weighted(item.executions, None)
                    for item in body
                    if (item.call == "direct" and item.callee is None) or item.call == "unknown"
                ]
            ),
            known_direct_callees={
                callee: _sum([item.executions for item in body if item.call == "direct" and item.callee == callee])
                for callee in sorted({item.callee for item in body if item.callee is not None})
            },
            stack_reads=_sum(reads),
            stack_writes=_sum(writes),
            stack_read_bytes=_sum([_weighted(item.executions, item.stack_read_bytes) for item in body]),
            stack_write_bytes=_sum([_weighted(item.executions, item.stack_write_bytes) for item in body]),
            repeated_stack_sites=_sum(repeated),
        )
    if consumed != by_address.keys():
        raise ValueError("instruction outside selected function extents")
    record = dict(
        schema="execution_boundary_summary_v1",
        functions=rows,
        program_sha256=program_sha256,
        census_sha256=census_sha256,
        provider_sha256=provider_sha256,
        provider_records_sha256=canonical_sha256(
            dict(
                functions=[vars(item) for item in ordered],
                instructions=[vars(by_address[address]) for address in sorted(by_address)],
            )
        ),
        unknown=[
            "Cross-call/block/loop dependency ordering",
            "Memory/stack address chronology and physical traffic",
            "Peak simultaneous stack usage",
            "Instruction service time and cycles",
        ],
    )
    record["summary_sha256"] = canonical_sha256(record)
    return record


def boundary_features(summary: Mapping, *, entry_function: str) -> dict[str, float | None]:
    """Normalize dynamic totals to an explicitly selected complete entry call.

    Selected extents must cover the intended observation; a missing callee is
    explicitly counted, and no recursive stack depth or call order is inferred.
    Static repeated-stack-site and frame features are not divided by entries.
    """
    if summary.get("schema") != "execution_boundary_summary_v1" or summary.get("summary_sha256") != canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    ):
        raise ValueError("boundary summary identity changed")
    functions = summary["functions"]
    if entry_function not in functions:
        raise ValueError("selected entry function is unavailable")
    entries = functions[entry_function]["entries"]
    features = {}
    for field in (
        "executed_instructions",
        "direct_calls",
        "indirect_calls",
        "unresolved_direct_calls",
        "stack_reads",
        "stack_writes",
        "stack_read_bytes",
        "stack_write_bytes",
        "frame_entry_bytes",
    ):
        total = _sum([row[field] for row in functions.values()])
        features["/boundaries/" + field + "_per_entry"] = total / entries if total is not None and entries else None
    features["/boundaries/repeated_stack_sites"] = _sum([row["repeated_stack_sites"] for row in functions.values()])
    frames = [row["frame_bytes"] for row in functions.values() if row["executed_instructions"] != 0]
    features["/boundaries/max_executed_frame_bytes"] = (
        None
        if None in frames or any(row["executed_instructions"] is None for row in functions.values())
        else max(frames, default=0)
    )
    for pointer, value in features.items():
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("nonfinite or negative boundary feature: " + pointer)
    return features


@dataclass(frozen=True)
class BoundaryDomain:
    domain_sha256: str
    bounds: tuple[tuple[str, float, float], ...]
    evidence_sha256: str

    def __post_init__(self) -> None:
        if not is_sha256(self.domain_sha256) or not is_sha256(self.evidence_sha256) or not self.bounds:
            raise ValueError("boundary domain requires exact domain/evidence and nonempty bounds")
        if len({row[0] for row in self.bounds}) != len(self.bounds):
            raise ValueError("duplicate boundary domain pointer")
        for pointer, lower, upper in self.bounds:
            if (
                not isinstance(pointer, str)
                or not pointer.startswith("/boundaries/")
                or any(
                    isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in (lower, upper)
                )
                or not 0 <= lower <= upper
            ):
                raise ValueError("invalid boundary domain pointer or bounds")

    def admit(self, features: Mapping[str, float | None], *, domain_sha256: str) -> dict:
        """Check this unpriced boundary subdomain; never approve a ranker."""
        reasons = []
        if domain_sha256 != self.domain_sha256:
            reasons.append("target, timing scope or execution regime differs")
        for pointer, lower, upper in self.bounds:
            value = features.get(pointer)
            if value is None:
                reasons.append(pointer + " is UNKNOWN")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= value <= upper:
                reasons.append(pointer + " is outside the training boundary domain")
        return dict(
            status="unknown" if reasons else "supported_boundary_subdomain",
            reasons=reasons,
            evidence_sha256=self.evidence_sha256,
            cycles="UNKNOWN",
            ranking_approved=False,
        )


def derive_boundary_domain(
    training: Sequence[Mapping], *, pointers: Sequence[str], entry_function: str, domain_sha256: str
) -> BoundaryDomain:
    """Derive unpriced bounds from training only, with no held-out expansion."""
    if not training or not pointers or len(set(pointers)) != len(pointers) or not is_sha256(domain_sha256):
        raise ValueError("training, distinct boundary pointers and exact domain required")
    vectors = [boundary_features(summary, entry_function=entry_function) for summary in training]
    bounds = []
    for pointer in pointers:
        values = [vector.get(pointer) for vector in vectors]
        if None in values:
            raise ValueError("training boundary feature is UNKNOWN or unavailable: " + str(pointer))
        bounds.append((pointer, min(values), max(values)))
    return BoundaryDomain(
        domain_sha256,
        tuple(bounds),
        canonical_sha256(
            dict(
                training=[summary["summary_sha256"] for summary in training],
                pointers=list(pointers),
                entry_function=entry_function,
                domain_sha256=domain_sha256,
            )
        ),
    )
