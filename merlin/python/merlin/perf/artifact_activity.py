"""Lift one emitted target artifact into program-scope optimization evidence.

The command buffer says what the compiler *declared*.  This module answers the complementary
question: what did the lowered artifact actually issue?  It deliberately reuses the target-derived
funct table, endpoint role vocabulary, CCA facets, and role-to-owner provenance already maintained by
Merlin.  No target opcode or compiler filename is named here.

This is structural evidence, not a cycle predictor.  It can count issued movement, configuration,
compute, loop-descriptor and synchronization instructions and can expose a visible DMA/wait overlap
shape.  It cannot infer dynamic occupancy, cache contention, or elapsed cycles from static code; those
remain UNKNOWN until a reduced warm profile supplies counters.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.kernels import asm_provenance
from merlin.kernels.cca import CCA, CommunicationFacet, ComputeFacet, DispatchFacet, MemoryFacet
from merlin.kernels.decode.rocc import funct_table_for
from merlin.kernels.endpoints import endpoints_for


_MOVEMENT_ROLES = frozenset({
    "operand_load", "weight_load", "move", "readout", "commit", "dma",
})
_COMPUTE_ROLES = frozenset({"accumulate", "elementwise"})


@dataclass(frozen=True)
class TaggedInstruction:
    """The small interface consumed by the existing role/CCA analysis."""

    index: int
    roles: tuple[str, ...]
    identity: str = ""
    instruction_class: str = "UNKNOWN"


def _funct_names(table: Mapping[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for raw, name in (table.get("names") or {}).items():
        try:
            out[int(raw)] = str(name)
        except (TypeError, ValueError):
            continue
    return out


def _dma_overlap(stream: Sequence[TaggedInstruction]) -> tuple[bool | None, int | None]:
    """Return the same observable DMA-to-wait shape used by the CCA role lifter.

    Architectural fences are tagged as ``sync`` by this adapter because the target decoder has
    already identified them structurally.  A DMA with no visible later wait remains UNKNOWN; an
    interlock or a decoder blind spot cannot be distinguished from this artifact alone.
    """
    issues = [row.index for row in stream if "dma" in row.roles]
    if not issues:
        return None, None
    waits = [row.index for row in stream if "sync" in row.roles]
    if not waits:
        return None, None
    gaps: list[int] = []
    for issue in issues:
        wait = next((index for index in waits if index > issue), None)
        if wait is None:
            continue
        gaps.append(sum(
            1 for row in stream
            if issue < row.index < wait and not set(row.roles).intersection({"dma", "sync"})
        ))
    if not gaps:
        return None, None
    return any(gap > 0 for gap in gaps), max(gaps)


def analyze_artifact_activity(trace: Mapping[str, Any], *, target: str,
                              op: str = "model") -> dict[str, Any]:
    """Summarize a decoded lowered artifact using only target-derived semantic roles.

    ``trace`` is the output of :func:`merlin.targetgen.rocc.decode.decode_text`.  A custom
    instruction whose encoding is recognized but has no endpoint role is kept in the explicit
    ``named_without_role`` bucket.  An unrecognized encoding is kept in ``unknown``.  Neither is
    silently treated as useful work or as zero movement.
    """
    instructions = trace.get("instructions")
    if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
        raise ValueError("decoded trace instructions must be a sequence")
    if not instructions:
        return {
            "schema": "emitted_artifact_activity_v1",
            "status": "UNKNOWN",
            "target": target,
            "reason": ("the lowered artifact contains no decoded instructions; absence is not zero "
                       "movement or a complete encoding"),
            "instruction_count": 0,
            "endpoint_instruction_count": None,
            "class_histogram": {},
            "identity_histogram": {},
            "role_counts": {},
            "issued": {
                "movement_instructions": None,
                "compute_instructions": None,
                "configuration_instructions": None,
                "loop_descriptor_instructions": None,
                "synchronization_instructions": None,
                "dma_instructions": None,
            },
            "encoding_resolution": {
                "status": "UNKNOWN",
                "unknown_instruction_indices": [],
                "named_without_role": [],
                "reason": "there is no target instruction stream to decode",
            },
            "program_cca": None,
            "artifact_opportunities": [],
            "role_to_compiler_owner": {},
            "dynamic_unknowns": {
                "cycles": "UNKNOWN: static artifact analysis is not a timing measurement",
                "occupancy": "UNKNOWN: requires executed event/counter evidence",
                "contention": "UNKNOWN: requires executed resource evidence",
                "resident_across_calls": "UNKNOWN: no target instruction stream was emitted",
            },
        }

    table = funct_table_for(target)
    names = _funct_names(table)
    endpoints = endpoints_for(target)
    tagged: list[TaggedInstruction] = []
    class_histogram: dict[str, int] = {}
    identity_histogram: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    endpoint_count = 0
    unknown: list[int] = []
    named_without_role: list[dict[str, Any]] = []

    for fallback_index, raw in enumerate(instructions):
        if not isinstance(raw, Mapping):
            unknown.append(fallback_index)
            continue
        index = raw.get("index")
        index = int(index) if isinstance(index, int) and not isinstance(index, bool) else fallback_index
        instruction_class = str(raw.get("class") or "UNKNOWN")
        class_histogram[instruction_class] = class_histogram.get(instruction_class, 0) + 1
        funct = raw.get("funct")
        identity = names.get(funct, "") if isinstance(funct, int) and not isinstance(funct, bool) else ""
        roles = tuple(sorted({
            role for endpoint in endpoints for role in endpoint.roles_of(identity)
        })) if identity else ()

        # A CPU/architectural fence is outside an endpoint's funct table, but the decoder identified
        # it exactly.  Carry it into the shared semantic vocabulary so program synchronization is not
        # invisible merely because it sits around, rather than inside, the accelerator endpoint.
        if instruction_class == "FENCE":
            roles = tuple(sorted(set(roles).union({"sync"})))
        if funct is not None:
            endpoint_count += 1
            if identity:
                identity_histogram[identity] = identity_histogram.get(identity, 0) + 1
                if not roles:
                    named_without_role.append({"index": index, "identity": identity})
            else:
                unknown.append(index)
        elif instruction_class == "UNKNOWN":
            unknown.append(index)
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
        tagged.append(TaggedInstruction(index, roles, identity, instruction_class))

    def count_any(wanted: frozenset[str]) -> int:
        return sum(bool(wanted.intersection(row.roles)) for row in tagged)

    config_count = count_any(frozenset({"config"}))
    compute_count = count_any(_COMPUTE_ROLES)
    movement_count = count_any(_MOVEMENT_ROLES)
    loop_count = count_any(frozenset({"loop_descriptor"}))
    sync_count = count_any(frozenset({"sync"}))
    dma_count = count_any(frozenset({"dma"}))
    overlap, issue_to_wait = _dma_overlap(tagged)
    role_evidence = sum(1 for row in tagged if row.roles)

    dispatch = DispatchFacet(
        n_dispatches=endpoint_count or None,
        config_fraction=(config_count / endpoint_count) if endpoint_count else None,
        descriptor_reuse=(config_count <= 1) if compute_count else None,
        loop_offloaded=(bool(loop_count) if (compute_count or loop_count) else None),
        dma_overlap=overlap,
        dma_issue_to_wait=issue_to_wait,
    )
    communication = CommunicationFacet(
        mechanism="dma" if dma_count else None,
        copy_compute_overlap=overlap,
        fences=sync_count,
    )
    cca = CCA(
        op=op,
        backend=[target],
        compute=ComputeFacet(op=op),
        memory=MemoryFacet(dma_pattern="burst" if dma_count else None),
        dispatch=dispatch,
        communication=communication,
        scope="program",
        provenance={
            "level": "emitted_target_artifact",
            "source": trace.get("source"),
            "confidence": "low" if unknown or named_without_role or not role_evidence else "high",
            "role_counts": dict(sorted(role_counts.items())),
            "unknown_instruction_indices": sorted(set(unknown)),
            "named_without_role": named_without_role,
        },
    )

    engine = ""
    engines = {endpoint.engine for endpoint in endpoints if endpoint.engine}
    if len(engines) == 1:
        engine = next(iter(engines))
    opportunities = asm_provenance.opportunities(
        role_counts, engine=engine, total=len(tagged), family=None)
    role_owners = {
        role: asm_provenance.provenance_of_role(role).to_dict()
        for role in sorted(role_counts)
    }

    return {
        "schema": "emitted_artifact_activity_v1",
        "status": "decoded",
        "target": target,
        "instruction_count": len(tagged),
        "endpoint_instruction_count": endpoint_count,
        "class_histogram": dict(sorted(class_histogram.items())),
        "identity_histogram": dict(sorted(identity_histogram.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "issued": {
            "movement_instructions": movement_count,
            "compute_instructions": compute_count,
            "configuration_instructions": config_count,
            "loop_descriptor_instructions": loop_count,
            "synchronization_instructions": sync_count,
            "dma_instructions": dma_count,
        },
        "encoding_resolution": {
            "status": "complete" if not unknown and not named_without_role else "partial",
            "unknown_instruction_indices": sorted(set(unknown)),
            "named_without_role": named_without_role,
        },
        "program_cca": cca.to_dict(),
        "artifact_opportunities": [row.to_dict() for row in opportunities],
        "role_to_compiler_owner": role_owners,
        "dynamic_unknowns": {
            "cycles": "UNKNOWN: static artifact analysis is not a timing measurement",
            "occupancy": "UNKNOWN: requires executed event/counter evidence",
            "contention": "UNKNOWN: requires executed resource evidence",
            "resident_across_calls": (
                "UNKNOWN here: use the differential trace residency check and command-buffer liveness"
            ),
        },
    }
