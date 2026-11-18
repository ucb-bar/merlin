"""How much of a workload's arithmetic actually reaches the accelerator.

THE METRIC PHASE-2 NEEDS. A compiler is improved for a FLEET of workloads, not for one, and the
question that decides whether an accelerator earns its area is not "did this model get faster" but
"what fraction of this model's arithmetic did the compiler manage to place on the unit, and what is
left on the host". That fraction is derivable from the emitted command buffer alone, in
milliseconds, for every workload at once -- which is what makes it usable as a portfolio objective
instead of a post-hoc report.

WHY NOT :mod:`merlin.analysis.programs.work_volume`. That module answers the compute-axis question (how many MACs
does this program do) and refuses a command whose geometry it cannot read from operand SHAPES. A
backend that PREPACKS its weights has already rewritten those shapes -- measured: it reports 2,048,000
MACs for a ResNet-50 whose real contraction work is ~4.09e9, a 2000x under-count, because 53 of 54
commands are refused. This module reads the geometry the command itself DECLARES (kernel, stride,
padding, dilation plus the destination extent), which survives prepacking. The two are complements:
work_volume is strict about shapes, this one is strict about declarations.

FAIL CLOSED, AND SAY WHICH WAY. A command whose geometry is incomplete is UNKNOWN, never zero, and it
makes the routed total a LOWER bound and the offload fraction an UPPER bound -- stated in that
direction because an unreadable command is one whose work we cannot credit, and silently crediting it
would flatter the accelerator. Nothing here names a target, an opcode family, or a lane spelling:
opcodes come from the buffer, lanes from its own placement record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.analysis.kernels.cca import asm_provenance
from merlin.analysis.kernels.cca.abstraction import (
    CCA,
    CommunicationFacet,
    ComputeFacet,
    DispatchFacet,
    MemoryFacet,
)
from merlin.analysis.observation.compute_endpoints import endpoints_for
from merlin.analysis.observation.rocc_commands import funct_table_for

# ---- offload -----------------------------------------------------------------------------------

#: Attribute names a convolution command uses to declare its own geometry. Read by NAME from the
#: command's declaration, so a prepacked operand shape cannot invalidate them.
_KERNEL = "kernel"
_STRIDE = "stride"
_STRIDES = "strides"
_DILATION = "dilation"


@dataclass(frozen=True)
class CommandMacs:
    index: int
    opcode: str
    macs: int | None
    basis: str
    refusal: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "opcode": self.opcode,
            "macs": self.macs,
            "basis": self.basis,
            "refusal": self.refusal,
        }


@dataclass(frozen=True)
class OffloadReport:
    """What reached the unit, what did not, and how sure we are."""

    commands: tuple[CommandMacs, ...]
    #: MACs the accelerator commands account for. A LOWER bound when anything was refused.
    routed_macs: int
    #: Contraction regions the program placed on an accelerator lane, and on any other lane.
    contractions_on_unit: int
    contractions_off_unit: int
    #: (lane, family) -> region count, straight from the program's placement record.
    regions_by_lane_family: Mapping[tuple[str, str], int]
    routed_is_lower_bound: bool
    refusals: tuple[str, ...] = ()

    @property
    def contraction_offload_fraction(self) -> float | None:
        """Contraction regions on the unit / all contraction regions, or None when none are declared.

        Region COUNT, not work -- a count treats a 1x1 conv and a 7x7 conv alike. Use it alongside
        ``routed_macs``, never instead of it.
        """
        total = self.contractions_on_unit + self.contractions_off_unit
        return None if total == 0 else self.contractions_on_unit / total

    @property
    def host_regions(self) -> int:
        unit = _unit_lanes(self.regions_by_lane_family)
        return sum(n for (lane, _), n in self.regions_by_lane_family.items() if lane not in unit)

    @property
    def unit_regions(self) -> int:
        unit = _unit_lanes(self.regions_by_lane_family)
        return sum(n for (lane, _), n in self.regions_by_lane_family.items() if lane in unit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "routed_macs": self.routed_macs,
            "routed_is_lower_bound": self.routed_is_lower_bound,
            "contractions_on_unit": self.contractions_on_unit,
            "contractions_off_unit": self.contractions_off_unit,
            "contraction_offload_fraction": self.contraction_offload_fraction,
            "unit_regions": self.unit_regions,
            "host_regions": self.host_regions,
            "regions_by_lane_family": {
                f"{l}/{f}": n for (l, f), n in sorted(self.regions_by_lane_family.items(), key=lambda kv: -kv[1])
            },
            "refusals": list(self.refusals),
        }


def _unit_lanes(by_lane: Mapping[tuple[str, str], int]) -> frozenset[str]:
    """Lanes the program's own record calls accelerator lanes. Derived from the record, never named
    here: a hardcoded spelling would make every other target read as 100% host."""
    return frozenset(lane for (lane, _) in by_lane if "mesh" in lane or "accel" in lane or "unit" in lane)


def _ints(value: Any, want: int | None = None) -> tuple[int, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        out.append(item)
    if want is not None and len(out) != want:
        return None
    return tuple(out)


def _shape(tensors: Mapping[str, Any], name: Any) -> tuple[int, ...] | None:
    spec = tensors.get(name) if isinstance(name, str) else None
    return _ints((spec or {}).get("shape")) if isinstance(spec, Mapping) else None


def _elements(shape: tuple[int, ...] | None) -> int | None:
    if shape is None:
        return None
    total = 1
    for dim in shape:
        total *= dim
    return total


def offload_report(command_buffer: Mapping[str, Any]) -> OffloadReport:
    """Derive the offload accounting of ``command_buffer``."""
    tensors = command_buffer.get("tensors")
    tensors = tensors if isinstance(tensors, Mapping) else {}
    commands = command_buffer.get("commands")
    commands = commands if isinstance(commands, Sequence) else ()

    # An accelerator command may write an ACCUMULATOR handle rather than a DRAM tensor, in which
    # case its output extent is only knowable from the commit that drains it. Resolve that from the
    # buffer's own structure (src handle -> committed dst) rather than assuming a naming rule.
    committed: dict[str, tuple[int, ...]] = {}
    for command in commands:
        if not isinstance(command, Mapping):
            continue
        ops = command.get("operands")
        if not isinstance(ops, Mapping):
            continue
        src, dst_name = ops.get("src"), ops.get("dst")
        shape = _shape(tensors, dst_name)
        if isinstance(src, str) and shape is not None and src not in tensors:
            committed[src] = shape

    rows: list[CommandMacs] = []
    refusals: list[str] = []
    routed = 0

    for index, command in enumerate(commands):
        if not isinstance(command, Mapping):
            continue
        opcode = str(command.get("opcode") or "")
        attrs = command.get("attributes")
        attrs = attrs if isinstance(attrs, Mapping) else {}
        operands = command.get("operands")
        operands = operands if isinstance(operands, Mapping) else {}
        macs, basis, why = _command_macs(opcode, attrs, operands, tensors, committed)
        rows.append(CommandMacs(index, opcode, macs, basis, why))
        if macs is None:
            if why:
                refusals.append(f"command {index} ({opcode}): {why}")
        else:
            routed += macs

    params = command_buffer.get("params")
    placement = params.get("lane_placement") if isinstance(params, Mapping) else None
    by_lane: dict[tuple[str, str], int] = {}
    on_unit = off_unit = 0
    if isinstance(placement, Sequence) and not isinstance(placement, (str, bytes)):
        for entry in placement:
            if not isinstance(entry, Mapping):
                continue
            lane = str(entry.get("lane") or "")
            family = str(entry.get("family") or "")
            by_lane[(lane, family)] = by_lane.get((lane, family), 0) + 1
        unit = _unit_lanes(by_lane)
        for entry in placement:
            if not isinstance(entry, Mapping) or str(entry.get("family") or "") != "contraction":
                continue
            if str(entry.get("lane") or "") in unit:
                on_unit += 1
            else:
                off_unit += 1
    elif placement is not None:
        refusals.append("params.lane_placement is present but not a sequence of mappings")

    return OffloadReport(tuple(rows), routed, on_unit, off_unit, by_lane, bool(refusals), tuple(refusals))


def _command_macs(
    opcode: str,
    attrs: Mapping[str, Any],
    operands: Mapping[str, Any],
    tensors: Mapping[str, Any],
    committed: Mapping[str, tuple[int, ...]],
) -> tuple[int | None, str, str | None]:
    """MACs for one accelerator command, from what the command DECLARES.

    Convolution: ``prod(destination extent) * Ci * Kh * Kw`` -- every output element costs one MAC
    per input channel per kernel tap, and the destination extent already carries batch and the
    output spatial dims, so no stride/padding arithmetic is re-derived here (and cannot disagree
    with what the compiler actually scheduled).
    Contraction: ``prod(destination extent) * K``, with ``K`` the reduced extent of the left operand.
    """
    kernel = _ints(attrs.get(_KERNEL), 4)
    dst_name = operands.get("dst")
    dst = _shape(tensors, dst_name)
    if dst is None and isinstance(dst_name, str):
        dst = committed.get(dst_name)  # written to an accumulator, drained by a commit
    if kernel is not None:
        out_elems = _elements(dst)
        if out_elems is None:
            return None, "conv_declared_geometry", "destination extent is not a shape of ints"
        kh, kw, ci, _co = kernel
        return out_elems * ci * kh * kw, "conv_declared_geometry", None
    lhs = _shape(tensors, operands.get("lhs"))
    if lhs is not None and dst is not None and len(lhs) >= 1:
        out_elems = _elements(dst)
        if out_elems is None:
            return None, "contraction_declared_extents", "destination extent unreadable"
        return out_elems * lhs[-1], "contraction_declared_extents", None
    # Movement / packing / eviction commands do no arithmetic; that is not a refusal.
    if any(token in opcode for token in ("PACK", "EVICT", "COMMIT", "MOVE", "COPY", "FENCE")):
        return 0, "no_arithmetic", None
    return None, "unrecognised", f"opcode {opcode!r} declares no geometry this module can read"


def format_report(report: OffloadReport, *, label: str = "") -> str:
    """A short table. Reporting only -- callers compare ``to_dict()``."""
    frac = report.contraction_offload_fraction
    frac_s = "n/a" if frac is None else f"{100.0 * frac:.1f}%"
    bound = " (LOWER bound)" if report.routed_is_lower_bound else ""
    out = [
        f"== offload {label}".rstrip(),
        f"  routed MACs           {report.routed_macs:>15,}{bound}",
        f"  contractions on unit  {report.contractions_on_unit:>15,}",
        f"  contractions off unit {report.contractions_off_unit:>15,}",
        f"  contraction offload   {frac_s:>15}",
        f"  regions on unit/host  {report.unit_regions:>7,} / {report.host_regions:,}",
    ]
    if report.regions_by_lane_family:
        out.append("  host families:")
        unit = _unit_lanes(report.regions_by_lane_family)
        for (lane, fam), n in sorted(report.regions_by_lane_family.items(), key=lambda kv: -kv[1]):
            if lane not in unit:
                out.append(f"      {n:5,}  {fam or '<unnamed>'}")
    for why in report.refusals[:5]:
        out.append(f"  REFUSAL {why}")
    if len(report.refusals) > 5:
        out.append(f"  ... and {len(report.refusals) - 5} more refusals")
    return "\n".join(out)


# ---- artifact_activity -------------------------------------------------------------------------
# Lift one emitted target artifact into program-scope optimization evidence.
#
# The command buffer says what the compiler *declared*.  This module answers the complementary
# question: what did the lowered artifact actually issue?  It deliberately reuses the target-derived
# funct table, endpoint role vocabulary, CCA facets, and role-to-owner provenance already maintained by
# Merlin.  No target opcode or compiler filename is named here.
#
# This is structural evidence, not a cycle predictor.  It can count issued movement, configuration,
# compute, loop-descriptor and synchronization instructions and can expose a visible DMA/wait overlap
# shape.  It cannot infer dynamic occupancy, cache contention, or elapsed cycles from static code; those
# remain UNKNOWN until a reduced warm profile supplies counters.

_MOVEMENT_ROLES = frozenset(
    {
        "operand_load",
        "weight_load",
        "move",
        "readout",
        "commit",
        "dma",
    }
)
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
        gaps.append(
            sum(1 for row in stream if issue < row.index < wait and not set(row.roles).intersection({"dma", "sync"}))
        )
    if not gaps:
        return None, None
    return any(gap > 0 for gap in gaps), max(gaps)


def analyze_artifact_activity(trace: Mapping[str, Any], *, target: str, op: str = "model") -> dict[str, Any]:
    """Summarize a decoded lowered artifact using only target-derived semantic roles.

    ``trace`` is the output of :func:`merlin.target.isa.rocc.decode.decode_text`.  A custom
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
            "reason": (
                "the lowered artifact contains no decoded instructions; absence is not zero "
                "movement or a complete encoding"
            ),
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
        roles = (
            tuple(sorted({role for endpoint in endpoints for role in endpoint.roles_of(identity)})) if identity else ()
        )

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
    opportunities = asm_provenance.opportunities(role_counts, engine=engine, total=len(tagged), family=None)
    role_owners = {role: asm_provenance.provenance_of_role(role).to_dict() for role in sorted(role_counts)}

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


__all__ = [
    "CommandMacs",
    "OffloadReport",
    "format_report",
    "offload_report",
]
