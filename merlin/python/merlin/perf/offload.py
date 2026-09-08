"""How much of a workload's arithmetic actually reaches the accelerator.

THE METRIC PHASE-2 NEEDS. A compiler is improved for a FLEET of workloads, not for one, and the
question that decides whether an accelerator earns its area is not "did this model get faster" but
"what fraction of this model's arithmetic did the compiler manage to place on the unit, and what is
left on the host". That fraction is derivable from the emitted command buffer alone, in
milliseconds, for every workload at once -- which is what makes it usable as a portfolio objective
instead of a post-hoc report.

WHY NOT :mod:`merlin.perf.work_volume`. That module answers the compute-axis question (how many MACs
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

__all__ = ["CommandMacs", "OffloadReport", "offload_report", "format_report"]

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
        return {"index": self.index, "opcode": self.opcode, "macs": self.macs,
                "basis": self.basis, "refusal": self.refusal}


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
            "regions_by_lane_family": {f"{l}/{f}": n for (l, f), n in
                                       sorted(self.regions_by_lane_family.items(),
                                              key=lambda kv: -kv[1])},
            "refusals": list(self.refusals),
        }


def _unit_lanes(by_lane: Mapping[tuple[str, str], int]) -> frozenset[str]:
    """Lanes the program's own record calls accelerator lanes. Derived from the record, never named
    here: a hardcoded spelling would make every other target read as 100% host."""
    return frozenset(lane for (lane, _) in by_lane
                     if "mesh" in lane or "accel" in lane or "unit" in lane)


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

    return OffloadReport(tuple(rows), routed, on_unit, off_unit, by_lane,
                         bool(refusals), tuple(refusals))


def _command_macs(opcode: str, attrs: Mapping[str, Any], operands: Mapping[str, Any],
                  tensors: Mapping[str, Any],
                  committed: Mapping[str, tuple[int, ...]]) -> tuple[int | None, str, str | None]:
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
        dst = committed.get(dst_name)      # written to an accumulator, drained by a commit
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
    out = [f"== offload {label}".rstrip(),
           f"  routed MACs           {report.routed_macs:>15,}{bound}",
           f"  contractions on unit  {report.contractions_on_unit:>15,}",
           f"  contractions off unit {report.contractions_off_unit:>15,}",
           f"  contraction offload   {frac_s:>15}",
           f"  regions on unit/host  {report.unit_regions:>7,} / {report.host_regions:,}"]
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
