"""Exact measured-work calibration and empirical ceilings for Phase 2."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.perf.envelope import Peak
from merlin.runtime.commandbuffer import batched_matmul_geometry

COMPUTE = "compute"


@dataclass(frozen=True)
class MeasuredPoint:
    """One shape whose work is known exactly and whose cycles were measured by the cycle oracle."""

    capsule: str
    macs: int
    cycles: int
    source: str
    # Reduction depths of the exact compute commands whose MACs were counted.  This is deliberately
    # target-independent: it comes from operand shapes in the command-buffer ABI.  An empty tuple
    # means the command buffer was measured before this evidence was recorded, not a depth of zero.
    reduction_depths: tuple[int, ...] = ()

    @property
    def achieved_rate(self) -> float:
        return self.macs / self.cycles


def harvest_measured_points(run_root: Path) -> tuple[list[MeasuredPoint], list[str]]:
    """Recover (exact work, measured cycles) pairs a completed run already produced.

    The work is the compiler's OWN emitted command buffer priced by
    :func:`merlin.perf.work_volume.work_from_command_buffer`, so it is what the program actually asks
    the array to do, not a shape someone declared. A buffer whose work is a lower bound is skipped
    rather than counted: a rate built on an understated numerator would flatter every candidate.

    Returns the points and the reasons any result was skipped, because a corpus that silently lost
    half its members is indistinguishable from one that never had them.
    """
    from merlin.perf.work_volume import work_from_command_buffer  # noqa: PLC0415

    points: dict[str, MeasuredPoint] = {}
    skipped: list[str] = []
    for result in sorted(Path(run_root).rglob("capsule_result.json")):
        try:
            document = json.loads(result.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable result is reported, never counted
            skipped.append(f"{result.parent.name}: capsule result is unreadable")
            continue
        cycles = ((document.get("tiers") or {}).get("L3") or {}).get("cycles")
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
            continue  # no cycle-oracle verdict here; not an error, just not a measured point
        buffer_path = result.parent / "generated" / "command_buffer.json"
        if not buffer_path.is_file():
            skipped.append(
                f"{result.parent.name}: measured {cycles} cycles but emitted no "
                "command buffer, so its work cannot be priced"
            )
            continue
        try:
            command_buffer = json.loads(buffer_path.read_text(encoding="utf-8"))
            work = work_from_command_buffer(command_buffer)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{result.parent.name}: command buffer did not price ({type(exc).__name__})")
            continue
        if work.is_lower_bound or not work.exact_macs or work.exact_macs <= 0:
            skipped.append(
                f"{result.parent.name}: work is a lower bound or zero "
                f"({len(work.refusals)} refusal(s)), so no rate may be built from it"
            )
            continue
        points.setdefault(
            result.parent.name,
            MeasuredPoint(
                result.parent.name,
                int(work.exact_macs),
                cycles,
                str(result.parent),
                _command_reduction_depths(command_buffer),
            ),
        )
    return sorted(points.values(), key=lambda p: p.capsule), skipped


def _command_reduction_depths(command_buffer: Mapping[str, Any]) -> tuple[int, ...]:
    """Exact contraction depths from the same command buffer whose work was priced.

    A rate reached at deep K is not an attainable rate for a shallow-K member on a machine whose
    fixed issue/fill cost is amortised along K.  Keep this small piece of geometry with every point
    so consumers can compare like with like.  Unsupported geometry returns no signature and must
    not be guessed into a cohort.
    """
    tensors = command_buffer.get("tensors")
    commands = command_buffer.get("commands")
    if not isinstance(tensors, Mapping) or not isinstance(commands, Sequence):
        return ()

    def shape(name: Any) -> tuple[int, ...] | None:
        spec = tensors.get(name) if isinstance(name, str) else None
        raw = spec.get("shape") if isinstance(spec, Mapping) else None
        if (
            not isinstance(raw, Sequence)
            or isinstance(raw, (str, bytes))
            or not raw
            or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in raw)
        ):
            return None
        return tuple(int(v) for v in raw)

    handles: dict[str, str] = {}
    depths: list[int] = []
    for raw in commands:
        if not isinstance(raw, Mapping):
            return ()
        opcode = str(raw.get("opcode") or "")
        operands = raw.get("operands") if isinstance(raw.get("operands"), Mapping) else {}
        if opcode == "RES_PACK":
            src, dst = operands.get("src"), operands.get("dst")
            if isinstance(src, str) and isinstance(dst, str) and shape(src):
                handles[dst] = src
            continue
        if opcode in ("MATMUL", "MATMUL_RESIDENT"):
            rhs = operands.get("rhs")
            if opcode == "MATMUL_RESIDENT":
                rhs = handles.get(rhs) if isinstance(rhs, str) else None
            lhs_shape, rhs_shape = shape(operands.get("lhs")), shape(rhs)
            if not lhs_shape or not rhs_shape or len(lhs_shape) != 2 or len(rhs_shape) != 2:
                return ()
            depths.append(lhs_shape[1])
        elif opcode == "BATCHED_MATMUL":
            try:
                geometry = batched_matmul_geometry(
                    shape(operands.get("a")),
                    shape(operands.get("w")),
                    shape(operands.get("dst")),
                    op="BATCHED_MATMUL reduction depth",
                )
            except ValueError:
                return ()
            depths.append(geometry.k)
        elif opcode == "ATTENTION_QK":
            lhs_shape = shape(operands.get("q"))
            if not lhs_shape or len(lhs_shape) != 2:
                return ()
            depths.append(lhs_shape[1])
        elif opcode == "ATTENTION_PV":
            lhs_shape = shape(operands.get("p"))
            if not lhs_shape or len(lhs_shape) != 2:
                return ()
            depths.append(lhs_shape[1])
        elif opcode == "CONV2D":
            weight = operands.get("weight")
            if isinstance(weight, str) and weight in handles:
                weight = handles[weight]
            weight_shape = shape(weight)
            if not weight_shape or len(weight_shape) != 2:
                return ()
            depths.append(weight_shape[0])
    return tuple(sorted(depths))


def achievable_ceiling(points: Sequence[MeasuredPoint], *, provenance: str) -> Peak:
    """The best rate anything actually reached -- derived, then falsified against every sample.

    :meth:`Peak.observed_ceiling` re-checks ``demand / rate <= busy`` on every point it was built
    from and returns UNKNOWN rather than a rate if any point violates it, so this cannot quietly
    become a nameplate number. ``merlin.perf.roofline`` will only admit a peak of this kind.
    """
    if not points:
        return Peak.unknown(COMPUTE, "mac", "no measured point priced its work", provenance=provenance)
    return Peak.observed_ceiling(COMPUTE, [(p.macs, p.cycles) for p in points], unit="mac", provenance=provenance)
