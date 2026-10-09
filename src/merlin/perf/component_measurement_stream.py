"""Bounded raw complete-stage events, without timer or stage authority.

The selected producer owns what its boundaries and counters mean. This parser
checks the original ordered roster only; it never computes elapsed cycles or
labels an invocation cold or warm.
"""

from __future__ import annotations

from dataclasses import dataclass

from merlin.perf.component_cost import COMPLETE_STAGES

BEGIN = b"MERLIN_MEASUREMENT_V1"
END = b"MERLIN_MEASUREMENT_END_V1"
UNKNOWN = (
    "stage_boundary_semantics",
    "counter_units_and_interval_integrity",
    "reset_clock_and_reuse_state",
    "cold_and_warm_complete_costs",
    "observer_integrity",
    "resource_and_physical_execution_domain",
    "runtime_stage_and_measurement_qualification",
    "held_applicability_ranking_error_and_interval_coverage",
)


@dataclass(frozen=True)
class RawMeasurementPlan:
    """Explicit original event order and whole-stream budgets; no capability."""

    stage_order: tuple[str, ...]
    max_console_bytes: int
    max_frame_bytes: int

    def record(self):
        if (
            type(self) is not RawMeasurementPlan
            or type(self.stage_order) is not tuple
            or len(self.stage_order) != len(COMPLETE_STAGES)
            or any(type(stage) is not str for stage in self.stage_order)
            or set(self.stage_order) != set(COMPLETE_STAGES)
            or type(self.max_console_bytes) is not int
            or type(self.max_frame_bytes) is not int
            or not 1 <= self.max_frame_bytes <= self.max_console_bytes <= 4 * 1024 * 1024
        ):
            raise ValueError("raw measurement requires a complete ordered stage roster and bounded stream")
        return {
            "schema": "merlin.raw_measurement_plan.v1",
            "stage_order": list(self.stage_order),
            "max_console_bytes": self.max_console_bytes,
            "max_frame_bytes": self.max_frame_bytes,
        }


def _unsigned(token):
    if not token or len(token) > 20 or any(byte not in b"0123456789" for byte in token):
        raise ValueError("raw measurement has an unsupported counter/control value")
    value = int(token)
    if value >= 1 << 64 or str(value).encode("ascii") != token:
        raise ValueError("raw measurement counter/control is outside its declared unsigned width")
    return value


def parse_measurement_stream(console: bytes, *, plan: RawMeasurementPlan):
    """Parse one closed ASCII segment of the exact actually consumed console."""
    selection = plan.record()
    if type(console) is not bytes or len(console) > plan.max_console_bytes:
        raise ValueError("raw measurement whole console exceeds its selected byte budget")
    lines = console.splitlines(keepends=True)
    reserved = [i for i, line in enumerate(lines) if line.startswith(b"MERLIN_MEASUREMENT")]
    if len(reserved) != 2 or lines[reserved[0]] != BEGIN + b"\n" or lines[reserved[1]] != END + b"\n":
        raise ValueError("raw measurement requires exactly one complete unambiguous frame")
    start, stop = reserved
    frame = b"".join(lines[start : stop + 1])
    if len(frame) > plan.max_frame_bytes or stop - start - 1 != 2 * len(plan.stage_order):
        raise ValueError("raw measurement omitted or exceeded its complete event roster")
    events = []
    for line, (stage, edge) in zip(
        lines[start + 1 : stop],
        ((stage, edge) for stage in plan.stage_order for edge in ("begin", "end")),
        strict=True,
    ):
        fields = line.removesuffix(b"\n").split(b" ")
        if not line.endswith(b"\n") or len(fields) != 4 or fields[:2] != [stage.encode("ascii"), edge.encode("ascii")]:
            raise ValueError("raw measurement stage order or boundary differs from its original plan")
        events.append({"stage": stage, "edge": edge, "counter": _unsigned(fields[2]), "control": _unsigned(fields[3])})
    return {
        "schema": "merlin.raw_measurement_events.v1",
        "plan": selection,
        "events": events,
        "frame_bytes": len(frame),
        "stage_roster_observed": list(plan.stage_order),
        "cold": None,
        "warm": None,
        "unknown": list(UNKNOWN),
        "scope": "ordered raw source-produced event accounting only; no elapsed cost or qualification",
    }
