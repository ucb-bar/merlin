"""Execution contracts for fast iteration and citable milestone measurements.

Search and certification use reduced witnesses and may not schedule a simulator execution longer
than ten minutes.  Full-size model execution is an optional, separate validation tier rather than a
Phase-2 prerequisite; when FireSim is available and explicitly used, it is admitted only through an
external queue.  This module validates requests and receipts; it deliberately does not shell out,
so no library caller can bypass the queue by importing a convenient runner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


ITERATION_MAX_SECONDS = 600.0
# Agent authoring and host-only whole-graph analysis do not execute a model.  Large global compiler
# changes may need more than the reduced-witness simulation ceiling, so keep their bound separate.
GLOBAL_AUTHORING_ROUND_MAX_SECONDS = 1200.0
# Full-graph compilation and static accounting never execute the model.  Keep their host deadline
# separate from the reduced-witness simulation limit.  This ceiling covers one ordered portfolio,
# not one member: four cold full-model lowers can exceed twenty minutes when the resource guard
# serializes workers, even though the normal two-worker wall time remains near ten minutes.
FULL_GRAPH_STATIC_ANALYSIS_MAX_SECONDS = 2400.0
STATIC_PLANNER_MAX_SECONDS = 300.0
FIRESIM_LIFECYCLE = (
    ("firesim", "kill"),
    ("firesim", "infrasetup"),
    ("firesim", "runworkload"),
    ("firesim", "kill"),
)
_PROFILE_METRICS = frozenset({
    "total_compute_cycles",
    "resource_busy_cycles",
    "movement_bytes",
    "movement_commands",
    "encoding_transitions",
    "movement_compute_overlap_cycles",
    "overlap_available_cycles",
    "idle_cycles",
    "critical_path_cycles",
})


def require_probe_execution(descriptor: Mapping[str, Any]) -> None:
    """Exclude complete model captures from the search oracle, even if they would run quickly.

    The model is compiled and accounted globally. Only a separate mechanism-equivalent probe may
    be executed for calibration. Passing this exclusion check is not proof of probe equivalence;
    the experiment must additionally bind the probe's mechanism and validity domain.
    """
    operation = descriptor.get("operation")
    operation = operation if isinstance(operation, Mapping) else {}
    performance = descriptor.get("performance")
    performance = performance if isinstance(performance, Mapping) else {}
    semantic = descriptor.get("semantic")
    semantic = semantic if isinstance(semantic, Mapping) else {}
    if (descriptor.get("kind") == "model" or operation.get("op") == "model"
            or semantic.get("generalization_axis") == "model"
            or performance.get("global_objective") is True
            or performance.get("measurement_scope") in ("full_model", "full_layer")):
        raise ValueError(
            "the full model/layer is a compile-only search objective; calibrate with a separate "
            "mechanism-equivalent probe, even when full execution would fit the time budget")


@dataclass(frozen=True)
class SimulationBudget:
    timeout_seconds: float
    reference_timeout_seconds: float
    tier: str = "iteration"

    def __post_init__(self) -> None:
        if self.tier != "iteration":
            raise ValueError("SimulationBudget is only for the reduced-witness iteration tier")
        for name, value in (("timeout_seconds", self.timeout_seconds),
                            ("reference_timeout_seconds", self.reference_timeout_seconds)):
            if value <= 0 or value > ITERATION_MAX_SECONDS:
                raise ValueError(
                    f"{name} must be in (0, {ITERATION_MAX_SECONDS:g}] seconds for iteration; "
                    "reduce the witness shape; optional full-size validation is a separate tier")


@dataclass(frozen=True)
class SimulationAdmission:
    admitted: bool
    estimated_seconds: float | None
    reason: str = ""


def admit_reduced_witness(*, estimated_cycles: int,
                          measured_cycles_per_second: float | None,
                          startup_seconds: float = 0.0,
                          budget: SimulationBudget) -> SimulationAdmission:
    """Admit only a witness whose evidence-backed runtime fits the bounded inner loop."""
    if estimated_cycles < 0 or startup_seconds < 0:
        raise ValueError("estimated cycles and startup seconds must be non-negative")
    if measured_cycles_per_second is None or measured_cycles_per_second <= 0:
        return SimulationAdmission(
            False, None,
            "simulator throughput is UNKNOWN; measure it before admitting a witness")
    estimate = startup_seconds + estimated_cycles / measured_cycles_per_second
    if estimate > budget.timeout_seconds:
        return SimulationAdmission(
            False, estimate,
            f"estimated {estimate:.3f}s exceeds the {budget.timeout_seconds:g}s per-execution "
            "iteration budget; reduce the witness shape")
    return SimulationAdmission(True, estimate)


@dataclass(frozen=True)
class WarmProfileContract:
    warmup_runs: int = 1
    measured_runs: int = 1
    captured_metrics: frozenset[str] = frozenset({"total_compute_cycles"})

    def __post_init__(self) -> None:
        if self.warmup_runs < 1:
            raise ValueError("a performance profile requires at least one unmeasured warm run")
        if self.measured_runs != 1:
            raise ValueError(
                "the iteration receipt must contain exactly one measured run after warmup")
        if "total_compute_cycles" not in self.captured_metrics:
            raise ValueError("total_compute_cycles is the required primary metric")
        extra = sorted(self.captured_metrics - _PROFILE_METRICS)
        if extra:
            raise ValueError(
                f"non-minimal profile metrics {extra}; capture compute cycles and only the "
                "resource/movement counters needed to explain them")


@dataclass(frozen=True)
class WarmComputeReceipt:
    workload: str
    total_compute_cycles: int
    contract: WarmProfileContract
    provenance: str
    resource_busy_cycles: tuple[tuple[str, int], ...] = ()
    movement_bytes: int | None = None
    movement_commands: int | None = None
    encoding_transitions: int | None = None
    movement_compute_overlap_cycles: int | None = None
    overlap_available_cycles: int | None = None
    idle_cycles: int | None = None
    critical_path_cycles: int | None = None

    def __post_init__(self) -> None:
        if not self.workload.strip() or not self.provenance.strip():
            raise ValueError("a warm compute receipt must name its workload and provenance")
        if self.total_compute_cycles < 0:
            raise ValueError("total compute cycles cannot be negative")
        if tuple(sorted(self.resource_busy_cycles)) != self.resource_busy_cycles:
            raise ValueError("resource busy counters must be sorted for stable receipts")
        if any(value < 0 for _, value in self.resource_busy_cycles):
            raise ValueError("resource busy counters cannot be negative")
        optional = (
            self.movement_bytes, self.movement_commands, self.encoding_transitions,
            self.movement_compute_overlap_cycles, self.overlap_available_cycles,
            self.idle_cycles, self.critical_path_cycles,
        )
        if any(value is not None and value < 0 for value in optional):
            raise ValueError("movement, encoding, and occupancy counters cannot be negative")
        bounded = (self.movement_compute_overlap_cycles, self.idle_cycles,
                   self.critical_path_cycles)
        if any(value is not None and value > self.total_compute_cycles for value in bounded):
            raise ValueError("overlap, idle, and critical-path cycles cannot exceed the measured run")
        if (self.movement_compute_overlap_cycles is not None
                and self.overlap_available_cycles is not None
                and self.movement_compute_overlap_cycles > self.overlap_available_cycles):
            raise ValueError("realized overlap cannot exceed overlap available cycles")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "profile": {
                "warmup_runs": self.contract.warmup_runs,
                "measured_runs": self.contract.measured_runs,
                "captured_metrics": sorted(self.contract.captured_metrics),
            },
            "total_compute_cycles": self.total_compute_cycles,
            "resource_busy_cycles": dict(self.resource_busy_cycles),
            "movement_bytes": self.movement_bytes,
            "movement_commands": self.movement_commands,
            "encoding_transitions": self.encoding_transitions,
            "movement_compute_overlap_cycles": self.movement_compute_overlap_cycles,
            "overlap_available_cycles": self.overlap_available_cycles,
            "idle_cycles": self.idle_cycles,
            "critical_path_cycles": self.critical_path_cycles,
            "provenance": self.provenance,
        }


def occupancy_from_warm_receipt(
        receipt: WarmComputeReceipt,
        resource_kinds: Mapping[str, object]):
    """Build planner occupancy from the minimal warm receipt, preserving every unknown.

    Resource roles come from the target adapter, never from counter spellings.  Aggregate busy
    counts alone do not establish overlap, idle time, or a critical path, so absent optional fields
    remain ``None`` and are named in ``missing``; downstream whole-model gates then refuse instead of
    treating an unmeasured quantity as zero.
    """
    from merlin.perf.decompose import ResourceKind
    from merlin.perf.global_planner import OccupancySummary

    busy = dict(receipt.resource_busy_cycles)
    unknown = sorted(set(busy) - set(resource_kinds))
    if unknown:
        raise ValueError(f"resource kinds are absent for warm counters {unknown}")

    def kind(value: object) -> ResourceKind:
        try:
            return value if isinstance(value, ResourceKind) else ResourceKind(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid resource kind {value!r}") from exc

    # Keep all adapter-declared resources in scope. Dropping an unobserved
    # engine changes the utilization denominator and can make a partial trace
    # look fully occupied. A known inactive engine needs an explicit zero.
    resolved = {name: kind(value) for name, value in resource_kinds.items()}
    compute = tuple(sorted(name for name, value in resolved.items()
                           if value is ResourceKind.COMPUTE))
    movement = tuple(sorted(name for name, value in resolved.items()
                            if value is ResourceKind.MOVEMENT))
    missing: list[str] = []
    for name, value in (
            ("at least one compute resource", compute or None),
            ("movement/compute overlap cycles", receipt.movement_compute_overlap_cycles),
            ("overlap available cycles", receipt.overlap_available_cycles),
            ("idle cycles", receipt.idle_cycles),
            ("critical path cycles", receipt.critical_path_cycles),
            ("physical movement bytes", receipt.movement_bytes),
            ("movement command count", receipt.movement_commands),
            ("executed encoding transition count", receipt.encoding_transitions)):
        if value is None:
            missing.append(name)
    return OccupancySummary(
        total_cycles=float(receipt.total_compute_cycles),
        busy_cycles=tuple((name, float(value)) for name, value in receipt.resource_busy_cycles),
        compute_resources=compute,
        movement_resources=movement,
        overlap_cycles=(float(receipt.movement_compute_overlap_cycles)
                        if receipt.movement_compute_overlap_cycles is not None else None),
        overlap_available_cycles=(float(receipt.overlap_available_cycles)
                                  if receipt.overlap_available_cycles is not None else None),
        idle_cycles=float(receipt.idle_cycles) if receipt.idle_cycles is not None else None,
        critical_path_cycles=(float(receipt.critical_path_cycles)
                              if receipt.critical_path_cycles is not None else None),
        movement_bytes=float(receipt.movement_bytes) if receipt.movement_bytes is not None else None,
        movement_commands=receipt.movement_commands,
        encoding_transitions=receipt.encoding_transitions,
        provenance=(receipt.provenance,),
        missing=tuple(missing),
    )


@dataclass(frozen=True)
class QueuedFireSimReceipt:
    queue_request_id: str
    queue_owned: bool
    commands: tuple[tuple[str, ...], ...]
    warm_profile: WarmComputeReceipt

    def __post_init__(self) -> None:
        if not self.queue_owned or not self.queue_request_id.strip():
            raise ValueError("FireSim milestone execution must be owned by the FireSim queue")
        prefixes = tuple(tuple(command[:2]) for command in self.commands)
        if prefixes != FIRESIM_LIFECYCLE:
            rendered = tuple(" ".join(item) for item in prefixes)
            expected = tuple(" ".join(item) for item in FIRESIM_LIFECYCLE)
            raise ValueError(
                f"FireSim lifecycle must be exactly {expected}, observed {rendered}")
        if any(len(command) < 2 for command in self.commands):
            raise ValueError("every FireSim lifecycle command must name its subcommand")
