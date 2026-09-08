"""Fast, accuracy-bounded selection for a whole-model optimization portfolio.

This module does not estimate a target by itself.  A host-owned analytical adapter supplies a
conservative cycle interval, data movement, an explicit occupancy timeline summary, encoding
conversion costs, and a calibrated risk score for each arm.  The shared evaluator then applies the
same quality budget and Pareto policy to every model.  Missing evidence remains missing: it is never
read as zero and can never retain a candidate.

The evaluator is deliberately target-neutral and simulation-free.  Reduced warm measurements may
calibrate an adapter, but no device runtime is called here and unlike model cycle counts are never
summed into a synthetic portfolio score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .global_planner import OccupancySummary

_DIRECTIONS = {
    "cycles": "min",
    "movement_bytes": "min",
    "compute_utilization": "max",
    "latency_hiding_efficiency": "max",
    "encoding_conversion_count": "min",
    "encoding_conversion_bytes": "min",
    "encoding_conversion_cycles": "min",
    "supported_work_placed_fraction": "max",
    "largest_connected_region_fraction": "max",
    "host_island_count": "min",
    "boundary_crossings": "min",
    "boundary_bytes": "min",
}

_LEVER_EFFECTS = {
    "whole_program_cycles": frozenset(("compute", "fusion", "lowering", "tiling")),
    "data_movement": frozenset(("encoding", "fusion", "movement", "residency")),
    "compute_occupancy": frozenset(("issue", "pipeline", "scheduling", "tiling")),
    "latency_hiding": frozenset(("issue", "movement", "pipeline", "scheduling")),
    "encoding_conversion": frozenset(("dtype", "encoding", "fusion", "movement")),
    "accelerator_coverage": frozenset(("fusion", "lowering", "movement", "tiling")),
    "host_islands": frozenset(("fusion", "lowering", "movement")),
    "quality_budget": frozenset(("approximation", "dtype", "numerics", "quantization")),
}


def _finite_nonnegative(value: Any, name: str, *, integer: bool = False) -> float | int:
    expected = int if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, expected):
        raise TypeError(f"{name} must be {'an integer' if integer else 'numeric'}")
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return int(value) if integer else float(value)


def _interval(value: CycleInterval | Mapping[str, Any] | None, name: str) -> CycleInterval:
    if isinstance(value, CycleInterval):
        return value
    if value is None:
        return CycleInterval.unknown(f"{name} was not supplied by the analytical adapter")
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a CycleInterval or mapping")
    provenance = tuple(str(item) for item in value.get("provenance") or ())
    missing = tuple(str(item) for item in value.get("missing") or ())
    lo, hi = value.get("lo"), value.get("hi")
    if lo is None and hi is None:
        return CycleInterval(None, None, provenance=provenance, missing=missing or (f"{name} is unresolved",))
    return CycleInterval(lo, hi, provenance=provenance, missing=missing)


def _occupancy(value: OccupancySummary | Mapping[str, Any] | None) -> OccupancySummary | None:
    if value is None or isinstance(value, OccupancySummary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("occupancy must be an OccupancySummary or mapping")
    busy = value.get("busy_cycles") or {}
    if not isinstance(busy, Mapping):
        raise TypeError("occupancy busy_cycles must be a mapping")
    return OccupancySummary(
        total_cycles=float(_finite_nonnegative(value.get("total_cycles"), "occupancy total_cycles")),
        busy_cycles=tuple(
            sorted(
                (str(key), float(_finite_nonnegative(amount, f"busy cycles for {key}"))) for key, amount in busy.items()
            )
        ),
        compute_resources=tuple(str(item) for item in value.get("compute_resources") or ()),
        movement_resources=tuple(str(item) for item in value.get("movement_resources") or ()),
        movement_elapsed_cycles=(
            None
            if value.get("movement_elapsed_cycles") is None
            else float(_finite_nonnegative(value["movement_elapsed_cycles"], "movement elapsed cycles"))
        ),
        overlap_cycles=(
            None
            if value.get("overlap_cycles") is None
            else float(_finite_nonnegative(value["overlap_cycles"], "overlap cycles"))
        ),
        overlap_available_cycles=(
            None
            if value.get("overlap_available_cycles") is None
            else float(_finite_nonnegative(value["overlap_available_cycles"], "overlap available cycles"))
        ),
        idle_cycles=(
            None
            if value.get("idle_cycles") is None
            else float(_finite_nonnegative(value["idle_cycles"], "idle cycles"))
        ),
        critical_path_cycles=(
            None
            if value.get("critical_path_cycles") is None
            else float(_finite_nonnegative(value["critical_path_cycles"], "critical path cycles"))
        ),
        movement_bytes=(
            None
            if value.get("movement_bytes") is None
            else float(_finite_nonnegative(value["movement_bytes"], "movement bytes"))
        ),
        movement_commands=(
            None
            if value.get("movement_commands") is None
            else int(_finite_nonnegative(value["movement_commands"], "movement commands", integer=True))
        ),
        encoding_transitions=(
            None
            if value.get("encoding_transitions") is None
            else int(_finite_nonnegative(value["encoding_transitions"], "encoding transitions", integer=True))
        ),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
        missing=tuple(str(item) for item in value.get("missing") or ()),
    )


@dataclass(frozen=True)
class QualityLimit:
    """One independently evaluated accuracy/error constraint."""

    metric: str
    direction: str
    threshold: float
    maximum_degradation: float | None = None

    def __post_init__(self) -> None:
        if not self.metric.strip() or self.direction not in ("at_most", "at_least"):
            raise ValueError("quality limits require a metric and at_most/at_least direction")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, (int, float))
            or not math.isfinite(float(self.threshold))
        ):
            raise ValueError("quality threshold must be finite")
        if self.maximum_degradation is not None:
            _finite_nonnegative(self.maximum_degradation, "maximum quality degradation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "threshold": float(self.threshold),
            "maximum_degradation": self.maximum_degradation,
        }


@dataclass(frozen=True)
class QualityBudget:
    """Explicit per-model quality contract; no implicit universal accuracy proxy."""

    limits: tuple[QualityLimit, ...]
    reference: str

    def __post_init__(self) -> None:
        if not self.reference.strip() or not self.limits:
            raise ValueError("a quality budget requires a reference and at least one limit")
        names = [limit.metric for limit in self.limits]
        if len(names) != len(set(names)):
            raise ValueError("quality metrics must be unique within one model budget")

    def to_dict(self) -> dict[str, Any]:
        return {"reference": self.reference, "limits": [limit.to_dict() for limit in self.limits]}


@dataclass(frozen=True)
class QualityObservation:
    values: tuple[tuple[str, float], ...]
    provenance: tuple[str, ...]
    complete: bool = True

    def __post_init__(self) -> None:
        if tuple(sorted(self.values)) != self.values or len(dict(self.values)) != len(self.values):
            raise ValueError("quality values must be unique and sorted")
        if any(not name.strip() for name, _ in self.values):
            raise ValueError("quality metric names cannot be empty")
        if any(isinstance(value, bool) or not math.isfinite(float(value)) for _, value in self.values):
            raise ValueError("quality observations must be finite numeric values")
        if not self.provenance:
            raise ValueError("quality observations require independent provenance")

    @property
    def value_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.values}

    def to_dict(self) -> dict[str, Any]:
        return {"values": self.value_map, "provenance": list(self.provenance), "complete": self.complete}


@dataclass(frozen=True)
class CoverageSummary:
    """Source-work placement and topology, independent of any accelerator vocabulary.

    ``work_unit`` is adapter-declared (for example exact arithmetic work or weighted source work).
    Raw operation count is not assumed to approximate benefit. Host islands are reported as
    ``(taxonomy, count, work)`` records so an agent can distinguish unsupported math, host control,
    and intentional boundary work without model-name rules.
    """

    supported_work_total: float
    supported_work_placed: float
    largest_connected_region_work: float
    connected_region_work: tuple[float, ...]
    host_islands: tuple[tuple[str, int, float], ...]
    boundary_crossings: int
    boundary_bytes: float
    work_unit: str
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "supported_work_total",
            "supported_work_placed",
            "largest_connected_region_work",
            "boundary_bytes",
        ):
            _finite_nonnegative(getattr(self, name), name)
        _finite_nonnegative(self.boundary_crossings, "boundary_crossings", integer=True)
        if self.supported_work_total <= 0:
            raise ValueError("supported source work total must be positive")
        if self.supported_work_placed > self.supported_work_total:
            raise ValueError("placed supported work cannot exceed supported source work")
        if self.largest_connected_region_work > self.supported_work_placed:
            raise ValueError("largest connected region cannot exceed placed work")
        if tuple(sorted(self.connected_region_work, reverse=True)) != self.connected_region_work:
            raise ValueError("connected region work must be descending")
        for work in self.connected_region_work:
            _finite_nonnegative(work, "connected region work")
        if self.supported_work_placed == 0 and (self.connected_region_work or self.largest_connected_region_work != 0):
            raise ValueError("zero placed work cannot have a connected accelerator region")
        if self.supported_work_placed > 0 and not self.connected_region_work:
            raise ValueError("placed work requires at least one connected region")
        if self.connected_region_work and self.connected_region_work[0] != self.largest_connected_region_work:
            raise ValueError("largest connected region must match the first region")
        if not math.isclose(
            sum(self.connected_region_work),
            self.supported_work_placed,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("connected regions must exactly partition placed source work")
        if not self.work_unit.strip() or not self.provenance:
            raise ValueError("coverage requires a work unit and provenance")
        if tuple(sorted(self.host_islands)) != self.host_islands:
            raise ValueError("host island taxonomy must be sorted")
        names = [name for name, _, _ in self.host_islands]
        if len(names) != len(set(names)):
            raise ValueError("host island taxonomy names must be unique")
        for name, count, work in self.host_islands:
            if not name.strip():
                raise ValueError("host island taxonomy names cannot be empty")
            _finite_nonnegative(count, f"host island count for {name}", integer=True)
            _finite_nonnegative(work, f"host island work for {name}")

    @property
    def supported_work_placed_fraction(self) -> float:
        return self.supported_work_placed / self.supported_work_total

    @property
    def largest_connected_region_fraction(self) -> float:
        return self.largest_connected_region_work / self.supported_work_total

    @property
    def host_island_count(self) -> int:
        return sum(count for _, count, _ in self.host_islands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported_work_total": self.supported_work_total,
            "supported_work_placed": self.supported_work_placed,
            "supported_work_placed_fraction": self.supported_work_placed_fraction,
            "largest_connected_region_work": self.largest_connected_region_work,
            "largest_connected_region_fraction": self.largest_connected_region_fraction,
            "connected_region_work": list(self.connected_region_work),
            "host_islands": [
                {"taxonomy": name, "count": count, "work": work} for name, count, work in self.host_islands
            ],
            "host_island_count": self.host_island_count,
            "boundary_crossings": self.boundary_crossings,
            "boundary_bytes": self.boundary_bytes,
            "work_unit": self.work_unit,
            "provenance": list(self.provenance),
        }


def _coverage(value: CoverageSummary | Mapping[str, Any] | None) -> CoverageSummary | None:
    if value is None or isinstance(value, CoverageSummary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("coverage must be a CoverageSummary or mapping")
    islands = value.get("host_islands") or ()
    if not isinstance(islands, Sequence) or isinstance(islands, (str, bytes)):
        raise TypeError("host_islands must be a sequence")
    parsed = []
    for island in islands:
        if not isinstance(island, Mapping):
            raise TypeError("host island taxonomy entries must be mappings")
        parsed.append(
            (
                str(island.get("taxonomy") or ""),
                int(_finite_nonnegative(island.get("count"), "host island count", integer=True)),
                float(_finite_nonnegative(island.get("work"), "host island work")),
            )
        )
    return CoverageSummary(
        supported_work_total=float(_finite_nonnegative(value.get("supported_work_total"), "supported_work_total")),
        supported_work_placed=float(_finite_nonnegative(value.get("supported_work_placed"), "supported_work_placed")),
        largest_connected_region_work=float(
            _finite_nonnegative(value.get("largest_connected_region_work"), "largest_connected_region_work")
        ),
        connected_region_work=tuple(
            float(_finite_nonnegative(item, "connected region work"))
            for item in value.get("connected_region_work") or ()
        ),
        host_islands=tuple(sorted(parsed)),
        boundary_crossings=int(
            _finite_nonnegative(value.get("boundary_crossings"), "boundary_crossings", integer=True)
        ),
        boundary_bytes=float(_finite_nonnegative(value.get("boundary_bytes"), "boundary_bytes")),
        work_unit=str(value.get("work_unit") or ""),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
    )


@dataclass(frozen=True)
class RooflineSummary:
    """Explicit adapter-composed lower bound and its limiting resource classes."""

    lower_bound_cycles: float
    resource_floors: tuple[tuple[str, float], ...]
    limiting_resources: tuple[str, ...]
    optimization_effects: tuple[str, ...]
    composition: str
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        _finite_nonnegative(self.lower_bound_cycles, "roofline lower bound")
        if tuple(sorted(self.resource_floors)) != self.resource_floors or len(dict(self.resource_floors)) != len(
            self.resource_floors
        ):
            raise ValueError("roofline resource floors must be unique and sorted")
        if not self.resource_floors or not self.limiting_resources:
            raise ValueError("roofline must name resource floors and at least one limiter")
        for resource, cycles in self.resource_floors:
            if not resource.strip():
                raise ValueError("roofline resource names cannot be empty")
            _finite_nonnegative(cycles, f"roofline floor for {resource}")
        if self.lower_bound_cycles < max(dict(self.resource_floors).values()):
            raise ValueError("composed roofline bound cannot be below a declared resource floor")
        if (
            tuple(sorted(self.limiting_resources)) != self.limiting_resources
            or len(set(self.limiting_resources)) != len(self.limiting_resources)
            or any(name not in dict(self.resource_floors) for name in self.limiting_resources)
        ):
            raise ValueError("roofline limiters must uniquely and stably name declared resources")
        if (
            not self.optimization_effects
            or tuple(sorted(self.optimization_effects)) != self.optimization_effects
            or len(set(self.optimization_effects)) != len(self.optimization_effects)
            or any(not effect.strip() for effect in self.optimization_effects)
        ):
            raise ValueError("roofline optimization effects must be unique, non-empty, and sorted")
        if not self.composition.strip() or not self.provenance:
            raise ValueError("roofline requires explicit composition and provenance")

    def to_dict(self, total_cycles: float | None = None) -> dict[str, Any]:
        headroom = (
            None if total_cycles is None or self.lower_bound_cycles == 0 else total_cycles / self.lower_bound_cycles
        )
        attainment = None if total_cycles is None or total_cycles == 0 else self.lower_bound_cycles / total_cycles
        return {
            "lower_bound_cycles": self.lower_bound_cycles,
            "resource_floors": dict(self.resource_floors),
            "limiting_resources": list(self.limiting_resources),
            "optimization_effects": list(self.optimization_effects),
            "composition": self.composition,
            "provenance": list(self.provenance),
            "headroom_to_lower_bound": headroom,
            "attainment_fraction": attainment,
        }


def _roofline(value: RooflineSummary | Mapping[str, Any] | None) -> RooflineSummary | None:
    if value is None or isinstance(value, RooflineSummary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("roofline must be a RooflineSummary or mapping")
    floors = value.get("resource_floors") or {}
    if not isinstance(floors, Mapping):
        raise TypeError("roofline resource_floors must be a mapping")
    return RooflineSummary(
        lower_bound_cycles=float(_finite_nonnegative(value.get("lower_bound_cycles"), "roofline lower bound")),
        resource_floors=tuple(
            sorted(
                (str(name), float(_finite_nonnegative(cycles, f"roofline floor for {name}")))
                for name, cycles in floors.items()
            )
        ),
        limiting_resources=tuple(str(item) for item in value.get("limiting_resources") or ()),
        optimization_effects=tuple(str(item) for item in value.get("optimization_effects") or ()),
        composition=str(value.get("composition") or ""),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
    )


@dataclass(frozen=True)
class AnalyticalMetrics:
    """One arm's host-produced whole-model analytical evidence."""

    cycles: CycleInterval
    movement_bytes: float | None
    movement_scope: str
    occupancy: OccupancySummary | None
    coverage: CoverageSummary | None
    roofline: RooflineSummary | None
    encoding_conversion_count: int | None
    encoding_conversion_bytes: float | None
    encoding_conversion_cycles: CycleInterval
    risk_score: float | None
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, interval in (
            ("cycles", self.cycles),
            ("encoding conversion cycles", self.encoding_conversion_cycles),
        ):
            if interval.resolved and not all(math.isfinite(float(endpoint)) for endpoint in (interval.lo, interval.hi)):
                raise ValueError(f"{name} must be finite")
        for name in ("movement_bytes", "encoding_conversion_bytes"):
            value = getattr(self, name)
            if value is not None:
                _finite_nonnegative(value, name)
        if self.movement_scope not in ("physical", "unavailable"):
            raise ValueError("movement scope must be physical or unavailable")
        if self.movement_bytes is not None and self.movement_scope != "physical":
            raise ValueError("known movement bytes must describe physical movement")
        if self.encoding_conversion_count is not None:
            _finite_nonnegative(self.encoding_conversion_count, "encoding_conversion_count", integer=True)
        if self.risk_score is not None:
            _finite_nonnegative(self.risk_score, "risk_score")
            if self.risk_score > 1:
                raise ValueError("risk_score must be in [0, 1]")
        if not self.provenance:
            raise ValueError("analytical metrics require provenance")
        if self.roofline is not None and self.cycles.resolved and self.roofline.lower_bound_cycles > self.cycles.lo:
            raise ValueError("roofline lower bound exceeds the analytical cycle interval")
        if self.occupancy is not None and self.cycles.resolved:
            if not math.isfinite(self.occupancy.total_cycles) or any(
                not math.isfinite(value) for _, value in self.occupancy.busy_cycles
            ):
                raise ValueError("occupancy counters must be finite")
            for name in (
                "movement_elapsed_cycles",
                "overlap_cycles",
                "overlap_available_cycles",
                "idle_cycles",
                "critical_path_cycles",
                "movement_bytes",
            ):
                value = getattr(self.occupancy, name)
                if value is not None and not math.isfinite(value):
                    raise ValueError(f"occupancy {name} must be finite")
            if not math.isclose(self.occupancy.total_cycles, float(self.cycles.hi), rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("occupancy total must equal the conservative cycle endpoint")
        if (
            self.occupancy is not None
            and self.occupancy.movement_bytes is not None
            and self.movement_bytes is not None
            and not math.isclose(self.occupancy.movement_bytes, self.movement_bytes, rel_tol=0.0, abs_tol=1e-9)
        ):
            raise ValueError("movement bytes disagree with the occupancy timeline")
        if (
            self.occupancy is not None
            and self.occupancy.encoding_transitions is not None
            and self.encoding_conversion_count is not None
            and self.occupancy.encoding_transitions != self.encoding_conversion_count
        ):
            raise ValueError("encoding conversion count disagrees with the occupancy timeline")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AnalyticalMetrics:
        encoding = value.get("encoding_conversions") or {}
        if not isinstance(encoding, Mapping):
            raise TypeError("encoding_conversions must be a mapping")
        count = encoding.get("count")
        byte_count = encoding.get("bytes")
        movement = value.get("movement_bytes")
        risk = value.get("risk_score")
        return cls(
            cycles=_interval(value.get("cycles"), "whole-model cycles"),
            movement_bytes=(None if movement is None else float(_finite_nonnegative(movement, "movement_bytes"))),
            movement_scope=str(value.get("movement_scope") or "unavailable"),
            occupancy=_occupancy(value.get("occupancy")),
            coverage=_coverage(value.get("coverage")),
            roofline=_roofline(value.get("roofline")),
            encoding_conversion_count=(
                None if count is None else int(_finite_nonnegative(count, "encoding conversion count", integer=True))
            ),
            encoding_conversion_bytes=(
                None if byte_count is None else float(_finite_nonnegative(byte_count, "encoding conversion bytes"))
            ),
            encoding_conversion_cycles=_interval(encoding.get("cycles"), "encoding conversion cycles"),
            risk_score=(None if risk is None else float(_finite_nonnegative(risk, "risk_score"))),
            provenance=tuple(str(item) for item in value.get("provenance") or ()),
        )

    def objectives(self) -> dict[str, float | CycleInterval | None]:
        return {
            "cycles": self.cycles,
            "movement_bytes": self.movement_bytes,
            "compute_utilization": (None if self.occupancy is None else self.occupancy.compute_utilization),
            "latency_hiding_efficiency": (None if self.occupancy is None else self.occupancy.latency_hiding_efficiency),
            "encoding_conversion_count": self.encoding_conversion_count,
            "encoding_conversion_bytes": self.encoding_conversion_bytes,
            "encoding_conversion_cycles": self.encoding_conversion_cycles,
            "supported_work_placed_fraction": (
                None if self.coverage is None else self.coverage.supported_work_placed_fraction
            ),
            "largest_connected_region_fraction": (
                None if self.coverage is None else self.coverage.largest_connected_region_fraction
            ),
            "host_island_count": (None if self.coverage is None else self.coverage.host_island_count),
            "boundary_crossings": (None if self.coverage is None else self.coverage.boundary_crossings),
            "boundary_bytes": (None if self.coverage is None else self.coverage.boundary_bytes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycles": self.cycles.to_dict(),
            "movement_bytes": self.movement_bytes,
            "movement_scope": self.movement_scope,
            "occupancy": None if self.occupancy is None else self.occupancy.to_dict(),
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "roofline": (
                None
                if self.roofline is None
                else self.roofline.to_dict(None if not self.cycles.resolved else float(self.cycles.hi))
            ),
            "encoding_conversions": {
                "count": self.encoding_conversion_count,
                "bytes": self.encoding_conversion_bytes,
                "cycles": self.encoding_conversion_cycles.to_dict(),
            },
            "risk_score": self.risk_score,
            "provenance": list(self.provenance),
        }


@dataclass(frozen=True)
class FastEvaluationPolicy:
    """Conservative portfolio retention policy.

    All known objectives participate in Pareto regression checks. ``required_objectives`` controls
    which unknowns block retention.  Risk and interval-width caps make model uncertainty explicit.
    """

    required_objectives: tuple[str, ...] = tuple(_DIRECTIONS)
    maximum_regression_fraction: tuple[tuple[str, float], ...] = tuple((name, 0.0) for name in _DIRECTIONS)
    minimum_improvement_fraction: float = 0.0
    maximum_risk_score: float = 0.25
    maximum_cycle_interval_width_fraction: float = 0.25
    require_improvement: bool = True
    require_roofline: bool = True
    global_benefit_objectives: tuple[str, ...] = (
        "cycles",
        "movement_bytes",
        "compute_utilization",
        "latency_hiding_efficiency",
        "encoding_conversion_count",
        "encoding_conversion_bytes",
        "encoding_conversion_cycles",
        "boundary_crossings",
        "boundary_bytes",
    )

    def __post_init__(self) -> None:
        unknown = sorted(set(self.required_objectives) - set(_DIRECTIONS))
        if unknown or len(self.required_objectives) != len(set(self.required_objectives)):
            raise ValueError(f"invalid or duplicate required objectives: {unknown}")
        unknown_benefit = sorted(set(self.global_benefit_objectives) - set(_DIRECTIONS))
        if unknown_benefit or len(self.global_benefit_objectives) != len(set(self.global_benefit_objectives)):
            raise ValueError(f"invalid or duplicate global benefit objectives: {unknown_benefit}")
        if not isinstance(self.require_improvement, bool) or not isinstance(self.require_roofline, bool):
            raise TypeError("fast-evaluation boolean policies must be booleans")
        regressions = dict(self.maximum_regression_fraction)
        if len(regressions) != len(self.maximum_regression_fraction):
            raise ValueError("maximum regression objectives must be unique")
        if set(regressions) - set(_DIRECTIONS):
            raise ValueError("maximum regression policy names an unknown objective")
        for value in (
            *regressions.values(),
            self.minimum_improvement_fraction,
            self.maximum_risk_score,
            self.maximum_cycle_interval_width_fraction,
        ):
            _finite_nonnegative(value, "fast-evaluation policy fraction")
        if self.maximum_risk_score > 1:
            raise ValueError("maximum_risk_score must be in [0, 1]")

    @property
    def regression_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.maximum_regression_fraction}

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_objectives": list(self.required_objectives),
            "maximum_regression_fraction": self.regression_map,
            "minimum_improvement_fraction": self.minimum_improvement_fraction,
            "maximum_risk_score": self.maximum_risk_score,
            "maximum_cycle_interval_width_fraction": self.maximum_cycle_interval_width_fraction,
            "require_improvement": self.require_improvement,
            "require_roofline": self.require_roofline,
            "global_benefit_objectives": list(self.global_benefit_objectives),
        }


def _quality(value: QualityObservation | Mapping[str, Any] | None) -> QualityObservation | None:
    if value is None or isinstance(value, QualityObservation):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("quality observation must be a mapping")
    values = value.get("values") or {}
    if not isinstance(values, Mapping):
        raise TypeError("quality observation values must be a mapping")
    parsed = []
    for name, number in values.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(float(number)):
            raise TypeError(f"quality observation {name} must be finite and numeric")
        parsed.append((str(name), float(number)))
    return QualityObservation(
        values=tuple(sorted(parsed)),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
        complete=value.get("complete") is True,
    )


def _quality_gate(
    budget: QualityBudget, candidate: QualityObservation | None, baseline: QualityObservation | None
) -> dict[str, Any]:
    checks = []
    blockers: list[str] = []
    failures: list[str] = []
    candidate_values = {} if candidate is None else candidate.value_map
    baseline_values = {} if baseline is None else baseline.value_map
    if candidate is None or not candidate.complete:
        blockers.append("complete candidate quality observation is unavailable")
    for limit in budget.limits:
        value = candidate_values.get(limit.metric)
        base = baseline_values.get(limit.metric)
        threshold_passes = None
        degradation_passes = None
        if value is None:
            blockers.append(f"candidate quality metric {limit.metric} is unavailable")
        else:
            threshold_passes = value <= limit.threshold if limit.direction == "at_most" else value >= limit.threshold
            if not threshold_passes:
                failures.append(f"candidate quality metric {limit.metric} exceeds its budget")
        if limit.maximum_degradation is not None:
            if baseline is None or not baseline.complete:
                blockers.append("complete baseline quality observation is unavailable")
            if base is None:
                blockers.append(f"baseline quality metric {limit.metric} is unavailable")
            elif value is not None:
                degradation = (value - base) if limit.direction == "at_most" else (base - value)
                degradation_passes = degradation <= limit.maximum_degradation
                if not degradation_passes:
                    failures.append(f"candidate quality metric {limit.metric} regressed too far")
        checks.append(
            {
                **limit.to_dict(),
                "candidate": value,
                "baseline": base,
                "threshold_passes": threshold_passes,
                "degradation_passes": degradation_passes,
            }
        )
    status = "failed" if failures else "needs_evidence" if blockers else "passed"
    return {
        "status": status,
        "reference": budget.reference,
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "failures": sorted(set(failures)),
        "candidate_observation": None if candidate is None else candidate.to_dict(),
        "baseline_observation": None if baseline is None else baseline.to_dict(),
    }


def _uncertainty(interval: CycleInterval) -> float | None:
    if not interval.resolved:
        return None
    if interval.hi == 0:
        return 0.0
    return float(interval.hi - interval.lo) / float(interval.hi)


def _compare_objective(
    name: str,
    baseline: float | CycleInterval | None,
    candidate: float | CycleInterval | None,
    allowed_regression: float,
    minimum_improvement: float,
) -> dict[str, Any]:
    direction = _DIRECTIONS[name]
    if isinstance(baseline, CycleInterval) or isinstance(candidate, CycleInterval):
        if not isinstance(baseline, CycleInterval) or not isinstance(candidate, CycleInterval):
            return {
                "objective": name,
                "direction": direction,
                "status": "UNKNOWN",
                "reason": "the two arms use incompatible evidence forms",
            }
        if not baseline.resolved or not candidate.resolved:
            missing = (*baseline.missing, *candidate.missing)
            return {
                "objective": name,
                "direction": direction,
                "status": "UNKNOWN",
                "reason": "; ".join(dict.fromkeys(missing)),
            }
        # Every interval-valued objective is minimized.  Prove a non-regression with the candidate
        # upper endpoint against the baseline lower endpoint; overlapping intervals remain unknown.
        acceptable = candidate.hi <= baseline.lo * (1.0 + allowed_regression)
        regression = candidate.lo > baseline.hi * (1.0 + allowed_regression)
        improvement = candidate.hi < baseline.lo * (1.0 - minimum_improvement)
        return {
            "objective": name,
            "direction": direction,
            "status": "non_regression" if acceptable else "regression" if regression else "UNCERTAIN",
            "baseline": baseline.to_dict(),
            "candidate": candidate.to_dict(),
            "allowed_regression_fraction": allowed_regression,
            "robust_improvement": improvement,
            "conservative_speedup": (None if candidate.hi == 0 else baseline.lo / candidate.hi),
        }
    if baseline is None or candidate is None:
        return {
            "objective": name,
            "direction": direction,
            "status": "UNKNOWN",
            "reason": f"{name} was not supplied for both arms",
        }
    baseline_value, candidate_value = float(baseline), float(candidate)
    if direction == "min":
        acceptable = candidate_value <= baseline_value * (1.0 + allowed_regression)
        improvement = candidate_value < baseline_value * (1.0 - minimum_improvement)
        change = None if baseline_value == 0 else (candidate_value - baseline_value) / baseline_value
    else:
        # A fractional allowance is relative to the bounded [0,1] metric.  A zero baseline remains
        # comparable without division; candidate must not fall below zero.
        acceptable = candidate_value >= baseline_value * (1.0 - allowed_regression)
        improvement = candidate_value > baseline_value * (1.0 + minimum_improvement)
        change = None if baseline_value == 0 else (candidate_value - baseline_value) / baseline_value
    return {
        "objective": name,
        "direction": direction,
        "status": "non_regression" if acceptable else "regression",
        "baseline": baseline_value,
        "candidate": candidate_value,
        "candidate_minus_baseline_fraction": change,
        "allowed_regression_fraction": allowed_regression,
        "robust_improvement": improvement,
    }


def _surface_levers(
    comparisons: Sequence[Mapping[str, Any]],
    quality_status: str,
    surfaces: Sequence[Mapping[str, Any]],
    roofline: RooflineSummary | None,
    total_cycles: float | None,
) -> list[dict[str, Any]]:
    objective_lever = {
        "cycles": "whole_program_cycles",
        "movement_bytes": "data_movement",
        "compute_utilization": "compute_occupancy",
        "latency_hiding_efficiency": "latency_hiding",
        "encoding_conversion_count": "encoding_conversion",
        "encoding_conversion_bytes": "encoding_conversion",
        "encoding_conversion_cycles": "encoding_conversion",
        "supported_work_placed_fraction": "accelerator_coverage",
        "largest_connected_region_fraction": "accelerator_coverage",
        "host_island_count": "host_islands",
        "boundary_crossings": "data_movement",
        "boundary_bytes": "data_movement",
    }
    needed = []
    for comparison in comparisons:
        if comparison.get("status") not in ("non_regression",):
            lever = objective_lever[str(comparison["objective"])]
            if lever not in needed:
                needed.append(lever)
    if quality_status != "passed":
        needed.insert(0, "quality_budget")
    result = []
    for lever in needed:
        effects = _LEVER_EFFECTS[lever]
        matched = []
        for surface in surfaces:
            declared = frozenset(str(item) for item in surface.get("effects") or ())
            if not declared.intersection(effects):
                continue
            matched.append({key: surface.get(key) for key in ("id", "path", "symbol", "scope", "effects")})
        result.append(
            {
                "lever": lever,
                "addresses_effects": sorted(effects),
                "authorized_surfaces": matched,
                "authority": ("host_frozen_surfaces_only" if matched else "no_authorized_surface_matches"),
            }
        )
    if (
        roofline is not None
        and total_cycles is not None
        and roofline.lower_bound_cycles > 0
        and total_cycles > roofline.lower_bound_cycles
    ):
        effects = frozenset(roofline.optimization_effects)
        matched = []
        for surface in surfaces:
            declared = frozenset(str(item) for item in surface.get("effects") or ())
            if declared.intersection(effects):
                matched.append({key: surface.get(key) for key in ("id", "path", "symbol", "scope", "effects")})
        result.append(
            {
                "lever": "roofline_headroom",
                "limiting_resources": list(roofline.limiting_resources),
                "headroom_to_lower_bound": total_cycles / roofline.lower_bound_cycles,
                "addresses_effects": list(roofline.optimization_effects),
                "authorized_surfaces": matched,
                "authority": ("host_frozen_surfaces_only" if matched else "no_authorized_surface_matches"),
            }
        )
    return result


def evaluate_fast_portfolio(
    rows: Sequence[Mapping[str, Any]],
    *,
    quality_budgets: Mapping[str, QualityBudget],
    policy: FastEvaluationPolicy,
    authorized_surfaces: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    expected_models: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Apply fail-closed quality, analytical Pareto, and calibration-risk gates.

    Each row contains ``model_id``, ``baseline`` and ``candidate`` analytical metrics plus
    ``baseline_quality`` and ``candidate_quality``.  The model ID should be a content identity, not
    a mutable display name.  The result contains ratios only; cycles are never added across models.
    """

    expected = (
        tuple(expected_models) if expected_models is not None else tuple(str(row.get("model_id")) for row in rows)
    )
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected model identities must be non-empty and unique")
    by_model = {str(row.get("model_id")): row for row in rows}
    if len(by_model) != len(rows) or set(by_model) != set(expected):
        raise ValueError("fast-evaluation rows must exactly cover the expected portfolio")
    if set(quality_budgets) != set(expected):
        raise ValueError("quality budgets must exactly cover the expected portfolio")
    regressions = policy.regression_map
    model_results = []
    all_failures: list[str] = []
    all_blockers: list[str] = []
    any_global_improvement = False
    speedups = []
    for model_id in expected:
        raw = by_model[model_id]
        provider_failure = raw.get("provider_failure")
        if isinstance(provider_failure, Mapping):
            reason = (
                f"host analytical provider failed: {provider_failure.get('type')}: {provider_failure.get('reason')}"
            )
            model_results.append(
                {"model_id": model_id, "status": "needs_evidence", "reason": reason, "recommended_levers": []}
            )
            all_blockers.append(f"{model_id}: {reason}")
            continue
        try:
            baseline = (
                raw["baseline"]
                if isinstance(raw["baseline"], AnalyticalMetrics)
                else AnalyticalMetrics.from_mapping(raw["baseline"])
            )
            candidate = (
                raw["candidate"]
                if isinstance(raw["candidate"], AnalyticalMetrics)
                else AnalyticalMetrics.from_mapping(raw["candidate"])
            )
            baseline_quality = _quality(raw.get("baseline_quality"))
            candidate_quality = _quality(raw.get("candidate_quality"))
            if baseline.coverage is not None and candidate.coverage is not None:
                if baseline.coverage.work_unit != candidate.coverage.work_unit or not math.isclose(
                    baseline.coverage.supported_work_total,
                    candidate.coverage.supported_work_total,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                ):
                    raise ValueError("coverage arms disagree on source-work denominator")
        except (KeyError, TypeError, ValueError) as exc:
            model_results.append(
                {
                    "model_id": model_id,
                    "status": "needs_evidence",
                    "reason": f"invalid analytical adapter result: {exc}",
                    "recommended_levers": [],
                }
            )
            all_blockers.append(f"{model_id}: invalid analytical adapter result")
            continue
        quality = _quality_gate(quality_budgets[model_id], candidate_quality, baseline_quality)
        comparisons = []
        baseline_objectives, candidate_objectives = baseline.objectives(), candidate.objectives()
        for objective in _DIRECTIONS:
            comparisons.append(
                _compare_objective(
                    objective,
                    baseline_objectives[objective],
                    candidate_objectives[objective],
                    regressions.get(objective, 0.0),
                    policy.minimum_improvement_fraction,
                )
            )
        failures = [
            f"{model_id}: {item['objective']} regressed" for item in comparisons if item["status"] == "regression"
        ]
        blockers = [
            f"{model_id}: {item['objective']} is {item['status']}"
            for item in comparisons
            if item["objective"] in policy.required_objectives and item["status"] in ("UNKNOWN", "UNCERTAIN")
        ]
        for arm_name, metrics in (("baseline", baseline), ("candidate", candidate)):
            if metrics.risk_score is None:
                blockers.append(f"{model_id}: {arm_name} calibration risk is UNKNOWN")
            elif metrics.risk_score > policy.maximum_risk_score:
                failures.append(f"{model_id}: {arm_name} calibration risk exceeds policy")
            width = _uncertainty(metrics.cycles)
            if width is None:
                blockers.append(f"{model_id}: {arm_name} cycle uncertainty is UNKNOWN")
            elif width > policy.maximum_cycle_interval_width_fraction:
                failures.append(f"{model_id}: {arm_name} cycle interval is too wide")
            if policy.require_roofline and metrics.roofline is None:
                blockers.append(f"{model_id}: {arm_name} physical roofline is UNKNOWN")
        failures.extend(f"{model_id}: {reason}" for reason in quality["failures"])
        blockers.extend(f"{model_id}: {reason}" for reason in quality["blockers"])
        improvements = [item["objective"] for item in comparisons if item.get("robust_improvement") is True]
        any_global_improvement = any_global_improvement or any(
            objective in policy.global_benefit_objectives for objective in improvements
        )
        cycle = next(item for item in comparisons if item["objective"] == "cycles")
        if isinstance(cycle.get("conservative_speedup"), (int, float)):
            speedups.append(float(cycle["conservative_speedup"]))
        status = "reject" if failures else "needs_evidence" if blockers else "pareto_admissible"
        model_results.append(
            {
                "model_id": model_id,
                "status": status,
                "quality_gate": quality,
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
                "objectives": comparisons,
                "robust_improvements": improvements,
                "failures": sorted(set(failures)),
                "blockers": sorted(set(blockers)),
                "recommended_levers": _surface_levers(
                    comparisons,
                    quality["status"],
                    (authorized_surfaces or {}).get(model_id, ()),
                    candidate.roofline,
                    None if not candidate.cycles.resolved else float(candidate.cycles.hi),
                ),
            }
        )
        all_failures.extend(failures)
        all_blockers.extend(blockers)
    if all_failures:
        status = "reject"
    elif all_blockers:
        status = "needs_evidence"
    elif policy.require_improvement and not any_global_improvement:
        status = "reject"
        all_failures.append("portfolio: no robust global benefit; increased placement alone is insufficient")
    else:
        status = "retain"
    geomean = (
        None
        if len(speedups) != len(expected) or not speedups or any(value <= 0 for value in speedups)
        else math.exp(sum(math.log(value) for value in speedups) / len(speedups))
    )
    return {
        "schema": "phase2_fast_portfolio_evaluation_v1",
        "status": status,
        "selection": "accuracy_bounded_per_model_analytical_pareto_and_risk_gate",
        "models": model_results,
        "models_evaluated": len(model_results),
        "policy": policy.to_dict(),
        "quality_budgets": {model: quality_budgets[model].to_dict() for model in expected},
        "portfolio_conservative_cycle_speedup_geomean": geomean,
        "failures": sorted(set(all_failures)),
        "blockers": sorted(set(all_blockers)),
        "aggregation": "dimensionless per-model ratios only; model cycles are never summed",
        "execution": "host_analytical_only_no_complete_model_or_layer_simulation",
        "authority": (
            "candidate metrics and quality must come from the host-owned adapter; the candidate cannot self-certify"
        ),
    }


def unavailable_fast_evaluation(*, reason: str) -> dict[str, Any]:
    """Stable fail-closed record for experiments that have not installed an adapter."""
    if not reason.strip():
        raise ValueError("unavailable fast evaluation requires a reason")
    return {
        "schema": "phase2_fast_portfolio_evaluation_v1",
        "status": "not_configured",
        "reason": reason,
        "selection": "accuracy_bounded_per_model_analytical_pareto_and_risk_gate",
        "execution": "host_analytical_only_no_complete_model_or_layer_simulation",
        "unknown_metrics": [*list(_DIRECTIONS), "quality_budget", "calibration_risk"],
        "aggregation": "model cycles are never summed",
    }
