"""Fast, accuracy-bounded selection for a complete-model optimization portfolio.

The shared evaluator consumes host-owned analytical evidence; it never runs a target or a
complete-model simulator.  Every model keeps its own cycle unit and quality contract.  The only
portfolio aggregate is a dimensionless geomean of conservative per-model speedups.

Missing evidence is an inhabited state, never zero.  When no held-out quality corpus/evaluator is
installed, :func:`unavailable_fast_evaluation` records an exact-only fallback: structural search may
continue, but no approximate transformation gains promotion authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

from .global_planner import OccupancySummary

PORTFOLIO_MEMBER_COUNT = 4

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

# Coverage/topology remains a diagnostic and a Pareto constraint.  It is deliberately absent from
# this set: moving more operations onto an accelerator is not a global benefit if cycles, movement,
# conversions, occupancy, and boundaries do not improve.
_GLOBAL_BENEFIT_OBJECTIVES = (
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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        return CycleInterval(
            None,
            None,
            provenance=provenance,
            missing=missing or (f"{name} is unresolved",),
        )
    return CycleInterval(lo, hi, provenance=provenance, missing=missing)


def _occupancy(value: OccupancySummary | Mapping[str, Any] | None) -> OccupancySummary | None:
    if value is None or isinstance(value, OccupancySummary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("occupancy must be an OccupancySummary or mapping")
    busy = value.get("busy_cycles") or {}
    if not isinstance(busy, Mapping):
        raise TypeError("occupancy busy_cycles must be a mapping")
    optional_float = (
        "movement_elapsed_cycles",
        "overlap_cycles",
        "overlap_available_cycles",
        "idle_cycles",
        "critical_path_cycles",
        "movement_bytes",
    )
    optional_integer = ("movement_commands", "encoding_transitions")
    parsed: dict[str, Any] = {
        name: (
            None
            if value.get(name) is None
            else float(_finite_nonnegative(value[name], name.replace("_", " ")))
        )
        for name in optional_float
    }
    parsed.update(
        {
            name: (
                None
                if value.get(name) is None
                else int(_finite_nonnegative(value[name], name.replace("_", " "), integer=True))
            )
            for name in optional_integer
        }
    )
    return OccupancySummary(
        total_cycles=float(_finite_nonnegative(value.get("total_cycles"), "occupancy total cycles")),
        busy_cycles=tuple(
            sorted(
                (str(key), float(_finite_nonnegative(amount, f"busy cycles for {key}")))
                for key, amount in busy.items()
            )
        ),
        compute_resources=tuple(str(item) for item in value.get("compute_resources") or ()),
        movement_resources=tuple(str(item) for item in value.get("movement_resources") or ()),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
        missing=tuple(str(item) for item in value.get("missing") or ()),
        **parsed,
    )


@dataclass(frozen=True)
class QualityLimit:
    """One independently evaluated accuracy or numerical-error constraint."""

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
    """Explicit per-model quality contract; no universal proxy is inferred."""

    limits: tuple[QualityLimit, ...]
    reference: str
    profile: str = "custom"

    def __post_init__(self) -> None:
        if not self.reference.strip() or not self.limits or not self.profile.strip():
            raise ValueError("a quality budget requires a profile, reference, and at least one limit")
        metrics = [limit.metric for limit in self.limits]
        if len(metrics) != len(set(metrics)):
            raise ValueError("quality metrics must be unique within one model budget")
        if self.profile == "classification_top1" and self.limits != (
            QualityLimit("top1_degradation_percentage_points", "at_most", 0.7),
        ):
            raise ValueError("classification top-1 profile must enforce a 0.7 point degradation cap")
        if self.profile == "numerical_similarity" and self.limits != (
            QualityLimit("cosine_similarity", "at_least", 0.99),
            QualityLimit("normalized_root_mean_square_error", "at_most", 0.02),
        ):
            raise ValueError("numerical similarity profile must enforce cosine and normalized RMSE limits")

    @classmethod
    def classification_top1(cls, reference: str) -> QualityBudget:
        """At most 0.7 percentage-point top-1 degradation against the bound corpus."""
        return cls(
            (QualityLimit("top1_degradation_percentage_points", "at_most", 0.7),),
            reference,
            "classification_top1",
        )

    @classmethod
    def numerical_similarity(cls, reference: str) -> QualityBudget:
        """Cosine >= 0.99 and normalized RMSE <= 0.02 on the bound corpus."""
        return cls(
            (
                QualityLimit("cosine_similarity", "at_least", 0.99),
                QualityLimit("normalized_root_mean_square_error", "at_most", 0.02),
            ),
            reference,
            "numerical_similarity",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "reference": self.reference,
            "limits": [limit.to_dict() for limit in self.limits],
        }


@dataclass(frozen=True)
class FourModelQualitySchema:
    """Frozen quality policy for four content-addressed portfolio members."""

    mode: str
    ordered_member_sha256s: tuple[str, ...]
    budgets: tuple[tuple[str, QualityBudget], ...]
    reason: str

    def __post_init__(self) -> None:
        if len(self.ordered_member_sha256s) != PORTFOLIO_MEMBER_COUNT:
            raise ValueError("the quality schema requires exactly four portfolio members")
        if len(set(self.ordered_member_sha256s)) != PORTFOLIO_MEMBER_COUNT or any(
            not _is_sha256(member) for member in self.ordered_member_sha256s
        ):
            raise ValueError("portfolio members require distinct SHA-256 identities")
        if self.mode not in ("accuracy_bounded", "exact_only"):
            raise ValueError("quality schema mode must be accuracy_bounded or exact_only")
        budget_members = tuple(member for member, _ in self.budgets)
        if self.mode == "accuracy_bounded" and budget_members != self.ordered_member_sha256s:
            raise ValueError("accuracy-bounded budgets must cover all four members in portfolio order")
        if self.mode == "exact_only" and self.budgets:
            raise ValueError("exact-only fallback cannot carry approximate quality budgets")
        if not self.reason.strip():
            raise ValueError("quality schema must explain its authority or fallback")

    @property
    def budget_map(self) -> dict[str, QualityBudget]:
        return dict(self.budgets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "phase2_four_model_quality_schema_v1",
            "mode": self.mode,
            "ordered_member_sha256s": list(self.ordered_member_sha256s),
            "budgets": {member: budget.to_dict() for member, budget in self.budgets},
            "reason": self.reason,
            "approximation_allowed": self.mode == "accuracy_bounded",
        }


def standard_four_model_quality_schema(
    ordered_member_sha256s: Sequence[str],
    *,
    classification_member_sha256: str,
    corpus_sha256_by_member: Mapping[str, str] | None,
) -> FourModelQualitySchema:
    """Build the paper policy, falling back to exact semantics when any corpus is missing."""
    members = tuple(ordered_member_sha256s)
    if len(members) != PORTFOLIO_MEMBER_COUNT:
        raise ValueError("the standard quality schema requires exactly four members")
    if classification_member_sha256 not in members:
        raise ValueError("classification member must belong to the four-model portfolio")
    if (
        corpus_sha256_by_member is None
        or set(corpus_sha256_by_member) != set(members)
        or any(not _is_sha256(value) for value in corpus_sha256_by_member.values())
    ):
        return FourModelQualitySchema(
            "exact_only",
            members,
            (),
            "held-out quality corpus is absent or incomplete; approximate transformations are disabled",
        )
    budgets = tuple(
        (
            member,
            (
                QualityBudget.classification_top1(f"held-out corpus sha256:{corpus_sha256_by_member[member]}")
                if member == classification_member_sha256
                else QualityBudget.numerical_similarity(
                    f"held-out corpus sha256:{corpus_sha256_by_member[member]}"
                )
            ),
        )
        for member in members
    )
    return FourModelQualitySchema(
        "accuracy_bounded",
        members,
        budgets,
        "all four held-out corpora have exact identities",
    )


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
    """Supported source work placement and host/accelerator boundary topology."""

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
        _finite_nonnegative(self.boundary_crossings, "boundary crossings", integer=True)
        if self.supported_work_total <= 0:
            raise ValueError("supported source work total must be positive")
        if self.supported_work_placed > self.supported_work_total:
            raise ValueError("placed work cannot exceed supported source work")
        if self.largest_connected_region_work > self.supported_work_placed:
            raise ValueError("largest connected region cannot exceed placed work")
        if tuple(sorted(self.connected_region_work, reverse=True)) != self.connected_region_work:
            raise ValueError("connected region work must be descending")
        for work in self.connected_region_work:
            _finite_nonnegative(work, "connected region work")
        if self.supported_work_placed > 0 and not self.connected_region_work:
            raise ValueError("placed work requires at least one connected region")
        if self.supported_work_placed == 0 and self.connected_region_work:
            raise ValueError("zero placed work cannot have connected regions")
        if self.connected_region_work and self.connected_region_work[0] != self.largest_connected_region_work:
            raise ValueError("largest connected region must match the first region")
        if not math.isclose(
            sum(self.connected_region_work), self.supported_work_placed, rel_tol=0.0, abs_tol=1e-9
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
                {"taxonomy": name, "count": count, "work": work}
                for name, count, work in self.host_islands
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
    parsed_islands = []
    for island in islands:
        if not isinstance(island, Mapping):
            raise TypeError("host island entries must be mappings")
        parsed_islands.append(
            (
                str(island.get("taxonomy") or ""),
                int(_finite_nonnegative(island.get("count"), "host island count", integer=True)),
                float(_finite_nonnegative(island.get("work"), "host island work")),
            )
        )
    return CoverageSummary(
        supported_work_total=float(
            _finite_nonnegative(value.get("supported_work_total"), "supported work total")
        ),
        supported_work_placed=float(
            _finite_nonnegative(value.get("supported_work_placed"), "supported work placed")
        ),
        largest_connected_region_work=float(
            _finite_nonnegative(value.get("largest_connected_region_work"), "largest connected region")
        ),
        connected_region_work=tuple(
            float(_finite_nonnegative(item, "connected region work"))
            for item in value.get("connected_region_work") or ()
        ),
        host_islands=tuple(sorted(parsed_islands)),
        boundary_crossings=int(
            _finite_nonnegative(value.get("boundary_crossings"), "boundary crossings", integer=True)
        ),
        boundary_bytes=float(_finite_nonnegative(value.get("boundary_bytes"), "boundary bytes")),
        work_unit=str(value.get("work_unit") or ""),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
    )


@dataclass(frozen=True)
class RooflineSummary:
    """Adapter-composed physical lower bound; composition is never assumed here."""

    lower_bound_cycles: float
    resource_floors: tuple[tuple[str, float], ...]
    limiting_resources: tuple[str, ...]
    optimization_effects: tuple[str, ...]
    composition: str
    provenance: tuple[str, ...]

    def __post_init__(self) -> None:
        _finite_nonnegative(self.lower_bound_cycles, "roofline lower bound")
        if tuple(sorted(self.resource_floors)) != self.resource_floors:
            raise ValueError("roofline resource floors must be sorted")
        floors = dict(self.resource_floors)
        if len(floors) != len(self.resource_floors) or not floors:
            raise ValueError("roofline resource floors must be nonempty and unique")
        for resource, cycles in self.resource_floors:
            if not resource.strip():
                raise ValueError("roofline resource names cannot be empty")
            _finite_nonnegative(cycles, f"roofline floor for {resource}")
        if self.lower_bound_cycles < max(floors.values()):
            raise ValueError("composed roofline bound cannot be below a resource floor")
        if (
            tuple(sorted(self.limiting_resources)) != self.limiting_resources
            or not self.limiting_resources
            or len(set(self.limiting_resources)) != len(self.limiting_resources)
            or any(resource not in floors for resource in self.limiting_resources)
        ):
            raise ValueError("roofline limiters must uniquely name declared resources")
        if (
            tuple(sorted(self.optimization_effects)) != self.optimization_effects
            or not self.optimization_effects
            or len(set(self.optimization_effects)) != len(self.optimization_effects)
            or any(not effect.strip() for effect in self.optimization_effects)
        ):
            raise ValueError("roofline optimization effects must be nonempty, unique, and sorted")
        if not self.composition.strip() or not self.provenance:
            raise ValueError("roofline requires explicit composition and provenance")

    def to_dict(self, total_cycles: float | None = None) -> dict[str, Any]:
        return {
            "lower_bound_cycles": self.lower_bound_cycles,
            "resource_floors": dict(self.resource_floors),
            "limiting_resources": list(self.limiting_resources),
            "optimization_effects": list(self.optimization_effects),
            "composition": self.composition,
            "provenance": list(self.provenance),
            "headroom_to_lower_bound": (
                None
                if total_cycles is None or self.lower_bound_cycles == 0
                else total_cycles / self.lower_bound_cycles
            ),
            "attainment_fraction": (
                None if total_cycles is None or total_cycles == 0 else self.lower_bound_cycles / total_cycles
            ),
        }


def _roofline(value: RooflineSummary | Mapping[str, Any] | None) -> RooflineSummary | None:
    if value is None or isinstance(value, RooflineSummary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("roofline must be a RooflineSummary or mapping")
    floors = value.get("resource_floors") or {}
    if not isinstance(floors, Mapping):
        raise TypeError("roofline resource floors must be a mapping")
    return RooflineSummary(
        lower_bound_cycles=float(
            _finite_nonnegative(value.get("lower_bound_cycles"), "roofline lower bound")
        ),
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
    """One arm's host-produced whole-program analytical evidence."""

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
            if interval.resolved and not all(
                math.isfinite(float(endpoint)) for endpoint in (interval.lo, interval.hi)
            ):
                raise ValueError(f"{name} must be finite")
        if self.movement_scope not in ("physical", "unavailable"):
            raise ValueError("movement scope must be physical or unavailable")
        if self.movement_bytes is not None:
            _finite_nonnegative(self.movement_bytes, "movement bytes")
            if self.movement_scope != "physical":
                raise ValueError("known movement bytes must describe physical movement")
        if self.encoding_conversion_count is not None:
            _finite_nonnegative(self.encoding_conversion_count, "encoding count", integer=True)
        if self.encoding_conversion_bytes is not None:
            _finite_nonnegative(self.encoding_conversion_bytes, "encoding bytes")
        if self.risk_score is not None:
            _finite_nonnegative(self.risk_score, "risk score")
            if self.risk_score > 1:
                raise ValueError("risk score must be in [0, 1]")
        if not self.provenance:
            raise ValueError("analytical metrics require provenance")
        if self.roofline is not None and self.cycles.resolved:
            if self.roofline.lower_bound_cycles > float(self.cycles.lo):
                raise ValueError("roofline lower bound exceeds the analytical cycle interval")
        if self.occupancy is not None and self.cycles.resolved:
            if not math.isclose(
                self.occupancy.total_cycles, float(self.cycles.hi), rel_tol=0.0, abs_tol=1e-9
            ):
                raise ValueError("occupancy total must equal the conservative cycle endpoint")
        if (
            self.occupancy is not None
            and self.occupancy.movement_bytes is not None
            and self.movement_bytes is not None
            and not math.isclose(
                self.occupancy.movement_bytes, self.movement_bytes, rel_tol=0.0, abs_tol=1e-9
            )
        ):
            raise ValueError("movement bytes disagree with the occupancy timeline")
        if (
            self.occupancy is not None
            and self.occupancy.encoding_transitions is not None
            and self.encoding_conversion_count is not None
            and self.occupancy.encoding_transitions != self.encoding_conversion_count
        ):
            raise ValueError("encoding count disagrees with the occupancy timeline")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AnalyticalMetrics:
        encoding = value.get("encoding_conversions") or {}
        if not isinstance(encoding, Mapping):
            raise TypeError("encoding conversions must be a mapping")
        movement, count, byte_count, risk = (
            value.get("movement_bytes"),
            encoding.get("count"),
            encoding.get("bytes"),
            value.get("risk_score"),
        )
        return cls(
            cycles=_interval(value.get("cycles"), "whole-model cycles"),
            movement_bytes=(
                None if movement is None else float(_finite_nonnegative(movement, "movement bytes"))
            ),
            movement_scope=str(value.get("movement_scope") or "unavailable"),
            occupancy=_occupancy(value.get("occupancy")),
            coverage=_coverage(value.get("coverage")),
            roofline=_roofline(value.get("roofline")),
            encoding_conversion_count=(
                None if count is None else int(_finite_nonnegative(count, "encoding count", integer=True))
            ),
            encoding_conversion_bytes=(
                None if byte_count is None else float(_finite_nonnegative(byte_count, "encoding bytes"))
            ),
            encoding_conversion_cycles=_interval(encoding.get("cycles"), "encoding conversion cycles"),
            risk_score=(None if risk is None else float(_finite_nonnegative(risk, "risk score"))),
            provenance=tuple(str(item) for item in value.get("provenance") or ()),
        )

    def objectives(self) -> dict[str, float | CycleInterval | None]:
        return {
            "cycles": self.cycles,
            "movement_bytes": self.movement_bytes,
            "compute_utilization": (
                None if self.occupancy is None else self.occupancy.compute_utilization
            ),
            "latency_hiding_efficiency": (
                None if self.occupancy is None else self.occupancy.latency_hiding_efficiency
            ),
            "encoding_conversion_count": self.encoding_conversion_count,
            "encoding_conversion_bytes": self.encoding_conversion_bytes,
            "encoding_conversion_cycles": self.encoding_conversion_cycles,
            "supported_work_placed_fraction": (
                None if self.coverage is None else self.coverage.supported_work_placed_fraction
            ),
            "largest_connected_region_fraction": (
                None if self.coverage is None else self.coverage.largest_connected_region_fraction
            ),
            "host_island_count": None if self.coverage is None else self.coverage.host_island_count,
            "boundary_crossings": None if self.coverage is None else self.coverage.boundary_crossings,
            "boundary_bytes": None if self.coverage is None else self.coverage.boundary_bytes,
        }

    def to_dict(self) -> dict[str, Any]:
        total = None if not self.cycles.resolved else float(self.cycles.hi)
        return {
            "cycles": self.cycles.to_dict(),
            "movement_bytes": self.movement_bytes,
            "movement_scope": self.movement_scope,
            "occupancy": None if self.occupancy is None else self.occupancy.to_dict(),
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "roofline": None if self.roofline is None else self.roofline.to_dict(total),
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
    """Conservative per-model Pareto, uncertainty, roofline, and risk policy."""

    required_objectives: tuple[str, ...] = tuple(_DIRECTIONS)
    maximum_regression_fraction: tuple[tuple[str, float], ...] = tuple(
        (name, 0.0) for name in _DIRECTIONS
    )
    minimum_improvement_fraction: float = 0.0
    maximum_risk_score: float = 0.25
    maximum_cycle_interval_width_fraction: float = 0.25
    require_improvement: bool = True
    require_roofline: bool = True
    global_benefit_objectives: tuple[str, ...] = _GLOBAL_BENEFIT_OBJECTIVES

    def __post_init__(self) -> None:
        if not isinstance(self.require_improvement, bool) or not isinstance(self.require_roofline, bool):
            raise TypeError("fast-evaluation boolean policies must be booleans")
        if set(self.required_objectives) - set(_DIRECTIONS) or len(self.required_objectives) != len(
            set(self.required_objectives)
        ):
            raise ValueError("required objectives must be unique known objectives")
        if set(self.global_benefit_objectives) - set(_DIRECTIONS) or len(
            self.global_benefit_objectives
        ) != len(set(self.global_benefit_objectives)):
            raise ValueError("global benefit objectives must be unique known objectives")
        regressions = dict(self.maximum_regression_fraction)
        if len(regressions) != len(self.maximum_regression_fraction) or set(regressions) - set(
            _DIRECTIONS
        ):
            raise ValueError("maximum regression policy must name unique known objectives")
        for value in (
            *regressions.values(),
            self.minimum_improvement_fraction,
            self.maximum_risk_score,
            self.maximum_cycle_interval_width_fraction,
        ):
            _finite_nonnegative(value, "fast-evaluation policy fraction")
        if self.maximum_risk_score > 1:
            raise ValueError("maximum risk score must be in [0, 1]")

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
        raise TypeError("quality values must be a mapping")
    parsed = []
    for name, number in values.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"quality observation {name} must be numeric")
        parsed.append((str(name), float(number)))
    return QualityObservation(
        values=tuple(sorted(parsed)),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
        complete=value.get("complete") is True,
    )


def _quality_gate(
    budget: QualityBudget,
    candidate: QualityObservation | None,
    baseline: QualityObservation | None,
) -> dict[str, Any]:
    checks, blockers, failures = [], [], []
    candidate_values = {} if candidate is None else candidate.value_map
    baseline_values = {} if baseline is None else baseline.value_map
    if candidate is None or not candidate.complete:
        blockers.append("complete candidate quality observation is unavailable")
    for limit in budget.limits:
        value, base = candidate_values.get(limit.metric), baseline_values.get(limit.metric)
        threshold_passes = None
        degradation_passes = None
        if value is None:
            blockers.append(f"candidate quality metric {limit.metric} is unavailable")
        else:
            threshold_passes = (
                value <= limit.threshold if limit.direction == "at_most" else value >= limit.threshold
            )
            if not threshold_passes:
                failures.append(f"candidate quality metric {limit.metric} exceeds its budget")
        if limit.maximum_degradation is not None:
            if baseline is None or not baseline.complete:
                blockers.append("complete baseline quality observation is unavailable")
            if base is None:
                blockers.append(f"baseline quality metric {limit.metric} is unavailable")
            elif value is not None:
                degradation = value - base if limit.direction == "at_most" else base - value
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
    return {
        "status": "failed" if failures else "needs_evidence" if blockers else "passed",
        "profile": budget.profile,
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


def _compare(
    name: str,
    baseline: float | CycleInterval | None,
    candidate: float | CycleInterval | None,
    allowed_regression: float,
    minimum_improvement: float,
) -> dict[str, Any]:
    direction = _DIRECTIONS[name]
    if isinstance(baseline, CycleInterval) or isinstance(candidate, CycleInterval):
        if not isinstance(baseline, CycleInterval) or not isinstance(candidate, CycleInterval):
            return {"objective": name, "direction": direction, "status": "UNKNOWN"}
        if not baseline.resolved or not candidate.resolved:
            missing = (*baseline.missing, *candidate.missing)
            return {
                "objective": name,
                "direction": direction,
                "status": "UNKNOWN",
                "reason": "; ".join(dict.fromkeys(missing)),
            }
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
            "conservative_speedup": None if candidate.hi == 0 else baseline.lo / candidate.hi,
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
    else:
        acceptable = candidate_value >= baseline_value * (1.0 - allowed_regression)
        improvement = candidate_value > baseline_value * (1.0 + minimum_improvement)
    return {
        "objective": name,
        "direction": direction,
        "status": "non_regression" if acceptable else "regression",
        "baseline": baseline_value,
        "candidate": candidate_value,
        "candidate_minus_baseline_fraction": (
            None if baseline_value == 0 else (candidate_value - baseline_value) / baseline_value
        ),
        "allowed_regression_fraction": allowed_regression,
        "robust_improvement": improvement,
    }


def _recommended_levers(
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
        if comparison.get("status") != "non_regression":
            lever = objective_lever[str(comparison["objective"])]
            if lever not in needed:
                needed.append(lever)
    if quality_status != "passed":
        needed.insert(0, "quality_budget")

    def authorized(effects: frozenset[str]) -> list[dict[str, Any]]:
        result = []
        for surface in surfaces:
            declared = frozenset(str(item) for item in surface.get("effects") or ())
            if declared.intersection(effects):
                result.append(
                    {
                        key: surface.get(key)
                        for key in ("id", "path", "symbol", "scope", "effects")
                    }
                )
        return result

    result = []
    for lever in needed:
        effects = _LEVER_EFFECTS[lever]
        matched = authorized(effects)
        result.append(
            {
                "lever": lever,
                "addresses_effects": sorted(effects),
                "authorized_surfaces": matched,
                "authority": "host_frozen_surfaces_only" if matched else "no_authorized_surface_matches",
            }
        )
    if (
        roofline is not None
        and total_cycles is not None
        and roofline.lower_bound_cycles > 0
        and total_cycles > roofline.lower_bound_cycles
    ):
        effects = frozenset(roofline.optimization_effects)
        matched = authorized(effects)
        result.append(
            {
                "lever": "roofline_headroom",
                "limiting_resources": list(roofline.limiting_resources),
                "headroom_to_lower_bound": total_cycles / roofline.lower_bound_cycles,
                "addresses_effects": list(roofline.optimization_effects),
                "authorized_surfaces": matched,
                "authority": "host_frozen_surfaces_only" if matched else "no_authorized_surface_matches",
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
    """Apply quality, analytical Pareto, uncertainty, roofline, and risk gates."""
    expected = tuple(expected_models or (str(row.get("model_id")) for row in rows))
    if (
        len(expected) != PORTFOLIO_MEMBER_COUNT
        or len(set(expected)) != PORTFOLIO_MEMBER_COUNT
        or any(not _is_sha256(model) for model in expected)
    ):
        raise ValueError("fast evaluation requires four distinct content-addressed models")
    by_model = {str(row.get("model_id")): row for row in rows}
    if len(by_model) != len(rows) or set(by_model) != set(expected):
        raise ValueError("fast-evaluation rows must exactly cover the ordered portfolio")
    if set(quality_budgets) != set(expected):
        raise ValueError("quality budgets must exactly cover the ordered portfolio")

    regressions = policy.regression_map
    model_results, all_failures, all_blockers, speedups = [], [], [], []
    any_global_improvement = False
    for model_id in expected:
        raw = by_model[model_id]
        provider_failure = raw.get("provider_failure")
        if isinstance(provider_failure, Mapping):
            reason = (
                f"host analytical provider failed: {provider_failure.get('type')}: "
                f"{provider_failure.get('reason')}"
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
        baseline_objectives, candidate_objectives = baseline.objectives(), candidate.objectives()
        comparisons = [
            _compare(
                objective,
                baseline_objectives[objective],
                candidate_objectives[objective],
                regressions.get(objective, 0.0),
                policy.minimum_improvement_fraction,
            )
            for objective in _DIRECTIONS
        ]
        failures = [
            f"{model_id}: {item['objective']} regressed"
            for item in comparisons
            if item["status"] == "regression"
        ]
        blockers = [
            f"{model_id}: {item['objective']} is {item['status']}"
            for item in comparisons
            if item["objective"] in policy.required_objectives
            and item["status"] in ("UNKNOWN", "UNCERTAIN")
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
        improvements = [
            item["objective"] for item in comparisons if item.get("robust_improvement") is True
        ]
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
                "recommended_levers": _recommended_levers(
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
        "ordered_model_sha256s": list(expected),
        "policy": policy.to_dict(),
        "quality_budgets": {model: quality_budgets[model].to_dict() for model in expected},
        "portfolio_conservative_cycle_speedup_geomean": geomean,
        "failures": sorted(set(all_failures)),
        "blockers": sorted(set(all_blockers)),
        "aggregation": "dimensionless per-model ratios only; model cycles are never summed",
        "execution": "serialized_host_analytical_only_no_complete_model_or_layer_simulation",
        "maximum_parallel_model_evaluations": 1,
        "authority": "candidate metrics and quality come only from the bound host-owned adapter",
    }


def unavailable_fast_evaluation(*, reason: str) -> dict[str, Any]:
    """Stable exact-only record when held-out quality evidence is absent."""
    if not reason.strip():
        raise ValueError("exact-only fallback requires a reason")
    return {
        "schema": "phase2_fast_portfolio_evaluation_v1",
        "status": "exact_only_fallback",
        "reason": reason,
        "selection": "exact_semantics_only_until_quality_corpus_is_bound",
        "approximation_allowed": False,
        "execution": "serialized_host_analytical_only_no_complete_model_or_layer_simulation",
        "maximum_parallel_model_evaluations": 1,
        "unknown_metrics": [*_DIRECTIONS, "quality_budget", "calibration_risk"],
        "aggregation": "model cycles are never summed",
    }
