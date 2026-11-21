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

from merlin.analysis.optimization.evaluation.global_planner import OccupancySummary
from merlin.common import digest as _mdigest
from merlin.compiler.partition.global_plan import CycleInterval

# ---- phase2_portfolio --------------------------------------------------------------------------

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


_is_sha256 = _mdigest.is_sha256


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
        name: (None if value.get(name) is None else float(_finite_nonnegative(value[name], name.replace("_", " "))))
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
                (str(key), float(_finite_nonnegative(amount, f"busy cycles for {key}"))) for key, amount in busy.items()
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
    parameters: tuple[tuple[str, float], ...] = ()

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
        if type(self.parameters) is not tuple:
            raise TypeError("quality parameters must be an immutable tuple")
        if tuple(sorted(self.parameters)) != self.parameters or len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("quality parameters must be unique and sorted")
        for name, value in self.parameters:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("quality parameter names must be nonempty")
            _finite_nonnegative(value, "quality parameter")
        if self.profile in ("exact", "elementwise"):
            metric = "bitwise_mismatch_count" if self.profile == "exact" else "elementwise_violation_count"
            if self.limits != (QualityLimit(metric, "at_most", 0.0),):
                raise ValueError("exact and elementwise profiles require zero complete-output violations")
            required = () if self.profile == "exact" else ("atol", "rtol")
            if tuple(name for name, _ in self.parameters) != required:
                raise ValueError("elementwise quality requires explicit atol and rtol; exact has no parameters")

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

    @classmethod
    def exact(cls, reference: str) -> QualityBudget:
        """Zero bitwise mismatches over every output, including signed zeros."""
        return cls((QualityLimit("bitwise_mismatch_count", "at_most", 0.0),), reference, "exact")

    @classmethod
    def elementwise(cls, reference: str, *, atol: float, rtol: float) -> QualityBudget:
        """Count violations of abs(candidate-reference) <= atol+rtol*abs(reference).

        The independent observer owns full output coverage and nonfinite-value
        semantics. A maximum-error or cosine proxy cannot supply this metric.
        Its observation must bind these exact tolerance parameters.
        """
        return cls(
            (QualityLimit("elementwise_violation_count", "at_most", 0.0),),
            reference,
            "elementwise",
            (("atol", atol), ("rtol", rtol)),
        )

    @classmethod
    def task(cls, reference: str, *, limits: tuple[QualityLimit, ...]) -> QualityBudget:
        """Caller-defined task metrics and thresholds, with no default proxy."""
        return cls(limits, reference, "task")

    def to_dict(self) -> dict[str, Any]:
        result = {
            "profile": self.profile,
            "reference": self.reference,
            "limits": [limit.to_dict() for limit in self.limits],
        }
        if self.parameters:
            result["parameters"] = dict(self.parameters)
        return result


@dataclass(frozen=True)
class PortfolioQualitySchema:
    """Frozen explicit policy for any nonempty content-addressed portfolio."""

    mode: str
    ordered_member_sha256s: tuple[str, ...]
    budgets: tuple[tuple[str, QualityBudget], ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.ordered_member_sha256s:
            raise ValueError("the quality schema requires a nonempty portfolio")
        if len(set(self.ordered_member_sha256s)) != len(self.ordered_member_sha256s) or any(
            not _is_sha256(member) for member in self.ordered_member_sha256s
        ):
            raise ValueError("portfolio members require distinct SHA-256 identities")
        if self.mode not in ("accuracy_bounded", "exact_only"):
            raise ValueError("quality schema mode must be accuracy_bounded or exact_only")
        budget_members = tuple(member for member, _ in self.budgets)
        if self.mode == "accuracy_bounded" and budget_members != self.ordered_member_sha256s:
            raise ValueError("accuracy-bounded budgets must cover all members in portfolio order")
        if any(not isinstance(budget, QualityBudget) for _, budget in self.budgets):
            raise TypeError("quality schema requires typed QualityBudget records")
        if self.mode == "exact_only" and self.budgets:
            raise ValueError("exact-only fallback cannot carry approximate quality budgets")
        if not self.reason.strip():
            raise ValueError("quality schema must explain its authority or fallback")

    @property
    def budget_map(self) -> dict[str, QualityBudget]:
        return dict(self.budgets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "phase2_portfolio_quality_schema_v1",
            "mode": self.mode,
            "ordered_member_sha256s": list(self.ordered_member_sha256s),
            "budgets": {member: budget.to_dict() for member, budget in self.budgets},
            "reason": self.reason,
            "approximation_allowed": self.mode == "accuracy_bounded"
            and any(budget.profile != "exact" for _, budget in self.budgets),
        }


@dataclass(frozen=True)
class FourModelQualitySchema(PortfolioQualitySchema):
    """Compatibility wrapper retaining the historical four-member policy."""

    def __post_init__(self) -> None:
        if len(self.ordered_member_sha256s) != PORTFOLIO_MEMBER_COUNT:
            raise ValueError("the quality schema requires exactly four portfolio members")
        super().__post_init__()

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["schema"] = "phase2_four_model_quality_schema_v1"
        return result


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
                else QualityBudget.numerical_similarity(f"held-out corpus sha256:{corpus_sha256_by_member[member]}")
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
    parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if tuple(sorted(self.values)) != self.values or len(dict(self.values)) != len(self.values):
            raise ValueError("quality values must be unique and sorted")
        if any(not name.strip() for name, _ in self.values):
            raise ValueError("quality metric names cannot be empty")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for _, value in self.values
        ):
            raise ValueError("quality observations must be finite numeric values")
        if type(self.complete) is not bool:
            raise TypeError("quality observation completeness must be a boolean")
        if not self.provenance:
            raise ValueError("quality observations require independent provenance")
        if type(self.parameters) is not tuple:
            raise TypeError("quality observation parameters must be an immutable tuple")
        if tuple(sorted(self.parameters)) != self.parameters or len(dict(self.parameters)) != len(self.parameters):
            raise ValueError("quality observation parameters must be unique and sorted")
        for name, value in self.parameters:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("quality observation parameter names must be nonempty")
            _finite_nonnegative(value, "quality observation parameter")

    @property
    def value_map(self) -> dict[str, float]:
        return {name: float(value) for name, value in self.values}

    def to_dict(self) -> dict[str, Any]:
        result = {"values": self.value_map, "provenance": list(self.provenance), "complete": self.complete}
        if self.parameters:
            result["parameters"] = dict(self.parameters)
        return result


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
        if not math.isclose(sum(self.connected_region_work), self.supported_work_placed, rel_tol=0.0, abs_tol=1e-9):
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
        supported_work_total=float(_finite_nonnegative(value.get("supported_work_total"), "supported work total")),
        supported_work_placed=float(_finite_nonnegative(value.get("supported_work_placed"), "supported work placed")),
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
                None if total_cycles is None or self.lower_bound_cycles == 0 else total_cycles / self.lower_bound_cycles
            ),
            "attainment_fraction": (
                None if total_cycles is None or total_cycles == 0 else self.lower_bound_cycles / total_cycles
            ),
        }
