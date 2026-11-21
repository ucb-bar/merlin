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
from merlin.analysis.optimization.evaluation.portfolio_quality import (
    _DIRECTIONS,
    _GLOBAL_BENEFIT_OBJECTIVES,
    CoverageSummary,
    RooflineSummary,
    _coverage,
    _finite_nonnegative,
    _interval,
    _occupancy,
)
from merlin.common import digest as _mdigest
from merlin.compiler.partition.global_plan import CycleInterval

# ---- phase2_portfolio --------------------------------------------------------------------------


def _roofline(value: RooflineSummary | Mapping[str, Any] | None) -> RooflineSummary | None:
    from merlin.analysis.optimization.evaluation.portfolio_quality import RooflineSummary, _finite_nonnegative

    if value is None or isinstance(value, RooflineSummary):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("roofline must be a RooflineSummary or mapping")
    floors = value.get("resource_floors") or {}
    if not isinstance(floors, Mapping):
        raise TypeError("roofline resource floors must be a mapping")
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
            if interval.resolved and not all(math.isfinite(float(endpoint)) for endpoint in (interval.lo, interval.hi)):
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
            movement_bytes=(None if movement is None else float(_finite_nonnegative(movement, "movement bytes"))),
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
    maximum_regression_fraction: tuple[tuple[str, float], ...] = tuple((name, 0.0) for name in _DIRECTIONS)
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
        if set(self.global_benefit_objectives) - set(_DIRECTIONS) or len(self.global_benefit_objectives) != len(
            set(self.global_benefit_objectives)
        ):
            raise ValueError("global benefit objectives must be unique known objectives")
        regressions = dict(self.maximum_regression_fraction)
        if len(regressions) != len(self.maximum_regression_fraction) or set(regressions) - set(_DIRECTIONS):
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
    from merlin.analysis.optimization.evaluation.portfolio_quality import QualityObservation

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
    parameters = value.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise TypeError("quality observation parameters must be a mapping")
    return QualityObservation(
        values=tuple(sorted(parsed)),
        provenance=tuple(str(item) for item in value.get("provenance") or ()),
        complete=value.get("complete") is True,
        parameters=tuple(sorted((name, number) for name, number in parameters.items())),
    )


def _quality_gate(
    budget: QualityBudget,
    candidate: QualityObservation | None,
    baseline: QualityObservation | None,
) -> dict[str, Any]:
    from merlin.analysis.optimization.evaluation.portfolio_quality import QualityBudget, QualityObservation

    checks, blockers, failures = [], [], []
    candidate_values = {} if candidate is None else candidate.value_map
    baseline_values = {} if baseline is None else baseline.value_map
    if candidate is None or not candidate.complete:
        blockers.append("complete candidate quality observation is unavailable")
    if budget.parameters and candidate is not None and candidate.parameters != budget.parameters:
        blockers.append("candidate quality observation does not bind the selected metric parameters")
    for limit in budget.limits:
        value, base = candidate_values.get(limit.metric), baseline_values.get(limit.metric)
        threshold_passes = None
        degradation_passes = None
        if value is None:
            blockers.append(f"candidate quality metric {limit.metric} is unavailable")
        else:
            if budget.profile in ("exact", "elementwise") and (value < 0 or not value.is_integer()):
                blockers.append(f"candidate quality metric {limit.metric} is not a nonnegative integer count")
            threshold_passes = value <= limit.threshold if limit.direction == "at_most" else value >= limit.threshold
            if not threshold_passes:
                failures.append(f"candidate quality metric {limit.metric} exceeds its budget")
        if limit.maximum_degradation is not None:
            if baseline is None or not baseline.complete:
                blockers.append("complete baseline quality observation is unavailable")
            if budget.parameters and baseline is not None and baseline.parameters != budget.parameters:
                blockers.append("baseline quality observation does not bind the selected metric parameters")
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
    from merlin.analysis.optimization.evaluation.portfolio_quality import _DIRECTIONS

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


def _needs_evidence(model_id: str, reason: str) -> dict[str, Any]:
    """The row for a model whose analytical result could not be judged. It recommends no lever: a
    lever is a reading of measured evidence, and this row has none."""
    return {"model_id": model_id, "status": "needs_evidence", "reason": reason, "recommended_levers": []}


def _recommended_levers(
    comparisons: Sequence[Mapping[str, Any]],
    quality_status: str,
    surfaces: Sequence[Mapping[str, Any]],
    roofline: RooflineSummary | None,
    total_cycles: float | None,
) -> list[dict[str, Any]]:
    from merlin.analysis.optimization.evaluation.portfolio_quality import _LEVER_EFFECTS, RooflineSummary

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
                result.append({key: surface.get(key) for key in ("id", "path", "symbol", "scope", "effects")})
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
    from merlin.analysis.optimization.evaluation.portfolio_quality import _DIRECTIONS, QualityBudget, _is_sha256

    expected = tuple(str(row.get("model_id")) for row in rows) if expected_models is None else tuple(expected_models)
    if not expected or len(set(expected)) != len(expected) or any(not _is_sha256(model) for model in expected):
        raise ValueError("fast evaluation requires a nonempty portfolio of distinct content-addressed models")
    by_model = {str(row.get("model_id")): row for row in rows}
    if len(by_model) != len(rows) or set(by_model) != set(expected):
        raise ValueError("fast-evaluation rows must exactly cover the ordered portfolio")
    if set(quality_budgets) != set(expected):
        raise ValueError("quality budgets must exactly cover the ordered portfolio")
    if any(not isinstance(budget, QualityBudget) for budget in quality_budgets.values()):
        raise TypeError("quality budgets must contain typed QualityBudget records")

    regressions = policy.regression_map
    model_results, all_failures, all_blockers, speedups = [], [], [], []
    any_global_improvement = False
    for model_id in expected:
        raw = by_model[model_id]
        provider_failure = raw.get("provider_failure")
        if isinstance(provider_failure, Mapping):
            reason = (
                f"host analytical provider failed: {provider_failure.get('type')}: {provider_failure.get('reason')}"
            )
            model_results.append(_needs_evidence(model_id, reason))
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
            model_results.append(_needs_evidence(model_id, f"invalid analytical adapter result: {exc}"))
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
    from merlin.analysis.optimization.evaluation.portfolio_quality import _DIRECTIONS

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
