"""Portfolio reporting with a measured post-freeze gate, distinct from search estimates.

Timing provenance is supplied by an upstream receipt-admission layer; this module
does not authenticate receipts or launch measurements. Estimated rows still expose
useful ratios and opportunities, but cannot satisfy the measured-performance gate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .global_planner import GlobalPlanResult, OccupancySummary


WHOLE_MODEL_REPORT_SCHEMA = "whole_model_optimization_report_v1"


@dataclass(frozen=True)
class ModelPerformance:
    model: str
    baseline_cycles: float
    candidate_cycles: float
    legal_floor_cycles: float | None
    physical_floor_cycles: float | None
    heldout: bool
    occupancy: OccupancySummary | None = None
    plan_digest: str = ""
    provenance: tuple[str, ...] = ()
    cycle_basis: str = "unspecified"

    def __post_init__(self) -> None:
        if self.cycle_basis not in {"unspecified", "model_estimate", "measured"}:
            raise ValueError("cycle_basis must distinguish measured cycles from model estimates")
        if not self.model.strip() or self.baseline_cycles <= 0 or self.candidate_cycles <= 0:
            raise ValueError("a model performance row needs a name and positive cycle counts")
        for name in ("legal_floor_cycles", "physical_floor_cycles"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def speedup(self) -> float:
        return self.baseline_cycles / self.candidate_cycles

    @property
    def regression_fraction(self) -> float:
        return (self.candidate_cycles - self.baseline_cycles) / self.baseline_cycles

    @property
    def legal_attainment(self) -> float | None:
        if self.legal_floor_cycles is None:
            return None
        return self.legal_floor_cycles / self.candidate_cycles

    @property
    def physical_attainment(self) -> float | None:
        if self.physical_floor_cycles is None:
            return None
        return self.physical_floor_cycles / self.candidate_cycles

    @property
    def gap_closed(self) -> float | None:
        if self.legal_floor_cycles is None:
            return None
        gap = self.baseline_cycles - self.legal_floor_cycles
        if gap <= 0:
            return None
        return (self.baseline_cycles - self.candidate_cycles) / gap

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "baseline_cycles": self.baseline_cycles,
            "candidate_cycles": self.candidate_cycles,
            "cycle_basis": self.cycle_basis,
            "speedup": self.speedup,
            "regression_fraction": self.regression_fraction,
            "legal_floor_cycles": self.legal_floor_cycles,
            "legal_attainment": self.legal_attainment,
            "physical_floor_cycles": self.physical_floor_cycles,
            "physical_attainment": self.physical_attainment,
            "baseline_to_legal_gap_closed": self.gap_closed,
            "heldout": self.heldout,
            "plan_digest": self.plan_digest,
            "occupancy": self.occupancy.to_dict() if self.occupancy is not None else None,
            "provenance": list(self.provenance),
        }


def performance_from_plan(model: str, baseline_cycles: float, result: GlobalPlanResult, *,
                          heldout: bool, provenance: tuple[str, ...] = ()) -> ModelPerformance:
    if result.plan is None or not result.plan.cycles.resolved:
        raise ValueError(f"{model}: no resolved global plan")
    return ModelPerformance(
        model=model,
        baseline_cycles=float(baseline_cycles),
        candidate_cycles=float(result.plan.cycles.hi),
        legal_floor_cycles=result.legal_floor.cycles,
        physical_floor_cycles=result.physical_floor.cycles,
        heldout=heldout,
        occupancy=result.occupancy,
        plan_digest=result.plan.digest,
        provenance=provenance,
        cycle_basis="model_estimate",
    )


@dataclass(frozen=True)
class WholeModelGatePolicy:
    minimum_heldout_geomean_speedup: float = 2.0
    maximum_individual_regression: float = 0.02
    minimum_aggregate_gap_closed: float = 0.80
    minimum_legal_attainment: float = 0.90


@dataclass(frozen=True)
class OptimizationOpportunity:
    kind: str
    model: str
    cycles: float | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "model": self.model,
                "cycles": self.cycles, "detail": self.detail}


@dataclass(frozen=True)
class WholeModelReport:
    models: tuple[ModelPerformance, ...]
    heldout_geomean_speedup: float | None
    aggregate_gap_closed: float | None
    worst_legal_attainment: float | None
    worst_regression: float
    gates: tuple[tuple[str, bool], ...]
    opportunities: tuple[OptimizationOpportunity, ...]
    refusals: tuple[str, ...] = ()
    schema: str = WHOLE_MODEL_REPORT_SCHEMA

    @property
    def passed(self) -> bool:
        return not self.refusals and all(value for _, value in self.gates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "status": "pass" if self.passed else "refused",
            "models": [row.to_dict() for row in self.models],
            "aggregate": {
                "cycle_basis": (self.models[0].cycle_basis if self.models
                    and len({row.cycle_basis for row in self.models}) == 1 else "mixed_or_unspecified"),
                "heldout_geomean_speedup": self.heldout_geomean_speedup,
                "baseline_to_legal_gap_closed": self.aggregate_gap_closed,
                "worst_legal_attainment": self.worst_legal_attainment,
                "worst_regression": self.worst_regression,
            },
            "gates": dict(self.gates),
            "opportunities": [item.to_dict() for item in self.opportunities],
            "refusals": list(self.refusals),
        }


def _opportunities(rows: tuple[ModelPerformance, ...]) -> tuple[OptimizationOpportunity, ...]:
    found: list[OptimizationOpportunity] = []
    for row in rows:
        occupancy = row.occupancy
        if occupancy is None:
            found.append(OptimizationOpportunity(
                "missing_occupancy", row.model, None,
                "capture per-resource busy cycles and a dependency timeline before ranking levers"))
            continue
        compute_busy = occupancy.compute_busy_cycles
        if compute_busy is not None:
            compute_capacity = occupancy.total_cycles * len(occupancy.compute_resources)
            found.append(OptimizationOpportunity(
                "accelerator_bubbles", row.model, max(0.0, compute_capacity - compute_busy),
                "fill/drain, dependencies, and issue gaps leave declared compute capacity idle"))
        else:
            found.append(OptimizationOpportunity(
                "unknown_accelerator_bubbles", row.model, None,
                "busy cycles are needed for every declared compute resource before ranking idle capacity"))
        movement_elapsed = occupancy.movement_elapsed_cycles
        if movement_elapsed is None and len(occupancy.movement_resources) == 1:
            # One resource's busy time is its elapsed union; multiple resources
            # need overlap evidence, not a sum that double-counts their shared time.
            movement_elapsed = occupancy.busy.get(occupancy.movement_resources[0])
        if occupancy.overlap_cycles is None:
            found.append(OptimizationOpportunity(
                "unknown_latency_hiding", row.model, None,
                "compute/movement overlap was not observable"))
        elif movement_elapsed is not None:
            found.append(OptimizationOpportunity(
                "exposed_movement", row.model,
                max(0.0, movement_elapsed - occupancy.overlap_cycles),
                "movement elapsed cycles outside compute; not necessarily critical-path cycles"))
        else:
            found.append(OptimizationOpportunity(
                "unknown_exposed_movement", row.model, None,
                "multiple movement resources need their elapsed union before subtracting overlap"))
        if occupancy.encoding_transitions is None:
            found.append(OptimizationOpportunity(
                "unknown_encoding_transitions", row.model, None,
                "executed representation conversions were not counted"))
        elif occupancy.encoding_transitions:
            found.append(OptimizationOpportunity(
                "encoding_transitions", row.model, None,
                f"{occupancy.encoding_transitions} representation conversions remain in the plan"))
    return tuple(sorted(found, key=lambda item: (
        item.cycles is None, -(item.cycles or 0.0), item.model, item.kind)))


def evaluate_whole_models(rows: tuple[ModelPerformance, ...], *,
                          policy: WholeModelGatePolicy | None = None) -> WholeModelReport:
    """Evaluate admitted post-freeze model results, never require model simulation during search.

    Search can inspect estimates here without passing this measured gate. A measured
    label requires caller-supplied provenance; its authenticity is checked upstream.
    """
    pol = policy or WholeModelGatePolicy()
    if not rows:
        return WholeModelReport((), None, None, None, 0.0, (), (),
                                ("no whole-model results",))

    refusals: list[str] = []
    compatible: list[ModelPerformance] = []
    for row in rows:
        if row.cycle_basis != "measured" or not row.provenance:
            refusals.append(
                f"{row.model}: measured full-model cycle evidence is absent "
                f"(cycle_basis={row.cycle_basis}); estimates can guide search, not prove a timing win")
        occupancy = row.occupancy
        if (occupancy is None or occupancy.compute_utilization is None
                or any(name not in occupancy.busy for name in occupancy.movement_resources)
                or occupancy.latency_hiding_efficiency is None
                or occupancy.idle_cycles is None
                or occupancy.critical_path_cycles is None
                or occupancy.movement_bytes is None
                or occupancy.movement_commands is None
                or occupancy.encoding_transitions is None):
            refusals.append(
                f"{row.model}: whole-model occupancy/movement/latency-hiding evidence is incomplete")
        if row.legal_floor_cycles is None:
            refusals.append(f"{row.model}: legal roofline is UNKNOWN")
            continue
        if row.legal_floor_cycles > row.baseline_cycles:
            refusals.append(
                f"{row.model}: legal floor {row.legal_floor_cycles:g} exceeds baseline "
                f"{row.baseline_cycles:g}")
            continue
        if row.candidate_cycles + 1e-9 < row.legal_floor_cycles:
            refusals.append(
                f"{row.model}: candidate violates its legal floor; cost model is unsound")
            continue
        compatible.append(row)

    heldout = [row for row in rows if row.heldout]
    geomean = (math.exp(sum(math.log(row.speedup) for row in heldout) / len(heldout))
               if heldout else None)
    if not heldout:
        refusals.append("no held-out whole-model result")

    gap_denom = sum(row.baseline_cycles - float(row.legal_floor_cycles)
                    for row in compatible)
    gap_numer = sum(row.baseline_cycles - row.candidate_cycles for row in compatible)
    gap_closed = gap_numer / gap_denom if gap_denom > 0 else None
    if gap_closed is None:
        refusals.append("aggregate baseline-to-attainable gap is not defined")
    attainments = [row.legal_attainment for row in compatible
                   if row.legal_attainment is not None]
    worst_attainment = min(attainments) if attainments else None
    worst_regression = max((row.regression_fraction for row in rows), default=0.0)
    occupancy_complete = not any(
        "occupancy/movement/latency-hiding" in refusal for refusal in refusals)

    gates = (
        ("measured_whole_model_cycles", all(
            row.cycle_basis == "measured" and bool(row.provenance) for row in rows)),
        ("heldout_geomean_speedup", geomean is not None
         and geomean >= pol.minimum_heldout_geomean_speedup),
        ("individual_regression", worst_regression <= pol.maximum_individual_regression),
        ("aggregate_gap_closed", gap_closed is not None
         and gap_closed >= pol.minimum_aggregate_gap_closed),
        ("legal_roofline_attainment", worst_attainment is not None
         and worst_attainment >= pol.minimum_legal_attainment),
        ("whole_model_occupancy_evidence", occupancy_complete),
    )
    return WholeModelReport(
        tuple(rows), geomean, gap_closed, worst_attainment, worst_regression,
        gates, _opportunities(tuple(rows)), tuple(dict.fromkeys(refusals)))
