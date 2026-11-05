"""Preregistered independent component screening and conservative work ordering."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from merlin.common.jsonio import canonical_sha256
from merlin.perf.fast_estimate_validation import Observation, Predictor, cross_validate
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


@dataclass(frozen=True)
class ComponentScreenPolicy:
    minimum_rank_rate: float = 0.95
    minimum_decided: int = 100
    minimum_slice_decided: int = 20
    minimum_slices: int = 3
    minimum_predictions: int = 20
    maximum_relative_error: float = 0.10
    minimum_interval_coverage: float = 0.95

    def __post_init__(self):
        if not 0.5 < self.minimum_rank_rate <= 1 or not 0 < self.minimum_interval_coverage <= 1:
            raise ValueError("screening requires positive coverage and agreement above chance")
        if not math.isfinite(self.maximum_relative_error) or self.maximum_relative_error < 0:
            raise ValueError("screening maximum error must be finite and nonnegative")
        if any(
            type(x) is not int or x < 1
            for x in (self.minimum_decided, self.minimum_slice_decided, self.minimum_slices, self.minimum_predictions)
        ):
            raise ValueError("screening requires positive independent evidence counts")

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @property
    def sha256(self):
        return canonical_sha256(self.to_dict())


def qualify_component_screen(
    observations: Sequence[Observation],
    fit: Callable[[Sequence[Observation]], Predictor],
    *,
    calibration_sha256: str,
    policy: ComponentScreenPolicy = ComponentScreenPolicy(),
) -> dict[str, Any]:
    """Held workload groups authorize screening only after both existing gates."""
    if not observations:
        raise ValueError("component qualification requires held observations")
    report = cross_validate(
        observations,
        fit,
        **{key: value for key, value in policy.to_dict().items() if key != "minimum_interval_coverage"},
    )
    predictions = [row for row in report["predictions"] if row["prediction"]["resolved"]]
    contains = sum(row["contains_observation"] for row in predictions)
    coverage = contains / len(predictions) if predictions else None
    reasons = list(report["reasons"])
    if coverage is None or coverage < policy.minimum_interval_coverage:
        reasons.append("held interval coverage is below preregistered minimum")
    return {
        **report,
        "schema": "component_cycle_screen_validation_v1",
        "exposable": not reasons,
        "reasons": reasons,
        "calibration_sha256": calibration_sha256,
        "policy_sha256": policy.sha256,
        "policy": policy.to_dict(),
        "interval_coverage": {"n": len(predictions), "contains": contains, "rate": coverage},
        "promotion": "SCREENING_ONLY",
    }


@dataclass(frozen=True)
class ComponentOpportunity:
    id: str
    family: str
    regime: str
    baseline: CycleInterval
    candidate: CycleInterval
    evaluation_seconds: float | None
    legal: bool | None
    correct: bool | None


def rank_component_opportunities(
    opportunities: Sequence[ComponentOpportunity],
    *,
    work_shares: Mapping[str, float],
) -> dict[str, Any]:
    """Conservative saved complete cost per evaluation second with family diversity.

    Work shares must come from this independent corpus. Equal family shares are an
    explicit caller choice. No validation-model proportions are accepted or inferred.
    """
    if len({row.id for row in opportunities}) != len(opportunities):
        raise ValueError("component opportunity identities must be distinct")
    if (
        set(work_shares) != {row.id for row in opportunities}
        or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in work_shares.values())
        or not math.isclose(sum(work_shares.values()), 1.0)
    ):
        raise ValueError("independent corpus shares must cover the exact proposal roster and sum to one")
    rows = []
    for proposal in opportunities:
        status, saved, priority = "UNRESOLVED", None, None
        if any(value is not None and type(value) is not bool for value in (proposal.legal, proposal.correct)):
            raise ValueError("component opportunity gates must be PASS, FAIL or UNKNOWN as bool or None")
        if proposal.legal is False or proposal.correct is False:
            status = "REFUSAL"
        elif (
            proposal.legal is True
            and proposal.correct is True
            and proposal.baseline.resolved
            and proposal.candidate.resolved
        ):
            saved = proposal.baseline.lo - proposal.candidate.hi
            if saved > 0:
                status = "OPPORTUNITY"
            elif proposal.candidate.lo > proposal.baseline.hi:
                status = "REGRESSION"
            else:
                status = "TIE_OR_OVERLAP"
            cost = proposal.evaluation_seconds
            if cost is not None and not isinstance(cost, bool) and math.isfinite(cost) and cost > 0:
                priority = max(0.0, saved) * work_shares[proposal.id] / cost
        rows.append(
            {
                "id": proposal.id,
                "family": proposal.family,
                "regime": proposal.regime,
                "status": status,
                "conservative_saved_cycles": saved,
                "priority": priority,
                "evaluation_seconds": proposal.evaluation_seconds,
                "legal": proposal.legal,
                "correct": proposal.correct,
            }
        )
    decided = sorted(
        (r for r in rows if r["status"] == "OPPORTUNITY" and r["priority"] is not None),
        key=lambda row: (-row["priority"], row["id"]),
    )
    representatives, remaining, seen = [], [], set()
    for row in decided:
        key = row["family"], row["regime"]
        if key not in seen:
            representatives.append(row["id"])
            seen.add(key)
        else:
            remaining.append(row["id"])
    return {
        "schema": "component_opportunity_order_v1",
        "order": representatives + remaining,
        "evidence": rows,
        "work_shares_sha256": canonical_sha256(dict(work_shares)),
        "promotion": "SCREENING_ONLY",
    }


def cheapest_distinguishing_experiment(experiments: Sequence[Mapping[str, Any]], *, unresolved: Sequence[str]):
    """Choose only experiments that explicitly distinguish a remaining question."""
    questions = set(unresolved)
    eligible = []
    for row in experiments:
        cost = row.get("evaluation_seconds")
        if (
            isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost > 0
            and questions.intersection(row.get("distinguishes", ()))
        ):
            eligible.append(row)
    return min(eligible, key=lambda row: (row["evaluation_seconds"], row["id"])) if eligible else None


def validate_component_screen_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Replay held predictions, ranking and coverage before admitting a screen."""
    from merlin.common.digest import is_sha256
    from merlin.perf import rank_validation as rank

    policy = ComponentScreenPolicy()
    if (
        report.get("schema") != "component_cycle_screen_validation_v1"
        or report.get("promotion") != "SCREENING_ONLY"
        or report.get("policy") != policy.to_dict()
        or report.get("policy_sha256") != policy.sha256
        or not is_sha256(report.get("calibration_sha256"))
        or not is_sha256(report.get("domain_sha256"))
    ):
        raise ValueError("component screen report differs from preregistered policy")
    records = report.get("predictions")
    if not isinstance(records, list) or not records:
        raise ValueError("component screen report omits its held predictions")
    seen, intervals, programs, errors, workload_groups = set(), {}, [], [], {}
    for row in records:
        ident = row["id"]
        if ident in seen or ident != canonical_sha256([row["program"], row["workload"]]):
            raise ValueError("component held prediction identity is duplicated or invalid")
        seen.add(ident)
        if not is_sha256(row["program"]) or not is_sha256(row["workload"]):
            raise ValueError("component screen requires exact executable and workload identities")
        workload_groups.setdefault(row["workload"], set()).add(row["group"])
        measured = row["measured_cycles"]
        if isinstance(measured, bool) or not math.isfinite(measured) or measured <= 0:
            raise ValueError("component held measurement is invalid")
        prediction = row["prediction"]
        interval = CycleInterval(
            prediction["lo"], prediction["hi"], tuple(prediction["provenance"]), tuple(prediction["missing"])
        )
        if interval.to_dict() != prediction:
            raise ValueError("component held prediction interval is inconsistent")
        programs.append(rank.Program(row["workload"], ident, measured, row["group"]))
        if interval.resolved:
            if not math.isfinite(interval.lo) or not math.isfinite(interval.hi):
                raise ValueError("component screen contains a nonfinite prediction")
            error = abs((interval.lo + interval.hi) / 2 - measured) / measured
            if row.get("relative_error") != error or row.get("contains_observation") is not (
                interval.lo <= measured <= interval.hi
            ):
                raise ValueError("component held error/coverage claim differs from prediction")
            intervals[ident] = (interval.lo, interval.hi)
            errors.append(error)
        elif "relative_error" in row or "contains_observation" in row:
            raise ValueError("unresolved held prediction cannot claim error or coverage")
    if any(len(groups) != 1 for groups in workload_groups.values()):
        raise ValueError("all variants of a workload must share one held-out group")
    for row in records:
        expected = [other["program"] for other in records if other["group"] != row["group"]]
        if not isinstance(row["training_programs"], list) or Counter(row["training_programs"]) != Counter(expected):
            raise ValueError("component prediction training roster leaks or omits a held workload group")
    overall = rank.interval_agreement(rank.ordered_pairs(programs), intervals)
    slices = {
        group: rank.interval_agreement(rank.ordered_pairs([p for p in programs if p.group == group]), intervals)
        for group in sorted({p.group for p in programs})
    }
    replay_rank = rank.verdict(
        overall,
        slices,
        minimum_rate=policy.minimum_rank_rate,
        minimum_decided=policy.minimum_decided,
        minimum_slice_decided=policy.minimum_slice_decided,
        minimum_slices=policy.minimum_slices,
    )
    if replay_rank != report.get("ranking"):
        raise ValueError("component held ranking claim differs from exact pair ordering")
    n = len(intervals)
    contains = sum(row["contains_observation"] for row in records if row["id"] in intervals)
    coverage = {"n": n, "contains": contains, "rate": contains / n if n else None}
    if coverage != report.get("interval_coverage") or report.get("absolute_error", {}).get("n") != n:
        raise ValueError("component held aggregate coverage/count differs from predictions")
    if report.get("absolute_error", {}).get("maximum_relative") != (max(errors) if errors else None):
        raise ValueError("component held maximum error differs from predictions")
    qualified = (
        replay_rank["exposable"]
        and n >= policy.minimum_predictions
        and max(errors, default=math.inf) <= policy.maximum_relative_error
        and coverage["rate"] is not None
        and coverage["rate"] >= policy.minimum_interval_coverage
    )
    if report.get("exposable") is not qualified or (qualified and report.get("reasons") != []):
        raise ValueError("component screen qualification differs from held evidence")
    return dict(report)
