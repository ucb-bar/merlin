"""Guidance and declared goals over complete measured search observations.

Measured opportunities, matched evaluation costs and conservative roofline goals
share this projection owner. They retain separate bounds and evidence requirements;
none issues numerical, timing, compiler promotion or final acceptance authority.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from merlin.development.backend.optimization.contracts import StageGateError

SCHEMA = "merlin.phase2.roofline_priority.v1"

CONVERGED, CONTINUE, UNDECIDED, REFUTED = "converged", "continue", "undecided", "floor_refuted"


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _target(row):
    macs, rate = row.get("declared_macs"), row.get("achievable_macs_per_cycle")
    if _positive(macs) and _positive(rate):
        target = macs / rate
        if _positive(target):
            return target, "matched empirical throughput; attainable improvement unproven", False
    roofline = row.get("roofline") or {}
    floor = roofline.get("roofline_cycles")
    if roofline.get("status") == "derived" and _positive(floor):
        return floor, "hardware lower bound; gap is an upper bound on possible savings", True
    return None, "no matched throughput target or unrefuted hardware bound", False


def measured_priorities(members, weights):
    """Use the ranking objective's exact weights and every observed repeat.

    The caller owns feedback/source/corpus validation. A missing or inconsistent
    bound cannot remove a member's measured time from the objective. Costs are
    not inferred from guest cycles or wall time of an incomplete evaluation.
    """
    if (
        not members
        or set(members) != set(weights)
        or any(not _positive(value) for value in weights.values())
        or not math.isclose(sum(weights.values()), 1.0)
    ):
        raise StageGateError("search priorities require the exact complete ranking weights")
    rows = []
    for identity, repeats in sorted(members.items()):
        if not repeats or any(not _positive(row.get("candidate_gsim_cycles")) for row in repeats):
            raise StageGateError("search priorities require every complete measured repeat")
        weight = weights[identity]
        cycles = [row["candidate_gsim_cycles"] for row in repeats]
        targets = [_target(row) for row in repeats]
        target, basis, lower_bound = targets[0]
        if any(pair != targets[0] for pair in targets):
            target, basis = None, "repeat observations disagree on their bound or throughput target"
        if target is not None and lower_bound and target > min(cycles):
            target, basis = None, "observed timing refutes the supplied hardware bound"
        rows.append(
            {
                "family": identity[0],
                "capsule": identity[1],
                "weight": weight,
                "weighted_cycles": {"lo": weight * min(cycles), "hi": weight * max(cycles)},
                "target_cycles": target,
                "potential_weighted_savings": (
                    {"lo": weight * max(0, min(cycles) - target), "hi": weight * max(0, max(cycles) - target)}
                    if target is not None
                    else None
                ),
                "basis": basis,
                "evaluation_cost_seconds": None,
                "savings_per_evaluation_second": None,
            }
        )
    priced = [row for row in rows if row["potential_weighted_savings"] is not None]
    unknown = [row for row in rows if row["potential_weighted_savings"] is None]
    return {
        "ranked": sorted(
            priced, key=lambda row: (-row["potential_weighted_savings"]["lo"], row["family"], row["capsule"])
        ),
        "unknown": unknown,
        "weighted_cycles": {key: sum(row["weighted_cycles"][key] for row in rows) for key in ("lo", "hi")},
        "unpriced_weighted_cycles": {key: sum(row["weighted_cycles"][key] for row in unknown) for key in ("lo", "hi")},
        "evaluation_cost_status": "UNKNOWN: complete matched evaluation costs have not been observed",
        "scope": "complete generated roster; potential savings are guidance, never promotion or a stop proof",
    }


def _roofline_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _floor_document(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], str]:
    capacity = ((row.get("performance") or {}).get("capacity") or {}).get("roofline") or row.get("staged_roofline")
    source = (
        "performance.capacity.roofline"
        if ((row.get("performance") or {}).get("capacity") or {}).get("roofline")
        else "staged_roofline"
    )
    if isinstance(capacity, Mapping) and capacity.get("status") in ("derived", "partial"):
        return capacity, source
    roofline = row.get("roofline") or {}
    if (
        isinstance(roofline, Mapping)
        and roofline.get("status") == "derived"
        and _roofline_positive(roofline.get("roofline_cycles"))
    ):
        return roofline, "roofline"
    if isinstance(capacity, Mapping) and capacity:
        return capacity, source
    return (roofline if isinstance(roofline, Mapping) else {}), "roofline"


def floor_of(row: Mapping[str, Any]) -> float | None:
    """A numeric lower bound for exploration, including partial staged floors.

    Presence of a numeric floor alone supplies no convergence eligibility.
    """
    document, source = _floor_document(row)
    statuses = ("derived",) if source == "roofline" else ("derived", "partial")
    floor = document.get("roofline_cycles" if source == "roofline" else "floor_cycles")
    return float(floor) if document.get("status") in statuses and _roofline_positive(floor) else None


def _floor_evidence(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    document, source = _floor_document(row)
    return {
        "repeat": index,
        "source": source,
        "status": document.get("status", "unknown"),
        **{key: deepcopy(document[key]) for key in ("unresolved", "omitted", "reason", "reasons") if key in document},
    }


def declared_target(descriptors: Sequence[Mapping[str, Any]]) -> float:
    """The one ``attainment_min`` the cohort's frozen members declare, or a refusal."""
    targets = {
        float(((d.get("performance") or {}).get("capacity") or {}).get("roofline_target", {}).get("attainment_min"))
        for d in descriptors
        if _roofline_positive(
            ((d.get("performance") or {}).get("capacity") or {}).get("roofline_target", {}).get("attainment_min")
        )
    }
    if len(targets) != 1:
        raise ValueError(f"the cohort must declare exactly one roofline attainment target, found {sorted(targets)}")
    target = targets.pop()
    if not 0 < target <= 1:
        raise ValueError("a roofline attainment target must be in (0, 1]")
    return target


def roofline_priorities(members: Mapping[Any, Sequence[Mapping[str, Any]]], *, attainment_min: float) -> dict[str, Any]:
    """Rank ``{identity: [repeat rows]}`` by attainment and decide the declared stop.

    Each repeat row carries its measured ``cycles`` and its floor (:func:`floor_of`). Repeats must
    agree on the floor: the floor is a property of the member, not of a run.
    """
    if not _roofline_positive(attainment_min) or attainment_min > 1:
        raise ValueError("attainment_min must be in (0, 1]")
    if not members:
        raise ValueError("no measured members were supplied")
    ranked, unknown, partial, refuted = [], [], [], []
    for identity, repeats in sorted(members.items(), key=lambda item: str(item[0])):
        cycles = [row.get("cycles") for row in repeats]
        if not repeats or not all(_roofline_positive(c) for c in cycles):
            raise ValueError(f"{identity}: every repeat needs a positive measured cycle count")
        floors = {floor_of(row) for row in repeats}
        evidence = [_floor_evidence(row, index) for index, row in enumerate(repeats)]
        if len(floors) != 1 or None in floors:
            unknown.append(
                {
                    "member": identity,
                    "reason": "no single derived floor across its repeats",
                    "evidence": evidence,
                }
            )
            continue
        floor = floors.pop()
        if min(cycles) < floor:
            refuted.append({"member": identity, "floor_cycles": floor, "min_cycles": min(cycles), "evidence": evidence})
            continue
        eligible = all(
            row["status"] == "derived" and not row.get("unresolved") and not row.get("omitted") for row in evidence
        )
        if not eligible:
            partial.append(
                {
                    "member": identity,
                    "floor_cycles": floor,
                    "reason": "partial, unresolved or omitted floor terms cannot establish convergence",
                    "evidence": evidence,
                }
            )
        worst, best = floor / max(cycles), floor / min(cycles)
        ranked.append(
            {
                "member": identity,
                "floor_cycles": floor,
                "attainment": {"worst": worst, "best": best},
                "within_target": worst >= attainment_min,
                "gap_cycles": max(cycles) - floor,
                "convergence_eligible": eligible,
                "floor_status": "derived" if eligible else "partial",
                "floor_evidence": evidence,
            }
        )
    ranked.sort(key=lambda row: (row["attainment"]["worst"], str(row["member"])))
    below = [row for row in ranked if not row["within_target"]]
    if refuted:
        status = REFUTED
    elif below:
        status = CONTINUE
    elif unknown or partial:
        status = UNDECIDED
    else:
        status = CONVERGED
    return {
        "schema": SCHEMA,
        "attainment_min": attainment_min,
        "status": status,
        "next": [row["member"] for row in below],
        "ranked": ranked,
        "unknown_floor": unknown,
        "partial_floor": partial,
        "floor_refuted": refuted,
        "completion_blocked": bool(unknown or partial or refuted),
        "scope": "guidance over measured rows; stops only on every member's derived floor without unresolved or omitted terms",
    }


def target_progress(incumbent, *, maximum_ratio=None, roofline_attainment=None):
    """Require every configured goal, retaining the original worst repeat ranges.

    Ratio uses the selection's exact family/member weights. Roofline attainment
    instead requires every member independently, through roofline_priority's
    existing complete-floor, unresolved-term and subfloor-refutation checks.
    """
    configured = maximum_ratio is not None or roofline_attainment is not None
    result = {
        "status": "not_configured" if not configured else "unknown",
        "maximum_ratio": maximum_ratio,
        "roofline_attainment_min": roofline_attainment,
        "source_sha256": None if incumbent is None else incumbent["source_sha256"],
        "conditions": {},
        "scope": "complete measured development roster; no whole-model or final acceptance claim",
    }
    if not configured:
        return result
    if incumbent is None:
        result["reason"] = "incumbent has no complete correct measured observation"
        return result
    conditions = result["conditions"]
    if maximum_ratio is not None:
        worst = incumbent["ratio_hi"]
        conditions["cycle_ratio"] = {"worst": worst, "reached": worst <= maximum_ratio}
    if roofline_attainment is not None:
        roofline = incumbent.get("roofline_progress")
        conditions["roofline"] = {
            "status": None if roofline is None else roofline["status"],
            "reached": bool(roofline and roofline["status"] == CONVERGED),
            "completion_blocked": not roofline or roofline["completion_blocked"],
        }
    if all(row["reached"] for row in conditions.values()):
        result["status"] = "reached"
    elif any(row.get("completion_blocked") for row in conditions.values()):
        result["status"] = "unknown"
    else:
        result["status"] = "continue"
    return result


def project_costs(priorities, observations, *, source_sha256, baseline_sha256, corpus_sha256, certificate_sha256):
    """Keep all matching repeats and separate evaluator/resource contexts."""
    from merlin.development.backend.optimization.ordinary import ordinary_fast_feedback as FAST
    from merlin.development.backend.optimization.ordinary import ordinary_measurement_selection as SELECTION
    from merlin.development.backend.optimization.search import evaluation_costs as COSTS

    def _passing(observation, identity):
        result = observation.get("result") or {}
        matches = [row for row in result.get("cells", ()) if (row.get("family"), row.get("capsule")) == identity]
        if len(matches) != 1:
            return False
        row = matches[0]
        if observation["kind"] == FAST.SELECTED_KIND:
            return all((row.get(arm) or {}).get("functional_status") == "PASS" for arm in ("baseline", "candidate"))
        return row.get("measured") is True and row.get("comparable") is True

    opportunities = {
        (row["family"], row["capsule"]): row for key in ("ranked", "unknown") for row in priorities.get(key, ())
    }
    groups, excluded = {}, []
    for observation in observations:
        cost = observation.get("evaluation_cost") or {}
        if cost.get("status") != "observed":
            continue
        context = cost.get("context") or {}
        selected = observation.get("selected_members", ())
        member = context.get("member") or {}
        identity = member.get("family"), member.get("capsule")
        if (
            cost.get("schema") != COSTS.SCHEMA
            or observation.get("state") != "completed"
            or cost.get("outcome") != "completed"
            or observation.get("kind") not in (FAST.SELECTED_KIND, SELECTION.KIND)
            or context.get("kind") != observation["kind"]
            or cost.get("job_id") != observation.get("job_id")
            or selected != [list(identity)]
            or context.get("selected_members") != selected
            or context.get("selection_scope") != "selected public members"
            or any(
                type(context.get(key)) is not str or not context[key]
                for key in ("resource_context_sha256", "closure_sha256", "environment_sha256")
            )
            or observation.get("source_sha256") != source_sha256
            or (
                context.get("candidate_sha256"),
                context.get("baseline_sha256"),
                context.get("corpus_sha256"),
                context.get("certificate_sha256"),
            )
            != (source_sha256, baseline_sha256, corpus_sha256, certificate_sha256)
            or not _positive(cost.get("wall_seconds"))
            or cost.get("boundary") != COSTS.BOUNDARY
            or cost.get("cache_state") != "UNKNOWN"
            or not _passing(observation, identity)
        ):
            excluded.append({"job_id": observation.get("job_id"), "reason": "no matched successful singleton cost"})
            continue
        key = (
            context["kind"],
            context["resource_context_sha256"],
            context["closure_sha256"],
            context["environment_sha256"],
        )
        group = groups.setdefault(key, {"context": context, "members": {}})
        rows = group["members"].setdefault(identity, [])
        if any(
            item["evaluation_cost"]["context"]["member"] != member
            or item["evaluation_cost"]["context"]["resources"] != context["resources"]
            for item in rows
        ):
            excluded.append({"job_id": observation["job_id"], "reason": "singleton cost inputs disagree"})
            continue
        rows.append(observation)
    projected = []
    for _key, group in sorted(groups.items()):
        rows = []
        for identity, repeats in sorted(group["members"].items()):
            costs = [row["evaluation_cost"]["wall_seconds"] for row in repeats]
            opportunity = opportunities.get(identity, {}).get("potential_weighted_savings")
            savings = (
                {"lo": opportunity["lo"] / max(costs), "hi": opportunity["hi"] / min(costs)}
                if opportunity is not None
                else None
            )
            if savings is not None and any(not math.isfinite(value) for value in savings.values()):
                savings = None
            rows.append(
                {
                    "family": identity[0],
                    "capsule": identity[1],
                    "jobs": [row["job_id"] for row in repeats],
                    "evaluation_cost_seconds": {"lo": min(costs), "hi": max(costs)},
                    "potential_weighted_savings": opportunity,
                    "savings_per_evaluation_second": savings,
                    "cost_scope": "one observed baseline/candidate pair action; queue wait reported separately",
                    "basis": opportunities.get(identity, {}).get("basis", "no complete measured incumbent bound"),
                }
            )
        priced = [row for row in rows if row["savings_per_evaluation_second"] is not None]
        projected.append(
            {
                "kind": group["context"]["kind"],
                "resource_context_sha256": group["context"]["resource_context_sha256"],
                "closure_sha256": group["context"]["closure_sha256"],
                "environment_sha256": group["context"]["environment_sha256"],
                "ranked": sorted(
                    priced, key=lambda row: (-row["savings_per_evaluation_second"]["lo"], row["family"], row["capsule"])
                ),
                "unknown": [row for row in rows if row["savings_per_evaluation_second"] is None],
                "missing_cost_members": [list(key) for key in sorted(opportunities) if key not in group["members"]],
            }
        )
    return {
        "status": "observed" if projected else "UNKNOWN",
        "groups": projected,
        "excluded": excluded,
        "source_sha256": source_sha256,
        "scope": (
            "conditional incumbent opportunity per observed pair-action second; no future duration or gain prediction"
        ),
        "missing_cost_members": [
            list(key) for key in sorted(opportunities) if not any(key in g["members"] for g in groups.values())
        ],
        "authority": "diagnostic guidance only; contender order, promotion, plateau and final gates unchanged",
        "cache_state": "UNKNOWN",
        "cpu_cost": "UNKNOWN",
    }
