"""Conditional search opportunities divided by matched observed pair-action cost.

This pure projection prices no unseen compiler or parallel sweep. The normal
coordinator supplies reopened observations from its original live pool.
"""

from __future__ import annotations

import math

from . import evaluation_costs as COSTS
from . import ordinary_fast_feedback as FAST
from . import ordinary_measurement_selection as SELECTION


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _passing(observation, identity):
    result = observation.get("result") or {}
    matches = [row for row in result.get("cells", ()) if (row.get("family"), row.get("capsule")) == identity]
    if len(matches) != 1:
        return False
    row = matches[0]
    if observation["kind"] == FAST.SELECTED_KIND:
        return all((row.get(arm) or {}).get("functional_status") == "PASS" for arm in ("baseline", "candidate"))
    return row.get("measured") is True and row.get("comparable") is True


def project(priorities, observations, *, source_sha256, baseline_sha256, corpus_sha256, certificate_sha256):
    """Keep all matching repeats and separate evaluator/resource contexts."""
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
