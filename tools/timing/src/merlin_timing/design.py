"""Check whether a reviewed Phase 1 observation plan can test variant ranking."""

from __future__ import annotations

import itertools
import statistics

from . import archive as A
from . import contracts as C
from . import validation as V


def check(plan: dict) -> dict:
    if plan.get("schema") not in (A.PLAN, A.PLAN_V2):
        raise C.ContractError("design check requires an original reviewed archive observation plan")
    template = plan.get("dataset_template")
    C.dataset(template)
    if template["records"]:
        raise C.ContractError("observation plan template must not contain attached records")
    cases = {case["case_id"]: case for case in template["cases"]}
    observations = plan.get("observations")
    if not isinstance(observations, list) or len(observations) > C.MAX_RECORDS:
        raise C.ContractError("observation roster is missing or exceeds its count bound")
    labels: dict[str, list[int]] = {key: [] for key in cases}
    missing = set()
    for row in observations:
        if not isinstance(row, dict) or row.get("case_id") not in cases:
            raise C.ContractError("observation refers to an undeclared case")
        key, label = row["case_id"], row.get("label")
        if not isinstance(label, dict):
            raise C.ContractError("observation lacks its original label")
        C.closed(
            label,
            {
                "engine",
                "engine_sha256",
                "receipt_sha256",
                "execution_sha256",
                "status",
                "numeric",
                "measured_now",
                "cycles",
            },
            "design-check label",
        )
        if (
            label["engine"] != template["domain"]["label_engine"]
            or label["engine_sha256"] != template["domain"]["label_engine_sha256"]
            or label["execution_sha256"] != C.execution_identity(cases[key])
        ):
            raise C.ContractError("design-check label differs from its selected execution or simulator")
        C.sha(label["receipt_sha256"], "design-check original measurement receipt")
        value = label.get("cycles")
        if (
            label.get("status") != "pass"
            or label.get("numeric") != "pass"
            or label.get("measured_now") is not True
            or type(value) is not int
            or value <= 0
        ):
            missing.add(key)
        else:
            labels[key].append(value)
    measured = {key: statistics.median(values) for key, values in labels.items() if values and key not in missing}
    groups: dict[str, list[dict]] = {}
    for case in cases.values():
        if case["split"] == "test":
            groups.setdefault(case["workload_sha256"], []).append(case)
    possible_pairs = sum(len(group) * (len(group) - 1) // 2 for group in groups.values())
    if possible_pairs > C.MAX_PAIR_COMPARISONS:
        raise C.ContractError("held pair comparisons exceed the explicit resource limit")
    pair_families = {}
    eligible_pairs = []
    for group in groups.values():
        for left, right in itertools.combinations(group, 2):
            if left["mechanism_family"] != right["mechanism_family"]:
                raise C.ContractError("one held workload changes mechanism family")
            a, b = left["case_id"], right["case_id"]
            if a not in measured or b not in measured:
                continue
            separation = abs(measured[a] - measured[b]) / max(measured[a], measured[b])
            if separation > V.GATES["minimum_measured_separation"]:
                family = left["mechanism_family"]
                pair_families[family] = pair_families.get(family, 0) + 1
                eligible_pairs.append(
                    {"left": a, "right": b, "mechanism_family": family, "measured_separation": separation}
                )
    calibration_groups = {
        case["source_group_sha256"]
        for case in cases.values()
        if case["split"] == "calibration" and case["case_id"] in measured
    }
    training_families = {
        case["mechanism_family"] for case in cases.values() if case["split"] == "train" and case["case_id"] in measured
    }
    held_families = {case["mechanism_family"] for case in cases.values() if case["split"] == "test"}
    checks = {
        "eligible_held_pairs": len(eligible_pairs) >= V.GATES["minimum_eligible_held_pairs"],
        "mechanism_families": len(pair_families) >= V.GATES["minimum_mechanism_families"],
        "calibration_groups": len(calibration_groups) >= 2,
        "training_family_coverage": held_families <= training_families,
        "all_held_labels_available": all(key in measured for key, case in cases.items() if case["split"] == "test"),
    }
    if plan["schema"] == A.PLAN_V2:
        checks["all_counter_reviews_available"] = all(row.get("scope_binding") is not None for row in observations)
    return {
        "schema": "merlin.spike_timing_design_check.v1",
        "archive_sha256": plan["archive_sha256"],
        "plan_sha256": C.digest(plan),
        "ready_for_native_qualification_attempt": all(checks.values()),
        "checks": checks,
        "counts": {
            "held_cases": sum(case["split"] == "test" for case in cases.values()),
            "held_workloads": len(groups),
            "possible_held_pairs": possible_pairs,
            "measured_separated_held_pairs": len(eligible_pairs),
            "families_with_eligible_pairs": len(pair_families),
            "calibration_source_groups": len(calibration_groups),
        },
        "eligible_pairs_by_family": dict(sorted(pair_families.items())),
        "eligible_pairs": eligible_pairs,
        "limitations": (
            "This checks label and experiment design only; native Spike predictions, "
            "numerical receipts, and ranking accuracy are still unmeasured."
        ),
    }
