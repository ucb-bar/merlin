"""Diagnostic comparisons and missing original-measurement plans, without launchers."""

from __future__ import annotations

import itertools

from . import contracts as C
from . import fitting as F
from . import validation as V


def compare(screen: dict, cases: dict, *, qualification: dict | None = None) -> dict:
    F.model(screen)
    C.closed(cases, {"schema", "profile", "domain", "cases"}, "prediction cases")
    if cases["schema"] != C.CASES or cases["profile"] != screen["profile"]:
        raise C.ContractError("prediction cases differ from the screen profile")
    C.domain(cases["domain"])
    if not isinstance(cases["cases"], list) or len(cases["cases"]) > C.MAX_CASES:
        raise C.ContractError("prediction case count exceeds its limit")
    authorized = False
    if qualification is not None:
        # A supplied receipt is an exact content-addressed result, not a toggle.
        C.closed(qualification, {"dataset", "report"}, "qualification receipt")
        expected = V.validate(qualification["dataset"], screen)
        if expected != qualification["report"]:
            raise C.ContractError("qualification differs from its original held-out validation")
        authorized = expected["qualified"] is True
    rows, seen = [], set()
    for case in cases["cases"]:
        C.closed(case, {"case_id", "workload_sha256", "features"}, "prediction case")
        C.text(case["case_id"], "prediction case ID")
        C.sha(case["workload_sha256"], "prediction workload")
        if case["case_id"] in seen:
            raise C.ContractError("prediction case IDs must be unique")
        seen.add(case["case_id"])
        prediction, reason = F.predict(screen, case["features"], domain_sha256=C.digest(cases["domain"]))
        residuals = screen["residual_interval"]["relative_residual_quantiles"]
        interval = (
            [max(0.0, prediction * (1 + residuals[0])), max(0.0, prediction * (1 + residuals[1]))]
            if prediction is not None and residuals is not None
            else None
        )
        rows.append(
            {
                "case_id": case["case_id"],
                "workload_sha256": case["workload_sha256"],
                "estimated_cycles": prediction,
                "interval": interval if authorized else None,
                "diagnostic_interval": interval,
                "screen_terms": {
                    "fixed_cycles": screen["fixed_cycles"],
                    "features": [
                        {
                            "name": name,
                            "observed_count": case["features"][name],
                            "coefficient": screen["coefficients"][name],
                            "estimated_cycles": case["features"][name] * screen["coefficients"][name],
                        }
                        for name in screen["feature_names"]
                    ],
                    "interpretation": "correlated screening terms; not physical latency, stall or overlap attribution",
                }
                if prediction is not None
                else None,
                "status": "qualified_screen" if authorized and interval else "diagnostic",
                "reason": reason or (None if authorized else "no qualified held-out screen receipt"),
            }
        )
    groups = {}
    for row in rows:
        groups.setdefault(row["workload_sha256"], []).append(row)
    if sum(len(group) * (len(group) - 1) // 2 for group in groups.values()) > C.MAX_PAIR_COMPARISONS:
        raise C.ContractError("prediction pair comparisons exceed the explicit 20000-pair resource limit")
    rankings = []
    pairs = (pair for group in groups.values() for pair in itertools.combinations(group, 2))
    for left, right in pairs:
        a, b = left["interval"], right["interval"]
        winner = left["case_id"] if a and b and a[1] < b[0] else right["case_id"] if a and b and b[1] < a[0] else None
        rankings.append(
            {
                "left": left["case_id"],
                "right": right["case_id"],
                "predicted_faster": winner,
                "status": "separated_empirical_intervals" if winner else "unknown",
            }
        )
    return {
        "schema": "merlin.spike_timing_comparison.v1",
        "profile": screen["profile"],
        "prediction_kind": "empirical_screen",
        "event_engine_qualified": False,
        "public_feedback_eligible": authorized and screen["profile"] == "phase1-public",
        "model_sha256": C.digest(screen),
        "domain_sha256": C.digest(cases["domain"]),
        "cases": rows,
        "comparisons": rankings,
        "limitations": ["empirical screen only; intervals do not establish event-engine timing or physical bounds"],
    }


def compare_native(cases: dict, *, qualification: dict | None = None) -> dict:
    """Rank original native predictions only within their independently checked domain."""
    C.native_cases(cases)
    authorized, ranges, families, residuals = False, {}, set(), None
    validation = None
    if qualification is not None:
        C.closed(qualification, {"dataset", "report"}, "native qualification receipt")
        dataset = qualification["dataset"]
        validation = V.validate(dataset, prediction_kind="event_timing")
        if validation != qualification["report"]:
            raise C.ContractError("native qualification differs from its original held-out validation")
        if (
            dataset["profile"] != cases["profile"]
            or dataset["domain"] != cases["domain"]
            or dataset["feature_names"] != cases["feature_names"]
        ):
            raise C.ContractError("native cases differ from the qualification profile/domain/feature schema")
        authorized = validation["event_engine_qualified"] is True
        originals = C.grouped_records(dataset)
        training = [
            C.case_point(case, originals.get(case["case_id"], []), dataset["feature_names"])[0]
            for case in dataset["cases"]
            if case["split"] == "train"
        ]
        training = [point for point in training if point is not None]
        if training:
            ranges = {
                name: [
                    min(point["features"][name] for point in training),
                    max(point["features"][name] for point in training),
                ]
                for name in cases["feature_names"]
            }
            families = {point["case"]["mechanism_family"] for point in training}
        residuals = validation["residual_interval"]["relative_residual_quantiles"]
    rows, by_workload = [], {}
    for row in cases["cases"]:
        case = row["case"]
        prediction, reason = V.native_prediction(row, case, cases)
        numeric = row["numeric"]
        if reason is None and (numeric is None or numeric["status"] != "pass" or numeric["measured_now"] is not True):
            reason = "no independent current passing numerical receipt for this execution"
        if reason is None and qualification is None:
            reason = "no qualified held-out native event receipt"
        if reason is None and (
            not ranges
            or case["mechanism_family"] not in families
            or any(
                row["features"][name] is None or not lo <= row["features"][name] <= hi
                for name, (lo, hi) in ranges.items()
            )
        ):
            reason = "native observation is outside the declared training feature/family domain"
        interval = (
            [max(0.0, prediction * (1 + residuals[0])), max(0.0, prediction * (1 + residuals[1]))]
            if prediction is not None and reason is None and residuals is not None
            else None
        )
        if reason is None and not authorized:
            reason = "native held-out qualification gates were not satisfied"
        eligible = authorized and reason is None and interval is not None
        result = {
            "case_id": case["case_id"],
            "workload_sha256": case["workload_sha256"],
            "estimated_cycles": prediction,
            "interval": interval if eligible else None,
            "diagnostic_interval": interval,
            "status": "qualified_native" if eligible else "diagnostic",
            "public_feedback_eligible": eligible and cases["profile"] == "phase1-public",
            "reason": reason,
        }
        rows.append(result)
        by_workload.setdefault(case["workload_sha256"], []).append(result)
    if sum(len(members) * (len(members) - 1) // 2 for members in by_workload.values()) > C.MAX_PAIR_COMPARISONS:
        raise C.ContractError("native prediction pair comparisons exceed the explicit 20000-pair resource limit")
    rankings = []
    for members in by_workload.values():
        for left, right in itertools.combinations(members, 2):
            a, b = left["interval"], right["interval"]
            winner = (
                left["case_id"] if a and b and a[1] < b[0] else right["case_id"] if a and b and b[1] < a[0] else None
            )
            rankings.append(
                {
                    "left": left["case_id"],
                    "right": right["case_id"],
                    "predicted_faster": winner,
                    "status": "separated_empirical_intervals" if winner else "unknown",
                }
            )
    return {
        "schema": "merlin.spike_timing_comparison.v1",
        "profile": cases["profile"],
        "prediction_kind": "event_timing",
        "event_engine_qualified": authorized,
        "public_feedback_eligible": any(row["public_feedback_eligible"] for row in rows),
        "model_sha256": cases["domain"]["target_model_sha256"],
        "domain_sha256": C.digest(cases["domain"]),
        "qualification_sha256": C.digest(validation) if validation is not None else None,
        "cases": rows,
        "comparisons": rankings,
        "limitations": [
            "empirical residual intervals are not physical bounds or runtime guarantees",
            "source and numerical receipt admission must independently establish original provenance",
            "historical private inputs never supply public feedback",
        ],
    }


def probe_plan(document: dict, screen: dict | None = None) -> dict:
    C.dataset(document)
    if screen is not None:
        F.model(screen)
        if screen["profile"] != document["profile"] or screen["domain_sha256"] != C.digest(document["domain"]):
            raise C.ContractError("probe screen differs from dataset profile/domain")
    records = C.grouped_records(document)
    needs = []
    for case in document["cases"]:
        point, reason = C.case_point(case, records.get(case["case_id"], []), document["feature_names"])
        missing = []
        originals = records.get(case["case_id"], [])
        if not originals or any(
            row["feature_provenance"]["complete"] is not True
            or any(value is None for value in row["features"].values())
            for row in originals
        ):
            missing.append("original_feature_report")
        if originals and any(row["features"] != originals[0]["features"] for row in originals):
            missing.append("original_feature_report")
        if not originals or any(
            row["label"] is None
            or row["label"]["status"] != "pass"
            or row["label"]["numeric"] != "pass"
            or row["label"]["measured_now"] is not True
            or row["label"]["cycles"] is None
            for row in originals
        ):
            missing.append("fresh_passing_cycle_label")
        if point is not None and screen is not None:
            _, why = F.predict(screen, point["features"], domain_sha256=C.digest(document["domain"]))
            if why:
                reason = why
                missing.append("independent_domain_calibration")
        if not records.get(case["case_id"]) or any(row["event_report"] is None for row in records[case["case_id"]]):
            missing.append("original_native_event_report")
        if missing:
            needs.append(
                {
                    "case_id": case["case_id"],
                    "split": case["split"],
                    "source_group_sha256": case["source_group_sha256"],
                    "workload_sha256": case["workload_sha256"],
                    "mechanism_family": case["mechanism_family"],
                    **{key: case[key] for key in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")},
                    "required": sorted(set(missing)),
                    "reason": reason or "native timing has no original report",
                }
            )
    report = V.validate(document, screen) if screen is not None else None
    return {
        "schema": "merlin.spike_timing_probe_plan.v1",
        "profile": document["profile"],
        "dataset_sha256": C.digest(document),
        "domain_sha256": C.digest(document["domain"]),
        "launch_authority": False,
        "probes": needs,
        "gates": dict(V.GATES),
        "unmet_gates": sorted(name for name, passed in report["checks"].items() if not passed) if report else None,
        "limitations": ["plans request original evidence only; no generated measurements or simulation launcher"],
    }
