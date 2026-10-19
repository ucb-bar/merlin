"""Held source-group validation and explicit separation of screens and native engines."""

from __future__ import annotations

import itertools
import statistics

from . import contracts as C
from . import fitting as F
from . import observations as O

GATES = {
    "minimum_rank_agreement": 0.90,
    "minimum_measured_separation": 0.05,
    "maximum_median_absolute_relative_error": 0.15,
    "maximum_p95_absolute_relative_error": 0.30,
    "minimum_eligible_held_pairs": 20,
    "minimum_mechanism_families": 3,
    "minimum_declared_case_prediction_coverage": 0.80,
}


def native_prediction(record: dict, case: dict, document: dict) -> tuple[float | None, str | None]:
    """Read an original complete native report; feature regressions never enter here."""
    envelope = record["event_report"]
    if envelope is None:
        return None, "no original native event report"
    C.closed(envelope, {"sha256", "report"}, "event report envelope")
    report = envelope["report"]
    if not isinstance(report, dict) or C.digest(report) != C.sha(envelope["sha256"], "event report digest"):
        raise C.ContractError("event report digest differs from its original report")
    required = {
        "schema",
        "profile",
        "device_sha256",
        "domain_sha256",
        "model_sha256",
        "feature_engine_sha256",
        "elf_sha256",
        "input_sha256",
        "scope_sha256",
        "functional_status",
        "timing_complete",
        "estimated_cycles",
        "feature_counts",
        "unresolved",
        "cycle_breakdown",
    }
    if not required <= set(report):
        raise C.ContractError("native event report omits required identity/timing fields")
    # Descriptive extension fields remain digest-bound and are never interpreted
    # as alternate prices, predictions, qualification or identity overrides.
    if report["schema"] not in (O.REPORT_V1, O.REPORT_V2):
        raise C.ContractError("unsupported native event report")
    expected = {key: case[key] for key in ("device_sha256", "elf_sha256", "input_sha256", "scope_sha256")}
    expected.update(
        profile=document["profile"],
        domain_sha256=C.digest(document["domain"]),
        model_sha256=document["domain"]["target_model_sha256"],
        feature_engine_sha256=document["domain"]["feature_engine_sha256"],
    )
    if any(report[key] != value for key, value in expected.items()):
        raise C.ContractError("native report engine/model/profile/source/domain identity differs")
    if "domain" in report and report["domain"] != document["domain"]:
        raise C.ContractError("native report domain content differs from its pinned identity")
    if report["functional_status"] == "PASS" and "raw_functional_exit_code" in report:
        if type(report["raw_functional_exit_code"]) is not int or report["raw_functional_exit_code"] != 0:
            raise C.ContractError("native functional status contradicts its original exit code")
    if "execution_completed" in report:
        if type(report["execution_completed"]) is not bool:
            raise C.ContractError("native execution completion must be an explicit boolean")
        if report["functional_status"] == "PASS" and report["execution_completed"] is not True:
            raise C.ContractError("native functional status contradicts its incomplete execution")
    observed_features = report["feature_counts"]
    if not isinstance(observed_features, dict) or not set(document["feature_names"]) <= set(observed_features):
        raise C.ContractError("native report omits selected feature observations")
    selected_features = {name: observed_features[name] for name in document["feature_names"]}
    C.features(selected_features, document["feature_names"])
    if selected_features != record["features"]:
        raise C.ContractError("native selected features differ from this original feature observation")
    if report["schema"] == O.REPORT_V2:
        complete = O.feature_complete(report)
        if record["feature_provenance"]["complete"] is not complete:
            raise C.ContractError("native feature provenance contradicts its count observation")
        if not complete:
            return None, "native feature observation was incomplete"
    if report["functional_status"] != "PASS" or report["timing_complete"] is not True or report["unresolved"]:
        return None, "native event execution was incomplete or functionally unverified"
    if report["estimated_cycles"] is None:
        return None, "native event cycles are unknown"
    return C.number(report["estimated_cycles"], "native estimated cycles", positive=True), None


def validate(document: dict, screen: dict | None = None, *, prediction_kind: str = "empirical_screen") -> dict:
    C.dataset(document)
    if prediction_kind not in {"empirical_screen", "event_timing"}:
        raise C.ContractError("select empirical_screen or event_timing explicitly")
    if prediction_kind == "empirical_screen":
        if screen is None:
            raise C.ContractError("empirical validation requires a fitted screen")
        F.model(screen)
        if (
            screen["dataset_sha256"] != C.digest(document)
            or screen["profile"] != document["profile"]
            or screen["domain_sha256"] != C.digest(document["domain"])
        ):
            raise C.ContractError("screen is not bound to this exact dataset/profile/domain")
        # Refit from declared training sources. A hand-edited coefficient or range
        # cannot acquire a held-out qualification merely by retaining fit metadata.
        rebuilt = F.fit(
            document,
            include_fixed=screen["solver"]["include_fixed"],
            maximum_condition=screen["solver"]["maximum_condition"],
            training_intake=screen.get("training_intake", {}).get("policy", "all_declared"),
            loss=screen.get("fit_loss", "absolute_squared"),
        )
        if rebuilt != screen:
            raise C.ContractError("screen differs from its reproducible original training fit")
    if prediction_kind == "event_timing" and screen is not None:
        raise C.ContractError("native event timing cannot consume an empirical screen as its authority")
    records = C.grouped_records(document)
    cases = {case["case_id"]: case for case in document["cases"]}
    predictions, errors, refusals, calibration = {}, [], [], []
    held_ids = [key for key, case in cases.items() if case["split"] == "test"]
    training = [
        C.case_point(case, records.get(key, []), document["feature_names"])[0]
        for key, case in cases.items()
        if case["split"] == "train"
    ]
    training = [point for point in training if point is not None]
    ranges = (
        {
            name: (
                min(point["features"][name] for point in training),
                max(point["features"][name] for point in training),
            )
            for name in document["feature_names"]
        }
        if training
        else {}
    )
    families = {point["case"]["mechanism_family"] for point in training}
    for key, case in cases.items():
        if case["split"] == "train":
            continue
        point, reason = C.case_point(case, records.get(key, []), document["feature_names"])
        predicted = None
        if point is not None:
            if prediction_kind == "empirical_screen":
                predicted, reason = F.predict(screen, point["features"], domain_sha256=C.digest(document["domain"]))
            elif (
                not ranges
                or case["mechanism_family"] not in families
                or any(not lo <= point["features"][name] <= hi for name, (lo, hi) in ranges.items())
            ):
                reason = "native observation is outside the declared training feature/family domain"
            else:
                original = [native_prediction(row, case, document) for row in point["records"]]
                if any(value is None for value, _ in original):
                    reason = next(why for value, why in original if value is None)
                elif len({value for value, _ in original}) != 1:
                    reason = "repeat native timing predictions disagree"
                else:
                    predicted = original[0][0]
        if predicted is None:
            refusals.append({"case_id": key, "split": case["split"], "reason": reason})
        elif case["split"] == "calibration":
            calibration.append((key, case["source_group_sha256"], predicted, point["cycles"]))
        else:
            predictions[key] = {"predicted_cycles": predicted, "measured_cycles": point["cycles"]}
            errors.append(abs(predicted - point["cycles"]) / point["cycles"])
    residuals = F.residual_interval(calibration)
    pairs, by_family = [], {}
    by_workload = {}
    for key in held_ids:
        by_workload.setdefault(cases[key]["workload_sha256"], []).append(key)
    if sum(len(members) * (len(members) - 1) // 2 for members in by_workload.values()) > C.MAX_PAIR_COMPARISONS:
        raise C.ContractError("held pair comparisons exceed the explicit 20000-pair resource limit")
    for members in by_workload.values():
        for left, right in itertools.combinations(members, 2):
            if left not in predictions or right not in predictions:
                continue
            a, b = predictions[left], predictions[right]
            separation = abs(a["measured_cycles"] - b["measured_cycles"]) / max(
                a["measured_cycles"], b["measured_cycles"]
            )
            if separation <= GATES["minimum_measured_separation"]:
                continue
            correct = (
                (a["predicted_cycles"] - b["predicted_cycles"]) * (a["measured_cycles"] - b["measured_cycles"])
            ) > 0
            family = cases[left]["mechanism_family"]
            if family != cases[right]["mechanism_family"]:
                raise C.ContractError("one workload cannot change its mechanism family")
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "mechanism_family": family,
                    "measured_separation": separation,
                    "ranking_correct": correct,
                }
            )
            by_family.setdefault(family, []).append(correct)
    agreement = sum(pair["ranking_correct"] for pair in pairs) / len(pairs) if pairs else None
    coverage = len(predictions) / len(held_ids) if held_ids else 0.0
    median = statistics.median(errors) if errors else None
    p95 = F.quantile(errors, 0.95) if errors else None
    checks = {
        "rank_agreement": agreement is not None and agreement >= GATES["minimum_rank_agreement"],
        "median_error": median is not None and median <= GATES["maximum_median_absolute_relative_error"],
        "p95_error": p95 is not None and p95 <= GATES["maximum_p95_absolute_relative_error"],
        "eligible_pairs": len(pairs) >= GATES["minimum_eligible_held_pairs"],
        "mechanism_families": len(by_family) >= GATES["minimum_mechanism_families"],
        "declared_case_coverage": coverage >= GATES["minimum_declared_case_prediction_coverage"],
        "calibration_intervals": residuals["status"] == "empirical",
    }
    qualified = all(checks.values())
    return {
        "schema": C.REPORT,
        "prediction_kind": prediction_kind,
        "profile": document["profile"],
        "public_feedback_eligible": qualified and document["profile"] == "phase1-public",
        "qualified": qualified,
        "status": "qualified" if qualified else "diagnostic",
        "qualification_scope": "empirical_screen_only"
        if prediction_kind == "empirical_screen"
        else "native_event_timing",
        "event_engine_qualified": qualified and prediction_kind == "event_timing",
        "model_sha256": C.digest(screen)
        if prediction_kind == "empirical_screen"
        else document["domain"]["target_model_sha256"],
        "dataset_sha256": C.digest(document),
        "domain_sha256": C.digest(document["domain"]),
        "gates": dict(GATES),
        "checks": checks,
        "held_source_group_sha256s": sorted({cases[key]["source_group_sha256"] for key in held_ids}),
        "metrics": {
            "rank_agreement": agreement,
            "median_absolute_relative_error": median,
            "p95_absolute_relative_error": p95,
            "eligible_held_pairs": len(pairs),
            "mechanism_families": len(by_family),
            "declared_held_cases": len(held_ids),
            "predicted_held_cases": len(predictions),
            "prediction_coverage": coverage,
        },
        "families": {
            name: {"eligible_pairs": len(values), "rank_agreement": sum(values) / len(values)}
            for name, values in sorted(by_family.items())
        },
        "residual_interval": residuals,
        "predictions": predictions,
        "pairs": pairs,
        "refusals": refusals,
        "limitations": [
            "empirical residual intervals are not physical bounds or runtime guarantees",
            "synthetic fixtures do not qualify an actual engine or device",
            "statistical checks consume caller-supplied records; source/measurement admission remains separate",
        ],
    }
