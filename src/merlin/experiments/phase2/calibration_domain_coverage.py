"""Exact public-source coverage of a frozen native calibration domain.

Pure reporting only. Source-profile support never grants execution, current
prediction, capacity extrapolation, timing semantics or promotion authority.
Original intake/owners must supply the catalog and dataset; no labels are fitted.
"""

from __future__ import annotations

from collections import Counter

from merlin_timing import contracts as C
from merlin_timing import validation as V

SCHEMA = "merlin.phase2.calibration_domain_coverage.v1"


def _original_record(row, observed, case, dataset):
    label_available = (
        row.get("originally_measured") is True
        and row.get("original_numerical_status") == "pass"
        and type(row.get("cycles")) is int
        and row["cycles"] > 0
    )
    reason = None
    if not label_available:
        reason = "original measured numerical label unavailable"
    elif observed is None:
        reason = "original native observation unavailable"
    elif row.get("scope_binding") is None:
        reason = "original per-record counter review unavailable"
    elif row.get("native_observation_available") is not True:
        reason = "catalog has no original native observation"
    elif observed["feature_provenance"]["complete"] is not True:
        reason = "original feature observation incomplete"
    else:
        scope = row["scope_binding"]
        if any(scope.get(name) != case[name] for name in ("elf_sha256", "input_sha256", "scope_sha256")):
            raise ValueError("coverage changed original per-record counter scope")
        label = observed["label"]
        if (
            observed["case_id"] != case["case_id"]
            or label is None
            or label["receipt_sha256"] != row["label_receipt_sha256"]
            or label["cycles"] != row["cycles"]
            or label["engine"] != row["engine"]
            or label["engine_sha256"] != row["engine_sha256"]
            or label["status"] != "pass"
            or label["numeric"] != "pass"
            or label["measured_now"] is not True
        ):
            raise ValueError("coverage changed the original per-record cycle label")
        prediction, reason = V.native_prediction(observed, case, dataset)
        if prediction is None and reason is None:
            raise ValueError("missing native prediction needs an explicit refusal")
    return {
        "record_id": row["record_id"],
        "label_receipt_sha256": row["label_receipt_sha256"],
        "label_available": label_available,
        "counter_review_available": row.get("scope_binding") is not None,
        "native_available": reason is None,
        "reason": reason,
    }


def assess(catalog, dataset, requirements, *, calibration_plan=None):
    """Join every required exact profile to original source-grouped records.

    Each requirement has a unique ``identity`` and ``profile`` (an independently
    checked complete semantic profile or None). Other fields are retained as
    descriptive provenance. A shape-only or family-only match grants nothing.
    ``calibration_plan`` is passed to the existing capacity validator with no
    observations: source geometry alone cannot identify its measured schedules.
    """
    C.dataset(dataset)
    if (
        catalog.get("schema") != "merlin.phase2.retained_archive_feedback.v1"
        or catalog.get("domain") != dataset["domain"]
        or [entry["case"] for entry in catalog["cases"]] != dataset["cases"]
    ):
        raise ValueError("coverage needs the exact original catalog/domain/case roster")
    observed = {row["record_id"]: row for row in dataset["records"]}
    declared, cases = set(), []
    grouped = C.grouped_records(dataset)
    for entry in catalog["cases"]:
        case = entry["case"]
        if case["source"]["visibility"] != "public":
            raise ValueError("domain coverage cannot consume private source evidence")
        rows = []
        for row in entry["records"]:
            key = row["record_id"]
            if key in declared:
                raise ValueError("original record occurs more than once")
            declared.add(key)
            rows.append(_original_record(row, observed.get(key), case, dataset))
        point, reason = C.case_point(case, grouped.get(case["case_id"], []), dataset["feature_names"])
        usable = bool(rows) and all(row["native_available"] for row in rows) and point is not None
        cases.append(
            {
                "case": case,
                "semantic_profile": catalog["sources"].get(case["source"]["source_sha256"], {}).get("semantic_profile"),
                "records": rows,
                "native_usable": usable,
                "reason": None if usable else reason or "one or more original repeats unavailable",
            }
        )
    if not set(observed) <= declared:
        raise ValueError("dataset observation lacks its original archived record")
    requested, seen = [], set()
    for requirement in requirements:
        key, profile = requirement["identity"], requirement["profile"]
        if type(key) is not str or not key or key in seen or profile is not None and type(profile) is not dict:
            raise ValueError("coverage needs unique identities and exact profiles or explicit UNKNOWN")
        seen.add(key)
        matches = [
            row
            for row in cases
            if profile is not None
            and row["semantic_profile"] == profile
            and row["case"]["mechanism_family"] == profile.get("family")
        ]
        groups = {
            split: sorted(
                {
                    row["case"]["source_group_sha256"]
                    for row in matches
                    if row["case"]["split"] == split and row["native_usable"]
                }
            )
            for split in ("train", "calibration", "test")
        }
        requested.append(
            {
                "requirement": requirement,
                "status": "exact_public_train_profile_present" if groups["train"] else "UNKNOWN",
                "reason": (
                    "exact source semantics/geometry in original public TRAIN; current execution domain still required"
                    if groups["train"]
                    else "complete semantic profile unavailable"
                    if profile is None
                    else "no usable original public TRAIN source with this exact semantic profile"
                ),
                "source_groups_by_split": groups,
                "case_ids": [row["case"]["case_id"] for row in matches],
                "record_ids": [record["record_id"] for row in matches for record in row["records"]],
            }
        )
    result = {
        "schema": SCHEMA,
        "catalog_sha256": C.digest(catalog),
        "dataset_sha256": C.digest(dataset),
        "archive_sha256": catalog["archive_sha256"],
        "domain": dataset["domain"],
        "cases": cases,
        "requirements": requested,
        "counts": {
            "declared_records": len(declared),
            "available_native_records": sum(r["native_available"] for c in cases for r in c["records"]),
            "available_labels": sum(r["label_available"] for c in cases for r in c["records"]),
            "available_counter_reviews": sum(r["counter_review_available"] for c in cases for r in c["records"]),
            "source_groups": len({row["case"]["source_group_sha256"] for row in cases}),
            "source_groups_by_split": {
                split: len({row["case"]["source_group_sha256"] for row in cases if row["case"]["split"] == split})
                for split in ("train", "calibration", "test")
            },
            "native_cases_by_split": dict(Counter(row["case"]["split"] for row in cases if row["native_usable"])),
            "requirement_status": dict(Counter(row["status"] for row in requested)),
        },
        "measured_now": False,
        "timing_qualification_issued": False,
        "current_prediction_issued": False,
        "promotion_authority": "NONE",
        "limits": [
            "the current recognizer requires exact TRAIN semantics/geometry; this report grants no extrapolation",
            "calibration/held matches are reported separately and cannot supply TRAIN support",
            "capacity, bank, overlap and schedule alternatives require original exact schedule pairs",
            "complete current numerics, raw event domain, hardware/runtime/window and target semantics remain separate",
        ],
    }
    if calibration_plan is not None:
        from . import calibration_validation

        result["capacity_validation"] = calibration_validation.decide(calibration_plan, {})
        result["capacity_reason"] = "no exact measured source/schedule pair is supplied by this source-profile join"
    return result
