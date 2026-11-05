"""Native queue inference from original process and full numerical observations.

This consumes an independently qualified event engine; it neither fits prices
nor grants target semantics. Identical executions supply one prediction, while
both original functional arms remain required and visible.
"""

import hashlib
import json

from merlin.llvmlower.reviewed_pass_capabilities import ReviewedPassCapability
from merlin.runtime import _native_files as F

from .contracts import StageGateError, canonical_json


def verify_semantics(capability, binding):
    if (
        type(capability) is not ReviewedPassCapability
        or capability.mechanism != "native_timing_semantics"
        or capability.binding != binding
    ):
        raise StageGateError("native archive screening needs independently reviewed exact event/resource semantics")
    capability.revalidate()


def original_numerical_observation(directory, value, *, checked_source=None):
    """Reopen the original queued receipt and its live selection, never a verdict argument."""
    path = directory / "ordinary_observation.json"
    pin = F.pin(path, max_bytes=64 << 20)
    if pin["sha256"] != value.get("evidence_sha256"):
        raise StageGateError("native inference lost its original full numerical observation")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        raise StageGateError("native numerical observation changed during reopening")
    original = json.loads(raw)
    public = {name: item for name, item in original.items() if name != "original_grade"}
    if checked_source is not None:
        from merlin.targetgen.checked_source_model_diagnostic import (
            NUMERICAL_COUNTS_PROTOCOL,
            numerical_grade_matches_selection,
        )

        from .ordinary_checked_source import record

        if type(checked_source) is not tuple or len(checked_source) != 2:
            raise StageGateError("native diagnostic inference lost its original live complete-source selection")
        selected = record(*checked_source)
        if (not numerical_grade_matches_selection(selected, original.get("original_grade", {}))
                or public.pop("checked_source_diagnostic", None) != selected):
            raise StageGateError("native diagnostic inference lost its original live complete-source selection")
        if selected.get("verification_protocol") == NUMERICAL_COUNTS_PROTOCOL:
            from merlin.targetgen.checked_source_model_diagnostic_execution import reopen_numerical_counts_grade

            try:
                reopen_numerical_counts_grade(checked_source[0], original["original_grade"])
            except (OSError, ValueError, TypeError, KeyError) as error:
                raise StageGateError("native fast inference lost original source/ISA/full-output witnesses") from error
    if canonical_json(public) != canonical_json({k: v for k, v in value.items() if k != "evidence_sha256"}):
        raise StageGateError("native inference numerical receipt differs from its original returned observation")
    tiers = original.get("original_grade", {}).get("tiers", {})
    if original.get("functional_status") != "PASS" or any(
        tiers.get(tier, {}).get("status") != "pass" for tier in ("L0", "L1", "L2")
    ):
        raise StageGateError("native inference requires original complete L0/L1/L2 numerical checks")
    if F.pin(path, max_bytes=64 << 20) != pin:
        raise StageGateError("native numerical observation changed during consumption")
    return original, pin


def inference_case(*, key, member, family, dataset, manifest_sha256, report, directory, value, checked_source=None):
    """Reopen the original private numerical verdict for this exact queued arm."""
    from merlin_timing import contracts as C

    original, pin = original_numerical_observation(directory, value, checked_source=checked_source)
    path = directory / "ordinary_observation.json"
    if original.get("compiled", {}).get("elf_sha256") != report["elf_sha256"]:
        raise StageGateError("native numerical observation belongs to another executable")
    case = {
        "case_id": key,
        "source_group_sha256": C.digest({"original_public_source": member.source_sha256}),
        "mechanism_family": family,
        "source": {
            "visibility": "public",
            "phase": "phase2",
            "manifest_sha256": manifest_sha256,
            "source_sha256": member.source_sha256,
        },
        **{name: report[name] for name in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")},
    }
    case["workload_sha256"] = C.workload(case)
    row = {
        "case": case,
        "features": {name: report["feature_counts"][name] for name in dataset["feature_names"]},
        "event_report": {"sha256": C.digest(report), "report": report},
        "numeric": {
            "status": "pass",
            "receipt_sha256": pin["sha256"],
            "execution_sha256": C.execution_identity(case),
            "measured_now": True,
        },
    }
    if F.pin(path, max_bytes=64 << 20) != pin:
        raise StageGateError("native numerical observation changed during consumption")
    return row


def compare(requests, *, dataset, qualification):
    from merlin_timing import contracts as C
    from merlin_timing import feedback as B
    from merlin_timing import validation as V

    declaration = {
        "schema": C.NATIVE_CASES,
        "profile": dataset["profile"],
        "domain": dataset["domain"],
        "feature_names": dataset["feature_names"],
    }
    unique, aliases, evidence, signatures = {}, {}, {}, {}
    for row in requests:
        C.native_cases({**declaration, "cases": [row]})
        case = row["case"]
        if case["case_id"] in aliases:
            raise StageGateError("native queue inference requires a distinct identity for each queued arm")
        identity = tuple(case[name] for name in ("elf_sha256", "input_sha256", "scope_sha256"))
        prediction, reason = V.native_prediction(row, case, declaration)
        numeric = row["numeric"]
        if numeric is None or numeric["status"] != "pass" or numeric["measured_now"] is not True:
            raise StageGateError("native duplicate inference requires every original current numerical receipt")
        if row["event_report"] is None:
            raise StageGateError("native queue inference requires each original event observation")
        report = row["event_report"]["report"]
        scope = report.get("scope_binding")
        signature = (
            C.execution_identity(case),
            case["mechanism_family"],
            row["features"],
            prediction,
            reason,
            {name: scope.get(name) for name in ("begin_pc", "end_pc", "occurrence")} if scope else None,
        )
        prior = unique.setdefault(identity, row)
        # Two fresh process runs of the same bytes must agree on their actual
        # scoped prediction. Independent review/process receipts and whole-trace
        # diagnostics need not have identical hashes. Preserve each one's pins.
        # Never choose a faster repeat or average conflicting scoped results.
        if signatures.setdefault(identity, signature) != signature:
            raise StageGateError("identical native executions disagree; no representative may be selected")
        aliases[case["case_id"]] = prior["case"]["case_id"]
        evidence[case["case_id"]] = {
            "event_report_sha256": row["event_report"]["sha256"],
            "numeric_receipt_sha256": numeric["receipt_sha256"],
        }
    result = B.compare_native(
        {
            "schema": C.NATIVE_CASES,
            "profile": dataset["profile"],
            "domain": dataset["domain"],
            "feature_names": dataset["feature_names"],
            "cases": list(unique.values()),
        },
        qualification=qualification,
    )
    predictions = {row["case_id"]: row for row in result["cases"]}
    return {
        **result,
        "cases": [
            {
                **predictions[original],
                "case_id": key,
                "prediction_kind": "event_timing",
                "original_execution_evidence": evidence[key],
            }
            for key, original in aliases.items()
        ],
    }
