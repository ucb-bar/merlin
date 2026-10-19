"""Native queue inference from original process and full numerical observations.

This consumes an independently qualified event engine; it neither fits prices
nor grants target semantics. Identical executions supply one prediction, while
both original functional arms remain required and visible.
"""

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


def inference_case(*, key, member, family, dataset, manifest_sha256, report, directory, value):
    """Reopen the original private numerical verdict for this exact queued arm."""
    from merlin_timing import contracts as C

    path = directory / "ordinary_observation.json"
    pin = F.pin(path, max_bytes=64 << 20)
    if pin["sha256"] != value.get("evidence_sha256"):
        raise StageGateError("native inference lost its original full numerical observation")
    original = json.loads(path.read_bytes())
    public = {name: item for name, item in original.items() if name != "original_grade"}
    if canonical_json(public) != canonical_json({k: v for k, v in value.items() if k != "evidence_sha256"}):
        raise StageGateError("native inference numerical receipt differs from its original returned observation")
    tiers = original.get("original_grade", {}).get("tiers", {})
    if original.get("functional_status") != "PASS" or any(
        tiers.get(tier, {}).get("status") != "pass" for tier in ("L0", "L1", "L2")
    ):
        raise StageGateError("native inference requires original complete L0/L1/L2 numerical checks")
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

    unique, aliases = {}, {}
    for row in requests:
        case = row["case"]
        identity = tuple(case[name] for name in ("elf_sha256", "input_sha256", "scope_sha256"))
        prior = unique.setdefault(identity, row)
        # Two fresh process runs of the same bytes must agree on their actual
        # event observation. Never choose a faster repeat or average conflicts.
        if (
            C.execution_identity(case) != C.execution_identity(prior["case"])
            or row["features"] != prior["features"]
            or row["event_report"] != prior["event_report"]
            or case["mechanism_family"] != prior["case"]["mechanism_family"]
        ):
            raise StageGateError("identical native executions disagree; no representative may be selected")
        aliases[case["case_id"]] = prior["case"]["case_id"]
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
            {**predictions[original], "case_id": key, "prediction_kind": "event_timing"}
            for key, original in aliases.items()
        ],
    }
