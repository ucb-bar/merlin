"""Original queued numerics plus fresh boxed native event observations."""

import json
import math
from pathlib import Path

from merlin.runtime import _native_files as F

from . import _archive_source_domain as DOMAIN
from . import _native_direct_source_domain as DIRECT
from . import archive_count_observation as COUNTS
from . import archive_native_screen as NATIVE
from . import archive_screen_review as REVIEW
from .contracts import StageGateError, document_sha256, write_json

SCHEMA = "merlin.public_native_member_prediction.v1"


def unknown(owner, reason):
    return {
        "schema": SCHEMA,
        "status": "UNKNOWN",
        "reason": reason,
        "selection_sha256": document_sha256(owner.record()),
        "prediction": None,
        "receipt_sha256": None,
        "measured_cycles": None,
        "promotion_authority": "NONE",
    }


def observe(owner, policy, member, workspace, value, *, sandbox, remaining_s, include_unqualified_counts=False):
    """The caller already wrote the unchanged complete original arm receipt."""
    if include_unqualified_counts is not False:
        from .public_native_exploration import UNQUALIFIED_COUNT_PROTOCOL

        if include_unqualified_counts is not True or owner.count_protocol != UNQUALIFIED_COUNT_PROTOCOL:
            raise StageGateError("unqualified count observation requires its original explicit owner protocol")
    if value.get("functional_status") != "PASS":
        return unknown(owner, "original complete functional checks did not pass")
    if type(remaining_s) not in (float, int) or not math.isfinite(remaining_s) or remaining_s < 1:
        return unknown(owner, "original parent deadline has no count-observation allowance")
    reviewed = json.loads(owner.reviewed_json)
    source = (
        DIRECT.current(policy, member, workspace, value, json.loads(owner.catalog_json))
        if owner.source_domain_protocol is not None
        else DOMAIN.current(member, value["compiled"], workspace, json.loads(owner.catalog_json))
    )
    if source.get("status") != "source_domain_supported" and not include_unqualified_counts:
        return unknown(owner, source.get("reason") or "unsupported current joint public source domain")
    compiled = value["compiled"]
    matches = [
        path
        for path in F.members(workspace, max_files=10000)
        if path.is_file()
        and not path.is_symlink()
        and path.suffix == ".elf"
        and path.stat().st_size <= 64 << 20
        and F.digest(path) == compiled.get("elf_sha256")
    ]
    if not matches:
        return unknown(owner, "original observed queued executable unavailable")
    receipt = COUNTS.observe(
        matches[0],
        runtime=owner.selection.runtime,
        policy=sandbox,
        directory=workspace / "public_native_counts",
        input_sha256=document_sha256({"kind": "original_frozen_public_capsule", "sha256": member.source_sha256}),
        remaining_s=remaining_s,
        numerical_pass=True,
    )
    path = workspace / "public_native_counts_receipt.json"
    write_json(path, receipt)
    if receipt.get("status") != "observed":
        return unknown(owner, receipt.get("reason") or "original boxed count observation unavailable")
    report_path = Path(receipt["report"]["path"])
    if F.pin(report_path, max_bytes=16 << 20)["sha256"] != receipt["report"]["sha256"]:
        raise StageGateError("public native original count report changed")
    report = json.loads(report_path.read_bytes())
    if source.get("status") != "source_domain_supported":
        result = unknown(owner, "Fresh counts observed; timing UNKNOWN: " +
                         (source.get("reason") or "unsupported current joint public source domain"))
        result["receipt_sha256"] = F.digest(path)
        return result
    reason = REVIEW.domain_reason(report, reviewed["raw_activity_ranges"])
    if reason is not None:
        return unknown(owner, reason)
    request = NATIVE.inference_case(
        key="original_queued_arm",
        member=member,
        family=source["family"],
        dataset=reviewed["dataset"],
        manifest_sha256=policy.search_corpus.manifest_sha256,
        report=report,
        directory=workspace,
        value=value,
        checked_source=(policy._checked_source_diagnostic, policy._compiler_pass_selection),
    )
    comparison = NATIVE.compare(
        [request],
        dataset=reviewed["dataset"],
        qualification={"dataset": reviewed["dataset"], "report": reviewed["report"]},
    )
    prediction = comparison["cases"][0]
    if prediction.get("status") != "qualified_native" or prediction.get("public_feedback_eligible") is not True:
        return unknown(owner, prediction.get("reason") or "native held-domain prediction unavailable")
    result = {
        "schema": SCHEMA,
        "status": "predicted",
        "reason": prediction.get("reason"),
        "selection_sha256": document_sha256(owner.record()),
        "prediction": prediction,
        "receipt_sha256": F.digest(path),
        "measured_cycles": None,
        "promotion_authority": "NONE",
    }
    validate_projection(result, owner=owner)
    return result


def validate_projection(value, *, owner):
    fields = {
        "schema",
        "status",
        "reason",
        "selection_sha256",
        "prediction",
        "receipt_sha256",
        "measured_cycles",
        "promotion_authority",
    }
    if (
        type(value) is not dict
        or set(value) != fields
        or value["schema"] != SCHEMA
        or value["selection_sha256"] != document_sha256(owner.record())
        or value["status"] not in {"predicted", "UNKNOWN"}
        or value["measured_cycles"] is not None
        or value["promotion_authority"] != "NONE"
    ):
        raise StageGateError("public native projection changed its explicit unscored selection")
    prediction = value["prediction"]
    if value["status"] == "UNKNOWN":
        if type(value["reason"]) is not str or not value["reason"] or prediction is not None:
            raise StageGateError("unavailable public native prediction needs an explicit reason")
        return value
    interval = prediction.get("interval") if type(prediction) is dict else None
    if (
        not isinstance(interval, (tuple, list))
        or len(interval) != 2
        or any(type(number) not in (int, float) or not math.isfinite(number) or number <= 0 for number in interval)
        or interval[0] > interval[1]
        or prediction.get("prediction_kind") != "event_timing"
        or prediction.get("status") != "qualified_native"
        or prediction.get("public_feedback_eligible") is not True
        or type(value["receipt_sha256"]) is not str
    ):
        raise StageGateError("public native prediction lacks its qualified positive residual interval")
    return value
