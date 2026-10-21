"""Fixed-model exploratory cycles, including outside the calibrated domain."""

import json
import math
from pathlib import Path

from merlin.runtime import _native_files as F

from .contracts import StageGateError, document_sha256

PROTOCOL = "merlin.public_native_exploratory_estimate.v1"


def project(owner, counts, report_path):
    """Use an already verified original count report; never refit or reprice it."""
    pin = F.pin(report_path, max_bytes=16 << 20)
    if pin["sha256"] != counts["provenance"]["report_sha256"]:
        raise StageGateError("exploratory estimate lost its original count report")
    report = json.loads(Path(pin["path"]).read_bytes())
    model = json.loads(owner.selection.runtime.model_json)
    expected_domain = dict(model["domain"], target_model_sha256=F.digest(owner.selection.runtime.model_path))
    if report.get("domain") != expected_domain:
        raise StageGateError("exploratory estimate changed its fixed model comparison domain")
    reviewed = json.loads(owner.reviewed_json)
    ranges = reviewed["raw_activity_ranges"]
    raw = report["feature_counts"]
    deviations = []
    for name in sorted(set(raw) | set(ranges)):
        value, support = raw.get(name), ranges.get(name)
        if support is None or value is None or not support[0] <= value <= support[1]:
            deviations.append({"feature": name, "value": value, "training_range": support})
    cycles = report.get("estimated_cycles")
    available = type(cycles) in (int, float) and math.isfinite(cycles) and cycles > 0
    window = report.get("counter_windows")
    if (
        not isinstance(window, list) or len(window) != 1
        or window[0].get("estimated_cycles") != cycles
        or window[0].get("begin_pc") != counts["provenance"]["scope_binding"]["begin_pc"]
        or window[0].get("end_pc") != counts["provenance"]["scope_binding"]["end_pc"]
    ):
        raise StageGateError("exploratory cycles differ from the original selected counter window")
    unresolved = report.get("unresolved", [])
    result = {
        "schema": PROTOCOL, "status": "estimated" if available else "unavailable",
        "estimated_cycles": cycles if available else None,
        "model_sha256": report["model_sha256"],
        "comparison_domain_sha256": document_sha256(expected_domain),
        "scope_sha256": report["scope_sha256"], "report_sha256": pin["sha256"],
        "qualified_for_promotion": False,
        "event_accounting_complete": report.get("timing_complete") is True and unresolved == [],
        "unresolved_events": unresolved,
        "domain_deviations": deviations,
        "calibration_residuals": {
            "qualification_report_sha256": document_sha256(reviewed["report"]),
            "held_domain_metrics": reviewed["report"].get("metrics", {}),
            "applicability": "calibration-domain evidence; no out-of-domain error guarantee",
        },
        "uncertainty_interval_cycles": None,
        "scope": "fixed-model estimate for exploration; no measured latency or promotion authority",
    }
    validate(result, counts=counts)
    if F.pin(report_path, max_bytes=16 << 20) != pin:
        raise StageGateError("original exploratory report changed during projection")
    return result


def validate(value, *, counts):
    fields = {
        "schema", "status", "estimated_cycles", "model_sha256", "comparison_domain_sha256",
        "scope_sha256", "report_sha256", "qualified_for_promotion", "event_accounting_complete",
        "unresolved_events", "domain_deviations", "calibration_residuals",
        "uncertainty_interval_cycles", "scope",
    }
    if (
        type(value) is not dict or set(value) != fields or value["schema"] != PROTOCOL
        or value["status"] not in {"estimated", "unavailable"}
        or value["qualified_for_promotion"] is not False
        or type(value["event_accounting_complete"]) is not bool
        or type(value["unresolved_events"]) is not list or type(value["domain_deviations"]) is not list
        or type(value["calibration_residuals"]) is not dict
        or value["uncertainty_interval_cycles"] is not None
        or value["report_sha256"] != counts["provenance"]["report_sha256"]
        or value["scope_sha256"] != counts["provenance"]["scope_binding"]["scope_sha256"]
        or any(type(value[k]) is not str or len(value[k]) != 64
               or any(c not in "0123456789abcdef" for c in value[k])
               for k in ("model_sha256", "comparison_domain_sha256", "scope_sha256", "report_sha256"))
    ):
        raise StageGateError("exploratory estimate lost its explicit model/report/scope contract")
    cycles = value["estimated_cycles"]
    if (value["status"] == "estimated" and (
        type(cycles) not in (int, float) or not math.isfinite(cycles) or cycles <= 0
    )) or (value["status"] == "unavailable" and cycles is not None):
        raise StageGateError("exploratory estimate requires positive finite cycles or explicit unavailability")
    return value
