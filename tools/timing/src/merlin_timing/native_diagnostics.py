"""Explain original native observations without inventing scoped attribution.

Execution-window counts and predictions are distinct from whole-trace resource
bookings. Neither is simulator walltime, numerical equivalence or a physical
roofline bound. This reader grants no qualification and launches nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import contracts as C
from . import observations as O
from . import validation as V

SCHEMA = "merlin.native_timing_diagnostics.v1"
MAX_RESOURCES = 1024


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise C.ContractError("duplicate native model key")
        result[key] = value
    return result


def explain(dataset: dict, record_id: str, model_path: str | Path) -> dict:
    """Join one original record to its exact model artifact; no source owner issued."""
    C.dataset(dataset)
    if dataset["profile"] != "phase1-public":
        raise C.ContractError("public diagnostics cannot expose private historical observations")
    C.text(record_id, "record identity")
    records = [row for row in dataset["records"] if row["record_id"] == record_id]
    if len(records) != 1:
        raise C.ContractError("diagnostics need exactly one original declared record")
    record = records[0]
    case = next(row for row in dataset["cases"] if row["case_id"] == record["case_id"])
    path = Path(model_path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise C.ContractError("native model exceeds the 16 MiB input bound")
    raw = path.read_bytes()
    if len(raw) > 16 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != dataset["domain"]["target_model_sha256"]:
        raise C.ContractError("native model artifact differs from the original selected digest")
    model = json.loads(raw, object_pairs_hook=_unique)
    if (
        not isinstance(model, dict)
        or dict(model.get("domain", {}), target_model_sha256=hashlib.sha256(raw).hexdigest()) != dataset["domain"]
    ):
        raise C.ContractError("native model domain differs from original observations")
    prediction, reason = V.native_prediction(record, case, dataset)
    if record["event_report"] is None:
        raise C.ContractError("native diagnostic has no original event report")
    report = record["event_report"]["report"]
    resources = model.get("resources")
    if not isinstance(resources, list) or not 1 <= len(resources) <= MAX_RESOURCES:
        raise C.ContractError("native resource roster is absent or oversized")
    by_id = {}
    for resource in resources:
        if not isinstance(resource, dict) or type(resource.get("id")) is not int or resource["id"] in by_id:
            raise C.ContractError("native resource identities are malformed or repeated")
        C.text(resource.get("name"), "native resource name")
        C.number(resource.get("capacity"), "native resource capacity", positive=True)
        by_id[resource["id"]] = resource
    breakdown = report.get("cycle_breakdown")
    if not isinstance(breakdown, dict):
        raise C.ContractError("native resource breakdown is absent")
    # The current observer exposes whole-trace resource totals only. A future
    # scoped schema needs explicit interval/snapshot semantics before admission.
    if breakdown.get("scope") not in (None, "whole_trace") or breakdown.get("additive") is not False:
        raise C.ContractError("native resource scope/additivity needs an independently reviewed schema")
    rows = breakdown.get("resources")
    if not isinstance(rows, list) or len(rows) > MAX_RESOURCES:
        raise C.ContractError("native resource observations are absent or oversized")
    context, seen = [], set()
    for row in rows:
        C.closed(
            row,
            {"resource_id", "reservations", "busy_unit_cycles", "unknown_events", "peak_queue_entries"},
            "resource observation",
        )
        key = row["resource_id"]
        if type(key) is not int or key not in by_id or key in seen:
            raise C.ContractError("native observation refers to an undeclared/repeated resource")
        seen.add(key)
        for name in ("reservations", "busy_unit_cycles", "unknown_events", "peak_queue_entries"):
            if type(row[name]) is not int or not 0 <= row[name] < 2**64:
                raise C.ContractError("native resource counts need bounded unsigned integers")
        context.append(dict(row, name=by_id[key]["name"], capacity=by_id[key]["capacity"]))
    counts = report["feature_counts"]
    if not isinstance(counts, dict) or len(counts) > 1024:
        raise C.ContractError("native feature roster is absent or oversized")
    for name, value in counts.items():
        C.text(name, "native feature")
        if value is not None:
            C.number(value, "native observed feature")
    warnings = [
        "Resource totals include whole-trace setup and teardown; execution-window utilization and stalls are UNKNOWN."
    ]
    if any(row["unknown_events"] for row in context):
        warnings.append(
            "Whole-trace resource bookings include unknown events even if the selected execution window is complete."
        )
    return {
        "schema": SCHEMA,
        "dataset_sha256": C.digest(dataset),
        "record_id": record_id,
        "case_id": case["case_id"],
        "execution_binding": {
            key: case[key] for key in ("elf_sha256", "input_sha256", "scope_sha256", "device_sha256")
        },
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "report_sha256": C.digest(report),
        "execution_window": {
            "estimated_cycles": prediction,
            "reason": reason,
            "count_complete": O.feature_complete(report),
            "counts": dict(sorted(counts.items())),
            "numerical_equivalence": "requires independent complete-output admission",
            "scope_sha256": case["scope_sha256"],
            "original_measurement": {
                "label": record["label"],
                "fresh_measurement_in_this_invocation": False,
                "interpretation": "Original paid target-cycle label, separate from the native modeled prediction.",
            },
        },
        "resource_context": {
            "scope": breakdown.get("scope", "unspecified"),
            "additive": False,
            "whole_trace_timing_complete": report.get("whole_trace_timing_complete"),
            "resources": context,
            "resource_roster_complete": seen == set(by_id),
            "execution_window_utilization": None,
            "execution_window_stall_cycles": None,
            "physical_floor_authority": False,
        },
        "setup": {
            "compilation_seconds": None,
            "input_staging_seconds": None,
            "reason": "Not measured by native cycle reports; obtain original operator receipts separately.",
        },
        "warnings": warnings,
        "qualification_issued": False,
        "interpretation": (
            "Declared execution-window counts and modeled cycles; whole-trace nonadditive context. "
            "In-window packing/transfers/dispatch/host work remain execution cost. No walltime-to-cycle conversion."
        ),
    }
