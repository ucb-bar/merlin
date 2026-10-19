"""Bounded diagnostic fixtures never qualify a real simulator or device."""

import copy
import hashlib
import json

import pytest
from data_helpers import sample
from merlin_timing import cli
from merlin_timing import contracts as C
from merlin_timing import native_diagnostics as N
from test_event_validation import original_reports


def selection(tmp_path, *, private=False):
    data = sample(profile="historical-private" if private else "phase1-public")
    model = {
        "domain": {k: v for k, v in data["domain"].items() if k != "target_model_sha256"},
        "resources": [{"id": 3, "name": "selected.port", "capacity": 1}],
    }
    raw = (json.dumps(model, sort_keys=True) + "\n").encode()
    modelpath = tmp_path / "model.json"
    modelpath.write_bytes(raw)
    identity = hashlib.sha256(raw).hexdigest()
    data["domain"]["target_model_sha256"] = identity
    for row in data["records"]:
        row["feature_provenance"]["model_sha256"] = identity
    original_reports(data)
    row = data["records"][0]
    report = row["event_report"]["report"]
    report["whole_trace_timing_complete"] = False
    report["cycle_breakdown"] = {
        "scope": "whole_trace",
        "additive": False,
        "resources": [
            {
                "resource_id": 3,
                "reservations": 1000,
                "busy_unit_cycles": 2000,
                "unknown_events": 8,
                "peak_queue_entries": 1,
            }
        ],
    }
    row["event_report"]["sha256"] = C.digest(report)
    return data, row["record_id"], modelpath


def test_setup_does_not_become_kernel_utilization_or_cycles(tmp_path):
    args = selection(tmp_path)
    before = copy.deepcopy(args[0])
    result = N.explain(*args)
    assert result["execution_window"]["estimated_cycles"] == 35
    assert result["resource_context"]["resources"][0]["busy_unit_cycles"] == 2000
    assert result["resource_context"]["execution_window_utilization"] is None
    assert result["resource_context"]["execution_window_stall_cycles"] is None
    assert result["setup"]["compilation_seconds"] is None
    assert result["setup"]["input_staging_seconds"] is None
    assert result["resource_context"]["physical_floor_authority"] is False
    assert not result["qualification_issued"] and len(result["warnings"]) == 2
    assert args[0] == before


@pytest.mark.parametrize("change", ["scope", "additive", "unknown_resource", "duplicate", "negative", "boolean"])
def test_ambiguous_scope_or_corrupt_resource_context_refused(tmp_path, change):
    data, key, path = selection(tmp_path)
    record = data["records"][0]
    report = record["event_report"]["report"]
    breakdown = report["cycle_breakdown"]
    if change == "scope":
        breakdown["scope"] = "kernel"
    elif change == "additive":
        breakdown["additive"] = True
    elif change == "unknown_resource":
        breakdown["resources"][0]["resource_id"] = 4
    elif change == "duplicate":
        breakdown["resources"].append(copy.deepcopy(breakdown["resources"][0]))
    elif change == "negative":
        breakdown["resources"][0]["busy_unit_cycles"] = -1
    else:
        breakdown["resources"][0]["unknown_events"] = True
    record["event_report"]["sha256"] = C.digest(report)
    with pytest.raises(C.ContractError):
        N.explain(data, key, path)


def test_unknown_native_prediction_remains_unknown(tmp_path):
    data, key, path = selection(tmp_path)
    row = data["records"][0]
    row["event_report"]["report"]["timing_complete"] = False
    row["event_report"]["sha256"] = C.digest(row["event_report"]["report"])
    result = N.explain(data, key, path)
    assert result["execution_window"]["estimated_cycles"] is None
    assert result["execution_window"]["reason"]


def test_rebound_model_bytes_and_private_data_cannot_enter_public_feedback(tmp_path):
    data, key, path = selection(tmp_path)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(C.ContractError, match="digest"):
        N.explain(data, key, path)
    data, key, path = selection(tmp_path, private=True)
    with pytest.raises(C.ContractError, match="private"):
        N.explain(data, key, path)


def test_cli_keeps_scope_and_custody(tmp_path):
    data, key, path = selection(tmp_path)
    source = tmp_path / "dataset.json"
    output = tmp_path / "feedback.json"
    C.write(source, data)
    assert (
        cli.main(
            [
                "native-diagnostics",
                "--dataset",
                str(source),
                "--record-id",
                key,
                "--model-artifact",
                str(path),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = C.read(output)
    assert result["record_id"] == key and result["resource_context"]["scope"] == "whole_trace"
    assert result["dataset_sha256"] == C.digest(data)
