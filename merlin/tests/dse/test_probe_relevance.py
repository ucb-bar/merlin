"""Changed-mechanism routing does not silently upgrade a local calibration to a global claim."""
import hashlib
import json

import pytest

from merlin.perf.activity_schedule import ActivityEvent
from merlin.perf.mechanism_probe import derive_mechanism_signature
from merlin.perf.probe_relevance import classify_probe_relevance
from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation


def row(name, value=1):
    return {"class": name, "funct": 3, "rs1": {"kind": "const", "raw": value},
            "rs2": {"kind": "const", "raw": 0}}


def artifact(rows, suffix=""):
    text = json.dumps(rows) + suffix
    digest = hashlib.sha256(text.encode()).hexdigest()
    return ({"emission": {"candidate_lowered_sha256": digest},
             "diagnostics": {"captured_logical_graph": {"logical_dispatch_digest": "same-graph"}}},
            {"lowered_text": text, "candidate_lowered_sha256": digest,
             "decoded_trace": {"instructions": rows}})


def signature(name, value=1, *, mixed=False):
    events = ([ActivityEvent("transfer", "port", "movement", 0),
               ActivityEvent("body", "array", "compute", 0, ("transfer",))]
              if mixed else [ActivityEvent("body", "whole_motif", "compute", 0)])
    return derive_mechanism_signature(
        representations=[ValueRepresentation("local", "rows", "i8")], events=events,
        capacity_regime={"local": "fits"}, tile_shape=[3, 5], edge_cases=["tail"],
        repetition_semantics="initialized isolated primitive",
        instruction_semantics=[{"instructions": [{"class": name, "selector": 3,
                               "rs1": {"kind": "const", "value": value},
                               "rs2": {"kind": "const", "value": 0}}]}])


def classify(before, after, sampled):
    left, la = artifact(before)
    right, ra = artifact(after)
    return classify_probe_relevance(previous_analysis=left, current_analysis=right,
                                    previous_artifacts=la, current_artifacts=ra, signature=sampled)


def test_scalar_barrier_deletion_is_not_priced_by_unchanged_array_primitive():
    result = classify([row("barrier"), row("array")], [row("array")], signature("array"))
    assert result["status"] == "unrelated_to_changed_instructions"
    assert result["unsampled_changed_classes"] == ["barrier"]
    assert result["exact_previous_changed_payloads_in_probe"] == 0
    assert result["isolated_calibration_allowed"] and not result["global_cost_validated"]


def test_packet_encoding_payload_change_not_hidden_by_equal_class_counts():
    result = classify([row("packet", 1)], [row("packet", 2)], signature("packet", 2))
    assert result["status"] == "unresolved_changed_context"
    assert result["exact_previous_changed_payloads_in_probe"] == 0
    assert result["exact_current_changed_payloads_in_probe"] == 1
    assert not result["changed_context_validated"]


def test_reordered_traffic_requires_context_even_with_multi_resource_probe():
    result = classify([row("packet"), row("array")], [row("array"), row("packet")],
                      signature("packet", mixed=True))
    assert result["instruction_multiset_preserved"]
    assert result["status"] == "unresolved_changed_context"
    assert result["probe_event_evidence"]["dependency_edges"] == 1
    assert not result["global_cost_validated"]


def test_unchanged_trace_does_not_hide_host_encoding_change():
    left, la = artifact([row("array")])
    right, ra = artifact([row("array")], "changed host conversion")
    result = classify_probe_relevance(previous_analysis=left, current_analysis=right,
                                     previous_artifacts=la, current_artifacts=ra,
                                     signature=signature("array"))
    assert result["status"] == "unresolved_host_or_control_flow_change"


def test_stale_artifact_binding_fails_closed():
    left, la = artifact([row("array")])
    right, ra = artifact([row("array", 2)])
    ra["lowered_text"] += "tampered"
    with pytest.raises(ValueError, match="do not match"):
        classify_probe_relevance(previous_analysis=left, current_analysis=right,
                                 previous_artifacts=la, current_artifacts=ra,
                                 signature=signature("array"))


def test_initial_calibration_needs_no_prior_candidate_or_blanket_sweep():
    current, artifacts = artifact([row("array")])
    result = classify_probe_relevance(previous_analysis=None, current_analysis=current,
                                     previous_artifacts=None, current_artifacts=artifacts,
                                     signature=signature("array"))
    assert result["status"] == "calibration_only_no_previous_artifact"
    assert result["isolated_calibration_allowed"]
