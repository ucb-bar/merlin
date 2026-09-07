"""Whole-model work comparisons retain missing dimensions and conflicting changes."""
from copy import deepcopy

from merlin.perf.structural_delta import compare_full_model_structure


def _analysis():
    return {"diagnostics": {
        "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": "a" * 64},
        "verified_global_plan_emission": {"status": "verified", "tasks": 4, "host_activity": {
            "status": "derived", "load_payload_bytes": 100, "store_payload_bytes": 100,
            "static_allocation_payload_bytes": 100,
            "dynamic_operations": {"integer_arithmetic": 10, "floating_arithmetic": 20}}},
        "target_artifact_activity": {"candidate": {"status": "decoded", "issued": {
            "compute_instructions": 5, "configuration_instructions": 2,
            "movement_instructions": 4, "synchronization_instructions": 3}}},
        "arms": {"candidate": {"movement": {"exact_bytes": 100, "is_lower_bound": False}}},
        "model_contraction_placement": {"candidate": {"total_contraction_macs": 200}},
    }}


def test_fewer_fences_is_structural_not_a_cycle_verdict():
    before = _analysis()
    after = deepcopy(before)
    after["diagnostics"]["target_artifact_activity"]["candidate"]["issued"]["synchronization_instructions"] = 2
    result = compare_full_model_structure(before, after)
    assert result["decreased"] == ["issued_synchronization_instructions"]
    assert result["structural_relation"] == "decrease_in_observed_metrics"
    assert result["cycle_selection"] == "UNMEASURED"


def test_host_work_increase_exposes_a_global_tradeoff():
    before = _analysis()
    after = deepcopy(before)
    plan = after["diagnostics"]["verified_global_plan_emission"]
    plan["tasks"] = 2
    plan["host_activity"]["store_payload_bytes"] = 200
    result = compare_full_model_structure(before, after)
    assert result["structural_relation"] == "mixed_work_tradeoff"
    assert result["increased"] == ["host_store_payload_bytes"]


def test_unknown_does_not_cancel_and_changed_model_is_refused():
    before = _analysis()
    before["diagnostics"]["verified_global_plan_emission"]["host_activity"]["status"] = "UNKNOWN"
    result = compare_full_model_structure(before, deepcopy(before))
    assert "host_dynamic_integer_arithmetic" in result["unknown"]
    after = deepcopy(before)
    after["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"] = "b" * 64
    assert compare_full_model_structure(before, after)["status"] == "refused"


def test_post_backend_code_growth_is_a_tradeoff_not_hidden_by_ir_memory_savings():
    before = _analysis()
    before["emission"] = {"candidate_lowered_sha256": "c" * 64}
    before["diagnostics"]["machine_artifact_activity"] = {"candidate": {
        "status": "compiled", "source_sha256": "c" * 64,
        "instruction_sites": {"schema": "encoded_instruction_sites_v1", "total": 10}, "object_bytes": 100}}
    after = deepcopy(before)
    after["diagnostics"]["verified_global_plan_emission"]["host_activity"]["store_payload_bytes"] = 50
    after["diagnostics"]["machine_artifact_activity"]["candidate"]["instruction_sites"]["total"] = 11
    result = compare_full_model_structure(before, after)
    assert result["structural_relation"] == "mixed_work_tradeoff"
    assert "machine_instruction_sites_total" in result["increased"]
    after["diagnostics"]["machine_artifact_activity"]["candidate"]["source_sha256"] = "stale"
    assert "machine_instruction_sites_total" in compare_full_model_structure(before, after)["unknown"]


def test_ir_hygiene_with_identical_object_is_not_emitted_kernel_work_deletion():
    before = _analysis()
    before["emission"] = {"candidate_lowered_sha256": "c" * 64}
    before["diagnostics"]["machine_artifact_activity"] = {"candidate": {
        "status": "compiled", "source_sha256": "c" * 64, "object_sha256": "d" * 64}}
    after = deepcopy(before)
    after["emission"]["candidate_lowered_sha256"] = "e" * 64
    after["diagnostics"]["machine_artifact_activity"]["candidate"]["source_sha256"] = "e" * 64
    after["diagnostics"]["verified_global_plan_emission"]["host_activity"]["dynamic_operations"]["integer_arithmetic"] = 5
    result = compare_full_model_structure(before, after)
    assert "host_dynamic_integer_arithmetic" in result["decreased"]
    assert result["machine_object_comparison"]["status"] == "identical"
    assert "byte-identical" in result["required_next_evidence"][0]
    assert result["cycle_selection"] == "UNMEASURED"
    after["diagnostics"]["machine_artifact_activity"]["candidate"]["source_sha256"] = "stale"
    assert compare_full_model_structure(before, after)["machine_object_comparison"]["status"] == "UNKNOWN"


def test_equal_object_sizes_do_not_imply_equal_machine_code():
    before = _analysis()
    before["emission"] = {"candidate_lowered_sha256": "c" * 64}
    before["diagnostics"]["machine_artifact_activity"] = {"candidate": {
        "status": "compiled", "source_sha256": "c" * 64, "object_sha256": "d" * 64,
        "object_bytes": 100}}
    after = deepcopy(before)
    after["diagnostics"]["machine_artifact_activity"]["candidate"]["object_sha256"] = "e" * 64
    assert compare_full_model_structure(before, after)["machine_object_comparison"]["status"] == "different"


def test_hardware_loop_sites_are_visible_and_not_work_deletion():
    before = _analysis()
    after = deepcopy(before)
    before["diagnostics"]["target_artifact_activity"]["candidate"]["issued"]["loop_descriptor_instructions"] = 0
    issued = after["diagnostics"]["target_artifact_activity"]["candidate"]["issued"]
    issued.update(loop_descriptor_instructions=6, compute_instructions=0, movement_instructions=0)
    report = compare_full_model_structure(before, after)
    assert report["metrics"]["issued_loop_descriptor_instructions"]["delta"] == 6
    assert report["structural_relation"] == "mixed_work_tradeoff"
    assert "dispatch.loop_offloaded" in report["changed_cca_axes"]
    assert report["instruction_accounting"]["after"]["descriptor_expanded_work"] == "UNKNOWN"
    assert any("not work deletion" in reason for reason in report["required_next_evidence"])
    assert report["cycle_selection"] == "UNMEASURED"


def test_partial_decoder_zero_is_only_zero_classified_sites():
    before = _analysis()
    artifact = before["diagnostics"]["target_artifact_activity"]["candidate"]
    artifact["issued"]["compute_instructions"] = 0
    artifact["encoding_resolution"] = {"status": "partial", "unknown_instruction_indices": [2, 4],
                                      "named_without_role": ["unclassified"]}
    report = compare_full_model_structure(before, deepcopy(before))
    arm = report["instruction_accounting"]["before"]
    assert arm["classification_status"] == "partial"
    assert arm["unclassified_instruction_sites"] == 2
    assert arm["dynamic_compute_work"] == "UNKNOWN"
    assert "classified" in report["metrics"]["issued_compute_instructions"]["scope"]


def test_absent_loop_or_decode_coverage_does_not_become_zero():
    before = _analysis()
    report = compare_full_model_structure(before, deepcopy(before))
    assert report["instruction_accounting"]["before"]["loop_descriptor_sites"] is None
    assert report["instruction_accounting"]["before"]["unclassified_instruction_sites"] is None
    assert "issued_loop_descriptor_instructions" in report["unknown"]
