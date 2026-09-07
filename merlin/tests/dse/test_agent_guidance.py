"""Experiment agents receive verified source symbols, not guessed edit suggestions."""
from pathlib import Path

import pytest

from merlin.perf.agent_guidance import (
    build_compiler_edit_contract,
    guidance_for_emission_analysis,
    guidance_for_report,
    inspect_compiler_package,
)
from merlin.perf.global_planner import OccupancySummary
from merlin.perf.whole_model_report import ModelPerformance, evaluate_whole_models


def _package(root: Path, *, surface: bool = True) -> Path:
    package = root / "compiler"
    (package / "lowering").mkdir(parents=True)
    (package / "lowering" / "schedule.py").write_text(
        "class Scheduler:\n    def choose(self):\n        return None\n", encoding="utf-8")
    declared = "" if not surface else """
optimization_surfaces:
  - id: overlap-policy
    scope: heuristic
    path: lowering/schedule.py
    symbol: Scheduler.choose
    effects: [latency_hiding, movement]
    cca_axes: [dispatch.dma_overlap, communication.copy_compute_overlap]
    mechanism: overlap independent transfers with compute
    emitted_delta: dependency schedule changes while work stays equal
    validation: warm paired pipeline witness
    abandonment: emitted dependencies or warm cycles do not improve
"""
    (package / "manifest.yaml").write_text("""
components:
  emit:
    - lowering/
""" + declared, encoding="utf-8")
    return package


def _report():
    occupancy = OccupancySummary(
        total_cycles=200, busy_cycles=(("array", 100), ("dma", 100)),
        compute_resources=("array",), movement_resources=("dma",),
        overlap_cycles=20, overlap_available_cycles=100,
        idle_cycles=0, critical_path_cycles=200,
        movement_bytes=1000, movement_commands=4)
    return evaluate_whole_models((
        ModelPerformance("model", 1000, 200, 190, 150, True, occupancy),))


def test_manifest_surface_resolves_to_real_symbol_and_ranked_problem(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path))
    brief = guidance_for_report(_report(), inventory)

    movement = next(row for row in brief["ranked_actions"]
                    if row["opportunity"]["kind"] == "exposed_movement")
    assert movement["status"] == "actionable"
    assert movement["edit_surfaces"][0]["path"] == "lowering/schedule.py"
    assert movement["edit_surfaces"][0]["symbol"] == "Scheduler.choose"
    assert movement["edit_surfaces"][0]["line"] == 2
    assert movement["mapping_basis"] == "exact CCA axis"
    assert movement["edit_surfaces"][0]["cca_axis_status"] == {
        "communication.copy_compute_overlap": "BACKEND_STUB",
        "dispatch.dma_overlap": "LEVER",
    }


def test_missing_semantic_declaration_stays_unknown_but_source_is_indexed(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path, surface=False))

    assert inventory.surfaces == ()
    assert inventory.symbols
    assert "cannot guess" in inventory.missing[0]


def test_host_surface_declarations_resolve_against_actual_compiler_not_manifest(tmp_path):
    package = _package(tmp_path)
    declarations = [surface.to_dict() for surface in inspect_compiler_package(package).surfaces]
    manifest = package / "manifest.yaml"
    manifest.write_text("components:\n  emit:\n    - lowering/\n")
    assert not inspect_compiler_package(package).surfaces
    inventory = inspect_compiler_package(package, host_surface_declarations=declarations)
    assert len(inventory.surfaces) == 1
    assert inventory.surfaces[0].symbol == "Scheduler.choose"
    assert inventory.surfaces[0].line == 2
    assert not inventory.missing
    declarations[0]["symbol"] = "Scheduler.absent"
    with pytest.raises(ValueError, match="does not resolve to AST"):
        inspect_compiler_package(package, host_surface_declarations=declarations)


def test_edit_contract_names_actual_symbols_and_explicit_helper_extensions(tmp_path):
    inventory = inspect_compiler_package(_package(tmp_path))
    contract = build_compiler_edit_contract(inventory, helper_extensions=[{
        'directory': 'lowering', 'surface_ids': ['overlap-policy'],
        'reason': 'new scheduling helper for this declared mechanism',
    }])
    assert contract['existing_symbols'] == [{
        'surface_id': 'overlap-policy', 'path': 'lowering/schedule.py',
        'symbol': 'Scheduler.choose', 'kind': 'function', 'line': 2,
    }]
    assert len(contract['sha256']) == 64
    assert 'catalog is not enforcement' in contract['enforcement']
    assert 'source_operation_ids' in contract['work_order_required_fields']
    assert build_compiler_edit_contract(inventory)['helper_extensions'] == []


@pytest.mark.parametrize('directory,owners', [('.', ['overlap-policy']),
    ('../escape', ['overlap-policy']), ('evaluator', ['overlap-policy']),
    ('lowering', ['self-authorized-surface'])])
def test_contract_extension_cannot_escape_or_claim_undeclared_owner(tmp_path, directory, owners):
    inventory = inspect_compiler_package(_package(tmp_path))
    with pytest.raises(ValueError):
        build_compiler_edit_contract(inventory, helper_extensions=[{
            'directory': directory, 'surface_ids': owners, 'reason': 'test',
        }])


def test_global_catalog_exposes_unpriced_abi_arena_and_exact_epilogue_gaps(tmp_path):
    inventory = inspect_compiler_package(_package(tmp_path))
    brief = guidance_for_emission_analysis({}, inventory)
    gaps = {row['gap']: row for row in brief['gap_coverage']}
    for name in ('arena_lifetimes_and_reuse', 'entry_abi_and_runtime_overhead',
                 'exact_quantized_epilogues_and_residuals'):
        assert gaps[name]['evidence_status'].startswith('UNKNOWN')
        assert gaps[name]['cheap_validation']
    assert brief['compiler_edit_contract_template']['existing_symbols']


def test_host_payload_guides_edits_without_claiming_cycles(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path))
    analysis = {"verified_global_plan_emission": {
        "status": "verified", "host_activity": {
            "status": "derived", "load_payload_bytes": 120, "store_payload_bytes": 80,
            "top_tasks_by_scalar_memory_payload": [{"task": 2, "source_regions": ["activation"]}],
        }}}
    brief = guidance_for_emission_analysis(analysis, inventory)
    hotspot = next(row for row in brief["ranked_actions"] if row["kind"] == "host_memory_hotspot")
    assert hotspot["status"] == "actionable"
    assert hotspot["evidence"]["load_payload_bytes"] == 120
    assert "not DRAM or CPU cycles" in hotspot["evidence"]["scope"]
    assert brief["timing_status"] == "UNMEASURED"
    analysis["verified_global_plan_emission"]["status"] = "refused"
    refused = guidance_for_emission_analysis(analysis, inventory)
    assert not any(row["kind"] == "host_memory_hotspot" for row in refused["ranked_actions"])


def test_static_materialization_hotspot_visible_without_known_dynamic_cost(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path))
    allocation = {"buffer_root": "alloca:31", "allocation_operation_index": 31,
                  "allocation_task": "2", "static_allocation_payload_bytes": 123456,
                  "load_payload_bytes": None, "store_payload_bytes": None,
                  "source_operation_attribution": "UNKNOWN"}
    analysis = {"verified_global_plan_emission": {"status": "verified", "host_activity": {
        "status": "UNKNOWN", "artifact_sha256": "a" * 64,
        "top_allocations_by_static_payload": [allocation],
        "top_buffers_by_scalar_memory_payload": [],
    }}}
    brief = guidance_for_emission_analysis(analysis, inventory)
    hotspot = next(row for row in brief["ranked_actions"] if row["kind"] == "host_memory_hotspot")
    assert hotspot["evidence"]["top_allocations"] == [allocation]
    assert hotspot["evidence"]["artifact_sha256"] == "a" * 64
    assert hotspot["evidence"]["load_payload_bytes"] is None
    assert "not stack-frame size" in hotspot["evidence"]["scope"]
    assert brief["timing_status"] == "UNMEASURED"


def test_surface_cannot_point_at_a_symbol_that_does_not_exist(tmp_path) -> None:
    package = _package(tmp_path)
    manifest = package / "manifest.yaml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("Scheduler.choose", "Scheduler.missing"),
        encoding="utf-8")

    with pytest.raises(ValueError, match="does not resolve"):
        inspect_compiler_package(package)


def test_surface_rejects_metric_or_unknown_cca_axes(tmp_path) -> None:
    package = _package(tmp_path)
    manifest = package / "manifest.yaml"
    original = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        original.replace("dispatch.dma_overlap", "dispatch.n_dispatches"), encoding="utf-8")
    with pytest.raises(ValueError, match="non-editable CCA axes"):
        inspect_compiler_package(package)

    manifest.write_text(
        original.replace("dispatch.dma_overlap", "dispatch.not_an_axis"), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown CCA axes"):
        inspect_compiler_package(package)


def test_emission_brief_maps_global_evidence_to_exact_edit_coordinate(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path))
    analysis = {
        "arms": {
            "baseline": {"macs": 100, "movement": {"known_bytes": 100}},
            "candidate": {
                "macs": 100,
                "movement": {"known_bytes": 140, "is_lower_bound": False},
                "representation_activity": {
                    "emitted_encoding_transitions": {"status": "UNKNOWN",
                                                       "declared_directives": 2},
                    "occupancy": {"status": "UNKNOWN"},
                },
            },
        },
        "barriers": {"status": "counted", "removed": 0},
        "structural_levels": {"candidate": {"findings": []}},
    }

    brief = guidance_for_emission_analysis(analysis, inventory)

    movement = next(row for row in brief["ranked_actions"]
                    if row["kind"] == "movement_regression")
    assert movement["status"] == "actionable"
    assert movement["edit_surfaces"][0]["path"] == "lowering/schedule.py"
    assert movement["edit_surfaces"][0]["symbol"] == "Scheduler.choose"
    assert movement["edit_surfaces"][0]["line"] == 2
    assert movement["mapping_basis"] == "exact CCA axis"


def test_unmapped_emission_problem_points_to_the_manifest_declaration(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path, surface=False))
    analysis = {
        "arms": {"candidate": {"representation_activity": {
            "emitted_encoding_transitions": {"status": "UNKNOWN"},
            "occupancy": {"status": "UNKNOWN"},
        }}},
    }

    brief = guidance_for_emission_analysis(analysis, inventory)

    assert brief["ranked_actions"]
    assert all(row["status"] == "unmapped" for row in brief["ranked_actions"])
    assert all("manifest.yaml" in row["next_step"] for row in brief["ranked_actions"])


def test_emitted_artifact_and_trace_evidence_join_the_same_edit_loop(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path))
    analysis = {
        "arms": {
            "baseline": {"macs": 100, "movement": {"known_bytes": 100}},
            "candidate": {"status": "emitted", "macs": 100, "movement": {"known_bytes": 100},
                          "representation_activity": {
                              "emitted_encoding_transitions": {"status": "UNKNOWN"},
                              "occupancy": {"status": "UNKNOWN"}}},
        },
        "target_artifact_activity": {
            "baseline": {"status": "decoded", "issued": {
                "movement_instructions": 2, "configuration_instructions": 1,
                "synchronization_instructions": 2, "loop_descriptor_instructions": 1}},
            "candidate": {"status": "decoded", "issued": {
                "movement_instructions": 5, "configuration_instructions": 2,
                "synchronization_instructions": 3, "loop_descriptor_instructions": 0},
                "encoding_resolution": {"status": "complete"},
                "artifact_opportunities": [{
                    "axis": "dispatch.dma_overlap", "observation": "DMA is immediately awaited",
                    "change": "overlap its consumer", "status": "forkable",
                    "forkable_now": True}]},
        },
        "trace_conformance": {
            "status": "checked",
            "introduced_candidate_findings": ["residency: redundant load"],
        },
    }

    brief = guidance_for_emission_analysis(analysis, inventory)

    kinds = {row["kind"] for row in brief["ranked_actions"]}
    assert {"issued_movement_regression", "dispatch_configuration_regression",
            "artifact_synchronization_regression", "loop_offload_regression",
            "artifact_cca_opportunity", "trace_regression"} <= kinds
    opportunity = next(row for row in brief["ranked_actions"]
                       if row["kind"] == "artifact_cca_opportunity")
    assert opportunity["mapping_basis"] == "exact CCA axis"
    assert opportunity["edit_surfaces"][0]["symbol"] == "Scheduler.choose"
    assert brief["mechanism_coverage"] == {
        "host_scalar_work": "UNKNOWN",
        "arithmetic_demand": "observed",
        "whole_model_lowering": "emitted",
        "declared_movement_volume": "declared_exact",
        "issued_movement": "decoded",
        "target_encoding": "complete",
        "dispatch_shape": "decoded",
        "synchronization": "decoded",
        "residency_reload_defect": "checked",
        "occupancy": "UNKNOWN for full model; bound reduced profiles are reported separately as scoped evidence",
        "contention": "UNKNOWN for full model; executed reduced counters do not establish model-wide contention",
        "cycle_ordering": "UNMEASURED: cheap signals are not validated schedule rankers",
    }
    gaps = {row["gap"]: row for row in brief["gap_coverage"]}
    assert set(gaps) == {
        "whole_model_placement_and_coverage",
        "arithmetic_lowering",
        "encoding_and_layout",
        "movement_and_materialization",
        "residency_across_operations",
        "fusion_and_host_boundaries",
        "dispatch_and_loop_offload",
        "latency_hiding_and_double_buffering",
        "synchronization",
        "capacity_and_contention",
        "exact_quantized_epilogues_and_residuals",
        "arena_lifetimes_and_reuse",
        "entry_abi_and_runtime_overhead",
    }
    overlap_gap = gaps["latency_hiding_and_double_buffering"]
    assert overlap_gap["edit_status"] == "mapped"
    assert overlap_gap["edit_surfaces"][0]["symbol"] == "Scheduler.choose"
    assert "reduced warm counter run" in overlap_gap["cheap_validation"]
    assert gaps["encoding_and_layout"]["edit_status"] == "missing_verified_surface"


def test_declined_full_model_is_the_first_named_gap_not_zero_work(tmp_path) -> None:
    inventory = inspect_compiler_package(_package(tmp_path, surface=False))
    analysis = {"arms": {"candidate": {
        "status": "declined", "macs": None,
        "declined": {"op": "host_lane", "reason": "cannot roll the full model"},
        "movement": {"known_bytes": None, "is_lower_bound": True},
        "representation_activity": {
            "emitted_encoding_transitions": {"status": "UNKNOWN"},
            "occupancy": {"status": "UNKNOWN"}},
    }}}

    brief = guidance_for_emission_analysis(analysis, inventory)

    assert brief["ranked_actions"][0]["kind"] == "whole_model_lowering_declined"
    assert brief["ranked_actions"][0]["status"] == "unmapped"
    assert brief["mechanism_coverage"]["whole_model_lowering"] == "declined"
    assert brief["mechanism_coverage"]["arithmetic_demand"] == "UNKNOWN"
    assert brief["mechanism_coverage"]["declared_movement_volume"] == \
        "unavailable: lowering declined"


def test_bound_compiler_owned_plan_is_not_confused_with_shared_solver_or_costs(tmp_path):
    inventory = inspect_compiler_package(_package(tmp_path))
    plan = {key: "a" * 64 for key in ("candidate_sha256", "source_sha256", "logical_dispatch_digest",
        "plan_digest", "candidate_lowered_sha256", "candidate_command_buffer_sha256")}
    plan.update(status="verified", source_operations=19, tasks=3,
                emitted_operations_by_task={"0": 4, "1": 10, "2": 2}, compiler_temporaries=["temporary"],
                proof_scope="source/task/ABI/CFG only", numeric_equivalence="requires reduced witness")
    brief = guidance_for_emission_analysis({"verified_global_plan_emission": plan}, inventory)
    wiring = brief["global_planner_wiring"]
    assert wiring["status"] == "compiler_owned_plan_verified"
    assert wiring["compiler_owned_plan"]["identities"]["source_sha256"] == "a" * 64
    assert wiring["compiler_owned_plan"]["owned_emitted_operation_count"] == 16
    assert wiring["compiler_owned_plan"]["compiler_temporary_count"] == 1
    assert wiring["shared_exact_cover_search"]["status"] == "UNKNOWN"
    assert wiring["compiler_owned_plan"]["costs"] == "UNKNOWN"
    assert wiring["compiler_owned_plan"]["optimality"] == "UNPROVEN"
    assert "optimize and re-verify the existing" in wiring["required"]
    del plan["candidate_lowered_sha256"]
    unresolved = guidance_for_emission_analysis({"verified_global_plan_emission": plan}, inventory)
    assert unresolved["global_planner_wiring"]["status"] == "UNKNOWN"


def test_declined_or_refused_gap_evidence_never_becomes_ready(tmp_path):
    inventory = inspect_compiler_package(_package(tmp_path))
    for status in ("declined", "refused", "unsupported", "unavailable: compiler declined"):
        analysis = {"target_artifact_activity": {"candidate": {"status": status}},
                    "trace_conformance": {"status": status},
                    "structural_levels": {"candidate": {"status": status, "findings": []}}}
        brief = guidance_for_emission_analysis(analysis, inventory)
        gaps = {row["gap"]: row for row in brief["gap_coverage"]}
        assert gaps["residency_across_operations"]["coverage_status"] == "needs_evidence"
        assert gaps["synchronization"]["coverage_status"] == "needs_evidence"
        assert gaps["dispatch_and_loop_offload"]["coverage_status"] == "needs_evidence"
        assert gaps["fusion_and_host_boundaries"]["coverage_status"] == "needs_evidence"
