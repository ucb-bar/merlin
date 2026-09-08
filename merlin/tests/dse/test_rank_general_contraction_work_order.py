from __future__ import annotations

import copy
import hashlib
import json

import pytest

from merlin.benchharness import hash_tree
from merlin.perf.compiler_edit_scope import validate_edit_contract, validate_mechanism_catalog
from merlin.perf.rank_general_contraction_work_order import (
    MECHANISM_ID,
    build_rank_general_mechanism_documents,
    inventory_portfolio_rank_general_contractions,
    main,
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate(tmp_path):
    candidate = tmp_path / "candidate"
    source = candidate / "mlir_oot/lowering/model_lane.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class IntegerContraction:\n    pass\n\n"
        "def _map_positions():\n    return ()\n\n"
        "def _integer_contraction():\n    return None\n\n"
        "def _place_integer_contractions():\n    return ()\n\n"
        "class MixedBuilder:\n"
        "    def build(self):\n        return None\n"
        "    def _mesh(self):\n        return None\n"
    )
    return candidate


def _contract(candidate):
    rows = [
        ("integer_contraction_geometry", "IntegerContraction"),
        ("contraction_affine_map_recognition", "_map_positions"),
        ("rank_general_integer_contraction", "_integer_contraction"),
        ("contraction_lane_reclassification", "_place_integer_contractions"),
        ("global_partition", "MixedBuilder.build"),
        ("contraction_placement", "MixedBuilder._mesh"),
    ]
    contract = {
        "schema": "compiler_edit_contract_v1",
        "authority": "host-frozen test authority",
        "enforcement": "host gate",
        "existing_symbols": [
            {
                "surface_id": surface,
                "path": "mlir_oot/lowering/model_lane.py",
                "symbol": symbol,
                "kind": "class" if symbol == "IntegerContraction" else "function",
                "line": index + 1,
            }
            for index, (surface, symbol) in enumerate(rows)
        ],
        "helper_extensions": [],
        "protected_controls": [],
        "unmapped_mechanism": "refuse",
        "work_order_required_fields": [],
    }
    contract["sha256"] = _digest(contract)
    return contract


def _analysis(candidate_sha256, *, workload):
    nodes = [
        {
            "kind": "dispatch",
            "op": "kernel_0",
            "inputs": ["a0", "b0", "c0"],
            "outputs": ["o0"],
            "regions": 0,
            "captures": [],
            "prov": {"prov.family": "contraction", "prov.op": "opaque_a",
                     "prov.region_id": "r0"},
        },
        {
            "kind": "dispatch",
            "op": "kernel_1",
            "inputs": ["a1", "b1"],
            "outputs": ["o1"],
            "regions": 0,
            "captures": [],
            "prov": {"prov.family": "contraction", "prov.op": "opaque_b",
                     "prov.region_id": "r1"},
        },
        {
            "kind": "dispatch",
            "op": "kernel_2",
            "inputs": ["a2", "b2", "c2"],
            "outputs": ["o2"],
            "regions": 0,
            "captures": [],
            "prov": {"prov.family": "contraction", "prov.op": "opaque_c",
                     "prov.region_id": "r2"},
        },
        {
            "kind": "dispatch",
            "op": "kernel_3",
            "inputs": ["a3", "b3"],
            "outputs": ["o3"],
            "regions": 0,
            "captures": [],
            "prov": {"prov.family": "contraction", "prov.op": "opaque_d",
                     "prov.region_id": "r3", "prov.role": "contraction"},
        },
    ]
    buffers = {}
    for index in range(4):
        shape = [2, 3, 4] if index != 2 else [3, 4]
        for prefix, dtype in (("a", "i8"), ("b", "i16"), ("c", "i32")):
            buffers[f"{prefix}{index}"] = {
                "id": f"{prefix}{index}", "shape": shape, "dtype": dtype,
                "kind": "arg" if prefix != "c" else "intermediate",
            }
        buffers[f"o{index}"] = {
            "id": f"o{index}", "shape": shape, "dtype": "i32", "kind": "intermediate",
        }
    program = {
        "entry": "entry", "args": [], "buffers": buffers, "nodes": nodes, "results": ["o3"],
    }
    logical_digest = _digest(program)
    pins = {
        "source_sha256": "1" * 64,
        "lowered_sha256": "2" * 64,
        "command_buffer_sha256": "3" * 64,
        "compiler_sha256": candidate_sha256,
        "logical_dispatch_digest": logical_digest,
        "plan_digest": "6" * 64,
        "target_facts_sha256": "7" * 64,
        "host_verifier_policy_sha256": "8" * 64,
    }
    tasks = [
        {
            "task_index": 0, "declared_task_kind": "host",
            "source_op_indices": [0, 2, 3], "instruction_indices": [0],
            "owned_instruction_payload_sha256": "9" * 64,
        },
        {
            "task_index": 1, "declared_task_kind": "contraction",
            "source_op_indices": [1], "instruction_indices": [1],
            "owned_instruction_payload_sha256": "a" * 64,
        },
    ]
    placement_rows = [
        {
            "ordinal": 0, "region": "r0", "op": "linalg.generic", "source_op_index": 0,
            "mac_status": "derived",
            "mac_basis": "one proved yielded MAC per static affine-domain point",
            "parallel": [2, 3, 4], "reduction": [5], "macs": 120, "lane": "fallback",
        },
        {
            "ordinal": 1, "region": "r1", "op": "linalg.generic", "source_op_index": 1,
            "mac_status": "derived",
            "mac_basis": "one proved yielded MAC per static affine-domain point",
            "parallel": [2, 3, 4], "reduction": [5], "macs": 120, "lane": "engine",
        },
        {
            "ordinal": 2, "region": "r2", "op": "linalg.generic", "source_op_index": 2,
            "mac_status": "derived",
            "mac_basis": "one proved yielded MAC per static affine-domain point",
            "parallel": [3, 4], "reduction": [5], "macs": 60, "lane": "fallback",
        },
    ]
    plan = {
        "status": "verified", "source_operations": len(nodes), "tasks": len(tasks),
        "candidate_sha256": candidate_sha256, "source_sha256": pins["source_sha256"],
        "candidate_lowered_sha256": pins["lowered_sha256"],
        "candidate_command_buffer_sha256": pins["command_buffer_sha256"],
        "logical_dispatch_digest": logical_digest, "plan_digest": pins["plan_digest"],
    }
    return {
        "schema": "host_owned_whole_model_emission_analysis_v2",
        "candidate_sha256": candidate_sha256,
        "compiler_edit_scope": {"status": "allowed", "contract_sha256": None},
        "workload": workload,
        "emission": {
            "candidate_command_buffer_sha256": pins["command_buffer_sha256"],
            "candidate_lowered_sha256": pins["lowered_sha256"],
        },
        "diagnostics": {
            "verified_global_plan_emission": plan,
            "captured_logical_graph": {
                "schema": "captured_global_graph_v1", "status": "verified",
                "source_sha256": pins["source_sha256"],
                "logical_dispatch_digest": logical_digest,
                "nodes": len(nodes), "dispatch_program": program,
            },
            "task_instruction_evidence": {"candidate": {
                "schema": "task_instruction_evidence_v1",
                "status": "static_ownership_verified",
                "declared_source_plan_status": "verified",
                "binding": pins, "tasks": tasks,
            }},
            "model_contraction_placement": {"candidate": {
                "schema": "model_contraction_placement_v1", "status": "complete",
                "contraction_count": len(placement_rows), "contractions": placement_rows,
                "unresolved": [], "conflicting_regions": [],
            }},
        },
    }


def _record(candidate, contract):
    candidate_sha256 = hash_tree(candidate)["sha256"]
    identity = {
        "analysis": "full_graph_compile_and_static_only",
        "capsule": "opaque-member",
        "capsule_sha256": "b" * 64,
        "full_model_simulation_allowed": False,
        "required_lanes": [],
        "required_tiers": [],
        "role": "primary",
    }
    workload = {key: identity[key] for key in
                ("capsule", "capsule_sha256", "required_lanes", "required_tiers")}
    analysis = _analysis(candidate_sha256, workload=workload)
    analysis["compiler_edit_scope"]["contract_sha256"] = contract["sha256"]
    analysis["optimization_brief"] = {"compiler_edit_contract_template": copy.deepcopy(contract)}
    portfolio_identity = {
        "schema": "full_model_optimization_portfolio_v1", "members": [identity],
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }
    authority = {
        "schema": "host_frozen_compiler_edit_authority_v1",
        "contract": copy.deepcopy(contract),
        "contract_document_sha256": _digest(contract),
        "initial_candidate_sha256": candidate_sha256,
        "source_pins_checked": True,
    }
    return {
        "schema": "global_perf_iteration_v1", "candidate_sha256": candidate_sha256,
        "analysis": analysis,
        "portfolio": {
            "schema": "full_model_portfolio_iteration_v1",
            "candidate_sha256": candidate_sha256,
            "full_model_simulation_allowed": False,
            "selection": "multi_model_pareto_without_invented_static_cycle_total",
            "portfolio_sha256": _digest(portfolio_identity),
            "members_total": 1, "members_ready": 1,
            "members": [{"identity": identity, "analysis_ref": "/analysis",
                         "status": "completed"}],
        },
        "cross_run_static_analysis_binding": {"compiler_edit_authority": authority},
    }


def test_inventory_separates_host_opportunity_already_offloaded_and_uncaptured(tmp_path):
    candidate = _candidate(tmp_path)
    result = inventory_portfolio_rank_general_contractions(
        _record(candidate, _contract(candidate)), iteration_record_sha256="c" * 64)
    assert result["status"] == "ready_for_work_order"
    member = result["members"][0]
    assert member["source_operation_ids"] == [0]
    assert [row["source_operation_id"] for row in member["host_rejected"]] == [0]
    assert [row["source_operation_id"] for row in member["already_offloaded"]] == [1]
    assert member["refusals"][0]["source_operation_id"] == 2
    assert member["refusals"][0]["refusal_class"] == "canonical_rank"
    assert member["uncaptured"] == [{
        "source_operation_id": 3,
        "refusal_class": "uncaptured_contraction",
        "reasons": ["explicit source contraction role has no complete affine-domain MAC witness"],
    }]
    site = member["host_rejected"][0]
    assert site["iteration_rank"] == 4
    assert site["map_evidence"]["status"] == "proved"
    assert site["accumulator_evidence"] == {
        "dtype": "i32", "status": "proved", "source": "yielded_integer_mac_recurrence",
    }
    assert site["ownership_evidence"]["declared_task_kind"] == "host"


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("rank", "rank evidence"),
        ("maps", "affine-map evidence"),
        ("accumulator", "accumulator evidence"),
        ("ownership", "ownership evidence"),
    ],
)
def test_inventory_fails_each_site_closed_without_required_evidence(tmp_path, mutation, expected):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract(candidate))
    analysis = record["analysis"]
    row = analysis["diagnostics"]["model_contraction_placement"]["candidate"]["contractions"][0]
    if mutation == "rank":
        row["parallel"] = None
    elif mutation == "maps":
        row["mac_basis"] = "unproved"
    elif mutation == "accumulator":
        graph = analysis["diagnostics"]["captured_logical_graph"]["dispatch_program"]
        graph["buffers"]["o0"]["dtype"] = "f32"
        digest = _digest(graph)
        analysis["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"] = digest
        analysis["diagnostics"]["verified_global_plan_emission"]["logical_dispatch_digest"] = digest
        analysis["diagnostics"]["task_instruction_evidence"]["candidate"]["binding"][
            "logical_dispatch_digest"] = digest
    else:
        analysis["diagnostics"]["task_instruction_evidence"]["candidate"]["tasks"][0][
            "source_op_indices"].remove(0)

    result = inventory_portfolio_rank_general_contractions(record)
    member = result["members"][0]
    assert 0 not in member["source_operation_ids"]
    refused = next(row for row in member["uncaptured"] if row["source_operation_id"] == 0)
    assert any(expected in reason for reason in refused["reasons"])


def test_builder_emits_valid_single_mechanism_catalog_and_exact_work_order(tmp_path):
    candidate = _candidate(tmp_path)
    contract = _contract(candidate)
    record = _record(candidate, contract)
    documents = build_rank_general_mechanism_documents(
        record, candidate=candidate, expected_candidate_sha256=hash_tree(candidate)["sha256"],
        iteration_record_sha256="c" * 64,
    )
    assert documents["catalog"]["mechanisms"][0]["id"] == MECHANISM_ID
    assert len(documents["catalog"]["mechanisms"]) == 1
    validate_edit_contract(documents["edit_contract"], candidate)
    validate_mechanism_catalog(documents["catalog"], candidate, documents["edit_contract"])
    work = documents["work_order"]
    assert work["initial_candidate_sha256"] == hash_tree(candidate)["sha256"]
    assert work["round_start_candidate_sha256"] == hash_tree(candidate)["sha256"]
    assert work["source_operation_ids"] == []
    assert work["portfolio_site_bindings"][0]["source_operation_ids"] == [0]
    assert work["sha256"] == _digest({key: value for key, value in work.items()
                                      if key != "sha256"})


def test_cli_requires_exact_pins_and_writes_readonly_artifacts_once(tmp_path, capsys):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract(candidate))
    iteration = tmp_path / "iteration.json"
    iteration.write_text(json.dumps(record, sort_keys=True) + "\n")
    iteration.chmod(0o444)
    raw_sha = hashlib.sha256(iteration.read_bytes()).hexdigest()
    candidate_sha = hash_tree(candidate)["sha256"]
    output = tmp_path / "sealed"
    args = [str(iteration.resolve()), str(candidate.resolve()), str(output.resolve()),
            "--iteration-sha256", raw_sha, "--candidate-sha256", candidate_sha]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["opportunity_counts"] == [1]
    for name in ("compiler_edit_contract.json", "mechanism_catalog.json",
                 "mechanism_work_order.json", "source_site_inventory.json", "receipt.json"):
        path = output / name
        assert path.is_file() and path.stat().st_mode & 0o222 == 0
        assert report["artifacts"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        main(args)
