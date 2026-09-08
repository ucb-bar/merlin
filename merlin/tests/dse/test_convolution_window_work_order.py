from __future__ import annotations

import copy
import hashlib
import importlib
import json
import sys

import pytest

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root
from merlin.perf.compiler_edit_scope import validate_edit_contract, validate_mechanism_catalog
from merlin.perf.convolution_window_work_order import (
    MECHANISM_ID,
    build_convolution_window_mechanism_documents,
    inventory_portfolio_convolution_windows,
    main,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _candidate(tmp_path):
    candidate = tmp_path / "candidate"
    files = {
        "mlir_oot/lowering/source_model.py": (
            "class SourceBuilder:\n    pass\n\ndef mesh_eligible():\n    return False\n"
        ),
        "mlir_oot/lowering/task.py": "class Convolution:\n    pass\n",
        "mlir_oot/lowering/schedule.py": (
            "class Scheduler:\n"
            "    def convolution(self):\n        return None\n"
            "    def run(self):\n        return None\n"
            "    def contraction(self):\n        return None\n"
        ),
        "mlir_oot/codegen/emitter.py": ("class Emitter:\n    def emit_im2col_row(self):\n        return None\n"),
        "mlir_oot/lowering/plan.py": ("class Builder:\n    def _op_conv2d(self):\n        return None\n"),
        "mlir_oot/lowering/model.py": ("class MixedBuilder:\n    def build(self):\n        return None\n"),
    }
    for relative, source in files.items():
        path = candidate / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return candidate


def _contract():
    rows = [
        ("source_convolution_global_partition", "mlir_oot/lowering/source_model.py", "SourceBuilder"),
        ("source_convolution_mesh_eligibility", "mlir_oot/lowering/source_model.py", "mesh_eligible"),
        ("source_convolution_semantic_task", "mlir_oot/lowering/task.py", "Convolution"),
        ("source_convolution_capability_selected_schedule", "mlir_oot/lowering/schedule.py", "Scheduler.convolution"),
        ("affine_window_packing_specialization", "mlir_oot/codegen/emitter.py", "Emitter.emit_im2col_row"),
        ("global_issue_and_fences", "mlir_oot/lowering/schedule.py", "Scheduler.run"),
        ("pipeline_issue", "mlir_oot/lowering/schedule.py", "Scheduler.contraction"),
        ("convolution_route", "mlir_oot/lowering/plan.py", "Builder._op_conv2d"),
        ("global_partition", "mlir_oot/lowering/model.py", "MixedBuilder.build"),
    ]
    contract = {
        "schema": "compiler_edit_contract_v1",
        "authority": "host-frozen fixture authority",
        "enforcement": "host gate",
        "existing_symbols": [
            {"surface_id": surface, "path": path, "symbol": symbol, "line": index + 1}
            for index, (surface, path, symbol) in enumerate(rows)
        ],
        "helper_extensions": [],
        "protected_controls": [],
        "unmapped_mechanism": "refuse",
        "work_order_required_fields": [],
    }
    contract["sha256"] = _digest(contract)
    return contract


def _buffer(name, shape, dtype, kind="intermediate"):
    return {"id": name, "shape": shape, "dtype": dtype, "kind": kind, "arg_index": None}


def _conv_node(region, inputs, output, path="direct_contraction", role="contraction"):
    provenance = {
        "prov._pattern_hint": "conv2d",
        "prov.conv_path": path,
        "prov.family": "contraction",
        "prov.op": "conv2d",
        "prov.region_id": region,
    }
    if role is not None:
        provenance["prov.role"] = role
    return {
        "kind": "dispatch",
        "op": f"kernel_{region}",
        "inputs": inputs,
        "outputs": [output],
        "captures": [],
        "regions": 0,
        "prov": provenance,
    }


def _placement(source, region, parallel, reduction):
    return {
        "ordinal": source,
        "region": region,
        "op": "linalg.generic",
        "source_op_index": source,
        "mac_status": "derived",
        "mac_basis": "one proved yielded MAC per static affine-domain point",
        "parallel": parallel,
        "reduction": reduction,
        "macs": __import__("math").prod([*parallel, *reduction]),
        "lane": "on_mesh",
    }


def _analysis(candidate_sha256, workload):
    nodes = [
        _conv_node("direct", ["a0", "w0"], "o0"),
        _conv_node("streamed", ["a1", "w1"], "o1"),
        _conv_node("native", ["a2", "w2"], "o2"),
        _conv_node("materialized", ["a3"], "window", path="im2col_matmul", role=None),
        {
            "kind": "view",
            "op": "tensor.collapse_shape",
            "inputs": ["window"],
            "outputs": ["flat_window"],
            "captures": [],
            "regions": 0,
            "prov": {},
        },
        _conv_node("materialized", ["w3", "flat_window"], "o3", path="im2col_matmul"),
        {
            "kind": "dispatch",
            "op": "opaque_conv",
            "inputs": ["a4"],
            "outputs": ["o4"],
            "captures": [],
            "regions": 0,
            "prov": {"prov._pattern_hint": "conv2d", "prov.family": "contraction"},
        },
        _conv_node("refused", ["a5", "w5"], "o5"),
    ]
    buffers = {
        "a0": _buffer("a0", [1, 3, 8, 8], "i8", "arg"),
        "w0": _buffer("w0", [4, 3, 1, 1], "i8", "arg"),
        "o0": _buffer("o0", [1, 4, 8, 8], "i32"),
        "a1": _buffer("a1", [1, 3, 6, 6], "i8", "arg"),
        "w1": _buffer("w1", [4, 3, 3, 3], "i8", "arg"),
        "o1": _buffer("o1", [1, 4, 4, 4], "i32"),
        "a2": _buffer("a2", [1, 3, 6, 6], "i8", "arg"),
        "w2": _buffer("w2", [4, 3, 3, 3], "i8", "arg"),
        "o2": _buffer("o2", [1, 4, 4, 4], "i32"),
        "a3": _buffer("a3", [1, 3, 6, 6], "f32", "arg"),
        "window": _buffer("window", [3, 3, 3, 1, 4, 4], "f32"),
        "flat_window": _buffer("flat_window", [27, 16], "i8"),
        "w3": _buffer("w3", [4, 27], "i8", "arg"),
        "o3": _buffer("o3", [4, 16], "i32"),
        "a4": _buffer("a4", [1, 1, 1, 1], "i8", "arg"),
        "o4": _buffer("o4", [1, 1, 1, 1], "i8"),
        "a5": _buffer("a5", [1, 3, 6, 6], "i8", "arg"),
        "w5": _buffer("w5", [4, 3, 3, 3], "i8", "arg"),
        "o5": _buffer("o5", [1, 4, 4, 4], "i32"),
    }
    program = {"entry": "entry", "args": [], "buffers": buffers, "nodes": nodes, "results": ["o3"]}
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
            "task_index": 0,
            "declared_task_kind": "convolution",
            "source_op_indices": [0],
            "static_operation_counts": {},
            "owned_instruction_payload_sha256": "9" * 64,
        },
        {
            "task_index": 1,
            "declared_task_kind": "convolution",
            "source_op_indices": [1],
            "static_operation_counts": {"llvm.load": 4, "llvm.store": 4, "llvm.udiv": 8, "llvm.urem": 8},
            "owned_instruction_payload_sha256": "a" * 64,
        },
        {
            "task_index": 2,
            "declared_task_kind": "convolution",
            "source_op_indices": [2],
            "static_operation_counts": {},
            "owned_instruction_payload_sha256": "b" * 64,
        },
        {
            "task_index": 3,
            "declared_task_kind": "host",
            "source_op_indices": [3, 4],
            "static_operation_counts": {},
            "owned_instruction_payload_sha256": "c" * 64,
        },
        {
            "task_index": 4,
            "declared_task_kind": "contraction",
            "source_op_indices": [5],
            "static_operation_counts": {},
            "owned_instruction_payload_sha256": "d" * 64,
        },
        {
            "task_index": 5,
            "declared_task_kind": "host",
            "source_op_indices": [6],
            "static_operation_counts": {},
            "owned_instruction_payload_sha256": "e" * 64,
        },
        {
            "task_index": 6,
            "declared_task_kind": "convolution",
            "source_op_indices": [7],
            "static_operation_counts": {},
            "owned_instruction_payload_sha256": "f" * 64,
        },
    ]
    for task in tasks:
        task.update(
            instruction_indices=[],
            class_counts={},
            role_counts={},
            unknown_instruction_indices=[],
            instructions_without_decoded_fields=[],
            instructions_without_target_roles=[],
            descriptor_semantics="UNVERIFIED",
            classification_coverage="partial",
        )
    placements = [
        _placement(0, "direct", [1, 4, 8, 8], [3, 1, 1]),
        _placement(1, "streamed", [1, 4, 4, 4], [3, 3, 3]),
        _placement(2, "native", [1, 4, 4, 4], [3, 3, 3]),
        _placement(5, "materialized", [4, 16], [27]),
        _placement(7, "refused", [1, 4, 4, 4], [3, 3, 3]),
    ]
    # Make the last source site internally contradictory so it must be refused.
    placements[-1]["reduction"] = [3, 3]
    routes = [
        {
            "index": 1,
            "opcode": "CONV2D",
            "operands": {"ifm": "x", "weight": "w", "dst": "y"},
            "attributes": {"layout": "streamed_row_im2col"},
        },
        {
            "index": 3,
            "opcode": "CONV2D",
            "operands": {"ifm": "x", "weight": "w", "dst": "y"},
            "attributes": {"layout": "streamed_row_im2col"},
        },
        {
            "index": 5,
            "opcode": "CONV2D",
            "operands": {"ifm": "x", "weight": "w", "dst": "y"},
            "attributes": {"layout": "native_affine_window"},
        },
        {
            "index": 7,
            "opcode": "CONV2D",
            "operands": {"ifm": "x", "weight": "w", "dst": "y"},
            "attributes": {"layout": "streamed_row_im2col"},
        },
    ]
    plan = {
        "status": "verified",
        "source_operations": len(nodes),
        "tasks": len(tasks),
        "candidate_sha256": candidate_sha256,
        "source_sha256": pins["source_sha256"],
        "candidate_lowered_sha256": pins["lowered_sha256"],
        "candidate_command_buffer_sha256": pins["command_buffer_sha256"],
        "logical_dispatch_digest": logical_digest,
        "plan_digest": pins["plan_digest"],
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
                "schema": "captured_global_graph_v1",
                "status": "verified",
                "source_sha256": pins["source_sha256"],
                "logical_dispatch_digest": logical_digest,
                "nodes": len(nodes),
                "dispatch_program": program,
            },
            "task_instruction_evidence": {
                "candidate": {
                    "schema": "task_instruction_evidence_v1",
                    "status": "static_ownership_verified",
                    "declared_source_plan_status": "verified",
                    "binding": pins,
                    "tasks": tasks,
                }
            },
            "model_contraction_placement": {
                "candidate": {
                    "schema": "model_contraction_placement_v1",
                    "status": "complete",
                    "contraction_count": len(placements),
                    "contractions": placements,
                    "unresolved": [],
                    "conflicting_regions": [],
                }
            },
            "arms": {
                "candidate": {
                    "representation_activity": {
                        "schema": "command_buffer_representation_activity_v1",
                        "representation_directives": routes,
                    }
                }
            },
        },
    }


def _record(candidate, contract):
    candidate_sha256 = hash_tree(candidate)["sha256"]
    identity = {
        "analysis": "full_graph_compile_and_static_only",
        "capsule": "opaque-member",
        "capsule_sha256": "0" * 64,
        "full_model_simulation_allowed": False,
        "required_lanes": [],
        "required_tiers": [],
        "role": "primary",
    }
    workload = {key: identity[key] for key in ("capsule", "capsule_sha256", "required_lanes", "required_tiers")}
    analysis = _analysis(candidate_sha256, workload)
    analysis["compiler_edit_scope"]["contract_sha256"] = contract["sha256"]
    portfolio_identity = {
        "schema": "full_model_optimization_portfolio_v1",
        "members": [identity],
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
        "schema": "global_perf_iteration_v1",
        "candidate_sha256": candidate_sha256,
        "analysis": analysis,
        "portfolio": {
            "schema": "full_model_portfolio_iteration_v1",
            "candidate_sha256": candidate_sha256,
            "full_model_simulation_allowed": False,
            "selection": "multi_model_pareto_without_invented_static_cycle_total",
            "portfolio_sha256": _digest(portfolio_identity),
            "members_total": 1,
            "members_ready": 1,
            "members": [{"identity": identity, "analysis_ref": "/analysis", "status": "completed"}],
        },
        "cross_run_static_analysis_binding": {"compiler_edit_authority": authority},
    }


def test_inventory_distinguishes_all_window_route_classes(tmp_path):
    candidate = _candidate(tmp_path)
    result = inventory_portfolio_convolution_windows(_record(candidate, _contract()))
    assert result["status"] == "ready_for_work_order"
    member = result["members"][0]
    assert [row["source_operation_id"] for row in member["already_direct"]] == [0]
    assert [row["source_operation_id"] for row in member["streamed"]] == [1]
    assert [row["source_operation_id"] for row in member["already_native"]] == [2]
    assert [row["source_operation_id"] for row in member["materialized"]] == [5]
    assert [row["source_operation_id"] for row in member["uncaptured"]] == [6]
    assert [row["source_operation_id"] for row in member["refused"]] == [7]
    assert member["source_operation_ids"] == [1, 3, 4, 5]
    assert member["streamed"][0]["geometry"]["window_row_instances"] == 4
    assert member["materialized"][0]["dependency_path"] == [3, 4, 5]


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("geometry", "geometry"),
        ("buffer", "buffer"),
        ("ownership", "ownership"),
        ("plan", "plan"),
    ],
)
def test_site_assignment_fails_closed_without_complete_evidence(tmp_path, mutation, expected):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract())
    analysis = record["analysis"]
    if mutation == "geometry":
        analysis["diagnostics"]["model_contraction_placement"]["candidate"]["contractions"][1]["parallel"] = None
    elif mutation == "buffer":
        del analysis["diagnostics"]["captured_logical_graph"]["dispatch_program"]["buffers"]["a1"]
        program = analysis["diagnostics"]["captured_logical_graph"]["dispatch_program"]
        digest = _digest(program)
        analysis["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"] = digest
        analysis["diagnostics"]["verified_global_plan_emission"]["logical_dispatch_digest"] = digest
        analysis["diagnostics"]["task_instruction_evidence"]["candidate"]["binding"]["logical_dispatch_digest"] = digest
    elif mutation == "ownership":
        analysis["diagnostics"]["task_instruction_evidence"]["candidate"]["tasks"][1]["source_op_indices"] = []
    else:
        analysis["diagnostics"]["verified_global_plan_emission"]["status"] = "unverified"
    result = inventory_portfolio_convolution_windows(record)
    member = result["members"][0]
    assert 1 not in member["source_operation_ids"]
    evidence = json.dumps(member, sort_keys=True)
    assert expected in evidence


def test_builder_emits_validator_accepted_single_mechanism_work_order(tmp_path):
    candidate = _candidate(tmp_path)
    contract = _contract()
    record = _record(candidate, contract)
    candidate_sha = hash_tree(candidate)["sha256"]
    documents = build_convolution_window_mechanism_documents(
        record, candidate=candidate.resolve(), expected_candidate_sha256=candidate_sha, iteration_record_sha256="c" * 64
    )
    validate_edit_contract(documents["edit_contract"], candidate)
    validate_mechanism_catalog(documents["catalog"], candidate, documents["edit_contract"])
    assert [row["id"] for row in documents["catalog"]["mechanisms"]] == [MECHANISM_ID]
    work = documents["work_order"]
    assert work["source_operation_ids"] == []
    assert work["portfolio_site_bindings"][0]["source_operation_ids"] == [1, 3, 4, 5]

    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    global_experiment = importlib.import_module("run_global_perf_experiment")
    validator = object.__new__(global_experiment.GlobalPerfExperiment)
    validator.mechanism_catalog = documents["catalog"]
    validator.edit_contract = documents["edit_contract"]
    portfolio = {
        "schema": "full_model_optimization_portfolio_v1",
        "members": [record["portfolio"]["members"][0]["identity"]],
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }
    validator.portfolio_identity = portfolio
    validator.portfolio_identity_sha256 = _digest(portfolio)
    assert validator._validate_mechanism_work_order(work, candidate_sha256=candidate_sha) == work


def test_cli_writes_readonly_content_addressed_artifacts_once(tmp_path, capsys):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract())
    iteration = tmp_path / "iteration.json"
    iteration.write_text(json.dumps(record, sort_keys=True) + "\n")
    iteration.chmod(0o444)
    raw_sha = hashlib.sha256(iteration.read_bytes()).hexdigest()
    candidate_sha = hash_tree(candidate)["sha256"]
    output = tmp_path / "sealed"
    args = [
        str(iteration.resolve()),
        str(candidate.resolve()),
        str(output.resolve()),
        "--iteration-sha256",
        raw_sha,
        "--candidate-sha256",
        candidate_sha,
    ]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["streamed_counts"] == [1]
    assert report["materialized_counts"] == [1]
    for name in (
        "compiler_edit_contract.json",
        "mechanism_catalog.json",
        "mechanism_work_order.json",
        "source_site_inventory.json",
        "receipt.json",
    ):
        path = output / name
        assert path.is_file() and path.stat().st_mode & 0o222 == 0
        assert report["artifacts"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        main(args)
