from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root
from merlin.perf.compiler_edit_scope import validate_edit_contract, validate_mechanism_catalog
from merlin.perf.narrow_epilogue_work_order import (
    MECHANISM_ID,
    build_narrow_epilogue_mechanism_documents,
    inventory_portfolio_narrow_epilogues,
    main,
)


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    source = candidate / "mlir_oot/compiler.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def epilogue_semantics():\n    return None\n\n"
        "def target_readout():\n    return None\n\n"
        "class TargetReadoutIR:\n    pass\n\n"
        "def final_target_emission():\n    return None\n\n"
        "class HostPointwise:\n    pass\n\n"
        "def global_partition():\n    return None\n\n"
        "def contraction_placement():\n    return None\n"
    )
    return candidate


def _contract() -> dict:
    rows = [
        ("epilogue_semantics", "epilogue_semantics"),
        ("target_readout", "target_readout"),
        ("target_readout_ir", "TargetReadoutIR"),
        ("final_target_emission", "final_target_emission"),
        ("host_pointwise_and_residual", "HostPointwise"),
        ("global_partition", "global_partition"),
        ("contraction_placement", "contraction_placement"),
    ]
    contract = {
        "schema": "compiler_edit_contract_v1",
        "authority": "host-frozen test authority",
        "enforcement": "host gate",
        "existing_symbols": [
            {
                "surface_id": surface,
                "path": "mlir_oot/compiler.py",
                "symbol": symbol,
                "kind": "class" if symbol in {"TargetReadoutIR", "HostPointwise"} else "function",
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


def _buffer(name, dtype, *, shape=(4, 4), encoding="dense", layout="row_major",
            kind="intermediate"):
    return {
        "id": name,
        "shape": list(shape),
        "dtype": dtype,
        "kind": kind,
        "encoding": encoding,
        "layout": layout,
    }


def _root(index: int, output: str, *, dtype="i32") -> dict:
    return {
        "kind": "dispatch",
        "op": f"opaque_contraction_{index}",
        "inputs": [f"a{index}", f"b{index}"],
        "outputs": [output],
        "regions": 0,
        "captures": [],
        "prov": {
            "prov.family": "contraction",
            "prov.op": "opaque_contraction",
            "prov.region_id": f"r{index}",
            "prov.role": "contraction",
        },
    }


def _stage(index: int, stage: str, source: str, output: str, *, extras=(), roles=(),
           operation=None) -> dict:
    return {
        "kind": "dispatch",
        "op": f"opaque_stage_{index}",
        "inputs": [source, *extras],
        "outputs": [output],
        "regions": 0,
        "captures": [],
        "prov": {
            "prov.family": "elementwise",
            "prov.op": operation or stage,
            "prov.region_id": f"e{index}",
            "prov.epilogue_stage": stage,
            "prov.epilogue_operation": operation or stage,
            "prov.epilogue_operand_roles": list(roles),
        },
    }


def _record(candidate: Path, contract: dict) -> dict:
    # Six independent accelerator roots exercise the mutually-exclusive inventory outcomes.
    nodes = [
        _root(0, "acc0"),
        _stage(1, "acc_scale", "acc0", "scaled0", extras=("scale0",), roles=("scale",)),
        _stage(2, "requant", "scaled0", "out0"),
        _root(3, "out3", dtype="i8"),
        _root(4, "acc4"),
        _stage(5, "bias", "acc4", "biased4", extras=("bias4",), roles=("bias",)),
        _stage(6, "requant", "biased4", "out4"),
        _root(7, "acc7"),
        _stage(8, "bias", "acc7", "res7", extras=("skip7",), roles=("bias",),
               operation="add"),
        _stage(9, "requant", "res7", "out7"),
        _root(10, "acc10"),
        {
            "kind": "dispatch", "op": "opaque_cast", "inputs": ["acc10"],
            "outputs": ["float10"], "regions": 0, "captures": [],
            "prov": {"prov.family": "cast", "prov.op": "dtype_cast",
                     "prov.region_id": "f10"},
        },
        _root(12, "acc12"),
        {
            "kind": "view", "op": "opaque_unknown", "inputs": ["acc12"],
            "outputs": ["wide12"], "regions": 0, "captures": [], "prov": {},
        },
    ]
    buffers = {}
    for root_index in (0, 3, 4, 7, 10, 12):
        buffers[f"a{root_index}"] = _buffer(f"a{root_index}", "i8", kind="arg")
        buffers[f"b{root_index}"] = _buffer(f"b{root_index}", "i8", kind="arg")
    buffers.update({
        "acc0": _buffer("acc0", "i32", encoding="accumulator"),
        "scale0": _buffer("scale0", "i32", shape=(), kind="const"),
        "scaled0": _buffer("scaled0", "i32", encoding="accumulator"),
        "out0": _buffer("out0", "i8"),
        "out3": _buffer("out3", "i8"),
        "acc4": _buffer("acc4", "i32", encoding="accumulator"),
        "bias4": _buffer("bias4", "i32", shape=(4,), kind="arg"),
        "biased4": _buffer("biased4", "i32", encoding="accumulator"),
        "out4": _buffer("out4", "i8"),
        "acc7": _buffer("acc7", "i32", encoding="accumulator"),
        "skip7": _buffer("skip7", "i32", kind="arg"),
        "res7": _buffer("res7", "i32", encoding="accumulator"),
        "out7": _buffer("out7", "i8"),
        "acc10": _buffer("acc10", "i32", encoding="accumulator"),
        "float10": _buffer("float10", "f32"),
        "acc12": _buffer("acc12", "i32", encoding="accumulator"),
        "wide12": _buffer("wide12", "i32", encoding="accumulator"),
    })
    program = {"entry": "entry", "args": [], "buffers": buffers, "nodes": nodes,
               "results": ["out0", "out3", "out4", "out7", "float10", "wide12"]}
    candidate_sha = hash_tree(candidate)["sha256"]
    logical_digest = _digest(program)
    pins = {
        "source_sha256": "1" * 64,
        "lowered_sha256": "2" * 64,
        "command_buffer_sha256": "3" * 64,
        "compiler_sha256": candidate_sha,
        "logical_dispatch_digest": logical_digest,
        "plan_digest": "6" * 64,
        "target_facts_sha256": "7" * 64,
        "host_verifier_policy_sha256": "8" * 64,
    }
    accelerator = [0, 3, 4, 7, 10, 12]
    host = sorted(set(range(len(nodes))) - set(accelerator))
    tasks = [
        {"task_index": 0, "declared_task_kind": "contraction",
         "source_op_indices": accelerator, "instruction_indices": [0],
         "owned_instruction_payload_sha256": "9" * 64},
        {"task_index": 1, "declared_task_kind": "host",
         "source_op_indices": host, "instruction_indices": [1],
         "owned_instruction_payload_sha256": "a" * 64},
    ]
    placements = [{
        "ordinal": ordinal,
        "region": f"r{source_index}",
        "op": "linalg.generic",
        "source_op_index": source_index,
        "mac_status": "derived",
        "mac_basis": "one proved yielded MAC per static affine-domain point",
        "parallel": [4, 4],
        "reduction": [4],
        "macs": 64,
        "lane": "engine",
    } for ordinal, source_index in enumerate(accelerator)]
    plan = {
        "status": "verified", "source_operations": len(nodes), "tasks": len(tasks),
        "candidate_sha256": candidate_sha, "source_sha256": pins["source_sha256"],
        "candidate_lowered_sha256": pins["lowered_sha256"],
        "candidate_command_buffer_sha256": pins["command_buffer_sha256"],
        "logical_dispatch_digest": logical_digest, "plan_digest": pins["plan_digest"],
    }
    identity = {
        "analysis": "full_graph_compile_and_static_only",
        "capsule": "opaque-member",
        "capsule_sha256": "b" * 64,
        "full_model_simulation_allowed": False,
        "required_lanes": [],
        "required_tiers": [],
        "role": "primary",
    }
    workload = {field: identity[field] for field in
                ("capsule", "capsule_sha256", "required_lanes", "required_tiers")}
    analysis = {
        "schema": "host_owned_whole_model_emission_analysis_v2",
        "candidate_sha256": candidate_sha,
        "compiler_edit_scope": {"status": "allowed", "contract_sha256": contract["sha256"]},
        "workload": workload,
        "emission": {"candidate_command_buffer_sha256": pins["command_buffer_sha256"],
                     "candidate_lowered_sha256": pins["lowered_sha256"]},
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
                "contraction_count": len(placements), "contractions": placements,
                "unresolved": [], "conflicting_regions": [],
            }},
        },
    }
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
        "initial_candidate_sha256": candidate_sha,
        "source_pins_checked": True,
    }
    return {
        "schema": "global_perf_iteration_v1", "candidate_sha256": candidate_sha,
        "analysis": analysis,
        "portfolio": {
            "schema": "full_model_portfolio_iteration_v1",
            "candidate_sha256": candidate_sha,
            "full_model_simulation_allowed": False,
            "selection": "multi_model_pareto_without_invented_static_cycle_total",
            "portfolio_sha256": _digest(portfolio_identity),
            "members_total": 1, "members_ready": 1,
            "members": [{"identity": identity, "analysis_ref": "/analysis",
                         "status": "completed"}],
        },
        "cross_run_static_analysis_binding": {
            "target_descriptor_sha256": "c" * 64,
            "compiler_edit_authority": authority,
        },
    }


def _capability(record: dict, form: dict) -> dict:
    proof = {
        "id": "opaque-capability-form",
        "status": "verified",
        "form": copy.deepcopy(form),
        "form_sha256": _digest(form),
        "evidence": [{
            "kind": "executed_conformance",
            "locator": "immutable://opaque/exact-form",
            "sha256": "d" * 64,
            "scope": "exact stage operands, encoding, narrow layout and completion",
        }],
    }
    document = {
        "schema": "target_narrow_epilogue_capability_evidence_v1",
        "status": "verified",
        "target_descriptor_sha256": record["cross_run_static_analysis_binding"][
            "target_descriptor_sha256"],
        "target_instruction_facts_sha256": record["analysis"]["diagnostics"][
            "task_instruction_evidence"]["candidate"]["binding"]["target_facts_sha256"],
        "proofs": [proof],
    }
    document["sha256"] = _digest(document)
    return document


def test_inventory_distinguishes_every_narrow_readout_outcome(tmp_path):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract())
    without_capability = inventory_portfolio_narrow_epilogues(record)
    initial = without_capability["members"][0]
    assert [row["producer_source_operation_id"] for row in initial["missing_capability"]] == [0, 4]
    evidence = _capability(record, initial["missing_capability"][0]["capability_form"])

    result = inventory_portfolio_narrow_epilogues(
        record, capability_evidence=evidence,
        capability_evidence_file_sha256="e" * 64,
        iteration_record_sha256="f" * 64,
    )
    assert result["status"] == "ready_for_work_order"
    member = result["members"][0]
    assert [row["producer_source_operation_id"] for row in member["eligible"]] == [0]
    assert [row["producer_source_operation_id"] for row in member["already_narrow"]] == [3]
    assert [row["producer_source_operation_id"] for row in member["missing_capability"]] == [4]
    assert [row["producer_source_operation_id"]
            for row in member["residual_second_operand"]] == [7]
    assert [row["producer_source_operation_id"]
            for row in member["float_or_unsupported_stage"]] == [10]
    assert [row["producer_source_operation_id"] for row in member["uncaptured"]] == [12]
    assert member["ownership_failures"] == []
    assert member["source_operation_ids"] == [0, 1, 2]
    site = member["eligible"][0]
    assert site["capability_proof_id"] == "opaque-capability-form"
    assert site["capability_form"]["stage_sequence"] == ["acc_scale", "requant"]
    assert site["capability_form"]["output"] == {
        "dtype": "i8", "encoding": "dense", "layout": "row_major", "rank": 2,
        "width_bits": 8,
    }


def test_capability_must_bind_exact_target_facts_form_and_completion(tmp_path):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract())
    inventory = inventory_portfolio_narrow_epilogues(record)
    form = inventory["members"][0]["missing_capability"][0]["capability_form"]
    evidence = _capability(record, form)

    for mutation in ("target", "facts", "layout", "operand", "width", "completion",
                     "citation", "raw_pin"):
        changed = copy.deepcopy(evidence)
        raw_pin = "e" * 64
        if mutation == "target":
            changed["target_descriptor_sha256"] = "0" * 64
        elif mutation == "facts":
            changed["target_instruction_facts_sha256"] = "0" * 64
        elif mutation == "layout":
            changed["proofs"][0]["form"]["output"]["layout"] = "other"
        elif mutation == "operand":
            changed["proofs"][0]["form"]["stages"][0]["inputs"].pop()
        elif mutation == "width":
            changed["proofs"][0]["form"]["output"]["width_bits"] = 16
        elif mutation == "completion":
            changed["proofs"][0]["form"]["completion"] = "issued_without_completion"
        elif mutation == "citation":
            changed["proofs"][0]["evidence"] = []
        else:
            raw_pin = "not-a-pin"
        if mutation not in {"target", "facts", "citation", "raw_pin"}:
            changed["proofs"][0]["form_sha256"] = _digest(changed["proofs"][0]["form"])
        if mutation != "raw_pin":
            changed["sha256"] = _digest({key: value for key, value in changed.items()
                                         if key != "sha256"})
        result = inventory_portfolio_narrow_epilogues(
            record, capability_evidence=changed,
            capability_evidence_file_sha256=raw_pin)
        assert result["status"] == "not_ready"
        assert result["problems"]


def test_incomplete_source_ownership_is_a_named_failure_not_an_opportunity(tmp_path):
    candidate = _candidate(tmp_path)
    record = _record(candidate, _contract())
    record["analysis"]["diagnostics"]["task_instruction_evidence"]["candidate"]["tasks"][0][
        "source_op_indices"].remove(0)
    result = inventory_portfolio_narrow_epilogues(record)
    assert result["status"] == "not_ready"
    member = result["members"][0]
    assert member["eligible"] == []
    assert member["ownership_failures"]
    assert {row["producer_source_operation_id"] for row in member["ownership_failures"]} >= {0}


def test_builder_emits_validator_accepted_one_mechanism_documents(tmp_path):
    candidate = _candidate(tmp_path)
    contract = _contract()
    record = _record(candidate, contract)
    initial = inventory_portfolio_narrow_epilogues(record)["members"][0]
    evidence = _capability(record, initial["missing_capability"][0]["capability_form"])
    documents = build_narrow_epilogue_mechanism_documents(
        record,
        candidate=candidate.resolve(),
        expected_candidate_sha256=hash_tree(candidate)["sha256"],
        iteration_record_sha256="f" * 64,
        capability_evidence=evidence,
        capability_evidence_file_sha256="e" * 64,
    )
    validate_edit_contract(documents["edit_contract"], candidate)
    validate_mechanism_catalog(documents["catalog"], candidate, documents["edit_contract"])
    assert documents["catalog"]["mechanisms"] == [{
        "id": MECHANISM_ID,
        "selectors": documents["catalog"]["mechanisms"][0]["selectors"],
    }]
    scripts = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
    sys.path.insert(0, str(scripts))
    os.environ.setdefault("MERLIN_EXT_CHIPYARD", "/nonexistent/test-chipyard")
    global_experiment = importlib.import_module("run_global_perf_experiment")
    validator = object.__new__(global_experiment.GlobalPerfExperiment)
    validator.mechanism_catalog = documents["catalog"]
    validator.edit_contract = documents["edit_contract"]
    validator.portfolio_identity = {
        "schema": "full_model_optimization_portfolio_v1",
        "members": [record["portfolio"]["members"][0]["identity"]],
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }
    validator.portfolio_identity_sha256 = _digest(validator.portfolio_identity)
    validated = validator._validate_mechanism_work_order(
        documents["work_order"], candidate_sha256=hash_tree(candidate)["sha256"])
    assert validated == documents["work_order"]
    assert validated["portfolio_site_bindings"][0]["source_operation_ids"] == [0, 1, 2]


def test_cli_requires_readonly_pins_and_writes_once(tmp_path, capsys):
    candidate = _candidate(tmp_path)
    contract = _contract()
    record = _record(candidate, contract)
    initial = inventory_portfolio_narrow_epilogues(record)["members"][0]
    evidence = _capability(record, initial["missing_capability"][0]["capability_form"])
    iteration = tmp_path / "iteration.json"
    capability = tmp_path / "capability.json"
    iteration.write_text(json.dumps(record, sort_keys=True) + "\n")
    capability.write_text(json.dumps(evidence, sort_keys=True) + "\n")
    iteration.chmod(0o444)
    capability.chmod(0o444)
    output = tmp_path / "sealed"
    args = [
        str(iteration.resolve()), str(candidate.resolve()), str(capability.resolve()),
        str(output.resolve()),
        "--iteration-sha256", hashlib.sha256(iteration.read_bytes()).hexdigest(),
        "--candidate-sha256", hash_tree(candidate)["sha256"],
        "--capability-sha256", hashlib.sha256(capability.read_bytes()).hexdigest(),
    ]
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["eligible_counts"] == [1]
    for name in ("compiler_edit_contract.json", "mechanism_catalog.json",
                 "mechanism_work_order.json", "source_site_inventory.json", "receipt.json"):
        path = output / name
        assert path.is_file() and path.stat().st_mode & 0o222 == 0
        assert report["artifacts"][name]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(SystemExit):
        main(args)
