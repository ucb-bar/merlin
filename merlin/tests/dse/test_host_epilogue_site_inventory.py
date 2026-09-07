from __future__ import annotations

import copy
import hashlib
import json

from merlin.perf.host_epilogue_site_inventory import (
    inventory_host_epilogue_sites,
    inventory_portfolio_host_epilogue_sites,
    main,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _record():
    pins = {
        "source_sha256": "1" * 64,
        "lowered_sha256": "2" * 64,
        "command_buffer_sha256": "3" * 64,
        "compiler_sha256": "4" * 64,
        "logical_dispatch_digest": "",
        "plan_digest": "6" * 64,
        "target_facts_sha256": "7" * 64,
        "host_verifier_policy_sha256": "8" * 64,
    }
    nodes = [
        {"kind": "view", "op": "tensor.empty", "inputs": [], "outputs": ["b1"],
         "prov": {}, "regions": 0, "captures": []},
        {"kind": "view", "op": "arith.constant", "inputs": [], "outputs": ["b2"],
         "prov": {}, "regions": 0, "captures": []},
        {"kind": "view", "op": "tensor.splat", "inputs": ["b2"], "outputs": ["b3"],
         "prov": {}, "regions": 0, "captures": []},
        {"kind": "dispatch", "op": "kernel_a", "inputs": ["b0", "b3"], "outputs": ["b4"],
         "prov": {"prov.family": "quantize", "prov.op": "quantize_per_tensor",
                  "prov.region_id": "q0"}, "regions": 0, "captures": []},
        {"kind": "dispatch", "op": "kernel_b", "inputs": ["b4"], "outputs": ["b5"],
         "prov": {"prov.family": "elementwise", "prov.op": "add", "prov.region_id": "a0"},
         "regions": 0, "captures": []},
        {"kind": "dispatch", "op": "kernel_c", "inputs": ["b5"], "outputs": ["b6"],
         "prov": {"prov.family": "contraction", "prov.op": "matmul", "prov.region_id": "m0"},
         "regions": 0, "captures": []},
    ]
    buffers = {
        name: {"id": name, "shape": [4], "dtype": "i8", "kind": kind}
        for name, kind in {
            "b0": "arg", "b1": "intermediate", "b2": "const", "b3": "intermediate",
            "b4": "intermediate", "b5": "intermediate", "b6": "intermediate",
        }.items()
    }
    program = {"entry": "forward", "args": [0], "buffers": buffers,
               "nodes": nodes, "results": ["b6"]}
    pins["logical_dispatch_digest"] = _digest(program)
    tasks = [
        {"task_index": 0, "declared_task_kind": "host", "source_op_indices": [0, 1, 2, 3, 4]},
        {"task_index": 1, "declared_task_kind": "matrix", "source_op_indices": [5]},
    ]
    plan = {
        "status": "verified", "source_operations": len(nodes), "tasks": len(tasks),
        "candidate_sha256": pins["compiler_sha256"],
        "source_sha256": pins["source_sha256"],
        "candidate_lowered_sha256": pins["lowered_sha256"],
        "candidate_command_buffer_sha256": pins["command_buffer_sha256"],
        "logical_dispatch_digest": pins["logical_dispatch_digest"],
        "plan_digest": pins["plan_digest"],
    }
    analysis = {
        "schema": "host_owned_whole_model_emission_analysis_v2",
        "candidate_sha256": pins["compiler_sha256"],
        "emission": {
            "candidate_command_buffer_sha256": pins["command_buffer_sha256"],
            "candidate_lowered_sha256": pins["lowered_sha256"],
        },
        "diagnostics": {
            "verified_global_plan_emission": plan,
            "captured_logical_graph": {
                "schema": "captured_global_graph_v1", "status": "verified",
                "source_sha256": pins["source_sha256"],
                "logical_dispatch_digest": pins["logical_dispatch_digest"],
                "nodes": len(nodes), "dispatch_program": program,
            },
            "task_instruction_evidence": {"candidate": {
                "schema": "task_instruction_evidence_v1", "status": "static_ownership_verified",
                "declared_source_plan_status": "verified", "binding": pins, "tasks": tasks,
            }},
        },
    }
    return {"schema": "global_perf_iteration_v1", "candidate_sha256": pins["compiler_sha256"],
            "analysis": analysis}


def test_inventory_binds_exact_one_use_host_chain_without_workload_names():
    result = inventory_host_epilogue_sites(_record())
    assert result["status"] == "ready_for_source_site_binding"
    assert result["source_operation_ids"] == [2, 3, 4]
    assert len(result["chains"]) == 1
    chain = result["chains"][0]
    assert chain["pointwise_source_operation_ids"] == [3, 4]
    assert chain["materialization_source_operation_ids"] == [2]
    assert [(edge["producer"], edge["consumer"]) for edge in chain["one_use_edges"]] == [
        (2, 3), (3, 4)]
    assert chain["output_boundaries"][0]["consumer_task_indices"] == [1]
    assert any(row["source_operation_id"] == 0 and row["refusal_class"] == "fanout_or_boundary"
               for row in result["refusals"])
    assert set(result["bindings"]) >= {"source_sha256", "plan_digest",
                                       "command_buffer_sha256", "lowered_sha256"}


def test_inventory_refuses_fanout_and_records_every_consumer():
    record = _record()
    program = record["analysis"]["diagnostics"]["captured_logical_graph"]["dispatch_program"]
    program["nodes"][5]["inputs"].append("b4")
    digest = _digest(program)
    graph = record["analysis"]["diagnostics"]["captured_logical_graph"]
    graph["logical_dispatch_digest"] = digest
    plan = record["analysis"]["diagnostics"]["verified_global_plan_emission"]
    plan["logical_dispatch_digest"] = digest
    summary = record["analysis"]["diagnostics"]["task_instruction_evidence"]["candidate"]
    summary["binding"]["logical_dispatch_digest"] = digest

    result = inventory_host_epilogue_sites(record)
    assert result["status"] == "ready_for_source_site_binding"
    assert result["source_operation_ids"] == [2, 3]
    refused = next(row for row in result["refusals"] if row["source_operation_id"] == 4)
    assert refused["refusal_class"] == "fanout_or_boundary"
    assert refused["reasons"] == ["buffer b5 crosses task 0->1"]
    boundary = result["chains"][0]["output_boundaries"][0]
    assert boundary["fanout"] == 2
    assert boundary["consumer_source_operation_ids"] == [4, 5]


def test_inventory_fails_closed_on_hash_drift_or_missing_exact_graph():
    drifted = _record()
    drifted["analysis"]["emission"]["candidate_lowered_sha256"] = "9" * 64
    result = inventory_host_epilogue_sites(drifted)
    assert result["status"] == "not_ready"
    assert result["source_operation_ids"] == []
    assert "analysis emission and plan disagree on lowered_sha256" in result["problems"]

    missing = _record()
    del missing["analysis"]["diagnostics"]["captured_logical_graph"]["dispatch_program"]
    result = inventory_host_epilogue_sites(missing)
    assert result["status"] == "not_ready"
    assert result["missing_fields"] == [
        "analysis.diagnostics.captured_logical_graph.dispatch_program"]


def test_inventory_refuses_unknown_dispatch_semantics_without_guessing_from_symbol():
    record = _record()
    program = record["analysis"]["diagnostics"]["captured_logical_graph"]["dispatch_program"]
    program["nodes"][4]["op"] = "looks_like_relu_but_is_not_evidence"
    program["nodes"][4]["prov"] = {}
    digest = _digest(program)
    graph = record["analysis"]["diagnostics"]["captured_logical_graph"]
    graph["logical_dispatch_digest"] = digest
    plan = record["analysis"]["diagnostics"]["verified_global_plan_emission"]
    plan["logical_dispatch_digest"] = digest
    record["analysis"]["diagnostics"]["task_instruction_evidence"]["candidate"]["binding"][
        "logical_dispatch_digest"] = digest

    result = inventory_host_epilogue_sites(record)
    assert result["status"] == "ready_for_source_site_binding"
    assert result["source_operation_ids"] == [2, 3]
    refused = next(row for row in result["refusals"] if row["source_operation_id"] == 4)
    assert refused == {
        "source_operation_id": 4,
        "refusal_class": "semantic",
        "reasons": ["dispatch lacks both explicit prov.family and prov.op semantic labels"],
    }


def test_inventory_accepts_direct_analysis_record_and_never_reads_model_identity():
    record = _record()
    analysis = copy.deepcopy(record["analysis"])
    analysis["workload"] = {"capsule": "a_name_that_must_not_affect_selection"}
    assert inventory_host_epilogue_sites(analysis)["source_operation_ids"] == [2, 3, 4]


def _portfolio_record():
    record = _record()
    first_identity = {
        "analysis": "full_graph_compile_and_static_only",
        "capsule": "first",
        "capsule_sha256": "a" * 64,
        "full_model_simulation_allowed": False,
        "required_lanes": [],
        "required_tiers": [],
        "role": "primary",
    }
    second_identity = {
        "analysis": "full_graph_compile_and_static_only",
        "capsule": "second",
        "capsule_sha256": "b" * 64,
        "full_model_simulation_allowed": False,
        "required_lanes": ["lane"],
        "required_tiers": ["tier"],
        "role": "training",
    }
    first_workload = {key: first_identity[key] for key in
                      ("capsule", "capsule_sha256", "required_lanes", "required_tiers")}
    second_workload = {key: second_identity[key] for key in
                       ("capsule", "capsule_sha256", "required_lanes", "required_tiers")}
    record["analysis"]["workload"] = first_workload
    second_analysis = copy.deepcopy(record["analysis"])
    second_analysis["workload"] = second_workload
    identities = [first_identity, second_identity]
    portfolio_identity = {
        "schema": "full_model_optimization_portfolio_v1",
        "members": identities,
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }
    record["portfolio"] = {
        "schema": "full_model_portfolio_iteration_v1",
        "candidate_sha256": record["candidate_sha256"],
        "full_model_simulation_allowed": False,
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "portfolio_sha256": _digest(portfolio_identity),
        "members_total": 2,
        "members_ready": 2,
        "members": [
            {"identity": first_identity, "analysis_ref": "/analysis", "status": "completed"},
            {"identity": second_identity, "analysis": second_analysis, "status": "completed"},
        ],
    }
    return record


def test_portfolio_inventory_resolves_primary_alias_and_embedded_members_in_exact_order():
    result = inventory_portfolio_host_epilogue_sites(
        _portfolio_record(), iteration_record_sha256="c" * 64)
    assert result["status"] == "ready_for_portfolio_source_site_binding"
    assert [row["identity"]["capsule"] for row in result["members"]] == ["first", "second"]
    assert [row["analysis_location"] for row in result["members"]] == [
        "/analysis", "/portfolio/members/1/analysis"]
    assert [row["source_operation_ids"] for row in result["members"]] == [
        [2, 3, 4], [2, 3, 4]]
    asserted = result.pop("portfolio_site_inventory_sha256")
    assert asserted == _digest(result)


def test_portfolio_inventory_fails_closed_when_order_or_member_analysis_identity_drifts():
    reordered = _portfolio_record()
    reordered["portfolio"]["members"].reverse()
    result = inventory_portfolio_host_epilogue_sites(reordered)
    assert result["status"] == "not_ready"
    assert result["members"] == []
    assert "ordered member identities do not match the exact portfolio hash" in result["problems"]

    mismatched = _portfolio_record()
    mismatched["portfolio"]["members"][1]["analysis"]["workload"]["capsule"] = "first"
    result = inventory_portfolio_host_epilogue_sites(mismatched)
    assert result["status"] == "not_ready"
    assert result["members"] == []
    assert "portfolio member 1 analysis workload differs from its identity" in result["problems"]


def test_portfolio_inventory_cli_writes_once_and_reports_raw_hash(tmp_path, capsys):
    source = tmp_path / "iteration.json"
    source.write_text(json.dumps(_portfolio_record(), sort_keys=True) + "\n")
    output = tmp_path / "inventory.json"
    assert main([str(source), str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    artifact = json.loads(output.read_text())
    asserted = artifact.pop("portfolio_site_inventory_sha256")
    assert asserted == _digest(artifact)
    try:
        main([str(source), str(output)])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("immutable CLI output was overwritten")
