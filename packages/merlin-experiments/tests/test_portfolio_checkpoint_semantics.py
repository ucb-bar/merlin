"""Installed live/replay member selection uses exact bounded structural evidence."""

import copy
import hashlib
import json
import socket
import subprocess

import pytest
from merlin_experiments.phase2 import contracts
from merlin_experiments.phase2 import portfolio_checkpoint as checkpoint

from merlin.benchharness import hash_tree
from merlin.perf.historical_reference import reference_summary


@pytest.fixture(autouse=True)
def refuse_execution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("portfolio receipt verification must not execute processes or bind sockets")

    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(socket.socket, "bind", refused)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def analysis(candidate, identity, revision, *, payload=0, operations=0, tasks=1):
    emission = {
        "candidate_lowered_sha256": digest(f"lowered:{identity['capsule']}:{revision}"),
        "candidate_command_buffer_sha256": digest(f"commands:{identity['capsule']}:{revision}"),
    }
    plan = {
        "status": "verified",
        "candidate_sha256": candidate,
        "plan_digest": digest(f"plan:{revision}"),
        "logical_dispatch_digest": digest("graph"),
        "source_sha256": digest(identity["capsule"]),
        **emission,
        "tasks": tasks,
        "host_activity": {
            "load_payload_bytes": payload,
            "store_payload_bytes": 0,
            "static_allocation_payload_bytes": 0,
            "dynamic_operations": {"load": operations},
        },
    }
    return {
        "candidate_sha256": candidate,
        "workload": {"capsule_sha256": identity["capsule_sha256"]},
        "emission": emission,
        "diagnostics": {
            "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": digest("graph")},
            "verified_global_plan_emission": plan,
            "arms": {"candidate": {"status": "emitted"}},
        },
    }


def iteration(number, candidate, portfolio, revisions):
    analyses = [
        analysis(candidate, identity, revision)
        for identity, revision in zip(portfolio["members"], revisions, strict=True)
    ]
    dependencies = {"compiler_implementation_sha256": digest("compiler:" + candidate)}
    policy = {
        "candidate_sha256": candidate,
        "compiler_dependencies": dependencies,
        "target_sha256": digest("target"),
        "portfolio_sha256": contracts.document_sha256(portfolio),
    }
    return {
        "schema": "global_perf_iteration_v1",
        "iteration": number,
        "candidate_sha256": candidate,
        "compiler_dependencies": dependencies,
        "readiness": {"status": "ready_for_probe_admission"},
        "analysis": analyses[0],
        "static_comparison": {"previous_iteration": number - 1},
        "analysis_reuse_binding": {
            "schema": "global_static_analysis_reuse_binding_v1",
            **policy,
            "sha256": contracts.document_sha256(policy),
        },
        "portfolio": {
            "portfolio_sha256": contracts.document_sha256(portfolio),
            "candidate_sha256": candidate,
            "members_total": len(analyses),
            "members_ready": len(analyses),
            "members": [
                {
                    "identity": identity,
                    **(
                        {"analysis_ref": "/analysis", "static_comparison_ref": "/static_comparison"}
                        if index == 0
                        else {"analysis": member}
                    ),
                }
                for index, (identity, member) in enumerate(zip(portfolio["members"], analyses, strict=True))
            ],
        },
    }


def contexts(before, after, portfolio):
    return list(
        zip(
            checkpoint.recorded_portfolio_contexts(
                before, portfolio_identity=portfolio, target_sha256=digest("target"), arm="previous"
            ),
            checkpoint.recorded_portfolio_contexts(
                after, portfolio_identity=portfolio, target_sha256=digest("target"), arm="current"
            ),
            strict=True,
        )
    )


@pytest.fixture
def records():
    portfolio = {"members": [{"capsule": name, "capsule_sha256": digest(name)} for name in ("first", "second")]}
    return (
        portfolio,
        iteration(0, digest("before"), portfolio, [0, 0]),
        iteration(1, digest("after"), portfolio, [1, 1]),
    )


def test_selection_known_deltas_are_lexicographic_and_ties_follow_member_order(records):
    portfolio, before, after = records
    pairs = contexts(before, after, portfolio)
    selected = checkpoint.select_changed_portfolio_contexts(pairs)
    assert selected["portfolio_index"] == 0
    assert selected["stable_portfolio_order_tie_break_applied"] is True
    second = after["portfolio"]["members"][1]["analysis"]["diagnostics"]["verified_global_plan_emission"]
    second["host_activity"]["load_payload_bytes"] = 1
    first = after["analysis"]["diagnostics"]["verified_global_plan_emission"]
    first["host_activity"]["dynamic_operations"]["load"] = 100000
    selected = checkpoint.select_changed_portfolio_contexts(contexts(before, after, portfolio))
    assert selected["portfolio_index"] == 1
    assert selected["structural_host_work_delta"]["host_payload_bytes_absolute_delta"] == 1
    assert selected["performance_inference"] == "none"


def test_selection_ignores_plan_only_changes_and_refuses_no_changed_artifacts(records):
    portfolio, before, after = records
    pairs = contexts(before, after, portfolio)
    for previous, current in pairs:
        for field in ("lowered_sha256", "command_buffer_sha256"):
            current["member_binding"][field] = previous["member_binding"][field]
    with pytest.raises(ValueError, match="no portfolio member has a changed emitted artifact"):
        checkpoint.select_changed_portfolio_contexts(pairs)
    pairs[1][1]["member_binding"]["lowered_sha256"] = digest("changed")
    assert checkpoint.select_changed_portfolio_contexts(pairs)["portfolio_index"] == 1


def test_unknown_structural_metrics_do_not_become_zero_or_speedup(records):
    portfolio, before, after = records
    pairs = contexts(before, after, portfolio)
    for pair in pairs:
        for context in pair:
            context["analysis"]["diagnostics"]["verified_global_plan_emission"] = {}
    selected = checkpoint.select_changed_portfolio_contexts(pairs)
    assert selected["portfolio_index"] == 0
    assert selected["known_ranking_metrics"] == []
    assert selected["selection_basis"] == "stable_portfolio_order_no_known_structural_host_work"
    assert selected["structural_host_work_delta"]["host_payload_bytes_absolute_delta"] is None


@pytest.fixture
def semantic_case(tmp_path, records):
    portfolio, before, after = records
    snapshot = tmp_path / "previous_candidate"
    snapshot.mkdir()
    (snapshot / "compiler.py").write_text("def compile():\n    return 1\n")
    before = iteration(0, hash_tree(snapshot)["sha256"], portfolio, [0, 0])
    before["submitted_snapshot"] = str(snapshot)
    for path in (snapshot / "compiler.py", snapshot):
        path.chmod(path.stat().st_mode & ~0o222)
    previous_path = tmp_path / "iteration_0000.json"
    previous_path.write_bytes(contracts.canonical_json(before))
    previous_path.chmod(0o444)
    pairs = contexts(before, after, portfolio)
    selection = checkpoint.select_changed_portfolio_contexts(pairs)
    previous, current = pairs[selection["portfolio_index"]]
    member = {"selection": selection, "previous": previous["member_binding"], "current": current["member_binding"]}
    binding = member["current"]
    semantic = {
        "schema": "global_changed_region_semantic_receipt_v2",
        "iteration": 1,
        "previous_iteration_record": str(previous_path),
        "previous_iteration_record_sha256": contracts.sha256_file(previous_path),
        "previous_portfolio_iteration_sha256": contracts.document_sha256(before["portfolio"]),
        "portfolio_member_binding": member,
        "evidence": {"portfolio_member_binding": copy.deepcopy(member)},
        "binding": {
            "compiler_digest": binding["compiler_implementation_sha256"],
            "target_digest": digest("target"),
            "graph_digest": binding["logical_dispatch_digest"],
            "plan_digest": binding["plan_digest"],
        },
        "previous_artifact_sha256": member["previous"]["lowered_sha256"],
        "current_artifact_sha256": binding["lowered_sha256"],
        "scope": "selected changed mechanism and tested reduced domain only",
        "full_model_numerics_qualified": False,
        "global_speedup_proven": False,
        "full_model_cycles": None,
    }
    return semantic, after, portfolio, tmp_path, snapshot, previous_path


def verify_semantic(case):
    semantic, after, portfolio, root, _, _ = case
    return checkpoint.verify_changed_region_semantic_receipt(
        semantic, iteration=after, portfolio_identity=portfolio, target_sha256=digest("target"), experiment_root=root
    )


def test_semantic_replay_recomputes_live_member_selection(semantic_case):
    assert verify_semantic(semantic_case) == semantic_case[0]["portfolio_member_binding"]


@pytest.mark.parametrize(
    "mutation",
    [
        "selection",
        "member_order",
        "policy",
        "previous_writable",
        "previous_bytes",
        "snapshot_bytes",
        "snapshot_link",
        "legacy",
        "whole_model_claim",
    ],
)
def test_semantic_receipt_refuses_substituted_prior_or_member_evidence(semantic_case, mutation):
    semantic, after, _, _, snapshot, previous_path = semantic_case
    if mutation == "selection":
        semantic["portfolio_member_binding"]["selection"]["portfolio_index"] = 1
    elif mutation == "member_order":
        after["portfolio"]["members"].reverse()
    elif mutation == "policy":
        after["analysis_reuse_binding"]["target_sha256"] = digest("different")
    elif mutation == "previous_writable":
        previous_path.chmod(0o644)
    elif mutation == "previous_bytes":
        previous_path.chmod(0o644)
        previous_path.write_bytes(previous_path.read_bytes() + b"\n")
        previous_path.chmod(0o444)
    elif mutation == "snapshot_bytes":
        source = snapshot / "compiler.py"
        source.chmod(0o644)
        source.write_text("def compile():\n    return 2\n")
        source.chmod(0o444)
    elif mutation == "snapshot_link":
        snapshot.chmod(0o755)
        (snapshot / "linked.py").symlink_to(snapshot / "compiler.py")
        snapshot.chmod(0o555)
    elif mutation == "legacy":
        semantic["schema"] = "global_changed_region_semantic_receipt_v1"
    else:
        semantic["full_model_numerics_qualified"] = True
    with pytest.raises(ValueError):
        verify_semantic(semantic_case)


def test_historical_bundle_keeps_reference_scope_and_exact_digest(tmp_path):
    path = tmp_path / "historical.json"
    bundle = {"schema": "historical_reference_bundle_v1", "records": [], "summary": reference_summary([])}
    raw = contracts.canonical_json(bundle)
    path.write_bytes(raw)
    returned, summary = checkpoint.load_historical_reference(path, contracts.sha256_file(path))
    assert returned == raw
    assert summary["reference_count"] == 0
    assert summary["target_cycle_authority"] is False
    with pytest.raises(ValueError, match="digest changed"):
        checkpoint.load_historical_reference(path, "0" * 64)
    with pytest.raises(ValueError, match="outside candidate writes"):
        checkpoint.load_historical_reference(path, contracts.sha256_file(path), candidate_roots=[tmp_path])


@pytest.mark.parametrize("invalid", ["duplicate", "oversized", "summary"])
def test_historical_bundle_refuses_duplicate_fields_size_and_summary(tmp_path, invalid):
    path = tmp_path / "historical.json"
    raw = b'{"schema":"historical_reference_bundle_v1","records":[],"records":[]}'
    if invalid == "oversized":
        raw = b" " * (16 * 1024 * 1024 + 1)
    elif invalid == "summary":
        raw = json.dumps({"schema": "historical_reference_bundle_v1", "records": [], "summary": {}}).encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        checkpoint.load_historical_reference(path, contracts.sha256_file(path))


@pytest.fixture
def paired_case(records):
    _, _, record = records
    diagnostics = record["analysis"]["diagnostics"]
    artifact = record["analysis"]["emission"]["candidate_lowered_sha256"]
    work = {
        "source_task_index": 0,
        "source_op_indices": [0],
        "timed_command_count": 2,
        "timed_command_multiset": [{"kind": "move", "count": 2}],
    }
    execution = {
        "correct": True,
        "warmup_runs": 1,
        "measured_runs": 1,
        "total_compute_cycles": 10,
        "engine_provenance": {"binary_sha256": digest("engine")},
        "counter_profile": {
            "kind": "joint_engine_busy_cycles",
            "partition_proof": {"status": "proved"},
            "layout": {"complete": True},
            "active_union_cycles": 8,
            "idle_cycles": 2,
            "overlap_any_engine_cycles": 1,
            "busy_cycles_by_engine_token": {"move": 5},
        },
    }
    receipt = {
        "binding": {
            "graph_digest": diagnostics["captured_logical_graph"]["logical_dispatch_digest"],
            "plan_digest": diagnostics["verified_global_plan_emission"]["plan_digest"],
            "compiler_digest": record["compiler_dependencies"]["compiler_implementation_sha256"],
            "target_digest": digest("target"),
        },
        "scope": "controlled_fixed_work_slice",
        "model_artifact_sha256": artifact,
        "global_cost_validated": False,
        "global_speedup_proven": False,
        "projection_proof": {
            "status": "same_work_projection_verified",
            "after_artifact_sha256": artifact,
            "work_contract": work,
            "work_contract_sha256": contracts.document_sha256(work),
        },
        "executions": {"before": copy.deepcopy(execution), "after": copy.deepcopy(execution)},
    }
    return record, receipt


@pytest.mark.parametrize("change", [False, True])
def test_paired_feedback_retains_scoped_evidence_without_global_claim(paired_case, change):
    record, receipt = paired_case
    if change:
        receipt["executions"]["after"]["counter_profile"]["overlap_any_engine_cycles"] = 2
    feedback = checkpoint.paired_context_decision_feedback(record, receipt, target_sha256=digest("target"))
    assert feedback["status"] == (
        "controlled_resource_change_observed" if change else "no_observed_overlap_or_busy_work_change"
    )
    assert feedback["global_speedup_proven"] is False
    assert feedback["full_model_cycles"] is None
    assert feedback["pipeline_projection_admitted"] is False
    assert feedback["buffer_capacity_contract"] == "UNPROVED"


@pytest.mark.parametrize("mutation", ["engine", "partition", "warmup", "bounds"])
def test_paired_missing_counter_evidence_stays_unknown(paired_case, mutation):
    record, receipt = paired_case
    after = receipt["executions"]["after"]
    if mutation == "engine":
        after["engine_provenance"]["binary_sha256"] = digest("different-engine")
    elif mutation == "partition":
        after["counter_profile"]["partition_proof"]["status"] = "unproved"
    elif mutation == "warmup":
        after["warmup_runs"] = 0
    else:
        after["counter_profile"]["overlap_any_engine_cycles"] = 9
    feedback = checkpoint.paired_context_decision_feedback(record, receipt, target_sha256=digest("target"))
    assert feedback["status"] == "counter_evidence_unknown"
    assert feedback["missing"] and feedback["observation"] == {}
    assert feedback["global_speedup_proven"] is False


@pytest.mark.parametrize("mutation", ["work", "binding", "scope"])
def test_paired_substituted_work_binding_or_scope_refuses(paired_case, mutation):
    record, receipt = paired_case
    if mutation == "work":
        receipt["projection_proof"]["work_contract"]["timed_command_count"] += 1
    elif mutation == "binding":
        receipt["binding"]["target_digest"] = digest("other-target")
    else:
        receipt["scope"] = "full_model"
    with pytest.raises(ValueError):
        checkpoint.paired_context_decision_feedback(record, receipt, target_sha256=digest("target"))
