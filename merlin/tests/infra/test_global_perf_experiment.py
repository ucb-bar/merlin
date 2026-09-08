"""Macro search must recompile the graph and cannot inherit microbenchmark stopping semantics."""
from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.benchharness import hash_tree
from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "merlin/experiments/gemmini_perf_bench/scripts"
sys.path.insert(0, str(SCRIPTS))
G = importlib.import_module("run_global_perf_experiment")
PAS = G.PAS
SHA = {name: PAS._sha256(name.encode()) for name in ("target", "graph", "plan", "llvm", "buffer")}


@pytest.mark.parametrize("case,expected", [
    ("identical", "identical"), ("different", "different"),
    ("missing", "UNKNOWN"), ("unbound", "UNKNOWN"), ("refused", "UNKNOWN"),
])
def test_agent_brief_joins_existing_object_comparison_without_upgrading_claims(case, expected):
    from merlin.perf.structural_delta import compare_full_model_structure

    before = {"emission": {"candidate_lowered_sha256": "a" * 64}, "diagnostics": {
        "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": "b" * 64},
        "verified_global_plan_emission": {"status": "verified", "host_activity": {
            "status": "derived", "load_payload_bytes": 100, "store_payload_bytes": 100}},
        "machine_artifact_activity": {"candidate": {"status": "compiled",
            "source_sha256": "a" * 64, "object_sha256": "c" * 64}}}}
    after = copy.deepcopy(before)
    after["diagnostics"]["verified_global_plan_emission"]["host_activity"]["load_payload_bytes"] = 50
    machine = after["diagnostics"]["machine_artifact_activity"]["candidate"]
    if case == "different":
        machine["object_sha256"] = "d" * 64
    elif case == "missing":
        del after["diagnostics"]["machine_artifact_activity"]
    elif case == "unbound":
        machine["source_sha256"] = "e" * 64
    elif case == "refused":
        after["diagnostics"]["captured_logical_graph"]["logical_dispatch_digest"] = "f" * 64
    change = compare_full_model_structure(before, after)
    after["optimization_brief"] = {"ranked_actions": [{"action": "inspect host payload"}]}
    record = {"analysis": after, "static_comparison": {"structural_change": change}}
    original = copy.deepcopy(record)
    view = G.agent_analysis_view(record, complete_evidence="/perf-control/full.json")
    priority = view["analysis"]["optimization_brief"]["machine_code_evidence_priority"]
    assert priority["status"] == expected
    assert priority["global_performance_benefit"] == priority["numerical_equivalence"] == "UNKNOWN"
    assert "caller, setup, linked ELF and timing are separate" in priority["scope"]
    assert view["static_comparison"] == record["static_comparison"]
    assert view["analysis"]["optimization_brief"]["ranked_actions"] == after["optimization_brief"]["ranked_actions"]
    assert record == original
    if case != "refused":
        assert change["metrics"]["host_load_payload_bytes"]["delta"] == -50
    if case == "identical":
        assert "IR-only change" in priority["message"]
        assert "no demonstrated machine-code saving" in priority["message"]
        assert "hardware traffic savings" in priority["message"]
    else:
        assert "IR-only change" not in priority["message"]


def test_agent_brief_missing_comparison_does_not_invent_object_equality():
    record = {"analysis": {"optimization_brief": {}}}
    view = G.agent_analysis_view(record, complete_evidence="complete.json")
    priority = view["analysis"]["optimization_brief"]["machine_code_evidence_priority"]
    assert priority["status"] == "UNKNOWN"
    assert priority["comparison"] == {}
    assert record == {"analysis": {"optimization_brief": {}}}


@pytest.mark.parametrize("baseline_state,expected", [
    ("bound", "different"), ("missing", "UNKNOWN"), ("stale", "UNKNOWN"),
])
def test_metadata_only_edit_does_not_erase_cumulative_object_change(baseline_state, expected):
    from merlin.perf.structural_delta import compare_full_model_structure

    analysis = {"emission": {"baseline_lowered_sha256": "a" * 64,
                             "candidate_lowered_sha256": "b" * 64}, "diagnostics": {
        "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": "c" * 64},
        "verified_global_plan_emission": {"status": "verified"},
        "machine_artifact_activity": {
            "baseline": {"status": "compiled", "source_sha256": "a" * 64, "object_sha256": "d" * 64},
            "candidate": {"status": "compiled", "source_sha256": "b" * 64, "object_sha256": "e" * 64}}}}
    if baseline_state == "missing":
        del analysis["diagnostics"]["machine_artifact_activity"]["baseline"]
    elif baseline_state == "stale":
        analysis["diagnostics"]["machine_artifact_activity"]["baseline"]["source_sha256"] = "f" * 64
    record = {"analysis": analysis, "static_comparison": {"previous_iteration": 1,
        "structural_change": compare_full_model_structure(analysis, copy.deepcopy(analysis))}}
    original = copy.deepcopy(record)
    brief = G.agent_analysis_view(record, complete_evidence="full.json")["analysis"]["optimization_brief"]
    previous = brief["machine_code_evidence_priority"]
    cumulative = brief["optimization_baseline_machine_code_comparison"]
    assert previous["status"] == "identical"
    assert previous["comparison_arm"] == "preceding_analyzed_revision"
    assert previous["previous_iteration"] == 1
    assert "does not erase earlier changes" in previous["message"]
    assert cumulative["comparison_arm"] == "optimization_baseline"
    assert cumulative["status"] == expected
    assert cumulative["global_performance_benefit"] == cumulative["numerical_equivalence"] == "UNKNOWN"
    assert record == original


def setup_experiment(tmp_path, *, verified=True, primary_interface_bytes=9, **experiment_options):
    baseline, candidate, source = (tmp_path / name for name in ("base", "candidate", "model"))
    for path in (baseline, candidate, source):
        path.mkdir()
        (path / "source.txt").write_text(path.name)
    (source / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
    (source / "capsule.interface.mlir").write_bytes(b"x" * primary_interface_bytes)
    sentinel = PAS.StageE2ESentinel("real-model", str(source), str(source),
                                   PAS._exact_tree_record(source)["sha256"], ("lane",), ("L2",))
    calls = []

    def analyzer(base, current, objective, **kwargs):
        calls.append(hash_tree(current)["sha256"])
        candidate_sha = calls[-1]
        plan = {"status": "verified", "plan_digest": SHA["plan"],
                "candidate_sha256": candidate_sha, "logical_dispatch_digest": SHA["graph"],
                "source_sha256": PAS._sha256_file(
                    Path(objective.frozen_source_path) / "capsule.interface.mlir"),
                "candidate_lowered_sha256": SHA["llvm"],
                "candidate_command_buffer_sha256": SHA["buffer"], "emitted_dispatches": 2}
        return {
            "candidate_sha256": candidate_sha,
            "workload": {"capsule_sha256": objective.capsule_sha256},
            "emission": {"candidate_lowered_sha256": SHA["llvm"],
                         "candidate_command_buffer_sha256": SHA["buffer"]},
            "diagnostics": {
                "captured_logical_graph": {"status": "verified", "logical_dispatch_digest": SHA["graph"]},
                "verified_global_plan_emission": plan if verified else {},
                "arms": {"candidate": {"status": "emitted", "macs": 8192, "exact": True,
                                       "movement": {"known_bytes": 128, "exact_bytes": True}}},
            },
        }

    experiment = G.GlobalPerfExperiment(
        baseline=baseline, baseline_sha256=hash_tree(baseline)["sha256"],
        sentinel=sentinel, target="test-target", target_sha256=SHA["target"],
        output=tmp_path / "run", analyzer=analyzer, **experiment_options)
    return experiment, candidate, calls


def _static_cache_analyzer(calls):
    def analyzer(base, current, objective, **kwargs):
        candidate_sha = hash_tree(current)["sha256"]
        calls.append(candidate_sha)
        lowered = "module { func.func @compiled() }\n"
        command_text = '{"commands":[],"tensors":{}}\n'
        lowered_sha = PAS._sha256(lowered.encode())
        command_sha = PAS._sha256(command_text.encode())
        plan = {"status": "verified", "plan_digest": SHA["plan"],
                "candidate_sha256": candidate_sha, "logical_dispatch_digest": SHA["graph"],
                "source_sha256": PAS._sha256_file(
                    Path(objective.frozen_source_path) / "capsule.interface.mlir"),
                "candidate_lowered_sha256": lowered_sha,
                "candidate_command_buffer_sha256": command_sha, "emitted_dispatches": 0}
        analysis = {
            "candidate_sha256": candidate_sha,
            "workload": {"capsule_sha256": objective.capsule_sha256},
            "emission": {"candidate_lowered_sha256": lowered_sha,
                         "candidate_command_buffer_sha256": command_sha},
            "diagnostics": {
                "captured_logical_graph": {"status": "verified",
                                           "logical_dispatch_digest": SHA["graph"]},
                "verified_global_plan_emission": plan,
                "emission_execution": {"elapsed_seconds": 123.0},
                "arms": {"candidate": {"status": "emitted", "macs": 0, "exact": True,
                                       "movement": {"known_bytes": 0}}},
            },
            "timing_status": "UNMEASURED",
        }
        kwargs["artifact_sink"]({
            "lowered_text": lowered, "decoded_trace": {"instructions": []},
            "command_buffer": json.loads(command_text), "command_buffer_text": command_text,
            "interface": str(Path(objective.frozen_source_path) / "capsule.interface.mlir"),
            "candidate_sha256": candidate_sha, "candidate_lowered_sha256": lowered_sha,
            "candidate_command_buffer_sha256": command_sha,
            "task_instruction_evidence": {"status": "verified", "tasks": []},
        })
        return analysis
    return analyzer


def _portfolio_selection_analyzer(calls, *, changing_capsules, include_host_activity=True):
    changing_capsules = frozenset(changing_capsules)

    def analyzer(base, current, objective, **kwargs):
        candidate_sha = hash_tree(current)["sha256"]
        calls.append((objective.capsule, candidate_sha))
        changed = objective.capsule in changing_capsules
        revision = candidate_sha if changed else "stable-emission"
        lowered = f"module {{ func.func @compiled() }} // {objective.capsule}:{revision}\n"
        command_buffer = {"commands": [], "tensors": {}, "params": {
            "portfolio_fixture": {"capsule": objective.capsule, "revision": revision}}}
        command_text = json.dumps(command_buffer, sort_keys=True, separators=(",", ":")) + "\n"
        lowered_sha = PAS._sha256(lowered.encode())
        command_sha = PAS._sha256(command_text.encode())
        source = Path(objective.frozen_source_path) / "capsule.interface.mlir"
        revision_work = 100 if changed and Path(current, "source.txt").read_text() != "candidate" else 0
        plan = {
            "status": "verified",
            "plan_digest": PAS._document_sha256({
                "capsule": objective.capsule, "revision": revision}),
            "candidate_sha256": candidate_sha,
            "logical_dispatch_digest": PAS._document_sha256({"capsule": objective.capsule}),
            "source_sha256": PAS._sha256_file(source),
            "candidate_lowered_sha256": lowered_sha,
            "candidate_command_buffer_sha256": command_sha,
            "emitted_dispatches": 0,
        }
        if include_host_activity:
            plan.update(tasks=2 + int(bool(revision_work)), host_activity={
                "status": "derived",
                "load_payload_bytes": 10 + revision_work,
                "store_payload_bytes": 20,
                "static_allocation_payload_bytes": 30,
                "dynamic_operations": {"llvm.load": 1 + revision_work},
            })
        analysis = {
            "candidate_sha256": candidate_sha,
            "workload": {"capsule_sha256": objective.capsule_sha256},
            "emission": {"candidate_lowered_sha256": lowered_sha,
                         "candidate_command_buffer_sha256": command_sha},
            "diagnostics": {
                "captured_logical_graph": {"status": "verified",
                    "logical_dispatch_digest": plan["logical_dispatch_digest"]},
                "verified_global_plan_emission": plan,
                "arms": {"candidate": {"status": "emitted", "macs": 0, "exact": True,
                    "movement": {"known_bytes": 0, "exact_bytes": True}}},
            },
            "timing_status": "UNMEASURED",
        }
        kwargs["artifact_sink"]({
            "lowered_text": lowered, "decoded_trace": {"instructions": []},
            "command_buffer": command_buffer, "command_buffer_text": command_text,
            "interface": str(source), "candidate_sha256": candidate_sha,
            "candidate_lowered_sha256": lowered_sha,
            "candidate_command_buffer_sha256": command_sha,
            "task_instruction_evidence": {"status": "verified", "tasks": []},
        })
        return analysis

    return analyzer


def _portfolio_selection_experiment(tmp_path, monkeypatch, *, changing_capsules,
                                    name="selection", include_host_activity=True):
    snapshot, snapshot_sha = _make_test_source_snapshot(
        tmp_path, name + "_policy", "POLICY = 1\n")
    monkeypatch.setattr(G, "host_verification_policy_record", lambda: _test_host_policy(snapshot))
    extra_model = tmp_path / (name + "_secondary_model")
    extra_model.mkdir()
    (extra_model / "capsule.yaml").write_text(
        "interface_mlir: capsule.interface.mlir\n")
    (extra_model / "capsule.interface.mlir").write_text("module { func.func @secondary() }\n")
    extra = PAS.StageE2ESentinel(
        "secondary-model", str(extra_model), str(extra_model),
        PAS._exact_tree_record(extra_model)["sha256"], (), ())
    calls = []
    analyzer = _portfolio_selection_analyzer(
        calls, changing_capsules=changing_capsules,
        include_host_activity=include_host_activity)
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", analyzer)
    experiment, candidate = _make_static_cache_experiment(
        tmp_path, name, snapshot, snapshot_sha, analyzer, portfolio_sentinels=[extra])
    return experiment, candidate, calls


def _make_test_source_snapshot(tmp_path, name, policy_text):
    snapshot_tool = importlib.import_module("perf_snapshot")
    source = tmp_path / (name + "_source")
    policy_dir = source / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "verifier.py").write_text(policy_text)
    snapshot = tmp_path / (name + ".source")
    snapshot_tool.create(source, snapshot, output_root=tmp_path / "unused-out",
                         source_roots=("policy",))
    receipt = snapshot_tool.verify(snapshot)
    return snapshot, PAS._document_sha256(receipt["files"])


def _test_host_policy(snapshot):
    source = snapshot / "policy/verifier.py"
    sources = {str(source.resolve()): PAS._sha256_file(source)}
    relative = {"policy/verifier.py": PAS._sha256_file(source)}
    return {"schema": "global_host_verification_policy_v1", "sources": sources,
            "sha256": PAS._document_sha256(relative),
            "location_sha256": PAS._document_sha256(sources)}


def _make_static_cache_experiment(tmp_path, name, snapshot, snapshot_sha, analyzer, **options):
    root = tmp_path / name
    root.mkdir()
    baseline, candidate, model = (root / item for item in ("baseline", "candidate", "model"))
    for path, text in ((baseline, "compiler"), (candidate, "candidate")):
        path.mkdir()
        (path / "source.txt").write_text(text)
    model.mkdir()
    (model / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
    (model / "capsule.interface.mlir").write_text("module {}\n")
    sentinel = PAS.StageE2ESentinel(
        "cache-model", str(model), str(model), PAS._exact_tree_record(model)["sha256"], (), ())
    experiment = G.GlobalPerfExperiment(
        baseline=baseline, baseline_sha256=hash_tree(baseline)["sha256"], sentinel=sentinel,
        target="test-target", target_sha256=SHA["target"], output=root / "run",
        source_snapshot_root=snapshot, source_snapshot_files_sha256=snapshot_sha,
        analyzer=analyzer, **options)
    return experiment, candidate


def test_exact_cross_run_static_checkpoint_hit_recomputes_current_state_without_measurements(
        tmp_path, monkeypatch):
    first_snapshot, first_snapshot_sha = _make_test_source_snapshot(
        tmp_path, "first", "POLICY = 1\n")
    second_snapshot, second_snapshot_sha = _make_test_source_snapshot(
        tmp_path, "second", "POLICY = 1\n")
    active_policy = [_test_host_policy(first_snapshot)]
    monkeypatch.setattr(G, "host_verification_policy_record", lambda: copy.deepcopy(active_policy[0]))
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer(seed_calls := []))
    seed, seed_candidate = _make_static_cache_experiment(
        tmp_path, "seed", first_snapshot, first_snapshot_sha, PAS.analyze_whole_model_emission)
    seed.analyze(seed_candidate, hypothesis="produce exact static seed")
    checkpoint = seed.seal(seed_candidate)

    active_policy[0] = _test_host_policy(second_snapshot)
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer(new_calls := []))
    current, current_candidate = _make_static_cache_experiment(
        tmp_path, "current", second_snapshot, second_snapshot_sha,
        PAS.analyze_whole_model_emission)
    receipt = current.import_static_analysis_checkpoint(
        current_candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))

    assert receipt["status"] == "hit"
    assert len(seed_calls) == 1 and new_calls == []
    assert len(current.iterations) == 1
    row = current.iterations[0]
    assert row["readiness"]["status"] == "ready_for_probe_admission"
    assert row["probe_receipts"] == []
    assert row["static_comparison"]["structural_change"]["status"] == "initial_observation"
    assert row["analysis_reuse"]["probe_or_timing_receipts_reused"] is False
    assert row["analysis_reuse"]["semantic_or_decision_feedback_reused"] is False
    assert row["analysis"]["diagnostics"]["emission_execution"]["timing_evidence_imported"] is False
    assert "elapsed_seconds" not in row["analysis"]["diagnostics"]["emission_execution"]
    assert Path(current.current_artifacts(current_candidate)["interface"]).is_file()
    assert list(current.current_portfolio_artifacts(current_candidate)) == [
        current.sentinel.capsule_sha256]
    assert Path(row["submitted_snapshot"]).stat().st_mode & 0o222 == 0


def test_v11_v13_style_host_policy_byte_mutation_is_an_explicit_cache_miss(
        tmp_path, monkeypatch):
    first_snapshot, first_snapshot_sha = _make_test_source_snapshot(
        tmp_path, "v11", "POLICY = 11\n")
    second_snapshot, second_snapshot_sha = _make_test_source_snapshot(
        tmp_path, "v13", "POLICY = 13\n")
    active_policy = [_test_host_policy(first_snapshot)]
    monkeypatch.setattr(G, "host_verification_policy_record", lambda: copy.deepcopy(active_policy[0]))
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer(seed_calls := []))
    seed, seed_candidate = _make_static_cache_experiment(
        tmp_path, "v11_run", first_snapshot, first_snapshot_sha, PAS.analyze_whole_model_emission)
    seed.analyze(seed_candidate, hypothesis="v11 static analysis")
    checkpoint = seed.seal(seed_candidate)

    active_policy[0] = _test_host_policy(second_snapshot)
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer(new_calls := []))
    current, current_candidate = _make_static_cache_experiment(
        tmp_path, "v13_run", second_snapshot, second_snapshot_sha,
        PAS.analyze_whole_model_emission)
    miss = current.import_static_analysis_checkpoint(
        current_candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))
    assert miss["status"] == "miss"
    assert miss["reason"] == "host_verification_policy_content_changed"
    assert current.iterations == [] and new_calls == []
    current.analyze(current_candidate, hypothesis="cold v13 analysis after safe miss")
    assert len(new_calls) == 1


def test_cross_run_static_checkpoint_direct_api_rejects_raw_symlink_and_relative_path(
        tmp_path, monkeypatch):
    snapshot, snapshot_sha = _make_test_source_snapshot(
        tmp_path, "direct_path", "POLICY = 1\n")
    monkeypatch.setattr(G, "host_verification_policy_record", lambda: _test_host_policy(snapshot))
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer([]))
    seed, seed_candidate = _make_static_cache_experiment(
        tmp_path, "direct_path_seed", snapshot, snapshot_sha,
        PAS.analyze_whole_model_emission)
    seed.analyze(seed_candidate, hypothesis="produce exact static seed")
    checkpoint = seed.seal(seed_candidate)
    linked = tmp_path / "linked_checkpoint.json"
    linked.symlink_to(checkpoint)

    linked_current, linked_candidate = _make_static_cache_experiment(
        tmp_path, "direct_path_linked", snapshot, snapshot_sha,
        PAS.analyze_whole_model_emission)
    with pytest.raises(ValueError, match="absent, mutable, linked, or changed"):
        linked_current.import_static_analysis_checkpoint(
            linked_candidate, checkpoint=linked,
            checkpoint_sha256=PAS._sha256_file(checkpoint))

    relative_current, relative_candidate = _make_static_cache_experiment(
        tmp_path, "direct_path_relative", snapshot, snapshot_sha,
        PAS.analyze_whole_model_emission)
    with pytest.raises(ValueError, match="absent, mutable, linked, or changed"):
        relative_current.import_static_analysis_checkpoint(
            relative_candidate, checkpoint=Path("relative-checkpoint.json"),
            checkpoint_sha256=PAS._sha256_file(checkpoint))


def test_cross_run_static_checkpoint_direct_api_rejects_linked_snapshot_root(
        tmp_path, monkeypatch):
    snapshot, snapshot_sha = _make_test_source_snapshot(
        tmp_path, "snapshot_real", "POLICY = 1\n")
    monkeypatch.setattr(G, "host_verification_policy_record", lambda: _test_host_policy(snapshot))
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer([]))
    seed, seed_candidate = _make_static_cache_experiment(
        tmp_path, "snapshot_seed", snapshot, snapshot_sha,
        PAS.analyze_whole_model_emission)
    seed.analyze(seed_candidate, hypothesis="produce exact static seed")
    checkpoint = seed.seal(seed_candidate)
    linked_snapshot = tmp_path / "linked_snapshot.source"
    linked_snapshot.symlink_to(snapshot, target_is_directory=True)
    current, candidate = _make_static_cache_experiment(
        tmp_path, "snapshot_current", linked_snapshot, snapshot_sha,
        PAS.analyze_whole_model_emission)

    with pytest.raises(ValueError, match="source snapshot root is relative, linked, mutable, or absent"):
        current.import_static_analysis_checkpoint(
            candidate, checkpoint=checkpoint,
            checkpoint_sha256=PAS._sha256_file(checkpoint))


def test_changed_region_selects_secondary_when_primary_emission_is_unchanged(
        tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"})
    experiment.analyze(candidate, hypothesis="initial portfolio emission")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="change secondary lowering mechanism")

    selection = experiment.select_changed_portfolio_member(candidate)
    assert selection["portfolio_index"] == 1
    assert selection["capsule"] == "secondary-model"
    assert selection["changed_artifact_fields"] == [
        "plan_digest", "lowered_sha256", "command_buffer_sha256"]
    assert selection["performance_inference"] == "none"

    seen = []
    def provider(*, candidate, experiment, timeout_s, portfolio_member):
        selected = experiment.selected_changed_portfolio_context(candidate, portfolio_member)
        binding = {"selection": selected["selection"],
                   "previous": selected["previous"]["member_binding"],
                   "current": selected["current"]["member_binding"]}
        seen.append((portfolio_member["capsule"], selected["current"]["interface"].read_text()))
        return {"status": "passed", "portfolio_member_binding": binding}

    receipt = experiment.qualify_changed_region(candidate, provider=provider, timeout_s=30)
    assert seen == [("secondary-model", "module { func.func @secondary() }\n")]
    assert receipt["portfolio_member_binding"]["selection"] == selection
    current_binding = receipt["portfolio_member_binding"]["current"]
    assert receipt["binding"] == {
        "graph_digest": current_binding["logical_dispatch_digest"],
        "plan_digest": current_binding["plan_digest"],
        "compiler_digest": current_binding["compiler_implementation_sha256"],
        "target_digest": current_binding["target_sha256"],
    }
    assert receipt["full_model_cycles"] is None
    checkpoint = experiment.seal(candidate)
    consumed = G.consume_global_candidate(checkpoint)
    assert consumed["semantic_receipts"] == experiment.iterations[-1]["semantic_receipts"]


def test_changed_region_selects_primary_when_primary_is_the_changed_member(
        tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"cache-model"})
    experiment.analyze(candidate, hypothesis="initial portfolio emission")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="change primary lowering mechanism")

    selection = experiment.select_changed_portfolio_member(candidate)
    assert selection["portfolio_index"] == 0
    assert selection["capsule"] == "cache-model"


def _selected_member_semantic_provider(*, candidate, experiment, timeout_s, portfolio_member):
    selected = experiment.selected_changed_portfolio_context(candidate, portfolio_member)
    return {"status": "passed", "portfolio_member_binding": {
        "selection": selected["selection"], "previous": selected["previous"]["member_binding"],
        "current": selected["current"]["member_binding"]}}


def _qualified_secondary_checkpoint(tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"})
    experiment.analyze(candidate, hypothesis="initial portfolio")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="changed secondary")
    semantic = experiment.qualify_changed_region(
        candidate, provider=_selected_member_semantic_provider, timeout_s=30)
    checkpoint = experiment.seal(candidate)
    return experiment, candidate, semantic, checkpoint


@pytest.mark.parametrize("mutation", [
    "missing_prior", "changed_prior", "cross_run", "prior_portfolio_digest",
    "invented_previous_binding", "ranking_delta", "cross_member", "legacy",
])
def test_semantic_consumer_rederives_pinned_prior_portfolio(tmp_path, monkeypatch, mutation):
    experiment, candidate, semantic, checkpoint = _qualified_secondary_checkpoint(tmp_path, monkeypatch)
    document = PAS._mapping_file(checkpoint)
    reference = document["semantic_receipts"][0]
    semantic_path = Path(reference["path"])
    prior_path = Path(semantic["previous_iteration_record"])
    if mutation == "missing_prior":
        prior_path.unlink()
    elif mutation == "changed_prior":
        prior_path.chmod(0o644)
        prior_path.write_text(prior_path.read_text() + "\n")
        prior_path.chmod(0o444)
    elif mutation == "cross_run":
        elsewhere = tmp_path / "other_run"
        elsewhere.mkdir()
        copied = elsewhere / prior_path.name
        copied.write_bytes(prior_path.read_bytes())
        copied.chmod(0o444)
        semantic["previous_iteration_record"] = str(copied)
    elif mutation == "prior_portfolio_digest":
        semantic["previous_portfolio_iteration_sha256"] = "f" * 64
    elif mutation == "invented_previous_binding":
        # All duplicated claims agree, but the previous immutable iteration contradicts them.
        binding = semantic["portfolio_member_binding"]
        binding["previous"]["lowered_sha256"] = "f" * 64
        binding["selection"]["previous"] = copy.deepcopy(binding["previous"])
        semantic["previous_artifact_sha256"] = "f" * 64
        semantic["evidence"]["portfolio_member_binding"] = copy.deepcopy(binding)
    elif mutation == "ranking_delta":
        binding = semantic["portfolio_member_binding"]
        binding["selection"]["structural_host_work_delta"]["host_payload_bytes_absolute_delta"] += 1000
        semantic["evidence"]["portfolio_member_binding"] = copy.deepcopy(binding)
    elif mutation == "cross_member":
        semantic["portfolio_member_binding"]["selection"]["portfolio_index"] = 0
    else:
        semantic["schema"] = "global_changed_region_semantic_receipt_v1"
    semantic_path.chmod(0o644)
    semantic_path.write_text(json.dumps(semantic))
    semantic_path.chmod(0o444)
    reference["sha256"] = PAS._sha256_file(semantic_path)
    checkpoint.chmod(0o644)
    checkpoint.write_text(json.dumps(document))
    checkpoint.chmod(0o444)
    with pytest.raises(ValueError, match="semantic|preceding|legacy"):
        G.consume_global_candidate(checkpoint)


def test_secondary_semantic_supplement_binds_both_portfolios(tmp_path, monkeypatch):
    experiment, candidate, semantic, checkpoint = _qualified_secondary_checkpoint(tmp_path, monkeypatch)
    # Real old-policy subprocess verification is separately exercised by checkpoint consumers.
    monkeypatch.setattr(G, "verify_retained_global_checkpoint", G.consume_global_candidate)
    semantic_path = Path(experiment.iterations[-1]["semantic_receipts"][0]["path"])
    result = G.write_semantic_supplement(
        original_candidate_receipt=checkpoint,
        previous_iteration_receipt=Path(semantic["previous_iteration_record"]),
        semantic_receipt=semantic_path, output=tmp_path / "supplement.json")
    assert result["semantic_status"] == "passed"
    assert result["portfolio_member_binding"]["current"]["capsule"] == "secondary-model"
    assert result["full_model_numerics_qualified"] is False
    assert result["global_speedup_proven"] is False
    wrong_previous = tmp_path / "wrong_previous.json"
    wrong_previous.write_bytes(Path(semantic["previous_iteration_record"]).read_bytes())
    wrong_previous.chmod(0o444)
    with pytest.raises(ValueError, match="supplement preceding iteration"):
        G.write_semantic_supplement(
            original_candidate_receipt=checkpoint, previous_iteration_receipt=wrong_previous,
            semantic_receipt=semantic_path, output=tmp_path / "wrong_supplement.json")


def test_semantic_supplement_explicitly_refuses_legacy_schema(tmp_path):
    path = tmp_path / "old_supplement.json"
    path.write_text(json.dumps({"schema": "global_semantic_supplement_v1"}))
    with pytest.raises(ValueError, match="policy or scope"):
        G.consume_semantic_supplement(path)


def test_structural_selector_treats_exact_empty_operation_maps_as_known_zero():
    before = {"diagnostics": {"verified_global_plan_emission": {
        "host_activity": {"dynamic_operations": {}}}}}
    after = copy.deepcopy(before)
    delta = G.GlobalPerfExperiment._known_structural_host_work_delta(before, after)
    assert delta["host_dynamic_operations_absolute_delta"] == 0
    assert delta["host_payload_bytes_absolute_delta"] is None
    assert delta["planned_task_count_absolute_delta"] is None


def test_changed_region_selector_records_stable_fallback_when_all_host_work_is_unknown(
        tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"cache-model", "secondary-model"},
        include_host_activity=False)
    experiment.analyze(candidate, hypothesis="initial portfolio emission")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="same mechanism changes both portfolio members")

    selection = experiment.select_changed_portfolio_member(candidate)
    assert selection["portfolio_index"] == 0
    assert selection["selection_basis"] == \
        "stable_portfolio_order_no_known_structural_host_work"
    assert selection["known_ranking_metrics"] == []
    assert selection["stable_portfolio_order_tie_break_applied"] is True
    assert selection["performance_inference"] == "none"


def test_changed_region_member_artifact_tamper_refuses_before_provider(
        tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"})
    experiment.analyze(candidate, hypothesis="initial portfolio emission")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="change secondary lowering mechanism")
    secondary = experiment.portfolio_sentinels[1].capsule_sha256
    experiment._portfolio_artifacts[secondary]["candidate_lowered_sha256"] = "0" * 64
    called = []

    with pytest.raises(ValueError, match="binding changed"):
        experiment.qualify_changed_region(
            candidate, timeout_s=30,
            provider=lambda **kwargs: called.append(kwargs) or {"status": "passed"})
    assert called == []


def test_changed_region_cross_member_artifact_substitution_refuses(tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"})
    experiment.analyze(candidate, hypothesis="initial portfolio emission")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="change secondary lowering mechanism")
    primary, secondary = (member.capsule_sha256 for member in experiment.portfolio_sentinels)
    experiment._portfolio_artifacts[secondary] = experiment._portfolio_artifacts[primary]

    with pytest.raises(ValueError, match="binding changed"):
        experiment.select_changed_portfolio_member(candidate)


def test_changed_region_refuses_when_no_portfolio_emission_changed(tmp_path, monkeypatch):
    experiment, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules=set())
    experiment.analyze(candidate, hypothesis="initial portfolio emission")
    (candidate / "source.txt").write_text("candidate-v2")
    experiment.analyze(candidate, hypothesis="metadata-only compiler revision")
    called = []

    with pytest.raises(ValueError, match="no portfolio member has a changed emitted artifact"):
        experiment.qualify_changed_region(
            candidate, timeout_s=30,
            provider=lambda **kwargs: called.append(kwargs) or {"status": "passed"})
    assert called == []


def test_imported_static_portfolio_bundle_can_be_previous_changed_member_context(
        tmp_path, monkeypatch):
    seed, seed_candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"}, name="import_seed")
    seed.analyze(seed_candidate, hypothesis="produce portfolio static seed")
    checkpoint = seed.seal(seed_candidate)

    current, candidate, calls = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"}, name="import_current")
    receipt = current.import_static_analysis_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))
    assert receipt["status"] == "hit"
    assert calls == []
    assert current.current_portfolio_member_context(candidate, index=1)[
        "member_binding"]["capsule"] == "secondary-model"

    (candidate / "source.txt").write_text("candidate-v2")
    current.analyze(candidate, hypothesis="change secondary after imported static seed")
    selection = current.select_changed_portfolio_member(candidate)
    assert selection["portfolio_index"] == 1
    assert selection["capsule"] == "secondary-model"
    semantic = current.qualify_changed_region(
        candidate, provider=_selected_member_semantic_provider, timeout_s=30)
    assert Path(semantic["previous_iteration_record"]) == current.output / "iteration_0000.json"
    assert semantic["previous_iteration_record_sha256"] == PAS._sha256_file(
        current.output / "iteration_0000.json")
    assert G.consume_global_candidate(current.seal(candidate))["semantic_receipts"]


def test_imported_secondary_member_can_compile_with_exact_previous_production_policy(
        tmp_path, monkeypatch):
    import subprocess
    from merlin.perf.analysis_worker import IsolatedAnalysisWorker
    from merlin.perf import analysis_worker as worker
    from merlin.targetgen import oot_runner

    seed, seed_candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"},
        name="probe_import_seed")
    seed.analyze(seed_candidate, hypothesis="produce portfolio static seed")
    checkpoint = seed.seal(seed_candidate)

    current, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"},
        name="probe_import_current")
    delegate = current.analyzer
    factory_calls = []

    def sandbox_factory(baseline, package, scratch):
        factory_calls.append((Path(package), Path(scratch)))
        result = {}
        for arm, selected in (("baseline", Path(baseline)), ("candidate", Path(package))):
            result[arm] = {
                "package_path": str(selected.resolve()),
                "scratch_path": str(Path(scratch).resolve()),
                "compiler_dependencies": current._compiler_dependencies(selected),
                "command_prefix": ["bwrap", "--clearenv", "PAYLOAD"],
                "bwrap_argv_length": 2, "answer_surfaces": [], "overlay_trees": {},
            }
        return result

    class FixtureIsolatedWorker(IsolatedAnalysisWorker):
        calls = 0
        def __call__(self, baseline, package, objective, *, artifact_sink=None, **kwargs):
            type(self).calls += 1
            scratch = tmp_path / f"fixture_worker_scratch_{type(self).calls}"
            scratch.mkdir()
            self.completed_sandboxes = self.sandbox_factory(baseline, package, scratch)
            return delegate(baseline, package, objective,
                            artifact_sink=artifact_sink, **kwargs)

    current.analyzer = FixtureIsolatedWorker(
        stage_path=Path(PAS.__file__), sandbox_factory=sandbox_factory,
        output=tmp_path / "fixture_worker")
    # Preserve the production worker API while keeping this regression compiler-free.
    current._member_analyzer = lambda: current.analyzer
    imported = current.import_static_analysis_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))
    reconstruction = current.iterations[0]["analysis_reuse"]["compiler_sandbox_reconstruction"]
    assert imported["status"] == "hit"
    assert reconstruction["status"] == "prepared_from_current_trusted_factory"
    assert reconstruction["compiler_invoked"] is False
    assert current._compiler_sandbox_sha256[0] == reconstruction["policy_set_sha256"]
    imported_submission = Path(current.iterations[0]["submitted_snapshot"])
    assert factory_calls[0][0] == imported_submission

    (candidate / "source.txt").write_text("candidate-v2")
    current.analyze(candidate, hypothesis="change secondary after imported static seed")
    assert current.select_changed_portfolio_member(candidate)["portfolio_index"] == 1
    loaded, observed = [], []
    monkeypatch.setattr(oot_runner, "load_package",
                        lambda path: loaded.append(path) or object())
    monkeypatch.setattr(worker, "run_sandboxed_entrypoint",
        lambda *args, **kwargs: observed.append(kwargs["sandbox"])
        or subprocess.CompletedProcess([], 0, "module {}", ""))
    secondary_interface = current.selected_changed_portfolio_context(candidate)[
        "previous"]["interface"]
    result = current.compile_previous_probe_candidate(
        candidate, secondary_interface, tmp_path / "previous_secondary_probe", timeout_s=10)

    assert result.returncode == 0
    assert loaded == [imported_submission]
    assert observed[0]["package_path"] == str(imported_submission)
    assert observed[0]["compiler_dependencies"] == current.iterations[0]["compiler_dependencies"]


def test_imported_previous_probe_refuses_reconstructed_policy_tamper(tmp_path, monkeypatch):
    from merlin.perf.analysis_worker import IsolatedAnalysisWorker

    seed, seed_candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"},
        name="policy_tamper_seed")
    seed.analyze(seed_candidate, hypothesis="produce portfolio static seed")
    checkpoint = seed.seal(seed_candidate)
    current, candidate, _ = _portfolio_selection_experiment(
        tmp_path, monkeypatch, changing_capsules={"secondary-model"},
        name="policy_tamper_current")

    def sandbox_factory(baseline, package, scratch):
        return {arm: {"package_path": str(Path(selected).resolve()),
            "scratch_path": str(Path(scratch).resolve()),
            "compiler_dependencies": current._compiler_dependencies(Path(selected)),
            "command_prefix": ["bwrap", "--clearenv", "PAYLOAD"],
            "bwrap_argv_length": 2, "answer_surfaces": [], "overlay_trees": {}}
            for arm, selected in (("baseline", baseline), ("candidate", package))}

    current.analyzer = IsolatedAnalysisWorker(
        stage_path=Path(PAS.__file__), sandbox_factory=sandbox_factory,
        output=tmp_path / "tamper_worker")
    current.import_static_analysis_checkpoint(
        candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))
    current._compiler_sandboxes[0]["candidate"]["command_prefix"].append("tampered")
    scratch = tmp_path / "tampered_policy_probe"
    scratch.mkdir()
    with pytest.raises(ValueError, match="policy changed identity"):
        current._probe_sandbox(candidate, scratch)


def test_cross_run_static_checkpoint_bundle_tamper_fails_closed(tmp_path, monkeypatch):
    snapshot, snapshot_sha = _make_test_source_snapshot(tmp_path, "same", "POLICY = 1\n")
    monkeypatch.setattr(G, "host_verification_policy_record",
                        lambda: _test_host_policy(snapshot))
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer([]))
    seed, candidate = _make_static_cache_experiment(
        tmp_path, "tamper_seed", snapshot, snapshot_sha, PAS.analyze_whole_model_emission)
    row = seed.analyze(candidate, hypothesis="produce exact static seed")
    checkpoint = seed.seal(candidate)
    bundle = Path(row["static_analysis_bundle"]["path"])
    bundle.chmod(0o644)
    current, current_candidate = _make_static_cache_experiment(
        tmp_path, "tamper_current", snapshot, snapshot_sha, PAS.analyze_whole_model_emission)
    with pytest.raises(ValueError, match="artifact bundle is absent, mutable, linked, or changed"):
        current.import_static_analysis_checkpoint(
            current_candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))


def test_cross_run_static_checkpoint_analysis_option_change_is_a_cold_miss(tmp_path, monkeypatch):
    first_snapshot, first_snapshot_sha = _make_test_source_snapshot(
        tmp_path, "option_first", "POLICY = 1\n")
    second_snapshot, second_snapshot_sha = _make_test_source_snapshot(
        tmp_path, "option_second", "POLICY = 1\n")
    active_policy = [_test_host_policy(first_snapshot)]
    monkeypatch.setattr(G, "host_verification_policy_record", lambda: copy.deepcopy(active_policy[0]))
    monkeypatch.setattr(PAS, "analyze_whole_model_emission", _static_cache_analyzer([]))
    seed, candidate = _make_static_cache_experiment(
        tmp_path, "option_seed", first_snapshot, first_snapshot_sha,
        PAS.analyze_whole_model_emission, timeout_s=300)
    seed.analyze(candidate, hypothesis="produce exact static seed")
    checkpoint = seed.seal(candidate)
    active_policy[0] = _test_host_policy(second_snapshot)
    current, current_candidate = _make_static_cache_experiment(
        tmp_path, "option_current", second_snapshot, second_snapshot_sha,
        PAS.analyze_whole_model_emission, timeout_s=301)
    miss = current.import_static_analysis_checkpoint(
        current_candidate, checkpoint=checkpoint, checkpoint_sha256=PAS._sha256_file(checkpoint))
    assert miss["status"] == "miss"
    assert "analysis_options" in miss["reason"]
    assert current.iterations == []


def test_launcher_requires_static_analysis_seed_path_and_sha_together():
    launcher = importlib.import_module("launch_global_agent_experiment")
    with pytest.raises(SystemExit) as caught:
        launcher.main([
            "--campaign-config", "not-read.json", "--candidate", "candidate",
            "--output", "not-created", "--static-analysis-seed-checkpoint", "checkpoint.json",
        ])
    assert caught.value.code == 2


def test_cross_run_identity_is_path_neutral_but_content_and_order_strict(tmp_path):
    dep = {"candidate_sha256": SHA["graph"], "shared_sources": {"planner.py": SHA["plan"]},
           "selected_lazy_exports": {"merlin.plan": "merlin.planner"}}
    left_opt = {"schema": "global_optimization_baseline_v1", "selection": "explicit_host_seed",
                "path": "/old/run/optimization_baseline", "sha256": SHA["graph"],
                "compiler_dependencies": {**dep, "shared_source_root": "/old/source"},
                "reason": "fixed seed", "scope": "comparison", "phase1_regraded": False,
                "objective_numerical_qualification": "UNPROVEN"}
    right_opt = copy.deepcopy(left_opt)
    right_opt["path"] = "/new/run/optimization_baseline"
    right_opt["compiler_dependencies"]["shared_source_root"] = "/new/source"
    assert G._portable_optimization_baseline_binding(left_opt) == \
        G._portable_optimization_baseline_binding(right_opt)

    edit = {"schema": "host_frozen_compiler_edit_authority_v1",
            "initial_candidate_sha256": SHA["graph"],
            "contract_document_sha256": SHA["plan"],
            "contract": {"existing_symbols": [{"path": "compiler.py", "symbol": "lower"}]},
            "seed_path": "/old/run/edit_scope_seed"}
    moved_edit = {**edit, "seed_path": "/new/run/edit_scope_seed"}
    assert G._portable_edit_authority(edit) == G._portable_edit_authority(moved_edit)
    assert G._portable_historical_reference(
        {"path": "/old/history", "sha256": SHA["graph"], "summary": {"rows": 1}}) == \
        G._portable_historical_reference(
            {"path": "/new/history", "sha256": SHA["graph"], "summary": {"rows": 1}})
    assert G._portable_phase1_binding({"run_dir": "/old/phase1", "run_id": "p1",
                                       "evidence_sha256": {"freeze.json": SHA["buffer"]}}) == \
        G._portable_phase1_binding({"run_dir": "/new/phase1", "run_id": "p1",
                                    "evidence_sha256": {"freeze.json": SHA["buffer"]}})

    first_tool = tmp_path / "first-tool"
    second_tool = tmp_path / "second-tool"
    first_tool.write_bytes(b"same tool")
    second_tool.write_bytes(b"same tool")
    digest = PAS._sha256_file(first_tool)
    left_policy = {"schema": "machine_artifact_policy_identity_v1",
                   "compiler": {"path": str(first_tool), "resolved_path": str(first_tool),
                                "sha256": digest}, "flags": ["-O2"]}
    right_policy = copy.deepcopy(left_policy)
    right_policy["compiler"].update(path=str(second_tool), resolved_path=str(second_tool))
    assert G._portable_machine_build_policy(left_policy, verify_files=True) == \
        G._portable_machine_build_policy(right_policy, verify_files=True)
    second_tool.write_bytes(b"different tool")
    with pytest.raises(ValueError, match="executable or implementation changed"):
        G._portable_machine_build_policy(right_policy, verify_files=True)

    first = G.full_model_portfolio_identity([
        SimpleNamespace(capsule="a", capsule_sha256=SHA["graph"], required_lanes=(),
                        required_tiers=()),
        SimpleNamespace(capsule="b", capsule_sha256=SHA["plan"], required_lanes=(),
                        required_tiers=()),
    ])
    second = G.full_model_portfolio_identity([
        SimpleNamespace(capsule="b", capsule_sha256=SHA["plan"], required_lanes=(),
                        required_tiers=()),
        SimpleNamespace(capsule="a", capsule_sha256=SHA["graph"], required_lanes=(),
                        required_tiers=()),
    ])
    assert PAS._document_sha256(first) != PAS._document_sha256(second)


def test_host_only_full_graph_and_authoring_budgets_are_distinct_from_probe_ceiling(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path, timeout_s=2400)

    record = experiment.analyze(candidate, hypothesis="Exercise the host-only static ceiling")

    assert calls == [hash_tree(candidate)["sha256"]]
    assert record["allocated_seconds"] == pytest.approx(2400, abs=0.01)
    contract = json.loads((experiment.output / "experiment.json").read_text())
    assert contract["maximum_full_graph_static_analysis_seconds"] == 2400
    assert contract["maximum_reduced_witness_seconds"] == 600
    with pytest.raises(ValueError, match="authoring bounds"):
        G.run_global_agent_sequence(
            experiment, candidate, run_round=lambda *_args, **_kwargs: {},
            stage_root=tmp_path / "stage", max_rounds=1,
            total_authoring_seconds=1201, round_seconds=1201)
    (tmp_path / "too_long").mkdir()
    with pytest.raises(ValueError, match="2400-second host wall budget"):
        setup_experiment(tmp_path / "too_long", timeout_s=2401)


def test_launcher_and_controller_share_the_canonical_resume_portfolio_identity(tmp_path):
    training_source = tmp_path / "training_model"
    training_source.mkdir()
    (training_source / "capsule.yaml").write_text(
        "interface_mlir: capsule.interface.mlir\n", encoding="utf-8")
    (training_source / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    training = PAS.StageE2ESentinel(
        "training-model", str(training_source), str(training_source),
        PAS._exact_tree_record(training_source)["sha256"], (), ())
    experiment, _candidate, _calls = setup_experiment(
        tmp_path, portfolio_sentinels=(training,))
    launcher = importlib.import_module("launch_global_agent_experiment")

    expected = G.full_model_portfolio_identity(experiment.portfolio_sentinels)

    assert launcher.full_model_portfolio_identity is G.full_model_portfolio_identity
    assert experiment.portfolio_identity == expected
    assert experiment.portfolio_identity_sha256 == PAS._document_sha256(expected)
    assert expected["execution"] == "bounded_host_admitted_analysis_with_deterministic_record_order"


def test_launcher_refuses_static_analysis_above_distinct_host_ceiling():
    launcher = importlib.import_module("launch_global_agent_experiment")

    with pytest.raises(SystemExit) as caught:
        launcher.main([
            "--campaign-config", "not-read.json", "--candidate", "candidate",
            "--output", "not-created", "--iteration-seconds", "2401",
        ])

    assert caught.value.code == 2


def _portfolio_sentinel(tmp_path, name, *, interface_bytes=9):
    source = tmp_path / name
    source.mkdir()
    (source / "source.txt").write_text(name)
    (source / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
    (source / "capsule.interface.mlir").write_bytes(b"x" * interface_bytes)
    return PAS.StageE2ESentinel(
        name, str(source), str(source), PAS._exact_tree_record(source)["sha256"], (), ())


def test_portfolio_recompiles_every_full_graph_on_one_candidate_snapshot(tmp_path):
    extras = [_portfolio_sentinel(tmp_path, name) for name in ("language-model", "vision-language-model")]
    experiment, candidate, calls = setup_experiment(tmp_path, portfolio_sentinels=extras)

    first = experiment.analyze(candidate, hypothesis="cross-model global planning")
    assert len(calls) == 3
    assert first["portfolio"]["members_total"] == 3
    assert first["portfolio"]["members_ready"] == 3
    assert first["readiness"]["status"] == "ready_for_probe_admission"
    assert first["portfolio"]["members"][0]["analysis_ref"] == "/analysis"
    assert [row["identity"]["capsule"] for row in first["portfolio"]["members"]] == [
        "real-model", "language-model", "vision-language-model"]
    assert all(row.get("analysis", first["analysis"])["candidate_sha256"]
               == first["candidate_sha256"] for row in first["portfolio"]["members"])

    (candidate / "source.txt").write_text("second global proposal")
    second = experiment.analyze(candidate, hypothesis="cross-model data movement")
    assert len(calls) == 6
    assert second["portfolio"]["members_ready"] == 3
    assert all(row["static_comparison"]["previous_iteration"] == 0
               for row in second["portfolio"]["members"][1:])


def test_one_worker_portfolio_uses_lpt_shared_deadline_and_declared_record_order(
        tmp_path, monkeypatch):
    extras = [
        _portfolio_sentinel(tmp_path, "large-second", interface_bytes=900),
        _portfolio_sentinel(tmp_path, "small-third", interface_bytes=100),
        _portfolio_sentinel(tmp_path, "large-fourth", interface_bytes=900),
    ]
    experiment, candidate, _ = setup_experiment(
        tmp_path, primary_interface_bytes=100, portfolio_sentinels=extras, timeout_s=100)
    base_analyzer = experiment.analyzer
    clock = [0.0]
    monkeypatch.setattr(G.time, "monotonic", lambda: clock[0])
    budgets = []
    durations = {"real-model": 5.0, "large-second": 20.0,
                 "small-third": 5.0, "large-fourth": 10.0}

    def analyzer(base, current, objective, **kwargs):
        budgets.append((objective.capsule, kwargs["timeout_s"]))
        result = base_analyzer(base, current, objective, **kwargs)
        clock[0] += durations[objective.capsule]
        return result

    experiment.analyzer = analyzer
    record = experiment.analyze(candidate, hypothesis="bounded generic portfolio allocation")
    assert [name for name, _ in budgets] == [
        "large-second", "large-fourth", "real-model", "small-third"]
    assert [budget for _, budget in budgets] == pytest.approx([
        100.0, 80.0, 70.0, 65.0])
    allocations = [row["analysis_allocation"] for row in record["portfolio"]["members"]]
    assert [row["interface_bytes"] for row in allocations] == [100, 900, 100, 900]
    assert all(row["policy"] == record["portfolio"]["analysis_allocation_policy"]
               for row in allocations)
    assert [row["allocated_seconds"] for row in allocations] == pytest.approx([
        70.0, 100.0, 65.0, 80.0])
    assert record["portfolio"]["analysis_concurrency"]["admitted_workers"] == 1
    assert record["elapsed_seconds"] == 40.0


def test_measured_portfolio_allocation_preserves_floors_and_rolls_surplus(tmp_path):
    sentinels = [
        _portfolio_sentinel(tmp_path, "small", interface_bytes=100),
        _portfolio_sentinel(tmp_path, "large", interface_bytes=1000),
        _portfolio_sentinel(tmp_path, "middle", interface_bytes=500),
    ]
    measurements = {
        sentinels[0].capsule_sha256: 10.0,
        sentinels[1].capsule_sha256: 200.0,
        sentinels[2].capsule_sha256: 50.0,
    }
    first = G.portfolio_member_analysis_allocation(
        300.0, sentinels, emission_seconds_by_capsule_sha256=measurements)
    assert first["chance_floor_seconds"] == 50.0
    assert 50.0 <= first["allocated_seconds"] <= 200.0
    # Finishing the first arm early makes its unused time available to the exact
    # remaining members; the final member receives the entire remaining deadline.
    second = G.portfolio_member_analysis_allocation(
        280.0, sentinels[1:], emission_seconds_by_capsule_sha256=measurements)
    last = G.portfolio_member_analysis_allocation(
        190.0, sentinels[2:], emission_seconds_by_capsule_sha256=measurements)
    assert second["allocated_seconds"] <= 280.0 - second["chance_floor_seconds"]
    assert last["allocated_seconds"] == 190.0
    assert all(row["policy"] ==
               "bounded_equal_chance_floor_plus_measured_emission_cost_with_rolling_surplus"
               for row in (first, second, last))


def test_portfolio_concurrency_is_memory_admitted_and_refuses_pressure():
    admitted = G.portfolio_analysis_concurrency(
        requested_workers=4, members=4, memory_available_bytes=112 * G._GIB,
        minimum_memory_available_bytes=48 * G._GIB)
    assert admitted["admitted_workers"] == 4
    bounded = G.portfolio_analysis_concurrency(
        requested_workers=4, members=4, memory_available_bytes=80 * G._GIB,
        minimum_memory_available_bytes=48 * G._GIB)
    assert bounded["admitted_workers"] == 2
    with pytest.raises(TimeoutError, match="below the portfolio analysis admission floor"):
        G.portfolio_analysis_concurrency(
            requested_workers=4, members=4, memory_available_bytes=47 * G._GIB,
            minimum_memory_available_bytes=48 * G._GIB)


def test_lpt_portfolio_schedule_is_deterministic_and_projects_two_safe_workers():
    schedule = G.portfolio_concurrent_schedule([64.9, 213.5, 30.5, 241.7], workers=2)
    assert schedule["submission_order"] == [3, 1, 0, 2]
    assert schedule["worker_estimated_seconds"] == pytest.approx([272.2, 278.4])
    assert schedule["projected_wall_seconds"] == pytest.approx(278.4)
    assert sum(schedule["worker_estimated_seconds"]) == pytest.approx(
        sum([64.9, 213.5, 30.5, 241.7]))


def test_mandatory_reserve_uses_resource_admitted_lpt_makespan(tmp_path, monkeypatch):
    extras = [_portfolio_sentinel(tmp_path, name) for name in ("second", "third", "fourth")]
    experiment, _, _ = setup_experiment(
        tmp_path, portfolio_sentinels=extras, portfolio_analysis_workers=4,
        minimum_memory_available_bytes=64 * G._GIB)
    observations = {sentinel.capsule_sha256: {
        "emission_wall_seconds": 1.0, "observed_analysis_wall_seconds": seconds}
        for sentinel, seconds in zip(
            experiment.portfolio_sentinels, [64.9, 213.5, 30.5, 241.7], strict=True)}
    monkeypatch.setattr(experiment, "_baseline_emission_observations", lambda: observations)
    monkeypatch.setattr(experiment, "_portfolio_analysis_cost_estimates",
                        lambda: [64.9, 213.5, 30.5, 241.7])
    monkeypatch.setattr(G, "_host_memory_available_bytes", lambda: 96 * G._GIB)
    reserve = experiment.mandatory_analysis_reserve_seconds(420)
    assert reserve["analysis_concurrency"]["admitted_workers"] == 2
    assert reserve["analysis_schedule"]["projected_wall_seconds"] == pytest.approx(278.4)
    assert reserve["seconds"] == 298


def test_concurrent_portfolio_restores_declared_result_order(tmp_path, monkeypatch):
    import threading
    import time

    extras = [_portfolio_sentinel(tmp_path, name) for name in ("second", "third", "fourth")]
    experiment, candidate, _ = setup_experiment(
        tmp_path, portfolio_sentinels=extras, portfolio_analysis_workers=4,
        minimum_memory_available_bytes=16 * G._GIB)
    delegate = experiment.analyzer
    barrier = threading.Barrier(4)
    completed = []
    delays = {"real-model": 0.06, "second": 0.04, "third": 0.02, "fourth": 0.0}

    def analyzer(base, current, objective, **kwargs):
        barrier.wait(timeout=2)
        time.sleep(delays[objective.capsule])
        result = delegate(base, current, objective, **kwargs)
        completed.append(objective.capsule)
        return result

    experiment.analyzer = analyzer
    monkeypatch.setattr(G, "_host_memory_available_bytes", lambda: 80 * G._GIB)
    record = experiment.analyze(candidate, hypothesis="bounded concurrent portfolio")
    declared = [row["identity"]["capsule"] for row in record["portfolio"]["members"]]
    assert declared == ["real-model", "second", "third", "fourth"]
    assert completed == ["fourth", "third", "second", "real-model"]
    assert record["portfolio"]["analysis_concurrency"]["admitted_workers"] == 4
    assert record["portfolio"]["analysis_allocation_policy"] == \
        "shared_portfolio_deadline_with_measured_cost_lpt_admission"


def test_one_worker_observed_case_does_not_starve_long_member_with_local_slice(
        tmp_path, monkeypatch):
    extras = [_portfolio_sentinel(tmp_path, name) for name in ("tiny", "lstm", "smol")]
    experiment, candidate, _ = setup_experiment(
        tmp_path, portfolio_sentinels=extras, timeout_s=600,
        portfolio_analysis_workers=4, minimum_memory_available_bytes=64 * G._GIB)
    costs = [64.915176, 213.509303, 30.469780, 241.742972]
    durations = {"real-model": 96.336, "tiny": 213.509303,
                 "lstm": 30.469780, "smol": 241.742972}
    delegate = experiment.analyzer
    clock = [0.0]
    execution = []
    monkeypatch.setattr(G.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(G, "_host_memory_available_bytes", lambda: 80 * G._GIB)
    monkeypatch.setattr(experiment, "_portfolio_analysis_cost_estimates", lambda: costs)

    def analyzer(base, current, objective, **kwargs):
        execution.append((objective.capsule, kwargs["timeout_s"]))
        duration = durations[objective.capsule]
        assert kwargs["timeout_s"] >= duration
        result = delegate(base, current, objective, **kwargs)
        clock[0] += duration
        return result

    experiment.analyzer = analyzer
    record = experiment.analyze(candidate, hypothesis="share the exact portfolio deadline")
    assert [name for name, _ in execution] == ["smol", "tiny", "real-model", "lstm"]
    assert dict(execution)["real-model"] == pytest.approx(
        600 - durations["smol"] - durations["tiny"])
    assert dict(execution)["real-model"] > 96.336
    assert record["portfolio"]["members_ready"] == 4
    assert [row["identity"]["capsule"] for row in record["portfolio"]["members"]] == [
        "real-model", "tiny", "lstm", "smol"]
    assert record["elapsed_seconds"] == pytest.approx(sum(durations.values()))


def test_exact_baseline_emission_cache_reuses_identity_and_refuses_corruption(tmp_path):
    binding = {"root": str((tmp_path / "cache").resolve()),
               "compiler_dependencies_sha256": SHA["plan"]}
    identity = PAS.baseline_emission_cache_identity(
        baseline_sha256=SHA["graph"], capsule_sha256=SHA["buffer"],
        source_sha256=SHA["llvm"], target="test", compiler_dependencies_sha256=SHA["plan"],
        compiler_api_schema={"path": "/compiler-api/schema.json", "sha256": SHA["target"]},
        entrypoints=("emit_analysis_bundle",))
    assert PAS.load_baseline_emission_cache(binding, identity) is None
    cold = PAS.store_baseline_emission_cache(
        binding, identity, lowered_text="module {}\n", command_buffer_text='{"commands": []}\n',
        emission_wall_seconds=12.5)
    warm = PAS.load_baseline_emission_cache(binding, identity)
    assert warm is not None and warm["key"] == cold["key"]
    assert warm["lowered_text"] == "module {}\n"
    changed = dict(identity)
    changed["source_sha256"] = SHA["target"]
    assert PAS.load_baseline_emission_cache(binding, changed) is None
    lowered = Path(binding["root"]) / warm["key"] / "lowered.mlir"
    lowered.chmod(0o644)
    lowered.write_text("module { func.func @tampered() }\n")
    with pytest.raises(PAS.StageGateError, match="artifact digest changed"):
        PAS.load_baseline_emission_cache(binding, identity)


def test_timed_out_worker_can_seed_only_its_completed_baseline_emission(
        tmp_path, monkeypatch):
    from merlin.targetgen import oot_runner as OR

    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "compiler.py").write_text("VALUE = 1\n")
    sentinel = _portfolio_sentinel(tmp_path, "tiny-timeout")
    dependencies = {"candidate_sha256": hash_tree(baseline)["sha256"],
                    "shared_sources": {}, "selected_lazy_exports": {}}
    dependency_sha = G.compiler_dependency_content_sha256(dependencies)
    cache_binding = {"root": str((tmp_path / "cache").resolve()),
                     "compiler_dependencies_sha256": dependency_sha}
    seed = tmp_path / "seed"
    worker = seed / "host_analysis_workers" / "analysis_timeout"
    scratch = worker / "compiler_scratch" / "baseline"
    scratch.mkdir(parents=True)
    (seed / "global_iterations").mkdir()
    (seed / "global_iterations" / "experiment.json").write_text(json.dumps({
        "target": "test-target", "optimization_baseline_sha256": hash_tree(baseline)["sha256"],
        "optimization_baseline": {"compiler_dependencies": dependencies},
    }))
    interface = Path(sentinel.frozen_source_path) / "capsule.interface.mlir"
    (scratch / "interface.mlir").write_bytes(interface.read_bytes())
    (scratch / "command_buffer.json").write_text('{"commands": [], "tensors": {}}')
    (worker / "baseline_lowered.mlir").write_text("module {}\n")
    (worker / "baseline_emission.json").write_text(json.dumps({
        "schema": "compiler_emission_diagnostics_v1", "arm": "baseline",
        "entrypoints": [{"command": "emit_analysis_bundle", "returncode": 0}],
    }))
    (worker / "request.json").write_text(json.dumps({
        "baseline": str(baseline.resolve()), "kwargs": {"target": "test-target"},
        "sentinel": {"capsule_sha256": sentinel.capsule_sha256},
    }))
    (worker / "receipt.json").write_text(json.dumps({
        "schema": "bounded_host_analysis_worker_v1", "status": "timeout",
        "wall_seconds": 215.74,
    }))
    monkeypatch.setattr(OR, "load_package", lambda _path: object())
    monkeypatch.setattr(OR, "analysis_emission_entrypoints",
                        lambda _package: ("emit_analysis_bundle",))
    monkeypatch.setattr(PAS, "validate_whole_program_schema", lambda *_args, **_kwargs: None)
    schema = {"path": "/compiler-api/command_buffer.schema.json", "sha256": SHA["target"]}
    imported = G.seed_baseline_emission_cache_from_run(
        cache_binding=cache_binding, seed_run=seed, baseline=baseline,
        sentinels=(sentinel,), target="test-target", compiler_api_schema=schema)
    assert len(imported) == 1
    assert imported[0]["observed_analysis_wall_seconds"] == 215.74
    assert imported[0]["observed_analysis_status"] == "timeout"
    assert imported[0]["lowered_text"] == "module {}\n"


def test_portfolio_action_digest_resolves_primary_and_filters_exact_authority(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "second")
    experiment, candidate, _ = setup_experiment(tmp_path, portfolio_sentinels=[extra])
    record = experiment.analyze(candidate, hypothesis="inspect every member")
    surface = {"id": "global", "path": "compiler.py", "symbol": "Planner.run",
               "scope": "pass", "effects": ["movement"]}
    action = {"rank": 1, "kind": "movement", "status": "actionable",
              "detail": "delete materialization", "evidence": {"bytes": 128},
              "required_effects": ["movement"], "edit_surfaces": [surface,
                  {**surface, "id": "unapproved", "symbol": "Other.run"}]}
    optimization_order = {"schema": "macro_optimization_order_v1", "tiers": [{
        "tier": 1, "name": "whole_program_work_deletion"}]}
    record["analysis"]["optimization_brief"] = {
        "ranked_actions": [action], "optimization_order": optimization_order}
    record["analysis"]["diagnostics"]["arms"]["candidate"].update({
        "macs": 8192, "movement": {"known_bytes": 128},
        "representation_activity": {"command_counts": {"MATMUL": 2},
                                    "placement": {"lane_counts": {"on_mesh": 2}}},
    })
    digest = G.portfolio_action_digest(
        record, complete_evidence="INITIAL_FULL_MODEL_EVIDENCE.json",
        edit_contract={"existing_symbols": [{"surface_id": "global", "path": "compiler.py",
                                              "symbol": "Planner.run"}]})
    assert [row["identity"]["capsule"] for row in digest["members"]] == [
        "real-model", "second"]
    primary = digest["members"][0]
    assert primary["complete_unpruned_evidence"]["json_pointer"] == "/analysis"
    assert primary["totals"]["movement_known_bytes"] == 128
    assert primary["top_ranked_actions"][0]["authorized_edit_surfaces"] == [{
        **surface, "authority": "exact_host_frozen_existing_symbol"}]
    assert primary["optimization_order"] == optimization_order
    assert digest["optimization_order"] == optimization_order


def test_portfolio_member_failure_blocks_current_revision_and_seal(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "large-transformer")
    experiment, candidate, _ = setup_experiment(tmp_path, portfolio_sentinels=[extra])
    primary_analyzer = experiment.analyzer

    def analyzer(base, current, objective, **kwargs):
        if objective.capsule == "large-transformer":
            return {"candidate_sha256": hash_tree(current)["sha256"],
                    "workload": {"capsule_sha256": objective.capsule_sha256},
                    "diagnostics": {"arms": {"candidate": {"status": "emission_failed"}}},
                    "timing_status": "UNMEASURED"}
        return primary_analyzer(base, current, objective, **kwargs)

    experiment.analyzer = analyzer
    record = experiment.analyze(candidate, hypothesis="proposal must cover every training model")
    assert record["readiness"]["status"] == "blocked"
    assert record["portfolio"]["members_ready"] == 1
    assert any(value.startswith("portfolio:large-transformer:")
               for value in record["readiness"]["blockers"])
    with pytest.raises(ValueError, match="portfolio:large-transformer"):
        experiment.seal(candidate)


def test_blocked_portfolio_has_exact_non_promotable_authoring_checkpoint(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "large-transformer")
    experiment, candidate, _ = setup_experiment(tmp_path, portfolio_sentinels=[extra])
    primary_analyzer = experiment.analyzer

    def analyzer(base, current, objective, **kwargs):
        if objective.capsule == "large-transformer":
            return {"candidate_sha256": hash_tree(current)["sha256"],
                    "workload": {"capsule_sha256": objective.capsule_sha256},
                    "diagnostics": {"arms": {"candidate": {"status": "emission_failed"}}},
                    "timing_status": "UNMEASURED"}
        return primary_analyzer(base, current, objective, **kwargs)

    experiment.analyzer = analyzer
    record = experiment.analyze(candidate, hypothesis="repair unsupported portfolio member")
    assert record["readiness"]["status"] == "blocked"

    checkpoint = experiment.checkpoint_authoring(candidate, name="blocked_seed")
    consumed = G.consume_authoring_checkpoint(checkpoint)
    assert consumed["schema"] == "global_authoring_checkpoint_v1"
    assert consumed["candidate_sha256"] == record["candidate_sha256"]
    assert consumed["portfolio_members_ready"] == 1
    assert consumed["portfolio_members_total"] == 2
    assert consumed["promotion_status"] == "blocked_authoring_checkpoint"
    assert consumed["global_speedup_proven"] is False
    with pytest.raises(ValueError, match="invalid global candidate receipt"):
        G.consume_global_candidate(checkpoint)


def test_timeout_only_blocked_portfolio_retains_all_ready_authoring_checkpoint(
        tmp_path, monkeypatch):
    extra = _portfolio_sentinel(tmp_path, "large-transformer")
    experiment, candidate, _ = setup_experiment(
        tmp_path, portfolio_sentinels=[extra], timeout_s=10)
    primary_analyzer = experiment.analyzer
    now = [0.0]
    monkeypatch.setattr(G.time, "monotonic", lambda: now[0])

    def analyzer(*args, **kwargs):
        analysis = primary_analyzer(*args, **kwargs)
        now[0] += 6.0
        return analysis

    experiment.analyzer = analyzer
    record = experiment.analyze(candidate, hypothesis="retain completed portfolio evidence")
    assert record["portfolio"]["members_ready"] == record["portfolio"]["members_total"] == 2
    assert record["readiness"]["status"] == "blocked"
    assert record["readiness"]["blockers"] == ["iteration_wall_budget_exceeded"]

    checkpoint = experiment.checkpoint_authoring(candidate, name="timeout_seed")
    consumed = G.consume_authoring_checkpoint(checkpoint)
    assert consumed["portfolio_members_ready"] == consumed["portfolio_members_total"] == 2
    assert consumed["readiness"]["status"] == "blocked"
    assert consumed["promotion_status"] == "blocked_authoring_checkpoint"


def test_sustained_sequence_can_repair_blocked_initial_portfolio(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "large-transformer")
    experiment, candidate, _ = setup_experiment(tmp_path, portfolio_sentinels=[extra])
    primary_analyzer = experiment.analyzer

    def analyzer(base, current, objective, **kwargs):
        if (objective.capsule == "large-transformer"
                and (current / "source.txt").read_text() != "portfolio repaired"):
            return {"candidate_sha256": hash_tree(current)["sha256"],
                    "workload": {"capsule_sha256": objective.capsule_sha256},
                    "diagnostics": {"arms": {"candidate": {"status": "emission_failed"}}},
                    "timing_status": "UNMEASURED"}
        return primary_analyzer(base, current, objective, **kwargs)

    experiment.analyzer = analyzer
    inherited = []

    def author(current, *, round_index, round_timeout_s):
        inherited.append((round_index, (current / "source.txt").read_text()))
        (current / "source.txt").write_text("portfolio repaired")
        repaired = experiment.analyze(current, hypothesis="implement missing model support")
        assert repaired["readiness"]["status"] == "ready_for_probe_admission"
        return {"status": "authored"}

    result = G.run_global_agent_sequence(
        experiment, candidate, run_round=author, stage_root=tmp_path / "stage",
        max_rounds=1, total_authoring_seconds=30, round_seconds=30,
        on_round_failure="resume-last-checkpoint")
    assert inherited == [(0, "candidate")]
    assert result["checkpoints"][0]["role"] == "initial_blocked_authoring_seed"
    assert G.consume_authoring_checkpoint(
        Path(result["checkpoints"][0]["path"]))["portfolio_members_ready"] == 1
    assert G.consume_global_candidate(
        Path(result["last_good_checkpoint"]["path"]))["portfolio"]["members"]


def test_sustained_sequence_refuses_regression_of_ready_portfolio_member(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "large-transformer")
    experiment, candidate, _ = setup_experiment(tmp_path, portfolio_sentinels=[extra])
    primary_analyzer = experiment.analyzer

    def analyzer(base, current, objective, **kwargs):
        if (objective.capsule == "real-model"
                and (current / "source.txt").read_text() == "regress primary"):
            return {"candidate_sha256": hash_tree(current)["sha256"],
                    "workload": {"capsule_sha256": objective.capsule_sha256},
                    "diagnostics": {"arms": {"candidate": {"status": "emission_failed"}}},
                    "timing_status": "UNMEASURED"}
        return primary_analyzer(base, current, objective, **kwargs)

    experiment.analyzer = analyzer

    def author(current, *, round_index, round_timeout_s):
        (current / "source.txt").write_text("regress primary")
        experiment.analyze(current, hypothesis="bad trade between model families")
        return {"status": "authored"}

    with pytest.raises(ValueError, match="regressed previously verified portfolio members"):
        G.run_global_agent_sequence(
            experiment, candidate, run_round=author, stage_root=tmp_path / "stage",
            max_rounds=1, total_authoring_seconds=30, round_seconds=30,
            on_round_failure="stop")


def test_agent_view_compacts_secondary_portfolio_analysis(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "large-transformer")
    experiment, candidate, _ = setup_experiment(tmp_path, portfolio_sentinels=[extra])
    record = experiment.analyze(candidate, hypothesis="inspect every full graph")
    secondary = record["portfolio"]["members"][1]["analysis"]
    secondary["optimization_brief"] = {"ranked_actions": [{
        "rank": 1, "kind": "movement", "status": "actionable", "detail": "reuse encoding",
        "evidence": {"bytes": 10}, "required_effects": ["movement"],
        "edit_surfaces": [{"id": "layout", "path": "compiler/layout.py", "symbol": "plan",
                           "scope": "planner", "effects": ["movement"],
                           "validation": "intentionally omitted from compact view"}]}]}
    secondary["diagnostics"]["captured_logical_graph"] = {"nodes": list(range(1000))}

    view = G.agent_analysis_view(record, complete_evidence="/perf-control/full.json")
    compact = view["portfolio"]["members"][1]["analysis"]
    assert compact["schema"] == "portfolio_member_agent_summary_v1"
    assert "diagnostics" not in compact
    assert compact["optimization_brief"]["ranked_actions"][0]["edit_surfaces"] == [{
        "id": "layout", "path": "compiler/layout.py", "symbol": "plan",
        "scope": "planner", "effects": ["movement"]}]
    assert compact["complete_unpruned_evidence"]["json_pointer"] == "/portfolio/members/1/analysis"


@pytest.mark.parametrize("missing", ["path", "sha256", "wrong_sha256"])
def test_optimization_baseline_requires_exact_explicit_pair(tmp_path, missing):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "compiler.py").write_text("VALUE=1\n")
    options = {"optimization_baseline": seed,
               "optimization_baseline_sha256": hash_tree(seed)["sha256"]}
    if missing == "path":
        del options["optimization_baseline"]
    elif missing == "sha256":
        del options["optimization_baseline_sha256"]
    else:
        options["optimization_baseline_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="optimization baseline"):
        setup_experiment(tmp_path, **options)


def test_optimization_baseline_is_snapshot_not_phase1_or_previous_probe(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "compiler.py").write_text("VALUE=1\n")
    seed_sha = hash_tree(seed)["sha256"]
    verified = []
    class Phase1:
        def verify(self, baseline):
            verified.append(baseline)
            return {"submission_sha256": hash_tree(baseline)["sha256"],
                    "run_dir": str(tmp_path), "evidence_sha256": {}}
    experiment, candidate, _ = setup_experiment(tmp_path, phase1=Phase1(),
        optimization_baseline=seed, optimization_baseline_sha256=seed_sha,
        optimization_baseline_reason="external objective needs the explicit compiler feature")
    original_analyzer = experiment.analyzer
    compiled_bases = []
    def analyze(base, *args, **kwargs):
        compiled_bases.append(base)
        return original_analyzer(base, *args, **kwargs)
    experiment.analyzer = analyze
    # Only the host-captured copy is an input after construction.
    (seed / "compiler.py").write_text("VALUE=2\n")
    first = experiment.analyze(candidate, hypothesis="initial objective seed")
    (candidate / "source.txt").write_text("actual proposal")
    second = experiment.analyze(candidate, hypothesis="next proposal")
    assert all(path == experiment.baseline for path in verified)
    assert compiled_bases == [experiment.optimization_baseline] * 2
    assert experiment.optimization_baseline != seed
    assert hash_tree(experiment.optimization_baseline)["sha256"] == seed_sha
    assert not experiment.optimization_baseline.stat().st_mode & 0o222
    assert first["baseline_sha256"] == experiment.baseline_sha256 != seed_sha
    assert second["static_comparison"]["previous_iteration"] == first["iteration"]
    sealed = experiment.seal(candidate)
    consumed = G.consume_global_candidate(sealed)
    assert consumed["baseline_sha256"] == experiment.phase1_binding["submission_sha256"]
    assert consumed["optimization_baseline_sha256"] == seed_sha
    assert consumed["optimization_baseline"]["objective_numerical_qualification"] == "UNPROVEN"
    assert consumed["promotion_status"] == "unqualified_candidate_for_review"
    brief = G.agent_analysis_view(second, complete_evidence="complete.json")["analysis"]["optimization_brief"]
    assert brief["optimization_comparison_seed"]["sha256"] == seed_sha
    # Snapshot tampering stops both future actions and checkpoint consumption.
    frozen_source = experiment.optimization_baseline / "compiler.py"
    frozen_source.chmod(0o644)
    frozen_source.write_text("VALUE=3\n")
    with pytest.raises(ValueError, match="optimization baseline"):
        experiment._check_inputs()
    with pytest.raises(ValueError, match="optimization comparison baseline"):
        G.consume_global_candidate(sealed)


def test_optimization_baseline_shared_dependency_mutation_is_rejected(tmp_path):
    root = tmp_path / "shared" / "merlin"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("")
    helper = root / "seed_helper.py"
    helper.write_text("VALUE=1\n")
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "compiler.py").write_text("from merlin.seed_helper import VALUE\n")
    experiment, _, _ = setup_experiment(tmp_path, compiler_shared_source_root=root,
        optimization_baseline=seed, optimization_baseline_sha256=hash_tree(seed)["sha256"])
    helper.write_text("VALUE=2\n")
    with pytest.raises(ValueError, match="optimization baseline or shared"):
        experiment._check_inputs()


@pytest.mark.parametrize("legacy", [False, True])
def test_resume_never_silently_replaces_optimization_baseline(legacy):
    checkpoint = {"baseline_sha256": "a" * 64}
    if not legacy:
        checkpoint["optimization_baseline_sha256"] = "b" * 64
    expected = checkpoint.get("optimization_baseline_sha256", checkpoint["baseline_sha256"])
    G.validate_optimization_baseline_resume(checkpoint, optimization_baseline_sha256=expected)
    with pytest.raises(ValueError, match="optimization baseline differs"):
        G.validate_optimization_baseline_resume(checkpoint, optimization_baseline_sha256="c" * 64)


@pytest.mark.parametrize("option", ["--optimization-baseline", "--optimization-baseline-sha256"])
def test_launcher_requires_both_optimization_baseline_flags(option):
    launcher = importlib.import_module("launch_global_agent_experiment")
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--campaign-config", "not-read.json", "--candidate", "candidate",
                       "--output", "not-created", option, "a" * 64])
    assert caught.value.code == 2


def _historical_bundle(tmp_path):
    from merlin.perf.harvest import Observation
    from merlin.perf.historical_reference import bind_historical_reference, reference_summary
    raw = json.dumps({"capsule": "public-work", "label": "public", "target": "test-target",
        "tiers": {"L3": {"status": "pass", "cycles": 20, "engine": "test-engine",
            "sim_provenance": {"sha256": "e" * 64}}}}).encode()
    row = bind_historical_reference(
        Observation("submission", "public-work", "L3", "console", "rtl",
                    "total_cycles", 20, "cycles", status="pass"),
        target="test-target", receipt=raw, receipt_sha256=PAS._sha256(raw),
        receipt_path="public/public-work/capsule_result.json", public_workloads={"public-work"},
        artifacts={"kernel.elf": b"elf", "harness.c": b"caller"},
        artifact_sha256={"kernel.elf": PAS._sha256(b"elf"), "harness.c": PAS._sha256(b"caller")},
        executable_names=("kernel.elf",), harness_name="harness.c")
    bundle = {"schema": "historical_reference_bundle_v1", "records": [row],
              "summary": reference_summary([row]), "provenance": {}, "refusals": []}
    path = tmp_path / "reference.json"
    path.write_bytes(PAS._canonical_json(bundle))
    return path, PAS._sha256_file(path), bundle


@pytest.mark.parametrize("option", ["--historical-reference", "--historical-reference-sha256"])
def test_launcher_requires_both_historical_reference_flags(option):
    launcher = importlib.import_module("launch_global_agent_experiment")
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--campaign-config", "not-read.json", "--candidate", "candidate",
                       "--output", "not-created", option, "a" * 64])
    assert caught.value.code == 2


@pytest.mark.parametrize("mutation", ["bundle_sha", "record_sha", "summary", "authority", "target",
                                    "symlink", "candidate_writable"])
def test_historical_reference_explicit_bundle_fails_closed(tmp_path, mutation):
    path, digest, bundle = _historical_bundle(tmp_path)
    roots = ()
    if mutation == "bundle_sha":
        digest = "0" * 64
    elif mutation == "record_sha":
        bundle["records"][0]["cycles"] += 1
    elif mutation == "summary":
        bundle["summary"]["reference_count"] = 87
    elif mutation == "authority":
        record = bundle["records"][0]
        record["target_cycle_authority"] = True
        record["reference_sha256"] = PAS._sha256(json.dumps(
            {k: v for k, v in record.items() if k != "reference_sha256"},
            sort_keys=True, separators=(",", ":")).encode())
    elif mutation == "target":
        pass  # A valid bundle for a different selected target is still refused.
    elif mutation == "symlink":
        link = tmp_path / "reference-link.json"
        link.symlink_to(path)
        path = link
    else:
        roots = (tmp_path,)
    if mutation in ("record_sha", "summary", "authority"):
        path.write_bytes(PAS._canonical_json(bundle))
        digest = PAS._sha256_file(path)
    with pytest.raises(ValueError):
        G.load_historical_reference(path, digest,
            target="other-target" if mutation == "target" else "test-target", candidate_roots=roots)


def test_historical_reference_compact_feedback_and_readonly_control_copy(tmp_path):
    path, digest, bundle = _historical_bundle(tmp_path)
    experiment, candidate, _ = setup_experiment(tmp_path,
        historical_reference_path=path, historical_reference_sha256=digest)
    row = experiment.analyze(candidate, hypothesis="Inspect actual graph with historical context")
    original = copy.deepcopy(row)
    view = G.agent_analysis_view(row, complete_evidence="full.json")
    brief = view["analysis"]["optimization_brief"]["historical_reference"]
    assert brief["reference_count"] == brief["engine_group_count"] == 1
    assert brief["target_cycle_authority"] is brief["warm_calibration"] is False
    assert brief["full_model_cycle_ordering"] == "UNKNOWN"
    assert "host_verified_warm_window" in brief["missing_contracts"]
    assert "engine_groups" not in brief and "records" not in brief and "cycles" not in brief
    assert brief["artifact"] == {"path": "/perf-control/historical_reference.json", "sha256": digest}
    assert row == original
    control = tmp_path / "host_control"
    control.mkdir()
    workspace = tmp_path / "author_workspace"
    experiment.stage_historical_reference(control, workspace=workspace)
    staged = control / "historical_reference.json"
    assert staged.read_bytes() == path.read_bytes()
    assert staged.stat().st_mode & 0o222 == 0
    sealed = experiment.seal(candidate)
    assert G.consume_global_candidate(sealed)["historical_reference"] == experiment.historical_reference
    retained = Path(experiment.historical_reference["path"])
    retained.chmod(0o644)
    retained.write_bytes(b"changed")
    with pytest.raises(ValueError, match="historical reference"):
        experiment._check_inputs()
    with pytest.raises(ValueError, match="historical reference"):
        G.consume_global_candidate(sealed)


def test_absent_historical_reference_preserves_legacy_brief(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    row = experiment.analyze(candidate, hypothesis="Legacy no historical data")
    view = G.agent_analysis_view(row, complete_evidence="full.json")
    assert "historical_reference" not in view["analysis"]["optimization_brief"]


@pytest.mark.parametrize("missing", ["path", "sha256"])
def test_controller_requires_both_historical_reference_inputs(tmp_path, missing):
    path, digest, _ = _historical_bundle(tmp_path)
    kwargs = {"historical_reference_path": path, "historical_reference_sha256": digest}
    del kwargs["historical_reference_" + missing]
    with pytest.raises(ValueError, match="both explicit"):
        setup_experiment(tmp_path, **kwargs)


def test_historical_reference_cannot_originate_in_candidate_or_author_workspace(tmp_path):
    path, digest, _ = _historical_bundle(tmp_path)
    experiment, candidate, calls = setup_experiment(tmp_path,
        historical_reference_path=path, historical_reference_sha256=digest)
    moved = candidate / "reference.json"
    path.rename(moved)
    experiment.historical_reference_source = moved
    with pytest.raises(ValueError, match="candidate-writable"):
        experiment.analyze(candidate, hypothesis="Must not read candidate-selected reference")
    assert not calls
    moved.rename(path)
    experiment.historical_reference_source = path
    control = tmp_path / "control"
    control.mkdir()
    with pytest.raises(ValueError, match="writable author workspace"):
        experiment.stage_historical_reference(control, workspace=tmp_path)
    assert not list(control.iterdir())


def test_historical_reference_interpretation_sources_are_policy_pinned():
    sources = G.host_verification_policy_record()["sources"]
    for leaf in ("historical_reference.py", "harvest.py", "work_volume.py"):
        path = repo_root() / "merlin/python/merlin/perf" / leaf
        assert sources[str(path.resolve())] == PAS._sha256_file(path)


@pytest.mark.parametrize("verified", [False, True])
def test_analysis_only_uses_controller_once_without_probe_or_seal(tmp_path, monkeypatch, verified):
    launcher = importlib.import_module("launch_global_agent_experiment")
    experiment, candidate, calls = setup_experiment(tmp_path, verified=verified)
    configured = []
    monkeypatch.setattr(launcher, "configure_global_analysis", lambda *a, **kw: configured.append((a, kw)))
    monkeypatch.setattr(experiment, "seal", lambda *_a, **_kw: pytest.fail("analysis-only attempted sealing"))
    monkeypatch.setattr(launcher, "run_global_agent_round", lambda *_a, **_kw: pytest.fail("authoring launched"))
    status = launcher.run_analysis_only(experiment, candidate, stage_root=tmp_path,
        target_experiment="target", agent_inputs="inputs", frozen_functional="frozen",
        frozen_corpus_manifest="corpus")
    assert len(configured) == len(calls) == 1
    assert status == (0 if verified else 1)
    receipt = PAS._mapping_file(tmp_path / "analysis_only.json")
    assert all(receipt[name] is False for name in (
        "phase1_rerun", "authoring_launched", "simulators_executed", "candidate_sealed", "global_speedup_proven"))
    assert not list(experiment.output.glob("*candidate*.json"))


def test_each_proposal_recompiles_whole_graph_and_micro_plateau_cannot_stop(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)

    def propose(history):
        (candidate / "source.txt").write_text(f"candidate {len(history)}")
        if history:
            # Agent-visible context is a copy, not writable host evidence.
            history[-1]["candidate_sha256"] = "forged"
        return candidate, "Fuse full-graph producer/consumer regions"

    rows = experiment.run(propose, iterations=3)
    assert len(calls) == len(rows) == 3
    assert len(set(calls)) == 3
    assert [row["candidate_sha256"] for row in rows] == calls
    assert all(row["timing_status"] == "UNMEASURED_FULL_MODEL" for row in rows)
    assert all(row["probe_receipts"] == [] for row in rows)
    sealed = experiment.seal(candidate)
    assert G.consume_global_candidate(sealed)["global_speedup_proven"] is False


def test_input_graph_alone_does_not_admit_probe_or_seal(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path, verified=False)
    row = experiment.analyze(candidate, hypothesis="Improve layout propagation")
    assert row["readiness"]["blockers"] == ["candidate_global_plan_emission_unverified"]
    with pytest.raises(ValueError, match="global iteration is not ready"):
        experiment.seal(candidate)
    with pytest.raises(ValueError, match="global iteration is not ready"):
        experiment.measure_probe(candidate, admission_inputs={}, execute=lambda **_: pytest.fail("ran"))


@pytest.mark.parametrize("case", ["mapped", "unauthorized", "tampered"])
def test_host_guidance_surfaces_reach_full_graph_feedback_without_expanding_authority(tmp_path, case, monkeypatch):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def schedule():\n return 1\n")
    (candidate / "manifest.yaml").write_text("components:\n  emit: [compiler.py]\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "overlap", "path": "compiler.py", "symbol": "schedule"}],
        "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    declarations = [{"id": "overlap", "scope": "codegen", "path": "compiler.py", "symbol": "schedule",
        "effects": ["latency_hiding", "movement"], "cca_axes": ["dispatch.dma_overlap"],
        "mechanism": "Schedule independent transfers", "emitted_delta": "Changed issue ordering",
        "validation": "Source dependency proof and a matched short witness",
        "abandonment": "No emitted change or invalid dependency"}]
    if case == "unauthorized":
        declarations[0]["id"] = "candidate_self_grant"
        with pytest.raises(ValueError, match="outside.*authority"):
            experiment.freeze_edit_scope(candidate, contract, host_surface_declarations=declarations)
        assert not calls
        return
    experiment.freeze_edit_scope(candidate, contract, host_surface_declarations=declarations)
    if case == "tampered":
        experiment.edit_guidance_inventory = None
        with pytest.raises(ValueError, match="guidance.*changed"):
            experiment.analyze(candidate, hypothesis="Changed host guidance")
        assert not calls
        return
    # The input document remains caller-owned; subsequent changes cannot affect frozen metadata.
    declarations[0]["mechanism"] = "Invent a different operation"
    manifest = candidate / "manifest.yaml"
    manifest.write_text(manifest.read_text()+"optimization_surfaces:\n  - id: candidate_self_grant\n")
    current = experiment.inspect_optimization_surfaces(candidate)
    assert [item["id"] for item in current["surfaces"]] == ["overlap"]
    row = experiment.analyze(candidate, hypothesis="Expose the full approved global surface")
    brief = row["analysis"]["optimization_brief"]
    gap = next(item for item in brief["gap_coverage"] if item["gap"] == "latency_hiding_and_double_buffering")
    assert gap["edit_status"] == "mapped"
    assert gap["coverage_status"] == "needs_evidence"
    assert gap["edit_surfaces"][0]["mechanism"] == "Schedule independent transfers"
    assert brief["host_guidance_binding"]["permission_scope"] == "unchanged host-frozen edit contract"
    assert experiment.edit_contract == contract
    assert len(calls) == 1
    # Preparation must use the same frozen metadata, not the candidate's smaller
    # manifest or its deliberately forged optimization_surfaces declaration.
    from merlin.perf import source_convolution_preparation
    read_mapping = PAS._mapping_file
    monkeypatch.setattr(PAS, "_mapping_file", lambda path, **kwargs:
        {"entry": "frozen_entry"} if Path(path).name == "capsule.yaml"
        else read_mapping(path, **kwargs))
    monkeypatch.setattr(source_convolution_preparation, "prepare_source_convolution",
                        lambda **kwargs: {"status": "prepared"})
    preparation = experiment.prepare_source_convolution(candidate,
        comparison_arm="optimization_baseline", timeout_s=30)
    assert [surface["surface_id"] for surface in preparation["allowed_edit_surfaces"]] == ["overlap"]
    assert preparation["numerical_pass"] is False
    assert preparation["runtime_admitted"] is False


@pytest.mark.parametrize("case", ["pinned", "legacy_pin", "unpinned", "tampered", "linked"])
def test_launcher_loads_only_host_pinned_guidance_inventory(tmp_path, case):
    import launch_global_agent_experiment as launcher
    path = tmp_path / "inventory.json"
    raw = json.dumps({"surfaces": [{"id": "host_description"}]}).encode()
    path.write_bytes(raw)
    receipt = {"guidance_inventory_sha256": PAS._sha256(raw)}
    if case == "legacy_pin":
        receipt = {"unchanged_catalog_file_sha256": {"inventory.json": PAS._sha256(raw)}}
    elif case == "unpinned":
        receipt = {}
        path.write_text("invalid JSON must not be read")
    elif case == "tampered":
        path.write_bytes(raw+b" ")
    elif case == "linked":
        real = tmp_path / "real.json"
        path.rename(real)
        path.symlink_to(real)
    if case in {"tampered", "linked"}:
        with pytest.raises(ValueError, match="guidance inventory"):
            launcher.load_host_guidance_declarations(tmp_path / "edit_contract.json", receipt)
    else:
        result = launcher.load_host_guidance_declarations(tmp_path / "edit_contract.json", receipt)
        assert result == (None if case == "unpinned" else [{"id": "host_description"}])


def test_host_scope_checked_before_compile_and_again_by_sealed_consumer(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    compiler = candidate / "compiler.py"
    compiler.write_text("def allowed():\n return 1\ndef protected():\n return 2\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "lowering", "path": "compiler.py", "symbol": "allowed"}],
        "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    experiment.analyze(candidate, hypothesis="Initial frozen scope")
    compiler.write_text(compiler.read_text().replace("return 2", "return 9"))
    with pytest.raises(ValueError, match="host-frozen authority"):
        experiment.analyze(candidate, hypothesis="Unauthorized change")
    assert len(calls) == 1
    assert list(experiment.output.glob("edit_scope_refusal_*.json"))
    compiler.write_text("def allowed():\n return 7\ndef protected():\n return 2\n")
    experiment.analyze(candidate, hypothesis="Approved lowering change")
    sealed = experiment.seal(candidate)
    assert len(calls) == 2
    assert G.consume_global_candidate(sealed)["global_speedup_proven"] is False
    # A candidate cannot replace the host-frozen contract after successful compilation.
    experiment.edit_contract["existing_symbols"].append(
        {"surface_id": "forged", "path": "compiler.py", "symbol": "protected"})
    with pytest.raises(ValueError, match="edit authority"):
        experiment.validate_candidate_scope(candidate)


def _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, mechanisms):
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": name, "path": "compiler.py", "symbol": name}
        for name in ("optimize", "schedule")], "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    catalog = {"schema": "compiler_mechanism_catalog_v1",
               "contract_sha256": contract["sha256"], "mechanisms": mechanisms}
    catalog["sha256"] = PAS._document_sha256(catalog)
    path = tmp_path / "mechanism_catalog.json"
    path.write_bytes(PAS._canonical_json(catalog))
    path.chmod(0o444)
    binding = experiment.freeze_mechanism_catalog(path, PAS._sha256_file(path))
    return path, binding


def _test_mechanism_work_order(experiment, candidate, *, mechanism_id="epilogue"):
    candidate_sha256 = hash_tree(candidate)["sha256"]
    rows = []
    for index, sentinel in enumerate(experiment.portfolio_sentinels):
        rows.append({
            "capsule": sentinel.capsule, "capsule_sha256": sentinel.capsule_sha256,
            "compiler_sha256": candidate_sha256,
            "source_sha256": PAS._sha256_file(
                Path(sentinel.frozen_source_path) / "capsule.interface.mlir"),
            "plan_digest": SHA["plan"],
            "candidate_command_buffer_sha256": SHA["buffer"],
            "candidate_lowered_sha256": SHA["llvm"],
            "status": "eligible_sites_bound" if index == 0 else "complete_no_eligible_sites",
            "inventory": {"eligible_chains": 1 if index == 0 else 0},
            "source_operation_ids": [0, 1] if index == 0 else [],
            "chains": [{"producer": 0, "consumer": 1}] if index == 0 else [],
        })
    document = {
        "schema": "host_prepared_mechanism_work_order_v1",
        "status": "ready_for_authoring", "mechanism_id": mechanism_id,
        "catalog_sha256": experiment.mechanism_catalog["sha256"],
        "contract_sha256": experiment.edit_contract["sha256"],
        "initial_candidate_sha256": candidate_sha256,
        "round_start_candidate_sha256": candidate_sha256,
        "ordered_portfolio": experiment.portfolio_identity["members"],
        "portfolio_sha256": experiment.portfolio_identity_sha256,
        "source_operation_ids": [], "portfolio_site_bindings": rows,
    }
    document["sha256"] = PAS._document_sha256(document)
    return document


def _freeze_test_mechanism_work_order(
        experiment, candidate, tmp_path, *, mechanism_id="epilogue"):
    document = _test_mechanism_work_order(
        experiment, candidate, mechanism_id=mechanism_id)
    path = tmp_path / "mechanism_work_order.json"
    path.write_bytes(PAS._canonical_json(document))
    path.chmod(0o444)
    return path, experiment.freeze_mechanism_work_order(
        path, PAS._sha256_file(path), candidate=candidate)


def test_mechanism_catalog_gates_before_analysis_and_binds_iteration_checkpoint(
        tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _path, binding = _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    seed = experiment.analyze(candidate, hypothesis="bind exact seed")
    assert seed["round_mechanism_attribution"]["status"] == "initial_seed"
    experiment.begin_mechanism_round(candidate, round_index=0)
    compiler = candidate / "compiler.py"
    compiler.write_text(compiler.read_text().replace("return 1", "return 2", 1))

    gate = experiment.finalize_mechanism_round(candidate, round_index=0)
    assert gate["status"] == "allowed"
    assert gate["selected_mechanism_id"] == "epilogue"
    assert len(calls) == 1  # Attribution itself never compiles a model.
    row = experiment.analyze(candidate, hypothesis="delete exact epilogue materialization")
    assert len(calls) == 2
    assert row["compiler_mechanism_catalog"] == binding
    assert row["round_mechanism_attribution"]["candidate_sha256"] == row["candidate_sha256"]
    assert row["round_mechanism_attribution"]["selected_mechanism_id"] == "epilogue"
    checkpoint = experiment.seal(candidate)
    consumed = G.consume_global_candidate(checkpoint)
    assert consumed["compiler_mechanism_catalog"] == binding
    assert consumed["round_mechanism_attribution"] == row["round_mechanism_attribution"]


def test_multi_mechanism_candidate_is_refused_before_full_model_analyzer(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [
        {"id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}]},
        {"id": "latency", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "schedule"}]},
    ])
    experiment.analyze(candidate, hypothesis="bind exact seed")
    experiment.begin_mechanism_round(candidate, round_index=0)
    compiler = candidate / "compiler.py"
    compiler.write_text(compiler.read_text().replace("return 1", "return 2"))

    with pytest.raises(ValueError, match="one-mechanism policy"):
        experiment.analyze(candidate, hypothesis="attempt two mechanisms")
    assert len(calls) == 1
    refusal = PAS._mapping_file(next(experiment.output.glob("mechanism_analysis_refusal_*.json")))
    assert refusal["mechanism_ids"] == ["epilogue", "latency"]


def test_final_mechanism_gate_records_and_refuses_semantic_noop(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    experiment.analyze(candidate, hypothesis="bind exact seed")
    experiment.begin_mechanism_round(candidate, round_index=0)

    gate = experiment.finalize_mechanism_round(candidate, round_index=0)
    assert gate["status"] == "refused"
    assert gate["semantic_noop"] is True
    assert "no semantic compiler mechanism delta" in gate["violations"][-1]["reason"]
    with pytest.raises(ValueError, match="refused compiler mechanism round"):
        experiment.analyze(candidate, hypothesis="must not compile no-op round")
    assert len(calls) == 1


def test_mechanism_catalog_requires_raw_absolute_readonly_exact_file(tmp_path):
    experiment, candidate, _calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "optimize", "path": "compiler.py", "symbol": "optimize"}],
        "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    catalog = {"schema": "compiler_mechanism_catalog_v1",
               "contract_sha256": contract["sha256"], "mechanisms": [{
                   "id": "epilogue", "selectors": [{
                       "kind": "function", "path": "compiler.py", "symbol": "optimize"}]}]}
    catalog["sha256"] = PAS._document_sha256(catalog)
    real = tmp_path / "catalog.json"
    real.write_bytes(PAS._canonical_json(catalog))
    digest = PAS._sha256_file(real)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_catalog(Path("catalog.json"), digest)
    link = tmp_path / "catalog-link.json"
    link.symlink_to(real)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_catalog(link, digest)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_catalog(real, "0" * 64)
    real.chmod(0o444)
    binding = experiment.freeze_mechanism_catalog(real, digest)
    assert PAS._mapping_file(Path(binding["frozen_path"])) == catalog
    assert PAS._mapping_file(
        experiment.output / "compiler_mechanism_catalog_receipt.json") == binding
    frozen = Path(binding["frozen_path"])
    frozen.chmod(0o644)
    frozen.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="mechanism catalog changed"):
        experiment._check_inputs()


@pytest.mark.parametrize("flag", ["--mechanism-catalog", "--mechanism-catalog-sha256"])
def test_launcher_requires_mechanism_catalog_path_sha_and_edit_contract(flag, tmp_path):
    launcher = importlib.import_module("launch_global_agent_experiment")
    value = str((tmp_path / "catalog.json").resolve()) if flag.endswith("catalog") else "a" * 64
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--campaign-config", "not-read.json", "--candidate", "candidate",
                       "--output", "not-created", flag, value])
    assert caught.value.code == 2


def test_source_worker_receives_the_exact_mechanism_catalog_pin(tmp_path):
    launcher = importlib.import_module("launch_global_agent_experiment")
    path = (tmp_path / "catalog.json").resolve()
    digest = "a" * 64
    assert launcher._mechanism_catalog_worker_arguments(path, digest) == (
        "--mechanism-catalog", str(path), "--mechanism-catalog-sha256", digest)
    assert launcher._mechanism_catalog_worker_arguments(None, None) == ()


def test_mechanism_work_order_is_immutable_and_binds_current_member_artifacts(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    _path, binding = _freeze_test_mechanism_work_order(
        experiment, candidate, tmp_path)
    initial = experiment.analyze(candidate, hypothesis="bind exact work-order artifacts")
    required_lanes = initial["portfolio"]["members"][0]["identity"]["required_lanes"]
    initial["portfolio"]["members"][0]["identity"]["required_lanes"] = tuple(required_lanes)
    analysis_binding = experiment.bind_mechanism_work_order_analysis(initial)
    bound_seed = experiment.record_bound_mechanism_work_order_seed(candidate, initial)
    assert bound_seed["iteration"] == initial["iteration"] + 1
    assert bound_seed["mechanism_work_order_analysis"] == analysis_binding
    seed_checkpoint = experiment.seal(candidate, name="bound_seed")
    assert G.consume_global_candidate(seed_checkpoint)[
        "mechanism_work_order_analysis"] == analysis_binding
    assert analysis_binding["candidate_sha256"] == hash_tree(candidate)["sha256"]
    assert analysis_binding["members"][0]["site_binding_sha256"] == PAS._document_sha256(
        experiment.mechanism_work_order["portfolio_site_bindings"][0])
    assert PAS._mapping_file(Path(binding["frozen_path"])) == binding["work_order"]
    assert len(calls) == 1
    experiment.begin_mechanism_round(candidate, round_index=0)
    compiler = candidate / "compiler.py"
    compiler.write_text(compiler.read_text().replace("return 1", "return 2", 1))
    assert experiment.finalize_mechanism_round(
        candidate, round_index=0)["selected_mechanism_id"] == "epilogue"
    row = experiment.analyze(candidate, hypothesis="execute exact assigned mechanism")
    assert experiment.bind_mechanism_work_order_analysis(row) == analysis_binding
    assert row["compiler_mechanism_work_order"] == binding
    assert row["mechanism_work_order_analysis"] == analysis_binding
    checkpoint = experiment.seal(candidate)
    consumed = G.consume_global_candidate(checkpoint)
    assert consumed["compiler_mechanism_work_order"] == binding
    assert consumed["mechanism_work_order_analysis"] == analysis_binding
    analysis_receipt = experiment.output / "compiler_mechanism_work_order_analysis.json"
    analysis_receipt.chmod(0o644)
    analysis_receipt.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="work-order analysis"):
        G.consume_global_candidate(checkpoint)


def test_sustained_sequence_bootstraps_work_order_before_first_authoring_round(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    _freeze_test_mechanism_work_order(experiment, candidate, tmp_path)

    def author(current, *, round_index, round_timeout_s):
        experiment.begin_mechanism_round(current, round_index=round_index)
        current_analysis = experiment.analyze(
            current, hypothesis="inspect bound work-order seed")
        assert experiment.bind_mechanism_work_order_analysis(current_analysis) == \
            experiment.mechanism_work_order_analysis_binding
        compiler = current / "compiler.py"
        compiler.write_text(compiler.read_text().replace("return 1", "return 2", 1))
        assert experiment.finalize_mechanism_round(
            current, round_index=round_index)["status"] == "allowed"
        experiment.analyze(current, hypothesis="execute assigned epilogue mechanism")
        return {"status": "authored"}

    result = G.run_global_agent_sequence(
        experiment, candidate, run_round=author, stage_root=tmp_path / "stage",
        max_rounds=1, total_authoring_seconds=30, round_seconds=30,
        on_round_failure="stop")

    assert result["status"] == "budget_complete"
    assert result["failures"] == []
    assert len(calls) == 2
    assert experiment.iterations[0]["mechanism_work_order_analysis"] is None
    assert experiment.iterations[1]["mechanism_work_order_analysis"] == \
        experiment.mechanism_work_order_analysis_binding
    consumed = G.consume_global_candidate(Path(result["last_good_checkpoint"]["path"]))
    assert consumed["mechanism_work_order_analysis"] == \
        experiment.mechanism_work_order_analysis_binding


@pytest.mark.parametrize("mutation", ["candidate", "catalog", "portfolio", "self_hash",
                                      "flat_sites", "missing_member"])
def test_mechanism_work_order_identity_mutations_fail_closed(tmp_path, mutation):
    experiment, candidate, _calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    document = _test_mechanism_work_order(experiment, candidate)
    if mutation == "candidate":
        document["round_start_candidate_sha256"] = "0" * 64
    elif mutation == "catalog":
        document["catalog_sha256"] = "0" * 64
    elif mutation == "portfolio":
        document["portfolio_sha256"] = "0" * 64
    elif mutation == "flat_sites":
        document["source_operation_ids"] = [0]
    elif mutation == "missing_member":
        document["portfolio_site_bindings"] = []
    if mutation != "self_hash":
        document["sha256"] = PAS._document_sha256({
            key: value for key, value in document.items() if key != "sha256"})
    else:
        document["sha256"] = "0" * 64
    path = tmp_path / f"work-order-{mutation}.json"
    path.write_bytes(PAS._canonical_json(document))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="mechanism work order|mechanism work-order"):
        experiment.freeze_mechanism_work_order(
            path, PAS._sha256_file(path), candidate=candidate)


def test_mechanism_work_order_current_analysis_hash_drift_refuses_before_authoring(tmp_path):
    experiment, candidate, _calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    document = _test_mechanism_work_order(experiment, candidate)
    document["portfolio_site_bindings"][0]["plan_digest"] = "0" * 64
    document["sha256"] = PAS._document_sha256({
        key: value for key, value in document.items() if key != "sha256"})
    path = tmp_path / "work-order.json"
    path.write_bytes(PAS._canonical_json(document))
    path.chmod(0o444)
    experiment.freeze_mechanism_work_order(
        path, PAS._sha256_file(path), candidate=candidate)
    experiment.begin_mechanism_round(candidate, round_index=0)
    initial = experiment.analyze(candidate, hypothesis="current exact analysis")
    with pytest.raises(ValueError, match="differs from current analysis"):
        experiment.bind_mechanism_work_order_analysis(initial)


def test_mechanism_work_order_keeps_repeated_source_ids_bound_per_four_members(tmp_path):
    portfolio = []
    for index in range(3):
        source = tmp_path / f"training-{index}"
        source.mkdir()
        (source / "capsule.yaml").write_text("interface_mlir: capsule.interface.mlir\n")
        (source / "capsule.interface.mlir").write_bytes(b"x" * (index + 2))
        portfolio.append(PAS.StageE2ESentinel(
            f"training-{index}", str(source), str(source),
            PAS._exact_tree_record(source)["sha256"], (), ()))
    experiment, candidate, calls = setup_experiment(
        tmp_path, portfolio_sentinels=tuple(portfolio))
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    document = _test_mechanism_work_order(experiment, candidate)
    for row in document["portfolio_site_bindings"]:
        row["status"] = "eligible_sites_bound"
        row["inventory"] = {"eligible_chains": 1}
        row["source_operation_ids"] = [0, 1]
        row["chains"] = [{"producer": 0, "consumer": 1}]
    document["sha256"] = PAS._document_sha256({
        key: value for key, value in document.items() if key != "sha256"})
    path = tmp_path / "four-model-work-order.json"
    path.write_bytes(PAS._canonical_json(document))
    path.chmod(0o444)
    experiment.freeze_mechanism_work_order(
        path, PAS._sha256_file(path), candidate=candidate)
    experiment.begin_mechanism_round(candidate, round_index=0)
    initial = experiment.analyze(candidate, hypothesis="four exact graphs")
    binding = experiment.bind_mechanism_work_order_analysis(initial)
    assert len(binding["members"]) == 4
    assert len(calls) == 4
    assert [member["capsule"] for member in binding["members"]] == [
        member.capsule for member in experiment.portfolio_sentinels]


def test_mechanism_work_order_requires_raw_absolute_readonly_exact_file(tmp_path):
    experiment, candidate, _calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text(
        "def optimize():\n    return 1\n\ndef schedule():\n    return 1\n")
    _freeze_test_mechanism_catalog(experiment, candidate, tmp_path, [{
        "id": "epilogue", "selectors": [
            {"kind": "function", "path": "compiler.py", "symbol": "optimize"}],
    }])
    document = _test_mechanism_work_order(experiment, candidate)
    path = tmp_path / "work-order.json"
    path.write_bytes(PAS._canonical_json(document))
    digest = PAS._sha256_file(path)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_work_order(
            Path("work-order.json"), digest, candidate=candidate)
    link = tmp_path / "work-order-link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_work_order(link, digest, candidate=candidate)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_work_order(path, digest, candidate=candidate)
    path.chmod(0o444)
    with pytest.raises(ValueError, match="immutable absolute"):
        experiment.freeze_mechanism_work_order(path, "0" * 64, candidate=candidate)
    experiment.freeze_mechanism_work_order(path, digest, candidate=candidate)
    path.chmod(0o644)
    path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="mechanism work order changed"):
        experiment._check_inputs()


def test_exact_active_v14_work_order_matches_retained_four_model_analysis():
    root = PAS.repo_root()
    artifact = root / ("out/artifacts/perf-bench/gemmini/"
        "development_v14_epilogue_site_bindings_20260907/active_t01_01_work_order.json")
    catalog_path = root / ("out/artifacts/perf-bench/gemmini/"
        "development_v14_mechanism_catalog_20260907/active_t01_01_mechanism_catalog.json")
    contract_path = root / ("out/artifacts/perf-bench/gemmini/"
        "development_v14_retained_surface_catalog_20260907/edit_contract.json")
    retained = root / ("out/artifacts/perf-bench/gemmini/"
        "global_phase2_portfolio_r50_tiny_lstm_smol_w8a8_v13_20260907/global_iterations")
    candidate, iteration_path, checkpoint_path = (
        retained / "round_0000_candidate_submission", retained / "iteration_0001.json",
        retained / "round_0000_candidate.json")
    required = (artifact, catalog_path, contract_path, candidate, iteration_path, checkpoint_path)
    if not all(path.exists() for path in required):
        pytest.skip("exact local V14 host-preparation artifacts are not installed")
    from merlin.perf.compiler_edit_scope import validate_edit_contract, validate_mechanism_catalog
    work, catalog, contract = (PAS._mapping_file(path)
                               for path in (artifact, catalog_path, contract_path))
    checkpoint, iteration = PAS._mapping_file(checkpoint_path), PAS._mapping_file(iteration_path)
    assert PAS._sha256_file(artifact) == (
        "6f8b11ba12091f4e5936a57b3524c61ba60c366aa42f6b088a9c0e967a358f64")
    validate_edit_contract(contract, candidate)
    validate_mechanism_catalog(catalog, candidate, contract)
    validator = object.__new__(G.GlobalPerfExperiment)
    validator.mechanism_catalog, validator.edit_contract = catalog, contract
    validator.portfolio_identity = checkpoint["portfolio"]
    validator.portfolio_identity_sha256 = PAS._document_sha256(validator.portfolio_identity)
    validated = validator._validate_mechanism_work_order(
        work, candidate_sha256=hash_tree(candidate)["sha256"])
    assert validated["sha256"] == "0c38b0a3417b252630dbbe5fe0cfb4910a3d5e92aad8731922d7fc8ffa2dfedf"
    assert len(validated["portfolio_site_bindings"]) == 4
    for index, site in enumerate(validated["portfolio_site_bindings"]):
        analysis = (iteration["analysis"] if index == 0
                    else iteration["portfolio"]["members"][index]["analysis"])
        plan, emission = analysis["diagnostics"]["verified_global_plan_emission"], analysis["emission"]
        assert {key: site[key] for key in (
            "compiler_sha256", "source_sha256", "plan_digest",
            "candidate_command_buffer_sha256", "candidate_lowered_sha256")} == {
                "compiler_sha256": iteration["candidate_sha256"],
                "source_sha256": plan["source_sha256"], "plan_digest": plan["plan_digest"],
                "candidate_command_buffer_sha256": emission["candidate_command_buffer_sha256"],
                "candidate_lowered_sha256": emission["candidate_lowered_sha256"],
            }


@pytest.mark.parametrize("flag", ["--mechanism-work-order",
                                  "--mechanism-work-order-sha256"])
def test_launcher_requires_mechanism_work_order_path_sha_and_catalog(flag, tmp_path):
    launcher = importlib.import_module("launch_global_agent_experiment")
    value = str((tmp_path / "work-order.json").resolve()) if flag.endswith(
        "work-order") else "a" * 64
    with pytest.raises(SystemExit) as caught:
        launcher.main(["--campaign-config", "not-read.json", "--candidate", "candidate",
                       "--output", "not-created", flag, value])
    assert caught.value.code == 2


def test_source_worker_receives_the_exact_mechanism_work_order_pin(tmp_path):
    launcher = importlib.import_module("launch_global_agent_experiment")
    path = (tmp_path / "work-order.json").resolve()
    digest = "b" * 64
    assert launcher._mechanism_work_order_worker_arguments(path, digest) == (
        "--mechanism-work-order", str(path), "--mechanism-work-order-sha256", digest)
    assert launcher._mechanism_work_order_worker_arguments(None, None) == ()


def test_context_provider_installation_is_not_current_motif_applicability():
    analysis = {"emission": {"candidate_lowered_sha256": SHA["llvm"]}, "diagnostics": {
        "verified_global_plan_emission": {"status": "verified", "candidate_lowered_sha256": SHA["llvm"]},
        "queued_movement_context": {"schema": "queued_movement_context_candidates_v1",
                                    "artifact_sha256": SHA["llvm"], "motifs": []}}}
    empty = G.controlled_context_capability(analysis, provider_installed=True)
    assert empty["provider_installed"] and not empty["available"]
    assert empty["current_candidate_status"] == "no_extracted_supported_motifs"
    analysis["diagnostics"]["queued_movement_context"]["motifs"] = [{"index": 1}]
    present = G.controlled_context_capability(analysis, provider_installed=True)
    assert present["available"] and present["admission_verified"] is False
    assert not G.controlled_context_capability(analysis, provider_installed=False)["available"]
    analysis["diagnostics"]["queued_movement_context"]["artifact_sha256"] = SHA["buffer"]
    assert G.controlled_context_capability(analysis, provider_installed=True)["current_candidate_status"] == "UNKNOWN"


@pytest.mark.parametrize("bad_contract", [False, True])
@pytest.mark.parametrize("counter_mode", ["same", "missing", "different_engine"])
def test_paired_context_same_work_binding_and_scoped_receipt(tmp_path, bad_contract, counter_mode):
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Before schedule")
    (candidate / "source.txt").write_text("reordered schedule")
    experiment.analyze(candidate, hypothesis="After schedule")
    for index, field in ((0, "_previous_artifacts"), (1, "_artifacts")):
        setattr(experiment, field, {
            "candidate_sha256": experiment.iterations[index]["candidate_sha256"],
            "candidate_lowered_sha256": SHA["llvm"],
            "decoded_trace": {"instructions": [{}] * 100}})
    calls = []
    work = {"timed_command_count": 3, "timed_command_multiset": ["load", "compute"],
            "source_task_index": 7, "source_op_indices": [18]}
    input_contract = {"wrapper_sha256": SHA["llvm"], "input_bytes": 16}
    digest = PAS._document_sha256(work)
    input_digest = PAS._document_sha256(input_contract)

    def provider(**kwargs):
        arms = {}
        for arm in ("before", "after"):
            directory = tmp_path / arm
            directory.mkdir()
            (directory / "primitive.elf").write_bytes(arm.encode())
            source = {"source_artifact_sha256": SHA["llvm"], "slice_source_sha256": SHA["buffer"],
                      "work_contract_sha256": digest, "timed_source_instruction_indices": [1, 2, 3],
                      "source_instruction_indices": [0, 1, 2, 3], "full_model_executed": False,
                      "full_layer_executed": False, "source_task_compute_pair_count": 2,
                      "executed_compute_pair_count": 1}
            prepared = {"source_artifact_sha256": SHA["buffer"], "measurement_scope": "controlled_fixed_work_slice",
                        "simulator_executed": False, "timed_instruction_count": 3, "host_input_bytes": 16,
                        "output_storage_bytes": 16, "workdir": str(directory),
                        "elf_sha256": PAS._sha256_file(directory / "primitive.elf"),
                        "wrapper_sha256": SHA["llvm"], "primitive_mlir_sha256": SHA["buffer"],
                        "domain_digest": SHA["plan"]}
            arms[arm] = {"source_slice": source, "prepared": prepared,
                         "work_contract_sha256": digest,
                         "deterministic_input_contract_sha256": input_digest}
        if bad_contract:
            arms["after"]["work_contract_sha256"] = SHA["target"]

        def execute(*, arm, timeout_s):
            calls.append(arm)
            assert 0 < timeout_s <= 60
            result = {**arms[arm]["prepared"], "correct": True, "warmup_runs": 1, "measured_runs": 1,
                    "full_model_executed": False, "full_source_probe_executed": False,
                    "total_compute_cycles": 100 if arm == "before" else 99}
            if counter_mode != "missing":
                result["engine_provenance"] = {"binary_sha256": SHA["target"]
                    if arm == "before" or counter_mode == "same" else SHA["plan"]}
                result["counter_profile"] = {"kind": "joint_engine_busy_cycles",
                    "partition_proof": {"status": "proved"}, "layout": {"complete": True},
                    "active_union_cycles": 90, "idle_cycles": 10 if arm == "before" else 9,
                    "overlap_any_engine_cycles": 30,
                    "busy_cycles_by_engine_token": {"compute": 50, "movement": 70}}
            return result

        return {"execute": execute, "paired_context_inputs": {
            "before_binding": experiment.previous_probe_binding(candidate),
            "after_binding": experiment.current_probe_binding(candidate),
            "before_model_artifact_sha256": SHA["llvm"], "after_model_artifact_sha256": SHA["llvm"],
            "arms": arms, "work_contract_sha256": digest,
            "deterministic_input_contract": input_contract, "deterministic_input_contract_sha256": input_digest,
            "projection_proof": {"schema": "controlled_fixed_work_projection_v1",
                "status": "same_work_projection_verified", "scope": "controlled_fixed_work_slice",
                "future_computes_omitted_symmetrically": True, "global_cost_validated": False,
                "before_artifact_sha256": SHA["llvm"], "after_artifact_sha256": SHA["llvm"],
                "before_timed_indices": [1, 2, 3], "after_timed_indices": [1, 2, 3],
                "work_contract": work, "work_contract_sha256": digest}}}

    if bad_contract:
        with pytest.raises(ValueError, match="changed source identity, work"):
            experiment.compare_controlled_context(candidate, provider=provider, timeout_s=60)
        assert calls == []
    else:
        receipt = experiment.compare_controlled_context(candidate, provider=provider, timeout_s=60)
        assert calls == ["before", "after"]
        assert receipt["after_minus_before_cycles"] == -1
        assert receipt["global_speedup_proven"] is False
        assert receipt["full_model_cycles"] is None
        feedback = receipt["decision_feedback"]
        assert feedback["status"] == ("no_observed_overlap_or_busy_work_change"
            if counter_mode == "same" else "counter_evidence_unknown")
        assert feedback["source_op_indices"] == [18]
        assert feedback["pipeline_projection_admitted"] is False
        view = G.agent_analysis_view(experiment.iterations[-1], complete_evidence="complete.json")
        assert view["measurement_driven_next_step"] == feedback["next_step"]
        coverage = view["analysis"]["optimization_brief"]["scoped_mechanism_coverage"]
        assert coverage["full_model_occupancy"] == "UNKNOWN"
        assert "scoped_mechanism_coverage" not in experiment.iterations[-1]["analysis"].get("optimization_brief", {})
        sealed = experiment.seal(candidate)
        assert len(G.consume_global_candidate(sealed)["paired_context_receipts"]) == 1
        stale = copy.deepcopy(receipt)
        stale["binding"]["compiler_digest"] = SHA["target"]
        with pytest.raises(ValueError, match="not bound"):
            G.paired_context_decision_feedback(experiment.iterations[-1], stale, target_sha256=SHA["target"])
        (candidate / "source.txt").write_text("next schedule")
        following = experiment.analyze(candidate, hypothesis="Use prior measurement as search history")
        assert following["static_comparison"]["previous_revision_mechanism_feedback"]["applies_to_current_revision"] is False
        assert "decision_feedback" not in following


def test_exact_unchanged_analysis_reuses_identity_without_compiling(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    first = experiment.analyze(candidate, hypothesis="Inspect graph")
    second = experiment.analyze(candidate, hypothesis="Read the same evidence")
    assert len(calls) == len(experiment.iterations) == 1
    assert second["exact_analysis_reused"] is True
    assert first["candidate_sha256"] == second["candidate_sha256"]
    receipt = second["analysis_reuse_receipt"]
    assert PAS._sha256_file(Path(receipt["path"])) == receipt["sha256"]
    evidence = PAS._mapping_file(Path(receipt["path"]))
    assert evidence["duplicate_current_revision"] is True
    assert evidence["full_graph_compiler_invoked"] is False


def test_reverted_candidate_reuses_ready_static_analysis_as_new_iteration(tmp_path):
    extra = _portfolio_sentinel(tmp_path, "second-model")
    experiment, candidate, calls = setup_experiment(
        tmp_path, portfolio_sentinels=[extra])
    original = (candidate / "source.txt").read_text()
    first = experiment.analyze(candidate, hypothesis="Inspect original revision")
    # Later dynamic evidence belongs only to its originating iteration and must not
    # become evidence for the chronologically new revisit.
    experiment.iterations[0]["probe_receipts"].append({"path": "timed.json", "sha256": "a" * 64})
    experiment.iterations[0]["decision_feedback"] = {"status": "measured_elsewhere"}
    experiment._iteration_artifacts[0] = {
        "candidate_sha256": first["candidate_sha256"],
        "candidate_lowered_sha256": SHA["llvm"], "marker": "original"}
    source_policy = {"candidate": {
        "package_path": first["submitted_snapshot"],
        "compiler_dependencies": first["compiler_dependencies"]}}
    experiment._compiler_sandboxes[0] = copy.deepcopy(source_policy)

    (candidate / "source.txt").write_text("rejected speculative revision")
    second = experiment.analyze(candidate, hypothesis="Inspect speculative revision")
    experiment._iteration_artifacts[1] = {
        "candidate_sha256": second["candidate_sha256"],
        "candidate_lowered_sha256": SHA["llvm"], "marker": "speculative"}
    experiment._artifacts = copy.deepcopy(experiment._iteration_artifacts[1])
    (candidate / "source.txt").write_text(original)

    revisited = experiment.analyze(candidate, hypothesis="Return to qualified checkpoint")

    assert len(calls) == 4  # two models for each of the two novel revisions
    assert len(experiment.iterations) == 3
    assert revisited["iteration"] == 2
    assert revisited["candidate_sha256"] == first["candidate_sha256"]
    assert revisited["submitted_snapshot"] == first["submitted_snapshot"]
    assert revisited["hypothesis"] == "Return to qualified checkpoint"
    assert revisited["readiness"]["status"] == "ready_for_probe_admission"
    assert revisited["static_comparison"]["previous_iteration"] == 1
    assert revisited["portfolio"]["members"][1]["static_comparison"]["previous_iteration"] == 1
    assert revisited["portfolio"]["analysis_allocation_policy"] == \
        "exact_immutable_ready_iteration_reuse_no_compilation"
    assert revisited["portfolio"]["analysis_concurrency"]["admitted_workers"] == 0
    assert all(member["analysis_allocation"]["allocated_seconds"] == 0
               for member in revisited["portfolio"]["members"])
    reuse = revisited["analysis_reuse"]
    assert reuse["source_iteration"] == 0
    assert reuse["result_iteration"] == 2
    assert reuse["full_graph_compiler_invoked"] is False
    assert reuse["probe_or_timing_receipts_reused"] is False
    assert revisited["probe_receipts"] == []
    assert "decision_feedback" not in revisited
    assert experiment.current_artifacts(candidate)["marker"] == "original"
    assert experiment.previous_artifacts(candidate)["marker"] == "speculative"
    assert experiment._compiler_sandboxes[2] == source_policy
    stored = PAS._mapping_file(experiment.output / "iteration_0002.json")
    assert stored == experiment.iterations[2]
    sealed = G.consume_global_candidate(experiment.seal(candidate, name="revisited"))
    assert sealed["iteration"] == 2


@pytest.mark.parametrize("unsafe_source", ["blocked", "wall_timeout"])
def test_reverted_candidate_recompiles_an_unsafe_prior_result(
        tmp_path, monkeypatch, unsafe_source):
    experiment, candidate, _ = setup_experiment(
        tmp_path, timeout_s=10 if unsafe_source == "wall_timeout" else 300)
    delegate = experiment.analyzer
    original = (candidate / "source.txt").read_text()
    invocations = []
    fail_first = [True]
    now = [0.0]
    if unsafe_source == "wall_timeout":
        monkeypatch.setattr(G.time, "monotonic", lambda: now[0])

    def analyzer(base, current, objective, **kwargs):
        invocations.append((current / "source.txt").read_text())
        if fail_first[0] and unsafe_source == "blocked":
            fail_first[0] = False
            return {"candidate_sha256": hash_tree(current)["sha256"],
                    "workload": {"capsule_sha256": objective.capsule_sha256},
                    "diagnostics": {"arms": {"candidate": {"status": "emission_failed"}}},
                    "timing_status": "UNMEASURED"}
        result = delegate(base, current, objective, **kwargs)
        if fail_first[0]:
            fail_first[0] = False
            now[0] += 11.0
        return result

    experiment.analyzer = analyzer
    unsafe = experiment.analyze(candidate, hypothesis="Initial unsafe result")
    assert unsafe["readiness"]["status"] == "blocked"
    if unsafe_source == "wall_timeout":
        assert unsafe["readiness"]["blockers"] == ["iteration_wall_budget_exceeded"]
    (candidate / "source.txt").write_text("safe intervening revision")
    experiment.analyze(candidate, hypothesis="Compile intervening revision")
    (candidate / "source.txt").write_text(original)

    retried = experiment.analyze(candidate, hypothesis="Retry original bytes")

    assert len(invocations) == 3
    assert retried["iteration"] == 2
    assert retried["readiness"]["status"] == "ready_for_probe_admission"
    assert "analysis_reuse" not in retried


def test_reverted_candidate_does_not_reuse_across_compiler_dependency_change(
        tmp_path, monkeypatch):
    experiment, candidate, calls = setup_experiment(tmp_path)
    original = (candidate / "source.txt").read_text()
    experiment.analyze(candidate, hypothesis="Initial dependency closure")
    (candidate / "source.txt").write_text("intervening revision")
    experiment.analyze(candidate, hypothesis="Changed candidate")
    (candidate / "source.txt").write_text(original)
    dependencies = experiment._compiler_dependencies

    def changed_dependencies(package):
        record = dependencies(package)
        if (package.resolve() == candidate.resolve()
                or package.parent.resolve() == experiment.output.resolve()
                and package.name.startswith("submission_")):
            record = {**record, "test_dependency_epoch": 1}
        return record

    monkeypatch.setattr(experiment, "_compiler_dependencies", changed_dependencies)
    retried = experiment.analyze(candidate, hypothesis="New shared dependency closure")
    assert len(calls) == 3
    assert "analysis_reuse" not in retried
    assert retried["compiler_dependencies"]["test_dependency_epoch"] == 1


def test_reverted_candidate_recompiles_if_prior_snapshot_lost_immutability(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    original = (candidate / "source.txt").read_text()
    first = experiment.analyze(candidate, hypothesis="Capture immutable original")
    (candidate / "source.txt").write_text("intervening revision")
    experiment.analyze(candidate, hypothesis="Compile intervening revision")
    (candidate / "source.txt").write_text(original)
    submitted_file = Path(first["submitted_snapshot"]) / "source.txt"
    submitted_file.chmod(0o644)

    retried = experiment.analyze(candidate, hypothesis="Do not trust writable cache source")

    assert len(calls) == 3
    assert retried["iteration"] == 2
    assert "analysis_reuse" not in retried
    assert retried["submitted_snapshot"] != first["submitted_snapshot"]


def test_analysis_passes_frozen_host_verifier_policy_to_worker(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    delegate = experiment.analyzer

    def analyzer(*args, **kwargs):
        assert kwargs["host_verifier_policy_sha256"] == experiment.host_policy["sha256"]
        return delegate(*args, **kwargs)

    experiment.analyzer = analyzer
    experiment.analyze(candidate, hypothesis="Verify the host-bound baseline plan")


def test_reverted_candidate_reuses_earlier_exact_verified_analysis_for_sealing(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    original = (candidate / "source.txt").read_text()
    first = experiment.analyze(candidate, hypothesis="Inspect original revision")
    (candidate / "source.txt").write_text("speculative revision")
    experiment.analyze(candidate, hypothesis="Inspect speculative revision")
    (candidate / "source.txt").write_text(original)

    current = experiment._current(candidate)
    assert current["candidate_sha256"] == first["candidate_sha256"]
    assert len(calls) == 2
    sealed = G.consume_global_candidate(experiment.seal(candidate, name="reverted"))
    assert sealed["candidate_sha256"] == first["candidate_sha256"]


@pytest.mark.parametrize("failure", [None, "author", "author_first", "integrity", "capacity"])
def test_sustained_sequence_checkpoints_and_explicit_recovery(tmp_path, failure):
    experiment, candidate, _ = setup_experiment(tmp_path)
    invoked, inherited = [], []

    def author(current, *, round_index, round_timeout_s):
        invoked.append((round_index, round_timeout_s))
        inherited.append((current / "source.txt").read_text())
        (current / "source.txt").write_text(f"draft round {round_index}")
        experiment.analyze(current, hypothesis="actual full graph test analysis")
        if failure and round_index == (0 if failure == "author_first" else 1):
            broker_complete = failure not in ("integrity", "capacity")
            experiment._write(f"agent_round_{round_index:04d}.json", {
                "agent_exit_code": 1 if failure == "capacity" else -15,
                "audit": {"clean": failure != "integrity", "broker_invocations": []},
                "broker_evidence": {"all_required_succeeded": broker_complete}})
            if failure == "capacity":
                rounds = kwargs["stage_root"] / "rounds"
                rounds.mkdir(parents=True, exist_ok=True)
                (rounds / f"round_{round_index:02d}.codex_summary.json").write_text(json.dumps({
                    "exit_code": 1, "timed_out": False, "turns_started": 1,
                    "turns_usage_reported": 0,
                    "errors": ["Selected model is at capacity. Please try a different model."]}))
            raise ValueError("terminal author failure" if failure == "author" else "integrity failure")
        return {"status": "authored"}

    kwargs = dict(run_round=author, stage_root=tmp_path / "stage", max_rounds=3,
                  total_authoring_seconds=70, round_seconds=30,
                  on_round_failure="resume-last-checkpoint")
    if failure == "integrity":
        with pytest.raises(ValueError, match="integrity failure"):
            G.run_global_agent_sequence(experiment, candidate, **kwargs)
        assert len(invoked) == 2
        return
    result = G.run_global_agent_sequence(experiment, candidate, **kwargs)
    assert invoked == [(0, 30), (1, 30), (2, 10)]
    assert inherited == ["candidate", "candidate" if failure == "author_first" else "draft round 0",
                         "draft round 0" if failure in ("author", "capacity") else "draft round 1"]
    assert result["authoring_seconds_reserved"] == 70
    assert result["checkpoints"][0]["role"] == "initial_verified_seed_not_an_authored_result"
    assert len(result["checkpoints"]) == (3 if failure else 4)
    assert G.consume_global_candidate(Path(result["last_good_checkpoint"]["path"]))["global_speedup_proven"] is False
    if failure:
        assert result["failures"][0]["live_handle_restarted"] is False
        assert result["failures"][0]["retryable_capacity_failure"] is (failure == "capacity")
        assert (Path(result["failures"][0]["draft_path"]) / "source.txt").read_text() == (
            "draft round 0" if failure == "author_first" else "draft round 1")


@pytest.mark.parametrize(("budget", "reserve"), [(30, 0), (120, 30), (299, 30),
                                                  (300, 100), (600, 180)])
def test_macro_round_reserves_a_bounded_final_response_window(budget, reserve):
    assert G._agent_finalization_reserve_seconds(budget) == reserve


def test_clean_reverted_round_can_continue_without_admitting_a_candidate(tmp_path):
    terminal = {
        "agent_exit_code": 0,
        "candidate_sha256": SHA["graph"],
        "audit": {"clean": True},
        "broker_evidence": {
            "status": "refused",
            "reason": "global broker required actions did not complete: ['analyze-whole-model']",
        },
    }
    assert G._retryable_unchanged_round_failure(
        terminal, checkpoint_sha256=SHA["graph"])
    terminal["candidate_sha256"] = SHA["target"]
    assert not G._retryable_unchanged_round_failure(
        terminal, checkpoint_sha256=SHA["graph"])

    terminal.update({
        "candidate_sha256": SHA["graph"],
        "audit": {"clean": False, "hits": [{"kind": "invalid_broker_invocation"}]},
        "broker_evidence": {"all_required_succeeded": True},
    })
    assert G._retryable_unchanged_round_failure(
        terminal, checkpoint_sha256=SHA["graph"])
    terminal["audit"]["hits"].append({"kind": "candidate_code_copied_outside"})
    assert not G._retryable_unchanged_round_failure(
        terminal, checkpoint_sha256=SHA["graph"])


@pytest.mark.parametrize("exit_code", [0, 124])
def test_sequence_restarts_clean_unchanged_round_from_consumed_checkpoint(tmp_path, exit_code):
    experiment, candidate, _ = setup_experiment(tmp_path)
    observed = []

    def author(current, *, round_index, round_timeout_s):
        observed.append((round_index, (current / "source.txt").read_text()))
        if round_index == 0:
            experiment._write("agent_round_0000.json", {
                "agent_exit_code": exit_code,
                "candidate_sha256": hash_tree(current)["sha256"],
                "audit": {"clean": True},
                "broker_evidence": {
                    "status": "refused",
                    "reason": "global broker required actions did not complete: "
                              "['analyze-whole-model']",
                },
            })
            raise ValueError("clean reverted round made no admitted edit")
        (current / "source.txt").write_text("verified second round")
        experiment.analyze(current, hypothesis="second round full graph")
        return {"status": "authored"}

    result = G.run_global_agent_sequence(
        experiment, candidate, run_round=author, stage_root=tmp_path / "stage",
        max_rounds=2, total_authoring_seconds=60, round_seconds=30,
        on_round_failure="resume-last-checkpoint")
    assert observed == [(0, "candidate"), (1, "candidate")]
    assert result["failures"][0]["retryable_unchanged_round_failure"] is True
    assert result["failures"][0]["recovery"] == "next_budgeted_round_from_consumed_checkpoint"
    assert len(result["checkpoints"]) == 2


@pytest.mark.parametrize("failure", ["changed_draft", "dirty_audit", "binding_refusal",
                                    "missing_checkpoint_digest", "other_exit"])
def test_unchanged_timeout_recovery_does_not_admit_unbound_or_unsafe_rounds(failure):
    terminal = {
        "agent_exit_code": 124,
        "candidate_sha256": SHA["graph"],
        "audit": {"clean": True, "hits": []},
        "broker_evidence": {
            "status": "refused",
            "reason": "global broker required actions did not complete: ['analyze-whole-model']",
        },
    }
    if failure == "changed_draft":
        terminal["candidate_sha256"] = SHA["target"]
    elif failure == "dirty_audit":
        terminal["audit"] = {"clean": False,
                             "hits": [{"kind": "invalid_broker_invocation"}]}
        terminal["broker_evidence"] = {"all_required_succeeded": True}
    elif failure == "binding_refusal":
        terminal["broker_evidence"]["reason"] = "candidate binding changed"
    elif failure == "missing_checkpoint_digest":
        terminal.pop("candidate_sha256")
    else:
        terminal["agent_exit_code"] = 1
    assert not G._retryable_unchanged_round_failure(
        terminal, checkpoint_sha256=SHA["graph"])


@pytest.mark.parametrize("mutation", ["baseline", "checkpoint", "host_policy"])
def test_timeout_recovery_rechecks_checkpoint_and_host_bindings(tmp_path, monkeypatch, mutation):
    experiment, candidate, _ = setup_experiment(tmp_path)
    invoked = []

    def author(current, *, round_index, round_timeout_s):
        invoked.append(round_index)
        experiment._write("agent_round_0000.json", {
            "agent_exit_code": 124,
            "candidate_sha256": hash_tree(current)["sha256"],
            "audit": {"clean": True},
            "broker_evidence": {
                "status": "refused",
                "reason": "global broker required actions did not complete: ['analyze-whole-model']",
            },
        })
        if mutation == "baseline":
            (experiment.baseline / "source.txt").write_text("mutated baseline")
        elif mutation == "checkpoint":
            checkpoint = experiment.output / "initial_seed_candidate.json"
            checkpoint.chmod(0o644)
            checkpoint.write_text("{}")
        else:
            monkeypatch.setattr(G, "host_verification_policy_record", lambda: {"sha256": "changed"})
        raise ValueError("terminal timeout; retained binding no longer valid")

    with pytest.raises(ValueError, match="terminal timeout"):
        G.run_global_agent_sequence(
            experiment, candidate, run_round=author, stage_root=tmp_path / "stage",
            max_rounds=2, total_authoring_seconds=60, round_seconds=30,
            on_round_failure="resume-last-checkpoint")
    assert invoked == [0]
    failure = PAS._mapping_file(experiment.output / "continuation_failure_0000.json")
    assert failure["retryable_unchanged_round_failure"] is True
    assert failure["recovery"] == "stop"


def test_prior_round_context_carries_agent_memory_but_labels_it_untrusted(tmp_path):
    rounds = tmp_path / "rounds"
    audits = tmp_path / "global_iterations"
    rounds.mkdir()
    audits.mkdir()
    (rounds / "round_00.final.txt").write_text("Next: fuse the surviving temporary.\n")
    (audits / "agent_round_0000.json").write_text(json.dumps({
        "status": "refused", "candidate_sha256": SHA["graph"],
        "agent_exit_code": 0, "refusal_reasons": ["analysis deadline"],
    }))
    context = G._prior_round_context(tmp_path, 1)
    assert context["schema"] == "global_prior_round_context_v1"
    assert "untrusted" in context["interpretation"]
    assert context["rounds"][0]["agent_summary"].startswith("Next:")
    assert context["rounds"][0]["host_round_audit"]["status"] == "refused"


def test_concurrent_identical_analysis_waits_and_reuses_one_snapshot(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    experiment, candidate, calls = setup_experiment(tmp_path)
    entered, release = threading.Event(), threading.Event()
    delegate = experiment.analyzer

    def analyze(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        return delegate(*args, **kwargs)

    experiment.analyzer = analyze
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(experiment.analyze, candidate, hypothesis="First request")
        assert entered.wait(2)
        duplicate = pool.submit(experiment.analyze, candidate, hypothesis="Duplicate request")
        release.set()
        assert first.result()["iteration"] == 0
        assert duplicate.result()["exact_analysis_reused"] is True
    assert len(calls) == len(experiment.iterations) == 1


def test_probe_reuses_exact_prepared_policy_and_cannot_unmask_answers(tmp_path):
    from merlin.perf.analysis_worker import IsolatedAnalysisWorker
    experiment, candidate, _ = setup_experiment(tmp_path)
    row = experiment.analyze(candidate, hypothesis="Inspect graph")
    answer, scratch = tmp_path / "answers", tmp_path / "public_probe"
    answer.mkdir()
    scratch.mkdir()
    analyzer = IsolatedAnalysisWorker(stage_path=Path(PAS.__file__), output=tmp_path / "worker",
        sandbox_factory=lambda *_: pytest.fail("reconstructed an already prepared policy"))
    prefix = ["bwrap", "--clearenv", "--tmpfs", str(answer), "bash", "-c", 'exec "$@"', "perf-tool"]
    analyzer.completed_sandboxes = {"candidate": {
        "package_path": row["submitted_snapshot"], "compiler_dependencies": row["compiler_dependencies"],
        "command_prefix": prefix, "bwrap_argv_length": 4,
        "answer_surfaces": [{"path": str(answer), "kind": "dir"}]}}
    experiment.analyzer = analyzer
    policy = experiment._probe_sandbox(candidate, scratch)
    assert policy["package_path"] == row["submitted_snapshot"]
    assert policy["command_prefix"] == [*prefix[:4], "--bind", str(scratch), str(scratch), *prefix[4:]]
    with pytest.raises(ValueError, match="masked answer"):
        experiment._probe_sandbox(candidate, answer)


def test_native_semantic_probe_requires_existing_isolated_policy(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Inspect graph")
    with pytest.raises(ValueError, match="production isolated compiler policy"):
        experiment.run_native_probe(candidate, tmp_path, ["/usr/bin/true"], timeout_s=1)


def _comparison_arm_fixture(tmp_path):
    from merlin.perf.analysis_worker import IsolatedAnalysisWorker
    seed = tmp_path / "comparison_seed"
    seed.mkdir()
    (seed / "source.txt").write_text("comparison compiler, not functional baseline")
    experiment, candidate, _ = setup_experiment(tmp_path,
        optimization_baseline=seed, optimization_baseline_sha256=hash_tree(seed)["sha256"])
    experiment.analyze(candidate, hypothesis="Initial optimization comparison")
    row = experiment.iterations[0]
    interface = Path(experiment.sentinel.frozen_source_path) / "source.txt"
    row["analysis"]["diagnostics"]["captured_logical_graph"]["source_sha256"] = PAS._sha256_file(interface)
    lowered, buffer = "module {}", '{"commands": [], "tensors": {}}'
    lower_sha, buffer_sha = PAS._sha256(lowered.encode()), PAS._sha256(buffer.encode())
    row["analysis"]["emission"].update(baseline_lowered_sha256=lower_sha,
                                      baseline_command_buffer_sha256=buffer_sha)
    experiment._artifacts = {"candidate_sha256": row["candidate_sha256"],
        "candidate_lowered_sha256": SHA["llvm"], "interface": interface}
    experiment._baseline_artifacts = {"identity": {"baseline_sha256": experiment.optimization_baseline_sha256,
        "capsule_sha256": experiment.sentinel.capsule_sha256,"target":experiment.target},
        "lowered_text":lowered,"lowered_sha256":lower_sha,
        "command_buffer_text":buffer,"command_buffer_sha256":buffer_sha}
    answer = tmp_path / "answers"
    answer.mkdir()
    prefix = ["bwrap","--clearenv","--tmpfs",str(answer),"bash","-c",'exec "$@"',"perf-tool"]
    policy = {"package_path":str(experiment.optimization_baseline),
        "compiler_dependencies":copy.deepcopy(experiment.optimization_baseline_binding["compiler_dependencies"]),
        "command_prefix":prefix,"bwrap_argv_length":4,
        "answer_surfaces":[{"path":str(answer),"kind":"dir"}]}
    experiment._optimization_baseline_sandbox = copy.deepcopy(policy)
    experiment._optimization_baseline_sandbox_sha256 = PAS._document_sha256(policy)
    experiment.analyzer = IsolatedAnalysisWorker(stage_path=Path(PAS.__file__), output=tmp_path/"worker",
        sandbox_factory=lambda *_:pytest.fail("must not reconstruct comparison arm grants"))
    return experiment,candidate,policy,answer


def test_initial_comparison_accessor_is_not_a_previous_iteration_or_qualification(tmp_path):
    experiment,candidate,policy,answer = _comparison_arm_fixture(tmp_path)
    artifact = experiment.optimization_baseline_artifacts(candidate)
    binding = experiment.optimization_baseline_artifact_binding(candidate)
    assert artifact["compiler_sha256"] == experiment.optimization_baseline_sha256 != experiment.baseline_sha256
    assert binding["structural_plan_status"] == "UNVERIFIED"
    assert binding["numerical_qualification"] == "UNPROVEN"
    assert binding["phase1_qualification_extended"] is False
    assert "plan_digest" not in binding
    assert len(experiment.iterations) == 1
    with pytest.raises(ValueError,match="two submitted"):
        experiment.previous_artifacts(candidate)
    scratch = tmp_path/"public_scratch"
    scratch.mkdir()
    got = experiment._probe_sandbox(candidate,scratch,optimization_baseline=True)
    prefix=policy["command_prefix"]
    assert got["command_prefix"] == [*prefix[:4],"--bind",str(scratch),str(scratch),*prefix[4:]]
    assert policy["command_prefix"] == prefix
    with pytest.raises(ValueError,match="masked answer"):
        experiment._probe_sandbox(candidate,answer,optimization_baseline=True)


@pytest.mark.parametrize("mutation",["llvm","buffer","cache_identity","analyzed_hash","source","policy","policy_missing","closure"])
def test_comparison_arm_rejects_stale_artifacts_and_recorded_policy(tmp_path,monkeypatch,mutation):
    experiment,candidate,policy,_ = _comparison_arm_fixture(tmp_path)
    if mutation == "llvm":
        experiment._baseline_artifacts["lowered_text"] += " "
    elif mutation == "buffer":
        experiment._baseline_artifacts["command_buffer_text"] += " "
    elif mutation == "cache_identity":
        experiment._baseline_artifacts["identity"]["baseline_sha256"] = experiment.baseline_sha256
    elif mutation == "analyzed_hash":
        experiment.iterations[0]["analysis"]["emission"]["baseline_lowered_sha256"] = "a"*64
    elif mutation == "source":
        outside=tmp_path/"unfrozen.mlir"
        outside.write_text("model")
        experiment._artifacts["interface"]=outside
    elif mutation == "policy":
        experiment._optimization_baseline_sandbox["command_prefix"].append("--unshare-all")
    elif mutation == "policy_missing":
        experiment._optimization_baseline_sandbox=None
    elif mutation == "closure":
        original=experiment._compiler_dependencies
        monkeypatch.setattr(experiment,"_compiler_dependencies",lambda package:
            {**original(package),"changed":True} if package==experiment.optimization_baseline else original(package))
    scratch=tmp_path/"probe"
    scratch.mkdir()
    with pytest.raises(ValueError):
        experiment._probe_sandbox(candidate,scratch,optimization_baseline=True)


def test_comparison_probe_compiles_normal_manifest_entrypoints_with_same_masks(tmp_path,monkeypatch):
    import subprocess
    from merlin.perf import analysis_worker
    from merlin.targetgen import oot_runner
    experiment,candidate,policy,_ = _comparison_arm_fixture(tmp_path)
    loaded,calls=[],[]
    monkeypatch.setattr(oot_runner,"load_package",lambda path:loaded.append(path) or object())
    def emit(package,name,source,output=None,**kwargs):
        calls.append((name,kwargs))
        if output:
            output.write_text('{"commands":[],"tensors":{}}')
        return subprocess.CompletedProcess([],0,"module {}","")
    monkeypatch.setattr(analysis_worker,"run_sandboxed_entrypoint",emit)
    source=tmp_path/"tiny.mlir"
    source.write_text("module {}")
    result=experiment.compile_optimization_baseline_probe_candidate(candidate,source,tmp_path/"probe",
        timeout_s=10,emit_command_buffer=True)
    assert loaded==[experiment.optimization_baseline]
    assert [name for name,_ in calls]==["lower_target_to_llvm","emit_command_buffer"]
    assert calls[1][1]["timeout_s"] < calls[0][1]["timeout_s"] <= 10
    assert all(item["sandbox"]["package_path"]==policy["package_path"] for _,item in calls)
    assert result["command_buffer"]=={"commands":[],"tensors":{}}
    assert len(experiment.iterations)==1


def test_comparison_compilation_detects_mutation_during_entrypoint(tmp_path,monkeypatch):
    import subprocess
    from merlin.perf import analysis_worker
    from merlin.targetgen import oot_runner
    experiment,candidate,_,_ = _comparison_arm_fixture(tmp_path)
    monkeypatch.setattr(oot_runner,"load_package",lambda path:object())
    def emit(*args,**kwargs):
        experiment._baseline_artifacts["lowered_text"] += "changed"
        return subprocess.CompletedProcess([],0,"module {}","")
    monkeypatch.setattr(analysis_worker,"run_sandboxed_entrypoint",emit)
    source=tmp_path/"tiny.mlir"
    source.write_text("module {}")
    with pytest.raises(ValueError,match="artifact bytes changed"):
        experiment.compile_optimization_baseline_probe_candidate(candidate,source,tmp_path/"probe",timeout_s=10)


def test_successful_analysis_records_baseline_policy_without_relabeling_candidate(tmp_path):
    experiment,candidate,_ = setup_experiment(tmp_path)
    delegate=experiment.analyzer
    baseline_policy={"package_path":str(experiment.optimization_baseline),
        "compiler_dependencies":experiment.optimization_baseline_binding["compiler_dependencies"],
        "command_prefix":["bwrap","--clearenv","bash"],"bwrap_argv_length":2,"answer_surfaces":[]}
    def analyze(base,current,*args,**kwargs):
        result=delegate(base,current,*args,**kwargs)
        analyze.completed_sandboxes={"baseline":baseline_policy,"candidate":{
            "package_path":str(current),"compiler_dependencies":experiment._compiler_dependencies(current)}}
        return result
    experiment.analyzer=analyze
    experiment.analyze(candidate,hypothesis="Retain exact arm policies")
    assert experiment._optimization_baseline_sandbox==baseline_policy
    assert experiment._optimization_baseline_sandbox is not baseline_policy
    assert experiment._optimization_baseline_sandbox_sha256==PAS._document_sha256(baseline_policy)
    assert experiment._compiler_sandboxes[0]["candidate"]["package_path"] != baseline_policy["package_path"]


@pytest.mark.parametrize("timeout",[0,-1,float("inf"),float("nan"),601])
def test_comparison_probe_requires_finite_inner_budget(tmp_path,timeout):
    experiment,candidate,_,_ = _comparison_arm_fixture(tmp_path)
    source=tmp_path/"tiny.mlir"
    source.write_text("module {}")
    with pytest.raises(ValueError,match="bounded timeout"):
        experiment.compile_optimization_baseline_probe_candidate(candidate,source,tmp_path/"probe",timeout_s=timeout)


def test_native_policy_accessor_reuses_exact_policy_but_returns_private_copy(tmp_path, monkeypatch):
    experiment,candidate,policy,_ = _comparison_arm_fixture(tmp_path)
    scratch=tmp_path/"native"
    scratch.mkdir()
    calls=[]
    monkeypatch.setattr(experiment,"_probe_sandbox",lambda *args: calls.append(args) or policy)
    first=experiment.native_probe_policy(candidate,scratch)
    first["command_prefix"].append("not-a-grant")
    second=experiment.native_probe_policy(candidate,scratch)
    assert len(calls)==1 and second==policy
    assert "not-a-grant" not in second["command_prefix"]


@pytest.mark.parametrize("route_status", [None, "observed_known_class_presence_mismatch"])
def test_source_contraction_prepare_and_execute_are_separate_hash_bound_actions(tmp_path,monkeypatch,route_status):
    from merlin.perf import source_contraction_preparation
    experiment,candidate,_=setup_experiment(tmp_path)
    original=experiment.analyze(candidate,hypothesis="Source pair")
    original_mapping=PAS._mapping_file
    monkeypatch.setattr(PAS,"_mapping_file",lambda *args,**kwargs:{"entry":"host_entry"})
    calls=[]
    declared_refusal={"status":"refused","problems":["source operations are not fully covered: [0, 1]"]}
    nested_evidence={"status":"prepared","emitted_route_correspondence":"UNKNOWN",
        "arms":{"before":{"source_task_cfg_proof":declared_refusal,
            "short_execution_admission":{"schema":"short_initializer_execution_admission_v1",
                "status":"source_bound_numerical_probe"}}}}
    if route_status:
        nested_evidence["task_route_feedback"] = {"status": route_status,
            "timing_calibration_admissible": False, "scope": "known static class presence only"}
    monkeypatch.setattr(source_contraction_preparation,"prepare_source_contraction",
        lambda **kwargs:calls.append(kwargs) or nested_evidence)
    prepared=experiment.prepare_source_contraction(candidate,comparison_arm="optimization_baseline",
        source_op_index=3,max_m=2,max_n=4,max_k=5,timeout_s=100)
    assert calls[0]["entry"]=="host_entry" and 0<calls[0]["timeout_s"]<=60
    assert not prepared["numerical_pass"] and not prepared["runtime_admitted"]
    assert prepared["task_route_feedback"]["status"] == (route_status or "UNKNOWN")
    assert not prepared["task_route_feedback"]["timing_calibration_admissible"]
    provider_calls=[]
    def provider(**kwargs):
        provider_calls.append(kwargs)
        return {"status":"short_pair_passed","route_relevance":"UNKNOWN",
                **({"task_route_feedback": nested_evidence["task_route_feedback"]} if route_status else {})}
    receipt=experiment.qualify_source_contraction(candidate,preparation_sha256=prepared["preparation_sha256"],
        provider=provider,timeout_s=100)
    assert len(provider_calls)==1 and provider_calls[0]["prepared"]["status"]=="prepared"
    assert provider_calls[0]["prepared"]["arms"]["before"]["source_task_cfg_proof"]==declared_refusal
    assert 0<provider_calls[0]["timeout_s"]<=60
    assert not receipt["global_speedup_proven"] and not receipt["full_model_numerics_qualified"]
    assert receipt["task_route_feedback"]["status"] == (route_status or "UNKNOWN")
    assert receipt["evidence"]["status"] == "short_pair_passed"
    assert not receipt["task_route_feedback"]["timing_calibration_admissible"]
    if route_status:
        assert receipt["task_route_feedback"] is not nested_evidence["task_route_feedback"]
    assert experiment.iterations[-1]["readiness"]==original["readiness"]
    assert len(experiment.iterations[-1]["source_pair_receipts"])==1
    monkeypatch.setattr(PAS,"_mapping_file",original_mapping)
    sealed=experiment.seal(candidate)
    assert len(G.consume_global_candidate(sealed)["source_pair_receipts"])==1
    reference=experiment.iterations[-1]["source_contraction_preparation_receipts"][0]
    stored=PAS._mapping_file(Path(reference["path"]))
    assert stored["evidence"]["arms"]["before"]["source_task_cfg_proof"]==declared_refusal
    assert stored["runtime_admitted"] is False and stored["numerical_pass"] is False
    Path(reference["path"]).chmod(0o644)
    Path(reference["path"]).write_text("changed")
    with pytest.raises(ValueError,match="source-pair receipt changed"):
        G.consume_global_candidate(sealed)
    with pytest.raises(ValueError,match="changed"):
        experiment.qualify_source_contraction(candidate,preparation_sha256=prepared["preparation_sha256"],provider=provider,timeout_s=10)
    assert len(provider_calls)==1


@pytest.mark.parametrize("case",["unique","ambiguous","stale"])
def test_source_entry_fallback_is_unique_and_frozen_not_a_named_model(tmp_path,monkeypatch,case):
    from merlin.perf import source_contraction_preparation
    experiment,candidate,_=setup_experiment(tmp_path)
    experiment.analyze(candidate,hypothesis="Source entry")
    source=tmp_path/"source.mlir"
    text="module {func.func @arbitrary_name(){func.return}}"
    if case=="ambiguous":text="module {func.func @a(){func.return} func.func @b(){func.return}}"
    source.write_text(text)
    row=experiment.iterations[-1]
    row["analysis"]["diagnostics"]["captured_logical_graph"]["source_sha256"]=PAS._sha256(text.encode()) if case!="stale" else "0"*64
    monkeypatch.setattr(PAS,"_mapping_file",lambda *args,**kwargs:{})
    monkeypatch.setattr(experiment,"current_artifacts",lambda _: {"interface":source})
    calls=[]
    monkeypatch.setattr(source_contraction_preparation,"prepare_source_contraction",
        lambda **kwargs:calls.append(kwargs) or {"status":"UNKNOWN"})
    action=lambda:experiment.prepare_source_contraction(candidate,comparison_arm="previous",source_op_index=0,
        max_m=1,max_n=1,max_k=1,timeout_s=10)
    if case=="unique":
        action()
        assert calls[0]["entry"]=="arbitrary_name"
    else:
        with pytest.raises(ValueError):action()
        assert not calls


def test_source_pair_refusal_charges_elapsed_but_does_not_call_provider(tmp_path,monkeypatch):
    experiment,candidate,_=setup_experiment(tmp_path)
    experiment.analyze(candidate,hypothesis="Source pair")
    start=experiment.iterations[-1]["elapsed_seconds"]
    with pytest.raises(ValueError,match="current-iteration"):
        experiment.qualify_source_contraction(candidate,preparation_sha256="0"*64,
            provider=lambda **_:pytest.fail("unbound preparation cannot execute"),timeout_s=10)
    assert experiment.iterations[-1]["elapsed_seconds"]>start


@pytest.mark.parametrize("preparation_status",["prepared","UNKNOWN"])
def test_source_preparation_is_feedback_not_semantic_or_runtime_pass(tmp_path,monkeypatch,preparation_status):
    from merlin.perf import source_convolution_preparation
    experiment,candidate,_=setup_experiment(tmp_path)
    row=experiment.analyze(candidate,hypothesis="Inspect generic source placement")
    before=experiment.iterations[-1]["elapsed_seconds"]
    descriptor_reads=[]
    monkeypatch.setattr(PAS,"_mapping_file",lambda path,**kwargs:descriptor_reads.append(path) or {"entry":"frozen_entry"})
    seen=[]
    def prepare(**kwargs):
        seen.append(kwargs)
        return {"status":preparation_status,"source_convolution_opportunities":[{"source_op_index":6,"source_macs":1024}]}
    monkeypatch.setattr(source_convolution_preparation,"prepare_source_convolution",prepare)
    result=experiment.prepare_source_convolution(candidate,comparison_arm="optimization_baseline",timeout_s=100)
    assert descriptor_reads==[Path(experiment.sentinel.frozen_source_path)/"capsule.yaml"]
    assert seen[0]["entry"]=="frozen_entry" and seen[0]["comparison_arm"]=="optimization_baseline"
    assert 0 < seen[0]["timeout_s"] <= 60
    assert result["status"]==("runtime_pending" if preparation_status=="prepared" else "UNKNOWN")
    assert result["numerical_pass"] is False and result["runtime_admitted"] is False
    assert result["runtime_recipe_binding"]=="NOT_PREPARED"
    assert result["allowed_edit_surfaces"]==[]
    actual=experiment.iterations[-1]
    assert len(actual["preparation_receipts"])==1
    assert not actual.get("semantic_receipts") and actual["probe_receipts"]==[]
    assert actual["readiness"]==row["readiness"]
    assert actual["elapsed_seconds"]>before


def test_preparation_exposes_only_effect_matched_frozen_ast_authority(tmp_path,monkeypatch):
    from merlin.perf import source_convolution_preparation
    experiment,candidate,_=setup_experiment(tmp_path)
    (candidate/"compiler.py").write_text("def lower():\n return 1\n")
    contract={"schema":"compiler_edit_contract_v1","existing_symbols":[{
        "surface_id":"authorized_lower","path":"compiler.py","symbol":"lower"}],"helper_extensions":[]}
    contract["sha256"]=PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate,contract)
    experiment.analyze(candidate,hypothesis="Inspect generic lowering")
    monkeypatch.setattr(PAS,"_mapping_file",lambda *args,**kwargs:{"entry":"frozen_entry"})
    monkeypatch.setattr(source_convolution_preparation,"prepare_source_convolution",lambda **kwargs:{"status":"prepared"})
    inspected=[]
    def inventory(path):
        inspected.append(path)
        return SimpleNamespace(to_dict=lambda:{"surfaces":[
            {"id":"authorized_lower","path":"compiler.py","symbol":"lower","effects":["placement","movement"]},
            {"id":"candidate_added","path":"new.py","symbol":"unapproved","effects":["placement"]}]})
    monkeypatch.setattr(PAS,"inspect_compiler_package",inventory)
    result=experiment.prepare_source_convolution(candidate,comparison_arm="previous",timeout_s=30)
    assert inspected==[experiment.edit_scope_seed]
    assert [row["surface_id"] for row in result["allowed_edit_surfaces"]]==["authorized_lower"]
    assert result["edit_authority_sha256"]==experiment.edit_scope_binding["contract_document_sha256"]


def test_preparation_failure_charges_elapsed_without_semantic_receipt(tmp_path,monkeypatch):
    from merlin.perf import source_convolution_preparation
    experiment,candidate,_=setup_experiment(tmp_path)
    experiment.analyze(candidate,hypothesis="Inspect source")
    elapsed=experiment.iterations[-1]["elapsed_seconds"]
    monkeypatch.setattr(PAS,"_mapping_file",lambda *args,**kwargs:{"entry":"frozen_entry"})
    def fail(**kwargs):
        raise ValueError("missing baseline proof")
    monkeypatch.setattr(source_convolution_preparation,"prepare_source_convolution",fail)
    with pytest.raises(ValueError,match="missing baseline proof"):
        experiment.prepare_source_convolution(candidate,comparison_arm="optimization_baseline",timeout_s=30)
    assert experiment.iterations[-1]["elapsed_seconds"]>elapsed
    assert not experiment.iterations[-1].get("semantic_receipts")


def test_preparation_rejects_implicit_or_invalid_comparison_arm(tmp_path):
    experiment,candidate,_=setup_experiment(tmp_path)
    with pytest.raises(ValueError,match="explicitly"):
        experiment.prepare_source_convolution(candidate,comparison_arm="default",timeout_s=30)


def test_preparation_oracle_helpers_are_exact_host_policy_inputs():
    policy=G.host_verification_policy_record()
    root=repo_root()/"merlin/python/merlin"
    for relative in ("perf/source_convolution_preparation.py","perf/source_convolution_witness.py",
                     "targetgen/capsule_golden.py","targetgen/conv_geometry.py","runtime/commandbuffer.py","runtime/tensor.py"):
        path=root/relative
        assert policy["sources"][str(path.resolve())]==PAS._sha256_file(path)


def test_seal_uses_exact_analyzed_snapshot_not_later_python_cache(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Inspect graph")
    cache = candidate / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"interpreter cache")
    record = json.loads(experiment.seal(candidate).read_text())
    assert not (Path(record["candidate_path"]) / "__pycache__").exists()
    assert record["global_speedup_proven"] is False


def test_probe_pair_compiles_both_artifacts_under_one_sandbox(tmp_path, monkeypatch):
    import subprocess
    from merlin.perf import analysis_worker as worker
    from merlin.targetgen import oot_runner
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Inspect graph")
    sandbox = {"package_path": str(candidate), "command_prefix": ["bwrap", "--clearenv"]}
    experiment.analyzer = worker.IsolatedAnalysisWorker(
        stage_path=Path(PAS.__file__), sandbox_factory=lambda *_: {"candidate": sandbox},
        output=tmp_path / "worker")
    monkeypatch.setattr(oot_runner, "load_package", lambda _: object())
    calls = []

    def emit(package, name, interface, output_json=None, **kwargs):
        calls.append((name, kwargs))
        if output_json:
            output_json.write_text('{"commands": [], "tensors": {}}')
        return subprocess.CompletedProcess([name], 0, "module {}", "")

    monkeypatch.setattr(worker, "run_sandboxed_entrypoint", emit)
    source = tmp_path / "tiny.mlir"
    source.write_text("module {}")
    result = experiment.compile_probe_candidate(candidate, source, tmp_path / "probe",
                                                timeout_s=10, emit_command_buffer=True)
    assert result["lowered"].returncode == result["command_buffer_emission"].returncode == 0
    assert result["command_buffer"] == {"commands": [], "tensors": {}}
    assert [name for name, _ in calls] == ["lower_target_to_llvm", "emit_command_buffer"]
    assert all(kwargs["sandbox"] == sandbox for _, kwargs in calls)
    assert calls[1][1]["timeout_s"] < calls[0][1]["timeout_s"] <= 10


def test_previous_probe_uses_only_retained_predecessor_policy(tmp_path, monkeypatch):
    import subprocess
    from merlin.perf import analysis_worker as worker
    from merlin.targetgen import oot_runner
    experiment, candidate, _ = setup_experiment(tmp_path)
    previous = experiment.analyze(candidate, hypothesis="Before")
    (candidate / "source.txt").write_text("next compiler")
    experiment.analyze(candidate, hypothesis="After")
    experiment._previous_artifacts = {"candidate_sha256": previous["candidate_sha256"],
                                      "candidate_lowered_sha256": SHA["llvm"]}
    prefix = ["bwrap", "--clearenv", "bash", "-c", 'exec "$@"', "perf-tool"]
    old = {"package_path": previous["submitted_snapshot"],
           "compiler_dependencies": previous["compiler_dependencies"], "command_prefix": prefix,
           "bwrap_argv_length": 2, "answer_surfaces": []}
    experiment._compiler_sandboxes[0] = {"candidate": old}
    experiment.analyzer = worker.IsolatedAnalysisWorker(stage_path=Path(PAS.__file__),
        output=tmp_path / "worker", sandbox_factory=lambda *_: pytest.fail("must reuse prior policy"))
    experiment.analyzer.completed_sandboxes = {"candidate": {"package_path": str(candidate)}}
    loaded, observed = [], []
    monkeypatch.setattr(oot_runner, "load_package", lambda path: loaded.append(path) or object())

    def emit(*args, **kwargs):
        observed.append(kwargs["sandbox"])
        return subprocess.CompletedProcess([], 0, "module {}", "")

    monkeypatch.setattr(worker, "run_sandboxed_entrypoint", emit)
    source = tmp_path / "tiny.mlir"
    source.write_text("module {}")
    experiment.compile_previous_probe_candidate(candidate, source, tmp_path / "probe", timeout_s=10)
    assert loaded == [Path(previous["submitted_snapshot"])]
    assert observed[0]["compiler_dependencies"] == previous["compiler_dependencies"]
    experiment._compiler_sandboxes.clear()
    with pytest.raises(ValueError, match="no retained successful"):
        experiment.compile_previous_probe_candidate(candidate, source, tmp_path / "no_policy", timeout_s=10)


def test_submitted_snapshot_survives_concurrent_agent_edits_but_cannot_seal(tmp_path):
    experiment, candidate, calls = setup_experiment(tmp_path)
    delegate = experiment.analyzer
    original = hash_tree(candidate)["sha256"]

    def analyzer(*args, **kwargs):
        (candidate / "source.txt").write_text("next revision authored during analysis")
        return delegate(*args, **kwargs)

    experiment.analyzer = analyzer
    row = experiment.analyze(candidate, hypothesis="Inspect submitted revision")
    assert row["candidate_sha256"] == original
    assert hash_tree(Path(row["submitted_snapshot"]))["sha256"] == original
    with pytest.raises(ValueError, match="candidate changed"):
        experiment.seal(candidate)


def test_frozen_baseline_artifacts_reused_but_each_edit_still_compiles(tmp_path, monkeypatch):
    experiment, candidate, calls = setup_experiment(tmp_path)
    delegate = experiment.analyzer
    baseline_compiles = []

    def analyzer(*args, artifact_sink, baseline_artifacts, **kwargs):
        if baseline_artifacts is None:
            baseline_compiles.append(True)
        artifact_sink({"baseline_artifacts": {"host_owned_fixture": True}})
        return delegate(*args, **kwargs)

    monkeypatch.setattr(PAS, "analyze_whole_model_emission", analyzer)
    experiment.analyzer = analyzer
    experiment.analyze(candidate, hypothesis="First fullmodel compilation")
    (candidate / "source.txt").write_text("new transformation")
    experiment.analyze(candidate, hypothesis="Recompile changed candidate only")
    assert len(calls) == 2 and len(baseline_compiles) == 1
    (experiment.baseline / "source.txt").write_text("changed baseline")
    with pytest.raises(ValueError, match="frozen compiler changed"):
        experiment.analyze(candidate, hypothesis="Refuse stale baseline reuse")


def test_macro_receipts_allow_completion_order_but_reject_duplicate_indices(tmp_path):
    action = PAS.BrokerAction(PAS.E2E_ANALYSIS_ACTION, (PAS._HOST_E2E_ANALYSIS_SENTINEL,), (),
                             "full-model analysis", True)
    rows = [{"index": index, "receipt_schema_version": 1, "action": action.name,
             "state": "complete", "returncode": 0,
             **{key: SHA["target"] for key in ("bindings_command_sha256", "stdout_sha256",
                                               "stderr_sha256", "argv_sha256")}}
            for index in (1, 0)]
    path = tmp_path / "receipts.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    audit = {"broker_invocations": [{"action": action.name, "bindings_sha256": SHA["target"]}] * 2}
    assert G.verify_global_broker_receipts(path, actions=[action], audit=audit)["count"] == 2
    rows[0]["index"] = 0
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="schema is invalid"):
        G.verify_global_broker_receipts(path, actions=[action], audit=audit)


def test_new_shared_helper_overlays_older_readonly_grant(tmp_path):
    import shutil
    import subprocess
    if shutil.which("bwrap") is None:
        pytest.skip("requires the production bwrap dependency")
    old, current = tmp_path / "frozen", tmp_path / "current"
    old.mkdir()
    current.mkdir()
    (old / "existing.py").write_text("old existing\n")
    (current / "new_helper.py").write_text("new helper\n")
    argv = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--tmpfs", "/scratch",
            "--ro-bind", str(old), str(current)]
    dependencies = {"shared_source_root": str(current),
                    "shared_sources": {"new_helper.py": PAS._sha256_file(current / "new_helper.py")}}
    command = G.compiler_dependency_mounts(argv, dependencies, tmp_path / "overlay")
    observed = subprocess.run([*command, "/usr/bin/cat", str(current / "existing.py"),
                               str(current / "new_helper.py")], capture_output=True, text=True)
    assert observed.returncode == 0, observed.stderr
    assert observed.stdout == "old existing\nnew helper\n"


def test_changed_candidate_requires_new_graph_plan_analysis(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Delete representation round trip")
    (candidate / "source.txt").write_text("another compiler")
    with pytest.raises(ValueError, match="recompile its full graph"):
        experiment.seal(candidate)
    experiment.analyze(candidate, hypothesis="Re-evaluate next compiler")
    assert experiment.seal(candidate).is_file()


@pytest.mark.parametrize("field", ["candidate_sha256", "logical_dispatch_digest",
                                   "candidate_lowered_sha256", "candidate_command_buffer_sha256"])
def test_plan_must_bind_actual_candidate_graph_and_artifacts(tmp_path, field):
    experiment, candidate, _ = setup_experiment(tmp_path)
    row = experiment.analyze(candidate, hypothesis="Test artifact identity binding")
    analysis = copy.deepcopy(row["analysis"])
    analysis["diagnostics"]["verified_global_plan_emission"][field] = SHA["target"]
    assert "candidate_global_plan_binding_mismatch" in PAS.global_iteration_readiness(analysis)["blockers"]


def test_frozen_model_and_baseline_are_rechecked(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Optimize complete graph")
    (Path(experiment.sentinel.frozen_source_path) / "source.txt").write_text("different model")
    with pytest.raises(ValueError, match="complete-model objective changed"):
        experiment.analyze(candidate, hypothesis="Cannot substitute an easier graph")


def test_macro_broker_refuses_unrelated_micro_sweep_without_running_evaluator(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    action = PAS.BrokerAction(PAS.DEVELOPMENT_FEEDBACK_ACTION,
                             (PAS._HOST_FEEDBACK_SENTINEL,), (), "legacy micro feedback", False)

    class Evaluator:
        def evaluate(self, *_args, **_kwargs):
            pytest.fail("macro mode must not execute a corpus microbenchmark sweep")

    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(("bwrap",), (), "available_not_an_isolation_claim", True, True, True),
        None, candidate, (action,), tmp_path / "broker" / "receipts.jsonl",
        deadline=PAS.time.monotonic() + 60, max_calls=1, max_tool_seconds=30,
        feedback_evaluator=Evaluator(), feedback_round=0, global_experiment=experiment)
    result = broker.execute({"action": PAS.DEVELOPMENT_FEEDBACK_ACTION, "bindings": {}})
    assert result["returncode"] == 125
    assert broker.stop_verdict is None


def test_macro_broker_reserves_deadline_for_required_whole_model_analysis(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    inspect = PAS.BrokerAction("inspect", ("unused",), (), "inspect", False)
    analysis = PAS.BrokerAction(
        PAS.E2E_ANALYSIS_ACTION, (PAS._HOST_E2E_ANALYSIS_SENTINEL,), (), "analysis", True)
    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(("bwrap",), (), "available_not_an_isolation_claim", True, True, True),
        SimpleNamespace(), candidate, (inspect, analysis), tmp_path / "broker" / "receipts.jsonl",
        deadline=PAS.time.monotonic() + 30, max_calls=2, max_tool_seconds=30,
        mandatory_analysis_reserve_seconds=30)
    with pytest.raises(PAS.StageGateError, match="reserved for mandatory whole-model analysis"):
        broker.execute({"action": "inspect", "bindings": {}})
    # The required action still reaches its host handler during the same reserved
    # interval.  This fixture has no global objective, so the handler refuses with
    # returncode 125 after admission rather than executing a compiler.
    assert broker.execute({"action": PAS.E2E_ANALYSIS_ACTION, "bindings": {}})["returncode"] == 125
    rows = [json.loads(line) for line in (tmp_path / "broker" / "receipts.jsonl").read_text().splitlines()]
    assert [row["state"] for row in rows] == ["rejected", "complete"]


def test_macro_handoff_detects_post_seal_candidate_edit(tmp_path):
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Global plan revision")
    record = experiment.seal(candidate)
    # The editable proposal can continue evolving without changing the frozen handoff.
    (candidate / "source.txt").write_text("edited after sealing")
    document = G.consume_global_candidate(record)
    frozen_file = Path(document["candidate_path"]) / "source.txt"
    frozen_file.chmod(0o644)
    frozen_file.write_text("tampered frozen bytes")
    with pytest.raises(ValueError, match="candidate bytes changed"):
        G.consume_global_candidate(record)


@pytest.mark.parametrize("outcome", ["passed", "runner_failed", "wrong_observation", "over_budget"])
def test_optional_probe_is_bound_to_current_graph_and_remains_probe_only(tmp_path, monkeypatch, outcome):
    from merlin.perf.activity_schedule import ActivityEvent
    from merlin.perf.execution_policy import SimulationBudget, WarmComputeReceipt, WarmProfileContract
    from merlin.perf.mechanism_probe import (
        MechanismEvidence, ProbeBinding, ProbeObservation, derive_mechanism_signature)
    from merlin.xdsl_dialects.lowering.global_plan import ValueRepresentation

    experiment, candidate, _ = setup_experiment(tmp_path)
    row = experiment.analyze(candidate, hypothesis="Overlap model loads with contraction work")
    binding = ProbeBinding(SHA["graph"], SHA["plan"],
                           row["compiler_dependencies"]["compiler_implementation_sha256"], SHA["target"])
    signature = derive_mechanism_signature(
        representations=[ValueRepresentation("local", "row-major", "i8", "blocked")],
        events=[ActivityEvent("load", "dma", "movement", 0, movement_bytes=128),
                ActivityEvent("compute", "mesh", "compute", 0, depends_on=("load",))],
        capacity_regime={"scratch": "fits_double"}, tile_shape=(8, 8, 8),
        edge_cases=("aligned",), repetition_semantics="independent tiles",
        instruction_semantics=[{"operation": "load"}, {"operation": "matmul"}])
    model = MechanismEvidence(binding, signature, SHA["llvm"], 1024, "host extracted model")
    probe = MechanismEvidence(binding, signature, PAS._sha256(b"probe"), 4, "host extracted probe")
    inputs = dict(model=model, probe=probe, descriptor={"kind": "kernel"},
                  budget=SimulationBudget(30, 30), estimated_cycles=100,
                  measured_cycles_per_second=100)
    ran = []
    clock = [100.0]
    monkeypatch.setattr(G.time, "monotonic", lambda: clock[0])
    initial_elapsed = row["elapsed_seconds"]
    execution_seconds = 31.0 if outcome == "over_budget" else 7.0
    if outcome == "runner_failed":
        experiment.timeout_s = 8.0

    def execute(*, timeout_s):
        ran.append(timeout_s)
        clock[0] += execution_seconds
        if outcome == "runner_failed":
            raise TimeoutError("bounded runner stopped")
        if outcome == "wrong_observation":
            return None
        return ProbeObservation(
            probe, WarmComputeReceipt("short-independent-probe", 100,
                                      WarmProfileContract(), "warm counter receipt"),
            0.1, 1, probe.artifact_digest)

    if outcome != "passed":
        error, message = ((TimeoutError, "bounded runner stopped") if outcome == "runner_failed"
                          else (ValueError, "does not match" if outcome == "wrong_observation"
                                else "exceeded its bounded"))
        with pytest.raises(error, match=message):
            experiment.measure_probe(candidate, admission_inputs=inputs, execute=execute)
        assert experiment.iterations[-1]["elapsed_seconds"] == initial_elapsed + execution_seconds
        assert not experiment.iterations[-1]["probe_receipts"]
        if outcome == "runner_failed":
            with pytest.raises(ValueError, match="exceeds iteration wall budget"):
                experiment.measure_probe(candidate, admission_inputs=inputs,
                    execute=lambda **_: pytest.fail("retry exceeded the remaining iteration budget"))
        return
    receipt = experiment.measure_probe(candidate, admission_inputs=inputs, execute=execute)
    assert experiment.iterations[-1]["elapsed_seconds"] == initial_elapsed + execution_seconds
    assert ran == [30]
    assert receipt["full_model_cycles"] is None
    assert receipt["target_timing_authority"] is None
    assert receipt["observed_timing_identity"] is None
    assert "separately validated calibration" in receipt["timing_scope"]
    assert receipt["warmup_runs"] == receipt["measured_runs"] == 1
    assert receipt["binding"]["compiler_digest"] == row["compiler_dependencies"]["compiler_implementation_sha256"]
    sealed = experiment.seal(candidate)
    assert len(G.consume_global_candidate(sealed)["probe_receipts"]) == 1


def test_agent_cannot_replace_host_probe_bindings(tmp_path):
    # Scope guard is reached only after current full graph + plan have been validated.
    experiment, candidate, _ = setup_experiment(tmp_path)
    experiment.analyze(candidate, hypothesis="Try an uncertain mechanism")
    with pytest.raises(ValueError, match="current probe bindings"):
        experiment.measure_probe(candidate, admission_inputs={"current_binding": {}},
                                 execute=lambda **_: pytest.fail("an unadmitted probe ran"))


def test_shared_compiler_helper_edit_invalidates_the_iteration(tmp_path, monkeypatch):
    shared = tmp_path / "shared" / "python" / "merlin"
    shared.mkdir(parents=True)
    (shared / "__init__.py").write_text("")
    (shared / "helper.py").write_text("def transform(): return 1\n")
    monkeypatch.setattr(PAS, "merlin_dir", lambda: tmp_path / "shared")
    experiment, candidate, _ = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("from merlin.helper import transform\n")
    row = experiment.analyze(candidate, hypothesis="Use a shared generalized transform")
    assert list(row["compiler_dependencies"]["shared_sources"]) == ["__init__.py", "helper.py"]
    old_tree = hash_tree(candidate)["sha256"]
    (shared / "helper.py").write_text("def transform(): return 2\n")
    assert hash_tree(candidate)["sha256"] == old_tree
    with pytest.raises(ValueError, match="shared compiler dependencies changed"):
        experiment.seal(candidate)


def test_real_macro_round_transport_compiles_each_revision_without_micro_feedback(tmp_path, monkeypatch):
    experiment, candidate, calls = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def lower():\n return 1\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "lowering", "path": "compiler.py", "symbol": "lower"}], "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    mechanism_catalog = {"schema": "compiler_mechanism_catalog_v1",
        "contract_sha256": contract["sha256"], "mechanisms": [{
            "id": "lowering", "selectors": [
                {"kind": "function", "path": "compiler.py", "symbol": "lower"}]}]}
    mechanism_catalog["sha256"] = PAS._document_sha256(mechanism_catalog)
    mechanism_path = tmp_path / "active_mechanism_catalog.json"
    mechanism_path.write_bytes(PAS._canonical_json(mechanism_catalog))
    mechanism_path.chmod(0o444)
    experiment.freeze_mechanism_catalog(mechanism_path, PAS._sha256_file(mechanism_path))
    _freeze_test_mechanism_work_order(
        experiment, candidate, tmp_path, mechanism_id="lowering")
    # Qualification IO is tested separately; this fixture exercises the authoring transport.
    experiment.phase1_binding = {"test_fixture": "verified existing qualification"}
    action = PAS.BrokerAction(PAS.E2E_ANALYSIS_ACTION, (PAS._HOST_E2E_ANALYSIS_SENTINEL,), (),
                             "full-model analysis", True)
    original_broker = PAS._Broker
    brokers = []

    def broker(*args, **kwargs):
        result = original_broker(*args, **kwargs)
        brokers.append(result)
        return result

    def codex_round(*args, **kwargs):
        (candidate / "compiler.py").write_text("def lower():\n return 2\n")
        result = brokers[-1].execute({"action": PAS.E2E_ANALYSIS_ACTION, "bindings": {}})
        assert result["returncode"] == 0
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("{}\n")
        return 0, transcript, None

    monkeypatch.setattr(PAS, "_Broker", broker)
    monkeypatch.setattr(PAS, "_codex_round", codex_round)
    monkeypatch.setattr(PAS, "build_action_registry", lambda *_args, **kwargs: (action,))
    monkeypatch.setattr(PAS, "verify_answer_free_agent_inputs", lambda *_: None)
    monkeypatch.setattr(PAS, "inner_execution_policy", lambda *_: PAS.AgentSandboxPolicy(
        ("bwrap",), (), "available_not_an_isolation_claim", True, True, True))
    monkeypatch.setattr(PAS, "run_required_tool_probes", lambda *_: [])
    monkeypatch.setattr(PAS, "inspect_compiler_package", lambda *_: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(PAS, "_round_telemetry", lambda *_args, **_kwargs: {"complete": True})
    monkeypatch.setattr(PAS, "audit_codex_transcript", lambda *_: {
        "clean": True, "broker_invocations": [{"action": PAS.E2E_ANALYSIS_ACTION,
            "bindings_sha256": PAS._sha256(PAS._canonical_json([]))}]})
    result = G.run_global_agent_round(
        experiment, candidate, target_experiment=SimpleNamespace(), workspace=tmp_path,
        stage_root=tmp_path / "stage", agent_inputs=SimpleNamespace(),
        frozen_functional=SimpleNamespace(), frozen_corpus_manifest=tmp_path / "manifest.json",
        model="test", resolved_model="test", effort="high", codex_binary=Path("codex"),
        round_index=0, round_timeout_s=30, max_tool_calls=3)
    assert result["status"] == "authored"
    assert result["mechanism_attribution"]["status"] == "allowed"
    assert result["mechanism_attribution"]["selected_mechanism_id"] == "lowering"
    assert PAS._mapping_file(Path(result["mechanism_attribution"]["receipt"]["path"]))[
        "candidate_sha256"] == result["candidate_sha256"]
    assert result["global_speedup_proven"] is False
    assert len(calls) == 2 and calls[0] != calls[1]
    assert result["broker_evidence"]["successful_actions"] == [PAS.E2E_ANALYSIS_ACTION]
    assert brokers[0].stop_verdict is None
    task = (tmp_path / "TASK.md").read_text()
    context = json.loads((tmp_path / "STAGE_CONTEXT.json").read_text())
    assert "Do not run Python directly against any path in the candidate workspace" in task
    assert "Do not place shell or Python commands before or after a broker call" in task
    assert "exactly one coherent optimization mechanism per round" in task
    assert "must not include opportunistic unrelated edits" in task
    assert "host_frozen_mechanism_work_order" in task
    assert "Do not infer, substitute or add sites" in task
    assert "delete whole-program work and boundaries" in task
    assert "only then operator, tile, or local scalar cleanup" in task
    assert context["portfolio_action_digest"]["members"][0]["identity"]["capsule"] == "real-model"
    assert "mandatory_analysis_reserve" in context
    assert context["mandatory_analysis_reserve"]["seconds"] == 0
    assert context["host_post_authoring_validation"]["maximum_seconds"] == experiment.timeout_s
    assert context["host_post_authoring_validation"]["full_model_simulation_allowed"] is False
    assert context["host_frozen_mechanism_work_order"]["work_order"][
        "mechanism_id"] == "lowering"
    assert context["mechanism_work_order_analysis"]["candidate_sha256"] != result[
        "candidate_sha256"]


def test_macro_round_validates_final_bytes_after_authoring_with_independent_host_budget(
        tmp_path, monkeypatch):
    experiment, candidate, calls = setup_experiment(tmp_path, timeout_s=1200)
    (candidate / "compiler.py").write_text("def lower():\n return 1\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "lowering", "path": "compiler.py", "symbol": "lower"}],
        "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    experiment.phase1_binding = {"test_fixture": "verified existing qualification"}
    action = PAS.BrokerAction(
        PAS.E2E_ANALYSIS_ACTION, (PAS._HOST_E2E_ANALYSIS_SENTINEL,), (),
        "optional in-round full-model analysis", False)
    original_broker = PAS._Broker
    brokers = []
    analysis_timeouts = []
    original_analyze = experiment.analyze

    def analyze(*args, **kwargs):
        analysis_timeouts.append(kwargs.get("timeout_s"))
        return original_analyze(*args, **kwargs)

    def broker(*args, **kwargs):
        result = original_broker(*args, **kwargs)
        brokers.append(result)
        return result

    def codex_round(*args, **kwargs):
        (candidate / "compiler.py").write_text("def lower():\n return 2\n")
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("{}\n")
        return 0, transcript, None

    experiment.analyze = analyze
    monkeypatch.setattr(PAS, "_Broker", broker)
    monkeypatch.setattr(PAS, "_codex_round", codex_round)
    monkeypatch.setattr(PAS, "build_action_registry", lambda *_args, **kwargs: (action,))
    monkeypatch.setattr(PAS, "verify_answer_free_agent_inputs", lambda *_: None)
    monkeypatch.setattr(PAS, "inner_execution_policy", lambda *_: PAS.AgentSandboxPolicy(
        ("bwrap",), (), "available_not_an_isolation_claim", True, True, True))
    monkeypatch.setattr(PAS, "run_required_tool_probes", lambda *_: [])
    monkeypatch.setattr(PAS, "inspect_compiler_package", lambda *_: SimpleNamespace(to_dict=lambda: {}))
    monkeypatch.setattr(PAS, "_round_telemetry", lambda *_args, **_kwargs: {"complete": True})
    monkeypatch.setattr(PAS, "audit_codex_transcript", lambda *_: {
        "clean": True, "broker_invocations": []})

    result = G.run_global_agent_round(
        experiment, candidate, target_experiment=SimpleNamespace(), workspace=tmp_path,
        stage_root=tmp_path / "stage", agent_inputs=SimpleNamespace(),
        frozen_functional=SimpleNamespace(), frozen_corpus_manifest=tmp_path / "manifest.json",
        model="test", resolved_model="test", effort="high", codex_binary=Path("codex"),
        round_index=0, round_timeout_s=600, max_tool_calls=3)

    assert result["status"] == "authored"
    assert analysis_timeouts == [None, 1200]
    assert len(calls) == 2 and calls[0] != calls[1]
    assert brokers[0].deadline - PAS.time.monotonic() <= 420
    assert brokers[0].mandatory_analysis_reserve_seconds == 0
    assert result["host_post_authoring_validation"]["candidate_sha256"] == hash_tree(candidate)["sha256"]
    assert result["host_post_authoring_validation"]["maximum_seconds"] == 1200
    assert result["broker_evidence"]["required_actions"] == []


def test_cached_policy_rebind_preserves_masks_and_rejects_identity_drift(tmp_path):
    old, new, scratch, next_scratch, answers, overlay = (tmp_path / name for name in
        ("old", "new", "scratch", "next_scratch", "answers", "overlay"))
    for path in (old, new, scratch, next_scratch, answers, overlay):
        path.mkdir()
    (old / "compiler.py").write_text("def lower():\n return 1\n")
    (new / "compiler.py").write_text("def lower():\n return 2\n")
    (overlay / "helper.py").write_text("VALUE=1\n")
    dependencies = G.compiler_dependency_record(new)
    prefix = ["bwrap", "--clearenv", "--ro-bind", str(old), str(old),
              "--bind", str(scratch), str(scratch), "--tmpfs", str(answers),
              "--chdir", str(old), "bash", "-c", 'exec "$@"', "perf-tool"]
    cached = {"package_path": str(old), "scratch_path": str(scratch),
        "compiler_dependencies": G.compiler_dependency_record(old),
        "command_prefix": prefix, "bwrap_argv_length": 10,
        "overlay_trees": {str(overlay): PAS._exact_tree_record(overlay)["sha256"]},
        "answer_surfaces": [{"path": str(answers), "kind": "dir"}]}
    rebound = G.rebind_compiler_sandbox(cached, package=new, scratch=next_scratch, dependencies=dependencies)
    assert cached["command_prefix"] == prefix
    result = rebound["command_prefix"]
    assert result[2:5] == ["--ro-bind", str(new), str(old)]
    assert result[5:8] == ["--bind", str(next_scratch), str(scratch)]
    assert result[14:16] == ["--tmpfs", str(answers)]
    assert rebound["policy_reuse"]["answer_masks_changed"] is False
    with pytest.raises(ValueError, match="masked answer"):
        G.rebind_compiler_sandbox(cached, package=answers, scratch=next_scratch, dependencies=dependencies)
    (new / "compiler.py").write_text("def lower():\n return 3\n")
    with pytest.raises(ValueError, match="source identity"):
        G.rebind_compiler_sandbox(cached, package=new, scratch=next_scratch, dependencies=dependencies)
    (new / "compiler.py").write_text("def lower():\n return 2\n")
    (overlay / "helper.py").write_text("VALUE=2\n")
    with pytest.raises(ValueError, match="overlay changed"):
        G.rebind_compiler_sandbox(cached, package=new, scratch=next_scratch, dependencies=dependencies)
    forged = copy.deepcopy(dependencies)
    forged["shared_sources"]["extra.py"] = SHA["graph"]
    with pytest.raises(ValueError, match="closure changed"):
        G.rebind_compiler_sandbox(cached, package=new, scratch=next_scratch, dependencies=forged)


@pytest.mark.parametrize("separate_comparison", [False, True])
def test_factory_reuses_exact_policy_without_rebuilding_masks_for_each_revision(tmp_path, monkeypatch,
                                                                               separate_comparison):
    experiment, candidate, _ = setup_experiment(tmp_path)
    config, answers = tmp_path / "config.json", tmp_path / "answers"
    config.write_text("{}")
    answers.mkdir()
    frozen_contract = tmp_path / "frozen_phase1_contract"
    frozen_contract.mkdir()
    historical_schema = frozen_contract / "schema.json"
    historical_schema.write_text('{"historical_phase1_contract": true}')
    historical_sha = PAS._sha256_file(historical_schema)
    calls = []
    def inner(*args):
        calls.append(str(args[1]))
        assert args[4] == experiment.baseline  # /perf-functional-base keeps qualification identity.
        return PAS.AgentSandboxPolicy(("bwrap", "--clearenv", "--ro-bind", str(frozen_contract),
                                       "/historical-contract"), (), "test-only", True, True, True)
    monkeypatch.setattr(PAS, "verify_answer_free_agent_inputs", lambda *_: None)
    monkeypatch.setattr(PAS, "answer_surfaces", lambda *_: [SimpleNamespace(path=answers, kind="dir")])
    monkeypatch.setattr(PAS, "inner_execution_policy", inner)
    monkeypatch.setattr(PAS, "inner_command", lambda policy, *_: [*policy.argv, "bash", "-c", 'exec "$@"', "perf-tool", "PAYLOAD_MARKER"])
    monkeypatch.setattr(PAS.BW, "apply_answer_masks", lambda argv, _: [*argv, "--tmpfs", str(answers)])
    monkeypatch.setattr(PAS.BW, "coverage_gap", lambda *_: [])
    monkeypatch.setattr(G, "compiler_dependency_mounts", lambda argv, *_: argv)
    factory = G.global_compiler_sandbox_factory(target_experiment=SimpleNamespace(path=config),
        agent_inputs=SimpleNamespace(manifest_path=config), frozen_functional=SimpleNamespace(marker=config),
        frozen_corpus_manifest=config, qualification_baseline=experiment.baseline if separate_comparison else None)
    comparison = experiment.baseline
    if separate_comparison:
        comparison = tmp_path / "optimization-comparison"
        comparison.mkdir()
        (comparison / "compiler.py").write_text("VALUE=1\n")
    first_scratch, second_scratch = tmp_path / "scratch1", tmp_path / "scratch2"
    first_scratch.mkdir()
    second_scratch.mkdir()
    first = factory(comparison, candidate, first_scratch)
    schema = PAS.whole_program_schema_record()
    prefix = first["candidate"]["command_prefix"]
    schema_index = prefix.index(schema["path"])
    assert prefix[schema_index-1:schema_index+2] == ["--ro-bind", schema["path"], "/compiler-api/command_buffer.schema.json"]
    assert prefix[prefix.index("MERLIN_COMMAND_BUFFER_SCHEMA")+1] == "/compiler-api/command_buffer.schema.json"
    assert G.host_verification_policy_record()["sources"][schema["path"]] == schema["sha256"]
    assert prefix[prefix.index(str(frozen_contract))+1] == "/historical-contract"
    assert PAS._sha256_file(historical_schema) == historical_sha
    (candidate / "source.txt").write_text("new compiler implementation")
    second = factory(comparison, candidate, second_scratch)
    assert first["baseline"]["package_path"] == str(comparison)
    assert len(calls) == 2
    assert all(second[arm]["policy_reuse"]["answer_masks_changed"] is False for arm in second)
    assert first["candidate"]["compiler_dependencies"]["candidate_sha256"] != second["candidate"]["compiler_dependencies"]["candidate_sha256"]
    assert second["candidate"]["scratch_path"] == str(second_scratch)


@pytest.mark.parametrize("failure", ["scope", "baseline_integrity"])
def test_pre_action_scope_failure_has_terminal_receipt_and_http_refusal(tmp_path, monkeypatch, failure):
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    experiment, candidate, _ = setup_experiment(tmp_path)
    (candidate / "compiler.py").write_text("def allowed():\n return 1\n")
    contract = {"schema": "compiler_edit_contract_v1", "existing_symbols": [
        {"surface_id": "lowering", "path": "compiler.py", "symbol": "allowed"}], "helper_extensions": []}
    contract["sha256"] = PAS._document_sha256(contract)
    experiment.freeze_edit_scope(candidate, contract)
    if failure == "scope":
        (candidate / "protected.py").write_text("VALUE=1\n")
    else:
        (experiment.baseline / "source.txt").write_text("changed immutable baseline")
    action = PAS.BrokerAction("candidate-test", ("must-not-execute",), (), "test", False)
    receipts = tmp_path / "broker" / "receipts.jsonl"
    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(("bwrap",), (), "test-only", True, True, True),
        SimpleNamespace(), candidate, (action,), receipts, deadline=PAS.time.monotonic()+30,
        max_calls=3, max_tool_seconds=5, global_experiment=experiment)
    monkeypatch.setattr(PAS, "inner_command", lambda *_: pytest.fail("candidate execution reached"))
    with broker.serving() as (host, port):
        for index in range(2):
            request = Request(f"http://{host}:{port}/execute", method="POST",
                data=json.dumps({"action": action.name, "bindings": {}}).encode(),
                headers={"Content-Type": "application/json", "X-Perf-Token": broker.token})
            with pytest.raises(HTTPError) as caught:
                urlopen(request, timeout=5)
            assert caught.value.code == 400
            response = json.loads(caught.value.read())
            rows = [json.loads(line) for line in receipts.read_text().splitlines()]
            assert len(rows) == index+1
            assert rows[-1]["index"] == index
            assert rows[-1]["state"] == "rejected" and rows[-1]["returncode"] == 126
            assert broker.calls[-1]["state"] == "rejected"
            assert response["error"].startswith("StageGateError:")
            assert "compiler edit authority" in response["error"]
            private = experiment.output / "host_refusals" / f"round_None_call_{index}.txt"
            assert private.is_file() and "ValueError:" in private.read_text()
            assert ("host-frozen authority" if failure == "scope" else "frozen compiler changed") in private.read_text()


def test_unexpected_scope_checker_fault_is_not_normalized_as_a_policy_refusal(tmp_path, monkeypatch):
    experiment, candidate, _ = setup_experiment(tmp_path)
    def broken_checker(*_):
        raise RuntimeError("unexpected internal checker fault")
    monkeypatch.setattr(experiment, "validate_candidate_scope", broken_checker)
    action = PAS.BrokerAction("candidate-test", ("must-not-execute",), (), "test", False)
    broker = PAS._Broker(
        PAS.AgentSandboxPolicy(("bwrap",), (), "test-only", True, True, True),
        SimpleNamespace(), candidate, (action,), tmp_path / "broker" / "receipts.jsonl",
        deadline=PAS.time.monotonic()+30, max_calls=1, max_tool_seconds=5, global_experiment=experiment)
    monkeypatch.setattr(PAS, "inner_command", lambda *_: pytest.fail("candidate execution reached"))
    with pytest.raises(RuntimeError, match="unexpected internal checker fault"):
        broker.execute({"action": action.name, "bindings": {}})


@pytest.mark.parametrize("relative", ["perf/storage_encoding.py", "perf/structural_transitions.py", "perf/physical_transition_evidence.py", "runtime/storage_binding.py",
    "runtime/prepack_authority.py", "runtime/captured_constants.py", "frontends/argument_identity.py",
    "perf/host_physical_transition_qualifier.py", "perf/model_placement.py", "perf/model_macs.py"])
def test_storage_interpreter_content_change_invalidates_frozen_segment(tmp_path, monkeypatch, relative):
    original = PAS.repo_root() / "merlin/python/merlin" / relative
    copied = tmp_path / "host_interpreter.py"
    copied.write_bytes(original.read_bytes())
    sha_file = PAS._sha256_file
    # Route just this source read to test-owned bytes; production sources remain untouched.
    monkeypatch.setattr(PAS, "_sha256_file", lambda path:
        sha_file(copied if Path(path).resolve() == original.resolve() else path))
    experiment, candidate, calls = setup_experiment(tmp_path)
    assert str(original.resolve()) in experiment.host_policy["sources"]
    experiment.analyze(candidate, hypothesis="Bind original storage interpretation")
    copied.write_bytes(copied.read_bytes() + b"\n# changed storage interpreter\n")
    with pytest.raises(ValueError, match="host verification policy changed"):
        experiment.analyze(candidate, hypothesis="Must not reuse stale interpretation")
    assert len(calls) == 1


@pytest.mark.parametrize("bound", [True, False])
def test_agent_storage_view_compacts_maps_and_keeps_bound_unresolved_obligations(tmp_path, bound):
    experiment, candidate, _ = setup_experiment(tmp_path)
    record = experiment.analyze(candidate, hypothesis="Typed storage plan")
    plan = record["analysis"]["diagnostics"]["verified_global_plan_emission"]
    contracts = {f"tensor_{index}": {"contract": {"logical_shape": [2, 3], "physical_shape": [3, 2]},
        "caller_materialization": "requires artifact-bound pack/view evidence",
        "emitted_consumer_addressing": "requires artifact-bound address evidence"} for index in range(393)}
    plan["storage_encodings"] = contracts
    plan["physical_transition_evidence"] = {"status": "verified", "rows": [{"copy_bytes": 24}]}
    if not bound:
        plan["candidate_lowered_sha256"] = SHA["target"]
    original = copy.deepcopy(record)
    view = G.agent_analysis_view(record, complete_evidence="/perf-control/full_model_analysis_0000.json")
    assert record == original
    compact_plan = view["analysis"]["diagnostics"]["verified_global_plan_emission"]
    assert "storage_encodings" not in compact_plan
    summary = compact_plan["storage_encoding_summary"]
    assert summary["reported_contract_count"] == 393
    assert summary["source_plan_binding_verified"] is bound
    assert summary["details"]["path"] == "/perf-control/full_model_analysis_0000.json"
    assert summary["details"]["canonical_record_sha256"] == PAS._document_sha256(record)
    assert summary["details"]["encoding_map_sha256"] == PAS._document_sha256(contracts)
    assert summary["details"]["json_pointer"] == "/analysis/diagnostics/verified_global_plan_emission/storage_encodings"
    transition_summary = summary["physical_transition_evidence"]
    assert transition_summary["status"] == ("verified" if bound else "UNKNOWN")
    assert transition_summary["details"]["evidence_sha256"] == PAS._document_sha256(plan["physical_transition_evidence"])
    assert "rows" not in transition_summary
    for field in ("caller_materialization", "emitted_consumer_addressing"):
        assert summary[field]["reported_pending_count"] == 393
        assert summary[field]["status"] == ("UNRESOLVED" if bound else "UNKNOWN")
        assert summary[field]["verified"] is False
    assert len(summary["obligation_examples"]) == 3
    assert "tensor_392" not in json.dumps(view)
    assert len(json.dumps(summary)) < 4000
    assert view["analysis"]["optimization_brief"]["storage_encoding_obligations"] == summary


@pytest.mark.parametrize("bound", [True, False])
def test_baseline_storage_view_keeps_separate_binding_and_details(tmp_path, bound):
    experiment, candidate, _ = setup_experiment(tmp_path)
    record = experiment.analyze(candidate, hypothesis="Separate baseline storage evidence")
    analysis = record["analysis"]
    analysis["emission"]["baseline_lowered_sha256"] = PAS._sha256(b"baseline LLVM")
    analysis["emission"]["baseline_command_buffer_sha256"] = PAS._sha256(b"baseline CB")
    baseline = copy.deepcopy(analysis["diagnostics"]["verified_global_plan_emission"])
    baseline["candidate_sha256"] = experiment.optimization_baseline_sha256
    baseline["candidate_lowered_sha256"] = analysis["emission"]["baseline_lowered_sha256"]
    baseline["candidate_command_buffer_sha256"] = analysis["emission"]["baseline_command_buffer_sha256"]
    baseline["storage_encodings"] = {f"baseline_tensor_{i}": {
        "caller_materialization": "requires baseline evidence"} for i in range(393)}
    if not bound:
        baseline["candidate_lowered_sha256"] = "0" * 64
    analysis["diagnostics"]["verified_baseline_global_plan_emission"] = baseline
    original = copy.deepcopy(record)
    view = G.agent_analysis_view(record, complete_evidence="complete.json")
    compact = view["analysis"]["diagnostics"]["verified_baseline_global_plan_emission"]
    assert "storage_encodings" not in compact
    summary = compact["storage_encoding_summary"]
    assert summary["arm"] == "baseline"
    assert summary["reported_contract_count"] == 393
    assert summary["source_plan_binding_verified"] is bound
    assert summary["details"]["json_pointer"] == "/analysis/diagnostics/verified_baseline_global_plan_emission/storage_encodings"
    assert summary["details"]["canonical_record_sha256"] == PAS._document_sha256(record)
    assert "baseline_tensor_392" not in json.dumps(view)
    assert record == original
    assert len(json.dumps(summary)) < 4000
