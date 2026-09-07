"""Paired preparation consumes cached proofs and compiles only the selected short source."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from merlin.perf import source_contraction_preparation as P
from merlin.perf.source_program_pair import document_digest, text_digest


SOURCE = '''module {func.func @work(%a:tensor<3x9xi8>,%b:tensor<9x4xi8>,%c:tensor<3x4xi8>)->tensor<3x4xi8>{
%r=linalg.matmul ins(%a,%b:tensor<3x9xi8>,tensor<9x4xi8>) outs(%c:tensor<3x4xi8>)->tensor<3x4xi8>
func.return %r:tensor<3x4xi8>}}'''


def fixture(tmp_path, monkeypatch):
    source = tmp_path/"full-source.mlir"
    source.write_text(SOURCE)
    graph, policy = "g"*64, "p"*64
    def artifact(arm):
        cb = {"params": {"global_program_plan": {"tasks": [
            {"task_index": 0, "kind": "contraction", "source_op_indices": [0]}]}}}
        raw = json.dumps(cb, indent=2)+"\n"
        llvm = arm+" full emitted source"
        return {"interface": source, "lowered_text": llvm, "command_buffer": cb,
                "command_buffer_text": raw, "candidate_sha256": arm*64,
                "candidate_lowered_sha256": text_digest(llvm), "candidate_command_buffer_sha256": text_digest(raw)}
    before, after = artifact("a"), artifact("b")
    def proof(artifact):
        return {"status": "verified", "source_sha256": text_digest(SOURCE),
            "candidate_sha256": artifact["candidate_sha256"],
            "candidate_lowered_sha256": artifact["candidate_lowered_sha256"],
            "candidate_command_buffer_sha256": artifact["candidate_command_buffer_sha256"],
            "logical_dispatch_digest": graph, "plan_digest": document_digest(P.program_plan(artifact["command_buffer"]))}
    prior_proof, current_proof = proof(before), proof(after)
    baseline_binding = {"schema": "baseline_global_plan_evidence_binding_v1", "source_sha256": text_digest(SOURCE),
        "compiler_sha256": before["candidate_sha256"], "lowered_sha256": before["candidate_lowered_sha256"],
        "command_buffer_sha256": before["candidate_command_buffer_sha256"],
        "host_verifier_policy_sha256": policy, "evidence_sha256": document_digest(prior_proof)}
    calls, checks = [], []
    def compile(arm):
        def run(candidate, source, scratch, **kwargs):
            calls.append((arm, kwargs))
            assert set(kwargs) == {"timeout_s", "emit_command_buffer"}
            assert 0 < kwargs["timeout_s"] <= 60 and kwargs["emit_command_buffer"]
            assert source.read_text() != SOURCE
            assert "tensor<2x5xi8>" in source.read_text()
            cb = {"kernel_abi": {"kind": "whole_program"}, "params": {"global_program_plan": {
                "tasks": [{"task_index": 0, "kind": "contraction", "source_op_indices": [0]}],
                "entry_bindings": ["A", "B", "C"], "output_bindings": ["Y"]}}}
            return {"lowered": subprocess.CompletedProcess([], 0, arm+" short llvm", ""),
                    "command_buffer_emission": subprocess.CompletedProcess([], 0, "", ""), "command_buffer": cb}
        return run
    def verify(**kwargs):
        checks.append(kwargs)
        assert kwargs["source_text"] != SOURCE
        assert "short llvm" in kwargs["lowered_text"]
        return {"status": "verified", "candidate_lowered_sha256": text_digest(kwargs["lowered_text"])}
    monkeypatch.setattr(P, "verify_compiler_global_plan", verify)
    current_binding = object()
    experiment = SimpleNamespace(
        iterations=[{"analysis": {"diagnostics": {"verified_global_plan_emission": prior_proof}}},
                    {"analysis": {"diagnostics": {"verified_global_plan_emission": current_proof,
                        "captured_logical_graph": {"logical_dispatch_digest": graph},
                        "verified_baseline_global_plan_emission": prior_proof,
                        "baseline_global_plan_evidence_binding": baseline_binding}}}],
        host_policy={"sha256": policy}, current_artifacts=lambda _: after, previous_artifacts=lambda _: before,
        current_probe_binding=lambda _: current_binding,
        previous_probe_binding=lambda _: SimpleNamespace(to_dict=lambda: {"identity": "previous"}),
        optimization_baseline_artifacts=lambda _: {**before, "compiler_sha256": before["candidate_sha256"],
            "lowered_sha256": before["candidate_lowered_sha256"], "command_buffer_sha256": before["candidate_command_buffer_sha256"]},
        optimization_baseline_artifact_binding=lambda _: baseline_binding,
        compile_previous_probe_candidate=compile("previous"),
        compile_optimization_baseline_probe_candidate=compile("baseline"), compile_probe_candidate=compile("current"))
    return experiment, calls, checks


def prepare(tmp_path, experiment, **kwargs):
    return P.prepare_source_contraction(candidate=tmp_path, experiment=experiment,
        comparison_arm=kwargs.pop("comparison_arm", "previous"), source_op_index=kwargs.pop("source_op_index", 0),
        entry="work", max_m=2, max_n=3, max_k=5, output=tmp_path/"prepared", **kwargs)


@pytest.mark.parametrize("arm", ["previous", "optimization_baseline"])
def test_exact_normal_arms_three_input_abi_and_independent_typed_oracle(tmp_path, monkeypatch, arm):
    experiment, calls, checks = fixture(tmp_path, monkeypatch)
    if arm == "optimization_baseline":
        experiment.iterations = experiment.iterations[-1:]
        experiment.previous_artifacts = lambda _: pytest.fail("baseline is not predecessor")
    result = prepare(tmp_path, experiment, comparison_arm=arm)
    assert result["status"] == "prepared", result
    assert [row[0] for row in calls] == ["previous" if arm == "previous" else "baseline", "current"]
    assert len(checks) == 2 and result["extraction"]["input_shapes"] == [[2,5],[5,3],[2,3]]
    oracle = json.loads(Path(result["independent_oracle"]).read_text())
    assert len(oracle["cases"]) == 3 and oracle["output_dtype"] == "i8"
    for row in result["arms"].values():
        assert text_digest(Path(row["command_buffer_path"]).read_text()) == row["command_buffer_canonical_sha256"]
        assert text_digest(Path(row["lowered_path"]).read_text()) == row["lowered_sha256"]
    assert result["numerical_qualification"] == "UNPROVEN" and not result["runtime_admitted"]
    assert result["emitted_route_correspondence"].startswith("UNKNOWN")
    assert result["selected_task_emission_change"].startswith("UNKNOWN")
    assert not result["full_model_recompiled"] and not result["full_model_proofs_recomputed"]
    # This fixture has no target-owned summaries: ordinary preparation must
    # expose that gap, rather than silently omit route relevance from feedback.
    assert result["task_route_presence"]["status"] == "UNKNOWN"
    assert result["task_route_presence_sha256"] == document_digest(result["task_route_presence"])
    assert result["task_route_feedback"]["status"] == "UNKNOWN"


@pytest.mark.parametrize("mutation", ["proof", "nonplan_buffer", "ownership", "wrong_index", "baseline_policy"])
def test_invalid_full_pair_never_compiles_or_reverifies_model(tmp_path, monkeypatch, mutation):
    experiment, calls, checks = fixture(tmp_path, monkeypatch)
    kwargs = {}
    if mutation == "proof":
        experiment.iterations[-1]["analysis"]["diagnostics"]["verified_global_plan_emission"]["status"] = "UNKNOWN"
    elif mutation == "nonplan_buffer":
        experiment.current_artifacts(None)["command_buffer"]["tensors"] = {"forged": True}
    elif mutation == "ownership":
        artifact = experiment.current_artifacts(None)
        P.program_plan(artifact["command_buffer"])["tasks"][0]["kind"] = "host"
        artifact["command_buffer_text"] = json.dumps(artifact["command_buffer"])
        artifact["candidate_command_buffer_sha256"] = text_digest(artifact["command_buffer_text"])
        proof = experiment.iterations[-1]["analysis"]["diagnostics"]["verified_global_plan_emission"]
        proof["candidate_command_buffer_sha256"] = artifact["candidate_command_buffer_sha256"]
        proof["plan_digest"] = document_digest(P.program_plan(artifact["command_buffer"]))
    elif mutation == "wrong_index":
        kwargs["source_op_index"] = 1
    else:
        experiment.host_policy["sha256"] = "changed"
        kwargs["comparison_arm"] = "optimization_baseline"
    result = prepare(tmp_path, experiment, **kwargs)
    assert result["status"] == "UNKNOWN" and calls == checks == []


@pytest.mark.parametrize("mutation", ["missing_initializer_arg", "wrong_owner", "wrong_abi", "failed_proof", "compile_timeout", "changed_source"])
def test_short_boundary_and_compilation_fail_closed(tmp_path, monkeypatch, mutation):
    experiment, calls, _ = fixture(tmp_path, monkeypatch)
    original = experiment.compile_previous_probe_candidate
    def changed(*args, **kwargs):
        if mutation == "compile_timeout":
            raise subprocess.TimeoutExpired(["normal"], kwargs["timeout_s"])
        result = original(*args, **kwargs)
        cb = result["command_buffer"]
        if mutation == "missing_initializer_arg":
            P.program_plan(cb)["entry_bindings"].pop()
        elif mutation == "wrong_owner":
            P.program_plan(cb)["tasks"][0]["kind"] = "host"
        elif mutation == "wrong_abi":
            cb["kernel_abi"]["kind"] = "unknown"
        elif mutation == "changed_source":
            args[1].write_text("changed")
        return result
    experiment.compile_previous_probe_candidate = changed
    if mutation == "failed_proof":
        monkeypatch.setattr(P, "verify_compiler_global_plan", lambda **_: {"status": "UNKNOWN"})
    result = prepare(tmp_path, experiment)
    assert result["status"] == "UNKNOWN" and "independent_oracle" not in result
    assert (Path(result["workdir"])/"preparation.json").is_file()


def test_one_deadline_covers_both_compilations(tmp_path, monkeypatch):
    experiment, calls, _ = fixture(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(P, "monotonic", lambda: clock[0])
    for name in ("compile_previous_probe_candidate", "compile_probe_candidate"):
        original = getattr(experiment, name)
        def slow(*args, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            clock[0] += 31
            return result
        setattr(experiment, name, slow)
    result = prepare(tmp_path, experiment)
    assert result["status"] == "UNKNOWN" and "deadline" in result["reason"]
    assert [row[1]["timeout_s"] for row in calls] == [60,29]


def test_changed_current_binding_never_returns_prepared(tmp_path, monkeypatch):
    experiment, _, _ = fixture(tmp_path, monkeypatch)
    original = experiment.compile_probe_candidate
    def change(*args, **kwargs):
        result = original(*args, **kwargs)
        experiment.current_probe_binding = lambda _: object()
        return result
    experiment.compile_probe_candidate = change
    assert prepare(tmp_path, experiment)["status"] == "UNKNOWN"


@pytest.mark.parametrize("mutation",[None,"unproved","source","llvm","buffer","compiler","declared_proof"])
def test_separate_initializer_admission_keeps_declared_refusal_and_requires_exact_pins(tmp_path,monkeypatch,mutation):
    from merlin.perf import source_initializer_elision
    experiment,_,_=fixture(tmp_path,monkeypatch)
    declared={"status":"refused","problems":["source operations are not fully covered: [0, 1]"]}
    monkeypatch.setattr(P,"verify_compiler_global_plan",lambda **_:json.loads(json.dumps(declared)))
    def admit(**kwargs):
        result={"schema":"short_initializer_execution_admission_v1","status":"source_bound_numerical_probe",
            "source_sha256":text_digest(kwargs["source_text"]),"lowered_sha256":text_digest(kwargs["lowered_text"]),
            "command_buffer_sha256":document_digest(kwargs["command_buffer"]),
            "candidate_sha256":kwargs["candidate_sha256"],"declared_proof_sha256":document_digest(kwargs["declared_proof"])}
        if mutation=="unproved":result["status"]="UNKNOWN"
        elif mutation is not None:
            key={"source":"source_sha256","llvm":"lowered_sha256","buffer":"command_buffer_sha256",
                "compiler":"candidate_sha256","declared_proof":"declared_proof_sha256"}[mutation]
            result[key]="stale"
        return result
    # This is a wiring test; the separate helper's source/command legality tests
    # establish the narrow initializer predicate, not this stubbed admission.
    monkeypatch.setattr(source_initializer_elision,"numerical_probe_admission",admit,raising=False)
    result=prepare(tmp_path,experiment)
    assert result["short_verification"]["before"]==declared
    if mutation is None:
        assert result["status"]=="prepared"
        assert all(arm["source_task_cfg_proof"]==declared for arm in result["arms"].values())
        assert all(arm["short_execution_admission"]["status"]=="source_bound_numerical_probe" for arm in result["arms"].values())
        assert not result["runtime_admitted"] and result["numerical_qualification"]=="UNPROVEN"
        assert result["emitted_route_correspondence"].startswith("UNKNOWN")
    else:
        assert result["status"]=="UNKNOWN" and "independent_oracle" not in result


@pytest.mark.parametrize("timeout", [0, 61, True, float("inf")])
def test_invalid_budget_not_silently_clamped(tmp_path, monkeypatch, timeout):
    experiment, _, _ = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="budget"):
        prepare(tmp_path, experiment, timeout_s=timeout)
