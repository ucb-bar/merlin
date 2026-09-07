from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from merlin.perf import source_convolution_preparation as P


class Binding:
    def __init__(self,value):
        self.value=value
    def to_dict(self):
        return {"identity":self.value}


def fixture(tmp_path,monkeypatch):
    source=tmp_path/"actual.mlir"
    source.write_text("host-owned full source")
    before_sha,after_sha="a"*64,"b"*64
    graph_sha="c"*64
    def artifact(sha,kind):
        cb={"params":{"global_program_plan":{"tasks":[{"task_index":0,"kind":kind,"source_op_indices":[6]}]}}}
        raw=json.dumps(cb,indent=2)+"\n"
        return {"interface":source,"lowered_text":"module "+kind,"command_buffer":cb,
            "command_buffer_text":raw,
            "candidate_sha256":sha,"candidate_lowered_sha256":P._sha("module "+kind),
            "candidate_command_buffer_sha256":P._sha(raw)}
    before,after=artifact(before_sha,"host"),artifact(after_sha,"convolution")
    def proof(artifact):
        return {"status":"verified","source_sha256":P._sha(source.read_text()),
            "candidate_sha256":artifact["candidate_sha256"],"candidate_lowered_sha256":artifact["candidate_lowered_sha256"],
            "candidate_command_buffer_sha256":artifact["candidate_command_buffer_sha256"],
            "logical_dispatch_digest":graph_sha,"plan_digest":P._digest(P._plan(artifact["command_buffer"]))}
    before_proof,after_proof=proof(before),proof(after)
    policy_sha="d"*64
    evidence_binding={"schema":"baseline_global_plan_evidence_binding_v1",
        "source_sha256":P._sha(source.read_text()),"lowered_sha256":before["candidate_lowered_sha256"],
        "command_buffer_sha256":before["candidate_command_buffer_sha256"],"compiler_sha256":before_sha,
        "host_verifier_policy_sha256":policy_sha,"evidence_sha256":P._digest(before_proof)}
    previous_binding,current_binding=Binding("previous"),Binding("current")
    calls=[]
    def compile_arm(kind):
        def compile(candidate,interface,scratch,**kwargs):
            calls.append((kind,kwargs))
            assert kwargs["emit_command_buffer"] is True
            assert 0 < kwargs["timeout_s"] <= 60
            cb={"kernel_abi":{"kind":"whole_program"},"params":{"global_program_plan":{
                "tasks":[{"task_index":0,"kind":kind,"source_op_indices":[2]}],
                "entry_bindings":["input","weight"],"output_bindings":["output"]}}}
            return {"lowered":subprocess.CompletedProcess([],0,"short "+kind,""),
                "command_buffer_emission":subprocess.CompletedProcess([],0,"",""),"command_buffer":cb}
        return compile
    experiment=SimpleNamespace(iterations=[{"analysis":{"diagnostics":{"verified_global_plan_emission":before_proof}}},
        {"analysis":{"diagnostics":{"captured_logical_graph":{"logical_dispatch_digest":graph_sha},
            "verified_global_plan_emission":after_proof,"verified_baseline_global_plan_emission":before_proof,
            "baseline_global_plan_evidence_binding":evidence_binding}}}],host_policy={"sha256":policy_sha},
        current_artifacts=lambda _:after,previous_artifacts=lambda _:before,
        current_probe_binding=lambda _:current_binding,previous_probe_binding=lambda _:previous_binding,
        optimization_baseline_artifacts=lambda _:{**before,"compiler_sha256":before_sha,
            "lowered_sha256":before["candidate_lowered_sha256"],"command_buffer_sha256":before["candidate_command_buffer_sha256"]},
        optimization_baseline_artifact_binding=lambda _:evidence_binding,
        compile_optimization_baseline_probe_candidate=compile_arm("host"),
        compile_previous_probe_candidate=compile_arm("host"),compile_probe_candidate=compile_arm("convolution"))
    monkeypatch.setattr(P,"observe_model_macs",lambda text,**kwargs:[(None,SimpleNamespace(
        source_op_index=6 if text==source.read_text() else 2,status="derived",macs=1000))])
    extraction={"schema":"actual_source_convolution_witness_v1","input_layout":"nchw","weight_layout":"oihw",
        "output_layout":"nchw","input_shape":[1,1,2,2],"weight_shape":[1,1,1,1],"output_shape":[1,1,2,2],
        "stride":[1,1],"dilation":[1,1],"padding":[0,0,0,0],"macs":4,"source_indices":[6]}
    monkeypatch.setattr(P,"extract_source_convolution",lambda *args,**kwargs:("short source",copy.deepcopy(extraction)))
    verified=[]
    def verify(**kwargs):
        assert kwargs["source_text"]=="short source", "must not reverify full model during probe preparation"
        verified.append(kwargs)
        return {"status":"verified","candidate_lowered_sha256":P._sha(kwargs["lowered_text"])}
    monkeypatch.setattr(P,"verify_compiler_global_plan",verify)
    return experiment,calls,verified


@pytest.mark.parametrize("comparison_arm",["optimization_baseline","previous"])
def test_prepares_only_short_normal_manifest_arms_and_preserves_reference(tmp_path,monkeypatch,comparison_arm):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    if comparison_arm=="optimization_baseline":
        experiment.iterations=experiment.iterations[-1:]
        experiment.previous_artifacts=lambda _:pytest.fail("initial comparison is not a previous iteration")
        experiment.compile_previous_probe_candidate=lambda *a,**kw:pytest.fail("wrong arm")
    else:
        experiment.optimization_baseline_artifacts=lambda _:pytest.fail("wrong arm")
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm=comparison_arm,
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="prepared",result
    assert [kind for kind,_ in calls]==["host","convolution"]
    assert len(verified)==2
    assert result["numerical_qualification"]=="UNPROVEN"
    assert result["runtime_admitted"] is False and result["simulator_executed"] is False
    assert result["full_model_proofs_recomputed"] is False
    assert Path(result["independent_oracle"]).is_file()


@pytest.mark.parametrize("field",["status","candidate_sha256","candidate_lowered_sha256","candidate_command_buffer_sha256",
                                 "source_sha256","logical_dispatch_digest","plan_digest"])
def test_stale_full_plan_refuses_without_any_compilation(tmp_path,monkeypatch,field):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    proof=experiment.iterations[-1]["analysis"]["diagnostics"]["verified_global_plan_emission"]
    proof[field]="stale"
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert calls==verified==[]


def test_missing_baseline_cached_policy_proof_is_not_recomputed(tmp_path,monkeypatch):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    del experiment.iterations[-1]["analysis"]["diagnostics"]["baseline_global_plan_evidence_binding"]
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="optimization_baseline",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert "static proof preparation" in result["reason"]
    assert calls==verified==[]


@pytest.mark.parametrize("field",["tensors","kernel_abi","commands"])
def test_nonplan_command_buffer_mutation_refuses_before_compilation(tmp_path,monkeypatch,field):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    experiment.current_artifacts(tmp_path)["command_buffer"][field] = {"tampered":True}
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert "original hash-bound raw bytes" in result["reason"]
    assert calls==verified==[]


def test_original_raw_and_canonical_hashes_are_not_interchangeable(tmp_path,monkeypatch):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    artifact=experiment.current_artifacts(tmp_path)
    assert P._sha(artifact["command_buffer_text"])!=P._digest(artifact["command_buffer"])
    artifact["command_buffer_text"] = json.dumps(artifact["command_buffer"],sort_keys=True,separators=(",",":"))
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert calls==verified==[]


def test_json_scalar_types_cannot_alias_by_python_equality(tmp_path,monkeypatch):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    artifact=experiment.current_artifacts(tmp_path)
    # bool False compares equal to integer zero in Python, but is not the same JSON value.
    artifact["command_buffer"]["params"]["global_program_plan"]["tasks"][0]["task_index"]=False
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert "original hash-bound raw bytes" in result["reason"]
    assert calls==verified==[]


def test_short_route_must_match_changed_full_source_owner(tmp_path,monkeypatch):
    experiment,calls,verified=fixture(tmp_path,monkeypatch)
    experiment.compile_probe_candidate=experiment.compile_previous_probe_candidate
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert "route" in result["reason"]
    assert "independent_oracle" not in result


def test_short_compiler_timeout_retains_unknown_preparation_receipt(tmp_path,monkeypatch):
    experiment,_,_=fixture(tmp_path,monkeypatch)
    def timeout(*args,**kwargs):
        raise subprocess.TimeoutExpired(["normal-entrypoint"],kwargs["timeout_s"])
    experiment.compile_previous_probe_candidate=timeout
    result=P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
                                        entry="compute",output=tmp_path/"prepared")
    assert result["status"]=="UNKNOWN"
    assert "TimeoutExpired" in result["reason"]
    assert (Path(result["workdir"])/"preparation.json").is_file()


@pytest.mark.parametrize("timeout",[0,-1,float("inf"),61])
def test_preparation_budget_must_be_explicit_and_bounded(tmp_path,monkeypatch,timeout):
    experiment,_,_=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="budget"):
        P.prepare_source_convolution(candidate=tmp_path,experiment=experiment,comparison_arm="previous",
            entry="compute",output=tmp_path/"prepared",timeout_s=timeout)


@pytest.mark.parametrize("comparison_arm", ["optimization_baseline", "previous"])
def test_shared_pair_binder_reuses_exact_model_evidence_without_running_compiler(tmp_path, monkeypatch, comparison_arm):
    from merlin.perf.source_program_pair import bind_source_program_pair
    experiment, calls, verified = fixture(tmp_path, monkeypatch)
    pair = bind_source_program_pair(candidate=tmp_path, experiment=experiment, comparison_arm=comparison_arm)
    assert pair.source == "host-owned full source"
    assert pair.source_sha256 == P._sha(pair.source)
    assert pair.owners["before"][6]["kind"] == "host"
    assert pair.owners["after"][6]["kind"] == "convolution"
    expected = (experiment.compile_optimization_baseline_probe_candidate if comparison_arm == "optimization_baseline"
                else experiment.compile_previous_probe_candidate)
    assert pair.compile_before is expected
    assert calls == verified == []


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5, 1])
def test_shared_pair_binder_refuses_invalid_or_exceeded_source_bounds(tmp_path, monkeypatch, limit):
    from merlin.perf.source_program_pair import bind_source_program_pair
    experiment, calls, verified = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="bound"):
        bind_source_program_pair(candidate=tmp_path, experiment=experiment,
                                 comparison_arm="previous", max_source_bytes=limit)
    assert calls == verified == []


def test_shared_pair_binder_does_not_substitute_baseline_for_missing_previous(tmp_path, monkeypatch):
    from merlin.perf.source_program_pair import bind_source_program_pair
    experiment, calls, verified = fixture(tmp_path, monkeypatch)
    experiment.iterations = experiment.iterations[-1:]
    experiment.optimization_baseline_artifacts = lambda _: pytest.fail("implicit comparator substitution")
    with pytest.raises(ValueError, match="preceding analyzed iteration"):
        bind_source_program_pair(candidate=tmp_path, experiment=experiment, comparison_arm="previous")
    assert calls == verified == []
