"""Production receipt join binds cached summaries to independently selected bytes."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from merlin.perf import source_contraction_preparation as P
from merlin.perf import task_instruction_evidence as T
from merlin.perf.source_program_pair import document_digest as digest, text_digest


def scenario(tmp_path, monkeypatch, comparison_arm="optimization_baseline"):
    facts, policy = {"classes": {"load_execute": ["compute"], "loop_launch": ["compute"]}}, "f"*64
    monkeypatch.setattr(T, "target_instruction_facts", lambda target: facts)
    source, probe = "full host-owned source bytes", "short host-owned source bytes"
    (tmp_path / "interface.mlir").write_text(probe)
    extraction = {"source_sha256": text_digest(source), "probe_source_sha256": text_digest(probe), "source_op_index": 0}
    prepared = {"comparison_arm": comparison_arm, "source_op_index": 0, "probe_source_op_index": 0,
        "workdir": str(tmp_path), "extraction": extraction, "arms": {}}
    artifacts, full_proofs, summaries = {}, {}, {}
    for arm, compiler, instruction in (("before", "a"*64, "load_execute"), ("after", "b"*64, "loop_launch")):
        for domain, text in (("full", source), ("short", probe)):
            cb = {"params": {"global_program_plan": {"source_sha256": text_digest(text),
                "tasks": [{"task_index": 0, "source_op_indices": [0], "kind": "contraction"}]}}}
            raw = json.dumps(cb, sort_keys=True, separators=(",", ":"))
            llvm = domain + " " + arm + " host-retained LLVM bytes"
            proof = {"status": "verified", "source_sha256": text_digest(text),
                "candidate_sha256": compiler, "candidate_lowered_sha256": text_digest(llvm),
                "candidate_command_buffer_sha256": text_digest(raw), "logical_dispatch_digest": text_digest(text),
                "plan_digest": digest(P.program_plan(cb))}
            summary = {"schema": "task_instruction_evidence_v1", "status": "static_ownership_verified",
                "binding": T.task_instruction_binding(source_text=text, lowered_text=llvm,
                    command_buffer_text=raw, verified_plan=proof, target_facts=facts, host_policy_sha256=policy),
                "tasks": [{"task_index": 0, "source_op_indices": [0], "declared_task_kind": "contraction",
                    "instruction_indices": [0], "unknown_instruction_indices": [],
                    "instructions_without_target_roles": [], "class_counts": {instruction: 1},
                    "role_counts": {"compute": 1}, "owned_instruction_payload_sha256": text_digest(llvm)}]}
            if domain == "full":
                artifacts[arm] = {"lowered_text": llvm, "command_buffer_text": raw, "command_buffer": cb,
                    "compiler_sha256": compiler, "lowered_sha256": text_digest(llvm), "command_buffer_sha256": text_digest(raw)}
                full_proofs[arm], summaries[arm] = proof, summary
            else:
                lp, cp = tmp_path/(arm+".mlir"), tmp_path/(arm+".json")
                lp.write_text(llvm)
                cp.write_text(raw)
                prepared["arms"][arm] = {"lowered_path": str(lp), "command_buffer_path": str(cp),
                    "compiler_sha256": compiler, "lowered_sha256": text_digest(llvm),
                    "command_buffer_canonical_sha256": text_digest(raw), "source_task_cfg_proof": proof,
                    "task_instruction_evidence": summary, "task_instruction_evidence_sha256": digest(summary)}
    current = {"verified_global_plan_emission": full_proofs["after"],
        "verified_baseline_global_plan_emission": full_proofs["before"],
        "task_instruction_evidence": {"baseline": summaries["before"], "candidate": summaries["after"]}}
    previous = {"verified_global_plan_emission": full_proofs["before"],
        "task_instruction_evidence": {"candidate": summaries["before"]}}
    experiment = SimpleNamespace(target="test_target", host_policy={"sha256": policy},
        iterations=[{"analysis": {"diagnostics": previous}}, {"analysis": {"diagnostics": current}}])
    pair = SimpleNamespace(source=source, source_sha256=text_digest(source), graph_sha256=text_digest(source), artifacts=artifacts)
    return dict(candidate=tmp_path, experiment=experiment, pair=pair, prepared=prepared)


@pytest.mark.parametrize("comparison_arm", ["previous", "optimization_baseline"])
def test_join_uses_correct_full_arm_without_any_parser(tmp_path, monkeypatch, comparison_arm):
    args = scenario(tmp_path, monkeypatch, comparison_arm)
    monkeypatch.setattr(P, "parse_mlir_text", lambda *a, **k: pytest.fail("no model parse in cached join"))
    monkeypatch.setattr(P, "verify_compiler_global_plan", lambda **k: pytest.fail("no full verification in cached join"))
    result = P.task_route_presence_for_preparation(**args)
    assert result["status"] == "observed_known_class_presence_change_reproduced", result
    feedback = P.task_route_feedback(result)
    assert feedback["descriptor_semantic_equivalence"] == "UNKNOWN"
    assert not feedback["timing_calibration_admissible"] and not feedback["global_cost_validated"]


def test_mismatched_short_route_remains_explicit_scoped_feedback(tmp_path, monkeypatch):
    args = scenario(tmp_path, monkeypatch)
    item = args["prepared"]["arms"]["after"]
    item["task_instruction_evidence"]["tasks"][0]["class_counts"] = {"load_execute": 1}
    item["task_instruction_evidence_sha256"] = digest(item["task_instruction_evidence"])
    original = deepcopy(args["prepared"])
    result = P.task_route_presence_for_preparation(**args)
    assert result["status"] == "observed_known_class_presence_mismatch"
    assert args["prepared"] == original
    assert "retain its numerical/timing result" in P.task_route_feedback(result)["next_action"]


def test_provider_receipt_keeps_observations_when_route_mismatch_is_reported(tmp_path, monkeypatch):
    from merlin.perf.source_program_pair_provider import _record_task_route_presence
    args = scenario(tmp_path, monkeypatch)
    item = args["prepared"]["arms"]["after"]
    item["task_instruction_evidence"]["tasks"][0]["class_counts"] = {"different_dispatch": 1}
    item["task_instruction_evidence_sha256"] = digest(item["task_instruction_evidence"])
    # Receipt-content preservation test, not a simulated hardware execution.
    result = {"status": "passed", "arms": {"before": {"cycles": 215}, "after": {"cycles": 260}}}
    original = deepcopy(result)
    comparison = P.task_route_presence_for_preparation(**args)
    args["prepared"].update(task_route_presence=comparison, task_route_presence_sha256=digest(comparison))
    _record_task_route_presence(result, **args)
    assert all(result[key] == value for key, value in original.items())
    assert result["task_route_feedback"]["status"] == "observed_known_class_presence_mismatch"
    assert result["prepared_task_route_presence_matches"]
    assert not result["task_route_feedback"]["timing_calibration_admissible"]


@pytest.mark.parametrize("mutation", ["compiler_claim", "facts_claim", "policy_claim", "source_claim", "full_buffer", "short_bytes", "summary_sha", "missing"])
def test_independent_artifact_and_policy_binding_refuses_rehashed_claims(tmp_path, monkeypatch, mutation):
    args = scenario(tmp_path, monkeypatch)
    item = args["prepared"]["arms"]["after"]
    if mutation.endswith("_claim"):
        key = {"compiler_claim": "compiler_sha256", "facts_claim": "target_facts_sha256",
               "policy_claim": "host_verifier_policy_sha256", "source_claim": "source_sha256"}[mutation]
        item["task_instruction_evidence"]["binding"][key] = "0"*64
        item["task_instruction_evidence_sha256"] = digest(item["task_instruction_evidence"])
    elif mutation == "full_buffer":
        args["pair"].artifacts["before"]["command_buffer"]["extra"] = "not in emitted raw bytes"
    elif mutation == "short_bytes":
        from pathlib import Path
        Path(item["lowered_path"]).write_text("changed emitted bytes")
    elif mutation == "summary_sha":
        item["task_instruction_evidence_sha256"] = "0"*64
    else:
        item.pop("task_instruction_evidence")
    result = P.task_route_presence_for_preparation(**args)
    assert result["status"] == "UNKNOWN" and result.get("reason")
    assert not result["timing_calibration_admissible"]
