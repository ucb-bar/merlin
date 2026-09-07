"""Retained baseline ownership is actual host verification, never candidate-declared proof."""
from copy import deepcopy
import importlib
import json
import sys

import pytest

from merlin.benchharness import hash_tree
from merlin.common.paths import merlin_dir
from merlin.perf import compiler_plan_evidence
from merlin.targetgen import oot_runner
from merlin.targetgen.rocc import decode

sys.path.insert(0, str(merlin_dir() / "experiments/gemmini_perf_bench/scripts"))
PAS = importlib.import_module("perf_agent_stage")

SOURCE = '''builtin.module {
  func.func @forward() -> tensor<2xi32> {
    %0 = arith.constant dense<[3, 7]> : tensor<2xi32>
    func.return %0 : tensor<2xi32>
  }
}'''
LOWERED = '''builtin.module {
  llvm.func @kernel(%0: !llvm.ptr) {
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.getelementptr %0[%1] {merlin.global_task = 0 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return
  }
}'''


def fixture(tmp_path, monkeypatch, *, unknown=False, task_count=1, compiler_schema=None, tensor_role="output"):
    baseline, candidate, source = (tmp_path / name for name in ("base", "candidate", "source"))
    for path in (baseline, candidate, source):
        path.mkdir()
    (baseline / "compiler.py").write_text("BASE=1\n")
    (candidate / "compiler.py").write_text("CANDIDATE=1\n")
    (source / "capsule.yaml").write_text(json.dumps({"id": "fixture-model"}))
    source_text, lowered_text = SOURCE, LOWERED
    if task_count > 1:
        results = ", ".join(f"%v{i}" for i in range(task_count))
        types = ", ".join(["tensor<2xi32>"] * task_count)
        constants = "\n".join(f"%v{i} = arith.constant dense<[3, 7]> : tensor<2xi32>" for i in range(task_count))
        source_text = f"builtin.module {{ func.func @forward() -> ({types}) {{\n{constants}\nfunc.return {results} : {types}\n}} }}"
        arguments = ", ".join(f"%p{i}: !llvm.ptr" for i in range(task_count))
        accesses = "\n".join(f"%g{i} = llvm.getelementptr %p{i}[%index] {{merlin.global_task = {i} : i64}} : (!llvm.ptr, i64) -> !llvm.ptr, i32" for i in range(task_count))
        lowered_text = f"builtin.module {{ llvm.func @kernel({arguments}) {{\n%index = llvm.mlir.constant(0 : i64) : i64\n{accesses}\nllvm.return\n}} }}"
    (source / "capsule.interface.mlir").write_text(source_text)
    sentinel = PAS.StageE2ESentinel("fixture-model", str(source), str(source),
                                    PAS._exact_tree_record(source)["sha256"], (), ())
    buffer = {"commands": [], "tensors": {"result": {"shape": [2], "dtype": "i32", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "result"}], "outputs": ["result"]},
        "params": {"global_program_plan": {"schema": "mixed_program_plan_v1",
            "source_sha256": PAS._sha256(source_text.encode()), "source_op_count": 1,
            "tasks": [{"task_index": 0, "kind": "host", "source_op_indices": [0],
                "instruction_start": 1, "instruction_end": 2, "reads": [], "writes": ["result"]}],
            "schedule_instruction_count": 3, "prologue_instruction_range": [0, 1],
            "epilogue_instruction_range": [2, 3],
            "source_values": [{"op_index": 0, "result_index": 0, "tensor": "result"}],
            "entry_bindings": [], "output_bindings": ["result"]}}}
    if task_count > 1:
        names = [f"result{i}" for i in range(task_count)]
        buffer["tensors"] = {name: {"shape": [2], "dtype": "i32", "role": "output"} for name in names}
        buffer["kernel_abi"] = {"kind": "whole_program", "args": [{"tensor": name} for name in names], "outputs": names}
        plan = buffer["params"]["global_program_plan"]
        plan.update(source_op_count=task_count, tasks=[{"task_index": i, "kind": "host",
            "source_op_indices": [i], "instruction_start": i+1, "instruction_end": i+2,
            "reads": [], "writes": [name]} for i, name in enumerate(names)],
            schedule_instruction_count=task_count+2, epilogue_instruction_range=[task_count+1, task_count+2],
            source_values=[{"op_index": i, "result_index": 0, "tensor": name} for i, name in enumerate(names)],
            output_bindings=names)
    emits, verifies = [], []
    buffer.update(abi_version="0.1", target="fixture-target")
    for argument in buffer["kernel_abi"]["args"]:
        argument["access"] = "write"
    for tensor in buffer["tensors"].values():
        tensor["role"] = tensor_role
    def emit(package, interface, scratch, tag, timeout):
        emits.append(tag)
        payload = {} if unknown and tag == "baseline" else buffer
        return 0, lowered_text, json.dumps(payload)
    original = compiler_plan_evidence.verify_compiler_global_plan
    def verify(**kwargs):
        verifies.append((kwargs["candidate_sha256"], kwargs.get("parsed_lowered_module") is not None))
        return original(**kwargs)
    monkeypatch.setattr(compiler_plan_evidence, "verify_compiler_global_plan", verify)
    monkeypatch.setattr(oot_runner, "load_package", lambda path: path)
    # This fixture has no device instructions. Keep actual parsing and ownership
    # verification while avoiding an unrelated target-facts lookup.
    monkeypatch.setattr(decode, "decode_module", lambda *a, **kw: {"instructions": []})
    monkeypatch.setattr(PAS, "analyze_command_buffers", lambda *a, **kw: {"arms": {
        "baseline": {"status": "emitted"}, "candidate": {"status": "emitted"}}})
    monkeypatch.setattr(PAS, "inspect_compiler_package", lambda *_: None)
    monkeypatch.setattr(PAS, "guidance_for_emission_analysis", lambda *_: {})
    def analyze(cache=None, policy="a" * 64):
        retained = {}
        result = PAS.analyze_whole_model_emission(baseline, candidate, sentinel, timeout_s=10,
            peak_macs_per_cycle=None, achievable_macs_per_cycle=None, target="fixture-target",
            emit_pair_runner=emit, artifact_sink=retained.update, baseline_artifacts=cache,
            host_verifier_policy_sha256=policy, compiler_api_schema=compiler_schema)
        return result, retained["baseline_artifacts"]
    return analyze, emits, verifies, baseline, candidate


def test_host_schema_validates_compiler_that_does_not_self_validate(tmp_path, monkeypatch):
    analyze, emits, _, _, _ = fixture(tmp_path, monkeypatch,
        compiler_schema=PAS.whole_program_schema_record(), tensor_role="invented")
    with pytest.raises(PAS.StageGateError, match="baseline command buffer violates current compiler API schema"):
        analyze()
    assert emits == ["baseline"]


def test_task_instruction_summary_is_cached_with_exact_facts_and_no_baseline_reparse(tmp_path, monkeypatch):
    from merlin.kernels.decode import rocc
    from merlin.kernels import endpoints
    from merlin.perf import task_instruction_evidence
    analyze, emits, _, _, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(rocc,"funct_table_for",lambda _: {"names":{}})
    monkeypatch.setattr(endpoints,"endpoints_for",lambda _: [])
    facts={"revision":"host-facts-v1"}
    monkeypatch.setattr(decode,"isa_constants",lambda _: facts)
    calls=[]
    original=task_instruction_evidence.summarize_task_instructions
    def summarize(**kwargs):
        calls.append(kwargs["parsed_module"] is not None)
        return original(**kwargs)
    monkeypatch.setattr(task_instruction_evidence,"summarize_task_instructions",summarize)
    first,cache=analyze()
    first_summary=first["diagnostics"]["task_instruction_evidence"]["baseline"]
    assert first_summary["status"]=="static_ownership_verified"
    assert first_summary["tasks"][0]["source_op_indices"]==[0]
    assert first_summary["tasks"][0]["instruction_indices"]==[]
    assert first_summary["dynamic_instruction_counts"] is None
    second,_=analyze(json.loads(json.dumps(cache)))
    assert second["diagnostics"]["task_instruction_evidence"]["baseline"]==first_summary
    assert calls==[True,True,True] and emits==["baseline","candidate","candidate"]
    facts["revision"]="host-facts-v2"
    third,_=analyze(cache)
    assert third["diagnostics"]["task_instruction_evidence"]["baseline"]["status"]=="UNKNOWN"
    assert calls[-2:]==[False,True]  # changed facts never authorize reusing stale summary or reparsing


def test_current_schema_validated_with_real_plan_and_json_cache(tmp_path, monkeypatch):
    analyze, emits, _, _, _ = fixture(tmp_path, monkeypatch,
        compiler_schema=PAS.whole_program_schema_record(), task_count=12)
    first, cache = analyze()
    second, _ = analyze(json.loads(json.dumps(cache)))
    assert first["diagnostics"]["verified_baseline_global_plan_emission"]["status"] == "verified"
    assert second["diagnostics"]["verified_baseline_global_plan_emission"]["status"] == "verified"
    assert emits == ["baseline", "candidate", "candidate"]


def test_schema_changed_pin_rejected_before_candidate(tmp_path, monkeypatch):
    record = PAS.whole_program_schema_record()
    record["sha256"] = "0" * 64
    analyze, emits, _, _, _ = fixture(tmp_path, monkeypatch, compiler_schema=record)
    with pytest.raises(PAS.StageGateError, match="schema binding changed"):
        analyze()
    assert emits == ["baseline"]


def test_baseline_real_plan_verified_once_and_cache_reused(tmp_path, monkeypatch):
    analyze, emits, verifies, baseline, candidate = fixture(tmp_path, monkeypatch)
    first, cache = analyze()
    proof = first["diagnostics"]["verified_baseline_global_plan_emission"]
    assert proof["status"] == "verified"
    assert proof["numeric_equivalence"] == "requires independent mechanism witnesses"
    assert proof["candidate_sha256"] == hash_tree(baseline)["sha256"]
    assert cache["verified_global_plan_emission"] == proof
    assert cache["global_plan_evidence_binding"]["evidence_sha256"] == PAS._document_sha256(proof)
    assert cache["global_plan_evidence_binding"]["host_verifier_policy_sha256"] == "a" * 64
    assert verifies == [(hash_tree(baseline)["sha256"], True), (hash_tree(candidate)["sha256"], True)]
    (candidate / "compiler.py").write_text("CANDIDATE=2\n")
    original = deepcopy(cache)
    second, _ = analyze(cache)
    assert emits == ["baseline", "candidate", "candidate"]
    assert len(verifies) == 3
    assert second["diagnostics"]["verified_baseline_global_plan_emission"] == proof
    assert cache == original


def test_actual_multitask_proof_survives_worker_json_roundtrip(tmp_path, monkeypatch):
    analyze, emits, verifies, _, _ = fixture(tmp_path, monkeypatch, task_count=12)
    first, cache = analyze()
    assert first["diagnostics"]["verified_baseline_global_plan_emission"]["status"] == "verified"
    transported = json.loads(json.dumps(cache))
    assert "10" in transported["verified_global_plan_emission"]["declared_task_kinds"]
    second, next_cache = analyze(transported)
    assert emits == ["baseline", "candidate", "candidate"]
    assert len(verifies) == 3
    assert second["diagnostics"]["verified_baseline_global_plan_emission"] == transported["verified_global_plan_emission"]
    assert next_cache["global_plan_evidence_binding"] == transported["global_plan_evidence_binding"]


@pytest.mark.parametrize("mutation", ["source", "policy", "proof", "lowered", "buffer", "compiler"])
def test_baseline_cache_binding_mutation_fails_closed_before_candidate(tmp_path, monkeypatch, mutation):
    analyze, emits, verifies, _, _ = fixture(tmp_path, monkeypatch)
    _, cache = analyze()
    if mutation in ("source", "policy", "compiler"):
        key = {"source": "source_sha256", "policy": "host_verifier_policy_sha256",
               "compiler": "compiler_sha256"}[mutation]
        cache["global_plan_evidence_binding"][key] = "b" * 64
    elif mutation == "proof":
        cache["verified_global_plan_emission"]["status"] = "refused"
    else:
        cache["lowered_text" if mutation == "lowered" else "command_buffer_text"] += " "
    with pytest.raises(PAS.StageGateError, match="baseline.*identity changed|baseline.*binding changed"):
        analyze(cache)
    assert emits == ["baseline", "candidate"]
    assert len(verifies) == 2


def test_unknown_baseline_evidence_is_not_promoted_by_cache(tmp_path, monkeypatch):
    analyze, _, verifies, _, _ = fixture(tmp_path, monkeypatch, unknown=True)
    first, cache = analyze()
    assert first["diagnostics"]["verified_baseline_global_plan_emission"]["status"] == "UNKNOWN"
    second, _ = analyze(cache)
    assert second["diagnostics"]["verified_baseline_global_plan_emission"]["status"] == "UNKNOWN"
    assert len(verifies) == 3


def test_missing_policy_does_not_license_proof_reuse(tmp_path, monkeypatch):
    analyze, _, verifies, _, _ = fixture(tmp_path, monkeypatch)
    _, cache = analyze(policy=None)
    analyze(cache, policy=None)
    assert len(verifies) == 4
