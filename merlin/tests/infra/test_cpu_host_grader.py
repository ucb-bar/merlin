"""The CPU-host grader builds once and compiles sealed capsules in isolated invocations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from merlin.benchharness import capsule_descriptor
from merlin.benchharness.host_agent import create_compiler_seal

_GRADER_PATH = Path(__file__).resolve().parents[2] / "experiments" / "cpu_host_compiler_v0" / "grader.py"
_SPEC = importlib.util.spec_from_file_location("cpu_host_grader_under_test", _GRADER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
grader = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(grader)
_BROKER_PATH = _GRADER_PATH.with_name("trusted_search_broker.py")
_BROKER_SPEC = importlib.util.spec_from_file_location(
    "cpu_host_trusted_search_broker_under_test", _BROKER_PATH)
assert _BROKER_SPEC is not None and _BROKER_SPEC.loader is not None
broker = importlib.util.module_from_spec(_BROKER_SPEC)
_BROKER_SPEC.loader.exec_module(broker)


def test_useful_vector_evidence_rejects_neutral_side_computation():
    disassembly = """
0000000000000000 <merlin_capsule_run>:
   0:\t0005e407\tvle32.v v8,(a1)
   4:\t02840457\tvadd.vv v8,v8,v8
   8:\t0004a023\tsw zero,0(a4)
   c:\t00008067\tret
"""
    evidence = grader._useful_vector_dataflow(disassembly)
    assert evidence["source_vector_loads"] == ["vle32.v"]
    assert evidence["computed_vector_registers"] == ["v8"]
    assert evidence["output_vector_stores"] == []
    assert evidence["useful"] is False


def test_useful_vector_evidence_requires_computed_value_to_reach_output():
    disassembly = """
0000000000000000 <merlin_capsule_run>:
   0:\t0005e407\tvle32.v v8,(a1)
   4:\t00066487\tvle32.v v9,(a2)
   8:\t02940457\tvadd.vv v8,v8,v9
   c:\t0204e027\tvse32.v v8,(a4)
  10:\t00008067\tret
"""
    evidence = grader._useful_vector_dataflow(disassembly)
    assert evidence["output_vector_stores"] == ["vse32.v"]
    assert evidence["useful"] is True


def test_useful_vector_evidence_rejects_later_scalar_output_overwrite():
    disassembly = """
0000000000000000 <merlin_capsule_run>:
   0:\t0005e407\tvle32.v v8,(a1)
   4:\t00066487\tvle32.v v9,(a2)
   8:\t02940457\tvadd.vv v8,v8,v9
   c:\t0204e027\tvse32.v v8,(a4)
  10:\t0004a023\tsw zero,0(a4)
  14:\t00008067\tret
"""
    evidence = grader._useful_vector_dataflow(disassembly)
    assert evidence["output_scalar_overwrites"] == ["sw"]
    assert evidence["useful"] is False


def test_useful_vector_evidence_parses_gnu_objdump_separate_operand_column():
    disassembly = """
0000000000000000 <merlin_capsule_run>:
  6e:\t011586b3          \tadd\ta3,a1,a7
  74:\t0006c803          \tlbu\ta6,0(a3)
  7e:\t02050e07          \tvle8.v\tv28,(a2)
  84:\tefc86457          \tvwmul.vx\tv8,v28,a6
  8c:\t4a83ae57          \tvsext.vf2\tv28,v8
  90:\t038e0c57          \tvadd.vv\tv24,v24,v28
  9c:\t96ba              \tadd\ta3,a3,a4
  a0:\t0206ec27          \tvse32.v\tv24,(a3)
  aa:\t8082              \tret
"""
    evidence = grader._useful_vector_dataflow(disassembly)
    assert evidence["vector_instructions"] == [
        "vadd.vv", "vle8.v", "vse32.v", "vsext.vf2", "vwmul.vx"]
    assert evidence["source_vector_loads"] == ["vle8.v"]
    assert evidence["output_vector_stores"] == ["vse32.v"]
    assert evidence["useful"] is True


def test_kernel_cannot_emit_or_terminate_trusted_receipt_process(tmp_path):
    if shutil.which("clang") is None or shutil.which("bwrap") is None:
        pytest.skip("native isolated grader tools are unavailable")
    kernel = tmp_path / "forged.c"
    kernel.write_text(r'''
#include <stdio.h>
#include <stdlib.h>
typedef struct { unsigned x[7]; unsigned long y[4]; } params_t;
int merlin_capsule_run(const params_t *p, const void *a, const void *b,
                       const void *c, void *o) {
  (void)p; (void)a; (void)b; (void)c; (void)o;
  puts("MERLIN_TRUSTED_RESULT version=1 seed=1 nonce=1 memory=1 numeric=1");
  fflush(stdout); exit(0);
}
''', encoding="utf-8")
    row = _row("heldout", "contraction", "matmul", "fp32",
               {"M": 2, "N": 3, "K": 5}, "row_row")
    result = grader._grade_native(
        row, {"ok": True, "_kernel_path": str(kernel)}, {"matmul": 1},
        tmp_path / "native")
    assert result["status"] == "fail"
    assert result["checks"]["trusted_parent_receipts"] is False
    assert grader._kernel_source_is_receipt_isolated(kernel.read_text()) is False


@pytest.mark.parametrize("escape", [
    "__real_pthread_create(0,0,0,0)",
    "__real_pthread_join((pthread_t)0,0)",
    "__real_pthread_setaffinity_np((pthread_t)0,0,0)",
    "syscall(220)",
    "clone(0,0,0,0)",
    "dlsym(0, \"pthread_create\")",
    "fwrite(0,1,0,stdout)",
    "writev(1,0,0)",
    '__asm__ volatile("ecall")',
])
def test_kernel_source_isolation_rejects_thread_and_output_bypasses(escape):
    source = f"int merlin_capsule_run(void) {{ {escape}; return 0; }}\n"
    assert grader._kernel_source_is_receipt_isolated(source) is False


def _audit_metrics(harts=4):
    full = (1 << harts) - 1
    workers = full & ~1
    output_count = 1024
    return {
        "calls": 20, "audit_call": 9, "audit_wall_ns": 10,
        "audit_time_ticks": 10, "correctness_checks": 21,
        "pinned_hart_mask": full, "worker_hart_mask": workers,
        "productive_worker_hart_mask": workers,
        "pthread_create_attempts": harts - 1, "pthread_creates": harts - 1,
        "pthread_create_failures": 0, "pthread_completions": harts - 1,
        "pthread_affinity_attempts": harts, "pthread_affinity_successes": harts,
        "pthread_affinity_failures": 0, "minimum_worker_cpu_ns": 100,
        "counterfactual_create_attempts": harts - 1,
        "counterfactual_creates": harts - 1, "counterfactual_create_failures": 0,
        "counterfactual_suppressed_starts": harts - 1,
        "counterfactual_worker_dependence": 1,
        "audit_serialized_callbacks": harts - 1,
        "audit_output_elements": output_count,
        "audit_output_coverage": output_count,
        "audit_owner_min_elements": output_count // harts,
        "audit_owner_max_elements": (output_count + harts - 1) // harts,
        "audit_ownership_violations": 0,
        "audit_balanced_shards": 1,
    }


def test_k1_timing_authority_requires_every_retained_call_and_exact_audit_workers():
    assert grader._k1_timing_authority(_audit_metrics(), 4) == (True, True)


def test_k1_timing_authority_rejects_exact_pinned_busy_decoy_workers():
    metrics = _audit_metrics()
    # The ordinary audit is deliberately perfect, including nontrivial worker CPU time.  The
    # trusted suppression challenge reveals that scalar main-thread output does not need workers.
    metrics["counterfactual_worker_dependence"] = 0
    assert grader._k1_timing_authority(metrics, 4) == (True, False)


def test_k1_timing_authority_rejects_flag_only_workers_even_with_counterfactual_failure():
    metrics = _audit_metrics()
    metrics["audit_owner_min_elements"] = 0
    metrics["audit_owner_max_elements"] = metrics["audit_output_elements"]
    metrics["audit_balanced_shards"] = 0
    assert metrics["counterfactual_worker_dependence"] == 1
    assert grader._k1_timing_authority(metrics, 4) == (True, False)


def test_failed_spike_gate_stops_before_k1_with_per_arm_evidence():
    row = {"id": "public-capsule", "family": "contraction"}
    passed = {"status": "pass", "checks": {
        "rvv_correctness": True, "instruction_evidence": True,
        "vlen_256": True, "cycle_measurement": True},
        "kernel_text_sha256": "a" * 64}
    failed = {**passed, "status": "fail", "checks": {
        **passed["checks"], "instruction_evidence": False},
        "vector_dataflow": {"useful": False, "vector_instructions": ["vle8.v"]}}
    compiled = {("parent", row["id"]): {"ok": True},
                ("candidate", row["id"]): {"ok": True}}
    with pytest.raises(grader.TrustedEvaluationFailure) as raised:
        grader._require_pre_k1_spike_gates(
            capsules=[row], compiled=compiled, compiled_k1=compiled,
            spike_records={("parent", row["id"]): passed,
                           ("candidate", row["id"]): failed})
    assert raised.value.evidence["k1_programs_started"] == 0
    assert raised.value.evidence["spike_gates"][row["id"]]["parent"]["passed"] is True
    candidate = raised.value.evidence["spike_gates"][row["id"]]["candidate"]
    assert candidate["passed"] is False
    assert candidate["checks"]["instruction_evidence"] is False
    assert candidate["vector_dataflow"]["vector_instructions"] == ["vle8.v"]


@pytest.mark.parametrize(("field", "value", "expected"), [
    ("correctness_checks", 1, (False, True)),
    ("pthread_create_attempts", 4, (True, False)),
    ("pthread_create_failures", 1, (True, False)),
    ("pthread_affinity_attempts", 5, (True, False)),
    ("pthread_affinity_failures", 1, (True, False)),
    ("pthread_completions", 4, (True, False)),
    ("productive_worker_hart_mask", 0, (True, False)),
    ("minimum_worker_cpu_ns", 0, (True, False)),
    ("counterfactual_suppressed_starts", 0, (True, False)),
    ("counterfactual_worker_dependence", 0, (True, False)),
    ("audit_serialized_callbacks", 2, (True, False)),
    ("audit_output_coverage", 1023, (True, False)),
    ("audit_owner_min_elements", 0, (True, False)),
    ("audit_ownership_violations", 1, (True, False)),
    ("audit_balanced_shards", 0, (True, False)),
])
def test_k1_timing_authority_rejects_cached_excess_failed_or_empty_work(
        field, value, expected):
    metrics = _audit_metrics()
    metrics[field] = value
    assert grader._k1_timing_authority(metrics, 4) == expected


def test_private_search_prebuild_preserves_real_submitted_manifest(tmp_path, monkeypatch):
    source = tmp_path / "submitted"
    source.mkdir()
    manifest = {
        "version": 1,
        "build": {"command": ["cmake", "-S", ".", "-B", "build"],
                  "then": ["cmake", "--build", "build"]},
        "compiler": {"command": ["build/compiler", "{input_mlir}", "{output_dir}",
                                  "{mode}", "{harts}", "{vlen_bits}"]},
        "policy": "policy.json",
    }
    (source / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (source / "policy.json").write_text("{}\n")

    def fake_build(package):
        (package / "build").mkdir()
        (package / "build" / "compiler").write_text("private executable\n")
        (package / "build" / "compiler").chmod(0o755)
        loaded = yaml.safe_load((package / "manifest.yaml").read_text())
        return loaded, {"commands": [
            {"command": loaded["build"]["command"], "returncode": 0,
             "wall_seconds": 0.1, "stdout_tail": "", "stderr_tail": ""},
            {"command": loaded["build"]["then"], "returncode": 0,
             "wall_seconds": 0.1, "stdout_tail": "", "stderr_tail": ""},
        ], "policy_sha256": hashlib.sha256((package / "policy.json").read_bytes()).hexdigest()}

    monkeypatch.setattr(grader, "_build", fake_build)
    destination = tmp_path / "private"
    receipt = grader.prepare_prebuilt_search_package(
        submission=source, destination=destination, build_override=["/bin/true"])
    assert yaml.safe_load((source / "manifest.yaml").read_text())["build"] == manifest["build"]
    private = yaml.safe_load((destination / "manifest.yaml").read_text())
    assert private["build"] == {"command": ["/bin/true"]}
    assert (destination / "build" / "compiler").is_file()
    assert receipt["real_build_commands"] == [manifest["build"]["command"],
                                               manifest["build"]["then"]]


def test_private_search_prebuild_rejects_noop_build_even_with_noncanonical_true(tmp_path,
                                                                                monkeypatch):
    source = tmp_path / "submitted"
    source.mkdir()
    manifest = {
        "version": 1, "build": {"command": ["/usr/bin/true"]},
        "compiler": {"command": ["build/compiler", "{input_mlir}", "{output_dir}",
                                  "{mode}", "{harts}", "{vlen_bits}"]},
        "policy": "policy.json",
    }
    (source / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (source / "policy.json").write_text("{}\n")

    def fake_noop(package):
        loaded = yaml.safe_load((package / "manifest.yaml").read_text())
        return loaded, {"commands": [{"command": ["/usr/bin/true"], "returncode": 0,
                                      "wall_seconds": 0.01, "stdout_tail": "",
                                      "stderr_tail": ""}],
                        "policy_sha256": hashlib.sha256(
                            (package / "policy.json").read_bytes()).hexdigest()}

    monkeypatch.setattr(grader, "_build", fake_noop)
    with pytest.raises(grader.GradeError, match="made no package change|compiler entrypoint"):
        grader.prepare_prebuilt_search_package(
            submission=source, destination=tmp_path / "private", build_override=["/bin/true"])


@pytest.mark.parametrize(("producer", "reason"), [
    ("failure", "submission build failed"),
    ("timeout", "submission build timed out"),
])
def test_submission_build_failures_are_typed_and_retain_stage_evidence(
        tmp_path, monkeypatch, producer, reason):
    package = tmp_path / "submission"; package.mkdir()
    (package / "policy.yaml").write_text("{}\n")
    (package / "manifest.yaml").write_text(yaml.safe_dump({
        "version": 1, "build": {"command": ["python3", "build.py"]},
        "compiler": {"command": ["compiler", "{input_mlir}", "{output_dir}",
                                  "{mode}", "{harts}", "{vlen_bits}"]},
        "policy": "policy.yaml",
    }))

    class Failed:
        returncode = 7
        stdout = "partial output"
        stderr = "compile error"

    def run(*_args, **_kwargs):
        if producer == "timeout":
            raise grader.SandboxTimeout(1800)
        return Failed()

    monkeypatch.setattr(grader, "_run_sandbox", run)
    with pytest.raises(grader.TreatmentBuildFailure) as captured:
        grader._build(package)
    assert captured.value.failure_class == "treatment_build_fail"
    assert captured.value.reason == reason
    evidence = captured.value.evidence
    assert evidence["failed_stage_index"] == 0
    assert evidence["commands"][0]["command"] == ["python3", "build.py"]
    if producer == "timeout":
        assert evidence["timeout_seconds"] == 1800
        assert evidence["commands"][0]["outcome"] == "timeout"
    else:
        assert evidence["returncode"] == 7
        assert evidence["commands"][0]["stderr_tail"] == "compile error"


def test_broker_preserves_treatment_build_failure_but_not_missing_controller_tool():
    submitted = grader.TreatmentBuildFailure(
        "submission build failed", {"commands": [], "failed_stage_index": 0})
    missing_controller_tool = grader.GradeError("trusted LLVM tool is absent")
    assert broker._broker_failure_class(submitted) == "treatment_build_fail"
    assert broker._broker_failure_class(missing_controller_tool) == "harness_invalid"


def test_compile_invocation_uses_label_blind_paths_and_cleared_environment(tmp_path, monkeypatch):
    package = tmp_path / "baseline_label" / "submission"
    package.mkdir(parents=True)
    compiler = package / "compiler"
    compiler.write_text("#!/bin/sh\nexit 1\n")
    compiler.chmod(0o755)
    manifest = {"compiler": {"command": ["compiler", "--input", "{input_mlir}",
                                               "--output", "{output_dir}", "--mode", "{mode}",
                                               "--harts", "{harts}", "--vlen", "{vlen_bits}"]}}
    row = _row("train", "contraction", "matmul", "fp32", {"M": 2, "N": 3, "K": 5},
               "row_row")
    captured = []
    class Result:
        returncode = 1
        stdout = ""
        stderr = ""
    monkeypatch.setattr(grader, "_run_sandbox",
                        lambda argv, **kwargs: captured.append(argv) or Result())
    grader._compile_one(package, manifest, row, "rvv", {"matmul": 1}, tmp_path / "work")
    argv = captured[0]
    assert "--clearenv" in argv
    assert "--chdir" in argv and argv[argv.index("--chdir") + 1] == "/package"
    assert argv[-11:] == ["compiler", "--input", "/work/ro_0", "--output", "/work/output",
                          "--mode", "rvv", "--harts", "1", "--vlen", "256"]


@pytest.mark.parametrize(("timed_out_stage", "reason"), [
    ("c_syntax", "C syntax check timed out"),
    ("mlir_verifier", "MLIR verifier timed out"),
])
def test_l0_syntax_and_verifier_timeouts_are_retained_candidate_failures(
        tmp_path, monkeypatch, timed_out_stage, reason):
    package = tmp_path / "submission"; package.mkdir()
    compiler = package / "compiler"; compiler.write_text("compiler\n"); compiler.chmod(0o755)
    manifest = {"compiler": {"command": [
        "compiler", "{input_mlir}", "{output_dir}", "{mode}", "{harts}", "{vlen_bits}"]}}
    row = _row("heldout", "contraction", "matmul", "fp32",
               {"M": 2, "N": 3, "K": 5}, "row_row")
    work_root = tmp_path / "work"

    class Compiled:
        returncode = 0
        stdout = "compiler output"
        stderr = ""

    def compile_candidate(*_args, **_kwargs):
        output = work_root / f"{row['id']}_scalar" / "output"
        source = "int merlin_capsule_run(void) { return 0; }\n"
        (output / "kernel.c").write_text(source)
        (output / "lowered.mlir").write_text("module { func.func @lowered() }\n")
        (output / "metadata.json").write_text(json.dumps({
            "version": 1, "capsule_sha256": row["sha256"],
            "requested_mode": "scalar", "actual_mode": "scalar",
            "fallback_used": False, "harts": 1, "vlen_bits": 256,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "transformations": ["generic"], "vlen_policy": "not_applicable",
            "tail_policy": "not_applicable",
        }))
        return Compiled()

    calls = 0

    class ToolResult:
        returncode = 0
        stdout = ""
        stderr = ""

    def tool(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if timed_out_stage == "c_syntax" or calls == 2:
            raise grader.subprocess.TimeoutExpired(["trusted-tool"], 60, stderr="partial")
        return ToolResult()

    monkeypatch.setattr(grader, "_run_sandbox", compile_candidate)
    monkeypatch.setattr(grader, "_llvm_tool", lambda name: Path(f"/trusted/{name}"))
    monkeypatch.setattr(grader.subprocess, "run", tool)
    result = grader._compile_one(
        package, manifest, row, "scalar", {"matmul": 1}, work_root)
    assert result["ok"] is False
    assert result["reason"] == reason
    assert result["timed_out_stage"] == timed_out_stage
    assert result["timeout_seconds"] == 60
    assert result["checks"]["c_syntax"] is (timed_out_stage == "mlir_verifier")
    assert result["checks"]["mlir_verifier"] is False
    assert result["source_sha256"] == result["metadata"]["source_sha256"]


def test_post_campaign_compiler_seal_is_verified_before_grading(tmp_path):
    package = _submission(tmp_path)
    seal = create_compiler_seal(
        workspace=tmp_path, search_seal={"status": "not_required"})
    seal_path = tmp_path / "compiler_seal.json"
    seal_path.write_text(json.dumps(seal))
    assert grader._verify_compiler_seal(package, seal_path)["status"] == "pass"
    (package / "policy.yaml").write_text("changed: true\n")
    with pytest.raises(grader.GradeError, match="compiler seal failed"):
        grader._verify_compiler_seal(package, seal_path)


def test_grader_rejects_post_seal_search_tree_mutation(tmp_path):
    package = _submission(tmp_path)
    search = package / "search"
    search.mkdir()
    (search / "search_record.json").write_text('{"status":"converged"}\n')
    seal = create_compiler_seal(
        workspace=tmp_path, search_seal={"status": "not_required"})
    seal_path = tmp_path / "compiler_seal.json"
    seal_path.write_text(json.dumps(seal))
    assert grader._verify_compiler_seal(package, seal_path)["status"] == "pass"

    (search / "search_record.json").write_text('{"status":"changed"}\n')
    with pytest.raises(grader.GradeError, match="compiler seal failed"):
        grader._verify_compiler_seal(package, seal_path)


def _row(split: str, family: str, operation: str, dtype: str, shape: dict, layout: str,
         state="stateless", core_count: int = 1) -> dict:
    identity = {"family": family, "operation": operation, "dtype": dtype, "shape": shape,
                "layout": layout, "state": state, "core_count": core_count}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return {"id": f"{family}-{operation}-{digest[:16]}", "sha256": digest, "split": split,
            **identity}


def test_public_descriptor_renderer_is_the_exact_grader_input_abi(tmp_path):
    codes = grader._codes(list(capsule_descriptor.conformance_rows()), "operation")
    assert codes == capsule_descriptor.OPERATION_CODE
    assert grader._codes([capsule_descriptor.conformance_rows()[1]], "operation") == codes

    mlir_opt = Path(__file__).resolve().parents[3] / "third_party/llvm-install/bin/mlir-opt"
    if not mlir_opt.is_file():
        pytest.skip("trusted mlir-opt is unavailable")
    for row in capsule_descriptor.conformance_rows():
        public = capsule_descriptor.render_capsule_mlir(row)
        assert grader._capsule_mlir(dict(row), codes) == public
        descriptor = tmp_path / f'{row["family"]}.mlir'
        descriptor.write_text(public, encoding="utf-8")
        verified = subprocess.run(
            [str(mlir_opt), "--allow-unregistered-dialect", str(descriptor)],
            capture_output=True, text=True, timeout=30)
        assert verified.returncode == 0, verified.stderr


def test_operation_codes_are_stable_across_split_specific_subsets():
    add = _row("train", "elementwise_map", "add", "fp32", {"length": 7}, "contiguous")
    unpack = _row(
        "heldout", "movement_layout", "unpack", "int8", {"working_set_bytes": 17},
        "operation_defined")
    assert grader._codes([add], "operation") == grader._codes([unpack], "operation")
    assert grader._codes([add], "operation")["add"] == 1
    assert grader._codes([unpack], "operation")["unpack"] == 29


def test_capsule_descriptor_rejects_unknown_or_nonpositive_public_fields():
    row = dict(capsule_descriptor.conformance_rows()[0])
    row["operation"] = "private_heldout_operation"
    with pytest.raises(ValueError, match="outside the public enum tables"):
        capsule_descriptor.render_capsule_mlir(row)
    row = dict(capsule_descriptor.conformance_rows()[0])
    row["core_count"] = 0
    with pytest.raises(ValueError, match="positive integer"):
        capsule_descriptor.render_capsule_mlir(row)


def test_movement_byte_budget_uses_complete_dtype_elements_only():
    row = _row(
        "train", "movement_layout", "copy", "fp32", {"working_set_bytes": 67},
        "operation_defined")
    assert capsule_descriptor.dimensions(row) == (16, 4, 4, 8)
    assert "dim0 = 16 : i64" in capsule_descriptor.render_capsule_mlir(row)


def _split(path: Path, split: str) -> None:
    offset = {"train": 0, "validation": 100, "heldout": 200}[split]
    rows = [
        _row(split, "contraction", "matmul", "fp32", {"M": 3, "N": 5, "K": 7 + offset}, "row_row"),
        _row(split, "elementwise_map", "relu", "fp32", {"length": 17 + offset}, "contiguous"),
        _row(split, "reduction", "sum", "fp32", {"length": 31 + offset}, "contiguous"),
        _row(split, "movement_layout", "copy", "int8", {"working_set_bytes": 65 + offset},
             "operation_defined"),
        _row(split, "fusion_epilogue", "matmul_bias", "fp32",
             {"M": 2, "N": 3, "K": 5 + offset}, "row_row", state="fused_epilogue"),
        _row(split, "runtime_parallel", "static_partition", "fp32", {"work_items": 1024 + offset},
             "contiguous", state={"reuse_count": 2}, core_count=4),
    ]
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def _submission(root: Path, *, correct: bool = False, vector: bool = False,
                multicore: bool = False, busy_decoy: bool = False) -> Path:
    package = root / "submission"
    package.mkdir()
    (package / "policy.yaml").write_text("version: 1\ndispatch: generic\n", encoding="utf-8")
    compiler = r'''#!/usr/bin/python3
import argparse
import hashlib
import json
import re
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--input")
p.add_argument("--output-dir")
p.add_argument("--mode")
p.add_argument("--harts")
p.add_argument("--vlen-bits")
a = p.parse_args()
text = Path(a.input).read_text()
out = Path(a.output_dir)
out.mkdir(exist_ok=True)
digest = re.search(r'sha256 = "([0-9a-f]{64})"', text).group(1)
source = """#include <stdint.h>
typedef struct { uint32_t x[7]; uint64_t y[4]; } merlin_capsule_params_t;
/* MODE %s */
int merlin_capsule_run(const merlin_capsule_params_t *p, const void *a, const void *b,
                       const void *c, void *o) {
  (void)p; (void)a; (void)b; (void)c; (void)o; return 0;
}
""" % a.mode
if __CORRECT__:
  source = """#define _GNU_SOURCE
#include <stdint.h>
#include <string.h>
#if MC_ENABLED
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#endif
typedef struct { uint32_t version, family, operation, dtype, layout, harts, vlen_bits;
                 uint64_t dim0, dim1, dim2, state0; } merlin_capsule_params_t;
#if MC_ENABLED
typedef struct { const float *a,*b; float *o; uint64_t begin,end; unsigned hart;
                 atomic_uint_fast64_t *ready; } test_job_t;
static int pin_to(unsigned hart) {
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(hart,&set);
  return pthread_setaffinity_np(pthread_self(),sizeof(set),&set);
}
static void *test_worker(void *value) {
  test_job_t *j=value;
  if(pin_to(j->hart)) return (void *)1;
#if DECOY_ENABLED
  volatile float sink=0;
  for(unsigned repeat=0;repeat<65536;repeat++) for(uint64_t i=j->begin;i<j->end;i++)
    sink+=j->a[i]*0.000001f;
  (void)sink;
  atomic_fetch_or_explicit(j->ready,UINT64_C(1)<<j->hart,memory_order_release);
#else
  for(unsigned repeat=0;repeat<65536;repeat++) for(uint64_t i=j->begin;i<j->end;i++)
    j->o[i]=j->a[i]+j->b[i];
#endif
  return 0;
}
#endif
/* MODE %s */
int merlin_capsule_run(const merlin_capsule_params_t *p, const void *va, const void *vb,
                       const void *vc, void *vo) {
  const float *a=va,*b=vb,*c=vc; float *o=vo;
#if MC_ENABLED
  if (p->family == 6 && p->harts > 1) {
    pthread_t threads[8]; test_job_t jobs[8]; uint64_t chunk=(p->dim0+p->harts-1)/p->harts;
#if DECOY_ENABLED
    atomic_uint_fast64_t ready=ATOMIC_VAR_INIT(1);
    if(pin_to(0)) return 6;
#endif
    for(uint32_t h=1;h<p->harts;h++) {
      jobs[h]=(test_job_t){a,b,o,h*chunk,(h+1)*chunk<p->dim0?(h+1)*chunk:p->dim0,h,
#if DECOY_ENABLED
                           &ready
#else
                           0
#endif
      };
      if(pthread_create(&threads[h],0,test_worker,&jobs[h])) return 7;
    }
    jobs[0]=(test_job_t){a,b,o,0,chunk<p->dim0?chunk:p->dim0,0,0};
#if DECOY_ENABLED
    for(uint32_t h=1;h<p->harts;h++) pthread_join(threads[h],0);
    if(atomic_load_explicit(&ready,memory_order_acquire)!=
       ((UINT64_C(1)<<p->harts)-1)) return 8;
    for(uint64_t i=0;i<p->dim0;i++) o[i]=a[i]+b[i];
#else
    test_worker(&jobs[0]);
    for(uint32_t h=1;h<p->harts;h++) pthread_join(threads[h],0);
#endif
    return 0;
  }
#endif
  if (p->family == 1) {
    for(uint64_t i=0;i<p->dim0;i++) for(uint64_t j=0;j<p->dim1;j++) {
      float z=0; for(uint64_t k=0;k<p->dim2;k++) z+=a[i*p->dim2+k]*b[k*p->dim1+j];
      o[i*p->dim1+j]=z;
    }
  } else if (p->family == 2) {
    for(uint64_t i=0;i<p->dim0;i++) o[i]=a[i]>0?a[i]:0;
  } else if (p->family == 3) {
    float z=0; for(uint64_t i=0;i<p->dim0;i++) z+=a[i]; o[0]=z;
  } else if (p->family == 4) {
    memcpy(vo,va,p->dim0);
  } else if (p->family == 5) {
    for(uint64_t i=0;i<p->dim0;i++) for(uint64_t j=0;j<p->dim1;j++) {
      float z=0; for(uint64_t k=0;k<p->dim2;k++) z+=a[i*p->dim2+k]*b[k*p->dim1+j];
      o[i*p->dim1+j]=z+c[j];
    }
  } else {
    for(uint64_t i=0;i<p->dim0;i++) o[i]=a[i]+b[i];
  }
  return 0;
}
""" % a.mode
  source = source.replace("MC_ENABLED", "1" if __MULTICORE__ and a.mode == "rvv_multicore" else "0")
  source = source.replace("DECOY_ENABLED", "1" if __BUSY_DECOY__ and a.mode == "rvv_multicore" else "0")
if __VECTOR__ and a.mode != "scalar":
  head, marker, tail = source.rpartition("  return 0;\n}")
  source = head + """  __asm__ volatile(
    "li t0, 1\\n\\tvsetvli t0, t0, e8, m1, ta, ma\\n\\tvle8.v v0, (%0)\\n\\tvadd.vx v0, v0, zero\\n\\tvse8.v v0, (%0)"
    : : "r"(vo) : "t0", "memory");
  return 0;
}""" + tail
(out / "kernel.c").write_text(source)
(out / "lowered.mlir").write_text(text + "// lowered " + a.mode + "\n")
meta = {
    "version": 1, "capsule_sha256": digest, "requested_mode": a.mode,
    "actual_mode": a.mode, "fallback_used": False, "harts": int(a.harts),
    "vlen_bits": int(a.vlen_bits),
    "vlen_policy": "not_applicable" if a.mode == "scalar" else "scalable_vl",
    "tail_policy": "not_applicable" if a.mode == "scalar" else "dynamic_vl",
    "transformations": ["generic-test-pass"],
    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
}
(out / "metadata.json").write_text(json.dumps(meta))
'''
    script = package / "compiler.py"
    compiler = compiler.replace("__CORRECT__", "1" if correct else "0")
    compiler = compiler.replace("__MULTICORE__", "1" if multicore else "0")
    compiler = compiler.replace("__BUSY_DECOY__", "1" if busy_decoy else "0")
    script.write_text(compiler.replace("__VECTOR__", "1" if vector else "0"), encoding="utf-8")
    script.chmod(0o755)
    manifest = {
        "version": 1,
        "build": {"command": ["/bin/true"]},
        "compiler": {"command": [
            "/usr/bin/python3", "compiler.py", "--input", "{input_mlir}",
            "--output-dir", "{output_dir}", "--mode", "{mode}",
            "--harts", "{harts}", "--vlen-bits", "{vlen_bits}",
        ]},
        "policy": "policy.yaml",
    }
    (package / "manifest.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return package


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_l0_grader_is_hermetic_and_requires_mode_specific_code(tmp_path):
    package = _submission(tmp_path)
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    _split(train, "train")
    _split(validation, "validation")
    _split(heldout, "heldout")
    target = tmp_path / "target.yaml"
    target.write_text("name: cpu\n")
    dialect = tmp_path / "dialect.yaml"
    dialect.write_text("target: cpu\n")
    args = argparse.Namespace(
        submission=package, target_contract=target, dialect_plan=dialect,
        train=train, validation=validation, heldout=heldout, run_l3=False)
    try:
        result = grader.grade(args)
    except grader.GradeError as exc:
        if "Operation not permitted" in str(exc) or "denied" in str(exc):
            pytest.skip(f"kernel disallows bubblewrap: {exc}")
        raise
    assert result["levels"]["L0"]["status"] == "pass"
    assert result["levels"]["L1"]["status"] == "fail"
    assert len(result["levels"]["L0"]["records"]) == 13
    assert all(result["levels"]["L0"]["scalar_rvv_source_change"].values())


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_l1_grader_executes_randomized_reference_and_memory_checks(tmp_path):
    package = _submission(tmp_path, correct=True)
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("train", "validation", "heldout")}
    for name, path in paths.items():
        _split(path, name)
    target = tmp_path / "target.yaml"; target.write_text("name: cpu\n")
    dialect = tmp_path / "dialect.yaml"; dialect.write_text("target: cpu\n")
    args = argparse.Namespace(submission=package, target_contract=target, dialect_plan=dialect,
                              run_l3=False, **paths)
    try:
        result = grader.grade(args)
    except grader.GradeError as exc:
        if "Operation not permitted" in str(exc) or "denied" in str(exc):
            pytest.skip(f"kernel disallows bubblewrap: {exc}")
        raise
    assert result["levels"]["L0"]["status"] == "pass"
    assert result["levels"]["L1"]["status"] == "pass"
    assert all(len(record["trials"]) == 3 for record in result["levels"]["L1"]["records"])
    assert all(trial["seed"] for record in result["levels"]["L1"]["records"]
               for trial in record["trials"])


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_l2_grader_rejects_semantics_neutral_vector_side_work(tmp_path):
    package = _submission(tmp_path, correct=True, vector=True)
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("train", "validation", "heldout")}
    for name, path in paths.items():
        _split(path, name)
    target = tmp_path / "target.yaml"; target.write_text("name: cpu\n")
    dialect = tmp_path / "dialect.yaml"; dialect.write_text("target: cpu\n")
    args = argparse.Namespace(submission=package, target_contract=target, dialect_plan=dialect,
                              run_l3=False, **paths)
    result = grader.grade(args)
    assert result["levels"]["L2"]["status"] == "fail"
    assert all(record["checks"]["rvv_correctness"] for record in result["levels"]["L2"]["records"])
    assert all(record["checks"]["vlen_256"] for record in result["levels"]["L2"]["records"])
    assert all(record["checks"]["instruction_evidence"] is False
               for record in result["levels"]["L2"]["records"])
    assert all(record["kernel_text_sha256"] for record in result["levels"]["L2"]["records"])


@pytest.mark.skipif(os.environ.get("MERLIN_RUN_K1_GRADER") != "1",
                    reason="set MERLIN_RUN_K1_GRADER=1 for the real-silicon grader test")
def test_l3_grader_records_exact_silicon_mode_vlen_harts_wall_and_rss(tmp_path):
    package = _submission(tmp_path, correct=True, vector=True, multicore=True)
    paths = {name: tmp_path / f"{name}.jsonl" for name in ("train", "validation", "heldout")}
    for name, path in paths.items():
        _split(path, name)
    target = tmp_path / "target.yaml"; target.write_text("name: cpu\n")
    dialect = tmp_path / "dialect.yaml"; dialect.write_text("target: cpu\n")
    result = grader.grade(argparse.Namespace(
        submission=package, target_contract=target, dialect_plan=dialect, run_l3=True, **paths))
    assert result["levels"]["L3"]["status"] == "pass"
    for record in result["levels"]["L3"]["records"]:
        assert all(record["checks"].values())
        assert record["metrics"]["vlenb"] == 32
        assert record["metrics"]["wall_ns"] > 0
        assert record["metrics"]["peak_rss_kb"] > 0


@pytest.mark.skipif(os.environ.get("MERLIN_RUN_K1_GRADER") != "1",
                    reason="set MERLIN_RUN_K1_GRADER=1 for the real-silicon grader test")
def test_l3_accepts_partitioned_exact_pinned_workers_on_public_capsule(tmp_path):
    package = _submission(tmp_path, correct=True, vector=True, multicore=True)
    manifest = yaml.safe_load((package / "manifest.yaml").read_text())
    row = _row("train", "runtime_parallel", "static_partition", "fp32",
               {"work_items": 1024}, "contiguous", state={"reuse_count": 2}, core_count=4)
    compile_record = grader._compile_one(
        package, manifest, row, "rvv_multicore", {"static_partition": 1},
        tmp_path / "compile")
    assert compile_record["ok"] is True
    record = grader._grade_k1(
        row, compile_record, {"static_partition": 1}, tmp_path / "k1", seed=1234567)
    assert record["status"] == "pass", json.dumps(record, indent=2)
    assert record["checks"]["per_call_correctness"] is True
    assert record["checks"]["audit_attribution"] is True
    assert record["metrics"]["correctness_checks"] == record["metrics"]["calls"] + 1
    assert record["metrics"]["counterfactual_worker_dependence"] == 1


@pytest.mark.skipif(os.environ.get("MERLIN_RUN_K1_GRADER") != "1",
                    reason="set MERLIN_RUN_K1_GRADER=1 for the real-silicon grader test")
def test_l3_rejects_scalar_main_with_exact_pinned_busy_decoy_workers(tmp_path):
    package = _submission(
        tmp_path, correct=True, vector=True, multicore=True, busy_decoy=True)
    manifest = yaml.safe_load((package / "manifest.yaml").read_text())
    row = _row("train", "runtime_parallel", "static_partition", "fp32",
               {"work_items": 1024}, "contiguous", state={"reuse_count": 2}, core_count=4)
    compile_record = grader._compile_one(
        package, manifest, row, "rvv_multicore", {"static_partition": 1},
        tmp_path / "compile")
    assert compile_record["ok"] is True
    record = grader._grade_k1(
        row, compile_record, {"static_partition": 1}, tmp_path / "k1", seed=1234567)
    assert record["status"] == "fail"
    assert record["checks"]["per_call_correctness"] is True
    assert record["checks"]["audit_attribution"] is False
    assert record["metrics"]["audit_serialized_callbacks"] == 3
    assert record["metrics"]["audit_output_coverage"] == 1024
    assert record["metrics"]["audit_owner_min_elements"] == 0
    assert record["metrics"]["audit_owner_max_elements"] == 1024
    assert record["metrics"]["audit_balanced_shards"] == 0
    assert record["metrics"]["counterfactual_worker_dependence"] == 1
    assert "worker_dependence=1" in record["monitor"]["child_stderr"]
    assert "audit_shards=0" in record["monitor"]["child_stderr"]
