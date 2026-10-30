"""Actual compilation of an index-width source exclusion; no proof credit."""

import json

from merlin.llvmlower import lower
from merlin.verify import compiler_transforms, pass_validation
from merlin.verify.speculative_gate import SpeculativeGate


def test_overflow_memory_source_uses_ordinary_DCE_without_a_proof_job(tmp_path, pass_gate_tools):
    source = """builtin.module {
func.func @forward(%input: memref<1xi32>) -> i32 {
  %unused = arith.constant 9223372036854775808 : index
  %zero = arith.constant 0 : index
  %value = memref.load %input[%zero] : memref<1xi32>
  func.return %value : i32
}
}"""
    options = compiler_transforms.Options(memory_source=True)
    ordinary = lower.lower_model(source, tmp_path / "ordinary", targets=(), formal_transforms=options)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=120)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({"prune_dead_pure_tensor_ops"}))
    checked = lower.lower_model(
        source, tmp_path / "selected", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == ordinary.ll_path.read_bytes()
    assert checked.stats["dead_tensor_ops_pruned"] == 1
    assert gate.verdicts == []
    coverage = json.loads((gate.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is True
    assert coverage["verified_boundaries"] == 0 and coverage["unverified_boundaries"] == 1
    assert coverage["boundaries"][0]["proof_credit"] is False


def test_all_memory_DCE_preserves_unverified_identity_qdq_coverage(tmp_path, pass_gate_tools):
    source = """builtin.module {
func.func @forward(%input: memref<1xi32>) -> i32 {
  %unused = arith.constant 7 : index
  %zero = arith.constant 0 : index
  %value = memref.load %input[%zero] : memref<1xi32>
  func.return %value : i32
}
}"""
    options = compiler_transforms.Options(memory_source=True)
    ordinary = lower.lower_model(source, tmp_path / "ordinary", targets=(), formal_transforms=options)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, pass_gate_tools)
    checked = lower.lower_model(
        source, tmp_path / "selected", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert validation.passes == frozenset(pass_validation.CHECKS)
    assert checked.ll_path.read_bytes() == ordinary.ll_path.read_bytes()
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    assert gate.verdicts[0]["detail"]["schema"] == "merlin.compiler-memory-dce.gate.v1"
    coverage = json.loads((gate.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is True
    assert coverage["verified_boundaries"] == 1 and coverage["unverified_boundaries"] == 1
    excluded = [row for row in coverage["boundaries"] if row["status"] == "unverified"]
    assert len(excluded) == 1 and excluded[0]["pass"] == "lower_quant_ext"
    assert excluded[0]["source_sha256"] == excluded[0]["final_sha256"] and excluded[0]["proof_credit"] is False
