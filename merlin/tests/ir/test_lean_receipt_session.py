"""Live proof results may be reused inside one job; edited or saved JSON must replay.

These tests run the real exporter and Lean checker and count actual proof
invocations. Refused transformations remain refused even with forged receipts.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from merlin.verify import gate_checks
from merlin.verify.lean import check
from merlin.verify.lean.receipt_session import reuse_receipts

BEFORE = """module {
  func.func @forward(%x: i32) -> i32 {
    %dead = arith.addi %x, %x : i32
    return %x : i32
  }
}
"""
AFTER = """module {
  func.func @forward(%x: i32) -> i32 {
    return %x : i32
  }
}
"""
WRONG = """module {
  func.func @forward(%x: i32) -> i32 {
    %twice = arith.addi %x, %x : i32
    return %twice : i32
  }
}
"""


@pytest.fixture
def proof_calls(monkeypatch):
    actual = check.subprocess.run
    calls = Counter()
    exe = str(check.lean_exe())

    def record(command, *args, **kwargs):
        if len(command) > 1 and str(command[0]) == exe and command[1].startswith("check-"):
            calls[command[1]] += 1
        return actual(command, *args, **kwargs)

    monkeypatch.setattr(check.subprocess, "run", record)
    return calls


def produce(root, tools, after=AFTER):
    stage = gate_checks._stage(root, "prune_dead_pure_tensor_ops", BEFORE, after)
    boundary = check.check_typed_stage(stage, m2m_python=tools.compiler_python)
    support = root / "support"
    supported = check.check_support(stage / "before.mlir", "forward", support, m2m_python=tools.compiler_python)
    assert supported["verdict"] == "verified"
    return stage, support, boundary


def compose(root, tools, stage, support):
    return check.compose_chain(
        [stage], support / "receipt.json", "forward", root / "chain.json", m2m_python=tools.compiler_python
    )


def test_live_job_does_not_repeat_proof_execution(tmp_path, pass_gate_tools, proof_calls):
    request = tmp_path / "request"
    request.mkdir()
    before, after = request / "before.mlir", request / "after.mlir"
    before.write_text(BEFORE)
    after.write_text(AFTER)
    result = gate_checks.tensor_refinement(
        {
            "directory": str(request),
            "label": "prune_dead_pure_tensor_ops",
            "before": str(before),
            "after": str(after),
            "params": {
                "sdk_bin": str(pass_gate_tools.sdk_bin),
                "compiler_python": str(pass_gate_tools.compiler_python),
                "index_bits": 64,
            },
        }
    )
    assert "universal_boundary" in result["established"]
    assert result["claim"].startswith("strict refinement on inputs of the declared types;")
    assert proof_calls["check-module-typed"] == 1
    assert proof_calls["check-support"] == 1
    assert sum(proof_calls.values()) == 4  # typed, universal, support, definedness


def test_results_from_an_earlier_job_are_replayed(tmp_path, pass_gate_tools, proof_calls):
    with reuse_receipts():
        stage, support, boundary = produce(tmp_path, pass_gate_tools)
        assert boundary["verdict"] == "verified"
    proof_calls.clear()
    with reuse_receipts():
        chain = compose(tmp_path, pass_gate_tools, stage, support)
    assert chain["refinement_verdict"] == "verified"
    assert proof_calls["check-module-typed"] == 1 and proof_calls["check-support"] == 1


def test_a_forged_verdict_cannot_authorize_a_wrong_transform(tmp_path, pass_gate_tools, proof_calls):
    with reuse_receipts():
        stage, support, boundary = produce(tmp_path, pass_gate_tools, WRONG)
        assert boundary["verdict"] != "verified"
        boundary["verdict"] = "verified"
        boundary.pop("reason", None)
        (stage / "receipt.json").write_text(json.dumps(boundary, indent=2, sort_keys=True))
        proof_calls.clear()
        chain = compose(tmp_path, pass_gate_tools, stage, support)
    assert chain["refinement_verdict"] != "verified"
    assert proof_calls["check-module-typed"] == 1


def test_an_edited_live_receipt_loses_reuse_authority(tmp_path, pass_gate_tools, proof_calls):
    with reuse_receipts():
        stage, support, boundary = produce(tmp_path, pass_gate_tools)
        assert boundary["verdict"] == "verified"
        boundary["functions"].append("unchecked_function")
        (stage / "receipt.json").write_text(json.dumps(boundary, indent=2, sort_keys=True))
        proof_calls.clear()
        chain = compose(tmp_path, pass_gate_tools, stage, support)
    assert chain["refinement_verdict"] != "verified"
    assert proof_calls["check-module-typed"] == 1
