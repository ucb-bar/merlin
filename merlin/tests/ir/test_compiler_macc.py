"""Draft MACC checks bind actual compiler outputs without granting policy approval."""

import copy
from pathlib import Path

import pytest
from lean_bf16_f32acc_cases import bmm, line_with, matmul, operands, replace_once, run_pass

from merlin.verify import compiler_macc as gate
from merlin.verify.gate_errors import GateRejected


def files(work: Path, source: str, output: str):
    before, after = work / "before.mlir", work / "after.mlir"
    before.write_text(source)
    after.write_text(output)
    return before, after


def check(tools, work: Path, source: str, output: str):
    before, after = files(work, source, output)
    return gate.check_boundary(
        before,
        after,
        work / "proof",
        sdk_bin=tools.sdk_bin,
        compiler_python=tools.compiler_python,
    )


@pytest.mark.parametrize("fmt", (["bf16", "f16"])[:2])
def test_actual_half_output_delivers_and_freshly_replays(pass_gate_tools, tmp_path, monkeypatch, fmt):
    source = bmm(fmt, batch=1, m=2, n=3, depth=5)
    output, count = run_pass(source)
    assert count == 1

    def forbidden(*args, **kwargs):
        pytest.fail("draft conformance must not install or consume policy approval")

    monkeypatch.setattr(gate.policy_gate, "require_admission", forbidden)
    completion = check(pass_gate_tools, tmp_path, source, output)
    detail = gate.validate_completion(completion)
    assert detail["status"] == "checked_draft_contract"
    assert detail["numerical_policy_verdict"] == "abstracted" and detail["policy_reviewed"] is False
    assert detail["original_source_equivalence"] is False and detail["native_fidelity"] is False
    assert len(detail["sites"]) == 1
    assert gate.THEOREM in detail["audit"]["axioms"]
    assert detail["audit_scope"] == "exact_roots_and_all_imported_project_declaration_safety"
    assert detail["full_imported_theorem_audit"] is False
    assert detail["audit"]["project_objects"]
    assert all(set(axioms) <= gate.policy_gate.STD3 for axioms in detail["audit"]["axioms"].values())
    assert gate.deliver(completion) == output
    with pytest.raises(GateRejected, match="exact live completion"):
        gate.deliver(copy.copy(completion))
    audit_output = Path(detail["audit"]["artifacts"][0]["path"])
    audit_bytes = audit_output.read_bytes()
    audit_output.write_bytes(audit_bytes + b" ")
    with pytest.raises(GateRejected, match="changed"):
        gate.deliver(completion)
    audit_output.write_bytes(audit_bytes)
    replayed = gate.replay(
        completion.report,
        tmp_path / "replay",
        sdk_bin=pass_gate_tools.sdk_bin,
        compiler_python=pass_gate_tools.compiler_python,
    )
    assert gate.deliver(replayed) == output
    Path(detail["after"]["path"]).write_text(output + "\n")
    with pytest.raises(GateRejected, match="changed"):
        gate.deliver(completion)
    with pytest.raises(GateRejected, match="changed"):
        gate.replay(
            completion.report,
            tmp_path / "stale-replay",
            sdk_bin=pass_gate_tools.sdk_bin,
            compiler_python=pass_gate_tools.compiler_python,
        )


@pytest.mark.parametrize("case", (["wrong-add", "nonzero-init", "transposed-map", "unrelated-change"])[:2])
def test_actual_wrong_output_cannot_reach_delivery(pass_gate_tools, tmp_path, case):
    source = (
        matmul(m=2, n=2, depth=2, transposed=True)
        if case == "transposed-map"
        else bmm(init="1.000000e+00" if case == "nonzero-init" else "0.000000e+00", batch=1, m=2, n=3, depth=5)
    )
    # Construct adversarial output independently after producer refusal or
    # correct map preservation; the checker must retain every former fault.
    produced_source = bmm(batch=1, m=2, n=3, depth=5) if case == "nonzero-init" else source
    output, count = run_pass(produced_source)
    assert count == 1
    if case == "transposed-map":
        output = replace_once(output, "-> (d1, d2)>", "-> (d2, d1)>")
    if case == "wrong-add":
        add = line_with(output, '"arith.addf"')
        left, right = operands(add)
        output = replace_once(output, add, add.replace(f"({left}, {right})", f"({right}, {left})"))
    elif case == "unrelated-change":
        output = replace_once(output, "value = 0.000000e+00 : bf16", "value = 1.000000e+00 : bf16")
    with pytest.raises(GateRejected):
        check(pass_gate_tools, tmp_path, source, output)
    assert not (tmp_path / "proof/draft-boundary.json").exists()
