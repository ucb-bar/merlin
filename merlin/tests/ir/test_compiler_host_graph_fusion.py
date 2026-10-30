"""Actual graph fusion, kernel refusals, compiler enforcement and fresh replay."""

import copy
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from host_graph_fusion_cases import contract, mutate, pair, source

from merlin.verify import compiler_host_graph_fusion as gate
from merlin.verify import host_graph_fusion_intake as intake
from merlin.verify import pass_validation
from merlin.verify.gate_errors import GateRejected, GateUnavailable, GateUnsupported
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


@pytest.fixture
def lean():
    executable = Path(os.environ.get("MERLIN_LEAN_SDK_BIN", "/UNSELECTED_SDK")) / "lean"
    if not executable.is_file():
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail("host graph kernel qualification requires MERLIN_LEAN_SDK_BIN")
        pytest.skip("host graph kernel qualification requires MERLIN_LEAN_SDK_BIN")
    return executable


@pytest.mark.parametrize(
    "context,shape,width",
    [(False, (7,), 8), (False, (2, 3), 32), (True, (3, 5), 16), (False, (1000003, 1000033), 64)],
)
def test_actual_complete_graph_and_shared_outputs(lean, tmp_path, context, shape, width):
    before, after = pair(context=context, shape=shape, width=width)
    candidate = intake.prepare(before, after)
    projection = json.loads(candidate.projection_json)
    assert projection["before"]["graph"] == projection["after"]["graph"]
    assert len(projection["fusion"]["outputs"]) == 3
    assert len(candidate.query) < 20000  # No tensor-element enumeration.
    completion = gate.check_boundary(before, after, tmp_path / "proof", lean=lean)
    assert gate.deliver(completion) == after
    report = gate.validate_completion(completion)
    assert report["audit"]["violations"] == []
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert intake.ROOT in report["statements"]
    assert report["native_fidelity"] is False
    assert report["compiler_enforcement"] is False  # This is a direct API check.


@pytest.mark.parametrize("fault", ["arithmetic", "call-input", "callee-output", "public-output", "suffix"])
def test_valid_actual_target_corruptions_fail_in_lean(lean, tmp_path, fault):
    before, after = pair(context=True)
    wrong = mutate(after, fault)
    intake.prepare(before, wrong)  # A genuine closed kernel refusal, not a parser refusal.
    with pytest.raises(GateRejected):
        gate.check_boundary(before, wrong, tmp_path / "refused", lean=lean)
    assert "is false" in (tmp_path / "refused/query.stdout").read_text()


def test_fresh_process_replay_and_no_saved_authority(lean, tmp_path):
    before, after = pair()
    completion = gate.check_boundary(before, after, tmp_path / "proof", lean=lean)
    for fabricated in (copy.copy(completion), gate.Completion(completion.identity, completion.report)):
        with pytest.raises(GateUnavailable):
            gate.deliver(fabricated)
    script = """import sys
from pathlib import Path
from merlin.verify.compiler_host_graph_fusion import replay, deliver
c = replay(sys.argv[1], Path(sys.argv[2]), lean=Path(sys.argv[3]))
assert deliver(c) == Path(sys.argv[4]).read_text()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            completion.report,
            str(tmp_path / "replay"),
            str(lean),
            str(tmp_path / "proof/final.mlir"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = Path(completion.report)
    report.write_text(report.read_text() + "\n")
    with pytest.raises(GateUnavailable, match="stale"):
        gate.deliver(completion)
    snapshot = tmp_path / "proof/final.mlir"
    snapshot.write_text(mutate(snapshot.read_text(), "arithmetic"))
    with pytest.raises(GateUnavailable, match="source changed"):
        gate.replay(completion.report, tmp_path / "tampered-replay", lean=lean)
    assert not (tmp_path / "tampered-replay").exists()


def test_changed_candidate_fields_and_missing_tool_refuse(tmp_path):
    before, after = pair()
    candidate = intake.prepare(before, after)
    for changed in (
        replace(candidate, query=candidate.query + "\n"),
        replace(candidate, final=mutate(after, "arithmetic")),
        replace(candidate, sources_json="[]"),
        replace(candidate, roots=("ActualHostFusion.accepted",)),
    ):
        with pytest.raises(ValueError, match="changed"):
            intake.ready(changed)
    with pytest.raises(GateUnavailable):
        gate.check_boundary(before, after, tmp_path / "proof", lean=tmp_path / "absent-lean")


def test_unknown_effects_and_uninitialized_public_outputs_refuse():
    raw, after = pair()
    unknown = raw.replace("%e0=", '"opaque.touch"() : () -> ()\n %e0=')
    with pytest.raises(GateUnsupported):
        intake.prepare(unknown, after)
    uninitialized = raw.replace("return %a1,%a2,%a0", "return %a1,%a2,%e0")
    with pytest.raises(GateUnsupported, match="uninitialized"):
        intake.prepare(uninitialized, after)


@pytest.mark.parametrize("fault", ["float", "poison", "initializer", "placement"])
def test_semantic_domain_exclusions_refuse(fault):
    raw, after = pair()
    if fault == "float":
        raw = (
            raw.replace("i32", "f32")
            .replace("arith.addi", "arith.addf")
            .replace("arith.muli", "arith.mulf")
            .replace("arith.subi", "arith.subf")
        )
    elif fault == "poison":
        raw = raw.replace("arith.addi %v,%w", "arith.addi %v,%w overflow<nsw>")
    elif fault == "initializer":
        raw = raw.replace("arith.addi %v,%w", "arith.addi %v,%init")
    elif fault == "placement":
        after = after.replace('"host"', '"device"')
    with pytest.raises(GateUnsupported):
        intake.prepare(raw, after)


def test_explicit_selector_and_worker_result_require_actual_claim():
    from merlin.verify.gate_worker import _require_builtin_result

    assert gate.SELECTOR in pass_validation.CHECKS
    assert gate.SELECTOR in pass_validation.DEFAULT_CHECKS
    with pytest.raises(GateRejected):
        _require_builtin_result(gate.CHECK, {"status": "checked", "compiler_enforcement": True})


def test_canonical_gate_refusal_quarantines_compiler_products(pass_gate_tools, tmp_path, monkeypatch):
    from merlin.llvmlower.lower import lower_model

    submit = pass_validation._submit_admitted

    def corrupt_boundary(name, before, after, **kwargs):
        # Inject a valid but wrong target at the compiler's actual gate boundary.
        # The canonical worker and its fresh kernel check remain unchanged.
        if name == gate.SELECTOR:
            after = mutate(after, "arithmetic")
        return submit(name, before, after, **kwargs)

    monkeypatch.setattr(pass_validation, "_submit_admitted", corrupt_boundary)
    worker = SpeculativeGate(tmp_path / "gate", interrupt=False, job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    raw = source(context=True)
    work = tmp_path / "rejected"
    with pytest.raises(GateFailure):
        lower_model(raw, work, targets=(), graph_optimizations=contract(raw, context=True), pass_validation=validation)
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "not_established"
    assert not (work / "model.ll").exists()
    assert (work / "model.ll.rejected").is_file()


@pytest.mark.parametrize("textual", [False, True])
def test_ordinary_compiler_gate_and_native_public_outputs(pass_gate_tools, tmp_path, textual):
    import numpy as np

    from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy
    from merlin.llvmlower.lower import lower_model

    raw = source(context=True)
    worker = SpeculativeGate(tmp_path / "gate", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    checked = lower_model(
        raw,
        tmp_path / "checked",
        targets=("host",),
        textual=textual,
        graph_optimizations=contract(raw, context=True),
        pass_validation=validation,
    )
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["compiler_enforcement"] is True and detail["fused_stages"] == 3
    x = np.array([-(2**31), 2**31 - 1, -1, 0, 17, 65537], dtype=np.int32).reshape(2, 3)
    y = np.array([1, -1, 7, 23, 65537, -3], dtype=np.int32).reshape(2, 3)
    a = x + y
    b = a * y
    expected = ((b + x) + (b - a), b, a)
    baseline = lower_model(raw, tmp_path / "baseline", targets=("host",), textual=textual)
    for result in (baseline, checked):
        outputs = [np.empty_like(x) for _ in range(3)]
        model = HostModel.load(
            str(result.host_so), image_policy=PrivateHostImagePolicy(result.host_so.parent.resolve())
        )
        model([(value.ctypes.data, value.shape) for value in (x, y, *outputs)])
        for actual, reference in zip(outputs, expected, strict=True):
            np.testing.assert_array_equal(actual, reference)
