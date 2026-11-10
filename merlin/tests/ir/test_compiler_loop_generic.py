"""Bounded generic-value canonical intake, authority and actual worker controls."""

import json
from pathlib import Path

import pytest

from merlin.transforms.certificates import ir_text
from merlin.transforms.unrolling import unroll_ordered_loops
from merlin.verify import compiler_loop_generic as gate
from merlin.verify.lean_graph_generic_parser import parse_generic

SOURCE = """builtin.module {
func.func private @prefix(tensor<4xf32>)->tensor<4xf32>
func.func private @tick(index,tensor<4xf32>)
func.func @forward(%left:tensor<4xf32>,%right:tensor<4xf32>)->tensor<4xf32> {
%r=func.call @prefix(%left):(tensor<4xf32>)->tensor<4xf32>
%l=arith.constant 0:index
%u=arith.constant 5:index
%s=arith.constant 1:index
scf.for %i=%l to %u step %s {func.call @tick(%i,%right):(index,tensor<4xf32>)->() scf.yield}
func.return %r:tensor<4xf32>
}}
"""


def pair():
    module = parse_generic(SOURCE)
    before = ir_text(module)
    assert unroll_ordered_loops(module, (2,), index_bits=64)
    return before, ir_text(module)


def test_actual_fresh_kernel_live_delivery_and_replay(tmp_path, pass_gate_tools):
    value = gate.prepare(*pair())
    lean = Path(pass_gate_tools.sdk_bin) / "lean"
    from merlin.verify import compiler_transforms as formal
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    worker = SpeculativeGate(tmp_path / "worker", job_timeout=900)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.UNROLL}))
    with pass_validation.validating(selected):
        result, compiler_report = formal.apply(
            value.source, tmp_path / "compiler", options=formal.Options(unroll_factor=2)
        )
    assert result == value.final
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["domain"] == gate.DOMAIN and detail["compiler_enforcement"] is True
    assert detail["native_fidelity_proved"] is False
    assert Path(detail["delivered"]["path"]).read_bytes() == value.final.encode()
    report = json.loads(Path(detail["receipt"]).read_bytes())
    assert report["candidate"] == value.describe()
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    assert set(report["statements"]) == set(value.roots)
    assert "Value" in report["statements"][gate.ROOT]
    # A worker's saved report cannot produce a live token in this process.
    with pytest.raises(gate.Unavailable, match="no authority"):
        gate.deliver(gate.Completion(value.identity, detail["receipt"]), value)
    locators = tmp_path / "replay-inputs"
    locators.mkdir()
    rows = []
    for label, text in (("source", value.source), ("final", value.final)):
        path = locators / (label + ".mlir")
        path.write_bytes(text.encode())
        rows.append(gate.file_ref(path))
    saved = json.loads(Path(detail["receipt"]).read_bytes())
    saved.update(buffers=rows, status="rejected", statements={}, audit={})
    locator = locators / "LOCATOR.json"
    locator.write_text(gate.canonical(saved))
    other, reconstructed = gate.replay(locator, lean=lean, destination=tmp_path / "replay", timeout=300)
    assert reconstructed == value
    assert gate.deliver(other, reconstructed) == value.final
    # Retain the real proof artifacts; only copied replay-input locators mutate.
    path = Path(rows[0]["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(gate.Unavailable, match="bound configuration"):
        gate.deliver(other, reconstructed)


def test_valid_wrong_capture_is_refused_by_lean_not_reported_unavailable(tmp_path, pass_gate_tools, monkeypatch):
    from xdsl.dialects import func

    from merlin.transforms import unrolling
    from merlin.verify import compiler_transforms as formal
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    original = unrolling.unroll_ordered_loops

    def wrong_producer(module, *args, **kwargs):
        # Second bulk tick reads the distinct same-typed %left, not captured %right.
        changed = original(module, *args, **kwargs)
        forward = next(op for op in module.body.block.ops if type(op) is func.FuncOp and op.sym_name.data == "forward")
        ticks = [op for op in forward.walk() if type(op) is func.CallOp and op.callee.root_reference.data == "tick"]
        assert changed and len(ticks) == 3
        ticks[1].operands = (ticks[1].operands[0], forward.body.block.args[0])
        module.verify()
        return changed

    monkeypatch.setattr(unrolling, "unroll_ordered_loops", wrong_producer)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=900)
    selected = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.UNROLL}))
    try:
        with pytest.raises(GateFailure):
            with pass_validation.validating(selected):
                formal.apply(SOURCE, tmp_path / "compiler", options=formal.Options(unroll_factor=2))
    finally:
        worker.close()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "not_established"
    assert worker.verdicts[0]["reason"].startswith("generic loop query left generated obligations unsolved")
    proofs = list(worker.directory.glob("*/generic-loop-proof"))
    assert len(proofs) == 1
    proof = proofs[0]
    assert all(json.loads(path.read_text())["returncode"] == 0 for path in proof.glob("import-*.process.json"))
    assert json.loads((proof / "query.process.json").read_text())["returncode"] == 1
    assert ": error: unsolved goals" in (proof / "query.stdout").read_text()
    assert not (proof / "REPORT.json").exists()
    coverage = json.loads((worker.directory / "coverage.json").read_text())
    assert coverage["compiler_completed"] is False and coverage["verified_boundaries"] == 0
