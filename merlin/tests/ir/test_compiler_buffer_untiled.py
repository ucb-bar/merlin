"""Actual untiled reuse has its own terminal simulation and retained target."""

import json
from pathlib import Path

import pytest
from test_compiler_buffer_identity import chain

from merlin.verify import buffer_reuse_gate as gate
from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain


@pytest.mark.parametrize("normalized,epochs", [(False, 2), (True, 2), (True, 1)])
def test_actual_three_buffer_projection_retains_untiled_target(normalized, epochs):
    buffers = chain(normalized=normalized, epochs=epochs)[:3]
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    assert candidate.final == buffers[-1]
    assert len(json.loads(candidate.raw_chain_json)) == 3
    assert ".untiled link0" in candidate.query
    assert '"arith.minui"' not in candidate.final
    assert "full_error_or_world_equivalence" in candidate.describe()
    assert candidate.describe()["full_error_or_world_equivalence"] is False


@pytest.mark.timeout(1800)
def test_untiled_complete_gate_delivery_and_fresh_replay(tmp_path, pass_gate_tools):
    buffers = chain(normalized=False, epochs=2)[:3]
    candidate = prepare_chain(buffers, allocator=ALLOCATOR)
    lean = Path(pass_gate_tools.sdk_bin) / "lean"
    result = gate.check(candidate, lean=lean, destination=tmp_path / "proof", timeout=300)
    assert result.status == "checked", result.reason
    delivered = []
    gate.deliver(result.completion, candidate, delivered.append)
    assert delivered == [buffers[-1]]
    report = json.loads(Path(result.completion.report).read_text())
    assert report["schema"] == "merlin.buffer-reuse.checked.v3"
    assert len(report["actual_buffers"]) == 3
    assert "actual4" not in report
    assert set(report["audit"]["axioms_used"]) <= gate.STD3
    replay = gate.replay(result.completion.report, lean=lean, destination=tmp_path / "replay", timeout=300)
    assert replay.status == "checked", replay.reason


def test_actual_untiled_changed_caller_is_not_hidden_by_the_model_hint():
    buffers = list(chain(normalized=False, epochs=2)[:3])
    buffers[-1] = buffers[-1].replace("arith.addi", "arith.subi", 1)
    with pytest.raises(ValueError):
        prepare_chain(tuple(buffers), allocator=ALLOCATOR)


def test_ordinary_compiler_gates_untiled_reuse_with_identical_output(tmp_path, pass_gate_tools):
    from merlin.llvmlower import lower
    from merlin.verify import compiler_transforms as transforms
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    raw = chain(normalized=False, epochs=2)[0]
    options = transforms.Options(buffer_allocator=ALLOCATOR)
    baseline = lower.lower_model(raw, tmp_path / "baseline", targets=(), formal_transforms=options)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=1800)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({transforms.BUFFER}))
    checked = lower.lower_model(
        raw, tmp_path / "checked", targets=(), formal_transforms=options, pass_validation=validation
    )
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    summary = checked.stats["formal_transforms"]
    assert transforms.TILE not in summary["changed"]
    assert transforms.BUFFER in summary["changed"]
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["full_error_or_world_equivalence"] is False
    assert detail["native_allocator_fidelity"] is False
    report = json.loads(Path(detail["receipt"]).read_text())
    assert len(report["actual_buffers"]) == 3


def test_normalization_only_storage_bundle_still_requires_actual_proof(tmp_path, pass_gate_tools):
    from merlin.verify import compiler_transforms as transforms
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    raw = chain(normalized=False, epochs=1)[0]
    options = transforms.Options(buffer_allocator=ALLOCATOR)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=1800)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({transforms.BUFFER}))
    with pass_validation.validating(validation):
        final, summary = transforms.apply(raw, tmp_path / "transforms", options=options)
    assert summary["changed"] == ["shape_view_normalization"]
    assert transforms.BUFFER in summary["unchanged"]
    assert summary["selected_checks"] == [transforms.BUFFER]
    assert worker.verdicts[0]["verdict"] == "accepted"
    report = json.loads(Path(worker.verdicts[0]["detail"]["receipt"]).read_text())
    assert Path(report["actual_buffers"][-1]["path"]).read_text() == final
