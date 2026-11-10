"""Actual untiled reuse has its own terminal simulation and retained target."""

import json
from pathlib import Path

import pytest
from test_compiler_buffer_identity import chain

from merlin.verify.buffer_reuse_intake import ALLOCATOR, prepare_chain


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
