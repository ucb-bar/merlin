"""The real schedule consumer joins its original owner before byte emission."""

import os
from dataclasses import replace
from pathlib import Path

import pytest
from test_configuration_elimination import native as native
from test_configuration_elimination import source
from test_selected_schedule_contracts import inputs

from merlin.compile.scheduling import schedule_pipeline
from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule
from merlin.llvmlower.toolchain import compiler_python
from merlin.semantic_compiler.allocation_types import StorageBank
from merlin.verify import pass_validation
from merlin.verify.compiler_configuration import SELECTOR
from merlin.verify.speculative_gate import SpeculativeGate


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_schedule_gate_precedes_native_byte_consumer(tmp_path, native, explicit):
    _, storage, encoding, limits, options = inputs(tmp_path)
    tools = pass_validation.CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), compiler_python())
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, tools, frozenset({SELECTOR}))
    with pass_validation.validating(validation):
        compiled = compile_schedule(
            source(),
            (),
            (StorageBank("unused", "other", 8, "bytes"),),
            storage,
            encoding,
            limits,
            budgets=ScheduleBudgets(),
            **options,
            configuration_qualification_identity="independent-neutral-state-machine-v1",
            pass_validation=validation if explicit else None,
        )
        assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    assert len(compiled.kernel.body) == 12
    assert worker.verdicts[0]["detail"]["representation"] == "schedule_kernel"
    image = b"".join(compiled.program.segments)
    actual, _ = native(image, 4, False)
    expected = bytes((((31 * i + 253) % 256) * 2 + 3) % 256 for i in range(67)) + bytes([91] * 5)
    assert actual == expected


def test_in_place_producer_mutation_refuses_before_any_proof_or_emitter(tmp_path, monkeypatch):
    _, storage, encoding, limits, options = inputs(tmp_path)
    original = source()
    original = replace(original, body=list(original.body))
    tools = pass_validation.CheckerTools(Path(os.environ["MERLIN_LEAN_SDK_BIN"]), compiler_python())
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, tools, frozenset({SELECTOR}))
    real_producer = schedule_pipeline.eliminate_redundant_configuration
    ran = []

    def corrupt(kernel, *args, **kwargs):
        ran.append(True)
        kernel.body.clear()
        return real_producer(kernel, *args, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("mutated source reached the proof worker")

    monkeypatch.setattr(schedule_pipeline, "eliminate_redundant_configuration", corrupt)
    monkeypatch.setattr(worker, "submit", forbidden)
    with pytest.raises(ValueError, match="changed the original Kernel before proof"):
        with pass_validation.validating(validation):
            compile_schedule(
                original,
                (),
                (StorageBank("unused", "other", 8, "bytes"),),
                storage,
                encoding,
                limits,
                budgets=ScheduleBudgets(),
                **options,
                configuration_qualification_identity="independent-neutral-state-machine-v1",
            )
    assert ran == [True] and worker.verdicts == []
