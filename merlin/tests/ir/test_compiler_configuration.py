"""Current actual producer, complete-entry proof, live delivery and refusal controls."""

import copy
import os
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import QUALIFICATION, TARGET, cases, declarations

from merlin.compile.scheduling.configuration import eliminate_redundant_configuration
from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Call, Const, Kernel
from merlin.transforms.certificates import ir_text
from merlin.verify import compiler_configuration as gate
from merlin.verify import compiler_configuration_intake as intake


def selection(work):
    facts, software = declarations()
    facts_path, software_path = work / "facts.json", work / "software.json"
    facts_path.write_bytes(facts)
    software_path.write_bytes(software)
    return ConfigurationIRSelection(read_scheduling_contracts(facts_path, software_path, target=TARGET), QUALIFICATION)


def actual(work, name="multifield"):
    chosen = selection(work)
    source = next(case.source for case in cases() if case.name == name)
    original = chosen.parse(source)
    result = chosen.apply(original)
    result.revalidate()
    return chosen, ir_text(original), ir_text(result.module)


@pytest.fixture
def work_path(tmp_path):
    return tmp_path


def test_actual_current_instruction_boundary_checks_and_replays(work_path, pass_gate_tools):
    chosen, source, final = actual(work_path)
    candidate = intake.prepare(source, final, selection=chosen)
    lean = Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean"
    completion = gate.check_candidate(candidate, lean=lean, destination=work_path / "proof")
    assert gate.deliver(completion, candidate) == final
    report = gate.validate_completion(completion, candidate)
    assert report["relation"] == intake.RELATION
    assert report["hardware_model_approved"] is False
    assert report["independently_qualified"] is False
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    with pytest.raises(gate.Unavailable):
        gate.deliver(copy.copy(completion), candidate)
    fresh, fresh_candidate = gate.replay(completion.report, lean=lean, destination=work_path / "replay")
    assert gate.deliver(fresh, fresh_candidate) == final


def test_bad_actual_output_refuses_even_with_producer_report(work_path, pass_gate_tools):
    chosen, source, final = actual(work_path)
    final = final.replace('"configuration_fixture_isa.work"', '"configuration_fixture_isa.unknown"', 1)
    candidate = intake.prepare(source, final, selection=chosen)
    lean = Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean"
    with pytest.raises(gate.NotEstablished):
        gate.check_candidate(candidate, lean=lean, destination=work_path / "proof")


def kernel_actual(work):
    chosen = selection(work)
    pair = Call("pair", (("slot", Const(0)), ("a", Const(3)), ("b", Const(5))))
    work_call = Call("work", (("slot", Const(0)), ("a", Const(0)), ("b", Const(0))))
    original = Kernel("run", (), (pair, work_call, pair, work_call, pair), (("source", "original"),))
    result = eliminate_redundant_configuration(original, chosen.selected_contracts.configuration_adapter(QUALIFICATION))
    assert len(result.kernel.body) == 3
    return chosen, original, result.kernel
