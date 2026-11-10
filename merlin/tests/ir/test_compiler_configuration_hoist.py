"""Real current producer/complete Kernel proof and independently corrupt outputs."""

import copy
import os
from dataclasses import replace
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import QUALIFICATION, TARGET, declarations

from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Call, Const, Kernel, Loop
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake


def call(name, slot=0, a=0, b=0):
    return Call(name, tuple((key, Const(value)) for key, value in (("slot", slot), ("a", a), ("b", b))))


def selection(work):
    facts, software = declarations()
    facts_path, software_path = work / "facts.json", work / "software.json"
    facts_path.write_bytes(facts)
    software_path.write_bytes(software)
    return ConfigurationIRSelection(read_scheduling_contracts(facts_path, software_path, target=TARGET), QUALIFICATION)


def actual(work, extent=3):
    from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration

    chosen = selection(work)
    kernel = Kernel(
        "commands",
        (),
        (
            call("pulse", 2, 11),
            Loop("i", extent, (call("pair", a=3, b=5), call("work"), call("work", a=7))),
            call("work", a=9),
        ),
        (("owner", "original"),),
    )
    result = hoist_loop_configuration(kernel, chosen.selected_contracts.configuration_adapter(QUALIFICATION))
    return chosen, kernel, result


@pytest.fixture
def work_path(tmp_path):
    return tmp_path


def test_real_current_producer_gate_delivery_and_fresh_replay(work_path, pass_gate_tools):
    chosen, original, result = actual(work_path)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    lean = Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean"
    completion = gate.check_candidate(candidate, lean=lean, destination=work_path / "proof")
    assert gate.deliver(completion, candidate) == result.kernel.text()
    report = gate.validate_completion(completion, candidate)
    assert report["relation"] == intake.RELATION
    assert report["production_enforcement"] is False
    assert set(report["audit"]["axioms_used"]) <= {"propext", "Classical.choice", "Quot.sound"}
    with pytest.raises(gate.Unavailable):
        gate.deliver(copy.copy(completion), candidate)
    fresh, replayed = gate.replay(completion.report, lean=lean, destination=work_path / "replay")
    assert gate.deliver(fresh, replayed) == result.kernel.text()


@pytest.mark.parametrize("mutation", (["drop-setup", "changed-work", "loop-metadata", "kernel-frame"])[:1])
def test_actual_corrupt_output_is_rejected_by_lean(work_path, mutation, pass_gate_tools):
    chosen, original, result = actual(work_path)
    final = result.kernel
    if mutation == "drop-setup":
        final = replace(final, body=(final.body[0], *final.body[2:]))
    elif mutation == "changed-work":
        loop = final.body[2]
        altered = replace(loop.body[0], args=call("work", a=1).args)
        final = replace(final, body=(*final.body[:2], replace(loop, body=(altered, *loop.body[1:])), *final.body[3:]))
    elif mutation == "loop-metadata":
        final = replace(final, body=(*final.body[:2], replace(final.body[2], var="changed"), *final.body[3:]))
    else:
        final = replace(final, attrs=(("owner", "mutant"),))
    candidate = intake.prepare(original.text(), final.text(), selection=chosen)
    with pytest.raises(gate.NotEstablished):
        gate.check_candidate(
            candidate, lean=Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean", destination=work_path / "proof"
        )


@pytest.mark.parametrize("extent", [0, -1, 1_000_001])
def test_unsupported_loop_domain_is_never_positive_coverage(work_path, extent):
    chosen, original, result = actual(work_path, extent=3)
    loop = original.body[1]
    outside = replace(original, body=(original.body[0], replace(loop, extent=extent), *original.body[2:]))
    reason = "positive bounded indexed loop domain required" if extent == 1_000_001 else "tripcount|canonical"
    with pytest.raises(ValueError, match=reason):
        intake.prepare(outside.text(), result.kernel.text(), selection=chosen)
