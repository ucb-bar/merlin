"""Real current producer/complete Kernel proof and independently corrupt outputs."""

import copy
import json
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


def test_current_source_intake_requires_no_historical_git(work_path):
    chosen, original, result = actual(work_path)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    intake.revalidate(candidate)
    assert "reference" not in json.loads(candidate.request_json)
    assert candidate.describe()["entries"] == ["commands"]
    assert candidate.describe()["hardware_model_approved"] is False
    assert candidate.describe()["production_enforcement"] is False
    assert candidate.roots[-1] == "ActualConfigurationHoist.complete_kernel"


@pytest.fixture
def work_path(tmp_path):
    return tmp_path


@pytest.mark.parametrize("field", ["query", "roots", "source", "final", "domain"])
def test_forged_or_changed_candidate_never_reaches_native(work_path, monkeypatch, field):
    chosen, original, result = actual(work_path)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    changed = replace(candidate, **{field: candidate.roots[:-1] if field == "roots" else "forged"})
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("forged hoist reached native proof runner"))
    with pytest.raises((ValueError, gate.Unavailable)):
        gate.check_candidate(changed, lean=Path("/never-opened"), destination=work_path / "proof")


def test_real_current_producer_gate_delivery_and_fresh_replay(work_path):
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


@pytest.mark.parametrize("mutation", ["drop-setup", "changed-work", "loop-metadata", "kernel-frame"])
def test_actual_corrupt_output_is_rejected_by_lean(work_path, mutation):
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
    with pytest.raises(ValueError, match="tripcount|canonical"):
        intake.prepare(outside.text(), result.kernel.text(), selection=chosen)


def test_checked_delivered_kernel_feeds_actual_c_emitter_and_native_failure_states(work_path):
    from configuration_hoist_native_consumer import compare

    chosen, original, result = actual(work_path)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    completion = gate.check_candidate(
        candidate, lean=Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean", destination=work_path / "proof"
    )
    delivered = gate.deliver(completion, candidate)
    rows = compare(original.text(), delivered, work_path / "native")
    assert len(rows) == 15
    (work_path / "native-observations.json").write_text(
        json.dumps(
            {
                "scope": "finite independent C-emitter/native state machine; no hardware qualification",
                "logical_observations_match": True,
                "cases": rows,
            },
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_hoist_producer_canonical_worker_then_native_consumer(work_path, pass_gate_tools, explicit):
    from configuration_hoist_native_consumer import compare

    from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    chosen, original, _ = actual(work_path)
    adapter = chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    if explicit:
        result = hoist_loop_configuration(
            original, adapter, selected_contracts=chosen.selected_contracts, pass_validation=validation
        )
    else:
        with pass_validation.validating(validation):
            result = hoist_loop_configuration(original, adapter, selected_contracts=chosen.selected_contracts)
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["compiler_enforcement"] is True
    assert detail["native_fidelity"] is False and detail["hardware_model_approved"] is False
    assert detail["domain"] == intake.DOMAIN and detail["relation"] == intake.RELATION
    assert detail["theorems"][-1] == "ActualConfigurationHoist.complete_kernel"
    assert len(compare(original.text(), result.kernel.text(), work_path / "native")) == 15


def test_installed_hoist_worker_rejects_actual_corrupt_target(work_path, pass_gate_tools):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    chosen, original, result = actual(work_path)
    wrong = replace(result.kernel, attrs=(("owner", "mutant"),))
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(
                gate.CHECK,
                before=original.text(),
                after=wrong.text(),
                params=gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin),
            )
    assert worker.verdicts[0]["verdict"] == "not_established"
