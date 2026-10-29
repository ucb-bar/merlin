"""Current actual producer, complete-entry proof, live delivery and refusal controls."""

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import QUALIFICATION, TARGET, call, cases, declarations, function
from configuration_elision_qualification_model import execute

from merlin.compile.scheduling.configuration import eliminate_redundant_configuration
from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts
from merlin.sched.ir import Call, Const, Kernel
from merlin.transforms.certificates import ir_text
from merlin.verify import compiler_configuration as gate
from merlin.verify import compiler_configuration_intake as intake
from merlin.verify import pass_validation
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


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


def test_current_intake_uses_selected_inputs_without_historical_git(work_path):
    chosen, source, final = actual(work_path)
    candidate = intake.prepare(source, final, selection=chosen)
    intake.revalidate(candidate)
    assert "reference" not in json.loads(candidate.request_json)
    assert candidate.describe()["entries"] == ["run"]
    assert candidate.describe()["hardware_model_approved"] is False
    assert candidate.describe()["independently_qualified"] is False


@pytest.fixture
def work_path(tmp_path):
    return tmp_path


@pytest.mark.parametrize("mutation", ["query", "roots", "domain", "final", "source"])
def test_forged_candidate_refuses_before_proof(work_path, monkeypatch, mutation):
    chosen, source, final = actual(work_path)
    candidate = intake.prepare(source, final, selection=chosen)
    changed = replace(candidate, **{mutation: candidate.roots[:-1] if mutation == "roots" else "forged"})
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("forged candidate reached proof runner"))
    with pytest.raises((ValueError, gate.Unavailable)):
        gate.check_candidate(changed, lean=Path("/never-opened"), destination=work_path / "proof")


def test_changed_selected_input_and_approval_flag_cannot_reopen(work_path):
    chosen = selection(work_path)
    request = intake.request(chosen)
    request["qualification_is_premise_only"] = False
    with pytest.raises(ValueError, match="cannot establish"):
        intake.reopen(request)
    request = intake.request(chosen)
    chosen.selected_contracts.facts_path.write_text("changed")
    with pytest.raises(ValueError, match="bytes changed"):
        intake.reopen(request)


def test_nonlive_completion_never_releases_output(work_path):
    chosen, source, final = actual(work_path)
    candidate = intake.prepare(source, final, selection=chosen)
    forged = gate.Completion(candidate.identity, str(work_path / "fake.json"))
    with pytest.raises(gate.Unavailable, match="no authority"):
        gate.deliver(forged, candidate)
    with pytest.raises(gate.Unavailable, match="no authority"):
        gate.deliver(copy.copy(forged), candidate)


def test_actual_current_instruction_boundary_checks_and_replays(work_path):
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


def test_bad_actual_output_refuses_even_with_producer_report(work_path):
    chosen, source, final = actual(work_path)
    final = final.replace('"configuration_fixture_isa.work"', '"configuration_fixture_isa.unknown"', 1)
    candidate = intake.prepare(source, final, selection=chosen)
    lean = Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean"
    with pytest.raises(gate.NotEstablished):
        gate.check_candidate(candidate, lean=lean, destination=work_path / "proof")


def test_installed_original_apply_holds_result_through_canonical_worker(work_path, pass_gate_tools):
    chosen = selection(work_path)
    original = chosen.parse(next(case.source for case in cases() if case.name == "multifield"))
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    result = chosen.apply(original, pass_validation=validation)
    result.revalidate()
    assert result.report["verification"]["removed_operations"] == 2
    assert len(worker.verdicts) == 1
    verdict = worker.verdicts[0]
    assert verdict["verdict"] == "accepted", verdict
    assert verdict["detail"]["compiler_enforcement"] is True
    assert verdict["detail"]["hardware_model_approved"] is False
    assert verdict["detail"]["relation"] == intake.RELATION


def test_installed_worker_refuses_actual_corrupt_output(work_path, pass_gate_tools):
    chosen, source, final = actual(work_path)
    final = final.replace('"configuration_fixture_isa.work"', '"configuration_fixture_isa.unknown"', 1)
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(
                gate.CHECK,
                before=source,
                after=final,
                params=gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin),
            )
    assert worker.verdicts[0]["verdict"] == "not_established"


def test_original_bare_instruction_module_is_gated_before_delivery(work_path, pass_gate_tools):
    from merlin.llvmlower.target_schedule import instruction_module

    chosen, original, _ = kernel_actual(work_path)
    original = replace(original, attrs=())
    module = instruction_module(original, chosen.selected_contracts)
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    result = chosen.apply(module, pass_validation=validation)
    assert result.report["verification"]["removed_operations"] == 2
    assert worker.verdicts[0]["verdict"] == "accepted"
    assert worker.verdicts[0]["detail"]["entries"] == ["<module>"]


def test_active_original_owner_automatically_gates_instruction_apply(work_path, pass_gate_tools):
    chosen = selection(work_path)
    module = chosen.parse(next(case.source for case in cases() if case.name == "multifield"))
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pass_validation.validating(validation):
        result = chosen.apply(module)
        result.revalidate()
        assert len(worker.verdicts) == 1
        assert worker.verdicts[0]["verdict"] == "accepted"


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_kernel_result_uses_original_owner_and_canonical_worker(work_path, pass_gate_tools, explicit):
    chosen, original, _ = kernel_actual(work_path)
    result = eliminate_redundant_configuration(original, chosen.selected_contracts.configuration_adapter(QUALIFICATION))
    worker = SpeculativeGate(work_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    if explicit:
        checked = gate.validated_kernel_result(chosen.selected_contracts, QUALIFICATION, original, result, validation)
    else:
        with pass_validation.validating(validation):
            checked = gate.validated_kernel_result(
                chosen.selected_contracts, QUALIFICATION, original, result, pass_validation.current()
            )
    assert checked is result
    assert worker.verdicts[0]["verdict"] == "accepted"
    assert worker.verdicts[0]["detail"]["representation"] == "schedule_kernel"


def test_stale_original_selection_is_unavailable_before_proof(work_path, pass_gate_tools):
    from merlin.verify.gate_errors import GateUnavailable

    chosen, before, after = actual(work_path)
    params = gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin)
    chosen.selected_contracts.facts_path.write_text("changed original facts")
    with pytest.raises(GateUnavailable, match="changed"):
        gate.check({"params": params, "before": before, "after": after, "directory": str(work_path)})


@pytest.mark.parametrize("name", ["unknown_effect", "unordered_effect"])
def test_unknown_or_nonerasable_effects_keep_the_actual_barrier(work_path, name):
    chosen, source, final = actual(work_path, name)
    candidate = intake.prepare(source, final, selection=chosen)
    projection = json.loads(candidate.projection_json)
    before = projection["before"]["entries"][0]["code"]
    after = projection["after"]["entries"][0]["code"]
    barrier_ids = {row["id"] for row in before if row["effect"] is None or not row["eraseAllowed"]}
    assert barrier_ids <= {row["id"] for row in after}


def kernel_actual(work):
    chosen = selection(work)
    pair = Call("pair", (("slot", Const(0)), ("a", Const(3)), ("b", Const(5))))
    work_call = Call("work", (("slot", Const(0)), ("a", Const(0)), ("b", Const(0))))
    original = Kernel("run", (), (pair, work_call, pair, work_call, pair), (("source", "original"),))
    result = eliminate_redundant_configuration(original, chosen.selected_contracts.configuration_adapter(QUALIFICATION))
    assert len(result.kernel.body) == 3
    return chosen, original, result.kernel


def test_current_actual_kernel_projection_preserves_full_frame(work_path):
    chosen, source, final = kernel_actual(work_path)
    candidate = intake.prepare(source.text(), final.text(), selection=chosen, representation="schedule_kernel")
    intake.revalidate(candidate)
    projected = json.loads(candidate.projection_json)
    assert len(projected["before"]["entries"][0]["code"]) == 5
    assert len(projected["after"]["entries"][0]["code"]) == 3
    assert projected["before"]["frame"] == projected["after"]["frame"]
    changed = replace(final, attrs=(("source", "mutant"),))
    mutant = intake.prepare(source.text(), changed.text(), selection=chosen, representation="schedule_kernel")
    assert mutant.query != candidate.query


def test_actual_current_kernel_is_checked_and_freshly_replayed(work_path):
    chosen, source, final = kernel_actual(work_path)
    candidate = intake.prepare(source.text(), final.text(), selection=chosen, representation="schedule_kernel")
    lean = Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean"
    completion = gate.check_candidate(candidate, lean=lean, destination=work_path / "proof")
    assert gate.deliver(completion, candidate) == final.text()
    report = gate.validate_completion(completion, candidate)
    assert report["candidate"]["representation"] == "schedule_kernel"
    fresh, replayed = gate.replay(completion.report, lean=lean, destination=work_path / "replay")
    assert gate.deliver(fresh, replayed) == final.text()


def test_dropping_an_initial_kernel_setter_is_rejected(work_path):
    chosen, source, final = kernel_actual(work_path)
    wrong = replace(final, body=final.body[1:])
    candidate = intake.prepare(source.text(), wrong.text(), selection=chosen, representation="schedule_kernel")
    with pytest.raises(gate.NotEstablished):
        gate.check_candidate(
            candidate, lean=Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean", destination=work_path / "proof"
        )


def test_external_call_and_complete_function_frame_are_retained(work_path):
    chosen = selection(work_path)
    pair = call("pair", a=3, b=5)
    source = (
        "module { func.func private @observer()\n"
        + function([pair, "func.call @observer() : () -> ()", pair, pair])
        + "\n}"
    )
    original = chosen.parse(source)
    result = chosen.apply(original)
    candidate = intake.prepare(ir_text(original), ir_text(result.module), selection=chosen)
    projection = json.loads(candidate.projection_json)
    assert projection["before"]["frame"] == projection["after"]["frame"]
    before = projection["before"]["entries"][0]["code"]
    observer = next(atom for atom in before if json.loads(atom["token"])["name"] == "func.call")
    assert observer["effect"] is None and observer["eraseAllowed"] is False
    assert observer in projection["after"]["entries"][0]["code"]


@pytest.mark.parametrize("fail_at", [None, 0, 1])
@pytest.mark.parametrize("initial_value", [-7, 0, 19])
def test_current_actual_ir_preserves_failure_terminal_state(work_path, fail_at, initial_value):
    chosen, source, final = actual(work_path)
    projection = json.loads(intake.prepare(source, final, selection=chosen).projection_json)

    def operations(program):
        rows = []
        for atom in program["entries"][0]["code"]:
            decoded = atom["decoded"]
            rows.append(
                {"instruction": decoded["instruction"], "bindings": decoded["arguments"]}
                if decoded
                else {"instruction": None, "operation": json.loads(atom["token"])["name"]}
            )
        return rows

    initial = {"configuration:0": initial_value, "configuration:1": initial_value + 1, "unrelated:9": 41}
    before = execute(operations(projection["before"]), initial, fail_observation=fail_at)
    after = execute(operations(projection["after"]), initial, fail_observation=fail_at)
    assert before == after
    assert before[0] == ("ok" if fail_at is None else "error")
