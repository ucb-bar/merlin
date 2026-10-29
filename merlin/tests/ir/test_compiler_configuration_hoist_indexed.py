"""Indexed unchanged body data, literal configuration fields, same conditional model."""

import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import QUALIFICATION
from configuration_hoist_native_consumer import compare
from test_compiler_configuration_hoist import actual as static_actual

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Const, Var, select
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake
from merlin.verify.configuration_hoist_bridge import Unsupported


def actual(work, expression):
    chosen, original, _ = static_actual(work)
    loop = original.body[1]
    first = replace(
        loop.body[1], args=tuple((name, expression if name == "a" else value) for name, value in loop.body[1].args)
    )
    original = replace(
        original, body=(original.body[0], replace(loop, body=(loop.body[0], first, *loop.body[2:])), *original.body[2:])
    )
    adapter = proposal_adapter(chosen)
    result = hoist_loop_configuration(original, adapter)
    assert result.report["hoisted_static_calls"] == 1
    return chosen, original, result


def proposal_adapter(chosen):
    from merlin.verify.compiler_configuration_hoist_indexed import proposal_adapter as make_proposal

    return make_proposal(chosen.selected_contracts, QUALIFICATION)


@pytest.mark.parametrize("expression", [Var("i"), Var("i") * 4 + 1, select("i", 2, Const(31), Var("i") * 2)])
def test_actual_indexed_data_preserves_full_expression_tokens(tmp_path, expression):
    chosen, original, result = actual(tmp_path, expression)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    intake.revalidate(candidate)
    projection = json.loads(candidate.projection_json)
    assert candidate.domain == intake.INDEXED_DOMAIN
    assert projection["indexed_data_analysis"]
    assert projection["all_index_model_premise_retained"] is True
    assert projection["native_or_encoder_fidelity_proved"] is False
    assert projection["before"]["atoms"] == projection["after"]["atoms"]
    assert "merlin.sched.ir.expr.Var" in candidate.query


def test_out_of_range_data_is_an_explicit_domain_refusal(tmp_path):
    chosen, original, result = actual(tmp_path, Var("i") + (1 << 32))
    with pytest.raises(ValueError, match="range/alignment"):
        intake.prepare(original.text(), result.kernel.text(), selection=chosen)


def test_dynamic_configuration_field_value_never_uses_approximate_bounds(tmp_path):
    chosen, original, result = actual(tmp_path, Var("i"))
    value = json.loads(chosen.selected_contracts.software_path.read_text())
    value["scheduling"]["configuration"]["work"]["reads"][0]["value"] = {"argument": "a"}
    chosen.selected_contracts.software_path.write_text(json.dumps(value))
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

    selected = read_scheduling_contracts(
        chosen.selected_contracts.facts_path,
        chosen.selected_contracts.software_path,
        target=chosen.selected_contracts.target,
    )
    chosen = ConfigurationIRSelection(selected, QUALIFICATION)
    with pytest.raises(ValueError, match="syntactic checked constant"):
        intake.prepare(original.text(), result.kernel.text(), selection=chosen)


def test_real_indexed_hoist_proof_and_native_observations(tmp_path):
    chosen, original, result = actual(tmp_path, Var("i") * 4 + 1)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    completion = gate.check_candidate(
        candidate, lean=Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean", destination=tmp_path / "proof"
    )
    delivered = gate.deliver(completion, candidate)
    assert len(compare(original.text(), delivered, tmp_path / "native")) == 15
    report = gate.validate_completion(completion, candidate)
    statement = report["statements"]["ActualConfigurationHoist.complete_kernel"]
    assert "PartialHoist.Models" in statement
    assert report["hardware_model_approved"] is False


def test_changing_retained_index_expression_is_a_genuine_lean_refusal(tmp_path):
    chosen, original, result = actual(tmp_path, Var("i") * 4 + 1)
    loop = result.kernel.body[2]
    changed = replace(
        loop.body[0], args=tuple((name, value + 1 if name == "a" else value) for name, value in loop.body[0].args)
    )
    final = replace(
        result.kernel,
        body=(*result.kernel.body[:2], replace(loop, body=(changed, *loop.body[1:])), *result.kernel.body[3:]),
    )
    candidate = intake.prepare(original.text(), final.text(), selection=chosen)
    with pytest.raises(gate.NotEstablished):
        gate.check_candidate(
            candidate, lean=Path(os.environ["MERLIN_LEAN_SDK_BIN"]) / "lean", destination=tmp_path / "proof"
        )


@pytest.mark.parametrize(
    "expression,extent",
    [
        (Var("i") * (1 << 64), 1),
        (select("i", 1 << 64, Const(1), Var("i")), 3),
        (select("i", 99, Const(1 << 64), Var("i")), 3),
        (select("i", 99, Var("i") * (1 << 64), Var("i")), 1),
    ],
)
def test_every_c_integer_literal_is_checked_even_for_singletons_and_unreachable_arms(expression, extent):
    from merlin.verify.compiler_configuration_hoist_indexed import bounds
    from merlin.verify.compiler_configuration_kernel import tree

    with pytest.raises(ValueError, match="literal exceeds"):
        bounds(tree(expression), {"i": (0, extent - 1)})


def test_alignment_congruence_is_not_approximated_from_endpoint_only():
    from merlin.verify.compiler_configuration_hoist_indexed import bounds
    from merlin.verify.compiler_configuration_kernel import tree

    _, _, divisor, remainder = bounds(tree(Var("i") * 4 + 4), {"i": (0, 4)})
    assert divisor % 4 == remainder % 4 == 0
    _, _, divisor, remainder = bounds(tree(select("i", 2, Const(7), Var("i") * 4)), {"i": (0, 4)})
    assert divisor % 4 != 0 or remainder % 4 != 0


@pytest.mark.parametrize("explicit", [False, True])
def test_public_iv_proposal_joins_canonical_worker_and_native_consumer(tmp_path, pass_gate_tools, explicit):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    chosen, original, baseline = actual(tmp_path, Var("i") * 4 + 1)
    adapter = proposal_adapter(chosen)
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    if explicit:
        result = hoist_loop_configuration(
            original, adapter, selected_contracts=chosen.selected_contracts, pass_validation=validation
        )
    else:
        with pass_validation.validating(validation):
            result = hoist_loop_configuration(original, adapter, selected_contracts=chosen.selected_contracts)
    assert result.kernel.text() == baseline.kernel.text()
    assert len(worker.verdicts) == 1 and worker.verdicts[0]["verdict"] == "accepted"
    detail = worker.verdicts[0]["detail"]
    assert detail["domain"] == intake.INDEXED_DOMAIN
    assert detail["native_fidelity"] is False and detail["hardware_model_approved"] is False
    report = json.loads(Path(detail["receipt"]).read_text())
    assert report["domain"] == intake.INDEXED_DOMAIN
    assert report["candidate"]["all_index_model_premise_retained"] is True
    assert report["candidate"]["data_argument_admission"] == "trusted_host_index64_bounds"
    assert len(compare(original.text(), result.kernel.text(), tmp_path / "native")) == 15


def test_actual_wrong_iv_target_is_rejected_by_canonical_worker(tmp_path, pass_gate_tools):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    chosen, original, result = actual(tmp_path, Var("i") * 4 + 1)
    loop = result.kernel.body[2]
    changed = replace(
        loop.body[0], args=tuple((name, value + 1 if name == "a" else value) for name, value in loop.body[0].args)
    )
    final = replace(
        result.kernel,
        body=(*result.kernel.body[:2], replace(loop, body=(changed, *loop.body[1:])), *result.kernel.body[3:]),
    )
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(
                gate.CHECK,
                before=original.text(),
                after=final.text(),
                params=gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin),
            )
    assert worker.verdicts[0]["verdict"] == "not_established"


def test_static_profile_and_source_roster_remain_stable_after_indexed_preparation(tmp_path):
    static_dir, indexed_dir = tmp_path / "static", tmp_path / "indexed"
    static_dir.mkdir()
    indexed_dir.mkdir()
    chosen, original, result = static_actual(static_dir)
    before = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    dynamic_chosen, dynamic_source, dynamic_result = actual(indexed_dir, Var("i"))
    dynamic = intake.prepare(dynamic_source.text(), dynamic_result.kernel.text(), selection=dynamic_chosen)
    after = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    assert before == after and before.domain == intake.DOMAIN
    assert dynamic.domain == intake.INDEXED_DOMAIN
    helper = "compiler_configuration_hoist_indexed.py"
    assert not any(row["path"].endswith(helper) for row in json.loads(before.sources_json))
    assert any(row["path"].endswith(helper) for row in json.loads(dynamic.sources_json))


@pytest.mark.parametrize("source_profile", ["static", "indexed"])
def test_profile_domain_cannot_be_forged_before_native(tmp_path, monkeypatch, source_profile):
    if source_profile == "static":
        chosen, original, result = static_actual(tmp_path)
    else:
        chosen, original, result = actual(tmp_path, Var("i"))
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    other = intake.INDEXED_DOMAIN if candidate.domain == intake.DOMAIN else intake.DOMAIN
    forged = replace(candidate, domain=other)
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("forged profile reached native"))
    with pytest.raises(gate.Unavailable, match="inputs or proof sources changed"):
        gate.check_candidate(forged, lean=Path("/never-opened"), destination=tmp_path / "proof")


def test_default_selected_adapter_iv_generation_remains_identity(tmp_path):
    chosen, original, _ = actual(tmp_path, Var("i") * 4 + 1)
    default = chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    result = hoist_loop_configuration(original, default)
    assert result.kernel.text() == original.text()
    assert result.report["hoisted_static_calls"] == 0


def test_shipped_proposal_never_changes_actual_data_operands(tmp_path):
    chosen, original, _ = actual(tmp_path, Var("i") * 4 + 1)
    call = original.body[1].body[1]
    before = call.args
    projected = proposal_adapter(chosen).project(call)
    assert projected.reads == (("configuration:0", 3), ("configuration:1", 5))
    assert call.args is before and call.arg("a") == Var("i") * 4 + 1


def test_shipped_proposal_requires_original_typed_selection():
    from merlin.verify.compiler_configuration_hoist_indexed import proposal_adapter as make_proposal

    with pytest.raises(ValueError, match="original typed"):
        make_proposal({}, QUALIFICATION)


def test_shipped_proposal_keeps_every_pure_setter_argument_constant(tmp_path):
    chosen, original, _ = static_actual(tmp_path)
    setter = original.body[1].body[0]
    dynamic = replace(setter, args=tuple((name, Var("i") if name == "slot" else value) for name, value in setter.args))
    assert proposal_adapter(chosen).project(dynamic) is None


def test_shipped_proposal_refuses_bool_and_foreign_expression_data(tmp_path):
    from merlin.sched.ir.expr import Expr

    class ForeignExpr(Expr):
        pass

    chosen, original, _ = actual(tmp_path, Var("i"))
    call = original.body[1].body[1]
    for bad in (Const(True), ForeignExpr()):
        invalid = replace(call, args=tuple((name, bad if name == "a" else value) for name, value in call.args))
        assert proposal_adapter(chosen).project(invalid) is None


def test_shipped_proposal_refuses_changed_original_inputs(tmp_path):
    chosen, original, _ = actual(tmp_path, Var("i"))
    proposal = proposal_adapter(chosen)
    chosen.selected_contracts.facts_path.write_text("changed original facts")
    with pytest.raises(ValueError, match="changed"):
        proposal.project(original.body[1].body[1])


def test_shipped_proposal_does_not_invent_dynamic_configuration_values(tmp_path):
    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

    chosen, original, _ = actual(tmp_path, Var("i"))
    software = json.loads(chosen.selected_contracts.software_path.read_text())
    software["scheduling"]["configuration"]["work"]["reads"][0]["value"] = {"argument": "a"}
    chosen.selected_contracts.software_path.write_text(json.dumps(software))
    selected = read_scheduling_contracts(
        chosen.selected_contracts.facts_path,
        chosen.selected_contracts.software_path,
        target=chosen.selected_contracts.target,
    )
    current = ConfigurationIRSelection(selected, QUALIFICATION)
    assert proposal_adapter(current).project(original.body[1].body[1]) is None


def wide_actual(work, expression, location, *, indexed=True):
    from configuration_elision_qualification_cases import TARGET, declarations

    from merlin.compile.scheduling.configuration_ir import ConfigurationIRSelection
    from merlin.compile.scheduling.selected_contracts import read_scheduling_contracts

    _, original, _ = static_actual(work)
    wide = work / "wide"
    wide.mkdir()
    facts, software = declarations(bits=64)
    (wide / "facts.json").write_bytes(facts)
    (wide / "software.json").write_bytes(software)
    chosen = ConfigurationIRSelection(
        read_scheduling_contracts(wide / "facts.json", wide / "software.json", target=TARGET), QUALIFICATION
    )
    loop = original.body[1]
    first = replace(
        loop.body[1],
        args=tuple((key, Var("i") if key == "a" and indexed else value) for key, value in loop.body[1].args),
    )
    if location == "inner":
        second = replace(
            loop.body[2], args=tuple((key, expression if key == "a" else value) for key, value in loop.body[2].args)
        )
        original = replace(
            original, body=(original.body[0], replace(loop, body=(loop.body[0], first, second)), *original.body[2:])
        )
    else:
        later = replace(
            original.body[2],
            args=tuple((key, expression if key == "a" else value) for key, value in original.body[2].args),
        )
        original = replace(
            original, body=(original.body[0], replace(loop, body=(loop.body[0], first, *loop.body[2:])), later)
        )
    result = hoist_loop_configuration(original, proposal_adapter(chosen))
    assert result.report["hoisted_static_calls"] == 1
    return chosen, original, result


@pytest.mark.parametrize(
    "location,expression",
    [
        ("inner", Const(1 << 63)),
        ("outer", Const(1 << 63)),
        ("inner", select("i", 99, Const(1 << 63), Var("i"))),
    ],
)
def test_index64_profile_checks_unsigned64_const_fast_paths_and_dead_arms(tmp_path, location, expression):
    from merlin.sched.ir.expr import ExprError, render

    # The selected unsigned64 field accepts these literals; the actual consumer
    # renders every literal using signed index64, including unreachable arms.
    with pytest.raises(ExprError, match="integer literal exceeds"):
        render(expression, c=True, index_bits=64)
    chosen, original, result = wide_actual(tmp_path, expression, location)
    with pytest.raises(Unsupported, match="literal exceeds"):
        intake.prepare(original.text(), result.kernel.text(), selection=chosen)


def test_static_unsigned64_profile_keeps_its_existing_semantics_without_native_claim(tmp_path):
    from merlin.sched.ir.expr import ExprError, render

    expression = Const(1 << 63)
    chosen, original, result = wide_actual(tmp_path, expression, "outer", indexed=False)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    intake.revalidate(candidate)
    assert candidate.domain == intake.DOMAIN
    assert candidate.describe()["data_argument_admission"] == "static_constants"
    with pytest.raises(ExprError, match="integer literal exceeds"):
        render(expression, c=True, index_bits=64)
