"""Actual complete flat Kernel scopes share one audited focused-hoist chain."""

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from configuration_elision_qualification_cases import QUALIFICATION
from configuration_hoist_multi_native_consumer import compare
from test_compiler_configuration_hoist import actual as static_actual
from test_compiler_configuration_hoist import call, selection
from test_compiler_configuration_hoist_indexed import actual as indexed_actual
from test_compiler_configuration_hoist_prefix import actual as prefix_actual

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Kernel, Loop, Var
from merlin.sched.ir.kernel import check_structure
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake
from merlin.verify.configuration_hoist_bridge import Unsupported


def actual(directory, case):
    chosen = selection(directory)
    first = [call("set", a=7), call("set", a=3), call("set", slot=1, a=5)]
    second = [call("pair", a=3, b=5), call("set", slot=2, a=13)]
    work0, work1 = call("work"), call("work", a=2)
    if case == "first-only":
        second = [call("pulse", a=3), call("pulse", slot=1, a=5)]
    if case == "second-only":
        first = [call("pulse", a=3), call("pulse", slot=1, a=5)]
    if case == "iv":
        work0 = replace(
            work0, args=tuple((key, Var("i") * 4 + 1 if key == "a" else value) for key, value in work0.args)
        )
        work1 = replace(
            work1, args=tuple((key, Var("j") * 2 + 1 if key == "a" else value) for key, value in work1.args)
        )
    source = Kernel(
        "commands",
        (),
        (
            call("pulse", 2, 11),
            Loop("i", 3, (*first, work0, call("work", a=7))),
            call("pulse", 2, 17),
            call("work", a=9),
            Loop("j", 3, (*second, work1, call("work", a=4))),
            call("work", a=1),
        ),
        (("owner", "original"),),
    )
    adapter = chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    if case == "iv":
        from merlin.verify.compiler_configuration_hoist_indexed import proposal_adapter

        adapter = proposal_adapter(chosen.selected_contracts, QUALIFICATION)
    result = hoist_loop_configuration(source, adapter)
    return chosen, source, result, adapter


@pytest.mark.parametrize(
    "case,focused,moved", [("both", 2, 5), ("first-only", 1, 3), ("second-only", 1, 2), ("iv", 2, 5)]
)
def test_actual_multi_loop_stage_chain_retains_entire_kernel_and_original_models(tmp_path, case, focused, moved):
    chosen, original, result, _ = actual(tmp_path, case)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    intake.revalidate(candidate)
    projection = json.loads(candidate.projection_json)
    assert candidate.domain == intake.MULTI_DOMAIN
    assert projection["actual_final_join"] is True
    assert len(projection["stages"]) == candidate.describe()["focused_loop_count"] == focused
    assert candidate.describe()["moved_setter_count"] == moved
    assert candidate.describe()["actual_loop_count"] == 2
    assert candidate.describe()["all_index_model_premise_retained"] is True
    assert "PartialHoist.Models fields literal before.atoms run" in candidate.query
    assert candidate.roots[-1] == "ActualConfigurationHoist.complete_kernel"
    sources = json.loads(candidate.proof_sources_json)
    assert len(sources) == 11 and sources[-1]["module"] == "Merlin.GraphStateConfigurationHoistFocus"


def test_invalid_zero_trip_kernel_refuses_before_any_positive_motion(tmp_path):
    chosen, original, _, adapter = actual(tmp_path, "both")
    loop = original.body[1]
    invalid = replace(original, body=(original.body[0], replace(loop, extent=0), *original.body[2:]))
    assert check_structure(invalid)
    with pytest.raises(ValueError, match="checked loop IR"):
        hoist_loop_configuration(invalid, adapter)
    with pytest.raises(ValueError, match="exact complete typed canonical Kernel"):
        intake.prepare(invalid.text(), original.text(), selection=chosen)


def test_current_profile_rejects_nested_kernel_instead_of_treating_it_as_flat(tmp_path):
    chosen, original, _, adapter = actual(tmp_path, "both")
    first, second = original.body[1], original.body[4]
    nested = replace(original, body=(replace(first, body=(second,)),))
    result = hoist_loop_configuration(nested, adapter)
    assert result.kernel.text() != nested.text()
    with pytest.raises(Unsupported):
        intake.prepare(nested.text(), result.kernel.text(), selection=chosen)


def test_context_copy_budget_refuses_before_building_unbounded_stages(tmp_path, monkeypatch):
    from merlin.verify import compiler_configuration_hoist_multi as multi
    from merlin.verify.compiler_configuration_kernel import tree

    chosen, original, result, _ = actual(tmp_path, "both")
    calls = []

    def bounded_copy(value):
        calls.append(value)
        if len(calls) > 1:
            pytest.fail("context stage copy reached before budget refusal")
        return copy.deepcopy(value)

    monkeypatch.setattr(multi, "copy", SimpleNamespace(deepcopy=bounded_copy))
    monkeypatch.setattr(multi, "_MAX_STAGE_WORK", 0)
    with pytest.raises(gate.Unavailable, match="stage expansion resource budget"):
        multi.project(
            tree(original),
            tree(result.kernel),
            isa=chosen.selected_contracts.isa.record(),
            effects=json.loads(chosen.selected_contracts.effects_json),
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "original_profile,forged_profile",
    [
        ("multi", "static"),
        ("multi", "indexed"),
        ("multi", "prefix"),
        ("static", "multi"),
        ("indexed", "multi"),
        ("prefix", "multi"),
    ],
)
def test_full_kernel_multi_profile_identity_cannot_be_forged(tmp_path, monkeypatch, original_profile, forged_profile):
    if original_profile == "multi":
        chosen, source, result, _ = actual(tmp_path, "both")
    elif original_profile == "prefix":
        chosen, source, result, _, _, _ = prefix_actual(tmp_path, "overwritten-final")
    elif original_profile == "indexed":
        chosen, source, result = indexed_actual(tmp_path, Var("i"))
    else:
        chosen, source, result = static_actual(tmp_path)
    candidate = intake.prepare(source.text(), result.kernel.text(), selection=chosen)
    domains = {
        "static": intake.DOMAIN,
        "indexed": intake.INDEXED_DOMAIN,
        "prefix": intake.PREFIX_DOMAIN,
        "multi": intake.MULTI_DOMAIN,
    }
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("forged profile reached canonical worker"))
    with pytest.raises((ValueError, gate.Unavailable)):
        gate.check_candidate(
            replace(candidate, domain=domains[forged_profile]),
            lean=Path("/never-opened"),
            destination=tmp_path / "proof",
        )


@pytest.mark.parametrize("case,explicit", [("both", False), ("second-only", True), ("first-only", False), ("iv", True)])
def test_public_multi_loop_producer_worker_live_consumer_and_replay(tmp_path, pass_gate_tools, case, explicit):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    chosen, original, baseline, adapter = actual(tmp_path, case)
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
    assert detail["domain"] == intake.MULTI_DOMAIN
    assert detail["native_fidelity"] is False and detail["hardware_model_approved"] is False
    report = json.loads(Path(detail["receipt"]).read_text())
    assert len(report["proof_sources"]) == 11
    assert report["candidate"]["actual_loop_count"] == 2
    assert len(compare(original.text(), result.kernel.text(), tmp_path / "native")) == 27
    if case == "both":
        completion, candidate = gate.replay(
            detail["receipt"], lean=Path(pass_gate_tools.sdk_bin) / "lean", destination=tmp_path / "fresh-replay"
        )
        assert candidate.domain == intake.MULTI_DOMAIN
        assert gate.deliver(completion, candidate) == result.kernel.text()


def corrupt(result, mutation):
    body = list(result.kernel.body)
    positions = [i for i, item in enumerate(body) if type(item) is Loop]
    last = positions[-1]
    if mutation == "extent":
        body[last] = replace(body[last], extent=4)
    elif mutation == "scope":
        moved = body[last - 2 : last]
        del body[last - 2 : last]
        body[1:1] = moved
    elif mutation == "body":
        loop = body[last]
        changed = replace(
            loop.body[0], args=tuple((key, value + 1 if key == "a" else value) for key, value in loop.body[0].args)
        )
        body[last] = replace(loop, body=(changed, *loop.body[1:]))
    else:
        return replace(result.kernel, attrs=(*result.kernel.attrs, ("unrelated", "changed")))
    return replace(result.kernel, body=tuple(body))


@pytest.mark.parametrize("mutation", ["extent", "scope", "body", "frame"])
def test_actual_sibling_scope_and_untouched_frame_mutants_are_kernel_refused(tmp_path, pass_gate_tools, mutation):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    chosen, original, baseline, _ = actual(tmp_path, "both")
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(
                gate.CHECK,
                before=original.text(),
                after=corrupt(baseline, mutation).text(),
                params=gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin),
            )
    assert worker.verdicts[0]["verdict"] == "not_established"
