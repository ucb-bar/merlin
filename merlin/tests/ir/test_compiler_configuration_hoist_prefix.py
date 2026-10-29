"""Real constant pure prefixes, original owner, complete failure-state equality."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from configuration_elision_qualification_cases import QUALIFICATION
from configuration_hoist_prefix_native_consumer import compare
from test_compiler_configuration_hoist import actual as static_actual
from test_compiler_configuration_hoist import call, selection
from test_compiler_configuration_hoist_indexed import actual as indexed_actual

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Const, Kernel, Loop, Var, select
from merlin.verify import compiler_configuration_hoist as gate
from merlin.verify import compiler_configuration_hoist_intake as intake
from merlin.verify.configuration_hoist_bridge import Unsupported


def actual(work, case):
    chosen = selection(work)
    setups, leading, remaining = {
        "two-disjoint": ([call("set", a=3), call("set", slot=1, a=5)], [], True),
        "overwritten-final": ([call("set", a=7), call("set", a=3), call("set", slot=1, a=5)], [], True),
        "identical-repeat": ([call("set", a=3), call("set", a=3), call("set", slot=1, a=5)], [], True),
        "pair-and-disjoint": ([call("pair", a=3, b=5), call("set", slot=2, a=11)], [], True),
        "identical-retained-write": (
            [call("set", a=7), call("set", a=3), call("set", slot=1, a=5)],
            [call("pulse", a=3)],
            True,
        ),
        "prefix-only": ([call("set", a=3), call("set", slot=1, a=5)], [], False),
        "two-disjoint-iv": ([call("set", a=3), call("set", slot=1, a=5)], [], True),
    }[case]
    work0 = call("work")
    if case == "two-disjoint-iv":
        work0 = replace(
            work0, args=tuple((key, Var("i") * 4 + 1 if key == "a" else value) for key, value in work0.args)
        )
    body = [*leading, work0, call("work", a=7)] if remaining else []
    before = Kernel(
        "commands",
        (),
        (call("pulse", 2, 11), Loop("i", 3, (*setups, *body)), call("work", a=9)),
        (("owner", "original"),),
    )
    adapter = chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    if case == "two-disjoint-iv":
        from merlin.verify.compiler_configuration_hoist_indexed import proposal_adapter

        adapter = proposal_adapter(chosen.selected_contracts, QUALIFICATION)
    result = hoist_loop_configuration(before, adapter)
    assert result.report["hoisted_static_calls"] == len(setups)
    return chosen, before, result, adapter, len(setups), remaining


@pytest.mark.parametrize(
    "case",
    [
        "two-disjoint",
        "overwritten-final",
        "identical-repeat",
        "pair-and-disjoint",
        "identical-retained-write",
        "prefix-only",
        "two-disjoint-iv",
    ],
)
def test_prefix_profile_uses_actual_maximal_source_prefix_and_finite_canonical_closure(tmp_path, case):
    chosen, original, result, _, count, _ = actual(tmp_path, case)
    candidate = intake.prepare(original.text(), result.kernel.text(), selection=chosen)
    intake.revalidate(candidate)
    assert candidate.domain == intake.PREFIX_DOMAIN
    assert candidate.describe()["moved_setter_count"] == count
    assert candidate.describe()["all_index_model_premise_retained"] is True
    sources = json.loads(candidate.proof_sources_json)
    assert len(sources) == 10
    assert sources[-1]["module"] == "Merlin.GraphStateConfigurationPrefixHoist"
    assert len(json.loads(candidate.projection_json)["move"]["setups"]) == count
    assert candidate.roots[-1] == "ActualConfigurationHoist.complete_kernel"


@pytest.mark.parametrize("bad", [call("pulse", a=7), call("reset")])
def test_earlier_overwritten_value_and_clobber_never_establish_prefix_motion(tmp_path, bad):
    chosen, original, _, adapter, _, _ = actual(tmp_path, "overwritten-final")
    loop = original.body[1]
    source = replace(
        original, body=(original.body[0], replace(loop, body=(*loop.body[:3], bad, *loop.body[3:])), *original.body[2:])
    )
    assert hoist_loop_configuration(source, adapter).kernel.text() == source.text()
    after = replace(
        source,
        body=(source.body[0], *loop.body[:3], replace(source.body[1], body=source.body[1].body[3:]), *source.body[2:]),
    )
    with pytest.raises(Unsupported, match="last-write-wins"):
        intake.prepare(source.text(), after.text(), selection=chosen)


@pytest.mark.parametrize(
    "original_profile,forged_profile",
    [
        ("prefix", "static"),
        ("prefix", "indexed"),
        ("static", "prefix"),
        ("indexed", "prefix"),
    ],
)
def test_prefix_profile_identity_cannot_be_forged(tmp_path, monkeypatch, original_profile, forged_profile):
    if original_profile == "prefix":
        chosen, before, result, _, _, _ = actual(tmp_path, "two-disjoint")
    elif original_profile == "static":
        chosen, before, result = static_actual(tmp_path)
    else:
        chosen, before, result = indexed_actual(tmp_path, Var("i"))
    candidate = intake.prepare(before.text(), result.kernel.text(), selection=chosen)
    domains = {"static": intake.DOMAIN, "indexed": intake.INDEXED_DOMAIN, "prefix": intake.PREFIX_DOMAIN}
    forged = replace(candidate, domain=domains[forged_profile])
    monkeypatch.setattr(gate.runner, "prove", lambda *a, **k: pytest.fail("forged profile reached Lean"))
    with pytest.raises((ValueError, gate.Unavailable)):
        gate.check_candidate(forged, lean=Path("/never-opened"), destination=tmp_path / "proof")


@pytest.mark.parametrize(
    "case,explicit",
    [
        ("overwritten-final", False),
        ("identical-retained-write", True),
        ("prefix-only", False),
        ("two-disjoint-iv", True),
    ],
)
def test_public_prefix_producer_worker_then_native_consumer(tmp_path, pass_gate_tools, case, explicit):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import SpeculativeGate

    chosen, original, baseline, adapter, count, remaining = actual(tmp_path, case)
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
    assert detail["domain"] == intake.PREFIX_DOMAIN
    assert detail["native_fidelity"] is False and detail["hardware_model_approved"] is False
    report = json.loads(Path(detail["receipt"]).read_text())
    assert report["candidate"]["moved_setter_count"] == count
    assert report["candidate"]["all_index_model_premise_retained"] is True
    assert (
        len(compare(original.text(), result.kernel.text(), tmp_path / "native", setters=count, body_work=remaining))
        == 15
    )


def test_canonical_prefix_worker_refuses_reversed_overwriting_setters(tmp_path, pass_gate_tools):
    from merlin.verify import pass_validation
    from merlin.verify.speculative_gate import GateFailure, SpeculativeGate

    chosen, original, result, _, _, _ = actual(tmp_path, "overwritten-final")
    body = result.kernel.body
    corrupt = replace(result.kernel, body=(body[0], body[2], body[1], *body[3:]))
    worker = SpeculativeGate(tmp_path / "worker", job_timeout=300)
    validation = pass_validation.PassValidation(worker, pass_gate_tools, frozenset({gate.SELECTOR}))
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            worker.submit(
                gate.CHECK,
                before=original.text(),
                after=corrupt.text(),
                params=gate.job_params(chosen, sdk_bin=pass_gate_tools.sdk_bin),
            )
    assert worker.verdicts[0]["verdict"] == "not_established"


@pytest.mark.parametrize(
    "location,expression",
    [
        ("inner", Const(1 << 63)),
        ("outer", Const(1 << 63)),
        ("inner", select("i", 99, Const(1 << 63), Var("i"))),
    ],
)
def test_prefix_index64_profile_refuses_unsigned64_literals_in_all_actual_calls(tmp_path, location, expression):
    from test_compiler_configuration_hoist_indexed import wide_actual

    from merlin.sched.ir.expr import ExprError, render
    from merlin.verify.compiler_configuration_hoist_indexed import proposal_adapter

    with pytest.raises(ExprError, match="integer literal exceeds"):
        render(expression, c=True, index_bits=64)
    chosen, source, _ = wide_actual(tmp_path, expression, location)
    loop = source.body[1]
    source = replace(
        source,
        body=(
            source.body[0],
            replace(loop, body=(loop.body[0], call("set", slot=2, a=11), *loop.body[1:])),
            *source.body[2:],
        ),
    )
    result = hoist_loop_configuration(source, proposal_adapter(chosen.selected_contracts, QUALIFICATION))
    assert result.report["hoisted_static_calls"] == 2
    with pytest.raises(Unsupported, match="literal exceeds"):
        intake.prepare(source.text(), result.kernel.text(), selection=chosen)
