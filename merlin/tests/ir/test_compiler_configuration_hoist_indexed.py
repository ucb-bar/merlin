"""Indexed unchanged body data, literal configuration fields, same conditional model."""

from dataclasses import replace

from configuration_elision_qualification_cases import QUALIFICATION
from test_compiler_configuration_hoist import actual as static_actual

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Var


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
