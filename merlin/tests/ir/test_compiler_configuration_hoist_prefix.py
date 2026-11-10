"""Real constant pure prefixes, original owner, complete failure-state equality."""

from dataclasses import replace

from configuration_elision_qualification_cases import QUALIFICATION
from test_compiler_configuration_hoist import call, selection

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Kernel, Loop, Var


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
