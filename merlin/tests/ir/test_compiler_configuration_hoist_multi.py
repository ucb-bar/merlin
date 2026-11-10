"""Actual complete flat Kernel scopes share one audited focused-hoist chain."""

from dataclasses import replace

from configuration_elision_qualification_cases import QUALIFICATION
from test_compiler_configuration_hoist import call, selection

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Kernel, Loop, Var


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
