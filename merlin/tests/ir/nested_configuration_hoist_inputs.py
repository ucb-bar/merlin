"""Actual positive counted Kernel inputs; synthetic effects grant no approval."""

from dataclasses import replace

from configuration_elision_qualification_cases import QUALIFICATION
from test_compiler_configuration_hoist import call, selection

from merlin.compile.scheduling.configuration_loops import hoist_loop_configuration
from merlin.sched.ir import Kernel, Loop, Var
from merlin.verify.compiler_configuration_hoist_indexed import proposal_adapter


def actual(work, case="inner"):
    chosen = selection(work)
    pair = call("pair", a=3, b=5)
    inner = Loop("j", 3, (pair, call("work"), call("work", a=7)))
    body = (Loop("i", 2, (inner, call("work", a=9))),)
    if case == "outer-and-inner":
        body = (Loop("i", 2, (call("set", a=3), Loop("j", 3, (call("set", slot=1, a=5), call("work"))), call("work"))),)
    elif case == "indexed":
        work_call = replace(
            call("work"),
            args=tuple(
                (key, Var("i") * 16 + Var("j") * 2 + 1 if key == "a" else value) for key, value in call("work").args
            ),
        )
        body = (Loop("i", 2, (replace(inner, body=(pair, work_call)),)),)
    elif case == "overlap":
        body = (
            Loop(
                "i",
                2,
                (
                    Loop(
                        "j",
                        3,
                        (
                            call("set", a=7),
                            call("set", a=3),
                            call("set", slot=1, a=5),
                            call("pulse", a=3),
                            call("work"),
                        ),
                    ),
                ),
            ),
        )
    elif case != "inner":
        raise ValueError("unknown nested test case")
    source = Kernel("commands", (), body, (("owner", "original"),))
    adapter = (
        proposal_adapter(chosen.selected_contracts, QUALIFICATION)
        if case == "indexed"
        else chosen.selected_contracts.configuration_adapter(QUALIFICATION)
    )
    result = hoist_loop_configuration(source, adapter)
    return chosen, source, result, adapter
