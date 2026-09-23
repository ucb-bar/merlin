"""Configuration as state: the misconfiguration that returns wrong data, and the config that is wasted.

The failure this exists against is not exotic -- it is what one target's binding already checks by hand,
field by field: "config_ld id 0 stride != the loop's", "config_st stride does not match", "config_ex
transposes differ from the loop's (or no config_ex)". Three spellings of one question, written out per
field because the IR had no way to state it. A call that assumes a configuration nothing established, or
established differently, runs on a device that IS configured -- just not for this call. It returns wrong
data promptly and reports nothing.

The second finding is the opposite kind: a configuration replaced before any call read it issued an
instruction, cost a dispatch slot, and changed nothing. That is a saving, not a defect, and it is
reported separately so a caller never has to choose between ignoring wrong answers and ignoring free
speedups.
"""

from __future__ import annotations

import pytest

from merlin.sched.check.config import check_config
from merlin.sched.ir import Kernel, TensorArg, call, loop
from merlin.sched.ir.expr import Var
from merlin.sched.ir.kernel import check_structure
from merlin.sched.primitives import Cursor, NotApplicable, hoist_config, proof_of

pytestmark = pytest.mark.target("gemmini", "atlas", "muon")

ARG = (TensorArg("a", (8,), "i8", "read"),)


# -- the misconfiguration --------------------------------------------------------------------------


def test_a_call_that_assumes_a_different_value_than_is_live_is_caught():
    k = Kernel(
        "k",
        ARG,
        (call("config_ld", sets={"ld_stride": 32}), call("mm", assumes={"ld_stride": 64})),
    )
    report = check_config(k)
    assert not report.ok
    assert "left ld_stride=32 live" in report.problems[0]
    assert "wrong data rather than an error" in report.problems[0], "the finding does not say what it costs"


def test_a_call_that_assumes_a_configuration_nobody_set_is_caught():
    """Distinct from a mismatch, and worse: the live value is whatever ran BEFORE this kernel, so the
    kernel's result is not a property of the kernel."""
    k = Kernel("k", ARG, (call("mm", assumes={"ld_stride": 64}),))
    assert "no earlier call set ld_stride" in check_config(k).problems[0]


def test_a_matching_configuration_is_clean():
    """The control. Without it every test here passes for a check that flags every assumption."""
    k = Kernel(
        "k",
        ARG,
        (call("config_ld", sets={"ld_stride": 64}), call("mm", assumes={"ld_stride": 64})),
    )
    assert check_config(k).ok


def test_the_most_recent_setter_wins():
    """Configuration is state, so a later set replaces an earlier one rather than conflicting with it."""
    k = Kernel(
        "k",
        ARG,
        (
            call("config_ld", sets={"ld_stride": 32}),
            call("mm", assumes={"ld_stride": 32}),
            call("config_ld", sets={"ld_stride": 64}),
            call("mm", assumes={"ld_stride": 64}),
        ),
    )
    assert check_config(k).ok and not check_config(k).dead


def test_configuration_set_outside_a_loop_is_live_inside_it():
    k = Kernel(
        "k",
        ARG,
        (call("config_ld", sets={"ld_stride": 64}), loop("i", 4, call("mm", assumes={"ld_stride": 64}))),
    )
    assert check_config(k).ok


def test_keys_are_independent():
    """A set of one key must not disturb another; otherwise a target with several config registers gets
    findings that depend on the order its unrelated configurations happen to appear in."""
    k = Kernel(
        "k",
        ARG,
        (
            call("config_ld", sets={"ld_stride": 64}),
            call("config_st", sets={"st_stride": 16}),
            call("mm", assumes={"ld_stride": 64, "st_stride": 16}),
        ),
    )
    assert check_config(k).ok


# -- the wasted configuration ----------------------------------------------------------------------


def test_a_configuration_replaced_before_anyone_reads_it_is_reported_as_dead():
    k = Kernel(
        "k",
        ARG,
        (
            call("config_ld", sets={"ld_stride": 32}),
            call("config_ld", sets={"ld_stride": 64}),
            call("mm", assumes={"ld_stride": 64}),
        ),
    )
    report = check_config(k)
    assert report.ok, "a wasted configuration is not a wrong answer"
    assert "changed nothing" in report.dead[0]


def test_a_configuration_that_is_read_before_being_replaced_is_not_dead():
    """The control: without it the test above would pass for a check that calls every replacement dead."""
    k = Kernel(
        "k",
        ARG,
        (
            call("config_ld", sets={"ld_stride": 32}),
            call("mm", assumes={"ld_stride": 32}),
            call("config_ld", sets={"ld_stride": 64}),
            call("mm", assumes={"ld_stride": 64}),
        ),
    )
    assert not check_config(k).dead


def test_re_setting_a_key_to_the_same_value_is_not_reported_as_dead():
    """It is redundant, not wasted-and-different; calling it dead would flag every idempotent re-config
    a loop body legitimately re-issues."""
    k = Kernel(
        "k",
        ARG,
        (call("config_ld", sets={"ld_stride": 64}), call("config_ld", sets={"ld_stride": 64}), call("mm")),
    )
    assert not check_config(k).dead


def test_the_report_says_which_keys_it_saw():
    """No problems over zero keys is an UNCONFIGURED kernel, not a correctly configured one."""
    assert check_config(Kernel("k", ARG, (call("mm"),))).keys == ()
    k = Kernel("k", ARG, (call("c", sets={"a": 1}), call("mm", assumes={"a": 1, "b": 2})))
    assert check_config(k).keys == ("a", "b")


# -- the IR carries it, and a plain kernel is untouched ---------------------------------------------


def test_a_kernel_with_no_configuration_prints_exactly_as_before():
    """The digest is the text, so adding this vocabulary must cost an unconfigured kernel nothing."""
    assert Kernel("k", ARG, (call("mm"),)).text() == "kernel k(a: i8[8] read)\n  mm()\n"


def test_configuration_prints_in_a_stable_order():
    """Two callers passing the same configuration in different order are the same schedule, so they must
    digest the same."""
    a = Kernel("k", ARG, (call("c", sets={"x": 1, "y": 2}),))
    b = Kernel("k", ARG, (call("c", sets={"y": 2, "x": 1}),))
    assert a.digest() == b.digest()
    assert "sets x=1,y=2" in a.text()


def test_a_configuration_value_naming_an_unbound_variable_is_a_structural_error():
    k = Kernel("k", ARG, (call("c", sets={"stride": Var("i")}),))
    assert any("unbound variables ['i']" in e for e in check_structure(k))


# -- the primitive ----------------------------------------------------------------------------------


def test_hoist_config_lifts_a_loop_invariant_configuration_out():
    k = Kernel(
        "k", ARG, (loop("i", 4, call("config_ld", sets={"ld_stride": 64}), call("mm", assumes={"ld_stride": 64})),)
    )
    assert proof_of(hoist_config) == "bounded_check"
    h = hoist_config(k, Cursor(loops=("i",), index=0))
    lines = [line.strip() for line in h.text().splitlines()[1:]]
    assert lines[0].startswith("config_ld"), f"the configuration was not lifted: {lines}"
    assert lines[1].startswith("for i"), lines
    assert check_config(h).ok, "the hoist broke the configuration it was supposed to preserve"


def test_hoisting_does_not_change_what_the_loop_computes():
    k = Kernel("k", ARG, (loop("i", 4, call("config_ld", sets={"s": 64}), call("mm", assumes={"s": 64})),))
    h = hoist_config(k, Cursor(loops=("i",), index=0))
    assert h.text().count("mm()") == 1 and "for i in 0..4" in h.text()


@pytest.mark.parametrize(
    "target,expect",
    [
        (call("config_ld", sets={"s": Var("i")}), "varies with i"),
        (call("mvin", x=1, sets={"s": 64}), "does more than configure"),
        (call("mm", assumes={"s": 64}), "sets no configuration"),
    ],
)
def test_hoist_config_refuses_what_it_cannot_justify(target, expect):
    k = Kernel("k", ARG, (loop("i", 4, target, call("other")),))
    with pytest.raises(NotApplicable, match=expect):
        hoist_config(k, Cursor(loops=("i",), index=0))


def test_hoist_config_refuses_when_another_call_in_the_loop_sets_the_same_key():
    """The second setter is what is live at the end of each iteration, so moving the first one out
    changes which value later iterations see -- silently, since both spellings still type-check."""
    k = Kernel(
        "k",
        ARG,
        (
            loop(
                "i",
                4,
                call("config_a", sets={"s": 64}),
                call("config_b", sets={"s": 32}),
                call("mm", assumes={"s": 32}),
            ),
        ),
    )
    with pytest.raises(NotApplicable, match="also sets"):
        hoist_config(k, Cursor(loops=("i",), index=0))


def test_hoist_config_refuses_to_empty_a_loop():
    k = Kernel("k", ARG, (loop("i", 4, call("config_ld", sets={"s": 64})),))
    with pytest.raises(NotApplicable, match="only statement"):
        hoist_config(k, Cursor(loops=("i",), index=0))


def test_hoist_config_refuses_a_cursor_that_is_not_in_a_loop():
    k = Kernel("k", ARG, (call("config_ld", sets={"s": 64}),))
    with pytest.raises(NotApplicable, match="at the top level"):
        hoist_config(k, Cursor(index=0))


# -- a transformation must not break the configuration it carried -----------------------------------
#
# Preserving the ANNOTATIONS is not the same as preserving the CONFIGURATION, and the difference is the
# whole reason this dependence had to move into the payload. `sets`/`assumes` can survive a rewrite
# verbatim while the rewrite changes which value is live where a call reads it -- the def-use chain is
# a property of ORDER, and order is exactly what a transformation edits. A separate test asserts the
# fields survive; these assert the meaning does.


def _configured() -> Kernel:
    """A valid configured nest: set once at the top, assumed in the innermost body."""
    return Kernel(
        "k",
        ARG,
        (
            call("config_ld", sets={"ld_stride": 64}),
            loop("i", 4, loop("j", 4, call("mm", assumes={"ld_stride": 64}))),
        ),
    )


def test_the_configured_nest_this_section_uses_is_valid_to_begin_with():
    """The control for everything below: a baseline that was already misconfigured would make every
    "still valid" assertion vacuous."""
    assert check_config(_configured()).ok


def test_check_config_would_notice_if_a_rewrite_lost_the_setter():
    """The other control. Without it the assertions below would pass for a check that never fires --
    which is precisely the state `check_config` is in on a kernel carrying no annotations at all."""
    k = Kernel("k", ARG, (loop("i", 4, loop("j", 4, call("mm", assumes={"ld_stride": 64}))),))
    assert not check_config(k).ok


@pytest.mark.parametrize(
    "name",
    ["divide_loop", "unroll", "peel"],
)
def test_a_semantics_preserving_primitive_leaves_the_configuration_valid(name):
    """Each of these claims to compute what the original computed. A kernel that reads a configuration
    nothing established computes something else, so the claim includes this."""
    from merlin.sched.primitives import divide_loop, peel, unroll

    k = _configured()
    inner, outer = Cursor(loops=("i", "j")), Cursor(loops=("i",))
    out = {
        "divide_loop": lambda: divide_loop(k, outer, 2),
        "unroll": lambda: unroll(k, inner),
        "peel": lambda: peel(k, inner),
    }[name]()
    report = check_config(out)
    assert report.ok, f"{name} returned a kernel whose configuration no longer holds: {report.problems}"
    assert not report.undecidable, f"{name} made a decided configuration undecidable: {report.undecidable}"
    # A clean report is NOT enough on its own, and this is the trap: a rewrite that DROPPED the
    # assumptions also reports clean, because a call that assumes nothing cannot be misconfigured.
    # Measured -- stripping every `assumes` from this kernel leaves `check_config` perfectly happy. So
    # the count is checked too: every surviving call must still declare what it reads.
    text = out.text()
    assert text.count("needs ld_stride=64") == text.count("mm()"), (
        f"{name} dropped an assumption while leaving the call, which reads as clean rather than as a defect:\n{text}"
    )


@pytest.mark.parametrize(
    "build,prim,expect",
    [
        (
            lambda: Kernel(
                "k",
                ARG,
                (
                    loop("i", 4, call("config_ld", sets={"s": 64})),
                    loop("j", 4, call("mm", assumes={"s": 64})),
                ),
            ),
            "fuse",
            "first of the two loops",
        ),
        (
            lambda: Kernel(
                "k",
                ARG,
                (loop("i", 4, call("config_ld", sets={"s": 64}), loop("j", 4, call("mm", assumes={"s": 64}))),),
            ),
            "reorder",
            "not perfect",
        ),
    ],
    ids=["fuse across a setter loop", "reorder around an interposed config"],
)
def test_a_rewrite_that_would_move_a_call_past_its_configuration_is_refused(build, prim, expect):
    """The two shapes where interleaving statements would change which value is live at a read.

    Both are refused today, and for reasons that are not about configuration at all -- fuse wants the
    cursor on the first of the two loops, reorder wants a perfect nest. That is worth pinning precisely
    BECAUSE it is incidental: the safety is currently a consequence of those structural preconditions,
    so a later relaxation of either would silently take this with it.
    """
    from merlin.sched.primitives import fuse, reorder

    with pytest.raises(NotApplicable, match=expect):
        {
            "fuse": lambda: fuse(build(), Cursor(loops=("i",))),
            "reorder": lambda: reorder(build(), Cursor(loops=("i",))),
        }[prim]()


# -- what a single walk of a loop body could not see ------------------------------------------------


def test_a_loop_whose_tail_overwrites_what_its_head_assumes_is_caught():
    """Correct on iteration 0 and wrong on every later one, which is the hard shape.

    An earlier version walked each loop body once and justified it by claiming a second pass would
    report nothing new. That was false, and this is the case it missed: iteration 0 sees what the
    statements before the loop left, every later iteration sees what the loop's own TAIL left."""
    k = Kernel(
        "k",
        ARG,
        (
            call("config_a", sets={"stride": 64}),
            loop("i", 4, call("mm", assumes={"stride": 64}), call("config_b", sets={"stride": 32})),
        ),
    )
    report = check_config(k)
    assert not report.ok
    assert "config_b left stride=32 live" in report.problems[0]


def test_a_loop_body_that_agrees_with_its_own_tail_stays_clean():
    """The control. Without it the test above passes for a check that flags every loop."""
    k = Kernel(
        "k",
        ARG,
        (
            call("config_a", sets={"stride": 32}),
            loop("i", 4, call("mm", assumes={"stride": 32}), call("config_b", sets={"stride": 32})),
        ),
    )
    assert check_config(k).ok


def test_a_finding_inside_a_loop_is_reported_once():
    """The body is walked twice; the reader should not have to know that to read the report."""
    k = Kernel("k", ARG, (loop("i", 4, call("mm", assumes={"stride": 64})),))
    assert len(check_config(k).problems) == 1


# -- a value that outlives the loop that bound it ---------------------------------------------------


def test_a_config_set_from_a_loop_variable_is_undecidable_after_the_loop():
    """`sets s=i` leaves a different value on every iteration, and which one survives is arithmetic
    this check does not do. Comparing the rendered text `"i"` reported a MISMATCH on a correct kernel
    -- both directions of that error were measured, so the read is refused instead."""
    k = Kernel("k", ARG, (loop("i", 5, call("c", sets={"s": Var("i")})), call("mm", assumes={"s": 4})))
    report = check_config(k)
    assert report.problems == (), "a correct kernel was reported as misconfigured"
    assert len(report.undecidable) == 1
    assert "differs per iteration" in report.undecidable[0]


def test_two_loops_reusing_a_variable_name_do_not_falsely_agree():
    """The other direction: `sets s=i` in one loop and `needs s=j` in another both render as a bare
    variable, and comparing the text reported agreement between two unrelated numbers."""
    k = Kernel(
        "k",
        ARG,
        (loop("i", 4, call("c", sets={"s": Var("i")})), loop("j", 4, call("mm", assumes={"s": Var("j")}))),
    )
    report = check_config(k)
    assert report.problems == ()
    assert report.undecidable, "two unrelated loop variables were compared as equal"


def test_a_value_used_inside_the_loop_that_bound_it_is_still_decided():
    """The control for both: within one scope the variable IS the same variable, and the check must
    keep deciding there or it would refuse every loop-varying configuration."""
    k = Kernel("k", ARG, (loop("i", 4, call("c", sets={"s": Var("i")}), call("mm", assumes={"s": Var("i")})),))
    report = check_config(k)
    assert report.ok and not report.undecidable
