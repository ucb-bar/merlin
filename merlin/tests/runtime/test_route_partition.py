"""The admission partition must close: every buffer lands in exactly one route, and says which.

Each test here was checked by MUTATION — the guard it covers was removed and the test observed to
fail. A check that cannot fail is this repo's most frequent defect (15 recorded instances), and the
defect this module exists to catch is itself an instance of it: a compiler reporting success while
emitting nothing.
"""

from __future__ import annotations

import pytest

from merlin.runtime.route_partition import (
    DECLINE_WITHOUT_REASON, DECLINES_AND_EMITS, ROUTE_ACCEPT, ROUTE_DECLINE, ROUTE_VIOLATION,
    SILENT_NO_WORK, route_of)


def _buffer(**over):
    cb = {"abi_version": "0.1", "target": "any",
          "tensors": {"A0": {"shape": [4, 4], "dtype": "i8", "role": "input"},
                      "Y0": {"shape": [4, 4], "dtype": "i32", "role": "output"}},
          "commands": [{"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "A0", "dst": "Y0"}}]}
    cb.update(over)
    return cb


def test_a_buffer_that_emits_work_is_accepted():
    v = route_of(_buffer())
    assert v.route == ROUTE_ACCEPT
    assert v.ok and v.violations == []


def test_an_explicit_decline_with_a_reason_is_route_d():
    v = route_of(_buffer(commands=[], declined={"op": "rmsnorm", "reason": "not lowered yet"}))
    assert v.route == ROUTE_DECLINE
    assert v.ok
    assert v.reason == "not lowered yet" and v.declined_op == "rmsnorm"


def test_an_empty_buffer_with_no_decline_is_a_violation_not_a_decline():
    """The whole point. This buffer used to be indistinguishable from a correct decline."""
    v = route_of(_buffer(commands=[]))
    assert SILENT_NO_WORK in v.kinds
    # The route must NOT be D. An unexplained empty buffer reading as a decline is exactly how a
    # dropped program grades as "declined, not a failure".
    assert v.route == ROUTE_VIOLATION
    assert not v.ok


def test_the_silent_case_names_the_output_that_nothing_computes():
    """The reproducing defect dropped a declared output; the message has to say so or a reader
    cannot tell this apart from a program that legitimately produces nothing."""
    v = route_of(_buffer(commands=[]))
    detail = dict(v.violations)[SILENT_NO_WORK]
    assert "Y0" in detail
    assert "does not identify accepted work" in detail


def test_declining_and_emitting_at_once_breaks_disjointness():
    """Measured 2026-09-07: 30 real buffers decline an opcode and then emit it anyway."""
    v = route_of(_buffer(declined={"op": "MATMUL", "reason": "unsupported operation graph"}))
    assert DECLINES_AND_EMITS in v.kinds
    assert v.route == ROUTE_VIOLATION
    detail = dict(v.violations)[DECLINES_AND_EMITS]
    assert "MATMUL" in detail and "disjoint" in detail


def test_a_decline_must_carry_a_readable_reason():
    for declined in ({"op": "MATMUL"}, {}, True, "yes"):
        v = route_of(_buffer(commands=[], declined=declined))
        assert DECLINE_WITHOUT_REASON in v.kinds, declined
        assert v.route == ROUTE_VIOLATION, declined


def test_a_violation_is_never_reported_as_a_clean_route():
    """`route` and `violations` cannot disagree — the invariant a caller relies on."""
    for cb in (_buffer(commands=[]),
               _buffer(declined={"op": "X", "reason": "r"}),
               _buffer(commands=[], declined={})):
        v = route_of(cb)
        assert (v.route == ROUTE_VIOLATION) == bool(v.violations)
        assert v.ok == (not v.violations)


def test_a_buffer_declaring_no_tensors_at_all_is_still_classified():
    """The reproducing rmsnorm case declared inputs but NO output, and other buffers spell an empty
    `tensors` as a list. Neither may crash the classifier -- an unclassifiable buffer would drop back
    into the silence this module removes."""
    v = route_of({"abi_version": "0.1", "target": "any", "tensors": [], "commands": []})
    assert v.route == ROUTE_VIOLATION and SILENT_NO_WORK in v.kinds
    assert v.outputs == []


def test_missing_keys_do_not_make_a_buffer_look_accepted():
    """A buffer with no `commands` key at all must not read as accepted."""
    v = route_of({})
    assert v.route == ROUTE_VIOLATION and SILENT_NO_WORK in v.kinds


@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_emitted_opcodes_appear_in_the_disjointness_message(n):
    cmds = [{"opcode": f"OP{i}", "operands": {}} for i in range(n)]
    v = route_of(_buffer(commands=cmds, declined={"op": "OP0", "reason": "r"}))
    detail = dict(v.violations)[DECLINES_AND_EMITS]
    assert f"{n} command(s)" in detail
    assert "OP0" in detail
