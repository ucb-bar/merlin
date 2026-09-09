"""A simulator abort that names an out-of-bounds index must reach the agent as ITS bug, not as infra.

Three times now this runner has had to learn the same lesson: a fault the AGENT caused, surfaced as a
`tool_crash`, reads as an unfixable infrastructure problem and the agent repeats it every round. The
first two cases (`did not halt`, and the `*** FAILED ***` runtime trap) are already reclassified with
actionable text. This pins the third.

MEASURED 2026-09-08: one arm hit `terminate called after throwing an instance of 'std::out_of_range' /
vector::_M_range_check: __n (which is 1024) >= this->size() (which is 1024)` eight times across several
grades and never fixed it. 1024 is exactly `ACC_ROWS` for this target, i.e. a one-past-the-end
accumulator row -- but the message names neither the resource, nor the index, nor the limit, so there
was nothing to act on. No other arm triggered it, which made it read as that arm being broken rather
than as an unreadable diagnostic.

The bound is parsed STRUCTURALLY (library code may not import `re`), so a changed libstdc++ message
must degrade to the generic crash text rather than invent a bound.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.capsule_runner import _out_of_range_reason, _range_check_bounds

LIBSTDCXX = (
    "spike exited 255:\n"
    "Gemmini extension configured with:\n    dim = 16\n\n"
    "terminate called after throwing an instance of 'std::out_of_range'\n"
    "  what():  vector::_M_range_check: __n (which is 1024) >= this->size() (which is 1024)\n"
)


def test_the_bound_is_parsed_from_a_real_libstdcxx_abort():
    idx, size = _range_check_bounds(LIBSTDCXX)
    assert (idx, size) == (1024, 1024)


@pytest.mark.parametrize("msg", [
    "spike exited 255: some other failure",
    "",
    # right exception, but not the _M_range_check shape -> must NOT invent a bound
    "terminate called after throwing an instance of 'std::out_of_range'",
    # marker present but non-numeric: degrade rather than guess
    "_M_range_check: __n (which is nine) >= this->size() (which is 1024)",
    # only one bound present
    "_M_range_check: __n (which is 1024) >= this->size()",
])
def test_anything_else_degrades_instead_of_guessing(msg):
    assert _range_check_bounds(msg) == (None, None)


def test_the_reason_names_the_resource_the_index_and_the_limit():
    idx, size = _range_check_bounds(LIBSTDCXX)
    r = _out_of_range_reason("spike", idx, size)
    assert "1024" in r and "0..1023" in r, "must state the offending index and the valid range"
    assert "ACC_ROWS" in r, "must point at the capacity the package compiles against"
    assert "not a simulator fault" in r, "must not read as an infra problem the agent cannot fix"
    for word in ("out of bounds", "OUT OF BOUNDS"):
        if word.lower() in r.lower():
            break
    else:
        pytest.fail("the reason must say the access was out of bounds")


def test_it_is_actionable_where_the_old_message_was_not():
    """The regression in one assertion: the old text carried none of what a fix needs."""
    old = LIBSTDCXX
    new = _out_of_range_reason("spike", *_range_check_bounds(LIBSTDCXX))
    for token in ("accumulator", "scratchpad", "ACC_ROWS", "0..1023"):
        assert token not in old, f"precondition: the raw abort should not mention {token}"
        assert token in new, f"the replacement must mention {token}"
