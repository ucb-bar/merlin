"""A failed capture must report what failed, not whatever its subprocess printed last.

`stderr[-1500:]` was the excerpt, and a python subprocess interleaves warnings with tracebacks -- so a
capture whose warnings arrive AFTER its traceback reported the warnings. Measured on the lstmnetvit
roster model: the failure was attributed to

    UserWarning: The tensor attributes self.net.lstm._flat_weights[0], ... are not part of a single
    contiguous chunk of memory

which is not an error at all. Three roster models fail on gemmini and this one's stated reason sent a
reader to a benign PyTorch contiguity notice instead of the actual fault.

The two failure conditions were also collapsed into one message. A non-zero return code means the
worker died; a missing meta.json after rc=0 means it exited cleanly and produced no capture. Those
license different next steps.
"""
from __future__ import annotations

from merlin.targetgen.capsule_source import _stderr_cause


def test_the_last_traceback_wins_over_trailing_warnings():
    """The exact shape of the defect: a real error followed by a wall of warnings."""
    warnings = "UserWarning: flat_weights are not part of a single contiguous chunk\n" * 60
    err = ("Traceback (most recent call last):\n"
           '  File "loader.py", line 3\n'
           "ValueError: the actual fault\n")
    got = _stderr_cause(err + warnings)
    assert "last traceback" in got
    assert "ValueError: the actual fault" in got
    assert got.index("Traceback") < got.index("UserWarning") if "UserWarning" in got else True


def test_the_latest_traceback_wins_when_there_are_several():
    first = "Traceback (most recent call last):\nRuntimeError: an earlier retry\n"
    second = "Traceback (most recent call last):\nRuntimeError: the final failure\n"
    got = _stderr_cause(first + second)
    assert "the final failure" in got
    assert "an earlier retry" not in got


def test_with_no_traceback_the_tail_is_used_and_says_so():
    """A worker can fail without a python traceback, and the excerpt must not pretend otherwise."""
    got = _stderr_cause("ld: cannot find -lfoo\n" * 40)
    assert "no traceback found" in got
    assert "cannot find -lfoo" in got


def test_silence_is_reported_as_silence():
    """An empty excerpt reads as 'no reason given'; say that the worker wrote nothing."""
    assert "wrote nothing" in _stderr_cause("")
    assert "wrote nothing" in _stderr_cause("   \n  ")


def test_the_excerpt_is_bounded():
    got = _stderr_cause("Traceback (most recent call last):\n" + ("x" * 50_000), limit=500)
    assert len(got) < 1_000


def test_the_two_failure_conditions_are_distinguished_in_the_message():
    """A worker that died and one that produced nothing are different problems."""
    from merlin.common.paths import repo_root
    src = (repo_root() / "merlin" / "python" / "merlin" / "targetgen"
           / "capsule_source.py").read_text(encoding="utf-8")
    assert "worker exited non-zero" in src
    assert "wrote no meta.json" in src
