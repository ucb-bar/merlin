"""A recorded failure reason must name the error, not just the frames that led to it.

`_stderr_cause` sliced a traceback as `text[idx:idx + limit]`, which keeps the header and the stack
frames and cuts off the LAST line -- the only line carrying the exception type and message. A deep
traceback therefore produced a "cause" containing no error at all. That is how a capture failure was
recorded with a reason nobody could act on.
"""
from __future__ import annotations

from merlin.targetgen.capsule_source import _stderr_cause


def _deep_traceback(frames: int = 60, *, error: str) -> str:
    body = "".join(
        f'  File "/a/deliberately/long/path/to/module_{i}.py", line {i}, in a_function_name_{i}\n'
        f"    a_source_line_that_is_quite_long_{i}()\n"
        for i in range(frames))
    return "Traceback (most recent call last)\n" + body + error + "\n"


def test_a_deep_traceback_still_names_its_exception():
    """⚠️ REGRESSION. This is the whole point of the function and it was the part being discarded."""
    err = "ValueError: Could not guard on data-dependent expression int_oo"
    got = _stderr_cause(_deep_traceback(error=err))
    assert err in got, "the exception line must survive truncation; the frames are the expendable part"


def test_a_short_traceback_is_returned_whole():
    err = "RuntimeError: boom"
    text = "Traceback (most recent call last)\n  File \"a.py\", line 1, in f\n    g()\n" + err + "\n"
    got = _stderr_cause(text)
    assert got.endswith(err + "\n")
    assert "elided" not in got, "nothing was dropped, so nothing should claim to have been"


def test_truncation_says_it_truncated_and_how_much():
    got = _stderr_cause(_deep_traceback(error="ValueError: x"), limit=400)
    assert "elided" in got
    assert "characters of frames elided" in got, "a silent truncation is indistinguishable from a short error"


def test_the_last_traceback_wins_over_a_later_warning():
    """A python subprocess interleaves warnings with tracebacks, so the TAIL of stderr is often a
    benign warning emitted after the real error -- which is how a capture failure came to be
    attributed to an LSTM contiguity UserWarning."""
    text = (_deep_traceback(frames=3, error="ValueError: the real cause")
            + "/x/y.py:1: UserWarning: something harmless\n  warnings.warn(...)\n")
    got = _stderr_cause(text)
    assert "the real cause" in got
    assert got.startswith("--- last traceback")


def test_no_traceback_falls_back_to_the_tail_and_says_so():
    got = _stderr_cause("just some output\nand more\n")
    assert "no traceback found" in got
    assert "and more" in got


def test_empty_stderr_is_reported_as_empty_not_as_a_cause():
    assert "wrote nothing" in _stderr_cause("")
    assert "wrote nothing" in _stderr_cause("   \n  ")
