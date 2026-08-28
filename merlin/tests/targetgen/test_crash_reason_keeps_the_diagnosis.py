"""A clipped crash reason must keep the HEAD, because that is where the diagnosis lives.

`_msg[-300:]` threw the head away. The muon operand-binding error is one long sentence whose useful half
is first -- "could not derive harness operands: <which of three cases>" -- and whose second half is fixed
advisory prose that is identical for every failure. Tail-only clipping therefore rendered five radiance
capsules as

    cyclotron crash: es (outputs: ['Y0'])

where "es" is the last two letters of "shapes". Those capsules read as unexplained infra failures across
dozens of verdicts while the sentence that named them was discarded every single time.
"""
from __future__ import annotations

from merlin.targetgen.capsule_runner import _clip

# the real message, verbatim in shape: diagnosis first, identical advisory tail
REAL = ("could not derive harness operands: the declared operands could not be bound to the command's "
        "shapes (outputs: ['Y0']). Command shape ['MATMUL', 'SOFTMAX'] is NOT the problem — this harness "
        "binds operands from the `tensors` declarations, not from opcode names, so any opcode is "
        "acceptable provided every operand and result is declared as {name: {shape: [...], dtype: ..., "
        "role: input|weight|output}}. If your declarations are complete, this is a TOOLING gap, not a "
        "defect in the submitted artifact.")


def test_short_messages_are_untouched():
    assert _clip("boom", 300) == "boom"


def test_the_head_survives_clipping():
    """The failing behaviour exactly: the cause must be readable, not truncated to a word fragment."""
    out = _clip(REAL, 300)
    assert len(out) <= 300 + 8            # + the elision marker
    assert out.startswith("could not derive harness operands:"), out[:60]
    assert "could not be bound" in out, "the specific one of three cases was lost"


def test_the_tail_survives_too():
    """The tail carries the operand/shape specifics on other adapters, so both ends are kept."""
    out = _clip(REAL, 300)
    assert out.rstrip().endswith("defect in the submitted artifact."), out[-60:]
    assert "[…]" in out, "no elision marker — a reader cannot tell the middle was dropped"


def test_a_tail_only_clip_would_have_failed_this():
    """Guard against a revert: the old behaviour must not satisfy the head assertion above."""
    assert not REAL[-300:].startswith("could not derive harness operands:")


def test_the_runner_uses_the_clip_not_a_bare_slice():
    import inspect

    from merlin.targetgen import capsule_runner as CR

    src = inspect.getsource(CR)
    assert "_msg[-300:]" not in src, "a bare tail slice is back in the crash reason"
    assert "_msg[-260:]" not in src, "a bare tail slice is back in the fault reason"
    assert "_clip(_msg, 300)" in src and "_clip(_msg, 260)" in src


def test_no_bare_tail_slice_of_the_message_remains_anywhere():
    """FOUR sites clipped this message tail-only, not one. The `invocation failed: {_msg[-400:]}` site is
    the one the AGENT reads -- it produced `cyclotron invocation failed: es (outputs: ['Y0'])` -- so a fix
    that missed it would leave the agent with the same unreadable feedback it had all along."""
    import inspect

    from merlin.targetgen import capsule_runner as CR

    src = inspect.getsource(CR)
    assert "_msg[-" not in src, "a bare tail slice of the exception message is back"
