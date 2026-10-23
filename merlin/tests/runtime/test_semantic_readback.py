"""Lossless semantic framing refuses incomplete and contradictory evidence."""

import pytest

from merlin.runtime.backends.spike_model import SpikeModelError, parse_console


def console(extra):
    return extra + "\nOUT_BYTES 0 4 0 0 128 63\nDONE\n"


def test_flag_off_result_is_unchanged():
    observed = parse_console(console(""))
    assert observed["output_bytes"] == [b"\x00\x00\x80\x3f"]
    assert "semantic_readback" not in observed


def test_round_trip_layout_alias_and_initial_snapshot():
    observed = parse_console(console("PRE_BYTES 0 4 0 0 0 0\nOUT_META 0 0 0 1 1 1\nOUT_ALIAS 0 0"))
    assert observed["semantic_readback"] == dict(
        metadata=[dict(shape=[1], stride=[1], storage_offset=0, requires_grad=False)],
        aliases=[[0]],
        pre_bytes={"0": "00000000"},
    )


@pytest.mark.parametrize(
    "frame",
    [
        "OUT_META 0 0 0 1 1 1",
        "OUT_META 1 0 0 0\nOUT_ALIAS 0",
        "OUT_META 0 0 0 1 1 -1\nOUT_ALIAS 0",
        "OUT_META 0 0 0 0\nOUT_ALIAS 0 1",
        "PRE_BYTES 0 1 256\nOUT_META 0 0 0 0\nOUT_ALIAS 0",
        "PRE_BYTES 0 1 0\nPRE_BYTES 0 1 1\nOUT_META 0 0 0 0\nOUT_ALIAS 0",
        "OUT_META 0 0 0 0\nOUT_ALIAS 0 0 0",
    ],
)
def test_semantic_frames_fail_closed(frame):
    with pytest.raises(SpikeModelError):
        parse_console(console(frame))


def test_semantic_frames_after_completion_are_rejected():
    with pytest.raises(SpikeModelError):
        parse_console(console("OUT_META 0 0 0 0\nOUT_ALIAS 0") + "PRE_BYTES 0 0\n")


def test_unobservable_autograd_state_is_marked_not_fabricated():
    observed = parse_console(console("OUT_META 0 0 -1 1 1 1\nOUT_ALIAS 0"))["semantic_readback"]
    assert observed["metadata"][0]["requires_grad"] is None
    assert observed["metadata"][0]["imposed_by_harness"] == ["requires_grad"]
