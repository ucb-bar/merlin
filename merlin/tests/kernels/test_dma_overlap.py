"""dispatch.dma_overlap must answer whether movement OVERLAPPED, not whether movement EXISTS.

The field was derived as ``bool(counts["dma"])`` while documented as "bulk movement issued to
overlap with compute". Those are different questions, and the derivation could never return False
while any DMA was present -- so on a corpus whose every transfer is immediately awaited it reported
True everywhere, which is the opposite of the fact, on the axis carrying the largest available win.
"""
from __future__ import annotations

from merlin.kernels.cca import _dma_overlap


class _Insn:
    def __init__(self, *roles):
        self.roles = tuple(roles)


def _stream(*specs):
    return [_Insn(*(s if isinstance(s, tuple) else (s,))) for s in specs]


def test_issue_then_immediate_wait_is_NOT_overlap():
    """The shape that motivated the fix: DMA.LOAD followed straight by DMA.WAIT."""
    overlapped, gap = _dma_overlap(_stream("dma", "sync", "dma", "sync"))
    assert overlapped is False
    assert gap == 0


def test_work_between_issue_and_wait_IS_overlap():
    overlapped, gap = _dma_overlap(_stream("dma", "accumulate", "accumulate", "sync"))
    assert overlapped is True
    assert gap == 2


def test_one_overlapped_transfer_among_serial_ones_still_counts_as_overlap():
    """The facet asks whether movement is EVER issued to overlap; the metric carries how much."""
    overlapped, gap = _dma_overlap(
        _stream("dma", "sync", "dma", "accumulate", "sync", "dma", "sync"))
    assert overlapped is True
    assert gap == 1


def test_more_dma_and_more_sync_do_not_count_as_the_work_being_overlapped():
    """Back-to-back transfers are movement, not compute: a queue of DMAs awaiting each other is
    still a serial stream, and counting them as 'work' would manufacture overlap out of the very
    pattern the facet exists to detect."""
    overlapped, gap = _dma_overlap(_stream("dma", "dma", "sync"))
    assert overlapped is False
    assert gap == 0


# --------------------------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------------------------
def test_a_stream_with_no_dma_is_UNKNOWN_not_False():
    """False would read as a scheduling failure. There is simply no movement to overlap."""
    overlapped, gap = _dma_overlap(_stream("accumulate", "readout", "commit"))
    assert overlapped is None
    assert gap is None


def test_dma_with_no_sync_anywhere_is_UNKNOWN_not_True():
    """Either the hardware interlocks (overlap is implicit) or this decoder cannot see the waits.
    Claiming True would credit the target for an overlap nobody observed."""
    overlapped, gap = _dma_overlap(_stream("dma", "accumulate", "dma", "accumulate"))
    assert overlapped is None
    assert gap is None


def test_a_dma_never_awaited_in_this_stream_votes_on_nothing():
    """A trailing issue may be awaited in a caller, or the stream may be truncated. It must not be
    read as overlap, and it must not drag a measured result either way."""
    overlapped, gap = _dma_overlap(_stream("dma", "sync", "dma"))
    assert overlapped is False, "the one measurable transfer was strictly serial"
    assert gap == 0

    only_trailing, only_gap = _dma_overlap(_stream("sync", "dma"))
    assert only_trailing is None, "no DMA in this stream has a wait after it: nothing is measurable"
    assert only_gap is None


def test_an_empty_stream_is_UNKNOWN():
    assert _dma_overlap([]) == (None, None)


def test_instructions_with_no_roles_do_not_count_as_overlapped_work():
    """An undecoded instruction is not evidence of compute. Counting it would let a decoder's
    ignorance manufacture the overlap."""
    overlapped, gap = _dma_overlap([_Insn("dma"), _Insn(), _Insn(), _Insn("sync")])
    assert overlapped is False
    assert gap == 0
