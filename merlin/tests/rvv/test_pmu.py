"""Hardware performance-counter parsing — the axis that separates "emits too many instructions"
from "stalls on each instruction". Board-free: only the parse/derive contract is tested here."""
from __future__ import annotations

from merlin.rvvgen.pmu import PmuCounts, parse


def test_parses_counts_and_derives_ipc():
    c = parse("noise\nMERLIN_PMU cycles=8211267 instructions=3554877\nmore noise")
    assert c == PmuCounts(cycles=8211267, instructions=3554877)
    assert c.ipc == round(3554877 / 8211267, 4)
    assert c.as_dict() == {"pmu_cycles": 8211267, "pmu_instructions": 3554877, "pmu_ipc": c.ipc}


def test_unavailable_counter_is_none_never_zero():
    """A counter the kernel refused reads back as -1. Reporting that as 0 would look like a kernel
    that executed no instructions — i.e. an infinitely good result. It must be absent instead."""
    assert parse("MERLIN_PMU cycles=-1 instructions=-1") is None
    assert parse("MERLIN_PMU cycles=0 instructions=0") is None      # 0 cycles is not a measurement
    assert parse("no counter line here") is None
    assert parse("") is None
    assert parse("MERLIN_PMU cycles=abc instructions=12") is None   # malformed, not silently partial


def test_ipc_is_low_when_stalled_and_high_when_dense():
    """The diagnostic direction the beam reads: same instruction count, worse IPC == stalling."""
    dense, stalled = PmuCounts(cycles=1000, instructions=500), PmuCounts(cycles=5000, instructions=500)
    assert dense.ipc > stalled.ipc
