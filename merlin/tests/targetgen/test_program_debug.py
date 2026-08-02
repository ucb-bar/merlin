"""The lite debugger's REDACTION contract — the guardrail that keeps the golden un-leakable.

The debugger runs the AGENT'S kernel on the functional model, so DRAM holds only the given inputs + what
the kernel wrote — never the reference golden. The one region whose post-run contents equal the golden IFF
the kernel is correct is the declared OUTPUT region, so ``program_oracle._split_dump_regions`` must refuse
any requested DRAM window overlapping it (and cap total dumped bytes against scraping). These are pure,
target-agnostic checks — no model venv, no oracle, no target name.
"""
from __future__ import annotations

from merlin.targetgen.program_oracle import _split_dump_regions, _DEBUG_MAX_DUMP_BYTES


OUT_BASE, OUT_N = 0x1000, 256          # a stand-in output region [0x1000, 0x1100)


def _bases(regs):
    return [b for b, _ in regs]


def test_window_inside_output_is_rejected():
    allowed, rejected = _split_dump_regions([(0x1040, 64)], OUT_BASE, OUT_N)
    assert allowed == []
    assert len(rejected) == 1 and "output" in rejected[0]["reason"].lower()


def test_window_straddling_output_edges_is_rejected():
    # straddles the start boundary, and straddles the end boundary — both overlap, both refused
    allowed, rejected = _split_dump_regions([(0x0FC0, 128), (0x10C0, 128)], OUT_BASE, OUT_N)
    assert allowed == []
    assert len(rejected) == 2


def test_adjacent_non_overlapping_windows_allowed():
    # ends exactly at OUT_BASE, and starts exactly at OUT_BASE+OUT_N — touching but NOT overlapping
    before = (0x0F00, OUT_BASE - 0x0F00)          # [0x0F00, 0x1000)
    after = (OUT_BASE + OUT_N, 64)                # [0x1100, 0x1140)
    allowed, rejected = _split_dump_regions([before, after], OUT_BASE, OUT_N)
    assert rejected == []
    assert _bases(allowed) == [before[0], after[0]]


def test_input_region_below_output_is_allowed():
    allowed, rejected = _split_dump_regions([(0x0800, 256)], OUT_BASE, OUT_N)
    assert _bases(allowed) == [0x0800] and rejected == []


def test_nonpositive_length_rejected():
    allowed, rejected = _split_dump_regions([(0x0800, 0), (0x0800, -8)], OUT_BASE, OUT_N)
    assert allowed == []
    assert len(rejected) == 2 and all("non-positive" in r["reason"] for r in rejected)


def test_total_byte_cap_enforced_and_order_preserved():
    # three input windows well clear of the output; together they exceed the per-request cap, so the
    # tail is rejected while earlier ones (in order) are kept
    big = _DEBUG_MAX_DUMP_BYTES
    reqs = [(0x0100, big // 2), (0x4000, big // 2), (0x8000, big // 2)]
    allowed, rejected = _split_dump_regions(reqs, OUT_BASE, OUT_N)
    assert _bases(allowed) == [0x0100, 0x4000]          # first two fit, order preserved
    assert len(rejected) == 1 and "cap" in rejected[0]["reason"].lower()
    assert sum(n for _, n in allowed) <= _DEBUG_MAX_DUMP_BYTES


def test_empty_request_is_empty():
    assert _split_dump_regions([], OUT_BASE, OUT_N) == ([], [])
    assert _split_dump_regions(None, OUT_BASE, OUT_N) == ([], [])
