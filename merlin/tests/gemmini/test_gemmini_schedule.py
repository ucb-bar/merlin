"""Schedule-quality lever: activation-operand stationarity (capacity-fit panel residency).

The emitted RoCC schedule keeps the resident weight in the scratchpad already; the lever additionally
keeps the activation row-panel A[mi, 0:Kt] resident and REUSES it across the N sweep, instead of
re-moving each activation tile for every output column. Proven ONLY by a measured emitted-code delta
(MVIN count) — never a schedule-text diff — per the inert-lever lesson: a flat sweep would be an inert
lever. Bit-exact correctness of the resulting kernel is covered on the RTL oracle by
test_gemmini_mlir::test_gemmini_mlir_verilator_cert (this file guards the mechanism, cheaply).
"""
from __future__ import annotations

from merlin.runtime.backends import base as _bk
from merlin.targetgen.eval.gemmini_conformance import workload

gm = _bk.get_backend("gemmini").gemmini_codegen_mlir


def _mvin_count(cb: dict) -> int:
    text, _ = gm.emit_kernel_mlir(cb)
    tok = f"{gm.K_MVIN}, x0"          # the MVIN funct in the emitted .insn operand list
    return sum(1 for ln in text.splitlines() if tok in ln)


def _with_scratchpad_rows(rows, cb):
    """Emit `cb`'s MVIN count with the scratchpad-depth budget overridden (restores the derived one)."""
    saved = gm.SCRATCHPAD_ROWS
    try:
        gm.SCRATCHPAD_ROWS = rows
        return _mvin_count(cb)
    finally:
        gm.SCRATCHPAD_ROWS = saved


def test_panel_residency_cuts_mvins_on_a_tiled_n_sweep():
    """A wide-N tiled matmul (Kt=1, Nt=8) re-moved its single activation tile once per output column;
    panel residency loads it ONCE and reuses it — a strict MVIN reduction (the lever is not inert)."""
    cb = workload(reuse=1, epilogue=(), m=16, k=16, n=128)   # Mt=1, Kt=1, Nt=8
    legacy = _with_scratchpad_rows(None, cb)                  # fail-safe path == prior per-tile schedule
    resident = _with_scratchpad_rows(gm.SCRATCHPAD_ROWS, cb)  # derived-depth panel residency
    # weight mvins (Kt*Nt=8) are common; activations drop from Mt*Nt*Kt=8 to Mt*Kt=1.
    assert resident < legacy
    assert legacy - resident == 7


def test_square_multitile_reduction_matches_the_reuse_arithmetic():
    """64x64x64: activation mvins drop from Mt*Nt*Kt to Mt*Kt (weight mvins Kt*Nt unchanged)."""
    cb = workload(reuse=1, epilogue=(), m=64, k=64, n=64)     # Mt=Kt=Nt=4
    legacy = _with_scratchpad_rows(None, cb)
    resident = _with_scratchpad_rows(gm.SCRATCHPAD_ROWS, cb)
    Kt = Nt = Mt = 4
    assert legacy == Kt * Nt + Mt * Nt * Kt        # weight + per-column activation re-moves
    assert resident == Kt * Nt + Mt * Kt           # weight + one activation panel per row


def test_no_n_sweep_is_a_noop():
    """When N is a single tile (Nt=1) there is no cross-column reuse to win — the lever must NOT change
    the schedule (guards against a spurious 'win' that is really a different, unequal computation)."""
    cb = workload(reuse=1, epilogue=(), m=16, k=128, n=16)   # Nt=1
    assert _with_scratchpad_rows(None, cb) == _with_scratchpad_rows(gm.SCRATCHPAD_ROWS, cb)


def test_capacity_guard_falls_back_when_the_panel_would_not_fit():
    """Fail-safe: an on-chip store too small for weight+panel keeps the legacy per-tile schedule
    (capacity is derived and honored, never assumed to fit)."""
    cb = workload(reuse=1, epilogue=(), m=64, k=64, n=64)
    legacy = _with_scratchpad_rows(None, cb)
    tight = _with_scratchpad_rows(gm.DIM, cb)     # one-tile budget: weight+panel cannot fit
    assert tight == legacy
