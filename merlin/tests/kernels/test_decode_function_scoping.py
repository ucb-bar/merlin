"""Loop-scoped decoding must stay inside the function under analysis.

An object file is not one function. Besides the model's own code it carries whatever the lowering
pipeline emitted beside it, and those helpers have loops of their own. `innermost_loop()` selects by
SMALLEST span, so the moment `llvmlower.pipeline._dealloc_passes` added MLIR's
`bufferization-lower-deallocations`, the emitted `dealloc_helper` — a 3-instruction scalar
retain/free loop — won that contest against every GEMM K-loop. MEASURED on a 64x64x64 matmul:
`innermost_loop()` returned `(500, 506)` in `<dealloc_helper>` while the real K-loop `(164, 206)` in
`<forward>`, with its 4 `vfmacc.vf` and zero spills, was untouched. Nothing about the codegen had
changed; every loop-scoped count simply read the wrong body, and `_lift_envelope` additionally
reclassified the true K-loop as an OUTER loop and charged its calls to the tile epilogue.

These tests pin both halves of the fix: support-function loops are excluded from kernel loop
selection (fail-safe — an object that is only a helper still reports its structure), and a span never
reaches across a function boundary into its neighbour.
"""
from __future__ import annotations

from merlin.kernels import cca
from merlin.kernels.decode import rvv

# A vector K-loop in <forward> (span 12 bytes) beside a TIGHTER scalar loop in the compiler-emitted
# <dealloc_helper> (span 4 bytes) — the exact shape that mis-selected the innermost loop.
_TWO_FUNCS = """\
0000000000000000 <forward>:
       0: 0207ec07 	vle32.v	v24, (a5)
       4: b3c7dc57 	vfmacc.vf	v24, fa5, v28
       8: 0207ec07 	vle32.v	v24, (a5)
       c: fed896e3 	bne	a7, a3, 0x0 <forward>

0000000000000010 <dealloc_helper>:
      10: 00178793 	addi	a5, a5, 1
      14: 000980e7 	jalr	ra, 0x0(s3)
      18: fef896e3 	bne	a7, a3, 0x10 <dealloc_helper>
"""

_HELPER_ONLY = """\
0000000000000010 <dealloc_helper>:
      10: 00178793 	addi	a5, a5, 1
      14: fef896e3 	bne	a7, a3, 0x10 <dealloc_helper>
"""


def test_innermost_loop_skips_the_compiler_support_helper():
    s = rvv.decode_text(_TWO_FUNCS)
    assert sorted(s.loop_spans()) == [(0, 0xC), (0x10, 0x18)]   # both back-edges are decoded...
    assert s.kernel_loop_spans() == [(0, 0xC)]                  # ...only one is model compute
    assert s.innermost_loop() == (0, 0xC), "the K-loop, not the deallocator's tighter loop"
    assert s.count_in(s.innermost_loop(), "vfmacc.vf") == 1


def test_helper_only_object_still_reports_its_loops():
    """Fail-safe: excluding support functions must never leave a stream looking straight-line."""
    s = rvv.decode_text(_HELPER_ONLY)
    assert s.kernel_loop_spans() == [(0x10, 0x14)]
    assert s.innermost_loop() == (0x10, 0x14)


def test_span_does_not_reach_into_the_neighbouring_function():
    """A range that straddles two adjacent functions is confined to the one the span belongs to
    (anchored on its back-edge), so a loop-scoped count can never mix two bodies."""
    s = rvv.decode_text(_TWO_FUNCS)
    straddling = (8, 0x18)                      # tail of <forward> .. end of <dealloc_helper>
    addrs = [i.raw.addr for i in s.insns_in(straddling)]
    assert addrs == [0x10, 0x14, 0x18], "must not pull in <forward>'s instructions at 0x8"


def test_functions_and_in_function_scoping():
    s = rvv.decode_text(_TWO_FUNCS)
    assert s.functions() == ("forward", "dealloc_helper")
    fwd = s.in_function("forward")
    assert [i.raw.addr for i in fwd.insns] == [0, 4, 8, 0xC]
    assert fwd.loop_spans() == [(0, 0xC)]
    assert s.in_function("nosuch").insns == []


def test_envelope_does_not_charge_helper_calls_to_the_kernel():
    """`calls_in_loop` counts the OUTER loops (tile epilogue). With the helper excluded, the K-loop
    is the innermost and there is no outer loop left — so the helper's `jalr` is not billed as
    per-tile kernel overhead."""
    c = cca.lift_asm(rvv.decode_text(_TWO_FUNCS), op="matmul", source="ours")
    assert c.envelope.calls_in_loop == 0
