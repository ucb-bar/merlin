"""P5a: `envelope.calls_in_loop` must be reliable-or-honestly-UNKNOWN, never a misleading 0.

`calls_in_loop` is loop-scoped, so it is only as trustworthy as the decoded loop structure. On an
UNRELOCATED object (an unlinked `model.o`: every branch displacement is still a zero placeholder, so
each branch resolves to its OWN address and `loop_spans()` reads EMPTY) the count silently collapses
to 0 -- a per-tile `memrefCopy` in the hot loop would read as "no calls in any loop". (MEASURED on a
whole-model K1 build: 0 back-edge spans on the object vs 6017 in the linked ELF.)

The fix carries a reliability flag on the decoder (`InsnStream.spans_reliable()`, mirroring
`escape_audit.EscapeSite.depth_reliable`) so the envelope facet reports `calls_in_loop=None` when the
loop structure cannot be trusted. `runtime_calls` (from the object's undefined symbols, NOT from loop
structure) is unaffected -- it stays the trustworthy axis that names the escape on a whole-model fork.
"""
from __future__ import annotations

from pathlib import Path

from merlin.common.paths import merlin_dir
from merlin.kernels import cca
from merlin.kernels.decode import rvv

_DATA = merlin_dir() / "tests" / "data" / "cca_asm"

# A tiny loop with a real backedge (branch target < its own address) -- a linked/relocated stream.
_RELOCATED = """\
0000000080002000 <k>:
80002000: 0207ec07 	vle32.v	v24, (a5)
80002004: b3c7dc57 	vfmacc.vf	v24, fa5, v28
80002008: fed896e3 	bne	a7, a3, 0x80002000 <k>
"""

# The SAME loop as emitted in an UNRELOCATED object: the branch displacement is a zero placeholder,
# so llvm-objdump resolves the back-edge to its own address (`<k+0x8>`).
_UNRELOCATED = """\
0000000000000000 <k>:
       0: 0207ec07 	vle32.v	v24, (a5)
       4: b3c7dc57 	vfmacc.vf	v24, fa5, v28
       8: fed896e3 	bne	a7, a3, 0x8 <k+0x8>
"""


def test_self_targeting_branch_marks_spans_unreliable():
    reloc = rvv.decode_text(_RELOCATED)
    unrel = rvv.decode_text(_UNRELOCATED)
    assert reloc.spans_reliable() is True
    assert reloc.loop_spans(), "a relocated back-edge must produce a loop span"
    assert unrel.spans_reliable() is False
    assert unrel.loop_spans() == [], "unrelocated branches self-target, so no span resolves"


def test_calls_in_loop_is_none_when_spans_unreliable():
    """The envelope facet must report UNKNOWN (None), never a confident wrong 0, when the loop
    structure is unreadable."""
    c = cca.lift_asm(rvv.decode_text(_UNRELOCATED), op="matmul", source="ours")
    assert c.envelope.calls_in_loop is None


def test_relocated_stream_gives_trustworthy_calls_in_loop():
    c = cca.lift_asm(rvv.decode_text(_RELOCATED), op="matmul", source="ours")
    assert c.envelope.calls_in_loop == 0        # a real, trustworthy count (no call in the loop)


def test_runtime_calls_survive_unreliable_spans():
    """The `runtime_calls` axis comes from the object's undefined symbols, not loop structure, so it
    stays trustworthy even when `calls_in_loop` had to be nulled -- this is the axis that routes to
    erase_self_copy on a whole-model fork."""
    c = cca.lift_asm(rvv.decode_text(_UNRELOCATED), op="matmul", source="ours",
                     undefined_symbols=["memrefCopy", "malloc"])
    assert c.envelope.calls_in_loop is None
    assert c.envelope.runtime_calls == ("malloc", "memrefCopy")   # sorted intersection of escapes


def test_linked_f32_fixture_is_reliable_but_unlinked_dtype_fixtures_are_not():
    """The real fixtures demonstrate both arms: the f32 fixture is from a LINKED ELF (relocated ->
    reliable -> trustworthy calls_in_loop=0), while the newly added single-ukernel `.o` fixtures are
    unlinked (self-targeting branches -> unreliable -> calls_in_loop honestly None)."""
    f32 = rvv.decode_text((_DATA / "xnnpack_f32_gemm_rvv.objdump").read_text())
    assert f32.spans_reliable() is True
    assert cca.lift_asm(f32, op="matmul", source="expert").envelope.calls_in_loop == 0

    for name in ("xnnpack_qd8_gemm_rvv.objdump", "xnnpack_f16_gemm_rvv.objdump"):
        stream = rvv.decode_text((_DATA / name).read_text())
        assert stream.spans_reliable() is False, f"{name} is an unlinked object"
        assert cca.lift_asm(stream, op="matmul", source="expert").envelope.calls_in_loop is None
