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


def test_harvested_fixtures_are_linked_and_an_unlinkable_one_stays_honest():
    """Both arms on the real fixtures — but which fixture is in which arm has CHANGED, deliberately.

    This test used to assert that the dtype fixtures ARE unlinked, which made the defect the contract.
    The harvester now links with `-shared -nostdlib` before disassembling and GATES on
    `spans_reliable()`, refusing to write a fixture whose loop structure it cannot read. That matters
    because an unlinked expert teaches NOTHING loop-scoped: register_block, accumulator_resident,
    nr_is_vsetvlmax, calls_in_loop and the whole memory facet all lift as None, and `cca_compare` then
    skips those axes in silence. Measured on qd8 before the fix: 48 instructions, 0 loop spans; after:
    3 spans, and the expert finally teaches register_block=(1, vsetvlmax*4) and panel_reuse=True.

    f16 is still unlinked and stays in the honest-UNKNOWN arm, for a reason worth keeping: its ukernel
    needs `xnn_float16`, which upstream defines as EITHER native `_Float16` or a `uint16` struct
    wrapper depending on a `#if`. Choosing one in the header shim would change which instructions the
    kernel emits, i.e. fabricate the expert's instruction mix — so the harvester skips it with the
    compile error recorded rather than guessing.
    """
    for name in ("xnnpack_f32_gemm_rvv.objdump", "xnnpack_qd8_gemm_rvv.objdump"):
        stream = rvv.decode_text((_DATA / name).read_text())
        assert stream.spans_reliable() is True, f"{name} must be harvested LINKED"
        c = cca.lift_asm(stream, op="matmul", source="expert")
        assert c.envelope.calls_in_loop == 0, f"{name}: a trustworthy count, not UNKNOWN"
        assert c.compute.register_block is not None, (
            f"{name}: a linked expert must be able to teach the register block — that is the lesson")

    f16 = _DATA / "xnnpack_f16_gemm_rvv.objdump"
    if f16.is_file():
        stream = rvv.decode_text(f16.read_text())
        if not stream.spans_reliable():
            # still unlinked: the facet must be honestly UNKNOWN, never a confident 0
            assert cca.lift_asm(stream, op="matmul", source="expert").envelope.calls_in_loop is None


# The same relocated loop, but its back-edge is the COMPRESSED form. `rv64gcv` includes the C
# extension, so this is what -O2 actually emits for a tight loop -- the uncompressed `bne` above is
# the exception, not the rule.
_COMPRESSED_BACKEDGE = """\
0000000080002000 <k>:
80002000: 0207ec07 	vle32.v	v24, (a5)
80002004: b3c7dc57 	vfmacc.vf	v24, fa5, v28
80002008: f8fd     	c.bnez	a5, 0x80002000 <k>
"""

# An inner reduction loop whose back-edge is compressed, wrapped in an outer loop that spills the
# accumulator. The accumulator is resident in the REDUCTION: the spill is outside it.
#
# This is the shape that made the missing compressed forms consequential rather than cosmetic. With
# `c.bnez` unrecognized the inner span does not exist, `_fma_loop` falls back to the enclosing loop,
# the spill pair inside THAT loop is attributed to the reduction, and a resident kernel is reported as
# round-tripping its accumulator through memory every step.
_COMPRESSED_INNER_LOOP = """\
0000000080002000 <k>:
80002000: 02010427 	vs1r.v	v8, (sp)
80002004: 0207ec07 	vle32.v	v24, (a5)
80002008: b3c7dc57 	vfmacc.vf	v24, fa5, v28
8000200c: f8fd     	c.bnez	a5, 0x80002004 <k+0x4>
80002010: 02010407 	vl1re32.v	v8, (sp)
80002014: fed896e3 	bne	a7, a3, 0x80002000 <k>
"""


class TestCompressedBranchesAreBranches:
    """A compressed back-edge is still a back-edge; missing it silently mis-scopes every loop query."""

    def test_a_compressed_back_edge_produces_a_span(self):
        stream = rvv.decode_text(_COMPRESSED_BACKEDGE)
        assert stream.loop_spans() == [(0x80002000, 0x80002008)]
        assert stream.has_loop()

    def test_it_agrees_with_the_uncompressed_spelling_of_the_same_loop(self):
        # The two differ only in encoding, so no loop-scoped answer may depend on which one was used.
        compressed = rvv.decode_text(_COMPRESSED_BACKEDGE).loop_spans()
        plain = rvv.decode_text(_RELOCATED).loop_spans()
        assert len(compressed) == len(plain) == 1
        assert compressed[0][0] == plain[0][0]

    def test_the_reduction_loop_is_found_inside_an_enclosing_loop(self):
        stream = rvv.decode_text(_COMPRESSED_INNER_LOOP)
        spans = sorted(stream.loop_spans())
        assert (0x80002004, 0x8000200C) in spans, "the compressed inner back-edge is missing"
        assert (0x80002000, 0x80002014) in spans

    def test_residency_is_judged_on_the_reduction_not_the_enclosing_loop(self):
        # The payoff: a resident kernel must not be reported as spilling because the only loop the
        # decoder could see was the one that legitimately spills around the reduction.
        got = cca.lift_asm(rvv.decode_text(_COMPRESSED_INNER_LOOP), op="matmul", source="t")
        assert got.compute.accumulator_resident is True

    def test_a_register_indirect_compressed_jump_resolves_to_no_target(self):
        # `c.jr ra` is matched as a branch and then declined for having no static target, which is the
        # correct outcome -- a return is not a back-edge.
        stream = rvv.decode_text("0000000080002000 <k>:\n80002000: 8082     \tc.jr\tra\n")
        assert stream.loop_spans() == [] and stream.spans_reliable()
