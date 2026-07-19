"""Tests for the runtime-escape audit (`merlin.kernels.escape_audit`).

The audit's failure mode is not crashing -- it is confidently reporting "no escapes" for a kernel it
could not actually read. Two real ways that happened during development are pinned here:

  * an unlinked object has UNRELOCATED branch displacements, so every branch appears to target itself
    and loop depth reads as 0 for a call that in truth runs per tile;
  * a host ``objdump`` that cannot disassemble RISC-V exits non-zero, which must surface as UNKNOWN.

The disassembly-parsing tests run on fixed text, so they need no toolchain and no board.
"""
from __future__ import annotations

from merlin.kernels.escape_audit import (
    EscapeReport, EscapeSite, _backedge_spans, _callee, _tokenize_with_symbols, audit,
)

# A miniature linked-ELF disassembly: `forward` with a two-deep loop nest whose inner body calls
# memrefCopy, plus a prologue malloc outside every loop, plus a libc function that also calls
# memcpy -- the library noise the audit must NOT attribute to the compute region.
_DISASM = """
0000000000010000 <forward>:
   10000: 00000513     	c.addi	a0, 0x0
   10004: 000080e7     	jal	ra, 0x21360 <malloc>
   10008: 00000513     	c.addi	a1, 0x0
   1000c: 00000513     	c.addi	a2, 0x0
   10010: 000080e7     	jal	ra, 0x10974 <memrefCopy>
   10014: 00a66063     	bltu	a2, a0, 0x1000c <forward+0xc>
   10018: 00b6e063     	bltu	a3, a1, 0x10008 <forward+0x8>
   1001c: 00008067     	c.jr	ra

0000000000020000 <__libc_helper>:
   20000: 000080e7     	jal	ra, 0x20c20 <memcpy>
   20004: 00008067     	c.jr	ra
"""


def test_callee_reads_symbol_annotation_and_ignores_intra_function_targets():
    assert _callee(["ra", "0x21360 <malloc>"]) == "malloc"
    # a branch back into the same function is annotated <sym+off>: not a call to a named symbol
    assert _callee(["a2", "a0", "0x1000c <forward+0xc>"]) is None
    assert _callee(["ra"]) is None
    assert _callee([]) is None


def test_tokenizer_separates_function_headers_from_instructions():
    toks = list(_tokenize_with_symbols(_DISASM))
    headers = [(a, s) for a, s, _, _ in toks if s is not None]
    assert headers == [(0x10000, "forward"), (0x20000, "__libc_helper")]
    insns = [(a, m) for a, s, m, _ in toks if s is None]
    assert (0x10004, "jal") in insns
    assert len(insns) == 10          # 8 in forward + 2 in the libc helper


def test_backedge_spans_are_intra_procedural_and_nest():
    insns = [(a, m, o) for a, s, m, o in _tokenize_with_symbols(_DISASM) if s is None
             and a < 0x20000]
    spans = _backedge_spans(insns)
    # two back-edges: 0x10014 -> 0x1000c (inner) and 0x10018 -> 0x10008 (outer)
    assert sorted(spans) == [(0x10008, 0x10018), (0x1000C, 0x10014)]
    depth = lambda addr: sum(1 for lo, hi in spans if lo <= addr <= hi)  # noqa: E731
    assert depth(0x10004) == 0        # prologue malloc: outside every loop
    assert depth(0x10010) == 2        # memrefCopy: inside both loops


def test_unreadable_artifacts_report_unknown_not_clean(tmp_path):
    """A file that is not an object must come back UNKNOWN -- never as an escape-free kernel."""
    bogus = tmp_path / "not_an_object.o"
    bogus.write_text("this is not ELF")
    rep = audit(bogus, bogus)
    assert rep.readable is False
    assert rep.sites is None
    assert rep.in_loop_counts() is None
    assert rep.site_counts() is None
    assert rep.max_depth() is None


def test_report_separates_in_loop_escapes_from_prologue_escapes():
    rep = EscapeReport(
        obj="x.o", elf="x", scope=("forward",), undefined=("malloc", "memrefCopy"),
        sites=(
            EscapeSite(helper="malloc", caller="forward", addr=0x10004, loop_depth=0),
            EscapeSite(helper="memrefCopy", caller="forward", addr=0x10010, loop_depth=2),
        ),
    )
    assert rep.readable is True
    assert rep.site_counts() == {"malloc": 1, "memrefCopy": 1}
    # only the per-iteration escape is a suspect; the prologue allocation is real work
    assert rep.in_loop_counts() == {"memrefCopy": 1}
    assert rep.max_depth() == 2
    assert rep.helpers == ("malloc", "memrefCopy")


def test_loopless_compute_scope_is_flagged_suspect_not_clean():
    """The K1-object trap: unrelocated branches yield zero back-edges, so a per-tile escape reads as
    depth 0. A scope with no loop at all must be flagged for re-check, never trusted as clean."""
    loopless = EscapeReport(
        obj="x.o", elf="x", scope=("forward",), undefined=("memrefCopy",),
        sites=(EscapeSite(helper="memrefCopy", caller="forward", addr=0x100, loop_depth=0),),
        loops_seen=0,
    )
    assert loopless.readable is True
    assert loopless.in_loop_counts() == {}          # what it *appears* to say...
    assert loopless.loop_structure_suspect is True  # ...and why that must not be believed

    real = EscapeReport(
        obj="x.o", elf="x", scope=("forward",), undefined=("memrefCopy",),
        sites=(EscapeSite(helper="memrefCopy", caller="forward", addr=0x100, loop_depth=0),),
        loops_seen=7,
    )
    assert real.loop_structure_suspect is False     # loops exist; depth 0 is a real finding


def test_helpers_is_none_when_symbol_table_unreadable():
    rep = EscapeReport(obj="x.o", elf="x", scope=("forward",), undefined=None, sites=())
    assert rep.helpers is None          # UNKNOWN, not "no escapes"
    assert rep.in_loop_counts() == {}   # sites WERE read and were empty -- that part is known
