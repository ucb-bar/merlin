"""`InsnStream.insns_in` — a per-span index, not a per-span scan of the whole stream.

`insns_in` filtered the ENTIRE instruction stream on every call, and its callers ask per span. On a
30,397-instruction int8 whole-model objdump `cca.lift_asm` spent 33.3 s of 33.9 s inside it across
5,672 calls (`_fma_loop`, `_infer_accumulator_resident`, `_infer_register_block`, `analyze_memory`),
which made a 30-fork generation stall for ~27 minutes -- the search's real cost was reading its own
output rather than building or measuring it.

Both versions timed in the same moment on the same file: 6.31 s -> 0.14 s, ~45x. On a host loaded by
the beam itself the pre-index lift reached ~55 s, which is what produced the stall; that 55 s is a
wall observation under load, not a clean measure of the code, so the ratio quoted is the same-moment
one.

The class already cached a section index for exactly this reason on the sibling path
(`_span_section`); this is the same fix on the path that dominated.

Correctness is the point, not speed: these tests assert the indexed result is IDENTICAL to the scan
it replaces, stream order included, since every loop-scoped CCA facet (residency, spills, register
block, the whole memory facet) is counted through this method.
"""
import pathlib

import pytest

from merlin.common.paths import merlin_dir
from merlin.kernels.decode import rvv


def _scan(stream, span):
    """The implementation this replaced, kept as the oracle."""
    lo, hi = span
    sect = stream._span_section(span)
    return [i for i in stream.insns
            if lo <= i.raw.addr <= hi and (not sect or i.raw.section == sect)]


def _assert_same(stream, span):
    a, b = _scan(stream, span), stream.insns_in(span)
    assert [id(x) for x in a] == [id(x) for x in b], (
        f"span {span}: indexed result differs from the scan "
        f"({len(a)} vs {len(b)} instructions)")


_FIXTURES = merlin_dir() / "tests" / "data" / "cca_asm"


@pytest.mark.parametrize("name", sorted(p.name for p in _FIXTURES.glob("*.objdump")))
def test_indexed_lookup_matches_the_scan_on_every_loop_span_of_every_fixture(name):
    """Across the real fixtures -- linked and unlinked, expert and ours -- on every span the decoder
    actually finds, plus the degenerate ones."""
    stream = rvv.decode_text((_FIXTURES / name).read_text())
    spans = stream.loop_spans()
    for sp in spans:
        _assert_same(stream, sp)
    for sp in [(0, 0), (0, 1 << 40), (1 << 40, (1 << 40) + 8), (7, 3)]:
        _assert_same(stream, sp)          # empty, everything, out-of-range, inverted


def test_a_span_never_swallows_a_neighbouring_function():
    """The property the section anchoring exists for, and the one an index could most easily break:
    two functions whose address ranges are adjacent must not bleed into each other."""
    text = ("0000000000000000 <alpha>:\n"
            "   0:\t02b7f0d7          \tvfmacc.vv\tv1, v2, v3\n"
            "   4:\t02b7f0d7          \tvfmacc.vv\tv4, v5, v6\n"
            "\n"
            "0000000000000008 <beta>:\n"
            "   8:\t02b7f0d7          \tvfmul.vv\tv7, v8, v9\n"
            "   c:\t02b7f0d7          \tvfmul.vv\tv10, v11, v12\n")
    stream = rvv.decode_text(text)
    got = {i.raw.mnemonic for i in stream.insns_in((0, 4))}
    assert got == {"vfmacc.vv"}, f"span (0,4) reached into beta: {got}"
    _assert_same(stream, (0, 4))
    _assert_same(stream, (0, 12))         # spans both -> whatever the scan said, identically


def test_an_out_of_order_stream_falls_back_and_still_matches():
    """Bisection is only valid where addresses ascend. A stream assembled out of order must take the
    linear path for that section rather than silently returning a wrong slice."""
    text = ("0000000000000000 <alpha>:\n"
            "  10:\t02b7f0d7          \tvfmacc.vv\tv1, v2, v3\n"
            "   4:\t02b7f0d7          \tvfmul.vv\tv4, v5, v6\n"
            "   8:\t02b7f0d7          \tvfadd.vv\tv7, v8, v9\n")
    stream = rvv.decode_text(text)
    assert stream._section_buckets()[rvv._ALL_SECTIONS][2] is False, (
        "the ascending flag must detect this")
    for sp in [(0, 8), (4, 16), (0, 1 << 40)]:
        _assert_same(stream, sp)


def test_count_in_is_unchanged_by_the_index():
    stream = rvv.decode_text((_FIXTURES / "xnnpack_f32_gemm_rvv.objdump").read_text())
    for sp in stream.loop_spans():
        expected = sum(1 for i in _scan(stream, sp)
                       if i.raw.mnemonic.startswith(("vfmacc", "vle")))
        assert stream.count_in(sp, "vfmacc", "vle") == expected


def test_the_index_is_built_once_and_reused():
    stream = rvv.decode_text((_FIXTURES / "xnnpack_f32_gemm_rvv.objdump").read_text())
    assert stream._by_section is None
    stream.insns_in((0, 16))
    first = stream._by_section
    assert first is not None
    stream.insns_in((0, 32))
    assert stream._by_section is first, "the index must not be rebuilt per call"


def test_an_empty_stream_does_not_raise():
    stream = rvv.decode_text("")
    assert stream.insns_in((0, 16)) == []


def test_a_stream_with_no_section_headers_is_not_counted_twice():
    """The bug this index introduced, and the reason the real fixtures could not catch it.

    The whole-stream bucket was keyed on "", which COLLIDES with instructions whose section genuinely
    is "" -- a stream built straight from RawInsn, as the hermetic memory/facet tests do. So
    `setdefault("").append` appended the stream to itself and every loop-scoped count came back
    exactly doubled: `analyze_memory` reported fma_in_loop=2 for a one-FMA loop. Every objdump
    fixture carries symbol headers, so only a synthetic stream exposes it.
    """
    from merlin.kernels.decode.memory import analyze_memory
    from merlin.kernels.decode.objdump import RawInsn
    from merlin.kernels.decode.rvv import InsnStream, VInsn, VType

    vt = VType(sew=32, lmul=4.0, tail="ta", mask="ma")

    def vi(addr, mn, *ops):
        return VInsn(raw=RawInsn(addr=addr, mnemonic=mn, operands=list(ops)),
                     is_vector=mn.startswith("v"), vtype=vt if mn.startswith("v") else None)

    stream = InsnStream(insns=[
        vi(0x100, "vle32.v", "v12", "(a3)"),
        vi(0x104, "flw", "fa5", "0x0(s1)"),
        vi(0x108, "vfmacc.vf", "v8", "fa5", "v12"),
        vi(0x10c, "bne", "s1", "a5", "0x100"),
    ])
    assert all(i.raw.section == "" for i in stream.insns), "premise: no symbol headers"
    for sp in [(0x100, 0x10c), (0x100, 0x108), (0, 1 << 40)]:
        _assert_same(stream, sp)
    m = analyze_memory(stream)
    assert m is not None and m.fma_in_loop == 1, f"one FMA in the loop, got {m.fma_in_loop}"
    assert m.total_loads == 2, f"one vector + one scalar load, got {m.total_loads}"
