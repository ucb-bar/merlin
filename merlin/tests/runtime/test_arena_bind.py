"""``arena_bind`` — binding the emitted heap allocations to one static arena, and the one mistake
that would not announce itself.

An arena REUSES bytes. Every other kind of bug in this pass fails loudly: an unparsed module raises,
an unbindable allocation stays on the heap. Seating two buffers that are live at the same time on the
same bytes does not fail at all — the module verifies, links, runs, and returns numbers that are
wrong only where the two happened to overlap in time. So the tests here spend most of their effort on
that single property, from three directions: the placement must separate a deliberately constructed
overlap; the guard that checks the placement must be able to FAIL (shown by mutating the conflict
relation, because a check that cannot fail proves nothing); and the rewritten module must compute the
same number the heap version does, which is the only evidence that does not depend on the analysis
being right about itself.

The second thing these tests defend is that the pass is not vacuous. Refusing every allocation is
"safe" and worthless, and it is exactly what happened before the dead ``br i1 true`` edge into each
deallocation block was folded away: with the never-taken edge kept, no free provably ran, no buffer
was ever provably dead, and the arena degenerated to one slot per allocation.
"""
from __future__ import annotations

import ctypes
import subprocess

import pytest

from merlin.llvmlower import arena_bind as ab
from merlin.xdsl_dialects.lowering.arena_plan import ArenaPlanError, pack_disjoint

# ---------------------------------------------------------------------------------------------
# Fixtures: hand-written LLVM IR shaped exactly like what the memref lowering emits — a malloc, the
# align-to-64 ptrtoint round-up the lowering always follows it with, and a dealloc behind a
# `br i1 true` ownership guard.
# ---------------------------------------------------------------------------------------------

_PROLOGUE = """declare ptr @malloc(i64)

declare void @free(ptr)
"""


def _alloc(name: str, size: int) -> str:
    return f"""  %{name} = call ptr @malloc(i64 {size})
  %{name}i = ptrtoint ptr %{name} to i64
  %{name}j = add i64 %{name}i, 63
  %{name}k = urem i64 %{name}j, 64
  %{name}l = sub i64 %{name}j, %{name}k
  %{name}p = inttoptr i64 %{name}l to ptr"""


def _guarded_free(name: str, here: str, nxt: str) -> str:
    return f"""  br i1 true, label %{here}, label %{nxt}

{here}:
  call void @free(ptr %{name})
  br label %{nxt}

{nxt}:"""


#: ``a`` is still live when ``b`` is allocated: ``b``'s malloc sits inside ``a``'s window. The
#: function returns ``a[0] + b[0]``, so seating them on the same bytes turns 1.0 + 2.0 = 3.0 into
#: 2.0 + 2.0 = 4.0 — a wrong number, not a crash, which is the whole point.
OVERLAPPING = _PROLOGUE + f"""
define void @forward(ptr %out) {{
{_alloc("a", 128)}
  store float 1.000000e+00, ptr %ap, align 4
{_alloc("b", 128)}
  store float 2.000000e+00, ptr %bp, align 4
  %av = load float, ptr %ap, align 4
  %bv = load float, ptr %bp, align 4
  %sum = fadd float %av, %bv
  store float %sum, ptr %out, align 4
{_guarded_free("b", "fb", "m1")}
{_guarded_free("a", "fa", "m2")}
  ret void
}}
"""

#: ``a`` is dead before ``b`` is allocated, so one slot must serve both.
SEQUENTIAL = _PROLOGUE + f"""
define void @forward(ptr %out) {{
{_alloc("a", 128)}
  store float 1.000000e+00, ptr %ap, align 4
  %av = load float, ptr %ap, align 4
{_guarded_free("a", "fa", "m1")}
{_alloc("b", 128)}
  store float 2.000000e+00, ptr %bp, align 4
  %bv = load float, ptr %bp, align 4
  %sum = fadd float %av, %bv
  store float %sum, ptr %out, align 4
{_guarded_free("b", "fb", "m2")}
  ret void
}}
"""


def _bind(text: str):
    return ab.bind_arena(text)


# ---------------------------------------------------------------------------------------------
# The property the pass exists for
# ---------------------------------------------------------------------------------------------

def test_a_buffer_still_live_never_shares_bytes_with_the_next_allocation():
    """The deliberate overlap: ``b`` is allocated while ``a`` is still live, so the two may not
    share a byte. This is the failure that would corrupt every model silently."""
    out, rep = _bind(OVERLAPPING)
    assert rep.bound == 2, rep.to_dict()
    off_a = out.split("%a = getelementptr inbounds i8, ptr @merlin_arena, i64 ")[1].split("\n")[0]
    off_b = out.split("%b = getelementptr inbounds i8, ptr @merlin_arena, i64 ")[1].split("\n")[0]
    lo, hi = sorted((int(off_a), int(off_b)))
    assert hi >= lo + 128, f"live buffers seated {lo} and {hi}, closer than their 128-byte size"
    assert rep.arena_bytes >= 256


def test_the_overlap_guard_can_actually_fail():
    """MUTATION: the same two buffers, with the conflict relation emptied. The placement then seats
    them both at 0, and the guard that runs on the offsets actually emitted must reject it.

    Without this, ``_assert_no_conflicting_overlap`` passing proves nothing — a check that cannot
    fail reports success on a broken planner exactly as loudly as on a correct one.
    """
    offsets, total = pack_disjoint([("a", 128), ("b", 128)], {"a": set(), "b": set()})
    assert offsets == {"a": 0, "b": 0} and total == 128, (offsets, total)
    with pytest.raises(ab.ArenaBindError) as exc:
        ab._assert_no_conflicting_overlap(offsets, {"a": 128, "b": 128},
                                          {"a": {"b"}, "b": {"a"}})
    assert "overlapping bytes" in str(exc.value)


def test_the_binder_records_the_conflict_the_guard_would_catch():
    """The guard above is the last line of defence; this is the analysis that should mean it never
    fires. Read the conflict relation the binder derives for the deliberate overlap and check the
    two buffers are in it — a placement is only as good as the relation it is given."""
    conflicts = _conflicts_of(OVERLAPPING)
    assert conflicts["%a"] == {"%b"} and conflicts["%b"] == {"%a"}, conflicts


def test_buffers_whose_windows_do_not_meet_share_one_slot():
    """The vacuity check. Refusing everything is safe and worthless; a planner that never reuses a
    byte is indistinguishable from no planner at all. Here ``a`` is freed before ``b`` exists, so
    both must land at offset 0 and the arena must be ONE buffer wide."""
    out, rep = _bind(SEQUENTIAL)
    assert rep.bound == 2
    assert _conflicts_of(SEQUENTIAL) == {"%a": set(), "%b": set()}
    off_a = int(out.split("%a = getelementptr inbounds i8, ptr @merlin_arena, i64 ")[1].split("\n")[0])
    off_b = int(out.split("%b = getelementptr inbounds i8, ptr @merlin_arena, i64 ")[1].split("\n")[0])
    assert off_a == off_b == 0
    assert rep.arena_bytes == 128, rep.to_dict()
    assert rep.reuse_factor == 2.0


def test_a_dead_ownership_edge_is_not_read_as_a_real_successor():
    """Every dealloc the ownership passes emit sits behind ``br i1 true, label %free, label %skip``.
    Keeping the never-taken ``%skip`` edge makes every free look optional, so no buffer is ever
    provably dead — measured before this fold: 95 allocations bound and 1.00x reuse on a program
    whose buffers do not all coexist. The fold is what makes the analysis non-vacuous."""
    assert ab._label_targets("br i1 true, label %fa, label %m1") == ["%fa"]
    assert ab._label_targets("br i1 false, label %fa, label %m1") == ["%m1"]
    # a genuine condition keeps both edges
    assert ab._label_targets("br i1 %c, label %x, label %y") == ["%x", "%y"]


def _conflicts_of(text: str) -> dict:
    """Re-derive the binder's conflict relation for a module (the pass keeps it internal)."""
    lines = text.split("\n")
    start, end = ab._split_function(lines, "forward")
    blocks = ab._blocks_of(lines, start, end)
    by_label = {b.label: i for i, b in enumerate(blocks)}
    succs = [[by_label[s] for s in b.succs] for b in blocks]
    line_block = {li: (bi, pos) for bi, b in enumerate(blocks) for pos, li in enumerate(b.insts)}
    allocs = []
    frees: dict[str, int] = {}
    for b in blocks:
        for li in b.insts:
            f = ab._parse_free(lines[li].strip())
            if f:
                frees[f] = li
    for b in blocks:
        for li in b.insts:
            m = ab._parse_malloc(lines[li].strip())
            if not m:
                continue
            mb, mp = line_block[li]
            fb, fp = line_block[frees[m[0]]]
            allocs.append(ab._Alloc(name=m[0], size=m[1], malloc_line=li, malloc_block=mb,
                                    malloc_pos=mp, free_line=frees[m[0]], free_block=fb,
                                    free_pos=fp))
    win = {a.name: ab._window_blocks(a, succs) for a in allocs}

    def inside(a, blk, pos):
        if blk == a.malloc_block and blk == a.free_block:
            return a.malloc_pos < pos < a.free_pos
        if blk == a.malloc_block:
            return pos > a.malloc_pos
        if blk == a.free_block:
            return pos < a.free_pos
        return blk in win[a.name]

    out = {a.name: set() for a in allocs}
    for i, a in enumerate(allocs):
        for b in allocs[i + 1:]:
            if inside(a, b.malloc_block, b.malloc_pos) or inside(b, a.malloc_block, a.malloc_pos):
                out[a.name].add(b.name)
                out[b.name].add(a.name)
    return out


# ---------------------------------------------------------------------------------------------
# Executable evidence: the rewritten module must compute what the heap version computes
# ---------------------------------------------------------------------------------------------

def _clang():
    from merlin.llvmlower import toolchain
    try:
        return str(toolchain.clang())
    except Exception:                      # noqa: BLE001 -- no toolchain in this checkout
        return None


@pytest.mark.parametrize("source,expected", [(OVERLAPPING, 3.0), (SEQUENTIAL, 3.0)])
def test_the_rewritten_module_computes_the_same_number(tmp_path, source, expected):
    """Compile and RUN both versions. This is the only evidence that does not rest on the analysis
    being right about itself: if the arena seated the two buffers on top of each other, the
    overlapping module returns 4.0 (b's value read twice), not 3.0."""
    cc = _clang()
    if cc is None:
        pytest.skip("no clang in this checkout")
    bound, rep = _bind(source)
    assert rep.bound == 2
    results = {}
    for tag, text in (("heap", source), ("arena", bound)):
        ll = tmp_path / f"{tag}.ll"
        so = tmp_path / f"{tag}.so"
        ll.write_text(text)
        proc = subprocess.run([cc, "-O0", "-shared", "-fPIC", str(ll), "-o", str(so)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        lib = ctypes.CDLL(str(so))
        out = ctypes.c_float(0.0)
        lib.forward(ctypes.byref(out))
        results[tag] = out.value
    assert results["heap"] == pytest.approx(expected), results
    assert results["arena"] == pytest.approx(results["heap"]), results


# ---------------------------------------------------------------------------------------------
# Fail-closed: everything the analysis cannot prove stays on the heap
# ---------------------------------------------------------------------------------------------

LOOPED = _PROLOGUE + f"""
define void @forward(ptr %out) {{
  br label %top

top:
  %i = phi i64 [ 0, %0 ], [ %n, %body ]
{_alloc("a", 128)}
  store float 1.000000e+00, ptr %ap, align 4
  call void @free(ptr %a)
  br label %body

body:
  %n = add i64 %i, 1
  %c = icmp slt i64 %n, 4
  br i1 %c, label %top, label %done

done:
  ret void
}}
"""


def test_an_allocation_inside_a_loop_is_left_on_the_heap():
    """A site that runs more than once per call has more than one window per call, and the whole
    disjointness argument assumes exactly one. Binding it would give two simultaneously live
    instances the same address."""
    out, rep = _bind(LOOPED)
    assert rep.bound == 0 and rep.refusals == {"in_loop": 1}, rep.to_dict()
    assert out == LOOPED, "a module with nothing bindable must come back byte-identical"


ESCAPING = _PROLOGUE + """
declare void @sink(ptr)

define void @forward(ptr %out) {
  %a = call ptr @malloc(i64 128)
  %ai = ptrtoint ptr %a to i64
  %ap = inttoptr i64 %ai to ptr
  store ptr %a, ptr %out, align 8
  store float 1.000000e+00, ptr %ap, align 4
  br i1 true, label %fa, label %m1

fa:
  call void @free(ptr %a)
  br label %m1

m1:
  ret void
}
"""


def test_a_raw_pointer_written_to_caller_memory_is_left_on_the_heap():
    """The raw pointer stored through ``%out`` outlives the call, so the plan's claim that those
    bytes belong to another buffer afterwards is not true."""
    out, rep = _bind(ESCAPING)
    assert rep.bound == 0 and rep.refusals == {"raw_pointer_escapes": 1}, rep.to_dict()
    assert out == ESCAPING


DESCRIPTOR = _PROLOGUE + """
declare void @memrefCopy(i64, ptr, ptr)
declare void @capture(ptr)

define void @forward(ptr %out) {
  %a = call ptr @malloc(i64 128)
  %ai = ptrtoint ptr %a to i64
  %ap = inttoptr i64 %ai to ptr
  %d = insertvalue { ptr, ptr, i64 } poison, ptr %a, 0
  %d2 = insertvalue { ptr, ptr, i64 } %d, ptr %ap, 1
  store float 1.000000e+00, ptr %ap, align 4
  br i1 true, label %fa, label %m1

fa:
  call void @free(ptr %a)
  br label %m1

m1:
  ret void
}
"""


def test_a_memref_descriptor_use_is_admitted_only_when_no_callee_can_capture():
    """``insertvalue`` builds the memref descriptor the ``memref.copy`` lowering hands to
    ``@memrefCopy``; refusing it costs 15% of the allocations on the deepjscc int8 build (17 of 112).
    It is admitted only because nothing in the function can retain what it is given — so an unknown
    callee must switch the relaxation back off for the whole function, and this test pins BOTH
    directions rather than only the permissive one."""
    out, rep = _bind(DESCRIPTOR)
    assert rep.bound == 1 and rep.refusals == {}, rep.to_dict()
    assert "@merlin_arena" in out

    hostile = DESCRIPTOR.replace("  br i1 true, label %fa, label %m1",
                                 "  call void @capture(ptr %ap)\n  br i1 true, label %fa, label %m1")
    out2, rep2 = _bind(hostile)
    assert rep2.bound == 0 and rep2.refusals == {"raw_pointer_escapes": 1}, rep2.to_dict()
    assert out2 == hostile


DESCRIPTOR_ESCAPES = _PROLOGUE + """
declare void @memrefCopy(i64, ptr, ptr)

define void @forward(ptr %out) {
  %a = call ptr @malloc(i64 128)
  %ai = ptrtoint ptr %a to i64
  %ap = inttoptr i64 %ai to ptr
  %d = insertvalue { ptr, ptr, i64 } poison, ptr %a, 0
  %d2 = insertvalue { ptr, ptr, i64 } %d, ptr %ap, 1
  store { ptr, ptr, i64 } %d2, ptr %out, align 8
  br i1 true, label %fa, label %m1

fa:
  call void @free(ptr %a)
  br label %m1

m1:
  ret void
}
"""


def test_a_descriptor_written_through_a_caller_pointer_is_left_on_the_heap():
    """MUTATION of the case above: the same descriptor, stored through the caller's ``%out`` instead
    of a local ``alloca``. The arena address now outlives the call, past the point the plan hands
    those bytes to another buffer.

    This is what makes the ``insertvalue`` relaxation defensible rather than convenient. Checking
    only the RAW pointer's uses would pass this module: the value that escapes is the aggregate, and
    the raw pointer's own uses are still just ``ptrtoint``, ``insertvalue`` and its ``free``.
    """
    out, rep = _bind(DESCRIPTOR_ESCAPES)
    assert rep.bound == 0 and rep.refusals == {"descriptor_escapes": 1}, rep.to_dict()
    assert out == DESCRIPTOR_ESCAPES
    # ...and the same module with a LOCAL spill destination is accepted, so the check is discriminating
    local = DESCRIPTOR_ESCAPES.replace(
        "  %a = call ptr @malloc(i64 128)",
        "  %slot = alloca { ptr, ptr, i64 }, align 8\n  %a = call ptr @malloc(i64 128)").replace(
        "store { ptr, ptr, i64 } %d2, ptr %out, align 8",
        "store { ptr, ptr, i64 } %d2, ptr %slot, align 8\n"
        "  call void @memrefCopy(i64 4, ptr %slot, ptr %slot)")
    _, rep2 = _bind(local)
    assert rep2.bound == 1 and rep2.refusals == {}, rep2.to_dict()


def test_a_store_destination_is_read_structurally():
    """An aggregate type spells its members ``", ptr,"``; the destination operand is ``", ptr %x"``.
    Reading the wrong one would name a member type as the destination and refuse everything."""
    assert ab._store_destination(
        "store { ptr, ptr, i64 } %d2, ptr %slot, align 8") == "%slot"
    assert ab._store_destination("store ptr %a, ptr %out, align 8") == "%out"
    assert ab._store_destination("%x = load ptr, ptr %y") is None


def test_a_libm_name_is_matched_exactly_not_by_prefix():
    """``@sincosf`` and ``@frexpf`` write through a pointer out-parameter. Matching the pure-math
    names by prefix would admit them under ``@sin`` / ``@f`` and quietly re-enable the descriptor
    relaxation in a function that CAN capture."""
    assert ab._is_non_capturing("@sinf") and ab._is_non_capturing("@cosf")
    assert not ab._is_non_capturing("@sincosf")
    assert not ab._is_non_capturing("@frexpf")
    assert not ab._is_non_capturing("@modff")
    assert not ab._is_non_capturing("<indirect>")


def test_a_computed_malloc_size_is_left_on_the_heap():
    """A non-literal size is a dynamically shaped buffer. ``arena_plan`` refuses a dynamic extent
    for the same reason: a plan that looks complete over a buffer it could not size is an arena that
    is too small, and the failure lands as heap corruption far from the cause."""
    assert ab._parse_malloc("%a = call ptr @malloc(i64 %n)") is None
    assert ab._parse_malloc("%a = call ptr @malloc(i64 128)") == ("%a", 128)


def test_a_function_that_returns_a_value_is_refused_outright():
    """A returned value could carry an arena pointer past the point the plan gives those bytes to
    somebody else. That is a property of the whole module, not of one allocation, so it raises."""
    bad = _PROLOGUE + """
define ptr @forward(ptr %out) {
  %a = call ptr @malloc(i64 128)
  ret ptr %a
}
"""
    with pytest.raises(ab.ArenaBindError):
        _bind(bad)


def test_a_branch_whose_successors_cannot_be_read_raises_rather_than_binding():
    """A missing CFG edge produces a reachability answer that is wrong in the one direction nothing
    else checks — it makes windows look smaller than they are, which is how two live buffers end up
    sharing bytes. Refuse the module instead of transforming it on a partial CFG."""
    broken = OVERLAPPING.replace("  br i1 true, label %fb, label %m1",
                                 "  br unknownform")
    with pytest.raises(ab.ArenaBindError):
        _bind(broken)


# ---------------------------------------------------------------------------------------------
# The frozen-baseline invariant at the seam that calls this
# ---------------------------------------------------------------------------------------------

def test_lower_model_does_not_touch_the_ll_unless_the_flag_is_set(tmp_path, monkeypatch):
    """With the flag off the emitted ``.ll`` must be what the pipeline produced, byte for byte —
    the repo has been burned twice by an obviously-good change that regressed on silicon, so the
    unflagged build has to stay the frozen control."""
    from merlin.llvmlower import lower as L

    monkeypatch.setattr(L, "preprocess_text", lambda text: (text, {}))
    monkeypatch.setattr(L, "lower_to_llvm_ir", lambda *a, **k: OVERLAPPING)
    monkeypatch.delenv("MERLIN_STATIC_ARENA", raising=False)

    off = L.lower_model("ignored", tmp_path / "off", targets=())
    assert off.ll_path.read_text() == OVERLAPPING
    assert "static_arena" not in off.stats

    on = L.lower_model("ignored", tmp_path / "on", targets=(), static_arena=True)
    assert on.ll_path.read_text() != OVERLAPPING
    assert on.stats["static_arena"]["bound"] == 2
    assert "@merlin_arena" in on.ll_path.read_text()

    monkeypatch.setenv("MERLIN_STATIC_ARENA", "1")
    env = L.lower_model("ignored", tmp_path / "env", targets=())
    assert env.stats["static_arena"]["bound"] == 2


def test_pack_disjoint_refuses_a_zero_sized_block():
    with pytest.raises(ArenaPlanError):
        pack_disjoint([("a", 0)], {})
