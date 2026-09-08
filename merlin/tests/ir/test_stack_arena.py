"""Moving a whole-model entry frame's static temporaries into one arena.

THE DEFECT, MEASURED. A target declares a maximum static frame for its kernel entrypoint. The host
lane hoists one ``alloca`` per intermediate into that frame with no reuse, so the frame scales with
the model: 816 bytes on ResNet-50 -- which is why it went unnoticed -- and **99,897,984 bytes on
SmolVLA's flow_denoise against a 65,536-byte budget**, from 1,534 allocas whose declared sizes
account for 100.0% of the demand. Seating them in one ``.bss`` arena measured a **496-byte** frame.

WHY THE PASS REFUSES TO SHARE BYTES, and why that is the point rather than a shortcoming. Colouring
the allocations the way :mod:`merlin.llvmlower.arena_bind` colours the heap cannot work here:

* SmolVLA's largest single allocation is 489,000 bytes -- 7.5x the whole budget -- so perfect reuse
  still would not fit, and the win being claimed is moving the storage off a 64 KiB stack, not
  shrinking it;
* an ``alloca`` has no ``free``, so no two live ranges are provably disjoint and sharing would be a
  guess. Refusing to share is what makes the transform sound with no liveness analysis at all.

So the tests below pin the refusals, the placement's disjointness, and -- the property everything
else rests on -- that a module with nothing bindable comes back BYTE-IDENTICAL, which is what lets
the compiler apply this only to a frame it has measured over budget and leave every already-fitting
build exactly as it was.
"""
from __future__ import annotations

import pytest

from merlin.llvmlower import stack_arena as SA
from merlin.llvmlower.stack_arena import StackArenaError, bind_stack_arena


def _module(body: str, *, entry: str = "k", args: str = "ptr %0") -> str:
    return "\n".join([
        "; ModuleID = 'test'",
        f"define void @{entry}({args}) {{",
        *body.strip("\n").split("\n"),
        "}",
        "",
    ])


class TestItMovesWhatItCanProve:
    def test_a_constant_count_alloca_becomes_a_gep_into_one_arena(self):
        text = _module("""
  %1 = alloca float, i64 1024, align 64
  %2 = alloca i8, i64 16, align 16
  ret void
""")
        out, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 2 and report.n_refused == 0
        assert report.moved_bytes == 1024 * 4 + 16
        assert " = alloca " not in out
        assert out.count("getelementptr inbounds i8, ptr @merlin_stack_arena") == 2
        assert f"@{SA.STACK_ARENA_SYMBOL} = internal global" in out

    def test_the_arena_is_zeroinitialized_so_it_lands_in_bss(self):
        out, _ = bind_stack_arena(_module("  %1 = alloca i8, i64 8, align 8\n  ret void"),
                                  entry_symbol="k")
        line = next(l for l in out.split("\n") if l.startswith("@" + SA.STACK_ARENA_SYMBOL))
        assert "internal global" in line and "zeroinitializer" in line

    def test_the_original_alloca_is_recorded_in_the_rewritten_line(self):
        """A reader of the IR must be able to see what the slot used to be."""
        out, _ = bind_stack_arena(_module("  %1 = alloca i16, i64 50, align 64\n  ret void"),
                                  entry_symbol="k")
        gep = next(l for l in out.split("\n") if "getelementptr" in l)
        assert "was `alloca i16, i64 50, align 64`" in gep and "100 bytes" in gep

    def test_each_slot_keeps_at_least_its_requested_alignment(self):
        text = _module("""
  %1 = alloca i8, i64 1, align 1
  %2 = alloca float, i64 4, align 64
  %3 = alloca i8, i64 3, align 1
  %4 = alloca double, i64 2, align 128
  ret void
""")
        out, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 4
        offsets = {}
        for line in out.split("\n"):
            if "getelementptr" not in line:
                continue
            name = line.strip().split(" = ", 1)[0]
            offsets[name] = int(line.split("ptr @merlin_stack_arena, i64 ", 1)[1].split(" ", 1)[0])
        assert offsets["%2"] % 64 == 0
        assert offsets["%4"] % 128 == 0
        arena = next(l for l in out.split("\n") if l.startswith("@" + SA.STACK_ARENA_SYMBOL))
        assert "align 128" in arena, "the arena must be at least as aligned as its strictest slot"

    def test_slots_never_overlap(self):
        text = _module("\n".join(
            [f"  %{i} = alloca i8, i64 {7 * i + 1}, align {1 << (i % 4)}" for i in range(1, 40)]
            + ["  ret void"]))
        out, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 39
        placed = []
        for line in out.split("\n"):
            if "getelementptr" not in line:
                continue
            offset = int(line.split("ptr @merlin_stack_arena, i64 ", 1)[1].split(" ", 1)[0])
            size = int(line.split("` (", 1)[1].split(" bytes", 1)[0])
            placed.append((offset, size))
        placed.sort()
        for (offset, size), (next_offset, _) in zip(placed, placed[1:]):
            assert offset + size <= next_offset, f"{offset}+{size} overlaps {next_offset}"
        assert placed[-1][0] + placed[-1][1] <= report.arena_bytes

    def test_it_never_shares_bytes_and_says_so(self):
        text = _module("""
  %1 = alloca i8, i64 1000, align 64
  %2 = alloca i8, i64 1000, align 64
  ret void
""")
        _, report = bind_stack_arena(text, entry_symbol="k")
        block = report.to_dict()
        assert block["shares_bytes"] is False
        assert report.arena_bytes >= 2000, "two slots must not be overlaid"
        assert "no free" in block["why_no_sharing"]


class TestNothingBindableMeansNothingChanged:
    """The property the compiler's repair-on-failure wiring depends on."""

    def test_a_module_with_no_alloca_is_returned_BYTE_IDENTICAL(self):
        text = _module("  ret void")
        out, report = bind_stack_arena(text, entry_symbol="k")
        assert out == text, "not merely equivalent -- identical"
        assert report.n_bound == 0
        assert SA.STACK_ARENA_SYMBOL not in out

    def test_a_module_whose_every_alloca_is_refused_is_also_IDENTICAL(self):
        text = _module("""
  br label %loop
loop:
  %1 = alloca i8, i64 64, align 8
  br label %loop
""")
        out, report = bind_stack_arena(text, entry_symbol="k")
        assert out == text
        assert report.n_bound == 0 and report.n_refused == 1


class TestTheRefusals:
    def test_a_DYNAMIC_count_is_refused(self):
        text = _module("""
  %n = add i64 1, 2
  %1 = alloca float, i64 %n, align 64
  ret void
""")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 0 and report.n_refused == 1
        assert "static size" in report.refusals[0]["reason"]

    def test_an_alloca_ON_A_CFG_CYCLE_is_refused(self):
        """The one way this transform could produce wrong numbers: two live instances, one slot."""
        text = _module("""
  br label %loop
loop:
  %1 = alloca i8, i64 64, align 8
  br i1 true, label %loop, label %done
done:
  ret void
""")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 0 and report.n_refused == 1
        assert "two dynamic instances" in report.refusals[0]["reason"]
        assert report.remaining_stack_bytes == 64

    def test_a_self_edge_counts_as_a_cycle(self):
        text = _module("""
  br label %spin
spin:
  %1 = alloca i8, i64 8, align 8
  br label %spin
""")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 0 and report.n_refused == 1

    def test_an_UNREACHABLE_block_is_refused(self):
        text = _module("""
  ret void
orphan:
  %1 = alloca i8, i64 32, align 8
  ret void
""")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 0 and report.n_refused == 1
        assert "never be written" in report.refusals[0]["reason"]

    def test_a_NON_DEFAULT_address_space_is_refused(self):
        text = _module("  %1 = alloca i8, i64 8, align 8, addrspace(3)\n  ret void")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 0 and report.n_refused == 1

    def test_an_UNSIZEABLE_element_type_is_refused_not_guessed(self):
        text = _module("  %1 = alloca %struct.thing, i64 4, align 8\n  ret void")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 0 and report.n_refused == 1
        assert "not one this pass can size" in report.refusals[0]["reason"]

    def test_every_known_element_width_is_correct(self):
        for name, width in (("i8", 1), ("i16", 2), ("i32", 4), ("i64", 8), ("half", 2),
                            ("bfloat", 2), ("float", 4), ("double", 8), ("ptr", 8)):
            assert SA.element_bytes_of(name) == width
            _, report = bind_stack_arena(
                _module(f"  %1 = alloca {name}, i64 10, align 8\n  ret void"), entry_symbol="k")
            assert report.moved_bytes == width * 10, name

    def test_a_RECURSIVE_entrypoint_is_refused_outright(self):
        """A static arena is one object per module: the inner call would scribble on the outer's."""
        text = _module("""
  %1 = alloca i8, i64 8, align 8
  call void @k(ptr %0)
  ret void
""")
        with pytest.raises(StackArenaError, match="scribble on its own caller"):
            bind_stack_arena(text, entry_symbol="k")

    def test_an_absent_entrypoint_raises_rather_than_returning_nothing_moved(self):
        with pytest.raises(StackArenaError):
            bind_stack_arena(_module("  ret void"), entry_symbol="not_here")

    def test_an_empty_entry_symbol_is_refused(self):
        with pytest.raises(StackArenaError, match="target's declaration"):
            bind_stack_arena(_module("  ret void"), entry_symbol="")


class TestTheReportIsSelfDescribing:
    def test_it_states_the_non_reentrancy_it_introduces(self):
        """A contract change must travel with the transform, not live in a commit message."""
        _, report = bind_stack_arena(
            _module("  %1 = alloca i8, i64 8, align 8\n  ret void"), entry_symbol="k")
        block = report.to_dict()
        assert "NOT reentrant" in block["reentrancy"]
        assert block["schema"] == "merlin_stack_arena_bind_v1"

    def test_it_accounts_for_every_alloca_it_saw(self):
        text = _module("""
  %dyn = add i64 1, 1
  %1 = alloca i8, i64 64, align 8
  %2 = alloca float, i64 %dyn, align 8
  br label %loop
loop:
  %3 = alloca i8, i64 16, align 8
  br label %loop
""")
        _, report = bind_stack_arena(text, entry_symbol="k")
        assert report.n_bound == 1, "only the constant, reachable, acyclic one"
        assert report.n_refused == 2, "the dynamic count and the one on the cycle"
        assert report.n_bound + report.n_refused == 3

    def test_the_symbol_is_distinct_from_the_heap_arenas(self):
        """Both passes may run; one symbol would silently overlay two different sets of storage."""
        from merlin.llvmlower.arena_bind import ARENA_SYMBOL
        assert SA.STACK_ARENA_SYMBOL != ARENA_SYMBOL


class TestTheRealSmolVLAModule:
    """The acceptance test: the module the preflight actually rejected."""

    def _llvm(self):
        from merlin.common.paths import artifacts_dir
        path = (artifacts_dir() / "perf-bench" / "gemmini"
                / "smolvla_flow_denoise_bundle_20260908" / "compiler" / "kernel.ll")
        if not path.is_file():
            pytest.skip("no emitted smolvla kernel.ll in this tree")
        return path.read_text(encoding="utf-8")

    def test_the_emitted_frame_really_is_over_budget(self):
        """If this module ever stops being over budget, the tests below prove nothing."""
        text = self._llvm()
        total = 0
        for line in text.split("\n"):
            if " = alloca " not in line:
                continue
            parsed = SA._parse_alloca(line)
            assert parsed is not None, line
            _, elem, count, _ = parsed
            total += SA.element_bytes_of(elem) * count
        assert total > 65536, f"only {total} bytes; the defect this fixes is not present"

    def test_every_alloca_is_bound_and_none_refused(self):
        _, report = bind_stack_arena(self._llvm(), entry_symbol="gemmini_kernel")
        assert report.n_bound == 1534
        assert report.n_refused == 0
        assert report.remaining_stack_bytes == 0
        assert report.moved_bytes == 99871398

    def test_no_alloca_survives_and_the_arena_covers_them_all(self):
        out, report = bind_stack_arena(self._llvm(), entry_symbol="gemmini_kernel")
        assert " = alloca " not in out
        assert report.arena_bytes >= report.moved_bytes
        assert f"[{report.arena_bytes} x i8] zeroinitializer" in out

    def test_colouring_could_not_have_fixed_it(self):
        """The measurement that decides the design: the largest single slot exceeds the budget."""
        largest = 0
        for line in self._llvm().split("\n"):
            if " = alloca " not in line:
                continue
            _, elem, count, _ = SA._parse_alloca(line)
            largest = max(largest, SA.element_bytes_of(elem) * count)
        assert largest > 65536, (
            "if the largest allocation fits the budget, byte reuse WOULD be a candidate fix and "
            "this module's refusal to share needs re-arguing")
