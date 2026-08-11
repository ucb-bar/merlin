"""The emitted kernel is asserted from the OBJECT, not from the source it was generated out of.

The failure this defends against was invisible in source. A kernel wrote `asm volatile("vsetvli …")`
before each operand load, exactly as intended, and LLVM's vector-length insertion pass — which treats a
standalone asm block as opaque and does not track the length it establishes — left both loads running on
a length set for something else. Both read 16 bytes; one of them read 15 bytes past the end of its panel.
Every source-level check passed.

So the tests that matter here compile the generated kernel and read the instruction stream back: the
configure/load/configure/load/accumulate sequence must be contiguous, the row load must keep its
tail-undisturbed policy, and the two operand lengths must come from different registers.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.kernels.decode import opu as OA
from merlin.kernels.opu_kernel import KernelSpec, emit_microkernel, emit_reference_c
from merlin.targetgen.rtl import opu_isa as OI


class _Enc:
    """A derived-encoding stand-in with the same interface the emitter uses."""

    def __init__(self, opcode: int, funct3: int, funct6: int):
        self.opcode, self.funct3, self.funct6 = opcode, funct3, funct6

    @property
    def funct7(self) -> int:
        return (self.funct6 << 1) | 1

    def insn_r(self, rd: str, rs1: str, rs2: str) -> str:
        return f".insn r {self.opcode:#x}, {self.funct3:#x}, {self.funct7:#x}, {rd}, {rs1}, {rs2}"


_TABLE = {"ACC": _Enc(0x57, 2, 40), "MOVEIN": _Enc(0x57, 6, 42),
          "BCAST": _Enc(0x57, 6, 44), "READOUT": _Enc(0x57, 6, 46)}
_SPEC = KernelSpec(accumulate="ACC", broadcast="BCAST", readout="READOUT")


class TestRefusesToGuess:
    def test_a_disagreeing_derivation_cannot_be_emitted_from(self):
        # Emitting would bake in whichever source happened to be read.
        with pytest.raises(ValueError, match="cross-check"):
            emit_microkernel(_TABLE, _SPEC, derivation_ok=False)

    def test_a_missing_instruction_is_refused_rather_than_defaulted(self):
        with pytest.raises(ValueError, match="missing"):
            emit_microkernel({"ACC": _TABLE["ACC"]}, _SPEC)

    def test_the_emitted_words_come_from_the_table_not_from_literals(self):
        # Shift every funct6 and the emitted directives must move with it.
        shifted = {k: _Enc(v.opcode, v.funct3, v.funct6 + 2) for k, v in _TABLE.items()}
        base = emit_microkernel(_TABLE, _SPEC)
        moved = emit_microkernel(shifted, _SPEC)
        assert "0x51" in base and "0x51" not in moved
        assert "0x55" in moved


def _device_reduction_loop(src: str) -> str:
    """The reduction loop of the DEVICE tile body.

    The emitted file also contains a scalar tile body behind ``#ifdef OPU_SCALAR_TILE`` (the host build
    that validates the tiling loop), and it has a reduction loop too — so a slice that merely searched for
    the first ``for`` over the reduction would read the wrong branch and these assertions would be about
    the stand-in rather than about the code the device runs.
    """
    device = src[src.index("#else"):src.index("#endif")]
    # Anchored on the emitted comments rather than on the loop's own syntax: the loop is unrolled and the
    # left-operand register rotated (see KernelSpec.row_vreg_alt), so a slice keyed to `for (size_t kk`
    # silently stopped matching and these assertions began raising instead of checking anything.
    return device[device.index("One fused block per reduction step"):
                  device.index("Readout is row-serial")]


#: Matched WITH the opening paren and quote so the phrase "asm volatile" in the surrounding comment is
#: not counted as a block — it is, and that made the single-buffered form look like it emitted two.
_ASM = 'asm volatile("'


def _fused_blocks(loop: str) -> list[str]:
    """Each ``asm volatile`` fused block in the reduction, sliced up to its operand list."""
    out = []
    rest = loop
    while _ASM in rest:
        rest = rest[rest.index(_ASM):]
        body = rest[:rest.index("::")] if "::" in rest else rest
        out.append(body)
        rest = rest[len(_ASM):]
    return out


class TestGeneratedSource:
    def test_every_fused_block_keeps_the_whole_sequence_in_one_asm_block(self):
        # The rotated loop emits several fused blocks (one per register in the rotation, plus the peeled
        # odd step). EVERY one must carry the whole configure/zero/load/configure/load/accumulate
        # sequence: a block that split it would let the vector-length insertion pass reinterpret a length,
        # which is the historical failure.
        blocks = _fused_blocks(_device_reduction_loop(emit_microkernel(_TABLE, _SPEC)))
        assert len(blocks) >= 1
        for i, block in enumerate(blocks):
            for piece in ("vmv.v.i", "tu, ma", "vle8.v", ".insn r"):
                assert piece in block, f"block {i} is missing {piece}"

    def test_the_single_buffered_form_emits_exactly_one_block(self):
        # The opt-out still has to hold the property, so the check is about the sequence and not about
        # how many times it appears.
        spec = KernelSpec(accumulate="ACC", broadcast="BCAST", readout="READOUT", row_vreg_alt=None)
        blocks = _fused_blocks(_device_reduction_loop(emit_microkernel(_TABLE, spec)))
        assert len(blocks) == 1
        for piece in ("vmv.v.i", "tu, ma", "vle8.v", ".insn r"):
            assert piece in blocks[0], piece

    def test_the_rotation_uses_a_different_register_per_step(self):
        # The point of the rotation: the write following an accumulate must not target the register that
        # accumulate is still reading. If both blocks used one register it would rotate nothing.
        blocks = _fused_blocks(_device_reduction_loop(emit_microkernel(_TABLE, _SPEC)))
        rows = {f"v{n}" for n in (_SPEC.row_vreg, _SPEC.row_vreg_alt) if n is not None}
        used = {r for r in rows for b in blocks if f"vmv.v.i {r}," in b}
        assert used == rows, f"expected both left-operand registers to appear, got {used}"

    def test_the_row_load_is_tail_undisturbed_and_the_column_load_is_not(self):
        for block in _fused_blocks(_device_reduction_loop(emit_microkernel(_TABLE, _SPEC))):
            assert "tu, ma" in block, "zeroed row lanes only survive a tail-undisturbed load"
            assert "ta, ma" in block

    def test_the_scalar_stand_in_is_compile_time_selected_and_not_in_the_device_path(self):
        # It exists only so the tiling loop can be validated on a host; it must not be reachable in a
        # device build, or the kernel could quietly compute correct answers without using the unit.
        src = emit_microkernel(_TABLE, _SPEC)
        assert "#ifdef OPU_SCALAR_TILE" in src
        assert ".insn r" not in src[src.index("#ifdef OPU_SCALAR_TILE"):src.index("#else")]

    def test_the_reference_shares_the_kernels_signature(self):
        # Same signature so a host or in-image comparison can call either through one declaration.
        args = "(int32_t *c, const int8_t *at, const int8_t *b, const int32_t *bias,\n"
        tail = "size_t m, size_t n, size_t k)"
        for src in (emit_microkernel(_TABLE, _SPEC), emit_reference_c()):
            assert args in src and tail in src


def _compile(src: str, tmp_path: Path) -> Path:
    from merlin.llvmlower import toolchain
    if not toolchain.available():
        pytest.skip("needs the pinned clang")
    tmp_path.mkdir(parents=True, exist_ok=True)
    c = tmp_path / "k.c"
    c.write_text(src, encoding="utf-8")
    o = tmp_path / "k.o"
    p = subprocess.run([toolchain.clang(), "--target=riscv64-unknown-elf", "-march=rv64gcv",
                        "-mabi=lp64d", "-O2", "-c", str(c), "-o", str(o)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        pytest.fail(f"the generated kernel does not compile:\n{p.stderr[-2000:]}")
    return o


class TestEmittedCode:
    """Read back from the object. This is the layer the historical failure hid below."""

    @pytest.fixture
    def stream(self, tmp_path):
        from merlin.kernels.decode.objdump import tokenize
        obj = _compile(emit_microkernel(_TABLE, _SPEC) + emit_reference_c(), tmp_path)
        return tokenize(obj), obj

    def test_it_compiles_and_emits_every_declared_instruction(self, stream):
        _, obj = stream
        a = OA.audit_object(obj, _TABLE)
        assert a.counts["ACC"] >= 1
        assert a.counts["BCAST"] >= 1, "the accumulator must be initialised on the unit"
        assert a.counts["READOUT"] >= 1, "an accumulate with no readout computes nothing observable"
        assert a.unaccounted == (), a.unaccounted

    def test_no_instruction_inherits_its_operand_length(self, stream):
        _, obj = stream
        a = OA.audit_object(obj, _TABLE)
        assert a.unconfigured == (), a.unconfigured
        assert a.notes == (), a.notes

    def test_the_fused_sequence_is_contiguous_in_the_emitted_code(self, stream):
        # THE regression. Nothing may be inserted between configuring a length and the load or
        # accumulate that depends on it.
        insns, _ = stream
        decoded = OA.decode_stream(insns, _TABLE)
        acc_at = [d.index for d in decoded if d.identity == "ACC"]
        assert acc_at, "no accumulate was emitted"
        i = acc_at[0]
        before = [insns[j].mnemonic for j in range(i - 6, i)]
        assert before == ["vsetvli", "vmv.v.i", "vsetvli", "vle8.v", "vsetvli", "vle8.v"], before

    def test_the_two_operand_lengths_come_from_different_registers(self, stream):
        # One length for the rows and one for the columns. A single shared length is the narrow-operand
        # bug: the row load would run at the column count and read past its panel.
        insns, _ = stream
        decoded = OA.decode_stream(insns, _TABLE)
        i = next(d.index for d in decoded if d.identity == "ACC")
        row_cfg, col_cfg = insns[i - 4], insns[i - 2]
        assert row_cfg.mnemonic == "vsetvli" and col_cfg.mnemonic == "vsetvli"
        assert row_cfg.operands[1] != col_cfg.operands[1], (row_cfg.operands, col_cfg.operands)

    def test_the_row_length_is_established_tail_undisturbed(self, stream):
        insns, _ = stream
        decoded = OA.decode_stream(insns, _TABLE)
        i = next(d.index for d in decoded if d.identity == "ACC")
        assert "tu" in insns[i - 4].operands, insns[i - 4].operands
        assert "ta" in insns[i - 2].operands, insns[i - 2].operands

    def test_the_row_lanes_are_zeroed_at_the_maximum_length(self, stream):
        # Zeroing at `ml` instead of the maximum would leave the lanes past the panel undefined, which is
        # what lets a short panel multiply garbage into the accumulator.
        insns, _ = stream
        decoded = OA.decode_stream(insns, _TABLE)
        i = next(d.index for d in decoded if d.identity == "ACC")
        zero_cfg = insns[i - 6]
        assert zero_cfg.mnemonic == "vsetvli"
        assert "zero" in zero_cfg.operands[1], zero_cfg.operands


class TestAgainstTheRealDerivation:
    """The same emission from the actual RTL, so the kernel is generated from derived facts end to end."""

    @pytest.fixture
    def derived(self):
        # paths.env, not os.environ: the checkout lives in the gitignored `.env`, and reading only the
        # process environment made this skip even where the hardware was present.
        from merlin.common.paths import env as _env
        root = _env("MERLIN_CHIPYARD")
        if not root:
            pytest.skip("needs the hardware checkout ($MERLIN_CHIPYARD)")
        s = Path(root) / "generators/saturn"
        if not s.is_dir():
            pytest.skip(f"no saturn generator under {s}")
        d = OI.derive(consts=s / "src/main/scala/common/Consts.scala",
                      instructions=s / "src/main/scala/insns/Instructions.scala",
                      params=s / "src/main/scala/common/Parameters.scala",
                      funct6_enum="OPMFunct6", consts_container="HasVectorConsts",
                      insn_seq="opuInsns", opcode_name="opcVector",
                      form_funct3={"VV": "OPMVV", "VX": "OPMVX"})
        return OI.crosscheck(d, s / "benchmarks/common/bme.h",
                             pairs={"OPMACC": "VOPACC", "OPMVIN": "VMV_RV",
                                    "OPMVINBCAST": "OPMVINBCAST", "OPMVOUT": "VMV_VR"})

    def test_the_kernel_generated_from_real_derived_facts_compiles_and_audits_clean(self, derived, tmp_path):
        assert derived.ok, [c for c in derived.crosschecks if not c["agrees"]]
        spec = KernelSpec(accumulate="OPMACC", broadcast="OPMVINBCAST", readout="OPMVOUT")
        src = emit_microkernel(derived.encodings, spec, derivation_ok=derived.ok)
        obj = _compile(src + emit_reference_c(), tmp_path)
        a = OA.audit_object(obj, derived.encodings)
        assert a.counts["OPMACC"] >= 1 and a.counts["OPMVOUT"] >= 1
        assert a.unconfigured == () and a.unaccounted == ()


#: The kernel shape the historical failure had: each instruction in its own `asm volatile`, so the
#: vector-length insertion pass sees separate opaque blocks and the sequence is not contiguous.
_UNFUSED = """
#include <stdint.h>
#include <stddef.h>
void unfused(int32_t *c, const int8_t *at, const int8_t *b, size_t m, size_t n, size_t k,
             size_t ml, size_t nl) {
  for (size_t kk = 0; kk < k; ++kk) {
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" :: "r"(ml));
    asm volatile("vle8.v v5, (%0)" :: "r"(at + kk * m));
    asm volatile("vsetvli zero, %0, e8, m1, ta, ma" :: "r"(nl));
    asm volatile("vle8.v v4, (%0)" :: "r"(b + kk * n));
    asm volatile(".insn r 0x57, 0x2, 0x51, x1, x5, x4");
  }
}
"""


class TestTheGuardActuallyGuards:
    """A regression test that cannot fail is not protecting anything, so the unfused kernel — the shape
    the historical failure had — is compiled here and the checks are shown to reject it."""

    @pytest.fixture
    def unfused(self, tmp_path):
        from merlin.kernels.decode.objdump import tokenize
        return tokenize(_compile(_UNFUSED, tmp_path))

    def test_the_contiguity_check_rejects_the_unfused_kernel(self, unfused):
        decoded = OA.decode_stream(unfused, _TABLE)
        i = next(d.index for d in decoded if d.identity == "ACC")
        before = [unfused[j].mnemonic for j in range(i - 6, i)]
        assert before != ["vsetvli", "vmv.v.i", "vsetvli", "vle8.v", "vsetvli", "vle8.v"], (
            "the unfused kernel must not satisfy the contiguity assertion, or that assertion is vacuous")

    def test_control_flow_lands_inside_the_sequence_when_it_is_not_fused(self, unfused):
        # This is the observable difference: the loads and the accumulate are no longer adjacent, so
        # nothing constrains what the compiler may do between establishing a length and using it.
        decoded = OA.decode_stream(unfused, _TABLE)
        i = next(d.index for d in decoded if d.identity == "ACC")
        window = [unfused[j].mnemonic for j in range(i - 6, i)]
        assert any(m.startswith(("c.j", "c.b", "j", "b")) for m in window), window

    def test_the_unfused_kernel_never_zeroes_the_row_lanes(self, unfused):
        # Without the zeroing, lanes past a short panel hold undefined register contents and the
        # accumulate multiplies them into the result.
        assert not any(i.mnemonic == "vmv.v.i" for i in unfused)

    def test_the_fused_kernel_and_the_unfused_one_do_not_share_a_digest(self, unfused, tmp_path):
        # If they did, the inert-lever guard could not tell the fix from the bug.
        fused = _compile(emit_microkernel(_TABLE, _SPEC) + emit_reference_c(), tmp_path / "f")
        assert OA.audit_object(fused, _TABLE).digest != OA.audit(unfused, _TABLE).digest


class TestTilingIsNumericallyExact:
    """The tiling loop is checked numerically on the host, with no hardware involved.

    A compile-time switch swaps the tile body for a scalar stand-in while leaving the tiling loop --
    the tail bounds, the pointer arithmetic, the bias column offset -- untouched. There is ONE copy of
    that loop, so this validates the same code the device build runs rather than a re-implementation.
    Sweeping the tile edge exercises every tail alignment, including edges the available hardware does
    not have.
    """

    @pytest.fixture(scope="class")
    def source(self):
        return emit_microkernel(_TABLE, _SPEC) + emit_reference_c()

    def _host_lib(self, source, tmp_path, edge):
        import ctypes
        from merlin.llvmlower import toolchain
        if not toolchain.available():
            pytest.skip("needs the pinned clang")
        tmp_path.mkdir(parents=True, exist_ok=True)
        c = tmp_path / "host.c"
        c.write_text(source, encoding="utf-8")
        so = tmp_path / f"host_{edge}.so"
        p = subprocess.run([toolchain.clang(), "-O2", "-shared", "-fPIC", "-DOPU_SCALAR_TILE",
                            f"-DOPU_TILE_EDGE={edge}", str(c), "-o", str(so)],
                           capture_output=True, text=True)
        if p.returncode != 0:
            pytest.fail(f"host build failed:\n{p.stderr[-2000:]}")
        lib = ctypes.CDLL(str(so))
        fn = lib.opu_gemm_i8
        fn.restype = None
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_size_t] * 3
        return fn

    @pytest.mark.parametrize("edge", [4, 8, 16, 32])
    def test_every_corpus_case_is_exact_at_this_tile_edge(self, source, tmp_path, edge):
        import numpy as np

        from merlin.kernels import opu_corpus as OC
        fn = self._host_lib(source, tmp_path / f"e{edge}", edge)
        runnable, _ = OC.select(32)
        assert runnable, "the corpus must not be empty"
        for case in runnable:
            lhs, rhs, bias = case.operands()
            out = np.zeros((case.m, case.n), dtype=np.int32)
            fn(out.ctypes.data, lhs.ctypes.data, rhs.ctypes.data,
               bias.ctypes.data if bias is not None else None, case.m, case.n, case.k)
            # Exact integer equality: there is no rounding to tolerate, and an aggregate similarity
            # gate has previously accepted a kernel over 1000% wrong per element.
            assert (out == OC.reference(lhs, rhs, bias)).all(), f"{case.name} at tile edge {edge}"

    def test_a_shape_larger_than_the_edge_really_is_being_tiled(self, source, tmp_path):
        # Otherwise the sweep above would be passing because everything fit in one tile.
        import numpy as np

        from merlin.kernels import opu_corpus as OC
        fn = self._host_lib(source, tmp_path / "e4", 4)
        rng = np.random.default_rng(11)
        m, n, k = 13, 19, 7          # both extents several tiles wide with a short tail on each
        lhs = rng.integers(-8, 9, size=(k, m), dtype=np.int8)
        rhs = rng.integers(-8, 9, size=(k, n), dtype=np.int8)
        out = np.zeros((m, n), dtype=np.int32)
        fn(out.ctypes.data, lhs.ctypes.data, rhs.ctypes.data, None, m, n, k)
        assert (out == OC.reference(lhs, rhs)).all()

    def test_the_bias_column_offset_is_right_for_every_column_tile(self, source, tmp_path):
        # A bias pointer that was not advanced per column tile would be correct only in the first one.
        import numpy as np

        from merlin.kernels import opu_corpus as OC
        fn = self._host_lib(source, tmp_path / "e4b", 4)
        m, n, k = 6, 11, 3
        lhs = np.zeros((k, m), dtype=np.int8)          # zero operands, so the output IS the bias
        rhs = np.zeros((k, n), dtype=np.int8)
        bias = np.arange(100, 100 + n, dtype=np.int32)
        out = np.zeros((m, n), dtype=np.int32)
        fn(out.ctypes.data, lhs.ctypes.data, rhs.ctypes.data, bias.ctypes.data, m, n, k)
        assert (out == np.broadcast_to(bias, (m, n))).all(), out
        assert (out == OC.reference(lhs, rhs, bias)).all()
