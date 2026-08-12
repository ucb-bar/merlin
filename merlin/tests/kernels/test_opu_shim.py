"""The shim is asserted by RUNNING it, not by reading the C it generates.

Everything the shim does is a translation between two conventions — descriptor layout, operand layout,
extents, alignment — and a translation that is subtly wrong reads perfectly. So these tests compile the
emitted translation unit with the scalar stand-in for the unit, call the generated entry points through
real memref descriptors from ctypes, and compare against numpy. What that leaves unproven is the datapath
itself, which is exactly what the RTL certification covers and this cannot.

The host build defines ``OPU_SCALAR_TILE``, so no matrix instruction executes here. That is the point: the
tiling loop, the K-major pack, the stride checks and the fallback are all target-independent code, and
proving them on a host is what makes a failure on the device attributable to the datapath.
"""
from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.kernels.opu_kernel import KernelSpec
from merlin.llvmlower.opu_shim import (DEFAULT_SCRATCH_BYTES, emit_translation_unit,
                                       scratch_bytes_for)


class _Enc:
    """A derived-encoding stand-in with the interface the emitter uses.

    Values are arbitrary: under ``OPU_SCALAR_TILE`` none of these words is executed, and the tests that
    assert the words themselves are real live in ``test_opu_kernel``/``test_opu_isa``.
    """

    def __init__(self, opcode: int, funct3: int, funct6: int):
        self.opcode, self.funct3, self.funct6 = opcode, funct3, funct6

    @property
    def funct7(self) -> int:
        return (self.funct6 << 1) | 1

    def insn_r(self, rd: str, rs1: str, rs2: str) -> str:
        return f".insn r {self.opcode:#x}, {self.funct3:#x}, {self.funct7:#x}, {rd}, {rs1}, {rs2}"


_TABLE = {"ACC": _Enc(0x57, 2, 40), "BCAST": _Enc(0x57, 6, 44), "READOUT": _Enc(0x57, 6, 46)}
_SPEC = KernelSpec(accumulate="ACC", broadcast="BCAST", readout="READOUT")

#: The alignment a caller must honour. 16 is what a dLen=128 datapath derives; the value is a parameter
#: here because the shim takes it as one, and one test varies it to show it is not baked in.
_ALIGN = 16

#: Tile edge the host stand-in sweeps. The device reads its own from the hardware; on a host there is
#: nothing to ask, so it is supplied and swept to exercise the tail paths at several alignments.
_EDGES = (4, 8, 16, 32)


class _Memref2D(ctypes.Structure):
    _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                ("offset", ctypes.c_ssize_t),
                ("sizes", ctypes.c_ssize_t * 2), ("strides", ctypes.c_ssize_t * 2)]


def _unpack(buf, shape, strides):
    """One memref argument as the seven scalars the lowered convention passes.

    Written out rather than hidden in a helper struct because the SEVEN-SCALAR form is the thing under
    test: a shim that read the extents from the wrong position would still link and would still run.
    """
    base = ctypes.cast(buf, ctypes.c_void_p)
    return [base, base, ctypes.c_ssize_t(0),
            ctypes.c_ssize_t(shape[0]), ctypes.c_ssize_t(shape[1]),
            ctypes.c_ssize_t(strides[0]), ctypes.c_ssize_t(strides[1])]


def _build(signatures, tmp_path: Path, *, edge: int, align: int = _ALIGN,
           scratch_bytes: int | None = None) -> ctypes.CDLL:
    src = emit_translation_unit(_TABLE, signatures, spec=_SPEC, alignment_bytes=align,
                               scratch_bytes=scratch_bytes)
    tmp_path.mkdir(parents=True, exist_ok=True)
    c = tmp_path / "opu_shim.c"
    c.write_text(str(src), encoding="utf-8")
    so = tmp_path / "opu_shim.so"
    cmd = ["cc", "-O1", "-fPIC", "-shared", "-Wall", "-Werror",
           "-DOPU_SCALAR_TILE", f"-DOPU_TILE_EDGE={edge}", str(c), "-o", str(so)]
    got = subprocess.run(cmd, capture_output=True, text=True)
    if got.returncode != 0:
        pytest.fail(f"the emitted translation unit did not compile cleanly:\n{got.stderr[-3000:]}")
    return ctypes.CDLL(str(so))


def _call(lib, symbol: str, a: np.ndarray, b: np.ndarray, *, shape=None,
          a_strides=None, b_strides=None) -> np.ndarray:
    """Run one entry point on real buffers and return what it wrote into C.

    ``shape`` is the LOGICAL ``(m, n, k)``, which is what the descriptors advertise; ``a``/``b`` are the
    backing buffers, which may be larger. Keeping the two separate is the only way to hand the shim a
    strided view — deriving the extents from the buffers' own shapes would make every view look dense and
    the stride tests would pass without testing anything.
    """
    m, k = (shape[0], shape[2]) if shape else a.shape
    n = shape[1] if shape else b.shape[1]
    c = np.zeros((m, n), dtype=np.int32)
    fn = getattr(lib, symbol)
    fn.restype = _Memref2D
    fn.argtypes = None
    args = [*_unpack(a.ctypes.data_as(ctypes.c_void_p), (m, k), a_strides or (k, 1)),
            *_unpack(b.ctypes.data_as(ctypes.c_void_p), (k, n), b_strides or (n, 1)),
            *_unpack(c.ctypes.data_as(ctypes.c_void_p), (m, n), (n, 1))]
    fn(*args)
    return c


def _operands(m: int, n: int, k: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=(m, k), dtype=np.int8)
    b = rng.integers(-128, 128, size=(k, n), dtype=np.int8)
    return a, b


def _expected(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """int32 with wraparound, which is what the accumulator does."""
    return (a.astype(np.int64) @ b.astype(np.int64)).astype(np.int32)


# --------------------------------------------------------------------------------------------------
# The thing that matters: does the compiled shim compute the contraction?
# --------------------------------------------------------------------------------------------------


class TestComputesTheContraction:
    @pytest.mark.parametrize("edge", _EDGES)
    def test_it_agrees_with_numpy_at_every_tile_edge(self, tmp_path, edge):
        # Sweeping the edge is how the tail paths get exercised at every alignment rather than only at
        # the one the device happens to have.
        m, n, k = 12, 20, 9
        lib = _build({"merlin_opu_gemm_i8_0": (m, n, k)}, tmp_path / f"e{edge}", edge=edge)
        a, b = _operands(m, n, k)
        got = _call(lib, "merlin_opu_gemm_i8_0", a, b)
        np.testing.assert_array_equal(got, _expected(a, b))

    @pytest.mark.parametrize("m,n,k", [(1, 16, 8), (16, 1, 8), (1, 1, 1), (17, 33, 5), (32, 32, 1)])
    def test_it_agrees_on_the_shapes_the_corpus_protects(self, tmp_path, m, n, k):
        # The same degenerate and off-by-one extents the frozen corpus keeps: a vecmat, a matvec, a
        # single element, both extents one past a tile, and a reduction of length one.
        lib = _build({"merlin_opu_gemm_i8_0": (m, n, k)}, tmp_path, edge=16)
        a, b = _operands(m, n, k)
        np.testing.assert_array_equal(_call(lib, "merlin_opu_gemm_i8_0", a, b), _expected(a, b))

    def test_an_odd_reduction_length_is_not_dropped_by_the_unrolled_loop(self, tmp_path):
        # The reduction loop is unrolled by two with the odd step peeled (the register rotation the RTL
        # hazard forces). An off-by-one in the peel loses the last K-step, which is a WRONG ANSWER that
        # only shows up for odd K.
        lib = _build({"merlin_opu_gemm_i8_0": (8, 8, 7)}, tmp_path, edge=8)
        for k in (1, 2, 3, 7):
            a, b = _operands(8, 8, k)
            np.testing.assert_array_equal(_call(lib, "merlin_opu_gemm_i8_0", a, b), _expected(a, b))

    def test_spectformers_real_signatures_agree(self, tmp_path):
        # The five signatures the rewrite actually mints for spectformer, at reduced K so the test is
        # quick -- the extents that matter here are M and N, which drive the tiling and the pack.
        sigs = {"merlin_opu_gemm_i8_0": (256, 196, 8), "merlin_opu_gemm_i8_1": (196, 1024, 8),
                "merlin_opu_gemm_i8_2": (196, 256, 8), "merlin_opu_gemm_i8_3": (196, 768, 8),
                "merlin_opu_gemm_i8_4": (1, 1000, 8)}
        lib = _build(sigs, tmp_path, edge=32)
        for sym, (m, n, k) in sigs.items():
            a, b = _operands(m, n, k)
            np.testing.assert_array_equal(_call(lib, sym, a, b), _expected(a, b),
                                          err_msg=f"{sym} m={m} n={n} k={k}")


class TestTheCallingConvention:
    def test_every_signature_gets_its_own_linkable_symbol(self, tmp_path):
        # MLIR function types are monomorphic, so a missing symbol is a link error at whole-model build
        # time -- far from here and hard to attribute.
        sigs = {f"merlin_opu_gemm_i8_{i}": (8, 8, 4) for i in range(4)}
        lib = _build(sigs, tmp_path, edge=8)
        for sym in sigs:
            assert getattr(lib, sym) is not None

    def test_the_extents_come_from_the_descriptor_not_from_the_signature(self, tmp_path):
        # A shim that trusted its generated-for extents would compute the wrong shape here and, worse,
        # would read past the operands it was handed.
        lib = _build({"merlin_opu_gemm_i8_0": (64, 64, 64)}, tmp_path, edge=16)
        a, b = _operands(5, 7, 3)
        np.testing.assert_array_equal(_call(lib, "merlin_opu_gemm_i8_0", a, b), _expected(a, b))

    def test_the_returned_descriptor_describes_the_output(self, tmp_path):
        # The call site consumes the returned struct, so a wrong field there corrupts every later use of
        # the result even though the buffer itself is correct.
        lib = _build({"merlin_opu_gemm_i8_0": (6, 10, 4)}, tmp_path, edge=8)
        a, b = _operands(6, 10, 4)
        c = np.zeros((6, 10), dtype=np.int32)
        fn = lib.merlin_opu_gemm_i8_0
        fn.restype = _Memref2D
        got = fn(*_unpack(a.ctypes.data_as(ctypes.c_void_p), (6, 4), (4, 1)),
                 *_unpack(b.ctypes.data_as(ctypes.c_void_p), (4, 10), (10, 1)),
                 *_unpack(c.ctypes.data_as(ctypes.c_void_p), (6, 10), (10, 1)))
        assert (got.sizes[0], got.sizes[1]) == (6, 10)
        assert (got.strides[0], got.strides[1]) == (10, 1)
        assert got.aligned == c.ctypes.data


class TestTheLeftOperandPack:
    def test_a_non_contiguous_left_operand_is_read_through_its_strides(self, tmp_path):
        # The pack is the only place A's strides are honoured, so a view arriving from bufferization must
        # not be read as if it were dense. A dense read of this buffer picks up the padding columns and
        # produces a completely different answer, so this cannot pass by accident.
        lib = _build({"merlin_opu_gemm_i8_0": (8, 8, 4)}, tmp_path, edge=8)
        a, b = _operands(8, 8, 4)
        backing = np.full((8, 16), 99, dtype=np.int8)     # row stride 16 for a K=4 operand
        backing[:, :4] = a
        got = _call(lib, "merlin_opu_gemm_i8_0", backing, b, shape=(8, 8, 4), a_strides=(16, 1))
        np.testing.assert_array_equal(got, _expected(a, b))
        # A strided LHS is packed, not declined: the pack reads through strides by construction.
        assert lib.merlin_opu_fallbacks() == 0

    def test_a_column_major_left_operand_is_packed_correctly(self, tmp_path):
        # The transpose is the whole reason this module exists, so it has to be right for a left operand
        # that is ALREADY K-major: strides (1, M) rather than (K, 1). Getting the two mixed up transposes
        # twice and computes A^T @ B.
        lib = _build({"merlin_opu_gemm_i8_0": (6, 8, 5)}, tmp_path, edge=8)
        a, b = _operands(6, 8, 5)
        colmajor = np.asfortranarray(a)                   # same values, strides (1, 6)
        got = _call(lib, "merlin_opu_gemm_i8_0", colmajor, b, shape=(6, 8, 5), a_strides=(1, 6))
        np.testing.assert_array_equal(got, _expected(a, b))

    def test_a_left_operand_too_large_for_the_scratch_still_computes(self, tmp_path):
        # The scratch is static and sized at generation time. Overrunning it would be a memory-safety bug
        # in generated code, so the shim must decline instead -- and declining must still be CORRECT.
        lib = _build({"merlin_opu_gemm_i8_0": (32, 8, 32)}, tmp_path, edge=8, scratch_bytes=64)
        a, b = _operands(32, 8, 32)          # needs 1024 bytes, has 64
        np.testing.assert_array_equal(_call(lib, "merlin_opu_gemm_i8_0", a, b), _expected(a, b))
        assert lib.merlin_opu_fallbacks() >= 1, "a shape that cannot be packed must be counted, not hidden"

    def test_the_scratch_is_sized_to_the_largest_signature(self):
        sigs = {"a": (196, 1024, 256), "b": (256, 196, 768), "c": (196, 256, 1024)}
        assert scratch_bytes_for(sigs) == 196 * 1024        # 200704, the largest M*K
        assert scratch_bytes_for({}) == DEFAULT_SCRATCH_BYTES


class TestTheFallbackIsCorrectAndCounted:
    def test_a_strided_right_operand_falls_back_rather_than_misreading_it(self, tmp_path):
        # The kernel indexes b[kk*n + j], so a view with any other row stride would be read as dense and
        # would silently mix rows. It must fall back, and the fallback must be right.
        lib = _build({"merlin_opu_gemm_i8_0": (8, 8, 4)}, tmp_path, edge=8)
        a, b = _operands(8, 8, 4)
        backing = np.full((4, 32), 99, dtype=np.int8)      # row stride 32 for an N=8 operand
        backing[:, :8] = b
        got = _call(lib, "merlin_opu_gemm_i8_0", a, backing, shape=(8, 8, 4), b_strides=(32, 1))
        np.testing.assert_array_equal(got, _expected(a, b))
        assert lib.merlin_opu_fallbacks() == 1
        assert lib.merlin_opu_calls() == 1

    def test_the_counters_separate_the_fast_path_from_the_fallback(self, tmp_path):
        # Without this split a systematically-declined model grades perfectly and reports a speedup it
        # never got, which is the failure these counters exist to make visible.
        lib = _build({"merlin_opu_gemm_i8_0": (8, 8, 4)}, tmp_path, edge=8)
        a, b = _operands(8, 8, 4)
        _call(lib, "merlin_opu_gemm_i8_0", a, b)
        assert (lib.merlin_opu_calls(), lib.merlin_opu_fallbacks()) == (1, 0)
        backing = np.full((4, 32), 99, dtype=np.int8)
        backing[:, :8] = b
        _call(lib, "merlin_opu_gemm_i8_0", a, backing, shape=(8, 8, 4), b_strides=(32, 1))
        assert (lib.merlin_opu_calls(), lib.merlin_opu_fallbacks()) == (2, 1)

    def test_an_empty_reduction_writes_the_zero_init(self, tmp_path):
        # K=0 means the result is C_init, which the rewrite guarantees is zero. Leaving C untouched would
        # be right only by accident, since the buffer is not required to arrive zeroed.
        lib = _build({"merlin_opu_gemm_i8_0": (4, 4, 4)}, tmp_path, edge=4)
        c = np.full((4, 4), 99, dtype=np.int32)
        a = np.zeros((4, 0), dtype=np.int8)
        b = np.zeros((0, 4), dtype=np.int8)
        fn = lib.merlin_opu_gemm_i8_0
        fn.restype = _Memref2D
        fn(*_unpack(a.ctypes.data_as(ctypes.c_void_p), (4, 0), (0, 1)),
           *_unpack(b.ctypes.data_as(ctypes.c_void_p), (0, 4), (4, 1)),
           *_unpack(c.ctypes.data_as(ctypes.c_void_p), (4, 4), (4, 1)))
        np.testing.assert_array_equal(c, np.zeros((4, 4), dtype=np.int32))


class TestTheDeviceBuild:
    """The host build cannot see the matrix instructions at all, so it cannot tell whether they survived.

    These compile the SAME translation unit for the device — no ``OPU_SCALAR_TILE`` — and read the
    instruction stream back out of the object. That is the check the host tests structurally cannot make.
    """

    @pytest.fixture
    def device_object(self, tmp_path):
        from merlin.llvmlower import toolchain
        if not toolchain.available():
            pytest.skip("needs the pinned clang")
        src = emit_translation_unit(_TABLE, {"merlin_opu_gemm_i8_0": (196, 1024, 256),
                                             "merlin_opu_gemm_i8_1": (256, 196, 768)},
                                    spec=_SPEC, alignment_bytes=_ALIGN)
        tmp_path.mkdir(parents=True, exist_ok=True)
        c = tmp_path / "shim_dev.c"
        c.write_text(str(src), encoding="utf-8")
        o = tmp_path / "shim_dev.o"
        got = subprocess.run([toolchain.clang(), "--target=riscv64-unknown-elf", "-march=rv64gcv",
                              "-mabi=lp64d", "-O2", "-Wall", "-Werror", "-c", str(c), "-o", str(o)],
                             capture_output=True, text=True)
        if got.returncode != 0:
            pytest.fail(f"the emitted unit does not build for the device:\n{got.stderr[-3000:]}")
        return o

    def test_the_matrix_instructions_reach_the_object(self, device_object):
        from merlin.kernels.decode import opu as OA
        audit = OA.audit_object(device_object, _TABLE)
        assert audit.counts["ACC"] >= 1 and audit.counts["READOUT"] >= 1

    def test_no_matrix_instruction_runs_unconfigured_or_unaccounted_for(self, device_object):
        # The historical failure: an operand load left running on a vector length set for something else.
        # The audit is what detects it, and it has to hold for the code SHIPPED in the model, not only for
        # the kernel compiled on its own.
        from merlin.kernels.decode import opu as OA
        audit = OA.audit_object(device_object, _TABLE)
        assert audit.unconfigured == () and audit.unaccounted == ()

    def test_every_entry_point_is_exported(self, device_object):
        got = subprocess.run(["nm", "-g", str(device_object)], capture_output=True, text=True)
        if got.returncode != 0:
            pytest.skip("nm unavailable")
        for sym in ("merlin_opu_gemm_i8_0", "merlin_opu_gemm_i8_1", "merlin_opu_shim_gemm_i8",
                    "merlin_opu_calls", "merlin_opu_fallbacks"):
            assert sym in got.stdout, f"{sym} is not exported, so the model would not link"


class TestRefusesToGuess:
    def test_an_unresolved_encoding_derivation_is_not_emitted_from(self):
        with pytest.raises(ValueError, match="cross-check"):
            emit_translation_unit(_TABLE, {"s": (4, 4, 4)}, spec=_SPEC, alignment_bytes=_ALIGN,
                                  derivation_ok=False)

    def test_a_nonsense_alignment_is_refused(self):
        with pytest.raises(ValueError, match="alignment"):
            emit_translation_unit(_TABLE, {"s": (4, 4, 4)}, spec=_SPEC, alignment_bytes=0)

    def test_the_alignment_is_a_parameter_not_a_literal(self, tmp_path):
        # A baked alignment would be right on one datapath width and wrong on every other.
        a32 = emit_translation_unit(_TABLE, {"s": (4, 4, 4)}, spec=_SPEC, alignment_bytes=32)
        a16 = emit_translation_unit(_TABLE, {"s": (4, 4, 4)}, spec=_SPEC, alignment_bytes=16)
        assert "#define MERLIN_OPU_ALIGN 32" in a32
        assert "#define MERLIN_OPU_ALIGN 16" in a16

    def test_the_emitted_unit_carries_the_kernel_it_was_certified_with(self, tmp_path):
        # The microkernel must be the emitter's output, not a transcription: the accumulate word has to
        # move when the encoding table does.
        base = emit_translation_unit(_TABLE, {"s": (4, 4, 4)}, spec=_SPEC, alignment_bytes=_ALIGN)
        shifted = emit_translation_unit({k: _Enc(v.opcode, v.funct3, v.funct6 + 2)
                                         for k, v in _TABLE.items()},
                                        {"s": (4, 4, 4)}, spec=_SPEC, alignment_bytes=_ALIGN)
        assert "0x51" in base and "0x51" not in shifted
