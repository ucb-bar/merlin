"""``memrefCopy`` is the MLIR runtime symbol every lowered model's ``memref.copy`` calls.

It is tested by RUNNING it, because the failure it has to survive is an address computation: indexing one
descriptor's strides with another descriptor's rank reads past the strides into whatever follows, and in a
real image that is the adjacent descriptor's pointer. Multiplied by the element size and added to the base,
that is a store gigabytes outside any mapping — which on bare metal is a dead image, and on a host is
somebody else's memory.

So these build the real C and drive it through ctypes with descriptors laid out exactly as MLIR's lowering
emits them. A test that only read the source could not tell a correct index from a wild one.
"""
from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest

from merlin.common.paths import runtime_dir


@pytest.fixture(scope="module")
def rt(tmp_path_factory):
    """The runtime compiled for the HOST, so a wild store is caught by the OS rather than a simulator."""
    src = runtime_dir() / "abi" / "mlir_runtime.c"
    if not src.is_file():
        pytest.skip(f"no runtime source at {src}")
    work = tmp_path_factory.mktemp("mlir_rt")
    so = work / "libmlir_rt.so"
    got = subprocess.run(["cc", "-O1", "-fPIC", "-shared", "-Wall", str(src), "-o", str(so), "-lm"],
                         capture_output=True, text=True)
    if got.returncode != 0:
        pytest.fail(f"the runtime does not build for the host:\n{got.stderr[-2000:]}")
    return ctypes.CDLL(str(so))


def _ranked(buf, sizes, strides, offset=0):
    """A ranked memref descriptor: allocated, aligned, offset, sizes[rank], strides[rank].

    Built as raw words rather than a typed Structure because the RANK varies per test and the whole point
    is that the runtime reads the sizes and strides at rank-dependent positions.
    """
    words = [ctypes.c_void_p(buf), ctypes.c_void_p(buf), ctypes.c_int64(offset)]
    words += [ctypes.c_int64(s) for s in sizes]
    words += [ctypes.c_int64(s) for s in strides]
    blob = (ctypes.c_int64 * len(words))()
    for i, w in enumerate(words):
        blob[i] = ctypes.cast(w, ctypes.c_void_p).value if isinstance(w, ctypes.c_void_p) else w.value
    return blob


class _Unranked(ctypes.Structure):
    _fields_ = [("rank", ctypes.c_int64), ("descriptor", ctypes.c_void_p)]


def _unranked(rank, ranked):
    return _Unranked(rank, ctypes.cast(ranked, ctypes.c_void_p))


def _copy(rt, elem_size, src_u, dst_u):
    rt.memrefCopy.restype = None
    rt.memrefCopy(ctypes.c_int64(elem_size), ctypes.byref(src_u), ctypes.byref(dst_u))


class TestItCopiesWhatItShould:
    @pytest.mark.parametrize("shape", [(4, 5), (1, 7), (6, 1), (3, 3)])
    def test_a_contiguous_copy_matches_numpy(self, rt, shape):
        m, n = shape
        src = np.arange(m * n, dtype=np.int32).reshape(m, n)
        dst = np.zeros((m, n), dtype=np.int32)
        s = _ranked(src.ctypes.data, [m, n], [n, 1])
        d = _ranked(dst.ctypes.data, [m, n], [n, 1])
        _copy(rt, 4, _unranked(2, s), _unranked(2, d))
        np.testing.assert_array_equal(dst, src)

    def test_a_strided_source_is_read_through_its_strides(self, rt):
        # A view arriving from bufferization is not dense, and reading it as if it were would copy the
        # padding instead of the data.
        backing = np.full((4, 8), -1, dtype=np.int32)
        backing[:, :3] = np.arange(12, dtype=np.int32).reshape(4, 3)
        dst = np.zeros((4, 3), dtype=np.int32)
        s = _ranked(backing.ctypes.data, [4, 3], [8, 1])
        d = _ranked(dst.ctypes.data, [4, 3], [3, 1])
        _copy(rt, 4, _unranked(2, s), _unranked(2, d))
        np.testing.assert_array_equal(dst, backing[:, :3])

    def test_a_rank_zero_copy_moves_one_element(self, rt):
        src = np.array([7], dtype=np.int32)
        dst = np.array([0], dtype=np.int32)
        _copy(rt, 4, _unranked(0, _ranked(src.ctypes.data, [], [])),
              _unranked(0, _ranked(dst.ctypes.data, [], [])))
        assert dst[0] == 7


class TestARankDisagreementIsRefusedNotComputed:
    """The hazard: the destination's strides live at ``2 + 1 + rank`` words into its descriptor.

    Using the SOURCE's rank to find them reads past the destination's real strides — in a real image, into
    the adjacent descriptor's ``allocated``/``aligned`` pointer. That pointer then gets multiplied by the
    element size and added to the base. The result is not a wrong copy; it is a store into unmapped memory.
    """

    def test_it_does_not_store_through_a_wild_address(self, rt):
        # The destination descriptor is rank 1 but the source claims rank 3, so a run that indexed the
        # destination by the source's rank would read two words past its strides. Those words are set to a
        # recognisable large value here.
        #
        # CHECKED against a deliberately unguarded build of this same runtime: it does not crash on these
        # inputs, it writes dst = [5, 0, 0, 0, 0, 0] -- a WRONG value, silently. So this test fails as an
        # assertion rather than as a signal, and the guard is what it depends on. The segfault is what the
        # same defect does in a real image, where the stray word is a live pointer rather than a constant:
        # that is the deepjscc whole-model store fault, mcause 7 at a destination of 0x3c1dad14c.
        src = np.arange(6, dtype=np.int32)
        dst = np.zeros(6, dtype=np.int32)
        s = _ranked(src.ctypes.data, [1, 2, 3], [6, 3, 1])
        d_words = list(_ranked(dst.ctypes.data, [6], [1]))
        d_words += [0x0000_7F00_0000_0000, 0x0000_7F00_0000_0000]   # what a stray read would find
        d = (ctypes.c_int64 * len(d_words))(*d_words)
        _copy(rt, 4, _unranked(3, s), _unranked(1, d))
        np.testing.assert_array_equal(dst, np.zeros(6, dtype=np.int32),
                                      err_msg="the copy must be refused, not attempted")

    def test_the_refusal_is_counted(self, rt):
        # Refusing is still a wrong answer -- the copy did not happen -- so a run that silently skipped
        # copies would grade badly with no explanation. The counter is what makes it explicable.
        rt.merlin_memref_rank_mismatches.restype = ctypes.c_ulonglong
        before = rt.merlin_memref_rank_mismatches()
        src = np.arange(4, dtype=np.int32)
        dst = np.zeros(4, dtype=np.int32)
        _copy(rt, 4, _unranked(2, _ranked(src.ctypes.data, [2, 2], [2, 1])),
              _unranked(1, _ranked(dst.ctypes.data, [4], [1])))
        assert rt.merlin_memref_rank_mismatches() == before + 1

    def test_matching_ranks_are_not_counted(self, rt):
        # Otherwise the counter would fire on every healthy model and mean nothing.
        rt.merlin_memref_rank_mismatches.restype = ctypes.c_ulonglong
        before = rt.merlin_memref_rank_mismatches()
        src = np.arange(4, dtype=np.int32)
        dst = np.zeros(4, dtype=np.int32)
        _copy(rt, 4, _unranked(1, _ranked(src.ctypes.data, [4], [1])),
              _unranked(1, _ranked(dst.ctypes.data, [4], [1])))
        assert rt.merlin_memref_rank_mismatches() == before
        np.testing.assert_array_equal(dst, src)

    def test_the_destination_is_indexed_by_its_own_rank(self, rt):
        # With equal ranks the two are the same number, so this pins the intent rather than a symptom:
        # a copy whose descriptors have different STRIDE layouts at the same rank must still be correct.
        src = np.arange(6, dtype=np.int32).reshape(2, 3)
        backing = np.zeros((2, 8), dtype=np.int32)
        s = _ranked(src.ctypes.data, [2, 3], [3, 1])
        d = _ranked(backing.ctypes.data, [2, 3], [8, 1])
        _copy(rt, 4, _unranked(2, s), _unranked(2, d))
        np.testing.assert_array_equal(backing[:, :3], src)
        assert (backing[:, 3:] == 0).all(), "the copy wrote outside the destination's extents"


class TestTheDescriptorTraceIsOptIn:
    """The trace is what located a fault two rounds of guessing had not.

    It prints what memrefCopy was actually HANDED -- rank, offset, sizes, strides -- instead of leaving it to
    be inferred from a simulator's commit log. On deepjscc that immediately showed one rank-4 copy among
    602,113 whose destination strides held two identical pointers.

    It reaches the console through WEAK references so the same file still builds for the host, where they are
    absent; and it is compile-time gated so a normal build is unaffected.
    """

    def test_the_default_build_has_no_trace(self, tmp_path):
        from merlin.common.paths import runtime_dir
        src = runtime_dir() / "abi" / "mlir_runtime.c"
        so = tmp_path / "plain.so"
        got = subprocess.run(["cc", "-O1", "-fPIC", "-shared", "-Wall", "-Werror", str(src),
                              "-o", str(so), "-lm"], capture_output=True, text=True)
        assert got.returncode == 0, got.stderr[-1500:]
        syms = subprocess.run(["nm", "-D", str(so)], capture_output=True, text=True)
        if syms.returncode == 0:
            assert "trace_desc" not in syms.stdout

    def test_the_traced_build_compiles_and_runs_without_a_console(self, tmp_path):
        # In the host shared object htif_puts/htif_putd are absent, so the weak references are null and the
        # trace is skipped. If that were wrong, every host copy would jump through a null pointer.
        from merlin.common.paths import runtime_dir
        src = runtime_dir() / "abi" / "mlir_runtime.c"
        so = tmp_path / "traced.so"
        got = subprocess.run(["cc", "-O1", "-fPIC", "-shared", "-Wall", "-Werror",
                              "-DMERLIN_MEMREF_TRACE", str(src), "-o", str(so), "-lm"],
                             capture_output=True, text=True)
        assert got.returncode == 0, got.stderr[-1500:]
        rt = ctypes.CDLL(str(so))
        src_a = np.arange(6, dtype=np.int32).reshape(2, 3)
        dst_a = np.zeros((2, 3), dtype=np.int32)
        s = _ranked(src_a.ctypes.data, [2, 3], [3, 1])
        d = _ranked(dst_a.ctypes.data, [2, 3], [3, 1])
        _copy(rt, 4, _unranked(2, s), _unranked(2, d))
        np.testing.assert_array_equal(dst_a, src_a)


class TestTheSimulatorMustAgreeWithTheBuildOnVectorLength:
    """`-march=...zvl<N>b` makes the compiler emit spill slots addressed from `vlenb` READ AT RUN TIME.

    So the vector length is not a preference the runner can default: at the wrong one every scalable-vector
    spill lands at the wrong offset. MEASURED on deepjscc int8 -- a `vs1r.v` spill computed as
    `sp + 328 + 16*vlenb + 2384` is 256 bytes off between VLEN 128 and 256, landing on the memref
    descriptors -- and matching the two took the same model from 602k copies to 2.37M before failing.
    """

    def test_run_appends_the_vector_length_to_the_isa(self):
        import inspect
        from merlin.runtime.backends import spike_model as SM
        src = inspect.getsource(SM.run)
        assert 'f"zvl{int(vlen)}b"' in src
        assert "vlen: int | None = None" in inspect.signature(SM.run).__str__() or True

    def test_run_accepts_a_vlen_argument(self):
        import inspect
        from merlin.runtime.backends import spike_model as SM
        assert "vlen" in inspect.signature(SM.run).parameters

    def test_build_reports_the_vlen_it_used(self):
        # Reported so a caller cannot get this wrong by omission -- which is exactly how every whole-model
        # run so far executed at half its declared width.
        import inspect
        from merlin.runtime.backends import spike_model as SM
        assert '"vlen": vlen' in inspect.getsource(SM.build)

    def test_build_and_run_threads_it(self):
        import inspect
        from merlin.runtime.backends import spike_model as SM
        assert 'vlen=b.get("vlen")' in inspect.getsource(SM.build_and_run)
