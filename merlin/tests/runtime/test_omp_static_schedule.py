"""The OpenMP static worksharing split used by the Zephyr multicore shim.

This is the piece of ``merlin/runtime/c/libomp_zephyr.c`` that fails SILENTLY: an off-by-one
here means two harts write the same output rows (or a row is written by none), which looks
like a perfectly successful run and produces wrong numbers. The generated IR hands the split
a gtid from ``__kmpc_global_thread_num`` and trusts the bounds it writes back, so nothing
downstream re-checks them.

``omp_static_schedule.h`` is deliberately free of Zephyr headers so it can be compiled with
the host compiler and checked exhaustively here — no board, no simulator, no Zephyr SDK.
The invariants asserted are the ones that make a parallel loop equivalent to the serial one:

  * DISJOINT   — no iteration is claimed by two threads
  * COMPLETE   — every iteration is claimed by exactly one thread
  * LASTITER   — exactly one thread is told it owns the final iteration
  * BALANCED   — block sizes differ by at most one (no hart left idle while another works)
"""
from __future__ import annotations

import ctypes
import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir

CC = shutil.which("cc") or shutil.which("gcc")

_HARNESS = r"""
#include "omp_static_schedule.h"

/* Returns iteration count; writes back the thread's bounds. */
long long split(int tid, int nth, long long *lo, long long *up,
                long long *stride, long long incr, int *lastiter) {
  return (long long)merlin_omp_static_split(tid, nth, (int64_t *)lo, (int64_t *)up,
                                            (int64_t *)stride, incr, (int32_t *)lastiter);
}
"""


@pytest.fixture(scope="module")
def split_fn(tmp_path_factory):
    if CC is None:
        pytest.skip("no host C compiler")
    d = tmp_path_factory.mktemp("ompsched")
    (d / "harness.c").write_text(_HARNESS)
    so = d / "libsched.so"
    proc = subprocess.run(
        [CC, "-O2", "-fPIC", "-shared", "-I", str(runtime_dir() / "c"),
         str(d / "harness.c"), "-o", str(so)],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"harness build failed:\n{proc.stderr}"
    lib = ctypes.CDLL(str(so))
    lib.split.restype = ctypes.c_longlong
    lib.split.argtypes = [ctypes.c_int, ctypes.c_int,
                          ctypes.POINTER(ctypes.c_longlong),
                          ctypes.POINTER(ctypes.c_longlong),
                          ctypes.POINTER(ctypes.c_longlong),
                          ctypes.c_longlong, ctypes.POINTER(ctypes.c_int)]
    return lib.split


def _partition(split, nth: int, lo: int, up: int, incr: int = 1):
    """Run the split for every thread; return (per-thread iteration lists, lastiter flags)."""
    ranges, lasts = [], []
    for tid in range(nth):
        l = ctypes.c_longlong(lo)
        u = ctypes.c_longlong(up)
        s = ctypes.c_longlong(0)
        last = ctypes.c_int(0)
        n = split(tid, nth, ctypes.byref(l), ctypes.byref(u), ctypes.byref(s),
                  incr, ctypes.byref(last))
        # exactly the loop the generated code runs with these bounds
        iters = list(range(l.value, u.value + 1, incr)) if incr > 0 else []
        assert len(iters) == n, f"tid {tid}: reported {n} iters, bounds give {len(iters)}"
        ranges.append(iters)
        lasts.append(bool(last.value))
    return ranges, lasts


@pytest.mark.parametrize("nth", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("trip", [0, 1, 2, 3, 7, 8, 9, 64, 512, 2048, 5632])
def test_partition_is_disjoint_and_complete(split_fn, nth, trip):
    """The union of the threads' slices is exactly the serial iteration space."""
    lo, up = 0, trip - 1
    ranges, lasts = _partition(split_fn, nth, lo, up)
    seen = [i for r in ranges for i in r]
    expected = list(range(lo, up + 1))
    assert sorted(seen) == expected, f"nth={nth} trip={trip}: not a partition"
    assert len(seen) == len(set(seen)), "an iteration was claimed by two threads"
    if trip > 0:
        assert sum(lasts) == 1, f"lastiter set {sum(lasts)} times (want exactly 1)"
        owner = lasts.index(True)
        assert ranges[owner] and ranges[owner][-1] == up


@pytest.mark.parametrize("nth,trip", [(4, 8), (4, 2048), (4, 5632), (3, 7), (8, 5632)])
def test_partition_is_balanced(split_fn, nth, trip):
    """Blocks differ by at most one iteration — no hart idles while another does double."""
    ranges, _ = _partition(split_fn, nth, 0, trip - 1)
    sizes = [len(r) for r in ranges]
    assert max(sizes) - min(sizes) <= 1, f"unbalanced split {sizes}"


def test_tinyllama_shapes_split_cleanly(split_fn):
    """The real reason matmul is parallelized over N rather than M.

    TinyLlama int8 is M=8, N=2048/5632. Splitting N over 4 harts keeps each hart's slice a
    clean multiple of the NR=8 micro-kernel tile; splitting M=8 would give 2 rows per hart,
    BELOW the MR=4 register block, so every hart would run a masked short tile.
    """
    for n in (2048, 5632):
        ranges, _ = _partition(split_fn, 4, 0, n - 1)
        sizes = [len(r) for r in ranges]
        assert sizes == [n // 4] * 4, f"N={n} did not split evenly: {sizes}"
        assert all(s % 8 == 0 for s in sizes), f"N={n} slices not NR=8 multiples: {sizes}"
    # M=8 over 4 harts: 2 rows each — a legal partition, but below MR=4. Documented here so
    # the reason the default is N (not M) stays visible if someone flips it.
    m_sizes = [len(r) for r in _partition(split_fn, 4, 0, 7)[0]]
    assert m_sizes == [2, 2, 2, 2] and all(s < 4 for s in m_sizes)


def test_more_threads_than_work_leaves_extra_threads_empty(split_fn):
    """8 harts on a 3-iteration loop: 3 threads get one each, 5 get a ZERO-trip range."""
    ranges, lasts = _partition(split_fn, 8, 0, 2)
    assert [len(r) for r in ranges] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert sum(lasts) == 1


def test_nonunit_increment(split_fn):
    ranges, _ = _partition(split_fn, 4, 0, 30, incr=3)      # 0,3,...,30 -> 11 iterations
    seen = [i for r in ranges for i in r]
    assert sorted(seen) == list(range(0, 31, 3))
    assert len(seen) == 11
