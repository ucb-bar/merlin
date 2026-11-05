"""Execute complete products through emitted batch views, including plane reuse."""

import ctypes
import itertools
import shutil
import subprocess
import sys

import numpy as np
import pytest

from merlin.compile.scheduling.batched_dense import BatchedDenseProblem, BatchMatrixView
from merlin.compile.scheduling.dense_contracts import DenseLogicalPointerABI
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.logical_borrowed_abi import LogicalBorrowedABI
from merlin.llvmlower.logical_borrowed_batch_abi import emit_borrowed_batch_translation_unit


def capability():
    return LogicalBorrowedABI(
        DenseLogicalPointerABI("dense_row_major", (1, 1, 4),
                               ("read_only", "read_only", "fully_written"),
                               "logical_elements", "synchronous"),
        BorrowedBufferEffects("independent synchronous test primitive", True, True, True),
        "independent native integer format", 4096,
    )


def descriptor(array):
    rank = array.ndim

    class Descriptor(ctypes.Structure):
        _fields_ = [("allocated", ctypes.c_void_p), ("aligned", ctypes.c_void_p),
                    ("offset", ctypes.c_int64), ("sizes", ctypes.c_int64 * rank),
                    ("strides", ctypes.c_int64 * rank)]

    return Descriptor(array.ctypes.data, array.ctypes.data, 0,
                      (ctypes.c_int64 * rank)(*array.shape),
                      (ctypes.c_int64 * rank)(*(s // array.itemsize for s in array.strides)))


CASES = (
    BatchedDenseProblem((3,), BatchMatrixView((3, 2, 7), (0,)),
                        BatchMatrixView((3, 7, 5), (0,))),
    # All four independent plane products for every head, without duplicated inputs.
    BatchedDenseProblem((2, 2, 5), BatchMatrixView((2, 5, 3, 7), (0, 2)),
                        BatchMatrixView((2, 5, 7, 4), (1, 2))),
    # Shared RHS, transposed physical LHS, ragged matrix extents.
    BatchedDenseProblem((2, 3), BatchMatrixView((2, 3, 7, 1), (0, 1), True),
                        BatchMatrixView((7, 5), ())),
)


def primitive(problem):
    return f"""
#include <stdlib.h>
static uint64_t calls;
static int forbid_kernel;
uint64_t count(void){{return calls;}}
void kernel(void *a,void *b,void *out){{
 if(forbid_kernel)_Exit(77);calls++;
 for(int i=0;i<{problem.m};i++)for(int j=0;j<{problem.n};j++){{
  uint32_t acc=0;
  for(int k=0;k<{problem.k};k++)
   acc+=(uint32_t)((int8_t*)a)[i*{problem.k}+k]*(uint32_t)((int8_t*)b)[k*{problem.n}+j];
  ((uint32_t*)out)[i*{problem.n}+j]=acc;
 }}
}}
"""


@pytest.mark.parametrize("problem", CASES)
@pytest.mark.parametrize("borrow", (False, True))
def test_actual_batch_bridge_all_elements_and_plane_combinations(problem, borrow, tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C toolchain required")
    unit = emit_borrowed_batch_translation_unit(
        capability(), problem, symbol="batch", kernel="kernel", borrow_contiguous=borrow,
    )
    source = tmp_path / "batch.c"
    library = tmp_path / f"batch_{problem.products}_{problem.m}_{problem.n}_{borrow}.so"
    source.write_text(unit.text + primitive(problem))
    subprocess.run([cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
                    "-Wno-misleading-indentation", "-shared", "-fPIC", source, "-o", library], check=True)
    loaded = ctypes.CDLL(str(library))
    rng = np.random.default_rng(37)
    a = rng.integers(-7, 8, problem.lhs.shape, dtype=np.int8)
    b = rng.integers(-7, 8, problem.rhs.shape, dtype=np.int8)
    # A real noncontiguous writable view exercises global injectivity and fallback.
    storage = np.full((*problem.output_shape[:-1], problem.n * 2), -777, np.int32)
    out = storage[..., ::2]
    args = tuple(descriptor(v) for v in (a, b, out))
    loaded.batch(*(ctypes.byref(v) for v in args))
    expected = np.empty(problem.output_shape, np.int32)
    for at in itertools.product(*(range(n) for n in problem.batch_shape)):
        aa = a[tuple(at[i] for i in problem.lhs.batch_axes)]
        bb = b[tuple(at[i] for i in problem.rhs.batch_axes)]
        if problem.lhs.transposed:
            aa = aa.T
        if problem.rhs.transposed:
            bb = bb.T
        expected[at] = aa.astype(np.int64) @ bb.astype(np.int64)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(storage[..., 1::2], -777)
    assert loaded.count() == problem.products
    # Only source rank changes code size; no static expansion by head count.
    assert unit.text.count("batch_matrix(&s0,&s1,&s2)") == 1


def test_cross_batch_alias_refuses_before_any_primitive(tmp_path):
    problem = BatchedDenseProblem((3,), BatchMatrixView((3, 2, 2), (0,)),
                                 BatchMatrixView((3, 2, 2), (0,)))
    unit = emit_borrowed_batch_translation_unit(capability(), problem, symbol="batch", kernel="kernel")
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("native C toolchain required")
    source, library = tmp_path / "bad.c", tmp_path / "bad.so"
    source.write_text(unit.text + primitive(problem) + """
void bad(void){
 int32_t shared[64]={0};int8_t rhs[12]={0};
 merlin_lmemref_3d a={shared,shared,0,{3,2,2},{4,2,1}};
 merlin_lmemref_3d b={rhs,rhs,0,{3,2,2},{4,2,1}};
 /* Output starts at a later input batch; a per-slice-only check is insufficient. */
 merlin_lmemref_3d c={shared,shared,2,{3,2,2},{4,2,1}};
 forbid_kernel=1;batch(&a,&b,&c);
}
""")
    subprocess.run([cc, "-std=c11", "-O2", "-shared", "-fPIC", source, "-o", library], check=True)
    result = subprocess.run([sys.executable, "-c", f"import ctypes;ctypes.CDLL({str(library)!r}).bad()"])
    assert result.returncode < 0, "alias must trap before the primitive (which exits77)"


def test_invalid_projection_and_no_product_elision():
    with pytest.raises(ValueError, match="extent"):
        BatchedDenseProblem((3,), BatchMatrixView((2, 2, 7), (0,)), BatchMatrixView((7, 5), ()))
    with pytest.raises(ValueError, match="distinct"):
        BatchMatrixView((2, 2, 3, 7), (0, 0))
    assert CASES[1].products == 2 * 2 * 5
