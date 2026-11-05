"""The ordinary tensor host links and executes a checked runtime head loop."""

import hashlib
import subprocess

import numpy as np
from test_head_batching import source

from merlin.compile.scheduling.dense_contracts import DenseLogicalPointerABI
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.borrowed_effects import BorrowedBufferEffects
from merlin.llvmlower.integer_batched_matmul_calls import lower_integer_batched_matmul_program
from merlin.llvmlower.logical_borrowed_abi import LogicalBorrowedABI
from merlin.llvmlower.target_data_layout import _query
from merlin.llvmlower.toolchain import clang


def test_ordinary_source_host_and_all_plane_products(tmp_path):
    cc = clang()
    effects = BorrowedBufferEffects("independent native synchronous matrix fixture", True, True, True)
    capability = LogicalBorrowedABI(
        DenseLogicalPointerABI("dense_row_major", (1, 1, 4),
                               ("read_only", "read_only", "fully_written"),
                               "logical_elements", "synchronous"),
        effects, "independent native integer format", 4096,
    )

    def emit(call):
        path = tmp_path / f"{call.symbol}.c"
        path.write_text("#include <stdint.h>\n" + f"""
void {call.kernel_symbol}(void *aa,void *bb,void *yy){{
 int8_t *a=aa,*b=bb;uint32_t*y=yy;
 for(int m=0;m<{call.m};m++)for(int n=0;n<{call.n};n++){{
  uint32_t s=0;for(int k=0;k<{call.k};k++)
   s+=(uint32_t)(int32_t)a[m*{call.k}+k]*(uint32_t)(int32_t)b[k*{call.n}+n];
  y[m*{call.n}+n]=s;
 }}
}}
""")
        output = path.with_suffix(".ll")
        subprocess.run([cc, "-O2", "-fPIC", "-S", "-emit-llvm", path, "-o", output],
                       check=True, capture_output=True)
        return output.read_text(), ((path, hashlib.sha256(path.read_bytes()).hexdigest()),)

    program = lower_integer_batched_matmul_program(
        source(h=3, m=2, k=7, n=5, planes=2), tmp_path / "ordinary",
        select=lambda call: True, emit_matrix_kernel=emit,
        logical_abi=capability, data_layout=_query(cc, ("-O2", "-fPIC")), refold_heads=True,
    )
    library = tmp_path / "ordinary.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", program.host.ll_path,
                    *program.kernels, *program.bridges, "-o", library], check=True, capture_output=True)
    rng = np.random.default_rng(19)
    inputs = [rng.integers(-8, 9, shape, dtype=np.int8)
              for shape in ((3, 2, 7), (3, 2, 7), (3, 7, 5), (3, 7, 5))]
    expected = [a[h].astype(np.int32) @ b[h].astype(np.int32)
                for a in inputs[:2] for b in inputs[2:] for h in range(3)]
    outputs = [np.zeros_like(value) for value in expected]
    HostModel.load(str(library))([(x.ctypes.data, x.shape) for x in (*inputs, *outputs)])
    for actual, value in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(actual, value)
    assert len(program.preparation.calls) == 4
    assert sum(call.problem.products for call in program.preparation.calls) == 12
    # One static inner primitive call in each counted bridge; not twelve
    # statically outlined products, nor a reduction in accelerator work.
    for call, path in zip(program.preparation.calls, program.bridges, strict=True):
        assert path.read_text().count(call.kernel_symbol + "(") == 2  # declaration and inner call
