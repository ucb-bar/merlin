from __future__ import annotations

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from merlin.runtime.backends import outlined_int8_board as backend


class Memref2DI32(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.POINTER(ctypes.c_int32)),
        ("aligned", ctypes.POINTER(ctypes.c_int32)),
        ("offset", ctypes.c_ssize_t),
        ("sizes", ctypes.c_ssize_t * 2),
        ("strides", ctypes.c_ssize_t * 2),
    ]


def test_file_rewrite_is_independent_of_opu_imports(tmp_path, monkeypatch):
    import builtins
    import importlib
    import sys

    from merlin.common.ir_lock import IR_LOCK

    original_import = builtins.__import__

    def no_opu(name, globals=None, locals=None, fromlist=(), level=0):
        if any(part in name or part in (fromlist or ()) for part in ("passes_opu", "opu_shim")):
            pytest.fail("generic outlined backend imported OPU implementation")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setitem(sys.modules, "merlin.llvmlower.passes_opu", None)
    monkeypatch.setitem(sys.modules, "merlin.llvmlower.opu_shim", None)
    monkeypatch.setattr(builtins, "__import__", no_opu)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("rewrite launched a native process"))
    importlib.reload(backend)
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text("""module {
  func.func @forward(%a: tensor<4x8xi8>, %b: tensor<8x6xi8>) -> tensor<4x6xi32> {
    %z = arith.constant 0 : i32
    %e = tensor.empty() : tensor<4x6xi32>
    %f = linalg.fill ins(%z : i32) outs(%e : tensor<4x6xi32>) -> tensor<4x6xi32>
    %r = linalg.matmul ins(%a, %b : tensor<4x8xi8>, tensor<8x6xi8>)
                      outs(%f : tensor<4x6xi32>) -> tensor<4x6xi32>
    return %r : tensor<4x6xi32>
  }
}
""")
    with IR_LOCK:
        result = backend.rewrite_prepared_file(prepared, tmp_path)
    symbol = backend.SYMBOL_PREFIX + "_0"
    assert result.count == 1 and result.signatures == {symbol: (4, 6, 8)}
    assert backend.load_signatures(tmp_path) == result.signatures
    assert (tmp_path / backend.SIDECAR_NAME).is_file()
    text = prepared.read_text()
    assert "linalg.matmul" not in text and f"func.call @{symbol}" in text
    assert text.count('bufferization.access = "read"') == 2
    assert text.count('bufferization.access = "write"') == 1


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a host C compiler")
def test_scalar_standin_is_exact_i8_times_i8_to_i32(tmp_path):
    """The host build takes the scalar fallback and checks the descriptor ABI and arithmetic."""
    lib_path = tmp_path / "outlined.so"
    subprocess.run(
        [
            shutil.which("cc"),
            "-shared",
            "-fPIC",
            "-O2",
            str(backend._SHIM_SRC),
            "-o",
            str(lib_path),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(lib_path))
    fn = lib.merlin_outlined_gemm_i8_body
    pi8 = ctypes.POINTER(ctypes.c_int8)
    pi32 = ctypes.POINTER(ctypes.c_int32)
    iz = ctypes.c_ssize_t
    fn.argtypes = [pi8, pi8, iz, iz, iz, iz, iz, pi8, pi8, iz, iz, iz, iz, iz, pi32, pi32, iz, iz, iz, iz, iz]
    fn.restype = Memref2DI32

    a = np.array([[1, -2, 3, 4, -5], [-7, 6, 5, -4, 3], [2, 2, 2, 2, 2]], dtype=np.int8)
    b = np.array([[2, -1, 3, 4], [5, 6, -7, 8], [-3, 2, 1, -4], [7, -8, 9, 1], [4, 3, -2, -1]], dtype=np.int8)
    c = np.full((3, 4), 0x55555555, dtype=np.int32)
    ap = a.ctypes.data_as(pi8)
    bp = b.ctypes.data_as(pi8)
    cp = c.ctypes.data_as(pi32)
    ret = fn(ap, ap, 0, 3, 5, 5, 1, bp, bp, 0, 5, 4, 4, 1, cp, cp, 0, 3, 4, 4, 1)

    np.testing.assert_array_equal(c, a.astype(np.int32) @ b.astype(np.int32))
    assert not bool(ret.allocated), "the returned destination alias is borrowed, not owned"
    assert tuple(ret.sizes) == (3, 4)
    assert tuple(ret.strides) == (4, 1)


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a host C compiler")
def test_forced_scalar_cross_build_does_not_require_rvv_intrinsics(tmp_path):
    """The diagnostic build must bypass every RVV intrinsic on a RISC-V compiler."""
    obj = tmp_path / "forced_scalar.o"
    subprocess.run(
        [
            shutil.which("cc"),
            "-O2",
            "-DMERLIN_OUTLINED_FORCE_SCALAR=1",
            "-c",
            str(backend._SHIM_SRC),
            "-o",
            str(obj),
        ],
        check=True,
    )
    assert obj.is_file()


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a host C compiler")
def test_object_builder_exports_one_wrapper_per_signature(tmp_path):
    signatures = {
        "merlin_outlined_gemm_i8_0": (8, 2048, 2048),
        "merlin_outlined_gemm_i8_1": (8, 5632, 2048),
    }
    obj = backend.build_object(shutil.which("cc"), ["-O2"], signatures, tmp_path, parallel=False)
    symbols = subprocess.run(["nm", "-g", str(obj)], check=True, capture_output=True, text=True).stdout
    assert "merlin_outlined_gemm_i8_body" in symbols
    assert "merlin_outlined_gemm_i8_0" in symbols
    assert "merlin_outlined_gemm_i8_1" in symbols
    assert " T merlin_outlined_gemm_i8_0" in symbols
