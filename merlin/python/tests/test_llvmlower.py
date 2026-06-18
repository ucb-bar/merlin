"""Whole-model lowering: dequant lowering + upstream pipeline + host execution.

Gated on the external toolchain (m2m venv with torch-mlir, clang-23); auto-skips when
absent. The host execution is the verification oracle preceding spike RVV runs.
"""
from __future__ import annotations

import ctypes

import pytest

from merlin.llvmlower import toolchain
from merlin.xdsl_dialects import _common

pytestmark = pytest.mark.skipif(not _common.HAS_XDSL, reason="xDSL not installed")

SLICE = """
builtin.module {
  func.func @forward(%w: tensor<8x6xi8>, %s: tensor<6xf32>, %zp: tensor<6xi32>,
                     %x: tensor<4x8xf32>) -> tensor<4x6xf32> {
    %wf = "quant_ext.dequantize_per_channel"(%w, %s, %zp)
        <{axis = 1 : i64, input_dtype = "i8"}>
        : (tensor<8x6xi8>, tensor<6xf32>, tensor<6xi32>) -> tensor<8x6xf32>
    %e = tensor.empty() : tensor<4x6xf32>
    %c0 = arith.constant 0.0 : f32
    %f = linalg.fill ins(%c0 : f32) outs(%e : tensor<4x6xf32>) -> tensor<4x6xf32>
    %y = linalg.matmul ins(%x, %wf : tensor<4x8xf32>, tensor<8x6xf32>)
         outs(%f : tensor<4x6xf32>) -> tensor<4x6xf32>
    func.return %y : tensor<4x6xf32>
  }
}
"""


def test_dequant_lowering_emits_pure_upstream():
    from merlin.llvmlower.passes_xdsl import preprocess_text

    out, stats = preprocess_text(SLICE)
    assert stats["dequantize_lowered"] == 1
    assert "quant_ext" not in out
    assert "linalg.generic" in out
    assert "llvm.emit_c_interface" in out
    assert "arith.sitofp" in out


def test_weights_pack_against_real_manifest():
    from pathlib import Path

    from merlin.llvmlower.weights_pack import emit_c_table, pack

    base = Path("/scratch/agustin/projects/model2MLIR/workloads/smolvla")
    if not (base / "smolvla_int8.safetensors").is_file():
        pytest.skip("smolVLA int8 artifact not present")
    from merlin.llvmlower.weights_pack import missing_buffers

    manifest = base / "smolvla_int8.safetensors.manifest.json"
    st = base / "smolvla_int8.safetensors"
    entries = pack(manifest, st)
    assert len(entries) == 1106          # params; 4 buffers are runtime-computed
    assert entries[0].arg_index == 0
    total = sum(e.nbytes for e in entries)
    assert 400e6 < total < 600e6         # ~0.47 GB int8 weights
    table = emit_c_table(entries)
    assert "MERLIN_WEIGHT_COUNT 1106" in table
    missing = missing_buffers(manifest, st)
    assert len(missing) == 4
    assert all("rotary_emb" in name and "inv_freq" in name for _, name in missing)


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_slice_lowers_and_executes_on_host(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model

    res = lower_model(SLICE, tmp_path, targets=("host",))
    model = HostModel.load(str(res.host_so))

    w = (ctypes.c_int8 * 48)(*[(i % 7) - 3 for i in range(48)])
    s = (ctypes.c_float * 6)(*[0.5, 1.0, 0.25, 2.0, 1.5, 0.125])
    zp = (ctypes.c_int32 * 6)(*[1, 0, -2, 3, 0, 1])
    x = (ctypes.c_float * 32)(*[(i % 5) - 2 for i in range(32)])
    y = (ctypes.c_float * 24)()
    model([(ctypes.addressof(w), (8, 6)), (ctypes.addressof(s), (6,)),
           (ctypes.addressof(zp), (6,)), (ctypes.addressof(x), (4, 8)),
           (ctypes.addressof(y), (4, 6))])

    W = [[(r * 6 + c) % 7 - 3 for c in range(6)] for r in range(8)]
    X = [[(r * 8 + c) % 5 - 2 for c in range(8)] for r in range(4)]
    S, ZP = [0.5, 1.0, 0.25, 2.0, 1.5, 0.125], [1, 0, -2, 3, 0, 1]
    for i in range(4):
        for j in range(6):
            ref = sum(X[i][k] * (W[k][j] - ZP[j]) * S[j] for k in range(8))
            assert abs(y[i * 6 + j] - ref) < 1e-5


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_parallel_lowering_emits_openmp_and_default_does_not(tmp_path):
    """The gated multicore path (parallel=True) lowers via OpenMP -> __kmpc_* runtime calls
    (satisfied at link by the cross-built libomp); the DEFAULT scalar path must NOT — proving
    the multicore lowering is isolated and cannot leak into the shipping flow."""
    from merlin.llvmlower.passes_xdsl import preprocess_text
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    upstream, _ = preprocess_text(SLICE)
    par_ll = lower_to_llvm_ir(upstream, workdir=tmp_path / "par", parallel=True)
    assert "__kmpc_fork_call" in par_ll
    seq_ll = lower_to_llvm_ir(upstream, workdir=tmp_path / "seq")
    assert "__kmpc" not in seq_ll


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_slice_compiles_to_rvv_object(tmp_path):
    import subprocess

    from merlin.llvmlower.lower import lower_model
    from merlin.runtime.backends import spike

    res = lower_model(SLICE, tmp_path, targets=("riscv",))
    assert res.riscv_obj.is_file()
    if not spike.available():
        pytest.skip("chipyard objdump unavailable")
    dasm = subprocess.run(
        [spike.gcc_path().with_name("riscv64-unknown-elf-objdump"), "-d",
         res.riscv_obj], capture_output=True, text=True).stdout
    assert "_mlir_ciface_forward" in dasm
    assert "vset" in dasm  # auto-vectorized RVV
