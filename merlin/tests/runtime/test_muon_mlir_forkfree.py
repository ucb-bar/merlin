"""The radiance thesis path: an agent's LLVM-dialect MLIR 4th artifact is compiled FORK-FREE (stock LLVM
rv32 + the RTL-derived Muon re-encode, no clang-muon) into a cyclotron ELF and grades correct. Mirrors the
gemmini contract (agent emits MLIR, runner owns the harness) but for the SIMT/Muon target.

Kernel-ABI order is ``[weight] ++ [lhs] ++ [out]`` (the generic kernel_abi the harness derives from the cb),
so the reference kernel's params are ``(%W, %L, %O)`` and it computes ``O = L @ W``.
"""
import pytest

from merlin.runtime.backends import muon

pytestmark = pytest.mark.skipif(not muon.available("cyclotron"),
                                reason="cyclotron SIMT oracle / stock LLVM not available")

# A 2x2 gemm as LLVM-dialect MLIR (plain-pointer ABI, params in [weight, lhs, out] order).
_KERNEL_MLIR = """module {
  llvm.func @radiance_kernel(%W: !llvm.ptr, %L: !llvm.ptr, %O: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c2 = llvm.mlir.constant(2 : i64) : i64
    %c3 = llvm.mlir.constant(3 : i64) : i64
    %pw0 = llvm.getelementptr %W[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pw1 = llvm.getelementptr %W[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pw2 = llvm.getelementptr %W[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pw3 = llvm.getelementptr %W[%c3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %w0 = llvm.load %pw0 : !llvm.ptr -> f32
    %w1 = llvm.load %pw1 : !llvm.ptr -> f32
    %w2 = llvm.load %pw2 : !llvm.ptr -> f32
    %w3 = llvm.load %pw3 : !llvm.ptr -> f32
    %pl0 = llvm.getelementptr %L[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pl1 = llvm.getelementptr %L[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pl2 = llvm.getelementptr %L[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %pl3 = llvm.getelementptr %L[%c3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %l0 = llvm.load %pl0 : !llvm.ptr -> f32
    %l1 = llvm.load %pl1 : !llvm.ptr -> f32
    %l2 = llvm.load %pl2 : !llvm.ptr -> f32
    %l3 = llvm.load %pl3 : !llvm.ptr -> f32
    %m0a = llvm.fmul %l0, %w0 : f32
    %m0b = llvm.fmul %l1, %w2 : f32
    %o0 = llvm.fadd %m0a, %m0b : f32
    %m1a = llvm.fmul %l0, %w1 : f32
    %m1b = llvm.fmul %l1, %w3 : f32
    %o1 = llvm.fadd %m1a, %m1b : f32
    %m2a = llvm.fmul %l2, %w0 : f32
    %m2b = llvm.fmul %l3, %w2 : f32
    %o2 = llvm.fadd %m2a, %m2b : f32
    %m3a = llvm.fmul %l2, %w1 : f32
    %m3b = llvm.fmul %l3, %w3 : f32
    %o3 = llvm.fadd %m3a, %m3b : f32
    %po0 = llvm.getelementptr %O[%c0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %po1 = llvm.getelementptr %O[%c1] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %po2 = llvm.getelementptr %O[%c2] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %po3 = llvm.getelementptr %O[%c3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %o0, %po0 : f32, !llvm.ptr
    llvm.store %o1, %po1 : f32, !llvm.ptr
    llvm.store %o2, %po2 : f32, !llvm.ptr
    llvm.store %o3, %po3 : f32, !llvm.ptr
    llvm.return
  }
}
"""

_CB = {
    "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc", "lhs": "A", "rhs": "B"}},
                 {"opcode": "COMMIT", "operands": {"dst": "OUT", "src": "acc"}}],
    "tensors": {"A": {"shape": [2, 2], "dtype": "f32", "role": "input"},
                "B": {"shape": [2, 2], "dtype": "f32", "role": "weight"}},
    "canonical_inputs": {"A": {"values": [1, 2, 3, 4]}, "B": {"values": [5, 6, 7, 8]}},
}


def test_mlir_forkfree_grades_on_cyclotron(tmp_path):
    elf = muon.compile_mlir_forkfree(_KERNEL_MLIR, _CB, tmp_path, target="radiance")
    assert elf.is_file() and elf.stat().st_size > 0
    console, cycles, _ = muon.run_elf(elf, "cyclotron", timeout=300)
    outputs, _raw = muon.parse_output(console, cycles)
    assert outputs.get("OUT") == [[19.0, 22.0], [43.0, 50.0]], console


@pytest.mark.skipif(not muon.available("verilator"),
                    reason="RadianceTapeoutSim Verilator RTL sim / rv64 SoC-fuse toolchain not available")
def test_mlir_forkfree_certs_on_verilator_rtl(tmp_path):
    """The RTL cert (L3): the SAME fork-free ELF is fused into the rv64 SoC carrier and run to completion
    on the RadianceTapeoutSimConfig Verilator RTL sim. It is a COMPLETION + cycle-accurate perf cert (the
    harness $finish races the UART flush, so console outputs do not surface for grading — the verilator
    adapter reports ``completion_only`` and correctness stays the required cyclotron L2 gate's job)."""
    from merlin.targetgen import muon_oracles
    cb = dict(_CB, target="radiance")
    res = muon_oracles.verilator_muon_adapter()(cb, _KERNEL_MLIR, tmp_path, 600)
    assert res.get("completion_only") is True, res
    assert res.get("cycles") and res["cycles"] > 0, res
    assert res["toolchain"] == "fork-free", res
    assert res["oracle"]["derived_from_rtl"] is True, res
    assert "finished execution" in res["console"], res["console"][-800:]


def test_kernel_symbol_parsed_structurally():
    assert muon._kernel_symbol_from_mlir(_KERNEL_MLIR) == "radiance_kernel"


def test_is_mlir_artifact_detects_llvm_dialect():
    assert muon.is_mlir_artifact(_KERNEL_MLIR)
    assert not muon.is_mlir_artifact("#include <x>\nvoid radiance_kernel(){}")


def test_reference_mlir_emitter_grades_on_cyclotron(tmp_path):
    """The reference MLIR emitter (muon_codegen_mlir) emits an LLVM-dialect matmul loop nest that compiles
    fork-free and grades correct on cyclotron for a 16x16 gemm — the reference the agentic arms must match."""
    import numpy as np
    from merlin.runtime.backends.muon_codegen_mlir import emit_kernel_mlir
    rng = np.random.default_rng(0)
    m = k = n = 16
    a = rng.standard_normal((m, k)).astype(np.float32)
    w = rng.standard_normal((k, n)).astype(np.float32)
    ref = (a @ w).astype(np.float32)
    cb = {
        "target": "radiance",
        "tensors": {"W": {"shape": [k, n], "dtype": "f32", "role": "weight"},
                    "A0": {"shape": [m, k], "dtype": "f32", "role": "input"},
                    "acc0": {"shape": [m, n], "dtype": "f32"},
                    "Y0": {"shape": [m, n], "dtype": "f32"}},
        "commands": [{"opcode": "MATMUL", "operands": {"dst": "acc0", "lhs": "A0", "rhs": "W"}},
                     {"opcode": "COMMIT", "operands": {"dst": "Y0", "src": "acc0"}}],
        "canonical_inputs": {"W": {"values": [float(x) for x in w.flatten()]},
                             "A0": {"values": [float(x) for x in a.flatten()]}},
    }
    mlir = emit_kernel_mlir(cb, target="radiance")
    assert "llvm.func @radiance_kernel" in mlir
    elf = muon.compile_mlir_forkfree(mlir, cb, tmp_path, target="radiance")
    console, cycles, _ = muon.run_elf(elf, "cyclotron", timeout=300)
    outputs, _raw = muon.parse_output(console, cycles)
    got = np.array(next(iter(outputs.values())), dtype=np.float32)
    assert got.shape == (m, n) and float(np.max(np.abs(got - ref))) <= 0.03125, console
