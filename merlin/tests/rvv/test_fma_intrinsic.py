"""The optional FMA path stays fused and preserves unrelated libm lowering."""

import ctypes
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.fma_intrinsic import FEATURE, MARKER, ensure_registered
from merlin.llvmlower.impr_features import apply_pipeline
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang


def test_torchao_patch_conv_fma_crosses_scalarize_runner_as_rv64_fmadd(tmp_path):
    """Use the exact captured patch convolution to select its source FMA, then lower that operation."""
    configured = os.environ.get("MERLIN_SPECT_TORCHAO_CAPTURE")
    if not configured:
        pytest.skip("set MERLIN_SPECT_TORCHAO_CAPTURE to the exact source capture")
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower import pipeline as P
    from merlin.llvmlower.accum_microkernel import SCALARIZE_MARKER
    from merlin.llvmlower.torchao_cpu_patch_conv import TORCH_SOURCE_COMMIT, rewrite_torch216_avx2_patch_conv

    source_path = Path(configured) / "model.mlir"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        "9ae3f772ed2d192899cea502e5dec0098d5abf95f56f1707c7ba3cdfcf5481ab"
    )
    source = parse_mlir_file(source_path)
    assert rewrite_torch216_avx2_patch_conv(source, source_commit=TORCH_SOURCE_COMMIT, cpu_capability="AVX2") == 1
    fmas = [op for op in source.walk() if op.name == "math.fma"]
    assert len(fmas) == 1
    assert [str(value.type) for value in [*fmas[0].operands, *fmas[0].results]] == ["f32"] * 4

    # Isolate the real rewrite's scalar operation so the regression exercises
    # the same two-stage runner without building the whole 1.7 MiB model.
    isolated = (
        "module { func.func @forward(%a: f32, %b: f32, %c: f32) -> f32 { "
        f"%x = {fmas[0].name} %a, %b, %c : f32\n"
        "return %x : f32 } }"
    )
    ensure_registered()
    stages = SCALARIZE_MARKER + "," + P._upstream_pipeline(frozenset({FEATURE}))
    llvm = lower_to_llvm_ir(isolated, workdir=tmp_path / "lower", pipeline=stages)
    assert "@llvm.fma.f32" in llvm and "@fmaf" not in llvm
    cross = subprocess.run(
        [
            str(clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gcv_zvl512b",
            "-mabi=lp64d",
            "-O2",
            "-S",
            "-x",
            "ir",
            "-",
            "-o",
            "-",
        ],
        input=llvm,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "fmadd.s" in cross.stdout


def test_explicit_stage_after_loop_formation():
    ensure_registered()
    baseline = ["canonicalize", "func.func(convert-linalg-to-loops)", "convert-scf-to-cf"]
    assert apply_pipeline(baseline, set()) == baseline
    selected = apply_pipeline(baseline, {FEATURE})
    assert selected.index(MARKER) == selected.index(baseline[1]) + 1


def test_scalar_vector_and_unrelated_math_lowering(tmp_path):
    source = """module {
      func.func @scalar(%a: f32, %b: f32, %c: f32) -> (f32, f32) {
        %x = math.fma %a, %b, %c {prov.region_id = "test"} : f32
        %e = math.exp %a : f32
        return %x, %e : f32, f32
      }
      func.func @packed(%a: vector<4xf32>, %b: vector<4xf32>, %c: vector<4xf32>) -> vector<4xf32> {
        %x = math.fma %a, %b, %c : vector<4xf32>
        return %x : vector<4xf32>
      }
    }"""
    actual = lower_to_llvm_ir(source, workdir=tmp_path, features={FEATURE})
    assert "@llvm.fma.f32" in actual and "@llvm.fma.v4f32" in actual
    assert "@expf" in actual
    assert "@fmaf" not in actual


def test_composes_with_exact_roundeven_intrinsic(tmp_path):
    source = """module {
      func.func @value(%a: f32, %b: f32, %c: f32) -> f32 {
        %x = math.fma %a, %b, %c : f32
        %y = math.roundeven %x : f32
        return %y : f32
      }
    }"""
    actual = lower_to_llvm_ir(source, workdir=tmp_path, features={FEATURE, "lower_roundeven_to_intrinsic"})
    assert "@llvm.fma.f32" in actual and "@llvm.roundeven.f32" in actual
    assert "@fmaf" not in actual and "@roundevenf" not in actual


@pytest.mark.parametrize("width", [32, 64])
def test_executed_fma_matches_library_including_cancellation(tmp_path, width):
    compiler = clang()
    if not shutil.which(str(compiler)):
        pytest.skip("native clang unavailable")
    source = """module {
      func.func @value(%a: f32, %b: f32, %c: f32) -> f32 {
        %x = math.fma %a, %b, %c : f32
        return %x : f32
      }
    }"""
    if width == 64:
        source = source.replace("f32", "f64")
    dtype, bits = (np.float32, np.uint32) if width == 32 else (np.float64, np.uint64)
    ctype = ctypes.c_float if width == 32 else ctypes.c_double
    precision = 23 if width == 32 else 52
    lowered = lower_to_llvm_ir(source, workdir=tmp_path / "lower", features={FEATURE})
    # Fixture cleanup can reuse a directory while dlopen retains its old image.
    # Distinct ABI variants therefore need distinct loader identities.
    ll, library = tmp_path / "model.ll", tmp_path / f"model_fma_{width}.so"
    ll.write_text(lowered)
    subprocess.run([str(compiler), "-O2", "-fPIC", "-shared", str(ll), "-lm", "-o", str(library)], check=True)
    actual = ctypes.CDLL(str(library)).value
    reference = getattr(ctypes.CDLL("libm.so.6"), "fmaf" if width == 32 else "fma")
    for function in (actual, reference):
        function.argtypes = [ctype] * 3
        function.restype = ctype
    rng = np.random.default_rng(1024)
    rows = rng.standard_normal((1024, 3)).astype(dtype)
    rows = np.vstack(
        [
            rows,
            [dtype(1 + 2**-precision), dtype(1 - 2**-precision), -1],
            [np.finfo(dtype).max, 2, -np.finfo(dtype).max],
            [np.finfo(dtype).tiny, 0.5, 0],
            [-0.0, 1, -0.0],
        ]
    )
    for a, b, c in rows:
        got = dtype(actual(float(a), float(b), float(c)))
        expected = dtype(reference(float(a), float(b), float(c)))
        assert got.view(bits) == expected.view(bits)
    a, b, c = map(dtype, rows[1024])
    assert dtype(actual(float(a), float(b), float(c))) != dtype(a * b + c)
