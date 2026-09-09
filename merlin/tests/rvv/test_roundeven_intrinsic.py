"""The compact exact round-even lowering is a real, correctly placed compiler lever."""

import pytest

from merlin.llvmlower import impr_features as F
from merlin.llvmlower.pipeline import build_rvv_pipeline, lower_to_llvm_ir
from merlin.llvmlower.roundeven_intrinsic import FEATURE, MARKER, apply_for_test, ensure_registered


def test_rewrite_changes_only_roundeven_to_the_exact_llvm_intrinsic():
    src = """module {
      func.func @f(%x: f32) -> (f32, f32) {
        %r = math.roundeven %x : f32
        %e = math.exp %x : f32
        return %r, %e : f32, f32
      }
    }"""
    got, count = apply_for_test(src)
    assert count == 1
    assert "llvm.intr.roundeven" in got
    assert "math.roundeven" not in got
    assert "math.exp" in got, "unrelated math operations must retain the established libm path"


def test_feature_places_the_runner_marker_after_linalg_becomes_loops():
    ensure_registered()
    passes = F.apply_pipeline(["canonicalize", "func.func(convert-linalg-to-loops)",
                               "convert-scf-to-cf"], {FEATURE})
    assert passes.index(MARKER) == passes.index("func.func(convert-linalg-to-loops)") + 1
    assert passes.index(MARKER) < passes.index("convert-scf-to-cf")


def test_real_lowering_reaches_llvm_intrinsic_not_libm(tmp_path):
    src = """module {
      func.func @f(%a: memref<16xf32>, %b: memref<16xf32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %i = %c0 to %c16 step %c1 {
          %x = memref.load %a[%i] : memref<16xf32>
          %r = math.roundeven %x : f32
          memref.store %r, %b[%i] : memref<16xf32>
        }
        return
      }
    }"""
    got = lower_to_llvm_ir(src, workdir=tmp_path, features={FEATURE})
    assert "@llvm.roundeven.f32" in got
    assert "roundevenf" not in got


def test_roundeven_intrinsic_and_arithmetic_expansion_are_explicit_alternatives(tmp_path):
    ensure_registered()
    with pytest.raises(Exception, match="alternative exact lowerings"):
        lower_to_llvm_ir("module {}", workdir=tmp_path, vectorize=True,
                         features={FEATURE, "fuse_quantize_round_convert"})
