"""Scope-safe lowering for stack temporaries inside parallel structured loops."""
from __future__ import annotations

import pytest

from merlin.llvmlower import alloca_scope_lower as ASL
from merlin.llvmlower.toolchain import m2m_python


_needs_toolchain = pytest.mark.skipif(
    not m2m_python().is_file(), reason="m2m lowering toolchain unavailable")


@_needs_toolchain
def test_structured_alloca_scope_is_lowered_before_cfg_conversion():
    source = """
module {
  func.func @scope_in_loop(%n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %i = %c0 to %n step %c1 {
      memref.alloca_scope {
        %scratch = memref.alloca() : memref<4x15xi32>
        scf.for %j = %c0 to %c4 step %c1 {
          memref.store %c0_i32, %scratch[%j, %c0] : memref<4x15xi32>
        }
      }
    }
    return
  }
}
"""
    lowered, count = ASL.lower_text_for_test(source)
    assert count == 1
    assert "memref.alloca_scope" not in lowered
    assert "llvm.intr.stacksave" in lowered
    assert "llvm.intr.stackrestore" in lowered
    cfg = ASL.apply_passes_for_test(lowered, "convert-scf-to-cf")
    assert "cf.cond_br" in cfg
    assert "memref.alloca_scope" not in cfg


def test_scope_lowering_is_feature_gated_in_every_runner():
    from merlin.llvmlower import accum_microkernel, pipeline

    runners = [pipeline._RUNNER_SRC, pipeline._RUNNER_ACT_POLY_TAIL,
               accum_microkernel.run_source()]
    for source in runners:
        assert "_lower_structured_alloca_scopes" in source
        assert "len(sys.argv) > 16" in source
        assert "_POST_OPENMP_STAGES" in source
    lowering = open(pipeline.__file__, encoding="utf-8").read()
    assert 'omp and "merlin.rqfuse" in mlir_text' in lowering


def test_requested_scope_lowering_requires_a_positive_receipt(tmp_path):
    assert ASL.require_report("OK alloca_scope_pre_cfg 44\n", tmp_path) == 44
    assert (tmp_path / ASL.REPORT_FILE).read_text() == "lowered=44\n"
    with pytest.raises(ValueError, match="rewrote no scopes"):
        ASL.require_report("OK alloca_scope_pre_cfg 0\n", tmp_path)
    with pytest.raises(ValueError, match="expected one"):
        ASL.require_report("", tmp_path)
