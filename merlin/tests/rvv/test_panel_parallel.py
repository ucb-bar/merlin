"""Parallel transport for independent packed-weight and im2col panels.

Panel packing deliberately keeps the proven inner register block unchanged.  Its outer ``scf.for``
writes disjoint output slices, so after bufferization removes the tensor iter-arg that loop is the
safe worksharing boundary.  These tests exercise the exact Python-IR rewrite spliced into the real
lowering runner; a text-only assertion would not prove the resulting operation verifies.
"""
from __future__ import annotations

import subprocess

import pytest

from merlin.llvmlower import panel_parallel as PP
from merlin.llvmlower import pipeline as P
from merlin.llvmlower import toolchain


MARKED = """
module {
  func.func private @__merlin_parallel_panel_marker() -> ()
  func.func @forward(%dst: memref<32xi32>) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 7 : i32
    scf.for %i = %c0 to %c32 step %c1 {
      func.call @__merlin_parallel_panel_marker() : () -> ()
      memref.store %v, %dst[%i] : memref<32xi32>
    }
    return
  }
}
"""


def test_single_panel_has_no_marker_to_leak_after_loop_folding():
    assert PP.marker_ops(True, 1) == []
    assert PP.marker_ops(False, 32) == []
    marked = PP.marker_ops(True, 2)
    assert len(marked) == 1
    assert str(marked[0].callee) == "@__merlin_parallel_panel_marker"


def _rewrite(text: str, tmp_path) -> tuple[str, str]:
    src = tmp_path / "in.mlir"
    src.write_text(text, encoding="utf-8")
    script = tmp_path / "rewrite.py"
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + PP.RUNNER_PRELUDE + "\n"
        "ctx = ir.Context()\n"
        "mod = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        "n = _parallelize_panel_loops(ctx, mod)\n"
        "mod.operation.verify()\n"
        "print('N', n)\n"
        "print('MODULE')\n"
        "print(str(mod.operation))\n",
        encoding="utf-8")
    proc = subprocess.run([str(toolchain.m2m_python()), str(script), str(src)],
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    head, _, module = proc.stdout.partition("MODULE\n")
    return module, head


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_marked_bufferized_panel_loop_becomes_verified_parallel_loop(tmp_path):
    module, report = _rewrite(MARKED, tmp_path)
    assert "scf.parallel" in module
    assert "scf.for " not in module
    assert "memref.store" in module
    assert "N 1" in report
    assert "rewritten 1 refused 0" in report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_unmarked_loop_is_byte_semantically_untouched(tmp_path):
    module, report = _rewrite(
        MARKED.replace("func.call @__merlin_parallel_panel_marker() : () -> ()\n", ""), tmp_path)
    assert "scf.for " in module and "scf.parallel" not in module
    assert "N 0" in report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_tensor_iter_arg_is_refused_until_bufferization_proves_disjoint_writes(tmp_path):
    tensor_loop = """
module {
  func.func private @__merlin_parallel_panel_marker() -> ()
  func.func @forward(%arg: tensor<32xi32>) -> tensor<32xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %r = scf.for %i = %c0 to %c32 step %c1 iter_args(%t = %arg) -> tensor<32xi32>
         {
      func.call @__merlin_parallel_panel_marker() : () -> ()
      scf.yield %t : tensor<32xi32>
    }
    return %r : tensor<32xi32>
  }
}
"""
    module, report = _rewrite(tensor_loop, tmp_path)
    assert "scf.for " in module and "scf.parallel" not in module
    assert "N 0" in report
    assert "refused 1" in report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_identity_memref_and_status_carriers_are_removed_after_bufferization(tmp_path):
    identity_carriers = """
module {
  func.func private @__merlin_parallel_panel_marker() -> ()
  func.func @forward(%dst: memref<32xi32>, %ok: i1) -> (memref<32xi32>, i1) {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %v = arith.constant 7 : i32
    %r:2 = scf.for %i = %c0 to %c32 step %c1
        iter_args(%d = %dst, %s = %ok) -> (memref<32xi32>, i1) {
      func.call @__merlin_parallel_panel_marker() : () -> ()
      memref.store %v, %d[%i] : memref<32xi32>
      scf.yield %d, %s : memref<32xi32>, i1
    }
    return %r#0, %r#1 : memref<32xi32>, i1
  }
}
"""
    module, report = _rewrite(identity_carriers, tmp_path)
    assert "scf.parallel" in module and "scf.for " not in module
    assert "return %arg0, %arg1" in module
    assert "rewritten 1 refused 0" in report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_nested_parallel_region_is_refused_instead_of_emitting_nested_forks(tmp_path):
    nested = MARKED.replace(
        "memref.store %v, %dst[%i] : memref<32xi32>",
        "scf.parallel (%j) = (%c0) to (%c32) step (%c1) {\n"
        "        memref.store %v, %dst[%j] : memref<32xi32>\n"
        "        scf.reduce\n"
        "      }")
    module, report = _rewrite(nested, tmp_path)
    assert module.count("scf.parallel") == 1
    assert "scf.for " in module
    assert "N 0" in report and "nested_parallel 1" in report


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv missing")
def test_real_multicore_pipeline_emits_worksharing_and_erases_marker(tmp_path):
    """The runner splice must survive every pass between source IR and the OpenMP dialect."""
    ll = P.lower_to_llvm_ir(
        MARKED, workdir=tmp_path, vectorize=True, parallel_harts=8, parallel_chunks=[],
        features=frozenset({"prepack_weight_panels"}))
    assert "@__kmpc_fork_call" in ll
    assert "@__kmpc_for_static_init" in ll
    assert "__merlin_parallel_panel_marker" not in ll


def test_every_lowering_runner_carries_the_panel_rewrite_and_gate():
    """A feature-specific runner may not silently turn the multicore artifact back into serial."""
    from merlin.llvmlower import accum_microkernel

    runners = [P._RUNNER_SRC, P._RUNNER_ACT_POLY_TAIL, accum_microkernel.run_source()]
    for src in runners:
        assert "_parallelize_panel_loops" in src
        assert "len(sys.argv) > 10" in src
        assert "_MID_STAGES, _LATE_STAGES)" in src
    lowering = open(P.__file__, encoding="utf-8").read()
    assert "_grain_gate, _panel_parallel_gate]" in lowering


def test_report_is_fail_closed_and_persisted(tmp_path):
    line = "OK panel_parallel regions 155 rewritten 155 refused 0 nested_parallel 0\n"
    assert PP.require_complete_report(line, tmp_path) == {
        "regions": 155, "rewritten": 155, "refused": 0, "nested_parallel": 0}
    assert (tmp_path / PP.REPORT_FILE).is_file()
    for bad in ("", "OK panel_parallel regions 0 rewritten 0 refused 0 nested_parallel 0",
                "OK panel_parallel regions 2 rewritten 1 refused 1 nested_parallel 0"):
        with pytest.raises(ValueError):
            PP.require_complete_report(bad, tmp_path)
