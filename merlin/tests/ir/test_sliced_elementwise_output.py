"""The final consumer's static slice bounds the elementwise producer's work and storage."""

from __future__ import annotations

import subprocess

import pytest

from merlin.llvmlower import pipeline as P
from merlin.llvmlower import toolchain

M2M = pytest.mark.skipif(not toolchain.m2m_python().is_file(), reason="model2MLIR venv unavailable")

SOURCE = """module {
  func.func @forward(%input: memref<1x4x8xi32>, %output: memref<1x8xf32>) {
    %tmp = memref.alloc() {alignment = 64 : i64} : memref<1x4x8xf32>
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                    iterator_types = ["parallel", "parallel", "parallel"]}
      ins(%input : memref<1x4x8xi32>) outs(%tmp : memref<1x4x8xf32>) {
    ^bb0(%in: i32, %out: f32):
      %f = arith.sitofp %in : i32 to f32
      linalg.yield %f : f32
    }
    %slice = memref.subview %tmp[0, 3, 0] [1, 1, 8] [1, 1, 1] :
      memref<1x4x8xf32> to memref<1x1x8xf32, strided<[32, 8, 1], offset: 24>>
    %row = memref.collapse_shape %slice [[0], [1, 2]] :
      memref<1x1x8xf32, strided<[32, 8, 1], offset: 24>> into
      memref<1x8xf32, strided<[32, 1], offset: 24>>
    memref.copy %row, %output : memref<1x8xf32, strided<[32, 1], offset: 24>> to memref<1x8xf32>
    memref.dealloc %tmp : memref<1x4x8xf32>
    return
  }
}"""


def _rewrite(tmp_path, source: str) -> str:
    src, script = tmp_path / "input.mlir", tmp_path / "rewrite.py"
    src.write_text(source, encoding="utf-8")
    script.write_text(
        "import sys\nfrom torch_mlir import ir\n"
        + P.SLICED_OUTPUT_PRELUDE
        + "ctx = ir.Context()\n"
        + "with ctx:\n"
        + "  module = ir.Module.parse(open(sys.argv[1]).read(), ctx)\n"
        + "  _slice_unused_elementwise_outputs(ctx, module)\n"
        + "  print(module.operation)\n",
        encoding="utf-8",
    )
    return subprocess.run(
        [str(toolchain.m2m_python()), str(script), str(src)],
        capture_output=True, text=True, timeout=60, check=True,
    ).stdout


@M2M
def test_only_copied_slice_is_computed(tmp_path):
    out = _rewrite(tmp_path, SOURCE)
    assert "OK sliced_elementwise_output 1" in out
    assert "memref.alloc() {alignment = 64 : i64} : memref<1x1x8xf32>" in out
    assert "memref<1x1x8xi32, strided<[32, 8, 1], offset: 24>>" in out
    assert "memref<1x4x8xf32>" not in out
    assert out.count("memref.dealloc") == 1


@M2M
def test_second_consumer_preserves_full_computation(tmp_path):
    source = SOURCE.replace(
        "    memref.dealloc %tmp",
        "    %c0 = arith.constant 0 : index\n"
        "    %other = memref.load %tmp[%c0, %c0, %c0] : memref<1x4x8xf32>\n"
        "    memref.store %other, %output[%c0, %c0] : memref<1x8xf32>\n"
        "    memref.dealloc %tmp",
    )
    out = _rewrite(tmp_path, source)
    assert "OK sliced_elementwise_output 0" in out
    assert "memref<1x4x8xf32>" in out


def test_switch_is_explicit_and_present_in_every_runner(monkeypatch):
    monkeypatch.delenv("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", raising=False)
    assert P.SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER not in P._upstream_pipeline()
    monkeypatch.setenv("MERLIN_SLICE_UNUSED_ELEMENTWISE_OUTPUT", "1")
    assert P.SLICE_UNUSED_ELEMENTWISE_OUTPUT_MARKER in P._dealloc_passes()
    assert P.SLICED_OUTPUT_RUNNER in P._RUNNER_SRC
    assert P.SLICED_OUTPUT_RUNNER in P._activation_poly_runner()
