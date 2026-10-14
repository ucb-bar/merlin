"""Round trip of the Lean IR export (design D9): export → Lean decode/print → mlir-opt canonical print
must equal mlir-opt's canonical print of the original file."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from merlin.common.paths import out_dir, repo_root
from merlin.verify.lean.export import export_text, round_trip

LEAN_EXE = out_dir() / "build" / "lean" / "build" / "bin" / "merlin-lean"
MLIR_OPT = Path(os.environ.get("MERLIN_VERIFY_MLIR_OPT", "/home/jeremy/Projects/llvm-project/build/bin/mlir-opt"))

SAMPLE = """
func.func @f(%a: tensor<2x3xf32>, %b: tensor<2x3xi8>) -> tensor<2x3xf32> {
  %c = arith.constant dense<1.5> : tensor<2x3xf32>
  %k = arith.constant 3 : i32
  %e = tensor.empty() : tensor<2x3xf32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
                       iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<2x3xf32>) outs(%e : tensor<2x3xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = arith.addf %x, %x fastmath<nnan,ninf> : f32
    linalg.yield %s : f32
  } -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}
"""


def test_export_is_structural():
    doc = export_text(SAMPLE)
    assert doc["schema"] == "merlin.lean.ir.v1"
    (func,) = doc["module"]["ops"]
    body = func["regions"][0]["blocks"][0]["ops"]
    generic = next(op for op in body if op["name"] == "linalg.generic")
    maps = generic["attrs"].get("indexing_maps") or generic["props"]["indexing_maps"]
    assert maps["elements"][0]["kind"] == "affine_map"
    iters = generic["attrs"].get("iterator_types") or generic["props"]["iterator_types"]
    assert iters["elements"][0] == {"kind": "enum", "dialect": "linalg", "name": "iterator_type", "value": "parallel"}


@pytest.mark.skipif(not LEAN_EXE.exists() or not MLIR_OPT.exists(),
                    reason="merlin-lean (lake build) or mlir-opt not built on this machine")
def test_round_trip_identical(tmp_path):
    src = tmp_path / "sample.mlir"
    src.write_text(SAMPLE)
    result = round_trip(src, tmp_path / "work", lean_exe=LEAN_EXE, mlir_opt=MLIR_OPT)
    assert result.status == "identical", result.detail
