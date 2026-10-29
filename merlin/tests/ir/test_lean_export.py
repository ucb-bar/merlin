"""Round trip of the Lean IR export (design D9): export → Lean decode/print → mlir-opt canonical print
must equal mlir-opt's canonical print of the original file."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from merlin.common.paths import out_dir
from merlin.verify.lean.export import export_text, round_trip
from merlin.verify.tools import find_mlir_tool

LEAN_EXE = out_dir() / "build" / "lean" / "build" / "bin" / "merlin-lean"
_selected_opt = os.environ.get("MERLIN_VERIFY_MLIR_OPT") or find_mlir_tool("mlir-opt")
MLIR_OPT = Path(_selected_opt) if _selected_opt else None

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


@pytest.mark.skipif(
    not LEAN_EXE.exists() or MLIR_OPT is None or not MLIR_OPT.exists(),
    reason="merlin-lean (lake build) or mlir-opt not built on this machine",
)
def test_round_trip_identical(tmp_path):
    src = tmp_path / "sample.mlir"
    src.write_text(SAMPLE)
    result = round_trip(src, tmp_path / "work", lean_exe=LEAN_EXE, mlir_opt=MLIR_OPT)
    assert result.status == "identical", result.detail


# Forms the Lean printer must reproduce exactly through the authoritative exporter, and a form outside
# the exported fragment that must be refused rather than exported lossily.
AUTHORITATIVE = {
    "sample": SAMPLE,
    "shaped_i1": """
func.func @f() -> tensor<2x2xi1> {
  %0 = arith.constant dense<[[true, false], [false, true]]> : tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}
""",
    "quoted_symbols": """
func.func private @"with space"(i32) -> i32
func.func @"quote\\"name"(%a: i32) -> i32 {
  %0 = func.call @"with space"(%a) {"odd key" = 1 : i32} : (i32) -> i32
  return %0 : i32
}
""",
    "numeric_attributes": """
func.func @f(%a: i32, %b: i32, %x: f32, %y: f32) -> (i32, i1, f32) {
  %i = arith.addi %a, %b overflow<nsw, nuw> : i32
  %c = arith.cmpf uno, %x, %y fastmath<nnan> : f32
  %v = arith.mulf %x, %y fastmath<contract, reassoc> : f32
  return %i, %c, %v : i32, i1, f32
}
""",
    "integer_dense_and_signed_zero": """
func.func @f() -> (tensor<2x2xi8>, f32) {
  %a = arith.constant dense<[[0, -128], [127, -1]]> : tensor<2x2xi8>
  %z = arith.constant -0.0 : f32
  return %a, %z : tensor<2x2xi8>, f32
}
""",
}


@pytest.mark.parametrize("name", sorted(AUTHORITATIVE))
def test_authoritative_round_trip(tmp_path, lean_tools, name):
    src = tmp_path / f"{name}.mlir"
    src.write_text(AUTHORITATIVE[name])
    result = round_trip(
        src, tmp_path / "work", lean_exe=lean_tools.lean_exe, backend="mlir", m2m_python=lean_tools.m2m_python
    )
    assert result.status == "identical", result.detail


def test_branches_are_refused(tmp_path, lean_tools):
    src = tmp_path / "branch.mlir"
    src.write_text("""
func.func @f(%c: i1, %a: i32) -> i32 {
  cf.cond_br %c, ^bb1, ^bb2
^bb1:
  return %a : i32
^bb2:
  return %a : i32
}
""")
    result = round_trip(
        src, tmp_path / "work", lean_exe=lean_tools.lean_exe, backend="mlir", m2m_python=lean_tools.m2m_python
    )
    assert result.status == "export_failed" and "outside the exported fragment" in result.detail


def test_authoritative_comparison_detects_changed_export(tmp_path, monkeypatch, lean_tools):
    from merlin.verify.lean import export

    source = tmp_path / "add.mlir"
    source.write_text("""func.func @f(%a: i32, %b: i32) -> i32 {
  %r = arith.addi %a, %b : i32
  return %r : i32
}""")
    original = export.export_file

    def corrupt(src, dst, **kwargs):
        original(src, dst, **kwargs)
        doc = json.loads(dst.read_text())
        body = doc["module"]["ops"][0]["regions"][0]["blocks"][0]["ops"]
        assert body[0]["name"] == "arith.addi"
        body[0]["name"] = "arith.subi"
        dst.write_text(json.dumps(doc))

    monkeypatch.setattr(export, "export_file", corrupt)
    result = export.round_trip(
        source, tmp_path / "work", lean_exe=lean_tools.lean_exe, backend="mlir", m2m_python=lean_tools.m2m_python
    )
    assert result.status == "different", result.detail
