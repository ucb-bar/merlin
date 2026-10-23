"""Ordered source loops may request a fully checked integer recapture."""

import pytest
from xdsl.dialects.builtin import StringAttr

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_numeric import scalarized_contraction_candidates

SOURCE = """builtin.module {
  func.func @forward(%a: tensor<1x2xf32>,%b: tensor<2x1xf32>) -> tensor<1x1xf32> {
    %e=tensor.empty() : tensor<1x1xf32>
    %r=linalg.generic {
      indexing_maps=[affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]
    } outs(%e : tensor<1x1xf32>) {
    ^bb0(%unused: f32):
      %m=linalg.index 0 : index
      %n=linalg.index 1 : index
      %lo=arith.constant 0 : index
      %hi=arith.constant 2 : index
      %step=arith.constant 1 : index
      %zero=arith.constant 0.0 : f32
      %sum=scf.for %k=%lo to %hi step %step iter_args(%s=%zero) -> (f32) {
        %x=tensor.extract %a[%m,%k] : tensor<1x2xf32>
        %y=tensor.extract %b[%k,%n] : tensor<2x1xf32>
        %p=arith.mulf %x,%y : f32
        %v=arith.addf %s,%p : f32
        scf.yield %v : f32
      }
      linalg.yield %sum : f32
    } -> tensor<1x1xf32>
    func.return %r : tensor<1x1xf32>
  }}"""


def test_scalarized_family_requests_recapture_without_changing_or_admitting_source():
    m = parse_mlir_text(SOURCE)
    op = next(op for op in m.walk() if op.name == "linalg.generic")
    op.attributes["prov.family"] = StringAttr("contraction")
    original = str(m)
    assert scalarized_contraction_candidates(m) == [list(m.walk()).index(op)]
    assert str(m) == original
    assert not any("prov.declarations" in child.attributes for child in m.walk())


@pytest.mark.parametrize("family", ["movement", "elementwise", "unknown"])
def test_other_families_do_not_request_contraction_recapture(family):
    m = parse_mlir_text(SOURCE)
    next(op for op in m.walk() if op.name == "linalg.generic").attributes["prov.family"] = StringAttr(family)
    assert scalarized_contraction_candidates(m) == []


def test_literal_scaling_does_not_claim_a_bilinear_candidate():
    m = parse_mlir_text(SOURCE.replace("%p=arith.mulf %x,%y", "%p=arith.mulf %x,%zero"))
    next(op for op in m.walk() if op.name == "linalg.generic").attributes["prov.family"] = StringAttr("contraction")
    assert scalarized_contraction_candidates(m) == []
