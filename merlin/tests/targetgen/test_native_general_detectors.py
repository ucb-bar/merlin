"""General excluded constructions read from IR properties, never from operator names."""

import pytest
from mlir_fixtures import forward, generic, pointwise

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def contraction(table):
    return parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<2x4xf32>) -> tensor<2x3xf32> {{
      %w = arith.constant dense<{table}> : tensor<4x3xf32>
      %z = arith.constant 0.0 : f32
      %e = tensor.splat %z : tensor<2x3xf32>
      %r = linalg.generic {{indexing_maps = [affine_map<(i,j,k)->(i,k)>, affine_map<(i,j,k)->(k,j)>, affine_map<(i,j,k)->(i,j)>],
                            iterator_types = ["parallel", "parallel", "reduction"]}}
        ins(%x, %w : tensor<2x4xf32>, tensor<4x3xf32>) outs(%e : tensor<2x3xf32>) {{
        ^bb0(%a: f32, %b: f32, %acc: f32):
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %acc, %p : f32
          linalg.yield %s : f32
      }} -> tensor<2x3xf32>
      func.return %r : tensor<2x3xf32>
    }} }}""")


def decide(module, reason):
    """The decision with only ``reason`` excluded; a host decision must cite exactly that reason."""
    result = construction_evidence(module, excluded=[reason])
    if result["status"] == "host":
        assert {e["reason"] for e in result["evidence"]} == {reason}
    return result["status"]


@pytest.mark.parametrize(
    "table,status",
    [
        # DFT twiddles (real part, n = 4): +-1/4 and 0.
        ("[[0.25,0.25,0.25],[0.25,0.0,-0.25],[0.25,-0.25,0.25],[0.25,0.0,-0.25]]", "host"),
        # Interpolation weights.
        ("[[0.75,0.25,0.0],[0.25,0.75,0.0],[0.0,0.75,0.25],[0.0,0.25,0.75]]", "host"),
        ("[[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]", "eligible"),  # 0/1 selection, prefix sums
        ("[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5],[0.0,0.0,0.0]]", "eligible"),  # scaled identity
        ("1.0", "eligible"),  # ones (window sums)
    ],
)
def test_only_a_multi_valued_literal_coefficient_table_is_excluded(table, status):
    assert decide(contraction(table), "nonuniform_literal_coefficient_contraction") == status


def division(divisor, op="arith.divf"):
    prelude = f"%d = arith.constant {divisor} : f32\n%ds = tensor.splat %d : tensor<4xf32>\n"
    body = f'%v = "{op}"(%a, %b) : (f32, f32) -> f32'
    computation = generic([("%x", "tensor<4xf32>"), ("%ds", "tensor<4xf32>")], "tensor<4xf32>", body)
    return forward("%x:tensor<4xf32>", "tensor<4xf32>", prelude + computation)


@pytest.mark.parametrize("op", ["arith.divf", "arith.remf"])
@pytest.mark.parametrize("divisor,status", [("0.0", "host"), ("-0.0", "host"), ("2.0", "eligible")])
def test_only_division_by_a_closed_zero_forces_ieee_specials(op, divisor, status):
    assert decide(division(divisor, op), "forced_ieee_special_division") == status


def test_zero_divisor_in_a_discarded_branch_is_not_forced():
    # Smith-style complex division by a real literal: c/0 appears only in the
    # alternative a closed |re| >= |im| test discards.
    body = """%re = arith.constant -2.0 : f32
      %im = arith.constant 0.0 : f32
      %ar = math.absf %re : f32
      %ai = math.absf %im : f32
      %p = arith.cmpf oge, %ar, %ai : f32
      %t = arith.divf %im, %re : f32
      %s = arith.divf %re, %im : f32
      %d1 = arith.mulf %a, %t : f32
      %d2 = arith.mulf %a, %s : f32
      %v = arith.select %p, %d1, %d2 : f32"""
    module = pointwise(["tensor<4xf32>"], "tensor<4xf32>", body)
    assert construction_evidence(module, excluded=["forced_ieee_special_division"])["status"] == "eligible"


def reduction(accumulate):
    return parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<2x3xf32>) -> tensor<3xf32> {{
      %one = arith.constant 1.0 : f32
      %init = tensor.splat %one : tensor<3xf32>
      %r = linalg.reduce ins(%x : tensor<2x3xf32>) outs(%init : tensor<3xf32>) dimensions = [0]
        (%in: f32, %acc: f32) {{
          %p = {accumulate} %in, %acc : f32
          linalg.yield %p : f32
        }}
      func.return %r : tensor<3xf32>
    }} }}""")


@pytest.mark.parametrize("accumulate,status", [("arith.mulf", "host"), ("arith.addf", "eligible")])
def test_only_a_product_reduction_over_runtime_elements_is_excluded(accumulate, status):
    assert decide(reduction(accumulate), "runtime_product_reduction") == status
