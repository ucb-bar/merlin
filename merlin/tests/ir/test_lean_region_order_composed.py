"""One final all-region traversal composes arithmetic and comparison ordering."""

import pytest
from test_lean_defined_bounds import _chain
from test_lean_motion_independent import _stage
from test_lean_reshape_compose import _canonicalize, _verify_mlir

from merlin.verify.lean.check import check_typed_stage

SOURCE = """func.func @forward(%data: tensor<3xi13>, %initial: tensor<3xi13>) -> tensor<3xi13> {
  %capture = arith.constant 5 : i13
  %result = linalg.generic {
      indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]
    } ins(%data : tensor<3xi13>) outs(%initial : tensor<3xi13>) {
    ^bb0(%x: i13, %old: i13):
      %sum = arith.addi %capture, %x : i13
      %comparison = arith.cmpi slt, %capture, %sum : i13
      %value = arith.select %comparison, %sum, %old : i13
      linalg.yield %value : i13
    } -> tensor<3xi13>
  return %result : tensor<3xi13>
}"""


@pytest.mark.parametrize("bits", [32, 64])
def test_actual_arithmetic_and_comparison_use_one_region_ordering(tmp_path, lean_tools, bits):
    stage, before, after = _canonicalize(tmp_path, lean_tools.m2m_python, SOURCE)
    assert "arith.cmpi sgt" in after.read_text()
    assert before.read_bytes() != after.read_bytes()
    _chain(tmp_path, stage, "forward", lean_tools, bits)


def test_wrong_inverse_predicate_inside_actual_region_remains_refused(tmp_path, lean_tools):
    (tmp_path / "actual").mkdir()
    _, _, after = _canonicalize(tmp_path / "actual", lean_tools.m2m_python, SOURCE)
    target = after.read_text()
    assert "arith.cmpi sgt" in target
    mutant = target.replace("arith.cmpi sgt", "arith.cmpi slt")
    stage = _stage(tmp_path, SOURCE, mutant)
    _verify_mlir(stage / "after.mlir", lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
