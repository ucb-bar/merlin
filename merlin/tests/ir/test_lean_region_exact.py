"""Actual named-op parse bodies require exact normalization beyond generic regions."""

import json

import pytest
from test_lean_defined_bounds import _chain
from test_lean_motion_independent import _stage
from test_lean_reshape_compose import _verify_mlir

from merlin.llvmlower import passes_xdsl
from merlin.verify.lean import stages
from merlin.verify.lean.check import check_typed_stage

NAMED_SOURCE = """func.func @forward(%a: tensor<2x2xi32>, %b: tensor<2x2xi32>,
    %init: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %r = linalg.matmul ins(%a, %b : tensor<2x2xi32>, tensor<2x2xi32>)
    outs(%init : tensor<2x2xi32>) -> tensor<2x2xi32>
  return %r : tensor<2x2xi32>
}"""


def _parse(tmp_path):
    with stages.capture(tmp_path / "captured"):
        passes_xdsl.preprocess_text(NAMED_SOURCE)
    stage = stages.completed(tmp_path / "captured")[0]
    assert json.loads((stage / "meta.json").read_bytes())["boundary"] == "xdsl-parse"
    assert (stage / "before.mlir").read_bytes() != (stage / "after.mlir").read_bytes()
    target = (stage / "after.mlir").read_text()
    assert '"linalg.matmul"' in target and '"arith.addi"(%3, %2)' in target
    return stage


@pytest.mark.parametrize("bits", [32, 64])
def test_actual_named_matmul_parse_commutation_has_strict_chain(tmp_path, lean_tools, bits):
    stage = _parse(tmp_path)
    _chain(tmp_path, stage, "forward", lean_tools, bits)


@pytest.mark.parametrize("fault", ["left-input", "right-input", "accumulator", "output"])
def test_valid_named_parse_mutants_do_not_hide_behind_region_normalization(tmp_path, lean_tools, fault):
    captured = _parse(tmp_path)
    target = (captured / "after.mlir").read_text()
    replacements = {
        "left-input": ('"linalg.matmul"(%a, %b, %init)', '"linalg.matmul"(%b, %b, %init)'),
        "right-input": ('"linalg.matmul"(%a, %b, %init)', '"linalg.matmul"(%a, %a, %init)'),
        "accumulator": ('"linalg.matmul"(%a, %b, %init)', '"linalg.matmul"(%a, %b, %a)'),
        "output": ('"func.return"(%r)', '"func.return"(%init)'),
    }
    old, new = replacements[fault]
    assert old in target
    mutant = target.replace(old, new)
    stage = _stage(tmp_path, NAMED_SOURCE, mutant)
    for name in ("before.mlir", "after.mlir"):
        _verify_mlir(stage / name, lean_tools.m2m_python)
    assert check_typed_stage(stage, m2m_python=lean_tools.m2m_python)["verdict"] == "abstracted"
