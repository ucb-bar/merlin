"""Literal transport exposes a device fill without reading runtime inputs."""

import pytest
from mlir_fixtures import assert_unchanged, returned

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import expose_uniform_literal_carriers


@pytest.mark.parametrize("element,value", [("i64", "0"), ("i1", "1"), ("f32", "-0.0"), ("f32", "0xFFC00001")])
def test_singleton_rank_change_preserves_the_literal_attribute(element, value):
    module = parse_mlir_text(f"""module {{ func.func @forward() -> tensor<1x{element}> {{
      %c = arith.constant {value} : {element}
      %v = tensor.splat %c : tensor<{element}>
      %s = arith.constant dense<1> : tensor<1xi64>
      %r = tensor.reshape %v(%s) : (tensor<{element}>, tensor<1xi64>) -> tensor<1x{element}>
      return %r : tensor<1x{element}>
    }} }}""")
    before = next(op for op in module.walk() if op.name == "arith.constant").value
    assert expose_uniform_literal_carriers(module) == 1
    assert returned(module).owner.name == "tensor.splat"
    assert returned(module).owner.operands[0].owner.value == before
    assert expose_uniform_literal_carriers(module) == 0


@pytest.mark.parametrize("source", ["%x", "%opaque"])
def test_runtime_or_unknown_carrier_is_not_a_literal_fill(source):
    module = parse_mlir_text(f"""module {{ func.func @forward(%x:tensor<i64>) -> tensor<1xi64> {{
      %opaque = "fixture.opaque"(%x) : (tensor<i64>) -> tensor<i64>
      %s = arith.constant dense<1> : tensor<1xi64>
      %r = tensor.reshape {source}(%s) : (tensor<i64>, tensor<1xi64>) -> tensor<1xi64>
      return %r : tensor<1xi64>
    }} }}""")
    assert_unchanged(module, expose_uniform_literal_carriers)
