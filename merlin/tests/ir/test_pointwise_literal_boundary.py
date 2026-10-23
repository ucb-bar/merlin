"""A literal inside one scalar boundary has the same explicit tensor operand."""

import pytest
from mlir_fixtures import pointwise

from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives


@pytest.mark.parametrize("operation", ["mulf", "divf", "addf", "subf"])
def test_literal_weight_is_exposed_without_changing_the_typed_scalar_boundary(operation):
    body = f"%c=arith.constant 1.0 : f32\n%v=arith.{operation} %a,%c : f32"
    module = pointwise(["tensor<4xf32>"], "tensor<4xf32>", body)
    assert expose_pointwise_primitives(module) == 1
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    assert len(roots) == 1 and len(roots[0].inputs) == 2
    assert roots[0].inputs[1].owner.name == "tensor.splat"
    assert roots[0].body.block.first_op.name == "arith." + operation
    assert list(roots[0].body.block.first_op.operands) == list(roots[0].body.block.args[:2])
    module.verify()
