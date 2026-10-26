"""Representation axes do not change a selector's observable tensor boundary."""

import pytest
from mlir_fixtures import forward, generic, pointwise
from xdsl.dialects.builtin import StringAttr

from merlin.llvmlower.static_reshape_movement import flatten_internal_selectors, lift_scalar_producers


def packed_selector(*, escapes=False, dynamic_shape=False):
    body = generic(
        [("%x", "tensor<2x3xi8>")], "tensor<2x1x3xi8>", "", yields="%a",
        maps=["affine_map<(d0,d1,d2)->(d0,d2)>", "affine_map<(d0,d1,d2)->(d0,d1,d2)>"],
    )
    if escapes:
        return forward("%x:tensor<2x3xi8>", "tensor<2x1x3xi8>", body)
    shape = "%s" if dynamic_shape else "%shape"
    params = "%x:tensor<2x3xi8>" + (",%s:tensor<1xi64>" if dynamic_shape else "")
    body += f"""%shape=arith.constant dense<[6]> : tensor<1xi64>
      %out=tensor.reshape %r({shape}) : (tensor<2x1x3xi8>,tensor<1xi64>) -> tensor<6xi8>"""
    return forward(params, "tensor<6xi8>", body, ret="%out")


def test_private_packing_axes_flatten_to_the_same_element_selectors():
    module = packed_selector()
    assert flatten_internal_selectors(module, ranks=[0, 1, 2]) == 1
    operation = next(op for op in module.walk() if op.name == "linalg.generic")
    select = list(operation.get_indexing_maps())[0].data
    assert [select.eval([i], []) for i in range(6)] == [(i // 3, i % 3) for i in range(6)]
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert str(returned.operands[0].type) == "tensor<6xi8>"
    module.verify()


@pytest.mark.parametrize("options", [dict(escapes=True), dict(dynamic_shape=True)])
def test_observable_packing_rank_and_unproved_runtime_shape_are_retained(options):
    module = packed_selector(**options)
    source = str(module)
    assert flatten_internal_selectors(module, ranks=[0, 1, 2]) == 0
    assert str(module) == source


def test_one_element_components_keep_scalar_results_and_original_integer_types():
    module = pointwise(["tensor<i8>"], "tensor<i32>", """%wide=arith.extsi %a : i8 to i32
      %negative=arith.constant -1 : i32
      %v=arith.muli %wide,%negative : i32""")
    root = next(op for op in module.walk() if op.name == "linalg.generic")
    root.attributes["test.component"] = StringAttr("proved")
    assert lift_scalar_producers(module, attribute="test.component") == 1
    root = next(op for op in module.walk() if "test.component" in op.attributes)
    assert str(root.results[0].type) == "tensor<1xi32>"
    assert str(next(op for op in root.walk() if op.name == "arith.muli").results[0].type) == "i32"
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert str(returned.operands[0].type) == "tensor<i32>"
    module.verify()
