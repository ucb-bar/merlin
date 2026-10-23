"""Static reshape selectors preserve every logical storage word."""

from itertools import product
from math import prod

import pytest
from mlir_fixtures import assert_unchanged

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.static_reshape_movement import expose_static_reshape_movement


def typ(shape):
    return "tensor<" + "x".join([*(str(n) for n in shape), "f32"]) + ">"


@pytest.mark.parametrize("before,after", [((), (1,)), ((1,), ()), ((2, 3), (6,)), ((6,), (3, 2)), ((2, 1, 3), (3, 2))])
def test_coordinate_map_covers_identical_flat_storage(before, after):
    shape = ",".join(map(str, after))
    module = parse_mlir_text(f"""module {{ func.func @forward(%x:{typ(before)}) -> {typ(after)} {{
      %s=arith.constant dense<[{shape}]> : tensor<{len(after)}xi64>
      %r=tensor.reshape %x(%s) : ({typ(before)},tensor<{len(after)}xi64>) -> {typ(after)}
      return %r : {typ(after)}
    }} }}""")
    assert expose_static_reshape_movement(module) == 1
    module.verify()
    selected = next(op for op in module.walk() if op.name == "linalg.generic")
    mapping = list(selected.get_indexing_maps())[0].data
    for linear, coordinate in enumerate(product(*(range(n) for n in after))):
        original = mapping.eval(coordinate, ())
        assert sum(index * prod(before[axis + 1 :]) for axis, index in enumerate(original)) == linear
    assert [op.name for op in selected.body.block.ops] == ["linalg.yield"]


def test_runtime_shape_is_not_assumed_to_equal_the_result_annotation():
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<2x3xf32>,%s:tensor<1xi64>) -> tensor<6xf32> {
      %r=tensor.reshape %x(%s) : (tensor<2x3xf32>,tensor<1xi64>) -> tensor<6xf32>
      return %r : tensor<6xf32>
    }}""")
    assert_unchanged(module, expose_static_reshape_movement)
