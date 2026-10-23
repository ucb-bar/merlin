"""Shape-one coordinates are literal; other index payloads stay unproved."""

import pytest
from mlir_fixtures import assert_unchanged, forward, generic, returned

from merlin.llvmlower.structural_folds import fold_static_unit_indices


def source(axis):
    body = f"%i=linalg.index {axis} : index\n%v=arith.index_cast %i : index to i64"
    return forward("", "tensor<2x1x3xi64>", generic([], "tensor<2x1x3xi64>", body))


def test_unit_axis_is_a_uniform_zero_fill():
    module = source(1)
    assert fold_static_unit_indices(module) > 0
    module.verify()
    assert returned(module).owner.name == "tensor.splat"
    assert returned(module).owner.operands[0].owner.value.value.data == 0


@pytest.mark.parametrize("axis", [0, 2])
def test_nontrivial_coordinate_is_not_a_literal(axis):
    assert_unchanged(source(axis), fold_static_unit_indices)
