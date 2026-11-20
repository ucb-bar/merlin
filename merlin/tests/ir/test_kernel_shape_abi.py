"""Tensor emptiness is independent of positive logical launch bounds."""

import pytest

from merlin.frontends.kernel_shape_abi import validate_dimensions


def test_static_empty_tensor_axes_keep_their_actual_rank_without_runtime_bindings():
    spec = {"schema": "t2m.shared-tensor-abi.v2", "grid": [1]}
    assert validate_dimensions(spec, [{"shape": [2, 0, 3], "effect": "write"}], []) is False


@pytest.mark.parametrize("dimension", (-1, False, True, 0.0, ""))
def test_empty_shape_admission_does_not_accept_negative_or_untyped_dimensions(dimension):
    with pytest.raises(ValueError, match="dimensions"):
        validate_dimensions({"schema": "t2m.shared-tensor-abi.v2"}, [{"shape": [dimension]}], [])


@pytest.mark.parametrize("grid", ([0], [-1], [1 << 31], [False]))
def test_empty_tensors_grant_no_nonpositive_or_untyped_grid(grid):
    with pytest.raises(ValueError, match="grid bound"):
        validate_dimensions({"schema": "t2m.shared-tensor-abi.v2", "grid": grid}, [{"shape": [0]}], [])
