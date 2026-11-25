"""Reference storage must match the output ABI before numerical comparison."""

import importlib.util

import numpy as np
import pytest

from merlin.common.paths import repo_root


def canonical():
    path = repo_root() / "examples/gemmini/model-study/reference.py"
    spec = importlib.util.spec_from_file_location("captured_model_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.canonical_outputs


def test_boolean_storage_is_typed_without_changing_a_byte_or_float_state():
    mask = np.array([[0, 1, 1, 0]], dtype=np.int8)
    state = np.array([-0.0, 1.125, -7.25], dtype=np.float32)
    result, records = canonical()([mask, state], [([1, 4], "i1"), ([3], "f32")], finite_required=True)
    assert result[0].dtype == np.bool_ and result[0].tobytes() == mask.tobytes()
    assert result[1].dtype == state.dtype and result[1].tobytes() == state.tobytes()
    assert all(row["storage_bytes_unchanged"] for row in records)


def test_scalar_output_keeps_zero_rank_and_noncanonical_boolean_storage_refuses():
    scalar = np.array(-0.0, dtype=np.float32)
    result, _ = canonical()([scalar], [([], "f32")], finite_required=True)
    assert result[0].shape == () and result[0].tobytes() == scalar.tobytes()
    invalid = np.frombuffer(bytes([2]), dtype=np.bool_)
    assert bool(invalid[0])
    with pytest.raises(ValueError, match="noncanonical storage"):
        canonical()([invalid], [([1], "i1")], finite_required=True)


@pytest.mark.parametrize(
    "values,specs",
    [
        ([np.array([0, 2], dtype=np.int8)], [([2], "i1")]),
        ([np.array([0, 1], dtype=np.int32)], [([2], "i1")]),
        ([np.array([0, 1], dtype=np.float32)], [([2], "i1")]),
        ([np.zeros(2, dtype=np.float64)], [([2], "f32")]),
        ([np.zeros(2, dtype=np.int64)], [([2], "i32")]),
        ([np.zeros(2, dtype=np.float32)], [([1, 2], "f32")]),
        ([], [([2], "f32")]),
        ([np.array([np.nan], dtype=np.float32)], [([1], "f32")]),
    ],
)
def test_reference_refuses_invalid_storage_shape_narrowing_and_nonfinite(values, specs):
    with pytest.raises(ValueError):
        canonical()(values, specs, finite_required=True)
