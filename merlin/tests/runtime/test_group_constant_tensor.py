"""Packed group constants retain the scalar constructor's exact attribute bytes."""

from __future__ import annotations

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType, Signedness, TensorType

from merlin.llvmlower.group_host_call import _constant_tensor
from merlin.xdsl_dialects._common import text


def scalar_constant(array, element):
    return arith.ConstantOp(
        DenseIntOrFPElementsAttr.from_list(
            TensorType(element, list(array.shape)), [int(v) for v in array.reshape(-1).tolist()]
        )
    )


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("signedness", list(Signedness))
@pytest.mark.parametrize("source_kind", ["i", "u"])
def test_matching_integer_width_preserves_complete_scalar_attribute(width, signedness, source_kind):
    dtype = np.dtype(f"{source_kind}{width // 8}")
    bounds = np.iinfo(dtype)
    array = np.array([bounds.min, 0, 1, bounds.max], dtype=dtype).reshape(2, 2)
    element = IntegerType(width, signedness)
    try:
        expected = scalar_constant(array, element)
    except ValueError:
        with pytest.raises(ValueError):
            _constant_tensor(array, element)
    else:
        actual = _constant_tensor(array, element)
        actual.verify()
        assert actual.value == expected.value
        assert actual.value.data.data == expected.value.data.data
        assert text(actual) == text(expected)


@pytest.mark.parametrize("dtype", [">i2", ">u4", "<i8", "i1"])
@pytest.mark.parametrize("layout", ["transpose", "reverse", "empty", "scalar"])
def test_endian_order_views_and_empty_shapes_preserve_serialization(dtype, layout):
    array = np.arange(12, dtype=np.dtype(dtype)).reshape(3, 4)
    if layout == "transpose":
        array = array.T
    elif layout == "reverse":
        array = array[::-1, ::-1]
    elif layout == "empty":
        array = array[:0, :]
    else:
        array = array[1, 2].reshape(())
    element = IntegerType(array.dtype.itemsize * 8)
    actual, expected = _constant_tensor(array, element), scalar_constant(array, element)
    actual.verify()
    assert actual.value == expected.value
    assert text(actual) == text(expected)


@pytest.mark.parametrize(
    "array,width",
    [
        (np.array([-1.9, 0.4, 2.8], dtype=np.float32), 8),
        (np.array([False, True], dtype=np.bool_), 1),
        (np.array([-4, 3], dtype=np.int16), 3),
        (np.array([-128, 127], dtype=np.int64), 8),
        (np.array([-(1 << 63), (1 << 63) - 1], dtype=object), 64),
    ],
)
def test_other_storage_keeps_original_scalar_conversion(array, width):
    element = IntegerType(width)
    assert _constant_tensor(array, element).value == scalar_constant(array, element).value


@pytest.mark.parametrize("value", [128, -129, float("nan"), float("inf")])
def test_invalid_scalar_conversion_still_refuses(value):
    array = np.array([value], dtype=np.float64)
    element = IntegerType(8, Signedness.SIGNED)
    with pytest.raises((ValueError, OverflowError)):
        scalar_constant(array, element)
    with pytest.raises((ValueError, OverflowError)):
        _constant_tensor(array, element)


def test_packed_integer_path_does_not_call_scalar_element_normalizer(monkeypatch):
    def forbidden(*args):
        raise AssertionError("per-element normalization reached packed storage")

    monkeypatch.setattr(IntegerType, "get_normalized_value", forbidden)
    array = np.arange(-128, 128, dtype=np.int8)
    constant = _constant_tensor(array, IntegerType(8))
    assert constant.value.data.data == array.tobytes()
