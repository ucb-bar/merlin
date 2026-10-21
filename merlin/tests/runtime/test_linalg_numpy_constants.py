"""Packed integer reads preserve the scalar unpacker's complete values and storage."""

from __future__ import annotations

import numpy as np
import pytest
from xdsl.dialects import arith
from xdsl.dialects.builtin import BytesAttr, DenseIntOrFPElementsAttr, IntegerType, Signedness, TensorType, f32

from merlin.runtime import linalg_numpy as LN


def scalar_constant(op):
    value, rtype = op.properties["value"], op.results[0].type
    shape, element = LN._shape(rtype), LN._elem(rtype)
    values = list(value.get_values())
    if len(values) == 1 and int(np.prod(shape, dtype=np.int64)) != 1:
        array = np.full(shape, values[0], dtype=LN._compute_dtype(element))
    else:
        array = np.asarray(values, dtype=LN._compute_dtype(element)).reshape(shape)
    return LN._store(array, element)


def constant(array, element, shape=None):
    shape = array.shape if shape is None else shape
    return arith.ConstantOp(DenseIntOrFPElementsAttr(TensorType(element, shape), BytesAttr(array.tobytes())))


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("shape", [(2, 3), (), (0, 3)])
def test_packed_integer_values_shape_dtype_and_writeability_equal_scalar(width, shape):
    dtype = np.dtype(f"<i{width // 8}")
    limits = np.iinfo(dtype)
    array = np.array([limits.min, limits.max, -1, 0, 1, limits.max], dtype=dtype)
    array = array[: int(np.prod(shape))].reshape(shape)
    op = constant(array, IntegerType(width))
    expected, actual = scalar_constant(op), LN._constant(op)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert actual.flags.writeable == expected.flags.writeable is True
    if actual.size:
        original = op.value.data.data
        actual.flat[0] = 0
        assert op.value.data.data == original
        assert LN._constant(op).tobytes() == expected.tobytes()


@pytest.mark.parametrize("width", [8, 16, 32, 64])
@pytest.mark.parametrize("shape", [(2, 3), (0, 3)])
def test_single_packed_value_retains_existing_splat_and_empty_semantics(width, shape):
    op = constant(np.array([-3], dtype=f"<i{width // 8}"), IntegerType(width), shape)
    expected, actual = scalar_constant(op), LN._constant(op)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes() and actual.flags.writeable


def test_little_endian_payload_is_decoded_in_storage_order():
    op = arith.ConstantOp(
        DenseIntOrFPElementsAttr(TensorType(IntegerType(32), [2]), BytesAttr(b"\x04\x03\x02\x01\xfe\xff\xff\xff"))
    )
    assert LN._constant(op).tolist() == [0x01020304, -2]


def test_packed_read_does_not_unpack_per_element(monkeypatch):
    op = constant(np.arange(-128, 128, dtype=np.int8), IntegerType(8))

    def forbidden(*args):
        raise AssertionError("packed integer read called scalar unpack")

    monkeypatch.setattr(DenseIntOrFPElementsAttr, "get_values", forbidden)
    assert LN._constant(op).tobytes() == op.value.data.data


@pytest.mark.parametrize(
    "element,array", [(f32, np.array([np.nan, np.inf, -0.0], np.float32)), (IntegerType(1), np.array([0, 1], np.int8))]
)
def test_other_types_keep_scalar_reader(monkeypatch, element, array):
    op = constant(array, element)
    expected = scalar_constant(op)
    calls = []
    original = DenseIntOrFPElementsAttr.get_values

    def observed(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(DenseIntOrFPElementsAttr, "get_values", observed)
    actual = LN._constant(op)
    assert calls == [op.value]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("signedness", [Signedness.SIGNED, Signedness.UNSIGNED])
def test_unsupported_explicit_signedness_still_refuses(signedness):
    op = constant(np.array([1], np.int8), IntegerType(8, signedness))
    with pytest.raises(LN.LinalgNumpyError):
        scalar_constant(op)
    with pytest.raises(LN.LinalgNumpyError):
        LN._constant(op)


def test_malformed_payload_keeps_original_scalar_refusal():
    op = arith.ConstantOp(DenseIntOrFPElementsAttr(TensorType(IntegerType(32), [1]), BytesAttr(b"\x00")))
    with pytest.raises(ValueError) as expected:
        scalar_constant(op)
    with pytest.raises(ValueError) as actual:
        LN._constant(op)
    assert str(actual.value) == str(expected.value)


def test_different_attribute_and_result_type_keeps_scalar_conversion(monkeypatch):
    value = DenseIntOrFPElementsAttr.from_list(TensorType(IntegerType(8), [2]), [-128, 127])
    op = arith.ConstantOp.create(properties={"value": value}, result_types=[TensorType(IntegerType(16), [2])])
    expected = scalar_constant(op)
    calls = []
    original = DenseIntOrFPElementsAttr.get_values

    def observed(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(DenseIntOrFPElementsAttr, "get_values", observed)
    actual = LN._constant(op)
    assert calls == [value]
    assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()
