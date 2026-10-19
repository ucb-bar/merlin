"""Versioned floating reductions agree with source loops before integer quantization."""

from __future__ import annotations

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.runtime import linalg_numpy as LN


def source(width, *, kind="named", element="f32", flags="none", operation="arith.addf"):
    input_type, output_type = f"tensor<2x{width}x{element}>", f"tensor<2x{element}>"
    body = f'''^bb0(%value: {element}, %acc: {element}):
      %next = "{operation}"(%value, %acc) <{{fastmath = #arith.fastmath<{flags}>}}>
        : ({element}, {element}) -> {element}
      "linalg.yield"(%next) : ({element}) -> ()'''
    if kind == "named":
        header = '"linalg.reduce"(%x, %init) <{dimensions = array<i64: 1>}>'
    else:
        header = (
            '"linalg.generic"(%x, %init) <{indexing_maps = '
            "[affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0)>], "
            "iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "
            "operandSegmentSizes = array<i32: 1,1>}>"
        )
    return f"""module {{
      func.func @forward(%x: {input_type}, %init: {output_type}) -> {output_type} {{
        %out = {header} ({{{body}}}) : ({input_type}, {output_type}) -> {output_type}
        return %out : {output_type}
      }}
    }}"""


def evaluate(values, initial, **kwargs):
    return LN.evaluate(mq.parse(source(values.shape[-1], **kwargs)), [values, initial])[0]


@pytest.mark.parametrize("kind", ["named", "generic"])
def test_cancellation_and_original_initial_accumulator_follow_source_order(kind):
    values = np.array([[1e8, 1, 1, 1, -1e8, 1, 1, 1]] * 2, dtype=np.float32)
    got = evaluate(values, np.zeros(2, np.float32), kind=kind)
    np.testing.assert_array_equal(got, np.array([3, 3], np.float32))
    assert np.add.reduce(values[0]) == 0  # The old reference's independent pairwise fork.
    values = np.array([[1e20, -1e20, 3.14]] * 2, dtype=np.float32)
    got = evaluate(values, np.ones(2, np.float32), kind=kind)
    np.testing.assert_array_equal(got, np.array([3.14, 3.14], np.float32))
    assert np.float32(np.add.reduce(values[0]) + np.float32(1)) != got[0]


@pytest.mark.parametrize("kind", ["named", "generic"])
@pytest.mark.parametrize("flags", ["none", "contract", "nnan,ninf"])
def test_other_fastmath_flags_do_not_grant_reassociation(kind, flags):
    values = np.array([[1e8, 1, 1, 1, -1e8, 1, 1, 1]] * 2, dtype=np.float32)
    np.testing.assert_array_equal(evaluate(values, np.zeros(2, np.float32), kind=kind, flags=flags), [3, 3])


@pytest.mark.parametrize("kind", ["named", "generic"])
@pytest.mark.parametrize("flags", ["reassoc", "fast"])
def test_explicit_reassociation_keeps_the_existing_permitted_numpy_tree(kind, flags):
    values = np.array([[1e8, 1, 1, 1, -1e8, 1, 1, 1]] * 2, dtype=np.float32)
    got = evaluate(values, np.zeros(2, np.float32), kind=kind, flags=flags)
    np.testing.assert_array_equal(got, np.add.reduce(values, axis=-1))
    np.testing.assert_array_equal(got, [0, 0])


@pytest.mark.parametrize("kind", ["named", "generic"])
@pytest.mark.parametrize("flags", ["none", "reassoc"])
@pytest.mark.parametrize("element,large,dtype", [("f16", 2048, np.float16), ("bf16", 256, np.float32)])
def test_each_add_rounds_at_its_declared_type(kind, flags, element, large, dtype):
    values = np.array([[large, 1, 1, -large]] * 2, dtype=dtype)
    initial = np.zeros(2, dtype)
    if element == "bf16":
        values = (values.view(np.uint32) >> 16).astype(np.uint16)
        initial = (initial.view(np.uint32) >> 16).astype(np.uint16)
    result = evaluate(values, initial, kind=kind, element=element, flags=flags)
    np.testing.assert_array_equal(result, np.zeros(2, result.dtype))


@pytest.mark.parametrize("kind", ["named", "generic"])
def test_empty_reduction_preserves_negative_zero_initial_bits(kind):
    initial = np.array([-0.0, -0.0], np.float32)
    result = evaluate(np.empty((2, 0), np.float32), initial, kind=kind)
    np.testing.assert_array_equal(result.view(np.uint32), initial.view(np.uint32))


def test_named_multi_axis_reduction_uses_source_lexicographic_indices_with_noncontiguous_data():
    text = """module {
      func.func @forward(%x: tensor<2x2x4xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
        %out = "linalg.reduce"(%x,%init) <{dimensions = array<i64: 0,2>}> ({
        ^bb0(%value:f32,%acc:f32):
          %sum = arith.addf %value,%acc : f32
          linalg.yield %sum : f32
        }) : (tensor<2x2x4xf32>,tensor<2xf32>) -> tensor<2xf32>
        return %out : tensor<2xf32>
      }
    }"""
    first = np.array([[1e8, 1, 1, 1], [-1e8, 1, 1, 1]], np.float32)
    values = np.stack([first, first], axis=1)
    padded = np.empty((2, 2, 8), np.float32)
    padded[..., ::2] = values
    values = padded[..., ::2]
    assert not values.flags.c_contiguous
    got = LN.evaluate(mq.parse(text), [values, np.zeros(2, np.float32)])[0]
    np.testing.assert_array_equal(got, [3, 3])


@pytest.mark.parametrize("kind", ["named", "generic"])
def test_half_integer_quantization_observes_the_source_sum(kind):
    values = np.array([[1e8, 1, 1, 1, -1e8, 1, 1, 1]] * 2, np.float32)
    result = evaluate(values, np.zeros(2, np.float32), kind=kind)
    # Same strict RNE boundary as the prepared program; widening final tolerances cannot fix it.
    np.testing.assert_array_equal(np.rint(result / np.float32(2)).astype(np.int8), [2, 2])
    np.testing.assert_array_equal(np.rint(np.add.reduce(values, axis=-1) / np.float32(2)).astype(np.int8), [0, 0])


@pytest.mark.parametrize("kind", ["named", "generic"])
def test_ordered_resource_refusal_does_not_reassociate_silently(kind):
    values = np.ones((2, LN.SEQUENTIAL_REDUCTION_LIMIT + 1), np.float32)
    with pytest.raises(LN.LinalgNumpyError, match="source-ordered reduction"):
        evaluate(values, np.zeros(2, np.float32), kind=kind)
    got = evaluate(values, np.zeros(2, np.float32), kind=kind, flags="reassoc")
    np.testing.assert_array_equal(got, [LN.SEQUENTIAL_REDUCTION_LIMIT + 1] * 2)


def test_interpreter_agrees_with_an_independently_compiled_strict_source_loop(tmp_path):
    clang = shutil.which("clang")
    if clang is None:
        pytest.skip("host Clang unavailable; pure interpreter reduction tests still run")
    source_file, library = tmp_path / "source_loop.c", tmp_path / "source_loop.so"
    source_file.write_text("""float source_loop(const float *x,int n,float init){
      float acc=init;for(int i=0;i<n;i++)acc=x[i]+acc;return acc;
    }""")
    subprocess.run(
        [clang, "-shared", "-fPIC", "-O2", "-fno-fast-math", "-ffp-contract=off", str(source_file), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    loop = ctypes.CDLL(str(library)).source_loop
    loop.argtypes, loop.restype = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float], ctypes.c_float
    for values, init in [([1e8, 1, 1, 1, -1e8, 1, 1, 1], 0), ([1e20, -1e20, 3.14], 1)]:
        array = np.array([values] * 2, np.float32)
        expected = loop(array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(values), init)
        for kind in ("named", "generic"):
            got = evaluate(array, np.full(2, init, np.float32), kind=kind)
            np.testing.assert_array_equal(got, np.full(2, expected, np.float32))


def test_reference_policy_is_explicit_versioned_and_returns_fresh_data():
    policy = LN.numeric_contract()
    assert policy["version"] == "merlin.linalg_numpy.numeric.v2.strict_float_reductions"
    policy["version"] = "changed by caller"
    assert LN.numeric_contract()["version"] == LN.NUMERIC_CONTRACT_VERSION


@pytest.mark.parametrize("kind", ["named", "generic"])
def test_f64_does_not_narrow_to_f32(kind):
    data = np.array([[1e16, 1, 1, 1, -1e16, 1, 1, 1]] * 2, np.float64)
    result = evaluate(data, np.zeros(2, np.float64), kind=kind, element="f64")
    np.testing.assert_array_equal(result, np.array([3, 3], np.float64))
    assert result.dtype == np.float64


@pytest.mark.parametrize("kind", ["named", "generic"])
def test_nonfinite_intermediate_is_not_eliminated_by_reassociation(kind):
    maximum = np.finfo(np.float32).max
    data = np.array([[maximum, maximum, -maximum, -maximum]] * 2, np.float32)
    result = evaluate(data, np.zeros(2, np.float32), kind=kind)
    assert np.isposinf(result).all()
