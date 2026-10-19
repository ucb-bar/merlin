"""Selected scalar math semantics survive linalg evaluation, quantization and constant reuse."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.runtime import linalg_numpy as LN
from merlin.runtime.host_math_reference import host_math_context, reference_math


def program(element="f32"):
    return mq.parse(f"""module {{
      func.func @forward(%x:tensor<1x{element}>) -> tensor<1x{element}> {{
        %empty = tensor.empty() : tensor<1x{element}>
        %out = "linalg.generic"(%x, %empty) <{{
          indexing_maps = [affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types = [#linalg.iterator_type<parallel>],
          operandSegmentSizes = array<i32:1,1>
        }}> ({{ ^bb0(%v:{element}, %unused:{element}):
          %r = math.exp %v : {element}
          linalg.yield %r : {element}
        }}) : (tensor<1x{element}>,tensor<1x{element}>) -> tensor<1x{element}>
        return %out : tensor<1x{element}>
      }}
    }}""")


@pytest.fixture
def selected(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("needs a native C compiler")
    return reference_math("portable_exp_trig", tmp_path / "math")


def residual_probability(value):
    primary = np.rint(value * np.float32(127))
    residual = np.rint((value-primary/np.float32(127))*np.float32(16256))
    return primary, residual


def test_selected_exp_contract_fixes_residual_boundary_without_changing_numpy_default(selected):
    module = program()
    x = np.array([-0.049721017479896545], np.float32)
    native = LN.evaluate(module, [x])[0]
    original_contract = LN.numeric_contract()
    with host_math_context(selected):
        actual = LN.evaluate(module, [x])[0]
        assert LN.numeric_contract()["host_math"] == selected.identity()
    np.testing.assert_array_equal(actual, selected.evaluate("math.exp", x))
    assert residual_probability(actual)[0].item() == 121
    assert residual_probability(actual)[1].item() == -21
    # The same reference remains independent of compiler lowering; only the
    # explicitly declared scalar primitive has changed implementation.
    np.testing.assert_array_equal(LN.evaluate(module, [x])[0], native)
    assert LN.numeric_contract() == original_contract


def test_hoisted_results_cannot_cross_selected_math_implementations(selected):
    module = program()
    x = np.array([-0.049721017479896545], np.float32)
    cache = LN.ConstantCache()
    cache.mark(x)
    LN.evaluate(module, [x], constants=cache)
    with host_math_context(selected), pytest.raises(LN.LinalgNumpyError, match="cache numerical policy"):
        LN.evaluate(module, [x], constants=cache)
    with host_math_context(selected):
        selected_cache = LN.ConstantCache()
        selected_cache.mark(x)
        first = LN.evaluate(module, [x], constants=selected_cache)[0]
        second = LN.evaluate(module, [x], constants=selected_cache)[0]
        np.testing.assert_array_equal(first, second)


def test_explicit_f32_policy_does_not_silently_narrow_f64(selected):
    with host_math_context(selected), pytest.raises(LN.LinalgNumpyError, match="only f32"):
        LN.evaluate(program("f64"), [np.array([.1], np.float64)])
