"""Shared tanh/pow primitives retain explicit scope, accuracy and binary array semantics."""

import ctypes
import shutil
import subprocess
from decimal import Decimal, localcontext

import numpy as np
import pytest

from merlin.runtime.host_math import build_host_math, host_math_recipe
from merlin.runtime.host_math_reference import host_math_context, reference_math


@pytest.fixture
def neural(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("requires native C compiler")
    return reference_math("portable_neural_f32", tmp_path / "oracle")


def test_neural_policy_extends_only_selected_primitives(neural, tmp_path):
    assert neural.operations == ("math.exp", "math.sin", "math.cos", "math.tanh", "math.powf")
    original = reference_math("portable_exp_trig", tmp_path / "original")
    assert original.operations == ("math.exp", "math.sin", "math.cos")
    for op in ("math.tanh", "math.powf", "math.sqrt", "math.rsqrt"):
        with pytest.raises(ValueError, match="does not select"):
            original.evaluate(op, np.array([1], np.float32))
    assert "-std=c11" in host_math_recipe("portable_exp_trig").compile_flags
    assert "-Wl,--wrap=powf" in host_math_recipe("portable_neural_f32").link_flags


def test_pow_broadcast_arrays_and_special_values(neural):
    left = np.array([[2], [4], [10000]], np.float32)
    right = np.array([0, 0.5, 1, 2], np.float32)
    got = neural.evaluate("math.powf", left, right)
    assert got.shape == (3, 4) and got.dtype == np.dtype("float32")
    np.testing.assert_array_equal(got[[0, 1, 2], [2, 1, 1]], [2, 2, 100])
    x = np.array([-2, -2, -0.0, -0.0, 0, np.inf, -np.inf, np.nan, 1, -1], np.float32)
    y = np.array([3, 0.5, 3, -3, -1, -2, 3, 0, np.nan, np.inf], np.float32)
    special = neural.evaluate("math.powf", x, y)
    assert special[0] == -8 and np.isnan(special[1])
    assert special[2] == 0 and np.signbit(special[2])
    assert np.isneginf(special[3]) and np.isposinf(special[4])
    assert special[5] == 0 and np.isneginf(special[6])
    np.testing.assert_array_equal(special[7:], [1, 1, 1])
    with pytest.raises(ValueError, match="two binary32"):
        neural.evaluate("math.powf", left)
    with pytest.raises(ValueError, match="binary32"):
        neural.evaluate("math.powf", left, np.asarray([1], dtype="float64"))
    with pytest.raises(ValueError, match="one operand"):
        neural.evaluate("math.tanh", left, left)
    assert neural.evaluate("math.powf", np.float32(2), np.float32(3)).shape == ()


def test_tanh_and_pow_decimal_accuracy(neural):
    xs = np.array([-100, -10, -4, -1, -0.5, -1e-20, -0.0, 0.0, 1e-20, 0.5, 1, 4, 10, 100], np.float32)
    got = neural.evaluate("math.tanh", xs)
    for x, y in zip(xs, got, strict=True):
        with localcontext() as ctx:
            ctx.prec = 150
            value = Decimal(float(x))
            e = (2 * value).exp()
            expected = np.float32(float((e - 1) / (e + 1)))
        if x == 0:
            assert y == 0 and np.signbit(y) == np.signbit(x)
        else:
            assert abs(int(y.view("uint32")) - int(expected.view("uint32"))) <= 2
    x = np.array([0.001, 0.5, 2, 10, 10000, 1.0001], np.float32)
    y = np.array([0.25, -5, 0.5, -3, 0.03125, 123], np.float32)
    actual = neural.evaluate("math.powf", x, y)
    for base, exponent, value in zip(x, y, actual, strict=True):
        with localcontext() as ctx:
            ctx.prec = 120
            expected = np.float32(float(ctx.power(Decimal(float(base)), Decimal(float(exponent)))))
        assert abs(int(value.view("uint32")) - int(expected.view("uint32"))) <= 1
    special = neural.evaluate("math.tanh", np.asarray([np.inf, -np.inf, np.nan], np.float32))
    assert special[0] == 1 and special[1] == -1 and np.isnan(special[2])


def test_selected_runtime_symbols_match_array_oracle(tmp_path, neural):
    cc = shutil.which("cc")

    def run(command):
        subprocess.run(list(map(str, command)), check=True, capture_output=True, timeout=30)

    objects, flags = build_host_math("portable_neural_f32", tmp_path / "emitted", cc, ["-O2", "-fPIC"], run)
    caller = tmp_path / "caller.c"
    caller.write_text(
        "#include <math.h>\nfloat p(float x,float y){return powf(x,y);}\nfloat t(float x){return tanhf(x);}\n"
    )
    run([cc, "-O2", "-fPIC", "-fno-builtin", "-c", caller, "-o", tmp_path / "caller.o"])
    shared = tmp_path / "emitted.so"
    run([cc, "-shared", tmp_path / "caller.o", *objects, *flags, "-lm", "-o", shared])
    lib = ctypes.CDLL(str(shared))
    lib.p.argtypes = [ctypes.c_float, ctypes.c_float]
    lib.p.restype = ctypes.c_float
    lib.t.argtypes = [ctypes.c_float]
    lib.t.restype = ctypes.c_float
    x = np.linspace(-8, 8, 1001, dtype="float32")
    actual = np.asarray([lib.t(float(v)) for v in x], dtype="float32")
    np.testing.assert_array_equal(actual.view("uint32"), neural.evaluate("math.tanh", x).view("uint32"))
    x = np.float32(10000)
    y = np.arange(32, dtype="float32") / np.float32(32)
    actual = np.asarray([lib.p(float(x), float(v)) for v in y], dtype="float32")
    np.testing.assert_array_equal(actual.view("uint32"), neural.evaluate("math.powf", x, y).view("uint32"))


def test_linalg_binary_and_unary_operations_use_the_declared_policy(neural):
    from merlin.common import mlir_query as mq
    from merlin.runtime import linalg_numpy as LN

    module = mq.parse("""module {
      func.func @forward(%x:tensor<3xf32>, %y:tensor<3xf32>) -> tensor<3xf32> {
        %empty = tensor.empty() : tensor<3xf32>
        %out = "linalg.generic"(%x, %y, %empty) <{
          indexing_maps = [affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32:2,1>
        }> ({ ^bb0(%a:f32, %b:f32, %unused:f32):
          %p = math.powf %a, %b : f32
          %t = math.tanh %p : f32
          linalg.yield %t : f32
        }) : (tensor<3xf32>,tensor<3xf32>,tensor<3xf32>) -> tensor<3xf32>
        return %out : tensor<3xf32>
      }
    }""")
    x = np.array([0.5, 2, 10000], np.float32)
    y = np.array([1, 0.5, -0.03125], np.float32)
    with host_math_context(neural):
        got = LN.evaluate(module, [x, y])[0]
        assert LN.numeric_contract()["host_math"]["operations"] == list(neural.operations)
    expected = neural.evaluate("math.tanh", neural.evaluate("math.powf", x, y))
    np.testing.assert_array_equal(got.view("uint32"), expected.view("uint32"))


@pytest.mark.parametrize("operation", ["exp", "sin", "cos", "tanh", "powf"])
def test_binary64_operations_remain_native_under_binary32_policy(neural, monkeypatch, operation):
    from merlin.common import mlir_query as mq
    from merlin.runtime import linalg_numpy as LN

    scalar = "math.powf %a, %b" if operation == "powf" else f"math.{operation} %a"
    module = mq.parse(f"""module {{
      func.func @forward(%x:tensor<3xf64>, %y:tensor<3xf64>) -> tensor<3xf64> {{
        %empty = tensor.empty() : tensor<3xf64>
        %out = "linalg.generic"(%x, %y, %empty) <{{
          indexing_maps = [affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types = [#linalg.iterator_type<parallel>], operandSegmentSizes = array<i32:2,1>
        }}> ({{ ^bb0(%a:f64, %b:f64, %unused:f64):
          %value = {scalar} : f64
          linalg.yield %value : f64
        }}) : (tensor<3xf64>,tensor<3xf64>,tensor<3xf64>) -> tensor<3xf64>
        return %out : tensor<3xf64>
      }}
    }}""")
    x = np.asarray([0.004, 1.00000000000001, 1000], dtype="float64")
    y = np.asarray([0.125, -0.5, 1 / 359], dtype="float64")
    with np.errstate(over="ignore"):
        expected = LN.evaluate(module, [x, y])[0]

        def refuse_unselected_binary64(*args, **kwargs):
            pytest.fail("binary64 was dispatched to the binary32 adapter")

        monkeypatch.setattr(neural, "evaluate", refuse_unselected_binary64)
        with host_math_context(neural):
            actual = LN.evaluate(module, [x, y])[0]
            assert LN.numeric_contract()["host_math"]["element_type"] == "f32"
    assert actual.dtype == np.dtype("float64")
    np.testing.assert_array_equal(actual.view("uint64"), expected.view("uint64"))
