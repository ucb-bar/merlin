"""One selected math implementation owns reference and emitted scalar results."""

import ctypes
import shutil
import subprocess
from decimal import Decimal, localcontext

import numpy as np
import pytest

from merlin.runtime.host_math import build_host_math
from merlin.runtime.host_math_reference import active_host_math, host_math_context, reference_math


@pytest.fixture
def portable(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("requires native C compiler")
    return reference_math("portable_exp_trig", tmp_path)


def test_native_and_unknown_policy_do_not_resolve_tools_or_write(tmp_path):
    work = tmp_path / "absent"
    assert reference_math("native", work, "absent-compiler") is None
    assert not work.exists()
    for policy in (None, True, "portable_trig", "expf_via_double", "portable_exp_trig "):
        with pytest.raises(ValueError, match="host_math_policy"):
            reference_math(policy, work, "absent-compiler")
    assert not work.exists()
    assert active_host_math() is None


def test_array_contract_special_values_and_scope(portable):
    x = np.asarray([[0.0, -0.0, 1.0, -1.0], [float("inf"), float("-inf"), float("nan"), -104]], dtype="float32")
    out = portable.evaluate("math.exp", x[:, ::-1])
    assert out.shape == (2, 4) and out.dtype == np.dtype("float32")
    assert out[0, 2] == 1 and out[0, 3] == 1
    assert out[1, 0] == 0 and np.isnan(out[1, 1]) and out[1, 2] == 0 and np.isposinf(out[1, 3])
    for op in ("math.sin", "math.cos"):
        trig = portable.evaluate(op, x)
        assert np.isnan(trig[1, :3]).all()
    assert np.signbit(portable.evaluate("math.sin", np.asarray(-0.0, dtype="float32")))
    assert portable.evaluate("math.exp", np.asarray(0.0, dtype="float32")).shape == ()
    assert portable.evaluate("math.exp", np.empty((0, 3), dtype="float32")).shape == (0, 3)
    for value in (np.asarray([1.0], dtype="float64"), np.asarray([1], dtype="int32")):
        with pytest.raises(ValueError, match="binary32"):
            portable.evaluate("math.exp", value)
    with pytest.raises(ValueError, match="does not select"):
        portable.evaluate("math.exp2", x)


def test_exp_known_quantization_boundary_and_decimal_accuracy(portable):
    value = np.float32(-0.049721017479896545)
    got = portable.evaluate("math.exp", np.asarray([value], dtype="float32"))[0]
    with localcontext() as ctx:
        ctx.prec = 120
        exact = np.float32(float(Decimal(float(value)).exp()))
    assert got.tobytes() == exact.tobytes()
    primary = np.rint(got * np.float32(127))
    residual = np.rint((got - primary / np.float32(127)) * np.float32(16256))
    assert primary == 121 and residual == -21
    # Upstream's bound is approximately 0.502 ULP, rather than correct rounding for every input.
    values = np.asarray([-103.5, -100, -90, -16.125, -1, -0.03125, 0, 0.5, 1, 10.125, 80, 88, 88.7], dtype="float32")
    actual = portable.evaluate("math.exp", values)
    for x, y in zip(values, actual, strict=True):
        with localcontext() as ctx:
            ctx.prec = 120
            expected = np.float32(float(Decimal(float(x)).exp()))
        assert abs(int(y.view("uint32")) - int(expected.view("uint32"))) <= 1
    extremes = portable.evaluate("math.exp", np.asarray([-1e38, 1e38], dtype="float32"))
    assert extremes[0] == 0 and np.isposinf(extremes[1])


def test_emitted_recipe_and_array_oracle_agree_for_every_selected_function(tmp_path, portable):
    cc = shutil.which("cc")

    def run(command):
        subprocess.run(list(map(str, command)), check=True, capture_output=True, timeout=30)

    objects, flags = build_host_math("portable_exp_trig", tmp_path / "runtime", cc, ["-O2", "-fPIC"], run)
    caller = tmp_path / "caller.c"
    caller.write_text(
        "#include <math.h>\nfloat e(float x){return expf(x);}\n"
        "float s(float x){return sinf(x);}\nfloat c(float x){return cosf(x);}\n"
    )
    run([cc, "-O2", "-fPIC", "-fno-builtin", "-c", caller, "-o", tmp_path / "caller.o"])
    library = tmp_path / "runtime.so"
    run([cc, "-shared", tmp_path / "caller.o", *objects, *flags, "-lm", "-o", library])
    lib = ctypes.CDLL(str(library))
    values = np.asarray(
        [
            -103.5,
            -90,
            -16.125,
            -1,
            -0.049721017479896545,
            -0.0,
            0.0,
            0.5,
            1,
            10.125,
            80,
            88,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype="float32",
    )
    for symbol, op in (("e", "math.exp"), ("s", "math.sin"), ("c", "math.cos")):
        function = getattr(lib, symbol)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
        actual = np.asarray([function(float(x)) for x in values], dtype="float32")
        expected = portable.evaluate(op, values)
        assert np.array_equal(actual[~np.isnan(expected)].view("uint32"), expected[~np.isnan(expected)].view("uint32"))
        assert np.array_equal(np.isnan(actual), np.isnan(expected))


def test_context_restores_on_failure_and_identity_reopens_products(tmp_path, portable):
    assert active_host_math() is None
    with host_math_context(portable):
        assert active_host_math() is portable
        with pytest.raises(RuntimeError):
            with host_math_context(None):
                assert active_host_math() is None
                raise RuntimeError("intentional")
        assert active_host_math() is portable
    assert active_host_math() is None
    identity = portable.identity()
    assert identity["policy"] == "portable_exp_trig"
    assert {"scalar_source_sha256", "compiler", "object_sha256", "library_sha256"} <= set(identity)
    assert reference_math("portable_exp_trig", tmp_path).identity() == identity
    library = next(tmp_path.glob("portable_exp_trig/*/host_math_reference.so"))
    replacement = library.with_suffix(".changed")
    replacement.write_bytes(library.read_bytes() + b"changed")
    replacement.replace(library)
    with pytest.raises(RuntimeError, match="artifact changed"):
        portable.identity()
    with pytest.raises(RuntimeError, match="artifact changed"):
        portable.evaluate("math.exp", np.asarray([0.0], dtype="float32"))
    # Changed cache products are rebuilt, not accepted under their stale receipt.
    restored = reference_math("portable_exp_trig", tmp_path)
    assert restored.identity() == identity


def test_non_nearest_rounding_refuses(portable):
    # Standard C fesetround constants vary by platform; ask C for the selected constant.
    libc = ctypes.CDLL(None)
    if not hasattr(libc, "fegetround") or not hasattr(libc, "fesetround"):
        pytest.skip("fenv functions unavailable")
    original = libc.fegetround()
    # x86 glibc FE_DOWNWARD. Avoid touching fenv on other platforms.
    import platform

    if platform.machine() not in ("x86_64", "amd64"):
        pytest.skip("test uses x86 FE_DOWNWARD")
    try:
        assert libc.fesetround(0x400) == 0
        with pytest.raises(RuntimeError, match="round-to-nearest"):
            portable.evaluate("math.exp", np.asarray([0.0], dtype="float32"))
    finally:
        assert libc.fesetround(original) == 0
