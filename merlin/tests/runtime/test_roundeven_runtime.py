"""Actual runtime roundeven values must not depend on the caller's FP mode."""

import ctypes
import subprocess
import uuid

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.toolchain import clang


def test_runtime_values_and_nan_words_in_every_rounding_mode(tmp_path):
    compiler = clang()
    if not compiler.is_file():
        pytest.skip("selected native compiler unavailable")
    wrapper = tmp_path / "rounding.c"
    wrapper.write_text('''#include <fenv.h>
#include <math.h>
#include <stddef.h>
extern float roundevenf(float);
extern double roundeven(double);
static fenv_t incoming;
int save(void) { return fegetenv(&incoming); }
int restore(void) { return fesetenv(&incoming); }
int set_mode(int i) {
  const int modes[] = {FE_TONEAREST, FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
  return fesetround(modes[i]);
}
int current_mode(void) { return fegetround(); }
void run32(const float *x, float *y, size_t n, int mutant) {
  for (size_t i=0; i<n; ++i) y[i] = mutant ? rintf(x[i]) : roundevenf(x[i]);
}
void run64(const double *x, double *y, size_t n, int mutant) {
  for (size_t i=0; i<n; ++i) y[i] = mutant ? rint(x[i]) : roundeven(x[i]);
}
''')
    image = tmp_path / ("rounding_" + uuid.uuid4().hex + ".so")
    subprocess.run([str(compiler), "-O2", "-shared", "-fPIC", "-fno-builtin", "-Wl,-Bsymbolic-functions",
                    str(runtime_dir() / "abi/mlir_runtime.c"), str(wrapper), "-lm", "-o", str(image)],
                   check=True, capture_output=True)
    library = ctypes.CDLL(str(image))
    rng = np.random.default_rng(713)
    assert library.save() == 0
    try:
        assert library.set_mode(0) == 0
        cases = []
        for width, dtype, unsigned in [(32, np.float32, np.uint32), (64, np.float64, np.uint64)]:
            values = np.concatenate([
                np.array([-np.inf, -2.5, -1.5, -0.5, -0., 0., .5, 1.5, 2.5, np.inf], dtype=dtype),
                np.arange(-1024.5, 1025, .5, dtype=dtype),
                rng.integers(0, np.iinfo(unsigned).max, 131072, dtype=unsigned).view(dtype),
            ])
            with np.errstate(invalid="ignore"):
                expected = np.rint(values)
            call = getattr(library, f"run{width}")
            call.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            cases.append((values, expected, unsigned, call))
        for mode in range(4):
            assert library.set_mode(mode) == 0
            incoming = library.current_mode()
            for values, expected, unsigned, call in cases:
                actual = np.empty_like(values)
                call(values.ctypes.data, actual.ctypes.data, values.size, 0)
                assert library.current_mode() == incoming
                np.testing.assert_array_equal(actual.view(unsigned), expected.view(unsigned))
                if mode:
                    call(values.ctypes.data, actual.ctypes.data, values.size, 1)
                    assert not np.array_equal(actual.view(unsigned), expected.view(unsigned))
    finally:
        assert library.restore() == 0
