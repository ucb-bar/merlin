"""The integer i-GELU's error is its own QUANTIZATION STEP, not a defect.

Why this test exists. Grading the int8 datapath against an independent W8A8 reference
(torchao, which quantizes ``nn.Linear`` and leaves GELU exact in fp32) attributed most of one
model's whole-model residual to a single pass: replacing ``math.erf`` with the I-BERT integer
i-GELU moved SpectFormer's deviation from 0.035 to 0.160 of the output max, more than every
other quant pass combined. That is either a bug in the pass or the price of integerizing a
transcendental, and a whole-model number cannot tell the two apart.

It is the price. i-GELU quantizes the erf ARGUMENT to int8 with a per-row abs-max scale, so its
error is bounded by that scale's step -- and the test states exactly that, by measuring the real
emitted kernel against exact GELU and comparing the error to ``sx/2`` (half an int8 LSB of the
row). A genuine defect (a mis-sequenced polynomial, a wrong constant, the pass replacing the bare
erf rather than the whole GELU region) does NOT respect that budget; it produces error that does
not scale with the row's own quantization step, which the second assertion checks directly.

Compiles and runs on the host through the ordinary lowering path, with ``RTLD_LOCAL`` (no
``n_args``), so it coexists with other model loads in the same pytest process.
"""
from __future__ import annotations

import io
import math

import numpy as np
import pytest

from merlin.llvmlower import toolchain

_M, _N = 16, 256

_SRC = f"""builtin.module {{
  func.func @forward(%x: tensor<1x{_M}x{_N}xf32>) -> tensor<1x{_M}x{_N}xf32> {{
    %e = tensor.empty() : tensor<1x{_M}x{_N}xf32>
    %g = linalg.generic {{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                          affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                         iterator_types = ["parallel", "parallel", "parallel"]}}
        ins(%x : tensor<1x{_M}x{_N}xf32>) outs(%e : tensor<1x{_M}x{_N}xf32>) {{
    ^bb0(%a: f32, %o: f32):
      %h = arith.constant 5.000000e-01 : f32
      %one = arith.constant 1.000000e+00 : f32
      %isq = arith.constant 0.707106769 : f32
      %t = arith.mulf %a, %isq : f32
      %er = math.erf %t : f32
      %p = arith.addf %one, %er : f32
      %hx = arith.mulf %h, %a : f32
      %r = arith.mulf %hx, %p : f32
      linalg.yield %r : f32
    }} -> tensor<1x{_M}x{_N}xf32>
    func.return %g : tensor<1x{_M}x{_N}xf32>
  }}
}}"""

#: i-GELU's error budget, in half-int8-LSBs of the row scale it chooses. Measured max ratio over
#: the three activation regimes below is 1.44; 3.0 leaves headroom for host FP variation while
#: still rejecting an error that does not come from the quantization step.
_BUDGET_HALF_LSB = 3.0


def _exact_gelu(x: np.ndarray) -> np.ndarray:
    v = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + v(x / math.sqrt(2.0)))


def _run_igelu(tmp_path):
    from xdsl.printer import Printer

    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.lower import lower_model
    from merlin.llvmlower.passes_quant_int import lower_gelu_int

    mod = parse_mlir_text(_SRC)
    assert lower_gelu_int(mod) == 1                 # the whole GELU region, not the bare erf
    s = io.StringIO(); Printer(stream=s).print_op(mod)
    res = lower_model(s.getvalue(), tmp_path / "igelu", targets=("host",))
    model = HostModel.load(str(res.host_so))        # RTLD_LOCAL: no n_args

    def call(X: np.ndarray) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        Y = np.zeros_like(X)
        model([(X.ctypes.data, X.shape), (Y.ctypes.data, Y.shape)])
        return Y
    return call


@pytest.mark.skipif(not toolchain.available(), reason="m2m venv / clang-23 missing")
def test_igelu_error_is_within_its_own_quantization_step(tmp_path):
    call = _run_igelu(tmp_path)
    rng = np.random.default_rng(0)
    base = rng.standard_normal((1, _M, _N)).astype(np.float32)

    # three activation regimes, including the outlier-heavy one a per-row abs-max scale is most
    # sensitive to (a few large values per row inflate the step for the whole row).
    outl = base.copy()
    for r in range(_M):
        outl[0, r, rng.integers(0, _N, 4)] *= 20.0
    ratios = []
    for X in (base, (3.0 * base).astype(np.float32), outl):
        Y = call(X)
        G = _exact_gelu(X.astype(np.float64))
        err = float(np.abs(Y.astype(np.float64) - G).max())
        half_lsb = float((np.abs(X).max(axis=-1) / 127.0).mean() / 2.0)
        ratios.append(err / half_lsb)
        a, b = Y.ravel().astype(np.float64), G.ravel()
        assert float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) > 0.999
        assert err < _BUDGET_HALF_LSB * half_lsb, (err, half_lsb)

    # The error TRACKS the step rather than being a fixed offset: scaling the inputs 3x scales the
    # row step 3x and the error with it, so every regime lands in the same narrow band of
    # half-LSBs. A wrong constant or a mis-anchored rewrite would not.
    assert max(ratios) / min(ratios) < 4.0, ratios
