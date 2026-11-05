"""A GELU + requantization kernel in miniature: the W8A8 per-element trip under libm, and its fix.

Spectformer int8 missed the W8A8 gate's per-element bound on RISC-V (cos 0.9999935, rel 0.0064,
max_rel 0.16) while a host build of the same model matched the golden bit for bit. The cause: GELU's
``erff`` came from glibc in the golden and from the RISC-V toolchain's libm in the image, and the two
differ by one ulp on a few inputs; requantization turned that ulp into a whole quantization step.

This reproduces the mechanism on one kernel, on the same two builds the gate compares -- the host
lowering that writes a golden and the RISC-V lowering run on spike -- and grades it with the gate
itself (``zephyr_model._gate``):

1. under ``libm``, GELU is computed on both and the elements whose bits differ are found;
2. each such element gets a quantization step that puts the two libraries' GELU values on opposite
   sides of a rounding boundary, so the one-ulp difference becomes one quantization step, exactly as
   requantization does in the model. The gate then trips on the per-element bound alone: cosine and
   global rel stay inside their bounds, as they did for spectformer;
3. under ``portable``, the same kernel on the same inputs is bit-identical on both builds and the
   gate passes -- without any bound being loosened.

If the two C libraries ever agree on every input here, step 2 has nothing to reproduce and is
skipped; step 3 is still asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.llvmlower import optional_passes as OP
from merlin.llvmlower import portable_math as PM
from merlin.runtime import kernel_spike

pytestmark = pytest.mark.skipif(not kernel_spike.available(), reason="chipyard toolchain/spike not available")

N = 4096

# y = GELU(x) = 0.5 x (1 + erf(x / sqrt 2)); d = round(y / s) * s, the dequantized requantization.
_MLIR = f"""
#id = affine_map<(d0) -> (d0)>
module {{
  func.func @gelu_q(%x: tensor<{N}xf32>, %s: tensor<{N}xf32>) -> (tensor<{N}xf32>, tensor<{N}xf32>) {{
    %e0 = tensor.empty() : tensor<{N}xf32>
    %e1 = tensor.empty() : tensor<{N}xf32>
    %r:2 = linalg.generic {{indexing_maps = [#id, #id, #id, #id], iterator_types = ["parallel"]}}
        ins(%x, %s : tensor<{N}xf32>, tensor<{N}xf32>) outs(%e0, %e1 : tensor<{N}xf32>, tensor<{N}xf32>) {{
    ^bb0(%a: f32, %sc: f32, %o0: f32, %o1: f32):
      %c = arith.constant 0.707106769 : f32
      %h = arith.constant 0.5 : f32
      %one = arith.constant 1.0 : f32
      %z = arith.mulf %a, %c : f32
      %er = math.erf %z : f32
      %t = arith.addf %er, %one : f32
      %u = arith.mulf %a, %h : f32
      %y = arith.mulf %u, %t : f32
      %q = arith.divf %y, %sc : f32
      %qr = math.roundeven %q : f32
      %d = arith.mulf %qr, %sc : f32
      linalg.yield %y, %d : f32, f32
    }} -> (tensor<{N}xf32>, tensor<{N}xf32>)
    return %r#0, %r#1 : tensor<{N}xf32>, tensor<{N}xf32>
  }}
}}
"""


def _module():
    from merlin.frontends.linalg_mlir import parse_mlir_text

    return parse_mlir_text(_MLIR)


def _host(x, s, work):
    """The golden's build: the same module lowered for the host and run in-process."""
    from merlin.llvmlower.kernel_backend import compile_host, extract_kernel

    model = compile_host(extract_kernel(_module(), "gelu_q"), work)
    y, d = np.zeros(N, np.float32), np.zeros(N, np.float32)
    model([(a.ctypes.data, a.shape) for a in (x, s, y, d)])
    return y, d


def _device(x, s, work):
    run = kernel_spike.run(_module(), [x, s], work)
    return run.outputs[0], run.outputs[1]


def _same_bits(a, b) -> np.ndarray:
    return a.view(np.uint32) == b.view(np.uint32)


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.delenv(OP.ENV, raising=False)

    def select(name: str) -> None:
        monkeypatch.setenv(PM.SWITCH, "1" if name == PM.PORTABLE else "0")
        assert PM.policy() == name

    return select


def test_gelu_requantization_trips_the_gate_under_libm_and_passes_under_portable(policy, tmp_path):
    from merlin.runtime.backends.zephyr_model import _gate

    # GELU's range, plus one large activation so the output's scale is that of a real layer.
    x = np.append(np.linspace(-6, 6, N - 1), 60.0).astype(np.float32)
    ones = np.ones(N, np.float32)

    policy(PM.LIBM)
    y_host, _ = _host(x, ones, tmp_path / "libm_host_gelu")
    y_dev, _ = _device(x, ones, tmp_path / "libm_dev_gelu")
    differ = np.nonzero(~_same_bits(y_host, y_dev))[0]

    # One quantization step per element; at a differing element, a step that makes the two libraries'
    # GELU values round to different codes: 2 * (the smaller magnitude) puts that one exactly on a tie
    # (rounded to even, 0) and the larger one above it (1). Chosen where the golden's value is the
    # larger, is significant for the per-element term, and is small enough that one step stays inside
    # the global rel bound -- the shape of the spectformer failure.
    s = np.full(N, 1.0 / 16, np.float32)
    rms = float(np.sqrt(np.mean(y_host.astype(np.float64) ** 2)))
    larger_on_host = np.abs(y_host[differ]) > np.abs(y_dev[differ])
    step = 2 * np.abs(y_host[differ])
    crafted = differ[larger_on_host & (step > 0.02 * rms) & (step < 0.4)]
    smaller = np.where(np.abs(y_host) < np.abs(y_dev), y_host, y_dev)
    s[crafted] = (2 * smaller[crafted]).astype(np.float32)

    if len(crafted):
        _, golden = _host(x, s, tmp_path / "libm_host")
        _, device = _device(x, s, tmp_path / "libm_dev")
        verdict = _gate(device, {"w8a8": golden})
        assert not _same_bits(golden, device).all()
        assert verdict["w8a8_cos"] > 0.999 and verdict["w8a8_rel"] < 1e-2, verdict
        assert verdict["w8a8_max_rel"] > 0.05, verdict
        assert not verdict["ok"], "the per-element bound should reject a whole-step flip"

    policy(PM.PORTABLE)
    y_host, _ = _host(x, ones, tmp_path / "portable_host_gelu")
    y_dev, _ = _device(x, ones, tmp_path / "portable_dev_gelu")
    assert _same_bits(y_host, y_dev).all(), "GELU differs between the golden's build and the device"
    _, golden = _host(x, s, tmp_path / "portable_host")
    _, device = _device(x, s, tmp_path / "portable_dev")
    assert _same_bits(golden, device).all()
    verdict = _gate(device, {"w8a8": golden})
    assert verdict["ok"] and verdict["w8a8_max_rel"] == 0.0 and verdict["w8a8_rel"] == 0.0, verdict
    if not len(crafted):
        pytest.skip("the host and device C libraries agree on every GELU input here; nothing to reproduce")
