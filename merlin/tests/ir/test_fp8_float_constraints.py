"""An fp8 element type must also SATISFY the "is a float" constraints, not merely parse.

WHY THIS EXISTS. `register_fp8_types` taught xDSL's type parser the `f8E4M3FN` / `f8E5M2` bare
idents, which is only half the job. `arith.truncf` and `arith.extf` -- the ordinary quantize and
dequantize casts -- constrain operand and result to `floatingPointLike`, which is
`ContainerOf(AnyFloatConstr)`: an `AnyOf` over the six BUILTIN float classes. An fp8 type therefore
parsed and then failed VERIFICATION with `Unexpected attribute f8E4M3FN` -- a message that reads
like a parse error and is not one.

MEASURED: this made `arith.truncf %x : f32 to f8E4M3FN` unreadable, so the npu_model
`smolvla_requant` kernel could not be loaded at all -- on a target whose entire datapath is fp8.

The second half is reader PARITY. merlin has two context builders; only `xdsl_dialects._common`
registered fp8, so the same capsule loaded through one reader and failed `type expected` in the
other. Both now register it.
"""

from __future__ import annotations

import pytest

pytest.importorskip("xdsl")


def _ctx():
    from xdsl.context import Context
    from xdsl.dialects import arith, func, linalg, math, tensor
    from xdsl.dialects.builtin import Builtin

    import merlin.xdsl_dialects.fp8  # noqa: F401 - installs the hooks on import

    ctx = Context(allow_unregistered=True)
    for d in (Builtin, func.Func, arith.Arith, linalg.Linalg, tensor.Tensor, math.Math):
        ctx.load_dialect(d)
    return ctx


def _roundtrip(src: str) -> None:
    from xdsl.parser import Parser

    module = Parser(_ctx(), src).parse_module()
    module.verify()


def _cast(op: str, frm: str, to: str) -> str:
    return f"builtin.module {{ func.func @f(%x: {frm}) {{ %t = arith.{op} %x : {frm} to {to}\n func.return }} }}"


@pytest.mark.parametrize("fp8", ["f8E4M3FN", "f8E5M2"])
def test_quantizing_to_fp8_verifies(fp8):
    """THE REGRESSION: dropping the constraint patch fails here with 'Unexpected attribute'."""
    _roundtrip(_cast("truncf", "f32", fp8))


@pytest.mark.parametrize("fp8", ["f8E4M3FN", "f8E5M2"])
def test_dequantizing_from_fp8_verifies(fp8):
    _roundtrip(_cast("extf", fp8, "f32"))


def test_builtin_float_casts_are_unaffected():
    """The six builtin floats keep their own entries; this widens the set, it does not replace it."""
    _roundtrip(_cast("truncf", "f64", "f32"))
    _roundtrip(_cast("extf", "f16", "f32"))


@pytest.mark.parametrize("fp8", ["f8E4M3FN", "f8E5M2"])
def test_the_math_dialect_accepts_fp8_too(fp8):
    """`math` has its own `floatingPointLike`. Patching only `arith` leaves math ops refusing fp8."""
    _roundtrip(f"builtin.module {{ func.func @f(%x: {fp8}) {{ %t = math.absf %x : {fp8}\n func.return }} }}")


def test_the_float_constraint_still_refuses_a_non_float():
    """Widening must not become 'accept anything' -- an integer is still not floatingPointLike."""
    from xdsl.utils.exceptions import VerifyException

    with pytest.raises((VerifyException, Exception)):
        _roundtrip(_cast("truncf", "f32", "i8"))


def test_registration_is_idempotent():
    """It installs at import; a second call must not double-register or raise."""
    from merlin.xdsl_dialects.fp8 import register_fp8_float_constraints, register_fp8_types

    for _ in range(3):
        register_fp8_types()
        register_fp8_float_constraints()
    _roundtrip(_cast("truncf", "f32", "f8E4M3FN"))


def test_both_merlin_readers_register_fp8():
    """READER PARITY, IN A FRESH INTERPRETER.

    `frontends.linalg_mlir` and `xdsl_dialects._common` must EACH register fp8, or a capsule loads
    through one reader and fails `type expected` in the other. This must run in a subprocess: the
    hooks are process-wide, so any earlier test that imported `xdsl_dialects.fp8` would satisfy this
    one for free and the test would pass with `linalg_mlir` unpatched -- which it did, until this
    was isolated. The subprocess imports ONLY `linalg_mlir`.
    """
    import subprocess
    import sys

    program = (
        "from merlin.frontends.linalg_mlir import parse_mlir_text\n"
        "src = ('builtin.module attributes {prov.level = \"linalg-on-tensors\"} {\\n'\n"
        "       '  func.func @f(%x: tensor<4xf8E4M3FN>) -> tensor<4xf8E4M3FN> "
        "{ func.return %x : tensor<4xf8E4M3FN> }\\n}\\n')\n"
        "parse_mlir_text(src)\n"
    )
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, f"linalg_mlir did not register fp8 on its own:\n{done.stderr[-600:]}"


def test_an_fp8_requantize_kernel_loads():
    """The exact shape that could not be read: quantize then dequantize inside a linalg body."""
    src = """builtin.module attributes {prov.level = "linalg-on-tensors"} {
  func.func @requant(%x: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %out0 = tensor.empty() : tensor<32x32xf32>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%x : tensor<32x32xf32>) outs(%out0 : tensor<32x32xf32>) {
    ^bb0(%in: f32, %_: f32):
      %t = arith.truncf %in : f32 to f8E4M3FN
      %u = arith.extf %t : f8E4M3FN to f32
      linalg.yield %u : f32
    } -> tensor<32x32xf32>
    return %result : tensor<32x32xf32>
  }
}
"""
    from merlin.frontends.linalg_mlir import parse_mlir_text

    parse_mlir_text(src)
