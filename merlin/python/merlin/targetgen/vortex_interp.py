"""Interpret a Vortex capsule's interface MLIR on given inputs — an INDEPENDENT reference oracle.

The capsule's ``golden.yaml`` is the answer key for ONE fixed input draw (the public
:func:`vortex_oracle.operand_seed` fill). To grade a frozen package on a HELD-OUT draw (A.1 —
:mod:`capsule_resample`), the reference output for those fresh inputs has to be recomputed at grade
time. This module does that by executing the capsule's own interface IR — the authoritative contract —
through xDSL, so there is no second copy of the op semantics to drift from the emitters/goldens.

It is the library form of the cross-check in ``tests/targetgen/test_vortex_corpus_interpreted.py``
(which imports :func:`interpret_capsule` from here), lifted out so the test and the resample grader run
the SAME interpreter. xDSL is imported lazily: callers that never interpret (the common case) pay
nothing, and a host without xDSL gets a clear error only when it actually asks to interpret.

A handful of standard-dialect ops xDSL 0.68 does not implement are shimmed; each has exactly one
meaning (sign-extend, truncate, negate, min/max, select, divide, tanh, ``scf.while``). ``linalg.fill``
with a scalar operand is patched. ``linalg.conv_2d_nhwc_hwcf`` has no interpretation, so the one named
float-conv capsule is run through the equivalent explicit ``linalg.generic`` (:func:`generic_conv_form`).
"""
from __future__ import annotations

import math as _pymath
import pathlib
from typing import Any

import yaml

from . import vortex_oracle as VO

# `linalg.conv_2d_nhwc_hwcf` has no interpretation function in xDSL 0.68; the named-conv capsule is
# interpreted through generic_conv_form() instead. Any op here triggers that substitution.
NAMED_CONV_OPS = {"linalg.conv_2d_nhwc_hwcf"}

_STATE: dict[str, Any] = {}          # cached xDSL handles + shim classes, built once by _init()


class InterpUnavailable(RuntimeError):
    """xDSL is not importable / exposes a different interpreter API, so IR cannot be interpreted here."""


def available() -> bool:
    """True if the IR interpreter can run here (xDSL importable with the expected API). Never raises."""
    try:
        _init()
        return True
    except InterpUnavailable:
        return False


def _init() -> dict[str, Any]:
    """Import xDSL once and build the shim registry. Raises :class:`InterpUnavailable` if xDSL is absent
    or exposes a different interpreter API (the same conditions the interpreted test skips on)."""
    if _STATE:
        return _STATE
    try:
        import xdsl.interpreters.linalg as _xdsl_linalg
        from xdsl.dialects import arith, linalg, math, scf
        from xdsl.dialects.builtin import IntegerType, f32
        from xdsl.interpreter import (
            Interpreter,
            InterpreterFunctions,
            OpImplResult,
            impl,
            impl_terminator,
            register_impls,
        )
        from xdsl.interpreters import register_implementations
        from xdsl.interpreters.math import MathFunctions
        from xdsl.interpreters.shaped_array import ShapedArray
        from xdsl.interpreters.utils.ptr import TypedPtr
        from xdsl.parser import Parser
    except ImportError as exc:                                          # pragma: no cover
        raise InterpUnavailable(f"xdsl interpreter unavailable: {exc}") from exc

    @register_impls
    class ArithShims(InterpreterFunctions):
        """The `arith` ops xDSL 0.68 does not implement. Each has exactly one meaning."""

        @impl(arith.ExtSIOp)
        def run_extsi(self, interp, op, args):
            return (args[0],)                       # sign-extension preserves the integer value

        @impl(arith.TruncIOp)
        def run_trunci(self, interp, op, args):
            w = op.result.type.width.data
            m = args[0] & ((1 << w) - 1)
            return (m - (1 << w) if m >= (1 << (w - 1)) else m,)

        @impl(arith.NegfOp)
        def run_negf(self, interp, op, args):
            return (-args[0],)

        @impl(arith.MaxSIOp)
        def run_maxsi(self, interp, op, args):
            return (max(args[0], args[1]),)

        @impl(arith.MinSIOp)
        def run_minsi(self, interp, op, args):
            return (min(args[0], args[1]),)

        @impl(arith.SelectOp)
        def run_select(self, interp, op, args):
            return (args[1] if args[0] else args[2],)

        @impl(arith.DivfOp)
        def run_divf(self, interp, op, args):
            return (args[0] / args[1],)

    @register_impls
    class MathShims(InterpreterFunctions):
        """`math` ops xDSL 0.68 leaves out (it implements exp/log/sqrt). `tanh` has one meaning."""

        @impl(math.TanhOp)
        def run_tanh(self, interp, op, args):
            return (_pymath.tanh(args[0]),)

    @register_impls
    class ScfShims(InterpreterFunctions):
        """`scf.while`, absent from xDSL 0.68 (which has for/if/br only)."""

        @impl(scf.WhileOp)
        def run_while(self, interp, op, args):
            current = tuple(args)
            for _ in range(1_000_000):              # runaway guard, not a semantic bound
                interp.run_ssacfg_region(op.before_region, current, "while_before")
                vals = getattr(interp, "_merlin_condition", None)
                assert vals is not None, "scf.condition did not run"
                if not vals[0]:
                    return tuple(vals[1:])
                current = tuple(interp.run_ssacfg_region(op.after_region, tuple(vals[1:]), "while_after"))
            raise AssertionError("scf.while did not terminate")

        @impl_terminator(scf.ConditionOp)
        def run_condition(self, interp, op, args):
            interp._merlin_condition = tuple(args)
            return None, ()

    def _patch_linalg_fill() -> None:
        """xDSL's `run_fill` asserts a ShapedArray fill value; standard MLIR passes a scalar."""
        def fill_impl(ft, interp, op, values):
            operand, res = values[0], values[1]
            value = operand.data_ptr[0] if isinstance(operand, ShapedArray) else operand
            for i in range(len(res.data)):
                res.data_ptr[i] = value
            return OpImplResult((res,) if len(op.results) > 0 else (), None)
        getattr(_xdsl_linalg.LinalgFunctions, "__impl_dict")[linalg.ops.FillOp] = fill_impl

    _STATE.update(
        Interpreter=Interpreter, Parser=Parser, ShapedArray=ShapedArray, TypedPtr=TypedPtr,
        register_implementations=register_implementations, MathFunctions=MathFunctions,
        ArithShims=ArithShims, MathShims=MathShims, ScfShims=ScfShims,
        patch_fill=_patch_linalg_fill,
        XTYPE={"f32": f32, "i8": IntegerType(8), "i32": IntegerType(32)},
    )
    return _STATE


def uses_uninterpretable(mlir_text: str) -> bool:
    """True if the IR contains an op the interpreter cannot run directly (the named conv)."""
    return any(op in mlir_text for op in NAMED_CONV_OPS)


def generic_conv_form(capsule: dict) -> str:
    """The named float conv as the equivalent explicit `linalg.generic`, which the interpreter CAN run.

    `linalg.conv_2d_nhwc_hwcf` at unit stride/dilation computes
    `O[n,oh,ow,f] = sum I[n, oh+kh, ow+kw, c] * K[kh,kw,c,f]`. Interpreting that generic checks the same
    arithmetic the named op is defined to perform.
    """
    a = capsule["operation"]["attributes"]
    ifm = next(o for o in capsule["inputs"] if o["name"] == a["ifm"])
    out = next(o for o in capsule["inputs"] if o["role"] == "output")
    _, h, w, ci = ifm["shape"]
    kh, kw, co = a["kh"], a["kw"], a["co"]
    _, oh, ow, _ = out["shape"]
    ti = f"tensor<1x{h}x{w}x{ci}xf32>"
    tw = f"tensor<{kh}x{kw}x{ci}x{co}xf32>"
    to = f"tensor<1x{oh}x{ow}x{co}xf32>"
    return f'''
#cI = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh + kh, ow + kw, ic)>
#cW = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (kh, kw, ic, oc)>
#cO = affine_map<(n, oh, ow, oc, kh, kw, ic) -> (n, oh, ow, oc)>
module {{
  func.func @forward(%IFM: {ti}, %W: {tw}) -> {to} {{
    %z = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : {to}
    %init = linalg.fill ins(%z : f32) outs(%e : {to}) -> {to}
    %0 = linalg.generic {{indexing_maps = [#cI, #cW, #cO],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel",
                                           "reduction", "reduction", "reduction"]}}
         ins(%IFM, %W : {ti}, {tw}) outs(%init : {to}) {{
    ^bb0(%x: f32, %k: f32, %acc: f32):
      %p = arith.mulf %x, %k : f32
      %s = arith.addf %acc, %p : f32
      linalg.yield %s : f32
    }} -> {to}
    func.return %0 : {to}
  }}
}}
'''


def interpret_ir(mlir_text: str, inputs: dict[str, list], input_specs: list[dict]) -> list:
    """Execute `mlir_text`'s ``@forward`` on `inputs` (name -> flat values); -> flat output values.

    `input_specs` are the capsule's non-output operand dicts (name/role/shape/dtype), in call order.
    """
    from merlin.frontends.linalg_mlir import make_context
    S = _init()
    ctx = make_context()
    module = S["Parser"](ctx, mlir_text).parse_module()

    S["patch_fill"]()
    interp = S["Interpreter"](module)
    S["register_implementations"](interp, ctx)
    interp.register_implementations(S["MathFunctions"]())     # not in register_implementations
    interp.register_implementations(S["ArithShims"]())
    interp.register_implementations(S["MathShims"]())
    interp.register_implementations(S["ScfShims"]())

    args = []
    for spec in input_specs:
        conv = float if spec["dtype"] == "f32" else int
        args.append(S["ShapedArray"](
            S["TypedPtr"].new([conv(v) for v in inputs[spec["name"]]], xtype=S["XTYPE"][spec["dtype"]]),
            list(spec["shape"])))
    (res,) = interp.call_op("forward", args)
    return list(res.data)


def interpret_capsule(capsule_dir: str | pathlib.Path, *, salt: str | None = None,
                      mlir_text: str | None = None) -> tuple[list, dict]:
    """Interpret a capsule's interface IR on its reference inputs (public, or held-out if `salt` set).

    -> (flat output values, capsule dict). The named float conv is auto-substituted with its generic
    form (:func:`generic_conv_form`). Pass `mlir_text` to override the IR (used by the corpus test).
    """
    d = pathlib.Path(capsule_dir)
    cap = yaml.safe_load((d / "capsule.yaml").read_text())
    if mlir_text is None:
        text = (d / cap["interface_mlir"]).read_text()
        if uses_uninterpretable(text):
            text = generic_conv_form(cap)
    else:
        text = mlir_text
    inp = VO.reference_inputs(cap, salt=salt)
    specs = [s for s in cap["inputs"] if s["role"] != "output"]
    return interpret_ir(text, inp, specs), cap


def resampled_reference(capsule_dir: str | pathlib.Path, salt: str) -> dict[str, list]:
    """The reference output for a HELD-OUT input draw, shaped like ``golden.yaml``'s ``outputs``.

    Computed by interpreting the capsule's own interface IR (the authoritative contract) on the inputs
    the same `salt` produces — so a device run launched from that salt can be graded against it. Returns
    ``{<output operand name>: [flat values]}``.
    """
    values, cap = interpret_capsule(capsule_dir, salt=salt)
    out_name = next(o["name"] for o in cap["inputs"] if o["role"] == "output")
    return {out_name: values}
