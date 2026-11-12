from __future__ import annotations

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, cf, func, math
from xdsl.interpreter import Interpreter, InterpreterFunctions, impl, register_impls
from xdsl.interpreters.arith import ArithFunctions
from xdsl.interpreters.linalg import LinalgFunctions

from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.model.automatic_host import optimize_host_quantization, prepare_automatic_host
from merlin.compiler.lower.model.quant_round_check import verify_quantization_rewrite
from merlin.compiler.transforms.certificates import ir_text


def _source(width=8, dtype="f32", *, divisor="1.0"):
    return f'builtin.module {{\n      func.func @forward(%a: tensor<3x5x{dtype}>) -> tensor<3x5xi{width}> {{\n        %e = tensor.empty() : tensor<3x5xi{width}>\n        %r = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,\n          affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel","parallel"]}}\n          ins(%a:tensor<3x5x{dtype}>) outs(%e:tensor<3x5xi{width}>) {{\n        ^bb0(%x:{dtype},%o:i{width}):\n          %s = arith.constant {divisor} : {dtype}\n          %hi = arith.constant 127.0 : {dtype}\n          %lo = arith.constant -127.0 : {dtype}\n          %d = arith.divf %x,%s : {dtype}\n          %round = math.roundeven %d : {dtype}\n          %h = arith.minimumf %round,%hi : {dtype}\n          %l = arith.maximumf %h,%lo : {dtype}\n          %q = arith.fptosi %l : {dtype} to i{width}\n          linalg.yield %q : i{width}\n        }} -> tensor<3x5xi{width}>\n        func.return %r : tensor<3x5xi{width}>\n      }}\n    }}'


@register_impls
class _ScalarConversions(InterpreterFunctions):
    @impl(arith.SelectOp)
    def select(self, interpreter, op, args):
        return (args[1] if args[0] else args[2],)

    @impl(arith.DivfOp)
    def divide(self, interpreter, op, args):
        return (float(np.float32(args[0]) / np.float32(args[1])),)

    @impl(arith.FPToSIOp)
    def truncate(self, interpreter, op, args):
        return (int(args[0]),)

    @impl(arith.SIToFPOp)
    def widen(self, interpreter, op, args):
        return (float(np.float32(args[0])),)

    @impl(arith.NegfOp)
    def negate(self, interpreter, op, args):
        return (-args[0],)

    @impl(math.RoundEvenOp)
    def round_even(self, interpreter, op, args):
        return (float(np.rint(np.float32(args[0]))),)


def _scalar_outputs(module, values):
    body = next((op for op in module.walk() if op.name == "linalg.generic")).regions[0]
    interpreter = Interpreter(module)
    for functions in (ArithFunctions(), LinalgFunctions(), _ScalarConversions()):
        interpreter.register_implementations(functions, override=isinstance(functions, _ScalarConversions))
    return np.array([interpreter.run_ssacfg_region(body, (float(value), 0))[0] for value in values])


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "ArithFunctions",
    "Interpreter",
    "InterpreterFunctions",
    "LinalgFunctions",
    "_ScalarConversions",
    "_scalar_outputs",
    "_source",
    "arith",
    "builtin",
    "cf",
    "func",
    "impl",
    "ir_text",
    "math",
    "np",
    "optimize_host_quantization",
    "parse_mlir_text",
    "prepare_automatic_host",
    "pytest",
    "register_impls",
    "verify_quantization_rewrite",
]
