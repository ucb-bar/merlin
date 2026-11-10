"""Actual signed-i8 quantizer lowerings, captured forms, composition and mutations.

The checked subsection has static f32 data, rank-zero f32 scales, rank-zero i32/i64
zero points and the full signed-byte range. Tests invoke the production lowering,
authoritative MLIR exporter, Lean checker and strict refinement receipt chain.
"""

from __future__ import annotations

from merlin.common.paths import repo_root

QUANTIZE = '"quant_ext.quantize_per_tensor"'


def source(shape: str = "2x3x", width: int = 64) -> str:
    data, result = f"tensor<{shape}f32>", f"tensor<{shape}i8>"
    return f"""func.func @forward(%x: {data}, %s: tensor<f32>, %z: tensor<i{width}>) -> {result} {{
  %q = {QUANTIZE}(%x, %s, %z)
    <{{quant_min = -128 : i64, quant_max = 127 : i64, output_dtype = "int8"}}>
    : ({data}, tensor<f32>, tensor<i{width}>) -> {result}
  return %q : {result}
}}
"""


def captured_source() -> tuple[str, int, int]:
    forms: dict[tuple[str, str], str] = {}
    total = 0
    root = repo_root() / "merlin/contract/capsules/model"
    for path in sorted(root.glob("*/capsule.interface.mlir")):
        for line in path.read_text().splitlines():
            if QUANTIZE + "(" not in line:
                continue
            total += 1
            head, _, signature = line.rpartition(" : ")
            properties = head[head.index("<{") : head.index("}>") + 2]
            forms.setdefault((signature.strip(), properties), line.strip())
    params, body, results, types = [], [], [], []
    for i, ((signature, _), line) in enumerate(forms.items()):
        operands, _, result = signature.partition(" -> ")
        data, scale, zero = (t.strip() for t in operands.strip("()").split(", "))
        params += [f"%x{i}: {data}", f"%s{i}: {scale}", f"%z{i}: {zero}"]
        call = line[line.index(QUANTIZE) :]
        body.append(f"%q{i} = {QUANTIZE}(%x{i}, %s{i}, %z{i}){call[call.index(')') + 1 :]}")
        results.append(f"%q{i}")
        types.append(result.strip())
    program = (
        f"func.func @forward({', '.join(params)}) -> ({', '.join(types)}) {{\n"
        + "\n".join(body)
        + f"\nreturn {', '.join(results)} : {', '.join(types)}\n}}\n"
    )
    return program, total, len(forms)


def _wrong_multiply(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    mul = next(op for op in module.walk() if isinstance(op, arith.MulfOp))
    Rewriter.replace_op(mul, arith.AddfOp(mul.lhs, mul.rhs))


def _ignore_zero_point(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    add = next(op for op in module.walk() if isinstance(op, arith.AddfOp))
    Rewriter.replace_op(add, arith.AddfOp(add.lhs, add.lhs))


def _wrong_rounding(module):
    from xdsl.dialects import math
    from xdsl.rewriter import Rewriter

    op = next(op for op in module.walk() if isinstance(op, math.RoundEvenOp))
    Rewriter.replace_op(op, math.FloorOp(op.operands[0]))


def _wrong_reciprocal(module):
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    div = next(op for op in module.walk() if isinstance(op, arith.DivfOp))
    Rewriter.replace_op(div, arith.DivfOp(div.rhs, div.lhs))
