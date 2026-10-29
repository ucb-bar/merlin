"""Shared inputs for the f32-accumulation contract tests: sources, the actual pass, mutations of its
output, and independent NumPy references for every contract clause.

References are written from the contract (``Bf16F32AccContract``), not from the pass: f32 products
of extended operands, a sequential f32 accumulator from +0.0 in loop order, and one final
round-to-nearest-even conversion. ``fused=True`` evaluates each step as one correctly rounded
multiply-add, the behavior a ``contract`` license permits the trusted lowering to choose.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

BMM_MAPS = (
    "[affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, "
    "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]"
)
ITERS = "[" + ", ".join(["#linalg.iterator_type<parallel>"] * 3 + ["#linalg.iterator_type<reduction>"]) + "]"
SEGMENTS = "operandSegmentSizes = array<i32: 2, 1>"
STANDARD_BODY = (
    '%m = "arith.mulf"(%x, %y) <{{fastmath = #arith.fastmath<none>}}> : ({k}, {k}) -> {k}\n'
    '      %p = "arith.addf"(%c, %m) <{{fastmath = #arith.fastmath<none>}}> : ({k}, {k}) -> {k}'
)


def bmm(
    k: str = "bf16",
    *,
    batch: int = 2,
    m: int = 3,
    n: int = 4,
    depth: int = 5,
    init: str = "0.000000e+00",
    body: str = STANDARD_BODY,
) -> str:
    """A batched contraction in the form SmolVLA's attention captures it: splat destination, generic."""
    a, b, c = f"tensor<{batch}x{m}x{depth}x{k}>", f"tensor<{batch}x{depth}x{n}x{k}>", f"tensor<{batch}x{m}x{n}x{k}>"
    return f""""builtin.module"() ({{
  "func.func"() <{{function_type = ({a}, {b}) -> {c}, sym_name = "subject"}}> ({{
  ^bb0(%a: {a}, %b: {b}):
    %z = "arith.constant"() <{{value = {init} : {k}}}> {{prov.region_id = "matmul_1"}} : () -> {k}
    %s = "tensor.splat"(%z) {{prov.region_id = "matmul_1"}} : ({k}) -> {c}
    %r = "linalg.generic"(%a, %b, %s) <{{indexing_maps = {BMM_MAPS}, iterator_types = {ITERS}, {SEGMENTS}}}> ({{
    ^bb0(%x: {k}, %y: {k}, %c: {k}):
      {body.format(k=k)}
      "linalg.yield"(%p) : ({k}) -> ()
    }}) {{prov.region_id = "matmul_1", prov.aten = "aten.bmm.default"}} : ({a}, {b}, {c}) -> {c}
    "func.return"(%r) : ({c}) -> ()
  }}) : () -> ()
}}) : () -> ()
"""


def matmul(
    k: str = "bf16", *, m: int = 3, n: int = 4, depth: int = 5, transposed: bool = False, operand: str | None = None
) -> str:
    """A named ``linalg.matmul`` with a fill destination; ``transposed`` reads B as (N, K)."""
    operand = operand or k
    right = f"tensor<{n}x{depth}x{operand}>" if transposed else f"tensor<{depth}x{n}x{operand}>"
    a, c = f"tensor<{m}x{depth}x{operand}>", f"tensor<{m}x{n}x{k}>"
    b_map = "(d1, d2)" if transposed else "(d2, d1)"
    maps = f"[affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> {b_map}>, " + (
        "affine_map<(d0, d1, d2) -> (d0, d1)>]"
    )
    cast = ""
    if operand != k:
        cast = (
            f'%u = "arith.truncf"(%x) : ({operand}) -> {k}\n      %v = "arith.truncf"(%y) : ({operand}) -> {k}\n'
            f'      %m = "arith.mulf"(%u, %v) <{{fastmath = #arith.fastmath<none>}}> : ({k}, {k}) -> {k}'
        )
    else:
        cast = f'%m = "arith.mulf"(%x, %y) <{{fastmath = #arith.fastmath<none>}}> : ({k}, {k}) -> {k}'
    return f""""builtin.module"() ({{
  "func.func"() <{{function_type = ({a}, {right}) -> {c}, sym_name = "subject"}}> ({{
  ^bb0(%a: {a}, %b: {right}):
    %z = "arith.constant"() <{{value = 0.000000e+00 : {k}}}> : () -> {k}
    %e = "tensor.empty"() : () -> {c}
    %f = "linalg.fill"(%z, %e) <{{operandSegmentSizes = array<i32: 1, 1>}}> ({{
    ^bb0(%i: {k}, %o: {k}):
      "linalg.yield"(%i) : ({k}) -> ()
    }}) : ({k}, {c}) -> {c}
    %r = "linalg.matmul"(%a, %b, %f) <{{indexing_maps = {maps}, {SEGMENTS}}}> ({{
    ^bb0(%x: {operand}, %y: {operand}, %c: {k}):
      {cast}
      %p = "arith.addf"(%c, %m) <{{fastmath = #arith.fastmath<none>}}> : ({k}, {k}) -> {k}
      "linalg.yield"(%p) : ({k}) -> ()
    }}) : ({a}, {right}, {c}) -> {c}
    "func.return"(%r) : ({c}) -> ()
  }}) : () -> ()
}}) : () -> ()
"""


def run_pass(source: str) -> tuple[str, int]:
    """The production pass, unchanged, on its own parser and printer."""
    from merlin.llvmlower import passes_xdsl
    from merlin.xdsl_dialects._common import text

    module = passes_xdsl.parse_mlir_text(source)
    count = passes_xdsl.lower_bf16_matmul_f32acc(module)
    return text(module, generic=True), count


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, f"mutation anchor {old!r} occurs {text.count(old)} times"
    return text.replace(old, new)


def line_with(text: str, *needles: str) -> str:
    matches = [line for line in text.splitlines() if all(needle in line for needle in needles)]
    assert len(matches) == 1, f"{needles} matches {len(matches)} lines"
    return matches[0]


def operands(line: str) -> list[str]:
    """SSA operands of one generic-form op line, by structure rather than spelling."""
    inner = line.split('"(', 1)[1].split(")", 1)[0]
    return [part.strip() for part in inner.split(",") if part.strip()]


def result(line: str) -> str:
    return line.split("=", 1)[0].strip()


def reverse_reduction(text: str, *, batch: int = 2, m: int = 3, n: int = 4, depth: int = 5, fmt: str = "bf16") -> str:
    """Feed the accumulating generic K-reversed copies of both operands: identical products,
    accumulated in the opposite order. Valid MLIR, unlike a non-invertible reversed map."""
    acc = line_with(text, '"linalg.generic"', "iterator_type<reduction>")
    left, right, destination = operands(acc)
    indent = acc[: len(acc) - len(acc.lstrip())]
    parallel = ", ".join(["#linalg.iterator_type<parallel>"] * 3)

    def copy(name: str, value: str, shape: str, read: str) -> str:
        ty = f"tensor<{shape}x{fmt}>"
        return (
            f'{indent}{name}_e = "tensor.empty"() : () -> {ty}\n'
            f'{indent}{name} = "linalg.generic"({value}, {name}_e) <{{indexing_maps = '
            f"[affine_map<(d0, d1, d2) -> {read}>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], "
            f"iterator_types = [{parallel}], operandSegmentSizes = array<i32: 1, 1>}}> ({{\n"
            f"{indent}^bb0({name}_x: {fmt}, {name}_o: {fmt}):\n"
            f'{indent}  "linalg.yield"({name}_x) : ({fmt}) -> ()\n'
            f"{indent}}}) : ({ty}, {ty}) -> {ty}\n"
        )

    reversal = copy("%rev_left", left, f"{batch}x{m}x{depth}", f"(d0, d1, {depth - 1} - d2)") + copy(
        "%rev_right", right, f"{batch}x{depth}x{n}", f"(d0, {depth - 1} - d1, d2)"
    )
    rewired = acc.replace(f"({left}, {right}, {destination})", f"(%rev_left, %rev_right, {destination})")
    return replace_once(text, acc + "\n", reversal + rewired + "\n")


# --- 16-bit formats -----------------------------------------------------------------------------


def to_f32(bits: np.ndarray, fmt: str) -> np.ndarray:
    """Exact extension of stored 16-bit patterns to binary32 (clause 1)."""
    bits = np.asarray(bits, dtype=np.uint16)
    if fmt == "bf16":
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return bits.view(np.float16).astype(np.float32)


def from_f32(values: np.ndarray, fmt: str) -> np.ndarray:
    """Round-to-nearest-even conversion from binary32 (clause 5); inputs here are never NaN."""
    values = np.asarray(values, dtype=np.float32)
    if fmt == "bf16":
        bits = values.view(np.uint32).astype(np.uint64)
        return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return values.astype(np.float16).view(np.uint16)


def random_bits(rng: np.random.Generator, shape: tuple[int, ...], fmt: str, scale: float = 4.0) -> np.ndarray:
    return from_f32(rng.uniform(-scale, scale, shape).astype(np.float32), fmt)


def round_f32(value: Fraction) -> np.float32:
    """Correctly rounded (nearest, ties to even) binary32 of an exact rational, with overflow."""
    if value == 0:
        return np.float32(0.0)
    sign, magnitude = (-1 if value < 0 else 1), abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while Fraction(2) ** exponent > magnitude:
        exponent -= 1
    while Fraction(2) ** (exponent + 1) <= magnitude:
        exponent += 1
    quantum = Fraction(2) ** (max(exponent, -126) - 23)
    steps, remainder = divmod(magnitude / quantum, 1)
    steps = int(steps)
    if remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and steps % 2):
        steps += 1
    rounded = steps * quantum
    if rounded >= Fraction(2) ** 128:
        return np.float32(sign * np.inf)
    return np.float32(sign * float(rounded))


def step(acc: np.float32, x: np.float32, y: np.float32, fused: bool) -> np.float32:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        product = np.float32(x * y)
        exact = Fraction(float(x)) * Fraction(float(y)) if np.isfinite(x) and np.isfinite(y) else None
        if (
            not fused
            or exact is None
            or not np.isfinite(acc)
            or (np.isfinite(product) and Fraction(float(product)) == exact)
        ):
            return np.float32(acc + product)
        return round_f32(Fraction(float(acc)) + exact)


def contract_reference(
    left: np.ndarray, right: np.ndarray, fmt: str, *, order=None, init: float = 0.0, fused: bool = False
) -> np.ndarray:
    """The contract for a (B, M, K) x (B, K, N) contraction, scalar step by scalar step."""
    a, b = to_f32(left, fmt), to_f32(right, fmt)
    batch, rows, depth = a.shape
    cols = b.shape[2]
    order = list(range(depth)) if order is None else list(order)
    out = np.zeros((batch, rows, cols), dtype=np.float32)
    for p in range(batch):
        for i in range(rows):
            for j in range(cols):
                acc = np.float32(init)
                for t in order:
                    acc = step(acc, a[p, i, t], b[p, t, j], fused)
                out[p, i, j] = acc
    return from_f32(out, fmt)


def contract_reference_fast(left: np.ndarray, right: np.ndarray, fmt: str) -> np.ndarray:
    """Vectorized unfused contract: one f32 rounding per product and per add, in ascending K."""
    a, b = to_f32(left, fmt), to_f32(right, fmt)
    acc = np.zeros((a.shape[0], a.shape[1], b.shape[2]), dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(a.shape[2]):
            acc = (acc + (a[:, :, t, None] * b[:, None, t, :]).astype(np.float32)).astype(np.float32)
    return from_f32(acc, fmt)


def half_accumulation_reference(left: np.ndarray, right: np.ndarray, fmt: str) -> np.ndarray:
    """The source's own meaning: product and running sum each rounded to the 16-bit format."""
    a, b = to_f32(left, fmt), to_f32(right, fmt)
    batch, rows, depth = a.shape
    out = np.zeros((batch, rows, b.shape[2]), dtype=np.uint16)
    with np.errstate(over="ignore", invalid="ignore"):
        for p in range(batch):
            for i in range(rows):
                for j in range(b.shape[2]):
                    acc = np.float32(0.0)
                    for t in range(depth):
                        product = to_f32(from_f32(np.float32(a[p, i, t] * b[p, t, j]), fmt), fmt)
                        acc = to_f32(from_f32(np.float32(acc + product), fmt), fmt)
                    out[p, i, j] = from_f32(acc, fmt)
    return out
