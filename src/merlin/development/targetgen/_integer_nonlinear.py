#!/usr/bin/env python3
"""Integer-only softmax, GELU and layer norm -- runs INSIDE the capture venv.

A program whose contractions are all int8 (``_activation_contractions``) still computes its
nonlinear layers in floating point: every softmax evaluates ``exp`` and a division per element,
every GELU an ``erf``, every layer norm a reciprocal square root. On a core without a floating
point vector unit those are the bulk of the host's work, and they sit between two int8
contractions, so the program leaves the integer domain twice per layer only to come straight
back. This module replaces the three with the I-BERT integer algorithms (Kim et al., 2021),
evaluated per element in integer arithmetic only:

* **softmax** -- the row is put on a FIXED exponent grid ``S = ln2 / 2**K`` (the same grid
  ``merlin.compiler.lower.model.passes_quant_int.lower_softmax_int`` uses, and for the same reason: the
  attention mask's sentinels must not set the grid), ``exp`` is the integer 2nd-order polynomial
  with a power-of-two range reduction, and the probabilities are returned as INT8 VALUES relative
  to the row's largest term, ``p = round(127 * e / e(0))``, normalized by their own integer sum.
  So the int8 operand of the next contraction (``P @ V``) is exactly ``p`` -- its dynamic row
  quantization is the identity on these values -- and the normalization is one per-row scale;
* **GELU** -- the input is quantized per row to int16, ``|x|`` rescaled onto a fixed grid by one
  dyadic multiply, and ``gelu(x) = relu(x) - |x| h(|x|)`` evaluated with ``h`` a degree-6 fixed-point
  polynomial (Horner's rule), dequantized by one per-row scale. Not I-BERT's 2nd-order i-GELU: its
  own 1.8e-2 error was measured to move a whole model past the float floor (see ``GELU_COEFFS``);
* **layer norm** -- the input is quantized per row to int16, the mean, the variance and the
  integer square root (Newton's method, a fixed number of iterations) are integer, and the
  normalization is one integer multiply and shift per element by a per-row factor. The affine
  weight and bias stay per-channel constants applied at the output.

What stays floating point, and said: the per-ROW scalars (a scale, its reciprocal), and the one
dequantizing multiply per element each function ends with -- the value the next int8 contraction
re-quantizes. No ``exp``, ``erf``, ``rsqrt`` or per-element division is left.

This is a DIFFERENT NUMERICAL MODEL from the floating-point one. A capsule captured under it is
graded bit-for-bit against its own reference (the same algorithm), and its accuracy against the
floating-point model is a separate, measured statement -- never assumed.

Nothing here names a target, a model or a scheme. The integer exp's constants are restated from
``passes_quant_int`` (this module runs where merlin is not importable) and a test holds the two equal,
so the two integer exps cannot drift apart; the GELU polynomial is re-derived by a test.
"""

from __future__ import annotations

import math
from typing import Any

# ---- the integer exp, on the fixed exponent grid (restated from passes_quant_int) -------------
IEXP_A, IEXP_B, IEXP_C = 0.35815147, 1.35330989, 0.34401959
IEXP_SH = 30  # fixed-point fraction bits for the polynomial
IEXP_K = 8  # the exponent grid step is ln2 / 2**K
IEXP_CLAMP = -30.0  # exp(-30) ~ 1e-13: where a masked/saturated input decays to
IEXP_S = 0.6931471805599453 / (1 << IEXP_K)
IEXP_BQ = int(round(IEXP_B / IEXP_S))
IEXP_AQ = int(round(IEXP_A * (1 << IEXP_SH) * IEXP_S * IEXP_S))
IEXP_CQ = int(round(IEXP_C * (1 << IEXP_SH)))
IEXP_QMAX = int(round(-IEXP_CLAMP / IEXP_S))

# ---- the integer GELU ---------------------------------------------------------------------------
# gelu(x) = relu(x) - |x| * h(|x|), h(u) = Phi(-u) for the erf form (0.5 * (1 + tanh(-c(u + 0.044715u^3)))
# for the tanh form). h is a degree-6 polynomial on [0, GELU_U] (0 beyond: |x| h(|x|) < 1.3e-4 there),
# least squares weighted by u (the correction is |x| * h) on 40001 even points -- the same fit
# the test re-derives. Max |gelu error| 5.9e-4 (erf) / 5.7e-4 (tanh), against 1.8e-2 for the I-BERT
# 2nd-order i-GELU, whose error alone moved 7 of SmolVLA's 16000 trajectory values past the float floor.
GELU_U = 4.0
GELU_DEGREE = 6
GELU_COEFFS = {
    "none": (
        0.4986434897,
        -0.3828156593,
        -0.05509493827,
        0.1489309257,
        -0.06093190645,
        0.01055039015,
        -0.0006891086172,
    ),
    "tanh": (0.4990561916, -0.385437155, -0.04957723216, 0.1442767735, -0.05910758069, 0.01021412711, -0.0006654384426),
}
#: GELU's input grid (int16 per row), the fixed-point grid of |x| (GELU_F fraction bits), the
#: coefficients' fixed point (GELU_P bits), and the mantissa bits of the per-row rescale onto |x|'s grid.
GELU_QMAX = (1 << 15) - 1
GELU_F = 12
GELU_P = 24
GELU_MANT_BITS = 20

#: The int8 range every function quantizes its input or output to (symmetric, zero excluded).
Q8 = 127
#: Layer norm's input grid: int16, so the row's mean and variance keep ~4.5 decimal digits.
LN_QMAX = (1 << 15) - 1
#: Fraction bits of the normalized layer-norm value, and of its per-row reciprocal factor.
LN_OUT_SH = 14
LN_RECIP_SH = 16
#: Newton iterations of the integer square root. From ``2**LN_SQRT_START`` (at or above every root
#: an int16 row can produce) the iteration halves at worst until it is within a factor of two, then
#: converges quadratically; 24 covers the whole range with room. A test checks it against isqrt.
LN_SQRT_START = 16
LN_SQRT_ITERS = 24

#: What the mode replaced on its last forward: ``{function: count}``.
SOFTMAX, GELU, LAYER_NORM = "softmax", "gelu", "layer_norm"


#: The aten operators this module's integer arithmetic is captured as. A capture bridge without an
#: exact decomposition of each leaves it OPAQUE (an undefined symbol, not a slow path), so the capture
#: worker refuses ``--integer-nonlinear`` against such a bridge instead of producing that program.
REQUIRED_DECOMPOSITIONS = (
    "aten.bitwise_right_shift.Tensor",
    "aten.bitwise_right_shift.Tensor_Scalar",
    "aten.div.Tensor_mode",
)


def _shift_right_const(x: Any, n: int) -> Any:
    """Arithmetic ``x >> n`` (``n`` a python int) for an integer tensor ``x``: ``floor(x / 2**n)``.

    Captured as ``aten.bitwise_right_shift.Tensor_Scalar``, which the capture bridge decomposes to one
    ``arith.shrsi`` per element. (It used to be built as a float64 round trip -- cast, exact divide by
    the power of two, floor, cast back -- because the bridge left the shift opaque; that was the
    same value whenever ``|x| < 2**53``, at several times the instructions, and the bulk of the
    program's casts.)
    """
    import torch

    return torch.bitwise_right_shift(x, n)


def _shift_right_var(x: Any, k: Any) -> Any:
    """``x >> k`` for a per-element non-negative integer shift amount ``k`` (a tensor), as one
    ``arith.shrsi`` (see :func:`_shift_right_const`). ``k`` is cast to ``x``'s dtype here so the shift
    reads it without a promotion of ``x``."""
    import torch

    return torch.bitwise_right_shift(x, k.to(x.dtype))


def _quantize_row(xf: Any, s: Any, qmax: int) -> Any:
    """``round(xf / s)`` clamped to ``[-qmax-1, qmax]``, per row (last axis), through a REAL
    ``torchao.quantization.quant_primitives.quantize_affine`` call rather than a bare
    ``round``+``cast`` pair.

    Numerically this is exactly what a bare ``torch.round(xf / s).clamp(...)`` computes (measured:
    round-half-to-even, matching torch.round bit for bit, including the clamp bounds) -- the reason
    to route it through ``quantize_affine`` at all is that the m2m capture bridge decomposes it to a
    real ``quant_ext.quantize_per_channel`` marker op the way a captured model's own dynamic
    quantization already is (``merlin.development.targetgen._activation_contractions`` uses the identical call),
    which is what lets ``compute_groups._window_sum_of`` read the operand's format the same way
    ``_window_mean_of`` reads a captured mean's -- a bare round+cast leaves no such marker.
    """
    import torch
    from torchao.quantization.quant_primitives import quantize_affine

    block = tuple([1] * (xf.dim() - 1) + [xf.shape[-1]])
    scale = s.reshape(*xf.shape[:-1])
    zero_point = scale.new_zeros(scale.shape, dtype=torch.int64)
    dtype = torch.int8 if qmax == 127 else torch.int16
    return quantize_affine(xf, block, scale, zero_point, dtype, -qmax - 1, qmax)


def _floordiv(a: Any, b: Any) -> Any:
    """``floor(a / b)`` for an integer tensor ``a`` and an integer tensor or python int ``b``, rounding
    toward -infinity (``-7 // 2 == -4``), as one ``arith.floordivsi`` per element (captured as
    ``aten.div.Tensor_mode`` with ``rounding_mode='floor'``). Every divisor in this module is positive."""
    import torch

    return torch.div(a, b, rounding_mode="floor")


def iexp_int(q: Any) -> Any:
    """``exp(q * S)`` for integer ``q`` in ``[-IEXP_QMAX, 0]``, as an int64 at ``2**IEXP_SH``."""
    import torch

    q = q.to(torch.int64)
    z = _shift_right_const(0 - q, IEXP_K)  # floor(-q / 2**K); q<=0 so -q>=0, built as a subtraction
    # (`torch.neg` is opaque on an integer tensor in the capture bridge; `0 - q` is `aten.sub.Tensor`).
    r = q + z * (1 << IEXP_K)  # in (-2**K, 0]
    t = r + IEXP_BQ
    e = IEXP_AQ * t * t + IEXP_CQ
    return _shift_right_var(e, z)


def iexp_zero() -> int:
    """``iexp_int(0)``: the fixed-point value of the row's largest term."""
    return IEXP_AQ * IEXP_BQ * IEXP_BQ + IEXP_CQ


def integer_softmax(x: Any, dim: int = -1) -> Any:
    """Softmax over ``dim`` in integer arithmetic; returns ``x``'s dtype, contiguous as the framework's is
    (a caller may ``view`` it), per the module docstring."""
    import torch

    xf = x.float()
    m = xf.amax(dim=dim, keepdim=True)
    q = torch.round((xf - m) / IEXP_S).clamp(-IEXP_QMAX, 0).to(torch.int32)
    e = iexp_int(q)
    e0 = iexp_zero()
    p = _floordiv(e * Q8 + e0 // 2, e0)  # int8 values in [0, 127]
    total = p.sum(dim=dim, keepdim=True)
    return (p.float() / total.float()).to(x.dtype).contiguous()


def _row_scale(x: Any, qmax: int) -> Any:
    """Per-row (last axis) symmetric scale ``max|x| / qmax``, 1 for an all-zero row."""
    import torch

    amax = x.abs().amax(dim=-1, keepdim=True)
    return torch.where(amax > 0, amax / qmax, torch.ones_like(amax))


def _dyadic(v: Any, bits: int) -> tuple[Any, Any]:
    """``(m, k)`` int64 per row with ``m / 2**k ~ v`` and ``m`` near ``2**bits``: a scale as a requant does.

    Floating point on purpose, and per ROW (``v`` is one scale per row): it turns the row's float scale
    into a mantissa and an exponent, which is a float operation by definition -- a ``log``, a ``floor``
    and a multiply by ``2**k``. The per-element work it feeds is integer.
    """
    import torch

    # `aten.log2.default` is not in the capture bridge's decomposition table (only `log`, `aten.log.default`,
    # is); `log2(v) = log(v) / log(2)` is exact in the sense that matters here -- `k` is then floored and
    # clamped to an INTEGER, so the only requirement is that this land on the same integer `log2` would.
    log2v = torch.log(v) / math.log(2.0)
    k = torch.maximum(
        torch.minimum((bits - torch.floor(log2v)), torch.full_like(v, 62 - bits)), torch.zeros_like(v)
    ).to(torch.int64)
    m = torch.round(v.double() * torch.pow(2.0, k.double())).to(torch.int64)
    return m, k


def integer_gelu(x: Any, approximate: str = "none") -> Any:
    """GELU in integer arithmetic over a per-row int16 quantization of ``x``.

    ``gelu(x) = relu(x) - |x| h(|x|)``: ``|x|`` is brought onto a fixed grid by one dyadic multiply per
    element (the row's scale as a mantissa and a shift), ``h`` is a degree-6 polynomial evaluated by
    Horner's rule in fixed point with the same coefficients for every row, and the result stays on the
    row's grid until the one dequantizing multiply. ``approximate`` selects the form the call declared.
    """
    import torch

    coeffs = GELU_COEFFS["tanh" if approximate == "tanh" else "none"]
    xf = x.float()
    s = _row_scale(xf, GELU_QMAX)
    q = torch.round(xf / s).clamp(-GELU_QMAX, GELU_QMAX).to(torch.int64)
    m, k = _dyadic(s * float(1 << GELU_F), GELU_MANT_BITS)
    a = q.abs()
    # `.clamp()` on an integer tensor builds its bound as a FloatAttr regardless of the operand's
    # element type in the m2m capture bridge's clamp decomposition (a torch.export -> linalg import
    # bug external to this module), which then fails `VerifyException: Unexpected attribute i64` --
    # measured capturing this exact op. `minimum` against a full-value tensor is the same one-sided
    # clamp and decomposes through the ordinary elementwise-min path instead.
    u_max = torch.full_like(a, int(GELU_U * (1 << GELU_F)))
    u = torch.minimum(_shift_right_var(a * m, k), u_max)  # |x| at 2**F
    fixed = [int(round(c * (1 << GELU_P))) for c in coeffs]
    h = torch.full_like(u, fixed[-1])
    for c in reversed(fixed[:-1]):
        h = _shift_right_const(h * u, GELU_F) + c
    h = torch.where(u >= int(GELU_U * (1 << GELU_F)), torch.zeros_like(h), h)
    # `relu(q)` as `maximum(q, 0)`, the same integer-clamp-bug dodge as above.
    out = torch.maximum(q, torch.zeros_like(q)) * (1 << GELU_P) - a * h  # gelu(x) / s at 2**P
    return (out.float() * (s / float(1 << GELU_P))).to(x.dtype).contiguous()


def isqrt_int(n: Any) -> Any:
    """``max(1, floor(sqrt(n)))`` of a non-negative int64 tensor, by a fixed number of Newton steps.

    Never below 1: the root divides the next step (and the caller's normalization), so a zero root --
    a constant row whose epsilon rounds to nothing -- is held at the smallest nonzero one."""
    import torch

    y = torch.full_like(n, 1 << LN_SQRT_START)
    ones = torch.ones_like(n)
    for _ in range(LN_SQRT_ITERS):
        step = _shift_right_const(y + _floordiv(n, y), 1)
        # See the comment in `integer_gelu`: `.clamp()` on an integer tensor hits the capture bridge's
        # clamp decomposition bug (`Unexpected attribute i64`); `maximum`/`minimum` avoid it.
        y = torch.maximum(torch.minimum(y, step), ones)
    return y


def integer_layer_norm(x: Any, normalized_shape: Any, weight: Any = None, bias: Any = None, eps: float = 1e-5) -> Any:
    """Layer norm over the last axis with an integer mean, variance and square root."""
    import torch

    shape = (int(normalized_shape),) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    if len(shape) != 1:
        return None  # a multi-axis norm is left to the framework (and counted by the caller)
    n = int(shape[0])
    xf = x.float()
    s = _row_scale(xf, LN_QMAX)
    # `_quantize_row` computes the identical value as `torch.round(xf / s).to(torch.int64)` (the
    # numerics are unchanged -- verified round-half-to-even and clamp bounds match, bit for bit),
    # but through a real `quantize_affine` call, so the capture bridge decomposes this point to a
    # `quant_ext.quantize_per_channel` marker instead of an opaque round+cast: that marker is what
    # lets `compute_groups._window_sum_of` read this row's true int16 range off the op itself,
    # rather than off the float `xf` a bare round leaves as the only legible operand.
    q = _quantize_row(xf, s, LN_QMAX).to(torch.int64)
    mean = _floordiv(q.sum(dim=-1, keepdim=True), n)
    d = q - mean
    var = _floordiv((d * d).sum(dim=-1, keepdim=True), n)
    var = var + torch.round(eps / (s * s)).to(torch.int64)
    std = isqrt_int(var)
    factor = _floordiv(torch.full_like(std, 1 << (LN_OUT_SH + LN_RECIP_SH)), std)
    y = _shift_right_const(d * factor, LN_RECIP_SH)  # (x - mean) / std at 2**LN_OUT_SH
    out = y.float() * (1.0 / float(1 << LN_OUT_SH))
    if weight is not None:
        out = out * weight.float()
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype).contiguous()


class Census:
    """What the mode replaced, and what it left, on its LAST forward."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], int] = {}

    def add(self, fn: str, verdict: str) -> None:
        self.rows[(fn, verdict)] = self.rows.get((fn, verdict), 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sites": [{"function": f, "verdict": v, "count": n} for (f, v), n in sorted(self.rows.items())],
        }


def _mode_class():
    import torch
    from torch.overrides import TorchFunctionMode

    softmaxes = {torch.softmax, torch.nn.functional.softmax, torch.Tensor.softmax, torch.special.softmax}
    gelus = {torch.nn.functional.gelu}
    norms = {torch.nn.functional.layer_norm, torch.layer_norm}

    class IntegerNonlinearMode(TorchFunctionMode):
        def __init__(self, census: Census) -> None:
            super().__init__()
            self.census = census

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func in softmaxes and args:
                x = args[0]
                dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
                if dim is not None and x.is_floating_point() and x.dim() >= 1:
                    self.census.add(SOFTMAX, "integer")
                    y = integer_softmax(x, int(dim))
                    dtype = kwargs.get("dtype", args[2] if len(args) > 2 else None)
                    return y.to(dtype) if dtype is not None else y
                self.census.add(SOFTMAX, "left_floating")
            elif func in gelus and args and args[0].is_floating_point():
                self.census.add(GELU, "integer")
                return integer_gelu(args[0], str(kwargs.get("approximate", args[1] if len(args) > 1 else "none")))
            elif func in norms and args and args[0].is_floating_point():
                names = ("input", "normalized_shape", "weight", "bias", "eps")
                bound = dict(zip(names, args))
                bound.update(kwargs)
                y = integer_layer_norm(
                    bound["input"],
                    bound["normalized_shape"],
                    bound.get("weight"),
                    bound.get("bias"),
                    float(bound.get("eps", 1e-5)),
                )
                if y is not None:
                    self.census.add(LAYER_NORM, "integer")
                    return y
                self.census.add(LAYER_NORM, "left_floating: more than one normalized axis")
            return func(*args, **kwargs)

    return IntegerNonlinearMode


def install(model: Any) -> Census:
    """Route ``model``'s forward through the mode, OUTSIDE any mode already installed on it.

    Installed after ``_activation_contractions.install``, this mode is the outer one: the contraction
    mode sees every call first, and the softmax its spelled-out attention computes reaches this mode
    from inside it -- so a fused attention's softmax is integer too. Like that mode, it replaces the
    forward on the INSTANCE, so the export and the eager golden both run it.
    """
    import functools

    census = Census()
    mode_cls = _mode_class()
    original = model.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        census.rows.clear()
        with mode_cls(census):
            return original(*args, **kwargs)

    model.forward = forward
    return census
