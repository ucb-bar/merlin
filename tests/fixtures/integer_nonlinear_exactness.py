"""Old-vs-new exactness of ``_integer_nonlinear``'s integer helpers -- run INSIDE the capture venv.

usage: python integer_nonlinear_exactness.py <targetgen dir> <section> [<section> ...]

The four helpers used to compute integer floor-shift and floor-division through float64 (cast, divide,
floor, cast back), because the capture bridge left the integer ops opaque. They now use the integer
ops themselves. ``OLD_*`` below are the float64 versions VERBATIM, and every section compares old and
new with exact equality -- never a tolerance. A section prints ``ok <section> <what was covered>``.

Where an operand domain is enumerable it is enumerated (marked EXHAUSTIVE); where it is not, the
equality is an identity on ``|x| < 2**53`` (float64 holds every such integer exactly, a division by a
power of two is exact, and ``floor`` of an exact value is exact), and the section enumerates the dense
low range plus the cases where a float quotient could round across an integer.
"""

from __future__ import annotations

import importlib.util
import math
import sys

import torch

TARGETGEN = sys.argv[1]
torch.manual_seed(0)


# ---- the float64 helpers this module used before, verbatim ----------------------------------------
def OLD_shift_right_const(x, n):
    return torch.floor(x.double() / float(1 << n)).to(x.dtype)


def OLD_shift_right_var(x, k):
    return torch.floor(x.double() / torch.pow(2.0, k.double())).to(x.dtype)


def OLD_floordiv(a, b):
    bd = b.double() if hasattr(b, "double") else float(b)
    return torch.floor(a.double() / bd).to(a.dtype)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, f"{TARGETGEN}/_integer_nonlinear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NEW = _load("_integer_nonlinear_new")
OLD = _load("_integer_nonlinear_old")
OLD._shift_right_const = OLD_shift_right_const
OLD._shift_right_var = OLD_shift_right_var
OLD._floordiv = OLD_floordiv
I64 = torch.int64


def same(a, b, what):
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    if a.is_floating_point():
        width = {torch.float32: torch.int32, torch.float64: torch.int64, torch.bfloat16: torch.int16}[a.dtype]
        assert torch.equal(a.view(width), b.view(width)), what  # bit patterns, so -0.0 and NaN count
    else:
        assert torch.equal(a, b), (what, int((a != b).sum()))


def chunks(t, size=1 << 24):
    for i in range(0, t.numel(), size):
        yield t[i : i + size]


def dense_and_edges(dense_bits: int, top_bits: int = 52):
    """Every integer in ``[-2**dense_bits, 2**dense_bits]`` plus ``±(2**j + d)`` for j up to ``top_bits``."""
    dense = torch.arange(-(1 << dense_bits), (1 << dense_bits) + 1, dtype=I64)
    edges = sorted({s * ((1 << j) + d) for j in range(dense_bits, top_bits + 1) for d in range(-3, 4) for s in (1, -1)})
    return torch.cat([dense, torch.tensor(edges, dtype=I64)])


# ---- helper-level identities ---------------------------------------------------------------------
def s_shift_const():
    x = dense_and_edges(20)
    for n in range(0, 63):
        same(NEW._shift_right_const(x, n), OLD_shift_right_const(x, n), f"shift_const n={n}")
    return "x in [-2^20, 2^20] (all) + 2^j+-3 to 2^52, n in [0, 62]"


def s_shift_var():
    x = dense_and_edges(18)
    for k in range(0, 64):
        kk = torch.full_like(x, k)
        same(NEW._shift_right_var(x, kk), OLD_shift_right_var(x, kk), f"shift_var k={k}")
    return "x in [-2^18, 2^18] (all) + 2^j+-3 to 2^52, every k in [0, 63] as a tensor"


def s_floordiv():
    a = torch.arange(-(1 << 12), (1 << 12) + 1, dtype=I64)
    b = torch.arange(1, (1 << 12) + 1, dtype=I64)
    aa, bb = torch.meshgrid(a, b, indexing="ij")
    aa, bb = aa.reshape(-1), bb.reshape(-1)
    same(NEW._floordiv(aa, bb), OLD_floordiv(aa, bb), "floordiv dense")
    same(NEW._floordiv(aa, -bb), OLD_floordiv(aa, -bb), "floordiv dense negative divisor")
    # where a float quotient could round up across an integer: a = q*b - 1 just below a multiple,
    # for every divisor up to 2**16 and q placing a near 2**51
    b = torch.arange(1, (1 << 16) + 1, dtype=I64)
    for top in (20, 32, 40, 46, 51):
        q = torch.div(torch.full_like(b, 1 << top), b, rounding_mode="floor")
        for d in (-1, 0, 1):
            a = q * b + d
            same(NEW._floordiv(a, b), OLD_floordiv(a, b), f"floordiv near multiple top={top} d={d}")
            same(NEW._floordiv(-a, b), OLD_floordiv(-a, b), f"floordiv near multiple top={top} d={d} negated")
    for c in (1, 2, 3, 7, 768, 1 << 16, NEW.iexp_zero()):
        x = dense_and_edges(20, 51)
        same(NEW._floordiv(x, c), OLD_floordiv(x, c), f"floordiv by python int {c}")
    return (
        "a in [-2^12, 2^12] x b in [1, 2^12] (all pairs, both divisor signs); a = q*b-1, q*b, q*b+1 "
        "for every b in [1, 2^16] at 2^20..2^51; python-int divisors on [-2^20, 2^20] + edges"
    )


# ---- the callers' own operand domains -----------------------------------------------------------
def s_iexp():
    q = torch.arange(-NEW.IEXP_QMAX, 1, dtype=torch.int32)
    same(NEW.iexp_int(q), OLD.iexp_int(q), "iexp_int")
    return f"EXHAUSTIVE: every q in [-{NEW.IEXP_QMAX}, 0] (both shifts of iexp_int)"


def s_softmax_numerator():
    q = torch.arange(-NEW.IEXP_QMAX, 1, dtype=torch.int32)
    e0 = NEW.iexp_zero()
    new = NEW._floordiv(NEW.iexp_int(q) * NEW.Q8 + e0 // 2, e0)
    old = OLD_floordiv(OLD.iexp_int(q) * OLD.Q8 + e0 // 2, e0)
    same(new, old, "softmax numerator")
    assert int(new.min()) >= 0 and int(new.max()) == NEW.Q8
    return "EXHAUSTIVE: the int8 numerator for every q on the exponent grid"


def _horner(module, shift, u, coeffs):
    fixed = [int(round(c * (1 << module.GELU_P))) for c in coeffs]
    h = torch.full_like(u, fixed[-1])
    for c in reversed(fixed[:-1]):
        h = shift(h * u, module.GELU_F) + c
    return h


def s_gelu_horner():
    u = torch.arange(0, int(NEW.GELU_U * (1 << NEW.GELU_F)) + 1, dtype=I64)
    for form, coeffs in NEW.GELU_COEFFS.items():
        same(_horner(NEW, NEW._shift_right_const, u, coeffs), _horner(OLD, OLD_shift_right_const, u, coeffs), form)
    return f"EXHAUSTIVE: every u in [0, {int(NEW.GELU_U * (1 << NEW.GELU_F))}], both forms, all Horner steps"


def s_gelu_rescale():
    a = torch.arange(0, NEW.GELU_QMAX + 1, dtype=I64)
    top_k = 62 - NEW.GELU_MANT_BITS
    lo, hi = 1 << NEW.GELU_MANT_BITS, 1 << (NEW.GELU_MANT_BITS + 1)
    ms = {0, 1, lo - 1, lo, lo + 1, hi - 1, hi, (1 << 53) // NEW.GELU_QMAX}
    ms |= set(torch.randint(lo, hi + 1, (48,)).tolist())
    for m in sorted(ms):
        x = a * m
        for k in range(0, top_k + 1):
            kk = torch.full_like(x, k)
            same(NEW._shift_right_var(x, kk), OLD_shift_right_var(x, kk), f"gelu rescale m={m} k={k}")
    return (
        f"EXHAUSTIVE in |x|: every |x| in [0, {NEW.GELU_QMAX}] for every k in [0, {top_k}] and {len(ms)} "
        "mantissas (the extremes of the dyadic's range, the largest with |x|*m < 2^53, 48 drawn)"
    )


def s_isqrt():
    n = torch.arange(0, 1 << 24, dtype=I64)
    r = torch.arange(1, (1 << 16) + 1, dtype=I64)
    squares = torch.cat([r * r - 1, r * r, r * r + 1])
    top = torch.arange((1 << 32) - (1 << 20), (1 << 32) + (1 << 20), dtype=I64)
    pw = torch.tensor([(1 << j) + d for j in range(0, 53) for d in (-1, 0, 1) if (1 << j) + d >= 0], dtype=I64)
    for part, label in ((n, "dense"), (squares, "squares"), (top, "top"), (pw, "powers")):
        for c in chunks(part, 1 << 22):
            new = NEW.isqrt_int(c)
            same(new, OLD.isqrt_int(c), f"isqrt {label}")
    small = NEW.isqrt_int(torch.arange(0, 1 << 20, dtype=I64))
    assert small.tolist() == [max(1, math.isqrt(v)) for v in range(1 << 20)]
    return (
        "EXHAUSTIVE on [0, 2^24) with all 24 Newton steps; k^2-1, k^2, k^2+1 for every k in [1, 2^16]; "
        "[2^32-2^20, 2^32+2^20) (all); 2^j+-1 to 2^52"
    )


def s_layer_norm_integer_steps(width=768):
    total = torch.arange(-(NEW.LN_QMAX + 1) * width, NEW.LN_QMAX * width + 1, dtype=I64)
    for c in chunks(total):
        same(NEW._floordiv(c, width), OLD_floordiv(c, width), "ln mean")
    for c in chunks(torch.arange(0, 1 << 26, dtype=I64)):
        same(NEW._floordiv(c, width), OLD_floordiv(c, width), "ln variance (dense)")
    top = width * (2 * NEW.LN_QMAX + 1) ** 2
    q = torch.unique(torch.logspace(0, math.log10(top // width), 4000, dtype=torch.float64).to(I64))
    for d in range(-width, width + 1):
        c = q * width + d
        c = c[(c >= 0) & (c <= top)]
        same(NEW._floordiv(c, width), OLD_floordiv(c, width), "ln variance near multiples")
    std = torch.arange(1, (1 << 16) + 1, dtype=I64)
    num = torch.full_like(std, 1 << (NEW.LN_OUT_SH + NEW.LN_RECIP_SH))
    factor = NEW._floordiv(num, std)
    same(factor, OLD_floordiv(num, std), "ln factor")
    d = torch.arange(-(2 * NEW.LN_QMAX + 1), 2 * NEW.LN_QMAX + 2, dtype=I64)
    picks = sorted(
        {0, 1, 2, 3, 255, 256, 257, 4095, 4096, (1 << 16) - 1} | set(torch.randint(0, 1 << 16, (40,)).tolist())
    )
    for i in picks:
        x = d * factor[i]
        same(NEW._shift_right_const(x, NEW.LN_RECIP_SH), OLD_shift_right_const(x, NEW.LN_RECIP_SH), "ln normalize")
    return (
        f"width {width}: EXHAUSTIVE mean over every row sum of an int16 row; variance on [0, 2^26) (all) and "
        f"+-{width} around 4000 multiples up to its maximum; EXHAUSTIVE factor for every root in [1, 2^16]; "
        f"normalize for every d in the int16 difference range at {len(picks)} roots"
    )


# ---- end to end ----------------------------------------------------------------------------------
def _tensors():
    g = torch.Generator().manual_seed(1)
    yield "randn", torch.randn(4, 7, 96, generator=g)
    yield "randn*30", torch.randn(3, 768, generator=g) * 30
    yield "tiny", torch.randn(2, 768, generator=g) * 1e-4
    yield "huge", torch.randn(2, 768, generator=g) * 1e4
    yield "zeros", torch.zeros(2, 64)
    yield "constant", torch.full((2, 64), 3.25)
    yield "one_hot", torch.eye(64)[:5] * 50
    yield "ramp", torch.linspace(-12, 12, 3072).reshape(1, -1)
    yield "bf16", (torch.randn(3, 128, generator=g) * 4).to(torch.bfloat16)
    yield "attention_rows", torch.randn(1, 12, 64, 64, generator=g) * 6


def s_end_to_end():
    covered = []
    for name, x in _tensors():
        same(NEW.integer_gelu(x, "none"), OLD.integer_gelu(x, "none"), f"gelu {name}")
        same(NEW.integer_gelu(x, "tanh"), OLD.integer_gelu(x, "tanh"), f"gelu tanh {name}")
        masked = x.float().clone()
        masked[..., ::5] = float("-inf")
        for dim in (-1, 0):
            same(NEW.integer_softmax(x, dim), OLD.integer_softmax(x, dim), f"softmax {name} dim={dim}")
        same(NEW.integer_softmax(masked, -1), OLD.integer_softmax(masked, -1), f"softmax masked {name}")
        w = x.shape[-1]
        weight, bias = torch.randn(w), torch.randn(w)
        for eps in (1e-5, 1e-6):
            same(
                NEW.integer_layer_norm(x, w, weight, bias, eps),
                OLD.integer_layer_norm(x, w, weight, bias, eps),
                f"ln {name}",
            )
        same(NEW.integer_layer_norm(x, (w,)), OLD.integer_layer_norm(x, (w,)), f"ln no affine {name}")
        covered.append(name)
    return "integer_softmax/gelu(both forms)/layer_norm on " + ", ".join(covered)


SECTIONS = {
    f.__name__[2:]: f
    for f in (
        s_shift_const,
        s_shift_var,
        s_floordiv,
        s_iexp,
        s_softmax_numerator,
        s_gelu_horner,
        s_gelu_rescale,
        s_isqrt,
        s_layer_norm_integer_steps,
        s_end_to_end,
    )
}

if __name__ == "__main__":
    for section in sys.argv[2:]:
        print("ok", section, SECTIONS[section](), flush=True)
