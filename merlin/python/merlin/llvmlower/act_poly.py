"""Vectorizable polynomial approximation of transcendental activations (default-off feature).

The baseline RVV pipeline lowers ``math.erf``/``math.exp``/``math.tanh`` through
``convert-math-to-libm`` -> a SCALAR libm call (``erff``/``expf``/``tanhf``) in a loop. The
elementwise activation ``linalg.generic`` is never vectorized (the baseline transform schedule
only matches matmul/batch_matmul), so a GELU/sigmoid/SiLU activation runs as a scalar libm loop —
the ~11-18x gap vs XNNPACK's vectorized-polynomial RVV kernels measured in
``output/kernels/ceiling/cross_framework_ops_k1.md``.

This module is the GENERAL compiler capability that closes that gap: a transcendental -> inline
arith-polynomial rewrite over the math ops, run BEFORE vectorization so the resulting arith chain
(mul/add/sub/div/select + bitcast/shift for the 2^n exponent) vectorizes to ``vfmacc`` chains
instead of a scalar call. It is NOT a hand-written kernel and NOT op/shape-specific: it rewrites
every ``math.exp``/``math.erf``/``math.tanh`` in the module, so GELU (erf), sigmoid/SiLU (exp), and
tanh all benefit from one transform.

Accuracy: these are APPROXIMATIONS (f32 minimax), so the gate is cosine / relative-error vs the
libm reference, not bit-exact (see the feature's tests + the report). The polynomials:
  * exp(x): range-reduce x = n*ln2 + r, exp = 2^n * P5(r) (degree-6 Horner of the Taylor series,
    f32-refined), 2^n assembled via (int(n)+127)<<23 bitcast. max rel err ~6.6e-7 over |x|<~10.
  * erf(x): Abramowitz-Stegun 7.1.26 rational-in-t poly with t = 1/(1+p|x|) and an exp(-x^2) tail,
    sign restored via copysign. max abs err ~5.8e-7.
  * tanh(x) = 2/(1+exp(-2x)) - 1 (reuses the exp poly).
The exp coefficients/structure mirror XNNPACK's rr2-p5 range-reduction (the CEILING REFERENCE);
the erf uses the standard A&S 7.1.26 rational form. We emit the MLIR; XNNPACK's kernel is only the
coefficient/structure reference, never linked.

This file runs inside the model2MLIR venv (it needs the upstream MLIR Python bindings). It is
imported by the lowering runner ONLY when the ``vectorized_transcendental_activation`` feature is
enabled; with the feature off it is never imported and the pipeline is byte-identical.
"""
from __future__ import annotations


def rewrite_source() -> str:
    """Return the self-contained Python source of the rewriter, to be prepended to the lowering
    runner (which executes in the m2m venv). Kept as a source string so the runner stays a single
    self-contained script and no extra import path wiring into the m2m venv is needed."""
    return _REWRITER_SRC


# The rewriter body. Self-contained (only torch_mlir.{ir,dialects.arith,dialects.math}); defines
# `apply_activation_polynomial(module, ctx)` which the runner calls after Module.parse and before
# the PassManager runs. Emitted as a string so it can be prepended to the in-venv runner verbatim.
_REWRITER_SRC = r'''
# --- vectorizable transcendental-activation polynomial rewriter (default-off feature) ---
from torch_mlir.dialects import arith as _arith, math as _math

def _ap_is_vec(t):
    try:
        _ir.VectorType(t); return True
    except (ValueError, TypeError):
        return False

def _ap_i32():
    return _ir.IntegerType.get_signless(32)

def _ap_int_ty(t):
    if _ap_is_vec(t):
        vt = _ir.VectorType(t)
        return _ir.VectorType.get(vt.shape, _ap_i32())
    return _ap_i32()

def _ap_fconst(t, v):
    if _ap_is_vec(t):
        et = _ir.VectorType(t).element_type
        attr = _ir.DenseElementsAttr.get_splat(t, _ir.FloatAttr.get(et, float(v)))
    else:
        attr = _ir.FloatAttr.get(t, float(v))
    return _arith.ConstantOp(t, attr).result

def _ap_iconst(ity, v):
    if _ap_is_vec(ity):
        attr = _ir.DenseElementsAttr.get_splat(ity, _ir.IntegerAttr.get(_ap_i32(), int(v)))
    else:
        attr = _ir.IntegerAttr.get(_ap_i32(), int(v))
    return _arith.ConstantOp(ity, attr).result

def _ap_fma(a, b, c):
    # a*b + c as a single math.fma -> llvm.intr.fma -> RVV vfmacc (the fused MAC; emitting fma
    # explicitly makes the Horner chain a vfmacc chain independent of -ffp-contract).
    return _math.FmaOp(a, b, c).result

def _ap_horner(coeffs, r, t):
    # Estrin/Horner P(r) = (((c0*r + c1)*r + c2)... evaluated with fused multiply-adds.
    p = _ap_fconst(t, coeffs[0])
    for c in coeffs[1:]:
        p = _ap_fma(p, r, _ap_fconst(t, c))
    return p

def _ap_exp(x, t):
    # exp(x): x = n*ln2 + r ; exp = 2^n * polyexp(r). 2^n via (int(n)+127)<<23 bitcast.
    LOG2E = _ap_fconst(t, 1.4426950408889634)
    NEG_LN2HI = _ap_fconst(t, -0.6931471824645996)
    NEG_LN2LO = _ap_fconst(t, 1.904654323148236e-09)
    LO = _ap_fconst(t, -87.0); HI = _ap_fconst(t, 88.0)
    x = _arith.MaximumFOp(x, LO).result
    x = _arith.MinimumFOp(x, HI).result
    nf = _math.RoundEvenOp(_arith.MulFOp(x, LOG2E).result).result
    r = _ap_fma(nf, NEG_LN2HI, x)            # x - n*ln2_hi  (fused)
    r = _ap_fma(nf, NEG_LN2LO, r)            # r - n*ln2_lo  (fused)
    p = _ap_horner([0.0013888889, 0.008333334, 0.041666668, 0.16666667, 0.5, 1.0, 1.0], r, t)
    ity = _ap_int_ty(t)
    ni = _arith.FPToSIOp(ity, nf).result
    ni = _arith.AddIOp(ni, _ap_iconst(ity, 127)).result
    ni = _arith.ShLIOp(ni, _ap_iconst(ity, 23)).result
    scale = _arith.BitcastOp(t, ni).result
    return _arith.MulFOp(p, scale).result

def _ap_erf(x, t):
    # erf(x): Abramowitz-Stegun 7.1.26, sign via copysign, exp(-x^2) tail.
    a = [_ap_fconst(t, c) for c in (1.061405429, -1.453152027, 1.421413741,
                                    -0.284496736, 0.254829592)]
    pc = _ap_fconst(t, 0.3275911); one = _ap_fconst(t, 1.0); zero = _ap_fconst(t, 0.0)
    neg_one = _ap_fconst(t, -1.0)
    ax = _math.AbsFOp(x).result
    # sign(x) as +-1 via a select (math.copysign has no vector LLVM legalization in this build).
    # OGE (predicate 3) so x>=0 -> +1, else -1 (erf(0)=0 either way, so the sign of 0 is irrelevant).
    pred = _ir.IntegerAttr.get(_ir.IntegerType.get_signless(64), int(_arith.CmpFPredicate.OGE))
    pos = _arith.CmpFOp(pred, x, zero).result
    s = _arith.SelectOp(pos, one, neg_one).result
    tt = _arith.DivFOp(one, _ap_fma(pc, ax, one)).result          # 1/(1 + p*|x|)
    poly = a[0]
    for c in a[1:]:
        poly = _ap_fma(poly, tt, c)                               # fused Horner
    poly = _arith.MulFOp(poly, tt).result
    negx2 = _arith.NegFOp(_arith.MulFOp(ax, ax).result).result
    e = _ap_exp(negx2, t)
    y = _arith.SubFOp(one, _arith.MulFOp(poly, e).result).result  # 1 - poly*exp(-x^2)
    return _arith.MulFOp(s, y).result

def _ap_tanh(x, t):
    # tanh(x) = 2*sigmoid(2x) - 1 = 2/(1+exp(-2x)) - 1.
    two = _ap_fconst(t, 2.0); one = _ap_fconst(t, 1.0)
    e = _ap_exp(_arith.NegFOp(_arith.MulFOp(two, x).result).result, t)
    return _arith.SubFOp(_arith.DivFOp(two, _arith.AddFOp(one, e).result).result, one).result

_AP_BUILDERS = {"math.exp": _ap_exp, "math.erf": _ap_erf, "math.tanh": _ap_tanh}

def apply_activation_polynomial(module, ctx):
    """Replace every unary math.exp/erf/tanh in `module` with its inline arith polynomial.
    Returns the number of ops rewritten. Run before the pass manager / vectorization."""
    targets = []
    def _walk(op):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    if o.operation.name in _AP_BUILDERS and len(o.operands) == 1:
                        targets.append(o)
                    _walk(o.operation)
    _walk(module.operation)
    n = 0
    for o in targets:
        op = o.operation
        x = op.operands[0]; t = x.type
        with _ir.InsertionPoint(op), _ir.Location.unknown():
            new = _AP_BUILDERS[op.name](x, t)
        op.result.replace_all_uses_with(new)
        op.erase()
        n += 1
    return n
# --- end activation polynomial rewriter ---
'''
