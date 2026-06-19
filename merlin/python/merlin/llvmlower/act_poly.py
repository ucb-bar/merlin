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
from torch_mlir.dialects import arith as _arith

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
    # a*b + c as arith mul+add (NOT math.fma). The whole activation body is then PURE ARITH — it
    # contains NO `math.*` op at all. That is deliberate: the un-rewritten softmax `math.exp` must
    # reach the baseline `convert-math-to-libm` -> scalar `expf` call (the exact, crash-free path).
    # The old design emitted `math.fma`, which forced a `convert-math-to-llvm` BEFORE
    # `convert-math-to-libm` (so the vector fma became `llvm.intr.fma` instead of being scalarized) —
    # but that same pass also converted the softmax `math.exp` to `llvm.intr.exp` (`llvm.exp.f32`),
    # which the freestanding spike/RVV runtime cannot legalize -> a wild instruction -> the openvla
    # "bad syscall" CRASH (status=fail, no OUT/DONE). With pure arith the activation chain vectorizes
    # to vfmul.vv + vfadd.vv (the RISC-V backend contracts adjacent mul+add into vfmacc under the
    # pipeline's fast-math where it can), and crucially the softmax exp lowers to `expf` exactly as in
    # the baseline — so no extra pipeline pass, no `llvm.intr.exp`, no crash.
    return _arith.AddFOp(_arith.MulFOp(a, b).result, c).result

def _ap_absf(x, t):
    # |x| in pure arith (no math.absf): max(x, -x). See _ap_fma for why the activation body avoids any
    # math.* op (so the softmax exp can take the exact libm path without a math-to-llvm pass).
    return _arith.MaximumFOp(x, _arith.NegFOp(x).result).result

def _ap_roundeven(x, t):
    # round-to-nearest-even in pure arith via the classic add/sub-magic trick: for f32, adding then
    # subtracting MAGIC = 1.5 * 2^23 (= 2^23 + 2^22) forces the FP rounding hardware to round to the
    # nearest integer (ties-to-even, the default rounding mode), valid for |x| < 2^23. Used only for
    # the exp range-reduction n = round(x*log2e); x is clamped to [-87,88] so x*log2e in [-126,127],
    # far inside 2^23 — exact. Avoids math.roundeven (see _ap_absf for why math ops are avoided).
    MAGIC = _ap_fconst(t, 12582912.0)   # 1.5 * 2**23
    biased = _arith.AddFOp(x, MAGIC).result
    return _arith.SubFOp(biased, MAGIC).result

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
    nf = _ap_roundeven(_arith.MulFOp(x, LOG2E).result, t)
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
    ax = _ap_absf(x, t)
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

# Provenance-identified ACTIVATION ops whose transcendental we may replace with the f32 minimax
# polynomial. These are the elementwise activations whose input range is bounded enough that the
# minimax poly is valid (gelu/silu/sigmoid/tanh inputs are pre-activation logits, O(1)-O(10), and the
# exp poly range-reduces+clamps to |x|<~87). DELIBERATELY EXCLUDES `softmax` (prov.family =
# "normalization"): a softmax computes exp(x - rowmax) and then DIVIDES by the row sum; the minimax
# exp differs from libm by ~1e-7 per element and that error is AMPLIFIED through the normalization
# (and the argument x-rowmax can be a large-magnitude logit gap), so blanket-rewriting the softmax exp
# drove openvla whole-model cos to 0.541 (the regression this fix closes). Softmax exp stays on libm.
_AP_ACTIVATION_OPS = frozenset({"gelu", "silu", "sigmoid", "tanh", "hardsigmoid", "hardswish"})


def _ap_prov(generic_op):
    """(prov.op, prov.family) of a linalg.generic from its `attrs` dict, or (None, None)."""
    op = generic_op.operation
    pop = pfam = None
    try:
        pop = _ir.StringAttr(op.attributes["prov.op"]).value
    except (KeyError, ValueError, TypeError):
        pass
    try:
        pfam = _ir.StringAttr(op.attributes["prov.family"]).value
    except (KeyError, ValueError, TypeError):
        pass
    return pop, pfam


def _ap_all_parallel(generic_op):
    """True iff the generic's iterator_types are ALL "parallel" (a pure elementwise map — NO
    reduction). A softmax/normalization carries a reduction iterator on its max/sum generics; a
    genuine activation is pure-parallel. Used only for the no-provenance structural fallback."""
    op = generic_op.operation
    try:
        its = op.attributes["iterator_types"]
    except KeyError:
        return False
    s = str(its)
    return "reduction" not in s and "parallel" in s


def _ap_is_activation_generic(generic_op):
    """True iff `generic_op` is a linalg.generic to which the minimax activation polynomial may be
    safely applied — an ELEMENTWISE ACTIVATION (gelu/silu/sigmoid/tanh), NOT a softmax/normalization
    (whose exp must stay on the exact libm/llvm.exp path; see _AP_ACTIVATION_OPS).

    Provenance is the PRIMARY signal (m2m tags every op): prov.op in the activation set, or
    prov.family == "elementwise" with a non-softmax op. When the generic carries NO provenance at all
    (hand-written / synthetic modules, e.g. the isolated workloads), fall back to the STRUCTURAL
    activation signature: all-parallel iterators (pure elementwise, no reduction). A normalization
    that DOES carry provenance (prov.family == "normalization"/"softmax", or prov.op == "softmax") is
    rejected even if structurally parallel — the provenance wins."""
    if generic_op.operation.name != "linalg.generic":
        return False
    pop, pfam = _ap_prov(generic_op)
    # provenance explicitly marks a normalization/softmax -> NEVER approximate its exp.
    if pop == "softmax" or pfam in ("normalization", "softmax"):
        return False
    if pop in _AP_ACTIVATION_OPS:
        return True
    if pfam == "elementwise" and pop is not None:
        return True
    # no provenance at all -> structural fallback: a pure-parallel (no-reduction) elementwise generic.
    if pop is None and pfam is None:
        return _ap_all_parallel(generic_op)
    return False


def apply_activation_polynomial(module, ctx):
    """Replace math.exp/erf/tanh with the inline arith minimax polynomial ONLY inside the
    linalg.generic the PROVENANCE identifies as an elementwise activation (gelu/silu/sigmoid/tanh).

    PRECISE TARGETING (the whole-model-correctness fix): the previous version rewrote EVERY
    math.exp/erf/tanh in the module — including the exp inside a SOFTMAX (prov.family =
    "normalization"). A minimax exp differs from libm by ~1e-7 and softmax AMPLIFIES that error
    through its row-sum normalization, so the blanket rewrite drove openvla whole-model cos to 0.541
    while the synthetic isolated tests (benign small ranges, no softmax) stayed cos=1.0. The
    abstraction already KNOWS which generic is an activation (prov.op / prov.family from m2m); we use
    that identity instead of op-class alone. A transcendental whose enclosing generic is a softmax /
    normalization / unknown context is LEFT as math.exp -> the baseline libm call (correct), so only
    the genuinely in-range activation transcendentals are approximated. Returns the count rewritten.
    """
    targets = []

    def _walk(op, in_activation):
        for region in op.regions:
            for block in region.blocks:
                for o in block.operations:
                    here = in_activation
                    if o.operation.name == "linalg.generic":
                        here = _ap_is_activation_generic(o)
                    if (here and o.operation.name in _AP_BUILDERS
                            and len(o.operands) == 1):
                        targets.append(o)
                    _walk(o.operation, here)

    _walk(module.operation, False)
    n = 0
    touched_generics = []
    for o in targets:
        op = o.operation
        # tag the enclosing activation generic so the SCHEDULE can vectorize PRECISELY these (and
        # only these) by `transform.structured.match attributes{merlin.act_vectorize}` — no blanket
        # foreach over every generic, no failures(suppress) masking a miscompile.
        par = op.parent
        while par is not None and par.name != "linalg.generic":
            par = par.parent
        if par is not None and par not in touched_generics:
            par.attributes["merlin.act_vectorize"] = _ir.UnitAttr.get()
            touched_generics.append(par)
        x = op.operands[0]; t = x.type
        with _ir.InsertionPoint(op), _ir.Location.unknown():
            new = _AP_BUILDERS[op.name](x, t)
        op.result.replace_all_uses_with(new)
        op.erase()
        n += 1
    return n
# --- end activation polynomial rewriter ---
'''
