"""Fuse a ``roundeven -> clamp -> fptosi`` quantize chain into a rounding-mode-independent
pure-arith convert, so the generic carrying it stops being a scalar libm call site.

WHAT IT IS FOR, and why the size of it is not obvious from the op count.

The int8 activation quantize (``passes_quant_int``, three separate construction sites, all emitting
the identical five-op body) is::

    %v = arith.divf   %x, %s                : f32
    %r = math.roundeven %v                  : f32
    %a = arith.minimumf %r, %hi             : f32
    %b = arith.maximumf %a, %lo             : f32
    %q = arith.fptosi %b                    : f32 to i8

Five ops in the IR. But ``math.roundeven`` has no vector form on this path: ``convert-math-to-libm``
turns it into a CALL to ``roundevenf``, and the vector variant of that pass scalarizes a
``vector<Nxf32>`` back into N extracts + N calls + N inserts. So the op is a scalar libm call per
ELEMENT, and everything around it is scalar too because nothing will vectorize a loop containing a
call.

Two consequences, and the second is the one that makes this a lever rather than a cleanup:

1. The chain is expensive. `roundevenf` on this toolchain is a nine-instruction body whose payload is
   `fcvt.w.s` (round-to-nearest-even) followed by `fcvt.s.w` back to float, wrapped in a
   `|x| < 2^23` guard -- a guard that is DEAD here, because the very next ops clamp the result into a
   range where it cannot fire.
2. The chain is a VECTORIZATION BARRIER, and that barrier is enforced twice over. LLVM's loop
   vectorizer declines a loop containing the call; and, upstream of it, the ``merlin.vec_r{rank}``
   TAGGER (`runtime/backends/zephyr_model._prepare_model_mlir`) refuses outright any all-parallel
   generic with a ``math.*`` op in its body -- a refusal that is CORRECT, and that this pass is the
   way past. Its comment says so: those ops "are the OTHER lever's job". The activation lever
   (`vectorized_transcendental_activation`) is the other lever for `math.exp`/`erf`/`tanh`; this is
   the one for the quantize chain, and it composes the same way -- rewrite to pure arith BEFORE the
   tagger runs, and the existing per-rank arms claim the op with no new arm written.

That is why this feature ``implies`` `vectorize_non_contraction_generics`. On its own the rewrite is
roughly instruction-NEUTRAL in scalar form (it trades a call and its callee for a similar count of
inline arith), so naming it alone would measure a wash; the payoff is that the op becomes
vectorizable, and a default-off lever whose payoff needs a second default-off lever is an inert lever
(the reasoning `impr_features.ImprFeature.implies` and `_vec_noncontraction_hygiene` already record).

THE EQUIVALENCE, which is the whole correctness argument and is exact, not approximate.

Write `re` for IEEE roundToIntegralTiesToEven (what ``math.roundeven`` computes -- a specified
operation, NOT affected by the dynamic rounding mode), `C` for the clamp the chain applies, and
`tz` for truncate-toward-zero (what ``arith.fptosi`` computes). The baseline computes `tz(C(re(v)))`.

  LEMMA 1 (the clamp commutes with the rounding). If every clamp bound is an INTEGRAL float, then
  `C(re(v)) = re(C(v))`. `re` is monotone non-decreasing and fixes every integer, so for an upper
  bound `hi`: if `v <= hi` then `re(v) <= re(hi) = hi` and both sides are `re(v)`; if `v > hi` then
  `min(v,hi) = hi` gives `re(hi) = hi` on the right, and on the left `re(v) >= hi` so `min(re(v),hi)
  = hi` as well. Symmetrically for a lower bound. Chains of such bounds compose. This is why the
  pass REFUSES a non-integral bound rather than rounding it: the commutation is the only thing that
  lets the clamp move to where it can bound the argument.

  LEMMA 2 (the truncation is a no-op). `re(u)` is integer-valued, and after Lemma 1 it lies in
  `[lo, hi]`, which the pass has already checked is inside the destination integer type. So
  `tz(re(C(v))) = re(C(v))` exactly -- no truncation happens, and `fptosi`'s round-toward-zero never
  differs from the round-half-to-even that produced its input. This is the step that makes the two
  DIFFERENT rounding rules in the chain collapse into one.

  So the whole chain is exactly: clamp into `[lo, hi]`, then round half-to-even. Checked at every
  boundary the composition has:
    * `v` NaN: `arith.minimumf` is IEEE `minimum` and PROPAGATES NaN, so the baseline reaches
      `fptosi(NaN)`, which is POISON in LLVM. The rewrite reaches `fptosi(NaN)` too and is poison in
      the same place. Neither form is defined here and the pass does not pretend to fix it; it is
      recorded because the input can occur (an all-zero activation tensor gives `amax = 0`, `s = 0`,
      `0/0 = NaN`) and because it is the one input class where "bit-identical" is a statement about
      two undefined values rather than two defined ones.
    * `v = +/-inf`: `re` fixes them, the clamp pins them to `hi`/`lo`. The rewrite clamps first and
      gets `hi`/`lo` directly. Same.
    * `|v| >= 2^23`: `v` is already integral, so `re(v) = v` and the clamp pins it to `hi`/`lo`. The
      rewrite clamps first, so it never evaluates the round on a value this large at all -- which is
      exactly what makes the inline round below safe.
    * ties: `v = 126.5 -> 126`, `127.5 -> 127` (rounds to 128, clamps back), `-127.5 -> -127`,
      `-2.5 -> -2`, `2.5 -> 2`, `1.5 -> 2`, `-0.5 -> -0.0 -> 0`. The emitted form reproduces each.

HOW THE ROUND IS EMITTED, and why not the cheaper spellings.

After the clamp, `|c| <= max(|lo|,|hi|)`, which the pass has checked is under 2^23. In that range
round-half-to-even is exactly expressible in ordinary arith, with NO dependence on the dynamic
rounding mode::

    t  = fptosi(c)                # truncate toward zero; exact, |c| < 2^23
    ft = sitofp(t)                # exact
    d  = c - ft                   # EXACT (t is a multiple of ulp(c) when |c| < 2^23), |d| < 1
    bump = |d| > 1/2  or  (|d| == 1/2 and t is odd)
    q  = t + (bump ? sign(c) : 0)

`t` and `q` are computed in the DESTINATION integer type, not in a wider one: `q = re(c)` and
`c` is in `[lo, hi]`, so by Lemma 1's monotonicity `q` is in `[lo, hi]` too, a range the pass has
already required to fit that type -- so neither the convert nor the `+/-1` can overflow, and no
truncation back is needed.

REJECTED, deliberately, and both rejections are about the ROUNDING MODE:

  * the add/sub magic constant (`x + 1.5*2^23 - 1.5*2^23`, which `act_poly._ap_roundeven` uses for the
    exp range reduction). It is two ops instead of fourteen, and it is CORRECT only while the dynamic
    rounding mode is round-to-nearest-even, because it works by asking the FP adder to do the
    rounding. The baseline chain it replaces is rounding-mode INDEPENDENT, so using it here would
    introduce a dependence on `frm` that the code being replaced does not have -- on a target whose
    `frm` this repo neither sets nor derives, and in a pass that is supposed to be target-agnostic.
    (It is also wrong on `[2^22, 2^23)` with that constant, though the clamp hides that.)
  * emitting a float->int convert that names round-to-nearest-even directly. There is no such op in
    `arith` (`fptosi` is truncate, by definition) and reaching for `llvm.intr.*` inside a
    `linalg.generic` body would put an op there that the linalg vectorizer cannot vectorize, which
    defeats the entire purpose. Setting `frm` around the region and using the dynamic mode was
    considered and rejected for the same reason as the magic constant, plus a second one: `frm` is
    process-global state that every other float op in the emitted code reads, so setting it would
    change the rounding of code this pass never looked at.

The sequence below is therefore longer than either, and it is the only one of the three whose result
does not depend on a mode nobody in this repo derives.

TARGET-AGNOSTIC: the matcher names no target, no model, no shape, no dtype and no constant. The clamp
bounds, the destination integer width and the float type are all READ OFF THE IR being rewritten; the
only literals are 1/2 and 1, which are properties of round-half-to-even itself. Anything it cannot
establish -- a non-integral bound, a one-sided clamp, a bound outside the destination type, a value
used more than once, `fptoui` instead of `fptosi` -- is REFUSED and COUNTED, never approximated.
"""
from __future__ import annotations

#: Feature name. Registered on demand (see :func:`ensure_registered`) rather than at import, for the
#: reason `wholemodel_proposer` documents at length: `_composes` swallows the KeyError an unregistered
#: name raises and returns False, so a lever nobody registered is not declined, it is INVISIBLE.
FEATURE = "fuse_quantize_round_convert"

#: Largest magnitude a clamp bound may have for the inline round to be exact. At or above 2^23 an f32
#: has no fractional bits left, so `c - trunc(c)` stops being exact and `t` stops being representable
#: in a narrow destination type. NOT a tuning number and NOT a target fact -- it is where f32's
#: mantissa ends, and it is derived from the float type of the value being rewritten rather than
#: written down, so an f16 or f64 chain gets its own bound.
def _integral_limit(float_type) -> float:
    """The magnitude at which ``float_type`` runs out of fractional bits (2 ** mantissa bits)."""
    bits = {16: 11, 32: 24, 64: 53}.get(_float_width(float_type))
    if bits is None:
        return 0.0
    return float(2 ** (bits - 1))


def _float_width(t) -> int | None:
    from xdsl.dialects.builtin import Float16Type, Float32Type, Float64Type
    if isinstance(t, Float16Type):
        return 16
    if isinstance(t, Float32Type):
        return 32
    if isinstance(t, Float64Type):
        return 64
    return None


def _const_float(value) -> float | None:
    """The f32/f64 constant ``value`` holds, or None if it is not a float `arith.constant`."""
    from xdsl.dialects.builtin import FloatAttr
    owner = getattr(value, "owner", None)
    if owner is None or getattr(owner, "name", None) != "arith.constant":
        return None
    attr = owner.properties.get("value", owner.attributes.get("value"))
    if not isinstance(attr, FloatAttr):
        return None
    return float(attr.value.data)


def _single_use(value) -> bool:
    """``value`` feeds exactly one op. Rewriting a chain whose intermediate is read elsewhere would
    leave the original `math.roundeven` alive AND add the inline round, which is strictly worse than
    doing nothing -- so a shared intermediate is refused rather than duplicated."""
    return len(list(value.uses)) == 1


def _int_range(int_type) -> tuple[int, int]:
    w = int_type.width.data
    return (-(2 ** (w - 1)), 2 ** (w - 1) - 1)


def _match_chain(fptosi_op):
    """``(round_op, clamp_ops, lo, hi)`` for a fusable chain ending at ``fptosi_op``, else a reason.

    Walks BACKWARDS from the convert through a run of `arith.minimumf` / `arith.maximumf` whose other
    operand is an integral float constant, and requires the run to bottom out at `math.roundeven`.
    The interval is accumulated in the order the ops apply, so a chain that never bounds one side is
    caught rather than assumed.

    Returns either a 4-tuple or a string naming the refusal.
    """
    from xdsl.dialects.builtin import IntegerType

    res_t = fptosi_op.results[0].type
    if not isinstance(res_t, IntegerType) or res_t.width.data < 2:
        return "dest_not_multibit_int"
    imin, imax = _int_range(res_t)

    cur = fptosi_op.operands[0]
    ftype = cur.type
    if _float_width(ftype) is None:
        return "src_not_float"
    limit = _integral_limit(ftype)
    if limit <= 0.0:
        return "float_type_unhandled"

    lo, hi = float("-inf"), float("inf")
    clamps: list[tuple[str, float, object]] = []
    # Bounded walk: the chain is a handful of ops, and an unbounded loop over a malformed IR graph
    # would hang the build rather than fail it.
    for _ in range(8):
        owner = getattr(cur, "owner", None)
        name = getattr(owner, "name", None)
        if name == "math.roundeven":
            if not _single_use(cur):
                return "round_result_shared"
            if lo == float("-inf") or hi == float("inf"):
                # A one-sided clamp leaves the rounded value unbounded on the other side, so the
                # inline round has no range to be exact in. Refuse; do not guess a bound.
                return "clamp_not_two_sided"
            if lo > hi:
                return "clamp_empty"
            if not (abs(lo) < limit and abs(hi) < limit):
                return "clamp_bound_above_mantissa"
            if not (imin <= lo <= imax and imin <= hi <= imax):
                return "clamp_bound_outside_dest_int"
            return (owner, clamps, lo, hi)
        if name not in ("arith.minimumf", "arith.maximumf"):
            return "chain_root_not_roundeven"
        if not _single_use(cur):
            return "clamp_result_shared"
        a, b = owner.operands
        cval, nxt = _const_float(b), a
        if cval is None:
            cval, nxt = _const_float(a), b
        if cval is None:
            return "clamp_bound_not_constant"
        if cval != int(cval) if cval == cval and abs(cval) != float("inf") else True:
            # NaN / inf / non-integral bound: Lemma 1 does not hold, so the clamp cannot be moved
            # in front of the rounding. (`cval != cval` is the NaN test; no `math` import needed.)
            return "clamp_bound_not_integral"
        if name == "arith.minimumf":
            hi = min(hi, cval)
        else:
            lo = max(lo, cval)
        clamps.append((name, cval, owner))
        cur = nxt
    return "chain_too_long"


def fuse_round_clamp_convert(module, report_out: "dict | None" = None) -> int:
    """Rewrite every fusable ``roundeven -> clamp -> fptosi`` chain in ``module``; returns the count.

    Reports refusals BY REASON into ``report_out``. That is not decoration: a pass that rewrote
    nothing and a pass that could not reach anything both return 0, and only the counters separate
    them -- which is the failure mode this repo keeps re-learning about default-off levers.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, IntegerType
    from .passes_xdsl import carry_provenance

    report: dict = {} if report_out is None else report_out
    n = 0
    for op in list(module.walk()):
        if getattr(op, "name", None) != "arith.fptosi":
            continue
        matched = _match_chain(op)
        if isinstance(matched, str):
            key = f"refused_{matched}"
            report[key] = report.get(key, 0) + 1
            continue
        round_op, clamps, lo, hi = matched
        it: IntegerType = op.results[0].type
        v = round_op.operands[0]
        ftype = v.type
        block = op.parent_block()
        if block is None:
            report["refused_no_parent_block"] = report.get("refused_no_parent_block", 0) + 1
            continue

        new_ops: list = []

        def emit(new):
            new_ops.append(new)
            return new.results[0]

        # 1. The clamp, replayed on the UNROUNDED value in the original order (Lemma 1). Rebuilt
        #    rather than moved so the original ops can be detached wholesale below.
        cur = v
        for name, bound, _src in reversed(clamps):
            cst = emit(arith.ConstantOp(FloatAttr(bound, ftype)))
            cur = emit(arith.MinimumfOp(cur, cst) if name == "arith.minimumf"
                       else arith.MaximumfOp(cur, cst))
        c = cur
        # 2. Round half-to-even inside [lo, hi], in the destination integer type (Lemma 2).
        t = emit(arith.FPToSIOp(c, it))
        ft = emit(arith.SIToFPOp(t, ftype))
        d = emit(arith.SubfOp(c, ft))
        ad = emit(arith.MaximumfOp(d, emit(arith.NegfOp(d))))
        half = emit(arith.ConstantOp(FloatAttr(0.5, ftype)))
        gt = emit(arith.CmpfOp(ad, half, "ogt"))
        eqh = emit(arith.CmpfOp(ad, half, "oeq"))
        one_i = emit(arith.ConstantOp(IntegerAttr(1, it)))
        zero_i = emit(arith.ConstantOp(IntegerAttr(0, it)))
        odd = emit(arith.CmpiOp(emit(arith.AndIOp(t, one_i)), zero_i, "ne"))
        bump = emit(arith.OrIOp(gt, emit(arith.AndIOp(eqh, odd))))
        zero_f = emit(arith.ConstantOp(FloatAttr(0.0, ftype)))
        neg = emit(arith.CmpfOp(c, zero_f, "olt"))
        step = emit(arith.SelectOp(neg, emit(arith.ConstantOp(IntegerAttr(-1, it))), one_i))
        q = emit(arith.AddiOp(t, emit(arith.SelectOp(bump, step, zero_i))))

        carry_provenance(new_ops[-1], op, "fuse_quantize_round_convert")
        for new in new_ops:
            block.insert_op_before(new, op)
        op.results[0].replace_all_uses_with(q)
        # Detach the old chain from the convert back down to the round. Each op on it was checked
        # single-use, so nothing else can observe them.
        block.detach_op(op)
        for _name, _bound, clamp_op in clamps:
            clamp_op.parent_block().detach_op(clamp_op)
        round_op.parent_block().detach_op(round_op)
        n += 1
    report["rewrites"] = report.get("rewrites", 0) + n
    return n


def ensure_registered() -> str:
    """Register the feature if it is not already. Idempotent; returns :data:`FEATURE`."""
    from . import impr_features as F
    if FEATURE in F.known():
        return FEATURE
    F.register(F.ImprFeature(
        name=FEATURE,
        action_class="PASS",
        description=(
            "Fuse the int8 quantize chain `math.roundeven -> clamp -> arith.fptosi` into a "
            "rounding-mode-INDEPENDENT inline round, so the generic carrying it stops containing a "
            "`math.*` op and becomes claimable by the existing per-rank vectorize arms. The chain is "
            "emitted at three sites in `passes_quant_int` and is a scalar libm CALL per element: "
            "`convert-math-to-libm` turns `math.roundeven` into `roundevenf` and scalarizes the "
            "vector form back into per-lane extracts + calls, so nothing downstream will vectorize "
            "a loop containing it -- and the `merlin.vec_r{rank}` TAGGER refuses any all-parallel "
            "generic with a `math.*` body outright, which is why these ops carry no tag today. Same "
            "composition as `vectorized_transcendental_activation`: rewrite to pure arith BEFORE the "
            "tagger runs and the existing arms claim the op, with no new arm. EXACT, not "
            "approximate: the clamp commutes with round-half-to-even because every bound is integral "
            "(refused otherwise), which bounds the argument; and `fptosi`'s truncate-toward-zero is "
            "then a no-op on an already-integral value inside the destination type -- so the two "
            "different rounding rules in the chain collapse to one. The inline round uses "
            "trunc/convert-back/compare rather than the add-magic-constant trick, because the chain "
            "it replaces is rounding-mode independent and the magic trick is not, on a target whose "
            "`frm` this repo does not derive. REFUSES and COUNTS a non-integral bound, a one-sided "
            "clamp, a bound outside the destination integer type, a shared intermediate and "
            "`fptoui`. IMPLIES `vectorize_non_contraction_generics`: on its own the rewrite is "
            "roughly instruction-neutral in scalar form (a call and its callee traded for inline "
            "arith), so the payoff is entirely that the op becomes vectorizable -- and a default-off "
            "lever whose payoff needs a second default-off lever measures as a wash. Orthogonal to "
            "`quantize_before_gather`, which changes the SCALE (per-tensor vs per-row) and is a "
            "genuine numeric change: this pass does not touch the scale computation at all. "
            "NO SPEED CLAIM -- the wall is unmeasured, and on this repo a lever that removed ops and "
            "shrank the object has measured 1.09x SLOWER. Default-off; baseline byte-identical."),
        implies=frozenset({F.VEC_NONCONTRACTION_NAME}),
    ))
    return FEATURE
