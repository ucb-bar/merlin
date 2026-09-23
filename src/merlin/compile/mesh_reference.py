"""The host-side reference a mesh tile is checked against.

``_reference_on_datapath`` computes a tile the way the declared accumulator does, and
``_accum_rel_tolerance`` bounds how far a float accumulator may legitimately drift from it.
"""

from __future__ import annotations


def _accum_rel_tolerance(accum_dtype: str, k: int) -> float | None:
    """Relative tolerance for a K-deep accumulation in ``accum_dtype``, DERIVED from that format's mantissa
    width in the quant-format registry rather than picked. A bf16 accumulator carries 7 mantissa bits, so a
    352-deep reduction CANNOT be bit-exact against an f32 reference and a bit-exact gate there would report
    a correct mesh as broken. An integer accumulator does not round: it gets 0.0, i.e. bit-exact.

    The bound is the format's unit roundoff ``2**-(mant_bits+1)`` grown by ``sqrt(k)`` — the random-walk
    growth of a k-deep sequential sum — with a small safety factor. CALIBRATED, not guessed: a 32x352x128
    fp8xbf16 layer measured 6.7% max relative error on the large elements against an f32 reference, versus
    the 7.3% this predicts before the safety factor. An earlier version used ``2**-mant_bits`` and a factor
    of 8, which returned 117% for that same layer — a gate that wide accepts anything.

    Returns ``None`` when the format cannot be resolved — the caller must fail closed rather than pick a
    tolerance, because both defaults are wrong in one direction (too tight condemns a good mesh, too loose
    passes a broken one)."""
    from ..common import quant_formats as QF

    try:
        f = QF.get(accum_dtype)
    except KeyError:
        # Not a registry format. An MLIR integer spelling (iN) is still unambiguous: integer accumulation
        # is exact, so it gates bit-exact. Anything else is genuinely unresolved.
        body = accum_dtype[1:] if accum_dtype[:1] == "i" else ""
        return 0.0 if body.isdigit() else None
    if f.kind == "int_affine":
        return 0.0
    mant = int(f.mant_bits or 0)
    if not mant:
        return None
    return (2.0 ** -(mant + 1)) * max(1.0, float(k) ** 0.5) * 1.5


def _reference_on_datapath(A, W, binding):
    """``A @ W`` as the TARGET's datapath computes it — operands decoded the way its compute unit reads
    them, then accumulated in its declared accumulator format. The reference a mesh should be gated
    against, rather than an f32 one it was never going to reproduce.

    Takes the whole ``CorpusBinding`` rather than the accumulator dtype alone, and that is deliberate.
    Modelled on the accumulator only, this function was right about the half of the datapath it had been
    handed and silently wrong about the other half: atlas's MXU sees a signed zero wherever an operand's
    exponent field is zero (``E4M3Mul.scala``: ``aZero := aExp === 0.U``, declared as
    ``subnormal_operand_flush`` in the target's profile), and a reference that reads the operand at full
    precision grades that hardware as broken. One object carries the whole datapath, so a field added to
    it later reaches this model without a signature change.

    Two halves, both measured. A narrow-float accumulator rounds every partial sum, so an f32 reference
    disagrees with a perfectly correct device by design: a 32x352x128 fp8xbf16 layer measured 796 of 4096
    elements differing, max absolute error 22, purely from bf16 rounding — and rounding the running sum
    after each MAC reproduced that device output BIT-FOR-BIT on all 4096. On the operand side, one atlas
    capsule's 30 divergent elements were exactly its 30 subnormal codes.

    LIMIT, stated rather than implied: operands are flushed but NOT otherwise rounded into the operand
    format. Callers feed exactly-representable values (that is what makes a bit-exact gate possible), so
    rounding would be a no-op here; modelling it would mean writing a round-to-nearest-even encoder whose
    own correctness nothing checks, and a reference is only worth what its weakest step is worth.

    Returns ``None`` when the accumulator format cannot be resolved, or for an integer accumulator (exact:
    the plain product is already the right reference). Assumes sequential k-order accumulation, which is
    why the caller treats a mismatch as "not bit-exact" and falls back to the tolerance gate rather than a
    failure -- another device may reduce in a different order and still be correct."""
    import numpy as np

    from ..common import quant_formats as QF

    try:
        f = QF.get(binding.accum_dtype)
    except KeyError:
        return None
    if getattr(binding, "subnormal_operand_flush", False):
        from ..runtime import fp8_formats as FF

        try:
            min_normal, _max_finite = FF.normal_range(binding.operand_dtype)
        except KeyError:  # operand format unresolvable: fail closed
            return None
        A = np.where(np.abs(A) < min_normal, np.float32(0.0), A).astype(np.float32)
        W = np.where(np.abs(W) < min_normal, np.float32(0.0), W).astype(np.float32)
    if f.kind == "int_affine":
        return None
    mant, exp = int(f.mant_bits or 0), int(f.exp_bits or 0)
    if mant == 10 and exp == 5:  # IEEE half
        rnd = lambda x: x.astype("<f2").astype(np.float32)  # noqa: E731
    elif mant == 7 and exp == 8:  # bfloat16: top half of the f32 word, RNE

        def rnd(x):
            u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
            return (((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16).astype(np.uint32).view(np.float32)
    elif mant == 23 and exp == 8:  # f32: the product already is the reference
        return None
    else:
        return None
    acc = np.zeros((A.shape[0], W.shape[1]), dtype=np.float32)
    for i in range(A.shape[1]):
        acc = rnd(acc + np.outer(A[:, i], W[i, :]))
    return acc
