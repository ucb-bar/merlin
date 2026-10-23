"""A bare integer sum over a trailing window, recognized as a window mean of multiplier one.

An integer-arithmetic model (an I-BERT style layer norm, say) sums a row it has already quantized to a
narrow integer and keeps the raw accumulator: ``quantize -> narrowing cast -> linalg.reduce(addi)``, no
dequantize, never leaving the integer domain. That is a contraction a unit performs -- the row against
a column of ones -- with no readout scale at all, so it is stated as :attr:`Group.window_mean` with
``multiplier`` fixed at ``1.0`` and every consumer of a window mean reads it unchanged. Split out of
:mod:`.compute_groups` (which calls it while it forms host regions) to keep that module under the size
gate. Nothing here names a target.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from merlin.common import mlir_query as mq


def window_sum_of(working: Sequence[Any], stage_of: Mapping[int, Any]) -> tuple[dict | None, str | None]:
    """``(facts, None)`` for a bare integer sum over a trailing window, ``(None, why)`` when it is one
    whose extents cannot be read, else ``(None, None)``.

    The shape is exactly ``quantize -> narrowing cast -> linalg.reduce(addi)``, in that order (the
    quantize in this region or the one before it, since formation closes a region at a quantize): a real
    quantize marker states the row is a narrow integer at capture time, and ITS RESULT carries that
    dtype -- the reduce's own operand is already widened, so reading the dtype there would ask a unit
    about the accumulator's format instead of the operand's. A mask population count (a cast with no
    quantize) is not this form; neither is a floor-based softmax denominator.
    """
    from . import compute_groups as CG

    reduces = [m for m in working if mq.op_name(m) == "linalg.reduce"]
    if len(working) not in (2, 3) or len(reduces) != 1:
        return None, None
    reduce = reduces[0]
    others = [m for m in working if m is not reduce]
    body = [n for n in CG._body_ops(reduce) if n not in CG._BODY_NOISE]
    casts = [m for m in others if stage_of[id(m)].kind == CG.CAST]
    if body != ["arith.addi"] or len(casts) != 1 or getattr(reduce.operands[0], "owner", None) is not casts[0]:
        return None, None
    cast = casts[0]
    # THE QUANTIZE MAY CLOSE THE REGION BEFORE THIS ONE: group formation never joins across a
    # quantize, so the marker is either this region's third member or the producer of the cast's
    # operand. Either way it is the op that states the row's narrow format, and nothing else is.
    quantize = getattr(cast.operands[0], "owner", None)
    inside = [m for m in others if m is not cast]
    if inside and (len(inside) != 1 or inside[0] is not quantize):
        return None, None
    if not hasattr(quantize, "results") or CG.classify(quantize) is None or CG.classify(quantize).kind != CG.QUANTIZE:
        return None, None
    in_shape, acc_dtype = mq.type_shape_dtype(reduce.operands[0].type)
    _, narrow_dtype = mq.type_shape_dtype(quantize.results[0].type)
    out_shape, _ = mq.type_shape_dtype(reduce.results[0].type)
    reduced = CG._int_array(reduce, "dimensions")
    if not in_shape or reduced is None or any(not isinstance(e, int) or e <= 0 for e in in_shape):
        return None, "the reduction's extents or dimensions are not static"
    kept = len(in_shape) - len(reduced)
    if sorted(reduced) != list(range(kept, len(in_shape))):
        return None, f"the sum reduces dims {sorted(reduced)} of a rank-{len(in_shape)} tensor; not trailing"
    if any(d is None or not str(d).startswith("i") for d in (acc_dtype, narrow_dtype)):
        return None, "the summed operand or the quantize's own result is not a narrow integer"
    return {
        "reduce": reduce,
        "rows": CG._product(in_shape[:kept]),
        "window": CG._product(in_shape[kept:]),
        "multiplier": 1.0,
        "bound_lsb": 0,  # an integer accumulate of an already-narrowed row is exact
        "exactness": "exact",
        "out_elements": CG._product(out_shape),
        "in_dtype": narrow_dtype,
    }, None


def window_mean_refusal(mean: dict[str, Any], oracle: Any, dtype: str | None) -> str | None:
    """Why no unit takes the mean as a contraction, or ``None`` with ``mean['units']`` set.

    Two questions, both the ones a fake-quantized layer is asked: can a unit contract these
    integers, and does its readout hold the one scale the mean needs. ``mean['asked']`` records
    whether a contracting unit exists at all, so a target with none plans the region as before.
    """
    from .compute_groups import CONTRACTION, FAMILY_OF_STAGE, QUANTIZE

    contraction = oracle.ask(
        op=CONTRACTION, family="contraction", in_dtype=dtype, weight_dtype=dtype, rank=2, attached=False
    )
    if not contraction.admitted:
        mean["asked"], mean["refusal"] = False, contraction.refusal
        return contraction.reason or "no unit contracts these operands"
    mean["asked"] = True
    if mean.get("exactness") != "exact" or mean.get("multiplier") != 1:
        mean["refusal"] = "exact_arithmetic"
        return "floating mean reassociation requires full int32 readout and an exact rational host quantization binding"
    from merlin.common.quant_formats import get
    from merlin.runtime.exact_readout import prove_fp32_products

    try:
        fmt = get(dtype)
        bits = fmt.element_bits
        low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if fmt.signed else (0, 2**bits - 1)
        mean["product_proof"] = prove_fp32_products(1, low * mean["window"], high * mean["window"])
    except (KeyError, ValueError) as exc:
        mean["refusal"] = "exact_arithmetic"
        return str(exc)
    scaled = oracle.ask(
        op=QUANTIZE, family=FAMILY_OF_STAGE[QUANTIZE], in_dtype=dtype, attached=True, granularity="tensor"
    )
    if not scaled.admitted:
        mean["refusal"] = scaled.refusal
        return scaled.reason or "the readout does not hold the scale a mean needs"
    readout = getattr(oracle, "readout", None)
    for facet in getattr(readout, "facets", ()) or ():
        fits = getattr(facet, "sum_fits_accumulator", None)
        if callable(fits) and fits(mean["window"]) is False:
            mean["refusal"] = "scale_granularity"
            return (
                f"a window of {mean['window']} elements of {facet.element_dtype} can exceed the "
                f"{facet.accumulator_dtype} accumulator"
            )
    mean["units"] = contraction.units
    return None
