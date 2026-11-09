"""Explicit runtime-scale and site-agreement numerical recipe declarations.

Dynamic qparams remain computations when realized; this data construction does
not execute a framework, select placement or issue an accuracy verdict.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .recipe import (
    ACTIVATION_CONTRACTION as ACTIVATION_CONTRACTION,
)
from .recipe import (
    CANDIDATE_SCHEMA as CANDIDATE_SCHEMA,
)
from .recipe import (
    CONTRACTION as CONTRACTION,
)
from .recipe import (
    DERIVED as DERIVED,
)
from .recipe import (
    DYNAMIC as DYNAMIC,
)
from .recipe import (
    DYNAMIC_SCHEMA as DYNAMIC_SCHEMA,
)
from .recipe import (
    NON_REDUCTION as NON_REDUCTION,
)
from .recipe import (
    OPERAND as OPERAND,
)
from .recipe import (
    OPERAND_SUM as OPERAND_SUM,
)
from .recipe import (
    RESIDUAL_OPERAND as RESIDUAL_OPERAND,
)
from .recipe import (
    SCHEMA as SCHEMA,
)
from .recipe import (
    SITE_AGREEMENT as SITE_AGREEMENT,
)
from .recipe import (
    SITE_AGREEMENT_SCHEMA as SITE_AGREEMENT_SCHEMA,
)
from .recipe import (
    SOFTMAX_AXIS as SOFTMAX_AXIS,
)
from .recipe import (
    SOFTMAX_OUTPUT as SOFTMAX_OUTPUT,
)
from .recipe import (
    SOFTMAX_PARTNER as SOFTMAX_PARTNER,
)
from .recipe import (
    STATIC_HEADS as STATIC_HEADS,
)
from .recipe import (
    STATIC_PLANES as STATIC_PLANES,
)
from .recipe import (
    STATIC_PLANES_SCHEMA as STATIC_PLANES_SCHEMA,
)
from .recipe import (
    UNDERIVABLE as UNDERIVABLE,
)
from .recipe import (
    WEIGHT as WEIGHT,
)
from .recipe import (
    WINDOW_MEAN as WINDOW_MEAN,
)
from .recipe import (
    digest as digest,
)

_MAX_PLANES = 4


def with_dynamic_contractions(
    recipe: Mapping[str, Any],
    *,
    families: Sequence[str] = (ACTIVATION_CONTRACTION,),
    planes: Mapping[str, int] | None = None,
    smooth: bool = False,
    static_families: Sequence[str] | None = None,
    scales: str = NON_REDUCTION,
    context_output: str | None = None,
) -> dict[str, Any]:
    """``recipe`` with a ``dynamic`` block: these families' contractions take run-time scales.

    Three general mechanisms, each named by the block and none by a model:

    * dynamic scales on non-reduction axes (:data:`NON_REDUCTION`): the host takes each row's absmax
      at run time, quantizes, the device runs a plain int8 matmul, and the host rescales the i32
      result by the outer product of the row and column scales. The readout holds one scale per
      tensor, so these scales are host work by construction;
    * planes: an operand selected by ``planes`` (:data:`SOFTMAX_OUTPUT`, :data:`WEIGHT`) is split into
      k int8 planes, each the quantized residual of the ones before, with its own row scales: k
      device matmuls against the same other operand, summed on the host;
    * the smooth shift (:data:`SOFTMAX_AXIS`), applied only where it preserves the function.

    ``families`` must be contraction families of this recipe (``activation_contraction`` is added
    when asked for); ``static_families`` (default: the recipe's own) are the families still
    quantized with the recipe's static spec. Families listed in both take the dynamic form.

    ``scales=STATIC_HEADS`` keeps the same mechanisms with scales fixed at capture: one calibrated
    minmax scale per head of each operand, the smooth shift a calibrated per-(head, row) mean (folded
    into the producing Linear's bias where only layout steps lie between), and a softmax-produced
    operand replaced by ``exp(s - rowmax)`` in planes with fixed scales ``1 / (q * (q + 1)**j)``, the
    rows divided by their sum after the contraction (``_static_attention``). ``planes[OPERAND] = k``
    puts every other operand on a static per-head grid of k int8 digits (two digits: 15 bits), each
    pair of digit planes one device call with one scale; ``planes[SOFTMAX_PARTNER] = k`` selects bounded
    round-and-clamp residual planes for the operand a softmax output multiplies, each successive grid
    ``quant_max + 1`` times finer. ``planes[RESIDUAL_OPERAND] = k`` uses those bounded residuals for
    every ordinary operand, including both score operands; it replaces the historical ``OPERAND``
    selector, whose radix arithmetic stays unchanged. Activation contractions of a static recipe only.

    ``context_output="int8_value_grid"`` explicitly selects single-plane normalized
    probabilities on a 1/127 grid and requantizes each P×V product onto its value
    operand's grid before joining heads. Q×K still leaves as INT32. This changes
    the numerical recipe; it is not equivalent to the exp-plane formulation.
    """
    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED:
        raise ValueError("only a derived quant_recipe_v1 can be extended")
    activation = recipe.get("activation") or {}
    if not (activation.get("symmetric") and str(activation.get("dtype", "")).startswith("int")):
        raise ValueError("dynamic contractions need a symmetric integer activation format")
    wanted = list(dict.fromkeys(families))
    if not wanted or any(f not in (CONTRACTION, ACTIVATION_CONTRACTION) for f in wanted):
        raise ValueError(f"dynamic scales apply to contraction families only, not {wanted}")
    selected = dict(planes or {})
    if any(
        k not in (SOFTMAX_OUTPUT, WEIGHT, OPERAND, SOFTMAX_PARTNER, RESIDUAL_OPERAND)
        or type(v) is not int
        or not 1 <= v <= _MAX_PLANES
        for k, v in selected.items()
    ):
        raise ValueError(
            f"planes take {SOFTMAX_OUTPUT!r}/{WEIGHT!r}/{OPERAND!r}/{SOFTMAX_PARTNER!r}/{RESIDUAL_OPERAND!r} "
            f"-> 1..{_MAX_PLANES}, not {selected}"
        )
    if OPERAND in selected and RESIDUAL_OPERAND in selected:
        raise ValueError("operand and residual_operand selectors are mutually exclusive")
    for digits in (OPERAND, SOFTMAX_PARTNER, RESIDUAL_OPERAND):
        if digits in selected and (scales != STATIC_HEADS or selected[digits] > 3):
            raise ValueError(f"{digits!r} digit planes (at most 3) belong to static per-head scales")
    if scales not in (NON_REDUCTION, STATIC_HEADS):
        raise ValueError(f"scales are {NON_REDUCTION!r} or {STATIC_HEADS!r}, not {scales!r}")
    if scales == STATIC_HEADS and (
        wanted != [ACTIVATION_CONTRACTION] or WEIGHT in selected or activation.get("mode") != "static"
    ):
        raise ValueError("static per-head scales cover activation contractions of a static recipe only")
    if context_output is not None and (
        context_output != "int8_value_grid"
        or scales != STATIC_HEADS
        or activation.get("dtype") != "int8"
        or (activation.get("quant_min"), activation.get("quant_max")) != (-128, 127)
        or any(value != 1 for value in selected.values())
    ):
        raise ValueError("int8_value_grid context readout requires static scales and single-plane operands")
    kept = list(recipe.get("families") or ()) if static_families is None else list(static_families)
    unknown = [f for f in kept if f not in (recipe.get("families") or ())]
    if unknown:
        raise ValueError(f"static families {unknown} are not the recipe's own")
    body = json.loads(json.dumps(dict(recipe)))
    body["families"] = list(dict.fromkeys([*kept, *wanted]))
    for family in wanted:
        (body.get("unquantized") or {}).pop(family, None)
    body[DYNAMIC] = {
        "schema": DYNAMIC_SCHEMA,
        "families": wanted,
        "scales": scales,
        "planes": dict(sorted(selected.items())),
        "smooth": SOFTMAX_AXIS if smooth else None,
    }
    body.setdefault("why", {}).pop("dynamic.context_output", None)
    if context_output is not None:
        body[DYNAMIC]["context_output"] = context_output
        body.setdefault("why", {})["dynamic.context_output"] = (
            "explicit numerical variant: quantize normalized probabilities on the 1/127 grid, "
            "perform an integer contraction and requantize each matrix result onto its value operand grid; "
            "no post-contraction row-sum division and no change to score readouts"
        )
    body.setdefault("why", {})[DYNAMIC] = (
        "opted in: run-time scales on non-reduction axes (rows of the streamed operand, columns of the "
        "other, a stored weight's output channels), applied by the host to the exact i32 accumulator "
        "as an outer product; the readout's one scale cannot hold them"
        if scales == NON_REDUCTION
        else "opted in: one calibrated scale per head of each operand, so every per-head device call reads "
        "one scale; softmax probabilities as fixed-scale planes of exp(s - rowmax), rows divided by their sum "
        "on the host"
    )
    if context_output is not None:
        body["why"][DYNAMIC] = body["why"]["dynamic.context_output"]
    body["recipe_sha256"] = digest(body)
    return body


def site_agreement_of(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """The validated per-site bar of ``recipe``'s dynamic block, or None when it declares none."""
    import math

    block = recipe.get(DYNAMIC) if isinstance(recipe, Mapping) else None
    bar = block.get(SITE_AGREEMENT) if isinstance(block, Mapping) else None
    if bar is None:
        return None
    if (
        not isinstance(bar, Mapping)
        or set(bar) != {"schema", "cosine_min", "rel_rmse_max"}
        or bar.get("schema") != SITE_AGREEMENT_SCHEMA
        or any(type(bar.get(key)) is not float or not math.isfinite(bar[key]) for key in ("cosine_min", "rel_rmse_max"))
        or not 0.0 < bar["cosine_min"] <= 1.0
        or not bar["rel_rmse_max"] > 0.0
    ):
        raise ValueError(f"the dynamic block's {SITE_AGREEMENT} is not an exact {SITE_AGREEMENT_SCHEMA} bar")
    return dict(bar)


def with_site_agreement(recipe: Mapping[str, Any], *, cosine_min: float, rel_rmse_max: float, basis: str) -> dict:
    """``recipe`` whose dynamic contraction sites are each held to a declared agreement bar.

    The bar is the recipe's own: a capture records each site's measured agreement with the float
    contraction of its operands, and the sealed issuer refuses a site below it. ``basis`` says where the
    numbers came from (e.g. a measured per-site distribution and its margin); it is recorded, not judged.
    """
    if recipe.get("schema") != SCHEMA or recipe.get("status") != DERIVED or not isinstance(recipe.get(DYNAMIC), dict):
        raise ValueError("a site agreement bar extends a derived recipe that has a dynamic block")
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("a site agreement bar records its basis")
    body = json.loads(json.dumps(dict(recipe)))
    body[DYNAMIC][SITE_AGREEMENT] = {
        "schema": SITE_AGREEMENT_SCHEMA,
        "cosine_min": float(cosine_min),
        "rel_rmse_max": float(rel_rmse_max),
    }
    site_agreement_of(body)
    body.setdefault("why", {})[f"{DYNAMIC}.{SITE_AGREEMENT}"] = basis.strip()
    body["recipe_sha256"] = digest(body)
    return body
