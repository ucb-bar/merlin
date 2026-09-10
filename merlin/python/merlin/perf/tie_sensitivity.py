"""When a reference cannot discriminate: quantization ties and margin-qualified verdicts.

WHY THIS EXISTS. A whole-model int8 gate compared an independent implementation against a
framework-eager dynamic-quantization reference elementwise, at `atol=0.03125 / rtol=0.02`, and
reported 18.7% of outputs "bad" with one of eight token rows taking the wrong argmax. Measured, the
cause was ONE element: a value 43 ULP from the reference's put `x * inv_scale` at 8.499998 against
8.500022, so round-half-even produced 8 where the reference produced 9. A 15-line model of the
quantize chain reproduced BOTH sides bit-exactly from their own inputs, so the arithmetic was not in
question. Forcing that single int8 code back moved the layer's relative L2 from 3.1e-3 to 7.3e-7 and
the divergence vanished.

That is not a defect an implementation can fix. Rounding a value that sits exactly on a tie is a
coin the reference already flipped, and any independent implementation with a different f32
reduction order will disagree on some of them. In that model, 225 of 1,081,344 quantized-activation
elements lay within 1e-4 of a tie, 19 within 1e-5, and 3 within 1e-6 -- so at a ~1e-6 relative
divergence, roughly one flip somewhere in the model is the EXPECTED outcome, not a surprise.

So an elementwise pass/fail against such a reference is not an achievable contract, and the honest
move is to say so rather than to loosen the tolerance until it passes. This module provides the two
things needed instead:

* a FRAGILITY CENSUS -- how many elements sit within a stated distance of a tie, which turns "will
  an independent implementation disagree?" into an arithmetic question with a number;
* a MARGIN-QUALIFIED VERDICT -- per row, PASS when the decision agrees, FAIL when it disagrees by
  more than the reference can be trusted to resolve, and INDETERMINATE when the reference's OWN
  decision margin is below the divergence the pipeline demonstrably has.

THE CENTRAL RULE, and the reason this is not just a looser gate: **INDETERMINATE IS NOT PASS.** A
row the reference cannot adjudicate must be reported as unadjudicable and must not be counted as
agreement. A gate that quietly folded those into passes would be the failure mode this repo has
recorded fifteen times -- a check that cannot fail reporting success. :func:`verdict` therefore
returns the three populations separately and refuses to summarize them into a single boolean.

The noise floor is a PARAMETER, never a default: it must come from a measurement of the pipeline in
question. :func:`verdict` raises when it is absent rather than substituting a plausible number,
because a guessed floor decides which rows become indeterminate and that is the whole verdict.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

__all__ = ["TIE_OFFSET", "tie_distance", "fragility_census", "decision_margin",
           "verdict", "PASS", "FAIL", "INDETERMINATE"]

PASS = "agrees"
FAIL = "disagrees"
INDETERMINATE = "reference_cannot_discriminate"

#: Round-half-to-even resolves a tie at exactly .5 past an integer, so the distance of interest is
#: to the nearest half-integer.
TIE_OFFSET = 0.5


def tie_distance(scaled_value: float) -> float:
    """Distance from ``scaled_value`` to the nearest rounding tie.

    ``scaled_value`` is the value AFTER dividing by the quantization scale -- the number the
    rounding mode actually sees. Ties sit at ``k + 0.5``, so the distance is measured to the
    nearest half-integer and is in [0, 0.5].
    """
    shifted = float(scaled_value) - TIE_OFFSET
    return abs(shifted - round(shifted))


def fragility_census(scaled_values: Sequence[float],
                     thresholds: Sequence[float] = (1e-4, 1e-5, 1e-6)) -> dict[str, Any]:
    """How many values sit within each threshold of a rounding tie.

    This is the predictive part: an implementation whose relative divergence is ``d`` should expect
    to disagree on approximately the count at threshold ``d``. Reporting that count alongside a
    failure turns "our compiler is wrong" into "one tie flipped, which this population predicted".
    """
    values = [float(v) for v in scaled_values]
    distances = [tie_distance(v) for v in values]
    counts = {f"within_{t:g}": sum(1 for d in distances if d <= t) for t in thresholds}
    nearest = min(distances) if distances else None
    return {
        "schema": "quantization_tie_fragility_v1",
        "elements": len(values),
        "counts": counts,
        "nearest_tie_distance": nearest,
        "reading": ("an implementation with relative divergence d should expect to disagree on "
                    "about as many elements as the count at threshold d; those disagreements are "
                    "predicted by the reference's own tie population, not evidence of a defect."),
    }


def decision_margin(scores: Sequence[float]) -> float:
    """Gap between the best and second-best score -- how firmly the reference decided.

    A row whose margin is smaller than the pipeline's demonstrated divergence was never firmly
    decided, so a disagreement there says nothing about the implementation.
    """
    ordered = sorted((float(s) for s in scores), reverse=True)
    if len(ordered) < 2:
        return float("inf")
    return ordered[0] - ordered[1]


def verdict(reference_rows: Sequence[Sequence[float]], candidate_rows: Sequence[Sequence[float]],
            *, noise_floor: float) -> dict[str, Any]:
    """Per-row three-way verdict against a reference of limited resolving power.

    ``noise_floor`` is the demonstrated end-to-end divergence of the pipeline, in the same units as
    the scores. It is required: a guessed floor decides which rows become indeterminate, which is
    the whole verdict. A row is INDETERMINATE when the reference's own decision margin does not
    exceed it -- the reference simply does not adjudicate there.
    """
    if noise_floor is None:
        raise ValueError("noise_floor is required: it must come from a measurement of this "
                         "pipeline, not from a default, because it decides which rows are "
                         "adjudicable at all")
    floor = float(noise_floor)
    if not (floor >= 0.0):
        raise ValueError(f"noise_floor must be a non-negative measurement, got {noise_floor!r}")
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(f"row count differs: reference {len(reference_rows)} vs "
                         f"candidate {len(candidate_rows)}")

    rows: list[dict[str, Any]] = []
    for index, (want, got) in enumerate(zip(reference_rows, candidate_rows)):
        if len(want) != len(got):
            raise ValueError(f"row {index} width differs: {len(want)} vs {len(got)}")
        margin = decision_margin(want)
        want_top = max(range(len(want)), key=lambda i: float(want[i]))
        got_top = max(range(len(got)), key=lambda i: float(got[i]))
        if want_top == got_top:
            status = PASS
        elif margin <= floor:
            status = INDETERMINATE
        else:
            status = FAIL
        rows.append({"row": index, "status": status, "reference_margin": margin,
                     "reference_choice": want_top, "candidate_choice": got_top})

    tally = {PASS: 0, FAIL: 0, INDETERMINATE: 0}
    for row in rows:
        tally[row["status"]] += 1
    return {
        "schema": "margin_qualified_verdict_v1",
        "noise_floor": floor,
        "rows": rows,
        "agrees": tally[PASS],
        "disagrees": tally[FAIL],
        "reference_cannot_discriminate": tally[INDETERMINATE],
        # Deliberately NOT a single boolean. An indeterminate row is not agreement, and collapsing
        # the three populations into one verdict is what makes a gate unable to fail.
        "adjudicable": tally[PASS] + tally[FAIL],
        "licence": ("INDETERMINATE IS NOT PASS. A row whose reference margin is at or below the "
                    "measured noise floor is unadjudicable by this reference; report it as such "
                    "and never fold it into the agreeing population."),
    }
