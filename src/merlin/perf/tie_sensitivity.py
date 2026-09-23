"""When a reference cannot discriminate: quantization ties and margin-qualified verdicts.

WHY THIS EXISTS. A whole-model int8 gate compared an independent implementation against a
framework-eager dynamic-quantization reference elementwise, at `atol=0.03125 / rtol=0.02`, and
reported 18.7% of outputs "bad" with one of eight token rows taking the wrong argmax. Measured, the
argmax flip came from ONE activation element: a value 43 ULP from the reference's put
`x * inv_scale` at 8.499998 against 8.500022, so round-half-even produced 8 where the reference
produced 9. A 15-line model of the quantize chain reproduced BOTH sides bit-exactly from their own
inputs, so the arithmetic was not in question.

The counterfactual was then RUN, end to end on the simulator, and it carries a correction worth
stating because it first read as a refutation. Forcing that code in the one quantizer the analysis
named left the model at 7 of 8 rows -- unchanged. The tensor is consumed by TWO quantizers
(`mlp.gate_proj` and `mlp.up_proj` share it and its per-token scale), so the identical tie occurs
twice; forcing both recovered all 8 rows exactly, and made the layer exact again (relative L2
3.1e-3 -> 7.3e-7). A counterfactual over a shared value has to cover every consumer of it.

That is not a defect an implementation can fix. Rounding a value that sits on a tie is a coin the
reference already flipped, and any independent implementation with a different f32 reduction order
will disagree on some of them. In that model, 485 of 2,809,856 quantized activations lay within
1e-4 of a tie, 53 within 1e-5, 14 within 1e-6, and 6 sat exactly on one.

AND THE ELEMENTWISE DIVERGENCE IS NOT THAT TIE. With all eight rows' top-1 recovered, 44,493 of
256,000 logits still exceeded the tolerance and the whole-output relative L2 was still 3.5e-2,
spread evenly across every row including the six the forced element cannot causally reach. So the
tie explains the RANKING failure; the value spread is a population of the same phenomenon that no
single element accounts for and no tolerance recovers.

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

__all__ = [
    "TIE_OFFSET",
    "tie_distance",
    "fragility_census",
    "decision_margin",
    "classify",
    "verdict",
    "verdict_from_choices",
    "adjudicable_rows",
    "spread_envelope",
    "PASS",
    "FAIL",
    "INDETERMINATE",
]

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


def fragility_census(
    scaled_values: Sequence[float], thresholds: Sequence[float] = (1e-4, 1e-5, 1e-6)
) -> dict[str, Any]:
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
        "reading": (
            "an implementation with relative divergence d should expect to disagree on "
            "about as many elements as the count at threshold d; those disagreements are "
            "predicted by the reference's own tie population, not evidence of a defect."
        ),
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


def _check_floor(noise_floor: float) -> float:
    """The floor, validated. Stated ONCE so a second entry point cannot admit a floor this one
    refuses -- the shape bundle_gate.grades_unquantized_activations was written to avoid."""
    if noise_floor is None:
        raise ValueError(
            "noise_floor is required: it must come from a measurement of this "
            "pipeline, not from a default, because it decides which rows are "
            "adjudicable at all"
        )
    floor = float(noise_floor)
    if not (floor >= 0.0):
        raise ValueError(f"noise_floor must be a non-negative measurement, got {noise_floor!r}")
    return floor


def classify(agrees: bool, margin: float, *, noise_floor: float) -> str:
    """THE rule, in one place: agreement, or a real disagreement, or an unadjudicable decision.

    Every granularity this module offers -- a row's argmax, a pair's ordering, a whole output's
    spread -- is the same question asked of a different decision, so it is the same rule. Stating
    it once is not tidiness: a rule stated twice is a rule that can be half-fixed, and the half
    that stays permissive is the one that stops the gate from failing.
    """
    floor = _check_floor(noise_floor)
    if agrees:
        return PASS
    return INDETERMINATE if float(margin) <= floor else FAIL


def adjudicable_rows(reference_rows: Sequence[Sequence[float]], *, noise_floor: float) -> list[bool]:
    """Which rows this reference can decide at all -- computed from the REFERENCE ALONE.

    Deliberately independent of any candidate: the partition is a property of the reference's own
    resolving power, so it can be declared BEFORE a run and cannot be widened by a candidate that
    would like more of its disagreements excused.
    """
    floor = _check_floor(noise_floor)
    return [decision_margin(row) > floor for row in reference_rows]


def spread_envelope(single_flip_spread: float, expected_flips: float) -> dict[str, Any]:
    """How far the reference's OWN unresolved ties can move a whole output.

    ``single_flip_spread`` is the measured end-to-end movement produced by forcing ONE tie-adjacent
    quantized code the other way; ``expected_flips`` is how many elements the fragility census puts
    within the pipeline's demonstrated per-element divergence of a tie. Their product is the
    worst-case (perfectly in-phase) spread the reference cannot distinguish from itself.

    This exists because an elementwise tolerance is not the only thing a top-1 verdict fails to
    catch: a miscompile that keeps every argmax while wrecking the values sails through a
    ranking-only gate, and against a reference of limited resolving power the deeper rankings are
    themselves unadjudicable. A bound DERIVED from the reference's own tie population still bites.
    """
    if not (float(single_flip_spread) >= 0.0):
        raise ValueError(f"single_flip_spread must be a non-negative measurement, got {single_flip_spread!r}")
    if not (float(expected_flips) >= 0.0):
        raise ValueError(f"expected_flips must be a non-negative count, got {expected_flips!r}")
    return {
        "schema": "reference_spread_envelope_v1",
        "single_flip_spread": float(single_flip_spread),
        "expected_flips": float(expected_flips),
        "envelope": float(single_flip_spread) * float(expected_flips),
        "reading": (
            "a candidate whose spread from the reference exceeds this is differing by more "
            "than the reference's own unresolved ties can account for; one at or below it "
            "is inside what the reference cannot resolve, which is NOT the same as agreeing"
        ),
    }


def verdict_from_choices(
    reference_rows: Sequence[Sequence[float]], candidate_choices: Sequence[int], *, noise_floor: float
) -> dict[str, Any]:
    """The per-row verdict when only the candidate's CHOICE came back, not its scores.

    A console-limited harness reports one argmax per row and a digest, because 256,000 logits do
    not fit down a UART. That is enough to adjudicate: the margins come from the reference, which
    the host holds in full.
    """
    floor = _check_floor(noise_floor)
    if len(reference_rows) != len(candidate_choices):
        raise ValueError(f"row count differs: reference {len(reference_rows)} vs candidate {len(candidate_choices)}")
    rows: list[dict[str, Any]] = []
    for index, (want, got_top) in enumerate(zip(reference_rows, candidate_choices)):
        want_top = max(range(len(want)), key=lambda i: float(want[i]))
        margin = decision_margin(want)
        rows.append(
            {
                "row": index,
                "status": classify(int(got_top) == want_top, margin, noise_floor=floor),
                "reference_margin": margin,
                "reference_choice": want_top,
                "candidate_choice": int(got_top),
            }
        )
    return _tally(rows, floor)


def _tally(rows: list[dict[str, Any]], floor: float) -> dict[str, Any]:
    """The three populations, kept apart. There is deliberately no boolean here."""
    counts = {PASS: 0, FAIL: 0, INDETERMINATE: 0}
    for row in rows:
        counts[row["status"]] += 1
    return {
        "schema": "margin_qualified_verdict_v1",
        "noise_floor": floor,
        "rows": rows,
        "agrees": counts[PASS],
        "disagrees": counts[FAIL],
        "reference_cannot_discriminate": counts[INDETERMINATE],
        # Deliberately NOT a single boolean. An indeterminate row is not agreement, and collapsing
        # the three populations into one verdict is what makes a gate unable to fail.
        "adjudicable": counts[PASS] + counts[FAIL],
        "licence": (
            "INDETERMINATE IS NOT PASS. A row whose reference margin is at or below the "
            "measured noise floor is unadjudicable by this reference; report it as such "
            "and never fold it into the agreeing population."
        ),
    }


def verdict(
    reference_rows: Sequence[Sequence[float]], candidate_rows: Sequence[Sequence[float]], *, noise_floor: float
) -> dict[str, Any]:
    """Per-row three-way verdict against a reference of limited resolving power.

    ``noise_floor`` is the demonstrated end-to-end divergence of the pipeline, in the same units as
    the scores. It is required: a guessed floor decides which rows become indeterminate, which is
    the whole verdict. A row is INDETERMINATE when the reference's own decision margin does not
    exceed it -- the reference simply does not adjudicate there.
    """
    floor = _check_floor(noise_floor)
    if len(reference_rows) != len(candidate_rows):
        raise ValueError(f"row count differs: reference {len(reference_rows)} vs candidate {len(candidate_rows)}")

    rows: list[dict[str, Any]] = []
    for index, (want, got) in enumerate(zip(reference_rows, candidate_rows)):
        if len(want) != len(got):
            raise ValueError(f"row {index} width differs: {len(want)} vs {len(got)}")
        margin = decision_margin(want)
        want_top = max(range(len(want)), key=lambda i: float(want[i]))
        got_top = max(range(len(got)), key=lambda i: float(got[i]))
        rows.append(
            {
                "row": index,
                "status": classify(want_top == got_top, margin, noise_floor=floor),
                "reference_margin": margin,
                "reference_choice": want_top,
                "candidate_choice": got_top,
            }
        )
    return _tally(rows, floor)
