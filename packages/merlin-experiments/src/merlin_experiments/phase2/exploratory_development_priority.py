"""Pure, unqualified iteration guidance from already validated queued estimates.

The caller owns original source, count, numerical and completed-job custody.
This advisory projection neither validates those owners nor selects an incumbent.
"""

import copy
import math

from .contracts import StageGateError, canonical_json, document_sha256

SCHEMA = "merlin.exploratory_development_priority.v1"
ESTIMATE_SCHEMA = "merlin.public_native_exploratory_estimate.v1"


def _positive(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _signature(estimate):
    return tuple(estimate[k] for k in ("model_sha256", "comparison_domain_sha256", "scope_sha256"))


def _estimate(arm):
    if type(arm) is not dict or arm.get("functional_status") != "PASS":
        return None, "original complete functional arm unavailable"
    compiled = arm.get("compiled")
    if type(compiled) is not dict or compiled.get("status") != "observed" or not _sha(compiled.get("elf_sha256")):
        return None, "original compiled executable unavailable"
    native = arm.get("native_exploration")
    estimate = native.get("exploratory_estimate") if type(native) is dict else None
    if (
        type(estimate) is not dict or estimate.get("schema") != ESTIMATE_SCHEMA
        or estimate.get("qualified_for_promotion") is not False
        or estimate.get("status") != "estimated" or not _positive(estimate.get("estimated_cycles"))
        or any(not _sha(estimate.get(k)) for k in (
            "model_sha256", "comparison_domain_sha256", "scope_sha256", "report_sha256"
        ))
        or type(estimate.get("domain_deviations")) is not list
        or type(estimate.get("calibration_residuals")) is not dict
        or "event_accounting_complete" in estimate and type(estimate["event_accounting_complete"]) is not bool
    ):
        return None, "explicit finite exploratory estimate and provenance unavailable"
    if len(canonical_json([estimate["domain_deviations"], estimate["calibration_residuals"]])) > 64 << 10:
        raise StageGateError("exploratory estimate annotations exceed their descriptive input bound")
    return estimate, None


def decide(data, *, required_members, minimum_improvement=0.01):
    """Consume normal queue ``result.data``; recommend only weight-free dominance.

    Pass the original frozen SearchSettings.minimum_improvement when selected.
    An event-accounting flag absent or false leaves ratios visible but prevents
    latency-based parent guidance. Domain deviations are warnings, not gates to
    this explicitly exploratory advice. Formal qualification stays separate.
    """
    if (
        not _positive(minimum_improvement) or minimum_improvement >= 1
        or type(required_members) is not tuple or not 1 <= len(required_members) <= 4096
        or any(type(key) is not tuple or len(key) != 2
               or any(type(part) is not str or not part for part in key) for key in required_members)
        or len(set(required_members)) != len(required_members)
        or type(data) is not dict
    ):
        raise StageGateError("exploratory priority requires its exact declared roster and frozen improvement threshold")
    selected, cells = data.get("selected_members"), data.get("cells")
    if (
        type(selected) is not list or any(type(k) is not list or len(k) != 2
                                         or any(type(p) is not str for p in k) for k in selected)
        or sorted(selected) != sorted([list(k) for k in required_members])
        or type(cells) is not list or len(cells) > 4096
        or any(type(c) is not dict or any(type(c.get(k)) is not str for k in ("family", "capsule")) for c in cells)
    ):
        raise StageGateError("exploratory priority cannot drop or replace selected members")
    by_key = {(c["family"], c["capsule"]): c for c in cells}
    if len(by_key) != len(cells) or not set(required_members) <= by_key.keys():
        raise StageGateError("exploratory priority requires every original selected cell exactly once")
    try:
        feedback_sha256 = document_sha256(data)
    except (TypeError, ValueError, OverflowError) as error:
        raise StageGateError("exploratory priority requires finite serializable queued feedback") from error
    rows, signatures = [], set()
    for key in required_members:
        cell = by_key[key]
        estimates = [_estimate(cell.get(arm)) for arm in ("baseline", "candidate")]
        row = dict(family=key[0], capsule=key[1], status="unavailable", reason=None,
                   baseline_estimated_cycles=None, candidate_estimated_cycles=None,
                   candidate_to_baseline_ratio=None, event_accounting_complete=False,
                   recommendation=None, provenance=None)
        if any(estimate is None for estimate, _ in estimates):
            row["reason"] = "; ".join(arm + ": " + reason for arm, (_, reason) in
                                      zip(("baseline", "candidate"), estimates, strict=True) if reason)
        else:
            baseline, candidate = [estimate for estimate, _ in estimates]
            if _signature(baseline) != _signature(candidate):
                row["reason"] = "arm estimates use different model, comparison domain or scope"
            else:
                ratio = candidate["estimated_cycles"] / baseline["estimated_cycles"]
                same_elf = cell["baseline"]["compiled"]["elf_sha256"] == cell["candidate"]["compiled"]["elf_sha256"]
                if not _positive(ratio) or same_elf and baseline["estimated_cycles"] != candidate["estimated_cycles"]:
                    row["reason"] = "fixed-model estimate ratio is nonfinite or identical executable estimates disagree"
                else:
                    signatures.add(_signature(baseline))
                    complete = all(e.get("event_accounting_complete") is True for e in (baseline, candidate))
                    pair = ("diagnostic_pair_check" not in cell
                            or (cell["diagnostic_pair_check"] or {}).get("status") == "PASS")
                    row.update(
                        status="estimated", baseline_estimated_cycles=baseline["estimated_cycles"],
                        candidate_estimated_cycles=candidate["estimated_cycles"], candidate_to_baseline_ratio=ratio,
                        event_accounting_complete=complete and pair,
                        reason=("comparable exploratory estimates" if complete and pair else
                                "latency advice unavailable: event accounting or original pair check incomplete"),
                        provenance={
                            arm: {k: copy.deepcopy(e[k]) for k in (
                                "model_sha256", "comparison_domain_sha256", "scope_sha256", "report_sha256",
                                "domain_deviations", "calibration_residuals"
                            )}
                            for arm, e in zip(("baseline", "candidate"), (baseline, candidate), strict=True)
                        },
                    )
                    if complete and pair and ratio <= 1 - minimum_improvement:
                        row["recommendation"] = "candidate"
                    elif complete and pair and 1 / ratio <= 1 - minimum_improvement:
                        row["recommendation"] = "baseline"
        rows.append(row)
    coverage = all(row["status"] == "estimated" for row in rows) and len(signatures) == 1
    latency_coverage = coverage and all(row["event_accounting_complete"] for row in rows)
    recommendation = None
    if latency_coverage:
        ratios = [row["candidate_to_baseline_ratio"] for row in rows]
        if all(r <= 1 for r in ratios) and any(r <= 1 - minimum_improvement for r in ratios):
            recommendation = "candidate"
        elif all(r >= 1 for r in ratios) and any(1 / r <= 1 - minimum_improvement for r in ratios):
            recommendation = "baseline"
    return dict(
        schema=SCHEMA, status="EXPLORATORY", feedback_sha256=feedback_sha256,
        required_members=[list(k) for k in required_members], minimum_improvement=minimum_improvement,
        estimate_coverage_complete=coverage, latency_recommendation_coverage_complete=latency_coverage,
        members=rows, next_exploration_parent=recommendation,
        reason=("weight-free dominance advice" if recommendation else
                "abstain: incomplete/incompatible estimates, accounting, tradeoff or subthreshold change"),
        qualified_for_promotion=False, promotion_authority="NONE", measured_cycles=None,
        aggregate_objective=None, plateau_evidence=False,
        scope="advisory next-parent guidance only; original source/count custody belongs to the caller",
    )
