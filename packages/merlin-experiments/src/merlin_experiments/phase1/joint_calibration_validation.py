"""Strict local calibration admission; never broad timing qualification."""
from __future__ import annotations

import itertools
import math

from merlin_timing.fitting import quantile

PROFILE = "strict_per_case_v1"
GATES = dict(qualification_profile=PROFILE, max_case=.04, max_p95=.04,
             min_rank_agreement=.90, min_separation=.05,
             minimum_train_improvement=.02, max_condition=1000)


def _valid(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def local_validation(cases, labels, incumbent, candidate, *, gates):
    """Keep every declared held case; invalid or missing evidence prevents admission."""
    profile = gates.get("qualification_profile", "legacy_v1")
    if profile not in {PROFILE, "legacy_v1"}:
        raise ValueError("unsupported local calibration qualification profile")
    if profile == PROFILE and (gates.get("max_case") != .04 or gates.get("max_p95") != .04):
        raise ValueError("strict calibration profile requires the fixed four-percent gates")
    held = [case for case in cases if case["split"] == "held"]
    ids = [case["id"] for case in held]
    if len(ids) != len(set(ids)):
        raise ValueError("repeated held case identity")
    before, after, refusals, usable = [], [], [], set()
    for key in ids:
        if not all(_valid(values.get(key)) for values in (labels, incumbent, candidate)):
            refusals.append({"case_id": key, "reason": "missing, nonfinite or nonpositive timing"})
            continue
        before.append(abs(incumbent[key] - labels[key]) / labels[key])
        after.append(abs(candidate[key] - labels[key]) / labels[key])
        usable.add(key)
    pairs = []
    for a, b in itertools.combinations(held, 2):
        if a["id"] not in usable or b["id"] not in usable or a["capsule"]["sha256"] != b["capsule"]["sha256"]:
            continue
        x, y = labels[a["id"]], labels[b["id"]]
        if abs(x-y) / max(x,y) <= gates["min_separation"]:
            continue
        px, py = candidate[a["id"]], candidate[b["id"]]
        pairs.append((x-y)*(px-py) > 0)
    p95 = quantile(after, .95) if after else None
    old_p95 = quantile(before, .95) if before else None
    agreement = sum(pairs)/len(pairs) if pairs else None
    complete = bool(held) and len(usable) == len(held)
    maximum = max(after) if after else None
    admitted = bool(complete and p95 <= gates["max_p95"] and p95 <= old_p95
                    and sum(after) <= sum(before) and agreement is not None
                    and agreement >= gates["min_rank_agreement"]
                    and (profile == "legacy_v1" or maximum <= gates["max_case"]))
    return dict(status="limited_calibrated_provisional" if admitted else "not_promoted",
                admitted=admitted, p95=p95, incumbent_p95=old_p95,
                qualification_profile=profile, maximum_case_error=maximum,
                declared_held_cases=len(held), evaluated_held_cases=len(usable),
                all_declared_cases_evaluated=complete, refusals=refusals,
                materially_separated_pairs=len(pairs), rank_agreement=agreement,
                timing_qualified=False, broad_qualification=False,
                reason="local held-source accuracy and variant ordering only; full qualification gates unchanged")
