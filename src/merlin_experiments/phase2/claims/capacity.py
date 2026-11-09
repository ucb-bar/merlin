"""Decision procedure for the capacity cohort's per-regime PREDICTS claim.

A capacity-family member (:mod:`merlin_experiments.phase0.capacity_families`) carries, frozen at
generation from the target's facts, its family, variant, regime, the regime variable ``x`` and its
staged roofline floor. The claim is a SCALING law per regime, never a single point and never a
monotone "fewer is better":

* within one ``(family, variant, regime)`` group, measured cycles are affine in ``x`` with a positive
  slope, ``r^2`` and residuals inside the frozen thresholds;
* no measurement is below its member's staged roofline floor (a count below the floor refutes the
  floor, which is a statement about the facts and is reported as such).

Slopes are reported per regime, and so is the ratio between adjacent regimes of one variant, because
an optimization's profit is regime-dependent and may be negative: fewer commands can be slower once
overlap is lost. Nothing here prefers one regime's slope to another's.

Same verdict discipline as the other analyzers: ``REFUSED`` for incomplete or malformed evidence,
``REFUTED`` for complete evidence outside a frozen bound, ``ACCEPTED`` otherwise. Thresholds are read
from the contract, never defaulted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

ANALYZER = "perf_capacity_claim.analyze_capacity_claim/v1"
ACCEPTED, REFUTED, REFUSED = "ACCEPTED", "REFUTED", "REFUSED"
_GROUP_FIELDS = ("family", "variant", "regime")


class _Refusal(ValueError):
    pass


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ols(points: Sequence[tuple[float, float]]) -> tuple[float, float, float] | None:
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in points)
    if sxx == 0:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / sxx
    intercept = mean_y - slope * mean_x
    total = sum((y - mean_y) ** 2 for _, y in points)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    return intercept, slope, (1.0 if total == 0 else 1.0 - residual / total)


def _declaration(descriptors: object) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    if not isinstance(descriptors, Sequence) or isinstance(descriptors, str) or not descriptors:
        raise _Refusal("no capsule descriptors were supplied")
    contracts, members = [], {}
    for d in descriptors:
        perf = d.get("performance") if isinstance(d, Mapping) else None
        if not isinstance(perf, Mapping) or perf.get("claim") != "PREDICTS":
            raise _Refusal("every member must carry a PREDICTS performance block")
        capacity = perf.get("capacity")
        if not isinstance(capacity, Mapping):
            raise _Refusal(f"member {d.get('name')!r} carries no frozen capacity block")
        if any(not isinstance(capacity.get(k), str) or not capacity.get(k) for k in _GROUP_FIELDS):
            raise _Refusal(f"member {d.get('name')!r} does not freeze its family, variant and regime")
        if not _number(capacity.get("x")) or capacity["x"] <= 0:
            raise _Refusal(f"member {d.get('name')!r} does not freeze a positive regime variable")
        name = str(d.get("name") or "")
        if not name or name in members:
            raise _Refusal("members need unique names")
        members[name] = capacity
        contracts.append(perf.get("acceptance"))
    first = contracts[0]
    if not isinstance(first, Mapping) or any(c != first for c in contracts):
        raise _Refusal("members disagree about the frozen acceptance contract")
    if first.get("analyzer") != ANALYZER:
        raise _Refusal(f"the contract names a different analyzer {first.get('analyzer')!r}")
    fit, cohort, thresholds = first.get("fit"), first.get("cohort"), first.get("thresholds")
    if not all(isinstance(x, Mapping) for x in (fit, cohort, thresholds)):
        raise _Refusal("the contract is missing its fit, cohort or thresholds block")
    if list(fit.get("group_by") or []) != list(_GROUP_FIELDS):
        raise _Refusal(f"the contract must group fits by {list(_GROUP_FIELDS)}")
    if not isinstance(fit.get("dependent_metric"), str) or not fit["dependent_metric"]:
        raise _Refusal("the contract does not name its dependent metric")
    minimum = cohort.get("min_points_per_regime")
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 2:
        raise _Refusal("a rate and an intercept need at least two points per regime")
    for key in ("slope_min_exclusive", "r_squared_min_inclusive"):
        if not _number(thresholds.get(key)):
            raise _Refusal(f"the contract omits {key}")
    bound = thresholds.get("residual_bound")
    if not isinstance(bound, Mapping) or not all(
        _number(bound.get(k)) for k in ("absolute_floor_cycles", "observed_cycle_fraction")
    ):
        raise _Refusal("the residual bound is not fully specified")
    replicates = first.get("replicates")
    identities = replicates.get("identities") if isinstance(replicates, Mapping) else None
    if (
        not isinstance(identities, list)
        or not identities
        or replicates.get("exact_count") != len(identities)
        or len(set(identities)) != len(identities)
    ):
        raise _Refusal("the contract does not freeze its replicate identities")
    groups: dict[tuple[str, ...], set[float]] = {}
    for capacity in members.values():
        groups.setdefault(tuple(capacity[k] for k in _GROUP_FIELDS), set()).add(float(capacity["x"]))
    thin = sorted("/".join(key) for key, xs in groups.items() if len(xs) < minimum)
    if thin:
        raise _Refusal(f"regime group(s) {thin} have fewer than {minimum} distinct regime-variable values")
    return members, first


def preflight_capacity_claim(descriptors: object) -> dict[str, Any]:
    """Is the cohort decidable before any cycle is spent: groups, thresholds and replicates frozen."""
    try:
        members, contract = _declaration(descriptors)
    except _Refusal as exc:
        return {"verdict": REFUSED, "reason": str(exc)}
    groups = sorted({"/".join(c[k] for k in _GROUP_FIELDS) for c in members.values()})
    return {"verdict": "ADMISSIBLE", "members": len(members), "groups": groups, "analyzer": contract["analyzer"]}


def analyze_capacity_claim(descriptors: object, results: object) -> dict[str, Any]:
    """Decide the per-regime scaling claim over frozen descriptors and authored result rows."""
    try:
        members, contract = _declaration(descriptors)
    except _Refusal as exc:
        return {"verdict": REFUSED, "reason": str(exc)}
    if not isinstance(results, Sequence) or isinstance(results, str) or not results:
        return {"verdict": REFUSED, "reason": "no result rows were supplied"}
    metric = contract["fit"]["dependent_metric"]
    identities = [str(i) for i in contract["replicates"]["identities"]]
    seen: dict[tuple[str, str], float] = {}
    for row in results:
        if not isinstance(row, Mapping):
            return {"verdict": REFUSED, "reason": "a result row is not a mapping"}
        key = (str(row.get("capsule") or ""), str(row.get("replicate") or ""))
        if key[0] not in members:
            return {"verdict": REFUSED, "reason": f"result row names capsule {key[0]!r} outside the cohort"}
        if key in seen:
            return {"verdict": REFUSED, "reason": f"(capsule, replicate) {key} is reported twice"}
        value = row.get(metric)
        if not _number(value) or value <= 0:
            return {"verdict": REFUSED, "reason": f"{key} carries no positive {metric}"}
        seen[key] = float(value)
    expected = {(name, identity) for name in members for identity in identities}
    if set(seen) != expected:
        missing = sorted(expected - set(seen))
        return {
            "verdict": REFUSED,
            "reason": "the evidence is not the exact predeclared cohort",
            "missing": missing[:8],
        }

    thresholds = contract["thresholds"]
    bound = thresholds["residual_bound"]
    by_group: dict[tuple[str, ...], list[tuple[float, float, str]]] = {}
    floor_refutations, attainment = [], {}
    for (name, _replicate), cycles in seen.items():
        capacity = members[name]
        by_group.setdefault(tuple(capacity[k] for k in _GROUP_FIELDS), []).append((float(capacity["x"]), cycles, name))
        floor = (capacity.get("roofline") or {}).get("floor_cycles")
        if _number(floor) and floor > 0:
            if cycles < floor:
                floor_refutations.append({"capsule": name, "cycles": cycles, "floor_cycles": floor})
            else:
                attainment[name] = min(attainment.get(name, 1.0), floor / cycles)
    fits, reasons = {}, []
    for key in sorted(by_group):
        points = [(x, y) for x, y, _ in by_group[key]]
        fitted = _ols(points)
        label = "/".join(key)
        if fitted is None:
            return {"verdict": REFUSED, "reason": f"group {label} does not vary its regime variable"}
        intercept, slope, r2 = fitted
        breaches = [
            {"x": x, "cycles": y}
            for x, y in points
            if abs(y - (intercept + slope * x))
            > max(float(bound["absolute_floor_cycles"]), float(bound["observed_cycle_fraction"]) * y)
        ]
        fits[label] = {
            "intercept_cycles": intercept,
            "slope_cycles_per_unit": slope,
            "r_squared": r2,
            "x_name": members[by_group[key][0][2]].get("x_name"),
            "distinct_x": sorted({x for x, _ in points}),
            "breaches": breaches[:8],
        }
        if not slope > float(thresholds["slope_min_exclusive"]):
            reasons.append(f"{label}: slope {slope:.6g} is not above the frozen floor")
        if r2 < float(thresholds["r_squared_min_inclusive"]):
            reasons.append(f"{label}: r_squared {r2:.6g} is below the frozen minimum")
        if breaches:
            reasons.append(f"{label}: {len(breaches)} observation(s) exceed the frozen residual bound")
    if floor_refutations:
        reasons.append(
            f"{len(floor_refutations)} measurement(s) beat their staged roofline floor: an input the floor was "
            "derived from does not describe this machine"
        )
    knees = {}
    for family, variant in sorted({(k[0], k[1]) for k in by_group}):
        regimes = sorted(
            (k[2], fits["/".join(k)]["slope_cycles_per_unit"], min(fits["/".join(k)]["distinct_x"]))
            for k in by_group
            if k[:2] == (family, variant)
        )
        ordered = sorted(regimes, key=lambda r: r[2])
        knees[f"{family}/{variant}"] = [
            {"from": a[0], "to": b[0], "slope_ratio": (b[1] / a[1]) if a[1] else None}
            for a, b in zip(ordered, ordered[1:])
        ]
    measured = {
        "fits": fits,
        "regime_slope_ratios": knees,
        "attainment_by_capsule": attainment,
        "floor_refutations": floor_refutations[:8],
    }
    if reasons:
        return {"verdict": REFUTED, "reasons": reasons, "measured": measured}
    return {"verdict": ACCEPTED, "measured": measured}
