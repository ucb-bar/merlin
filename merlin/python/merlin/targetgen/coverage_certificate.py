"""The per-compilation coverage certificate — the artifact that makes Acceleratable Region Recall
auditable.

For every region of a compiled workload it records BOTH sides of the ARR ratio:

- the **decision** the generated compiler actually made (``accelerator`` / ``in_contract`` /
  ``cpu_fallback``) — read from a :func:`merlin.targetgen.routing.route_plan` result (the *numerator*);
- the **eligibility** the hardware declares — from the independent oracle
  :func:`merlin.targetgen.eligibility.is_eligible` over the target's ``semantic_capabilities`` (the
  *denominator*).

The gap between the two is the compiler deficiency ARR is designed to surface: a region the hardware
*could* run (eligible) but the compiler left on the CPU lane (``false_fallback``), or — the integrity
direction — a region routed to the accelerator that the oracle says is *not* eligible
(``accelerated_ineligible``, an acceleration-precision miss).

This module consumes the routing output (numerator) and the eligibility oracle (denominator); it is NOT
the oracle, so it may legitimately reference both.
"""
from __future__ import annotations

from merlin.targetgen import eligibility as _el
from merlin.targetgen import semantic_families as _sf


def _decision_map(plan: dict) -> dict[int, str]:
    """Map each RouteResult (by identity) to the compiler's offload decision."""
    dec: dict[int, str] = {}
    for r in plan.get("mesh", []):
        dec[id(r)] = "accelerator"
    for r in plan.get("fallback", []):
        dec[id(r)] = "in_contract"
    for r in plan.get("scalar_rvv", []):
        dec[id(r)] = "cpu_fallback"
    return dec


def _flops(demand, family: str | None) -> int:
    """Best-effort work estimate. A contraction with known extents is ``2*M*K*N`` (MAC = mul+add);
    everything else is 0 until the region carries an element count (kept honest, never guessed)."""
    if family == "contraction" and demand.m and demand.k and demand.n:
        return 2 * int(demand.m) * int(demand.k) * int(demand.n)
    return 0


def _ratio(num: int, den: int):
    """num/den, or ``None`` when the denominator is empty (no eligible work to recall)."""
    return (num / den) if den else None


def build(plan: dict, cap_map: dict, *, target: str | None = None) -> dict:
    """Build the coverage certificate from a ``route_plan`` result and a capability map.

    ``plan`` is the dict returned by :func:`merlin.targetgen.routing.route_plan_on` / ``route_plan``
    (needs ``results`` + the ``mesh``/``fallback``/``scalar_rvv`` buckets). ``cap_map`` is the
    independent denominator from :func:`merlin.targetgen.eligibility.capability_map_for_target`.
    """
    dec = _decision_map(plan)
    regions: list[dict] = []
    n_eligible = n_accelerated = n_eligible_accelerated = 0
    eligible_flops = accelerated_eligible_flops = 0
    accelerated_ineligible = 0

    for r in plan.get("results", []):
        d = r.demand
        desc = _el.RegionDescriptor(source=d.site or d.op, op=d.op, in_dtype=d.in_fmt,
                                    weight_dtype=d.weight_fmt, m=d.m, k=d.k, n=d.n)
        verdict = _el.is_eligible(desc, cap_map)
        family = verdict.family or _sf.from_op(d.op)
        decision = dec.get(id(r), "cpu_fallback")
        accelerated = decision == "accelerator"
        flops = _flops(d, family)

        regions.append({
            "source": d.site or d.op,
            "op": d.op,
            "semantic_family": family,
            "target_eligible": verdict.eligible,
            "eligibility_reason": verdict.reason,
            "decision": decision,
            "unit": r.unit,
            "gap": r.gap,
            "estimated_work_flops": flops,
        })

        if verdict.eligible:
            n_eligible += 1
            eligible_flops += flops
            if accelerated:
                n_eligible_accelerated += 1
                accelerated_eligible_flops += flops
        if accelerated:
            n_accelerated += 1
            if not verdict.eligible:
                accelerated_ineligible += 1

    false_fallback = sum(1 for reg in regions
                         if reg["target_eligible"] and reg["decision"] != "accelerator")

    return {
        "target": target,
        "denominator_source": "semantic_capabilities (independent eligibility oracle)",
        "n_regions": len(regions),
        "n_eligible": n_eligible,
        "n_accelerated": n_accelerated,
        "n_eligible_accelerated": n_eligible_accelerated,
        "false_fallback_count": false_fallback,
        "accelerated_ineligible_count": accelerated_ineligible,
        "eligible_flops": eligible_flops,
        "accelerated_eligible_flops": accelerated_eligible_flops,
        "metrics": {
            # the headline ARR numbers for this single compilation (None == no eligible work)
            "acceleratable_region_recall": _ratio(n_eligible_accelerated, n_eligible),
            "acceleratable_flop_recall": _ratio(accelerated_eligible_flops, eligible_flops),
            "acceleration_precision": _ratio(n_accelerated - accelerated_ineligible, n_accelerated),
        },
        "regions": regions,
    }


def for_target(plan: dict, target: str) -> dict:
    """Convenience: load the target's declared capability map and build the certificate."""
    cap_map = _el.capability_map_for_target(target)
    return build(plan, cap_map, target=target)
