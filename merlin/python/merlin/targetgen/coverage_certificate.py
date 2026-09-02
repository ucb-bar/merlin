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

⚠️ BOTH SIDES OF THE RATIO ARE BUILT FROM THE ROUTING DEMANDS, so a contraction the matcher never
recognised is in neither. It is not a false fallback and not an ineligible acceleration; it is simply
absent, and the recall it never entered reads high because of it. Measured on ``spectformer_int8_full``:
16 ``linalg.generic`` ops are attention's Q·Kᵀ and scores·V — 157.4 MMAC against 1702.2 MMAC of matched
work, so **8.5% of that model's contraction MACs were invisible to every number here**. ``linalg_mlir``
therefore lets the certificate price what the demands could not see and report it as
``denominator_completeness``, alongside a LOWER BOUND for BOTH recalls -- region and flop -- each
charging the whole unmatched mass to its own denominator. Every recall here is therefore a bracket, and
quoting only the upper half of one is the error this block exists to stop.
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


def denominator_completeness(linalg_mlir: str | None) -> dict | None:
    """Price the contraction work the ROUTING DEMANDS could not see, from the module itself.

    ``None`` when there is no module to read. On a module that will not parse this returns an ``error``
    entry rather than nothing: a certificate that silently omits the completeness block is
    indistinguishable from one whose denominator is provably complete, which is the direction that
    flatters us.
    """
    if not linalg_mlir:
        return None
    try:
        from merlin.common import mlir_query as _mq
        from merlin.common.ir_lock import IR_LOCK
        from merlin.xdsl_dialects.lowering.contraction_coverage import contraction_coverage
        with IR_LOCK:                     # xDSL's parser is not thread-safe; see common.ir_lock
            rep = contraction_coverage(_mq.parse(linalg_mlir))
    except Exception as exc:              # noqa: BLE001 — advisory, but the gap must stay visible
        return {"error": f"{type(exc).__name__}: {exc}",
                "note": "denominator completeness UNKNOWN — the module could not be priced, so the "
                        "recall below is an upper bound with no stated floor"}

    caveats: list[str] = []
    if rep.unlowered:
        caveats.append(
            f"{len(rep.unlowered)} contraction(s) worth {rep.unlowered_macs} MAC "
            f"({rep.unlowered_share:.1%} of all contraction MACs) stayed linalg.generic, so they never "
            f"became routing demands and appear in NEITHER side of the recall above")
    if rep.unpriceable:
        caveats.append(
            f"{len(rep.unpriceable)} contraction(s) could not be priced (no derivable loop extents), so "
            f"even the lower bound below is optimistic — they are counted as ops, never as work")
    return {
        "matched_contraction_macs": rep.lowered_macs,
        "unmatched_contraction_macs": rep.unlowered_macs,
        "unmatched_contraction_share": rep.unlowered_share,
        "n_unmatched_contractions": len(rep.unlowered),
        "n_unpriceable_contractions": len(rep.unpriceable),
        "unpriceable_result_types": list(rep.unpriceable),
        "unmatched": [{"result_type": u.result_type,
                       "loop_extents": {str(d): e for d, e in u.loop_extents},
                       "macs": u.macs} for u in rep.unlowered],
        "generic_labels": dict(rep.labels),
        "caveats": caveats,
    }


def executed_false_fallbacks(execution: dict | None) -> dict:
    """Which kernels the router ASSIGNED to the accelerator did not execute there.

    The execution-evidenced counterpart of ``false_fallback_count``, and it is computable for a reason
    worth stating: both halves are keyed by the SAME thing, the kernel symbol. ``mesh_route_symbols``
    is what the router assigned; the dispatch ledger records, per completed call, which lane it went to.
    No join to the plan's per-op demands is required -- that join does not exist, and this does not need
    it, so the measurable half of the question is answerable without inventing the unmeasurable half.

    ``status`` is ``not_measured`` when either side is absent, never an empty list of fallbacks: "no
    symbols fell back" and "nobody recorded which symbols ran" are opposite conclusions.
    """
    ex = execution or {}
    routed = ex.get("mesh_route_symbols")
    ledger = ex.get("dispatch_ledger")
    if not isinstance(routed, (list, tuple)) or not isinstance(ledger, list):
        return {"status": "not_measured",
                "detail": "the run recorded no route symbols or no dispatch ledger, so which assigned "
                          "kernel executed where was never observed"}
    on_accel = {str(e.get("symbol")) for e in ledger
                if isinstance(e, dict) and e.get("status") == "pass" and e.get("lane") == "on_mesh"}
    assigned = [str(x) for x in routed]
    fell_back = sorted(sym for sym in assigned if sym not in on_accel)
    return {"status": "measured", "n_routed": len(assigned),
            "n_executed_on_accelerator": len([s for s in assigned if s in on_accel]),
            "n_false_fallback": len(fell_back),
            # Named, not just counted: "4 kernels fell back" gives a reader nothing to act on.
            "false_fallback_symbols": fell_back[:64],
            "detail": "kernels the router assigned to the accelerator that no completed call placed "
                      "there; joined on the kernel symbol, which both records carry"}


def build(plan: dict, cap_map: dict, *, target: str | None = None,
          linalg_mlir: str | None = None, execution: dict | None = None) -> dict:
    """Build the coverage certificate from a ``route_plan`` result and a capability map.

    ``plan`` is the dict returned by :func:`merlin.targetgen.routing.route_plan_on` / ``route_plan``
    (needs ``results`` + the ``mesh``/``fallback``/``scalar_rvv`` buckets). ``cap_map`` is the
    independent denominator from :func:`merlin.targetgen.eligibility.capability_map_for_target`.

    ``execution`` is the run's ``mesh_execution`` record, when one exists. EVERY NUMBER BELOW IS
    DERIVED FROM THE PLAN, which lists what the router ASSIGNED and not what ran -- exactly the
    conflation already fixed on the lane side, where one submission assigned 15 matmuls to the mesh and
    fell back on all 15 at run time. The certificate therefore states its own evidence in ``arr_evidence``
    and, when execution accounting exists, carries it beside the plan numbers in ``execution_crosscheck``
    with ``agrees`` set. A disagreement means the recalls above describe intent that did not happen.

    ``linalg_mlir`` is the module those demands were derived FROM. Pass it and the certificate also
    reports what the demands missed; omit it and the completeness block is absent, which is itself
    reported (``denominator_completeness: null``) rather than read as "nothing was missed".
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

    # Work the matcher never turned into a demand. A MAC is a multiply AND an add, so it is 2 flops on
    # the same scale `_flops` uses -- mixing the two units would understate the correction by half.
    completeness = denominator_completeness(linalg_mlir)
    unmatched_flops = 2 * int(completeness.get("unmatched_contraction_macs") or 0) if completeness else 0
    unmatched_regions = int(completeness.get("n_unmatched_contractions") or 0) if completeness else 0

    # THE EVIDENCE, said out loud. A recall that reads as a measurement of the compiler when it is a
    # summary of the router is the same defect `lane_report` was hardened against, and this surface had
    # never been told about it.
    xcheck = None
    if execution:
        def _n(v):
            return None if v is None or isinstance(v, str) else int(v)
        routed, ran = _n(execution.get("matmul_layers_routed")), _n(execution.get("matmul_layers_on_mesh"))
        fell = _n(execution.get("matmul_layers_host_fallback"))
        xcheck = {"matmul_layers_routed": routed, "matmul_layers_on_mesh": ran,
                  "matmul_layers_host_fallback": fell,
                  # None, never True: "nobody could tell" is not "they agree".
                  "agrees": None if (routed is None or ran is None) else (routed == ran),
                  "why": ("the plan assigned `routed` contraction layers to the accelerator and `on_mesh` "
                          "of them executed there; when these differ, the recalls in this certificate "
                          "describe an intent the run did not carry out")}

    # The execution-evidenced half of the same question, reported BESIDE the plan-derived recalls rather
    # than replacing them: the recalls are per-region and this is per-kernel-symbol, so they are not the
    # same denominator and collapsing them would invent a number neither record supports.
    executed = executed_false_fallbacks(execution)

    return {
        "target": target,
        "denominator_source": "semantic_capabilities (independent eligibility oracle)",
        "arr_evidence": "routing_plan",
        "execution_crosscheck": xcheck,
        "executed_false_fallbacks": executed,
        "n_regions": len(regions),
        "n_eligible": n_eligible,
        "n_accelerated": n_accelerated,
        "n_eligible_accelerated": n_eligible_accelerated,
        "false_fallback_count": false_fallback,
        "accelerated_ineligible_count": accelerated_ineligible,
        "eligible_flops": eligible_flops,
        "accelerated_eligible_flops": accelerated_eligible_flops,
        "unmatched_contraction_flops": unmatched_flops,
        "unmatched_contraction_regions": unmatched_regions,
        "denominator_completeness": completeness,
        "metrics": {
            # the headline ARR numbers for this single compilation (None == no eligible work).
            # The two recalls are computed over the regions the matcher PRODUCED, which is why each
            # carries a lower bound beside it -- see the module docstring. Quote a recall WITH its
            # bound or not at all: alone, the upper one reads as a measurement when it is a ceiling.
            "acceleratable_region_recall": _ratio(n_eligible_accelerated, n_eligible),
            "acceleratable_flop_recall": _ratio(accelerated_eligible_flops, eligible_flops),
            # REGION recall's floor, by the same construction as the flop floor below. This is the one
            # people actually quote -- it is the plain "how many of the regions it could accelerate did
            # it" number -- and it was the only headline metric here with no floor beside it, so the
            # single most-cited figure was also the single least-bracketed one. Charging every unmatched
            # contraction to the denominator is deliberately the unflattering assumption: each is counted
            # as eligible and unaccelerated.
            "acceleratable_region_recall_lower_bound":
                _ratio(n_eligible_accelerated, n_eligible + unmatched_regions),
            # The same recall with every unmatched contraction charged to the denominator: the floor
            # under the number above, on the assumption (deliberately the unflattering one) that all of
            # that work was eligible and none of it was accelerated. True recall lies between the two;
            # they coincide exactly when the matcher missed nothing.
            "acceleratable_flop_recall_lower_bound":
                _ratio(accelerated_eligible_flops, eligible_flops + unmatched_flops),
            "acceleration_precision": _ratio(n_accelerated - accelerated_ineligible, n_accelerated),
        },
        "regions": regions,
    }


def for_target(plan: dict, target: str, *, linalg_mlir: str | None = None,
               execution: dict | None = None) -> dict:
    """Convenience: load the target's declared capability map and build the certificate."""
    cap_map = _el.capability_map_for_target(target)
    return build(plan, cap_map, target=target, linalg_mlir=linalg_mlir, execution=execution)
