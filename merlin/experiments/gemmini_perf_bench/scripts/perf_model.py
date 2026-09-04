"""Derive this target's performance model from its OWN sources, every run, and price the corpus.

Nothing here is hand-written about a machine. The ceiling comes from the target's RTL facts, the work
comes from the compiler's own emitted command buffers, and the achieved rates come from cycles a
previous run already measured. Point it at another target's facts and run root and it derives that
target's model instead -- there is no target name in this file.

**Why it exists.** The performance experiment could only learn a cycle count by running the cycle
oracle, and that oracle costs ``12.89 s + 4.60 ms per simulated cycle`` -- ~217 simulated cycles/s,
fitted on 170 phase-1 points spanning 161..28,118 cycles. A realistic convolution layer is ~451k
cycles, i.e. ~35 minutes per member per measurement and ~4.6 h per feedback call, so the large shapes
that most need optimising are exactly the ones the oracle cannot reach. Meanwhile phase 1 had already
measured 44 shapes spanning 4k..2.1M MACs and 269..28,118 cycles, and thrown the timing away.

**Two ceilings, never one.** The structural peak is what the array could retire if nothing ever
stalled; it is derived from the RTL and it is unreachable. The achievable ceiling is the best rate any
measured point actually reached. Reporting only the first invites chasing a number no program can
reach; reporting only the second hides how much of the machine is structurally unavailable. Both are
reported, and the gap between them is a fact about the machine rather than about the compiler.

**What this model may and may not be used for.** It prices work against a rate, so it is a *ballpark*
and its measured error is stated by :func:`prediction_error` rather than assumed -- measured at mean
-73.7% on gemmini, so it is NOT a substitute for the cycle oracle on any claim needing a magnitude.
Its sound uses are the two ceilings and the headroom RANKING between shapes.

⚠️ Ordering two schedules of the SAME shape via ``merlin.perf.differential`` does not work on a
partial-overlap machine -- see :func:`schedule_ordering`, which is wired and tested precisely so that
refusal is visible rather than rediscovered.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.perf.decompose import ResourceKind
from merlin.perf.envelope import Basis, Peak, ResourceDemand, resource_time

COMPUTE = "compute"


@dataclass(frozen=True)
class MeasuredPoint:
    """One shape whose work is known exactly and whose cycles were measured by the cycle oracle."""

    capsule: str
    macs: int
    cycles: int
    source: str

    @property
    def achieved_rate(self) -> float:
        return self.macs / self.cycles


@dataclass(frozen=True)
class Headroom:
    """How far one measured shape sits from each ceiling."""

    point: MeasuredPoint
    achievable_rate: float
    structural_rate: int

    @property
    def share_of_achievable(self) -> float:
        return self.point.achieved_rate / self.achievable_rate

    @property
    def share_of_structural(self) -> float:
        return self.point.achieved_rate / self.structural_rate

    @property
    def factor_to_achievable(self) -> float:
        """How many times faster this shape would run at the best rate anything actually reached."""
        return self.achievable_rate / self.point.achieved_rate


def structural_ceiling(rtl_facts_path: Path, target: str) -> tuple[int | None, str]:
    """The machine's structural MAC ceiling, derived from its own RTL facts.

    Delegates to :mod:`merlin.perf.contract`, which reads the discovered array's geometry (rows x
    cols x the multipliers per element its ``mac_idiom`` states) and refuses rather than inventing a
    peak when no array grounds the unit. More than one compute resource carrying a peak is also a
    refusal: picking one would be choosing which machine the number describes.
    """
    try:
        from merlin.perf.contract import derive_contract  # noqa: PLC0415
        contract = derive_contract(
            target, facts=json.loads(Path(rtl_facts_path).read_text(encoding="utf-8")))
    except Exception as exc:  # noqa: BLE001 - an underivable ceiling is reported, never guessed
        return None, f"peak is not derivable from this target's RTL facts ({type(exc).__name__})"
    peaks = []
    for resource in contract.resources:
        value = getattr((resource.terms or {}).get("peak_macs_per_cycle"), "value", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            peaks.append((resource.name, value))
    if not peaks:
        return None, "this target's RTL facts evidence no compute array, so it has no derived peak"
    if len(peaks) > 1:
        names = ", ".join(sorted(name for name, _ in peaks))
        return None, f"this target evidences several compute units ({names}); utilization needs one"
    name, value = peaks[0]
    return value, f"facts-derived peak of compute unit {name!r}"


def harvest_measured_points(run_root: Path) -> tuple[list[MeasuredPoint], list[str]]:
    """Recover (exact work, measured cycles) pairs a completed run already produced.

    The work is the compiler's OWN emitted command buffer priced by
    :func:`merlin.perf.work_volume.work_from_command_buffer`, so it is what the program actually asks
    the array to do, not a shape someone declared. A buffer whose work is a lower bound is skipped
    rather than counted: a rate built on an understated numerator would flatter every candidate.

    Returns the points and the reasons any result was skipped, because a corpus that silently lost
    half its members is indistinguishable from one that never had them.
    """
    from merlin.perf.work_volume import work_from_command_buffer  # noqa: PLC0415

    points: dict[str, MeasuredPoint] = {}
    skipped: list[str] = []
    for result in sorted(Path(run_root).rglob("capsule_result.json")):
        try:
            document = json.loads(result.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable result is reported, never counted
            skipped.append(f"{result.parent.name}: capsule result is unreadable")
            continue
        cycles = ((document.get("tiers") or {}).get("L3") or {}).get("cycles")
        if not isinstance(cycles, int) or isinstance(cycles, bool) or cycles <= 0:
            continue  # no cycle-oracle verdict here; not an error, just not a measured point
        buffer_path = result.parent / "generated" / "command_buffer.json"
        if not buffer_path.is_file():
            skipped.append(f"{result.parent.name}: measured {cycles} cycles but emitted no "
                           "command buffer, so its work cannot be priced")
            continue
        try:
            work = work_from_command_buffer(json.loads(buffer_path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{result.parent.name}: command buffer did not price ({type(exc).__name__})")
            continue
        if work.is_lower_bound or not work.exact_macs or work.exact_macs <= 0:
            skipped.append(f"{result.parent.name}: work is a lower bound or zero "
                           f"({len(work.refusals)} refusal(s)), so no rate may be built from it")
            continue
        points.setdefault(result.parent.name,
                          MeasuredPoint(result.parent.name, int(work.exact_macs), cycles,
                                        str(result.parent)))
    return sorted(points.values(), key=lambda p: p.capsule), skipped


def achievable_ceiling(points: Sequence[MeasuredPoint], *, provenance: str) -> Peak:
    """The best rate anything actually reached -- derived, then falsified against every sample.

    :meth:`Peak.observed_ceiling` re-checks ``demand / rate <= busy`` on every point it was built
    from and returns UNKNOWN rather than a rate if any point violates it, so this cannot quietly
    become a nameplate number. ``merlin.perf.roofline`` will only admit a peak of this kind.
    """
    if not points:
        return Peak.unknown(COMPUTE, "mac", "no measured point priced its work", provenance=provenance)
    return Peak.observed_ceiling(COMPUTE, [(p.macs, p.cycles) for p in points],
                                 unit="mac", provenance=provenance)


def rank_headroom(points: Iterable[MeasuredPoint], *, achievable: Peak,
                  structural: int) -> list[Headroom]:
    """Every measured shape, worst utilisation first -- the optimisation order.

    The ranking is against the ACHIEVABLE ceiling, because that is the rate something on this machine
    demonstrably reached. A shape far below it is evidence of a compiler gap; a shape at it has no
    headroom left to find without moving the ceiling itself.
    """
    if not achievable.known:
        return []
    return sorted((Headroom(p, float(achievable.value), structural) for p in points),
                  key=lambda h: h.share_of_achievable)


def prediction_error(points: Sequence[MeasuredPoint], peak: Peak) -> dict[str, Any]:
    """How wrong pricing work against a single rate is, per point and in aggregate.

    This is the number that licenses -- or refuses -- any use of the model in place of the oracle.
    It is reported, never assumed: a model whose error nobody measured is a model nobody may cite.
    """
    if not peak.known or not points:
        return {"status": "unavailable", "reason": "no ceiling or no points", "rows": []}
    rows = []
    for p in points:
        predicted = resource_time(
            ResourceDemand(COMPUTE, ResourceKind.COMPUTE, p.macs, "mac", Basis.MOVED), peak).cycles
        rows.append({"capsule": p.capsule, "macs": p.macs, "measured_cycles": p.cycles,
                     "predicted_cycles": predicted,
                     "relative_error": (predicted - p.cycles) / p.cycles})
    errors = sorted(row["relative_error"] for row in rows)
    return {"status": "measured", "n": len(rows),
            "mean_relative_error": sum(errors) / len(errors),
            "median_relative_error": errors[len(errors) // 2],
            "worst_relative_error": errors[0], "best_relative_error": errors[-1],
            "rows": sorted(rows, key=lambda r: r["relative_error"])}


def derive(run_root: Path, rtl_facts_path: Path, target: str) -> dict[str, Any]:
    """The whole model for one target, from that target's own run evidence and RTL facts."""
    points, skipped = harvest_measured_points(Path(run_root))
    structural, structural_basis = structural_ceiling(Path(rtl_facts_path), target)
    peak = achievable_ceiling(points, provenance=f"measured cycles from {Path(run_root).name}")
    ranked = rank_headroom(points, achievable=peak, structural=structural) if structural else []
    return {
        "schema_version": 1,
        "target": target,
        "evidence_root": str(run_root),
        "points": len(points),
        "skipped": skipped,
        "structural_mac_per_cycle": structural,
        "structural_basis": structural_basis,
        "achievable_mac_per_cycle": (float(peak.value) if peak.known else None),
        "achievable_basis": (peak.provenance if peak.known else peak.reason),
        "achievable_share_of_structural": (
            float(peak.value) / structural if peak.known and structural else None),
        "headroom": [{"capsule": h.point.capsule, "macs": h.point.macs,
                      "measured_cycles": h.point.cycles,
                      "achieved_mac_per_cycle": h.point.achieved_rate,
                      "share_of_achievable": h.share_of_achievable,
                      "share_of_structural": h.share_of_structural,
                      "factor_to_achievable": h.factor_to_achievable} for h in ranked],
        "prediction_error": prediction_error(points, peak),
    }


def render(model: Mapping[str, Any], *, limit: int = 12) -> str:
    """A short report: both ceilings, then the shapes furthest from the achievable one."""
    out = [f"target {model['target']}: {model['points']} measured points"]
    structural = model.get("structural_mac_per_cycle")
    achievable = model.get("achievable_mac_per_cycle")
    out.append(f"  structural ceiling {structural} mac/cycle  ({model.get('structural_basis')})")
    if achievable is None:
        out.append(f"  achievable ceiling UNAVAILABLE: {model.get('achievable_basis')}")
        return "\n".join(out)
    share = model.get("achievable_share_of_structural") or 0.0
    out.append(f"  achievable ceiling {achievable:.2f} mac/cycle  "
               f"({100 * share:.1f}% of structural -- the rest is not reachable by any measured program)")
    error = model.get("prediction_error") or {}
    if error.get("status") == "measured":
        out.append(f"  pricing work against that single rate is wrong by "
                   f"{100 * error['mean_relative_error']:+.1f}% on average "
                   f"(worst {100 * error['worst_relative_error']:+.1f}%), so it ballparks and does "
                   f"not replace the oracle")
    out.append(f"  furthest from the achievable ceiling:")
    for row in (model.get("headroom") or [])[:limit]:
        out.append(f"    {row['capsule'][:34]:34} {row['achieved_mac_per_cycle']:7.2f} mac/cyc "
                   f"{100 * row['share_of_achievable']:6.1f}% of achievable "
                   f"-> {row['factor_to_achievable']:5.1f}x headroom")
    return "\n".join(out)

# --- overlap: the falsifier this archetype's family contract is written against -------------------

def overlap_observation(label: str, counter_overlap: Mapping[str, Any], *,
                        work: str | None = None) -> Any:
    """Turn this target's OWN hardware counter reading into a falsifier observation.

    **The denominator is a deliberate choice, recorded here.** ``merlin.perf.falsifier`` documents
    ``available_cycles`` as the second-largest group's busy count -- the ceiling on overlap between
    the busiest PAIR. ``merlin.perf.hw_counters`` instead bounds the cycles in which two or more
    engines could have been busy at once, ``min(total_busy - busiest, total_busy // 2)``. On a
    two-engine machine they agree; on gemmini's three (EX/LD/ST) they do not: the measured probe has
    busy 84/53/43, so the pair bound is 53 while the >=2-busy bound is 90, and eta is 0.28 or 0.17
    depending on which is used.

    This passes through whatever the counter module computed, because that is the convention the
    measured artifact on disk already uses, and because a verdict is a COMPARISON -- the convention
    cancels between the two arms so long as both are read the same way. What must never happen is
    the two conventions being mixed across arms, which is why this is the single entry point.
    """
    from merlin.perf.falsifier import ENGINE_AXIS, EtaObservation  # noqa: PLC0415

    realised = counter_overlap.get("realised_cycles")
    available = counter_overlap.get("available_cycles")
    busy = dict(counter_overlap.get("busy_cycles") or {})
    engines = tuple(counter_overlap.get("engines") or sorted(busy))
    detail = (f"hardware counters over {counter_overlap.get('measurement_cycles')} cycles; "
              f"available_cycles is the >=2-busy bound from merlin.perf.hw_counters")
    if not isinstance(realised, int) or not isinstance(available, int) or available <= 0:
        return EtaObservation(label=label, realised_cycles=None, available_cycles=None,
                              engines=engines, busy=busy, axis=ENGINE_AXIS, work=work,
                              detail="the counter reading carries no usable overlap denominator")
    return EtaObservation(label=label, realised_cycles=realised, available_cycles=available,
                          engines=engines, busy=busy,
                          sampled_cycles=int(counter_overlap.get("measurement_cycles") or 0),
                          axis=ENGINE_AXIS, work=work, detail=detail)


def overlap_verdict(baseline: Any, candidate: Any, *, bit_exact: bool | None,
                    invariants_held: bool | None = None) -> dict[str, Any]:
    """Did the change actually buy overlap, or merely survive the hardware?

    Delegates to :func:`merlin.perf.falsifier.ab_decision`. Bit-exactness alone is explicitly NOT
    evidence: on an interlocked machine a reordering is correct by construction, so preserving the
    answer says nothing about whether it bought anything. ACCEPT requires bit-exactness, held
    invariants, AND a risen eta; anything unestablished is UNDETERMINABLE rather than a pass.
    """
    from merlin.perf.falsifier import ab_decision  # noqa: PLC0415

    decision = ab_decision(baseline, candidate, bit_exact=bit_exact,
                           invariants_held=invariants_held)
    return {"state": getattr(decision, "state", None),
            "reason": getattr(decision, "reason", ""),
            "baseline_eta": baseline.eta if hasattr(baseline, "eta") else None,
            "candidate_eta": candidate.eta if hasattr(candidate, "eta") else None}


def _composed_detail(c: Any) -> dict[str, Any]:
    """Everything a composed bound knows about itself, so a refusal can be read, not guessed."""
    return {"cycles": getattr(c, "cycles", None),
            "partial_cycles": getattr(c, "partial_cycles", None),
            "floor_cycles": getattr(c, "floor_cycles", None),
            "operator": getattr(getattr(c, "operator", None), "name", None),
            "eta": getattr(c, "eta", None),
            "overlap_saving": getattr(c, "overlap_saving", None),
            "unresolved": sorted(getattr(c, "unresolved", ()) or ()),
            "workload_fixed_cycles": getattr(c, "workload_fixed_cycles", None),
            "serial_fixed_cycles": getattr(c, "serial_fixed_cycles", None),
            "clamped_to_floor": getattr(c, "clamped_to_floor", None)}


def _cancellation_proof(a: Any, b: Any, demands_a: Mapping[str, Any] | None,
                        demands_b: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Per unresolved resource: the work each side asks of it, and whether the unknown cancels.

    This is the whole basis of an ordering verdict. Two schedules can be ordered without pricing
    either one ONLY where they leave the same resources unresolved AND ask each of them for the same
    work -- then whatever those resources cost, they cost the same on both sides and drop out of the
    difference. Reporting the per-resource evidence makes a refusal diagnosable instead of a verdict
    the reader has to take on faith.
    """
    def amount(demands: Mapping[str, Any] | None, name: str) -> Any:
        entry = (demands or {}).get(name)
        return getattr(entry, "amount", entry)

    rows = []
    for name in sorted(set(getattr(a, "unresolved", ()) or ()) | set(getattr(b, "unresolved", ()) or ())):
        left, right = amount(demands_a, name), amount(demands_b, name)
        rows.append({"resource": name, "demand_a": left, "demand_b": right,
                     "stated_on_both": left is not None and right is not None,
                     "cancels": left is not None and right is not None and left == right})
    return rows


def schedule_ordering(a: Any, b: Any, *, demands_a: Mapping[str, Any] | None = None,
                      demands_b: Mapping[str, Any] | None = None,
                      label_a: str = "a", label_b: str = "b") -> dict[str, Any]:
    """Rank two composed schedules without pricing either -- WHERE THE OPERATOR ALLOWS IT.

    ⚠️ Measured on gemmini 2026-09-03: this REFUSES on this target. The machine's composition is
    PARTIAL (measured eta 0.1667 over EX/LD/ST), and ``merlin.perf.differential`` refuses a partial
    operator by construction: "a partial-overlap operator credits pairs, so neither the magnitude nor
    the ordering of a resolved-part difference survives; refusing rather than approximating." It
    returns EXACT deltas under SUM and ordering under MAX, so it is the right tool for an in-order or
    a fully-overlapped machine and the wrong one here.

    The report carries the whole basis rather than a bare verdict: each side's composed internals, the
    separate ``comparable`` diagnosis, and the per-resource cancellation proof -- so a refusal says
    WHICH resource broke it and why, and an acceptance shows what was allowed to drop out.
    """
    from merlin.perf.differential import REFUSED, comparable, compare  # noqa: PLC0415

    ok, why = comparable(a, b, demands_a=demands_a, demands_b=demands_b)
    result = compare(a, b, demands_a=demands_a, demands_b=demands_b,
                     label_a=label_a, label_b=label_b)
    return {"basis": result.basis, "faster": result.faster,
            "delta_cycles": result.delta_cycles, "reason": result.reason,
            "usable": result.basis != REFUSED,
            "comparable": ok, "comparable_reason": why,
            "cancelled": list(getattr(result, "cancelled", ()) or ()),
            "cancellation_proof": _cancellation_proof(a, b, demands_a, demands_b),
            label_a: _composed_detail(a), label_b: _composed_detail(b)}


def rank_candidates(candidates: Mapping[str, Any], *,
                    demands: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Order candidates best-first, and report every pair that could NOT be compared.

    ``rank_schedules`` deliberately keeps an incomparable candidate in the ranking rather than
    dropping it: a candidate excluded for want of evidence is a hole in the search, not an answer
    about the candidate. The refusals are returned beside the order for exactly that reason.
    """
    from merlin.perf.differential import rank_schedules  # noqa: PLC0415

    if not candidates:
        return {"status": "unavailable", "reason": "no candidate was composed",
                "order": [], "refusals": []}
    order, refusals = rank_schedules(candidates, demands=demands)
    return {"status": "ranked", "order": list(order),
            "refusals": [{"reason": r.reason, "basis": r.basis} for r in refusals],
            "fully_comparable": not refusals}


def compare_per_engine(a: Mapping[str, Any], b: Mapping[str, Any], *,
                       demands_a: Mapping[str, Mapping[str, Any]] | None = None,
                       demands_b: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Order two schedules ENGINE BY ENGINE, so a trade is visible instead of averaged away.

    A single scalar verdict hides the case that matters most to a compiler: a change that speeds one
    engine while slowing another. ``compare_by_engine`` reports each engine separately, names the
    engines it could not decide, and flags a trade -- which is a different decision for the author
    than a uniform win.
    """
    from merlin.perf.differential import compare_by_engine  # noqa: PLC0415

    if not a or not b:
        return {"status": "unavailable", "reason": "one side composed no engine"}
    v = compare_by_engine(a, b, demands_a=demands_a, demands_b=demands_b)
    return {"status": "compared", "faster": v.faster, "basis": v.basis, "reason": v.reason,
            "total_delta_cycles": v.total_delta_cycles,
            "undecided_engines": list(v.undecided_engines or ()),
            "traded": bool(v.traded),
            "per_engine": {k: {"basis": c.basis, "faster": c.faster,
                               "delta_cycles": c.delta_cycles, "reason": c.reason}
                           for k, c in (v.per_engine or {}).items()}}


# --- what an oracle query COSTS, and whether this shape may be asked ------------------------------

def fit_oracle_cost_law(run_root: Path, *, substrate: str) -> Any:
    """Fit ``seconds = a + b*cycles + c*words`` from timing a completed run already recorded.

    Uses :func:`merlin.perf.oracle_cost.fit_cost_law`, which separates the terms by construction and
    marks any it could not isolate. With no LOAD probe ladder the per-word term stays UNKNOWN and
    every estimate is flagged a lower bound -- which is the honest state, not a defect: nothing here
    varied program size independently of cycles.

    Fitted on this target 2026-09-03 over 170 points spanning 161..28,118 cycles:
    ``12.89 s + 4.60 ms/cycle`` (~217 simulated cycles/s). An earlier hand fit over only the four
    perf capsules -- a 2x cycle range -- gave 9.30 ms/cycle and over-predicted the largest measured
    shape by 2x. Fitting a cost law on a narrow range and extrapolating is the error this replaces.
    """
    from merlin.perf.oracle_cost import CostSample, ProbeKind, fit_cost_law  # noqa: PLC0415

    samples = []
    for result in Path(run_root).rglob("capsule_result.json"):
        try:
            tier = ((json.loads(result.read_text(encoding="utf-8")).get("tiers") or {})
                    .get("L3") or {})
        except Exception:  # noqa: BLE001
            continue
        cycles, seconds = tier.get("cycles"), (tier.get("timing") or {}).get("sim_active_s")
        if (isinstance(cycles, int) and not isinstance(cycles, bool) and cycles > 0
                and isinstance(seconds, (int, float)) and seconds > 0):
            samples.append(CostSample(seconds=float(seconds), cycles=int(cycles), words=0,
                                      concurrency=1, kind=ProbeKind.CORPUS,
                                      label=result.parent.name))
    if not samples:
        return None
    return fit_cost_law(samples, substrate=substrate)


def oracle_affordability(law: Any, *, predicted_cycles: float,
                         budget_seconds: float, program_words: int = 0) -> dict[str, Any]:
    """Would asking the cycle oracle about this shape cost more than the budget allows?

    This is the gate that keeps a large shape out of the expensive tier BEFORE the tier is spent,
    rather than discovering the cost by paying it. `merlin.perf.preflight` already computes a
    per-tier wall estimate; what has never existed is a refusal that reads it, so this supplies one.

    The estimate is a LOWER bound whenever a term could not be isolated, so an affordable verdict on
    a lower bound is deliberately conservative in the wrong direction -- it is reported alongside,
    never hidden.
    """
    if law is None:
        return {"status": "undeterminable", "reason": "no oracle cost law could be fitted",
                "seconds": None, "affordable": None}
    estimate = law.estimate(int(max(0, predicted_cycles)), int(program_words))
    seconds = float(estimate.seconds)
    return {
        "status": "measured",
        "seconds": seconds,
        "budget_seconds": float(budget_seconds),
        "affordable": seconds <= float(budget_seconds),
        "is_lower_bound": bool(getattr(estimate, "is_lower_bound", False)),
        "excluded_terms": list(getattr(estimate, "excluded", ()) or ()),
        "reason": (f"the cycle oracle would need ~{seconds:,.0f} s for ~{predicted_cycles:,.0f} "
                   f"cycles against a {budget_seconds:,.0f} s budget"),
    }


# --- activity: the one type five more modules are waiting on --------------------------------------

def activity_source_from_counters(workload: str, overlap: Mapping[str, Any],
                                  kinds: Mapping[str, str], *,
                                  provenance: str = "") -> Any:
    """Build the per-resource activity record that the corpus-level analyses all consume.

    ``merlin.perf.attribution``, ``composer``, ``workload_roles`` and ``roofline`` every one take an
    :class:`~merlin.perf.decompose.ActivitySource`, and nothing in the performance experiment built
    one -- which is the single reason those four modules were unreachable.

    ``kinds`` maps each engine to its role and is DERIVED, never declared: the counter artifact
    records it under ``resource_role_binding`` with the probe method that established it. An engine
    with no derived role becomes OTHER rather than being guessed into compute or movement.

    ``partitioned=False`` is load-bearing. These counters have a per-combination layout (EX, EX+LD,
    EX+LD+ST, ...), so they can see two engines busy in the same cycle. A PARTITIONED source charges
    every cycle to exactly one owner and therefore reports zero overlap by construction, and
    ``headroom.composition_operator`` refuses such a source as overlap evidence -- correctly.
    """
    from merlin.perf.decompose import ActivitySource, Resource, ResourceKind  # noqa: PLC0415

    by_name = {"compute": ResourceKind.COMPUTE, "movement": ResourceKind.MOVEMENT,
               "fixed": ResourceKind.FIXED}
    busy = dict(overlap.get("busy_cycles") or {})
    resources = tuple(
        Resource(name=engine, kind=by_name.get(str(kinds.get(engine, "")).lower(),
                                               ResourceKind.OTHER),
                 busy_cycles=int(cycles))
        for engine, cycles in sorted(busy.items()))
    total = int(overlap.get("measurement_cycles") or 0)
    return ActivitySource(workload=workload, total_cycles=total, resources=resources,
                          partitioned=False, completion_observable=None,
                          provenance=provenance or "hardware counters, per-combination layout")


def classify_roles(sources: Sequence[Any]) -> dict[str, Any]:
    """Which workloads have a lever worth pulling, and which are only good for calibration.

    :func:`merlin.perf.workload_roles.classify_workloads` needs the WHOLE corpus, because two of its
    rules are corpus-relative: the modal binding kind, and the smallest resolvable activity quantum
    that sets the no-lever floor. A workload with too little headroom is NO_LEVER; one whose activity
    isolates a single term is CALIBRATION; the rest are OPTIMIZE.
    """
    from merlin.perf.workload_roles import classify_workloads  # noqa: PLC0415

    if not sources:
        return {"status": "unavailable", "reason": "no activity source was built", "roles": {}}
    split = classify_workloads(sources)
    entries = split.roles.values() if hasattr(split.roles, "values") else split.roles
    rows = [{"workload": r.workload, "role": r.role.name, "binding": r.binding,
             "binding_kind": str(r.binding_kind), "binding_share": r.binding_share,
             "headroom_cycles": r.headroom_cycles, "headroom_share": r.headroom_share,
             "rule": r.rule} for r in entries]
    return {"status": "classified",
            "modal_binding_kind": str(split.modal_binding_kind),
            "quantum_cycles": split.quantum_cycles,
            "headroom_floor_cycles": split.headroom_floor_cycles,
            "unavailable": [str(u) for u in (split.unavailable or ())],
            "by_role": {role: [r["workload"] for r in rows if r["role"] == role]
                        for role in ("OPTIMIZE", "CALIBRATION", "NO_LEVER", "UNKNOWN")},
            "rows": rows}


def attribute_gap(source: Any, *, envelope: Any = None) -> dict[str, Any]:
    """Where every measured cycle went, and how much of each bucket the structure explains.

    :func:`merlin.perf.attribution.attribute` splits the measured time into compute / dma / stall /
    control / host plus an always-emitted RESIDUAL, and -- given a structural envelope -- reports the
    gap between what each bucket cost and what it structurally had to cost. Without an envelope the
    structural side is UNKNOWN rather than zero, so a missing bound never reads as a closed gap.
    """
    from merlin.perf.attribution import attribute, buckets_from_kinds  # noqa: PLC0415

    kinds = {r.name: r.kind for r in source.resources}
    buckets = buckets_from_kinds(kinds, fixed_bucket="control", other_bucket="host")
    result = attribute(source, buckets=buckets, envelope=envelope)
    return {"workload": getattr(result, "workload", source.workload),
            "components": [{"bucket": c.bucket, "measured_cycles": c.measured_cycles,
                            "structural_cycles": c.structural_cycles,
                            "gap_cycles": c.gap_cycles,
                            "family": str(getattr(c, "family", "")),
                            "evidence_kind": getattr(c, "evidence_kind", "")}
                           for c in getattr(result, "components", ())]}


# --- schedulability: may this shape be asked at all, and at what price -----------------------------

def machine_budget(rtl_facts_path: Path, target: str) -> Any:
    """The machine limits a workload is checked against, derived from the target's own RTL.

    ``preflight.MachineBudget.from_machine_facts`` refuses a target that ships no ISA definition --
    it is built for the self-hosted-ISA archetype, and a command-buffer target legitimately has none
    ("'<target>' ships no ISA definition; no kernel can be encoded for it"). The budget itself is
    just the machine's limits, so it is derived here from the array geometry and datapath widths the
    facts already carry, and the limits nothing publishes stay ``None``.

    ``None`` is not "no limit": :mod:`merlin.perf.preflight` turns each one into an UNCHECKED refusal
    rather than a pass, which is the whole point -- a DRAM window nobody published cannot be checked,
    and a program longer than IMEM is not rejected by the device, it silently runs its prefix.
    """
    from merlin.perf.preflight import MachineBudget  # noqa: PLC0415
    from merlin.perf.workload_gen import tile_geometry  # noqa: PLC0415

    facts = json.loads(Path(rtl_facts_path).read_text(encoding="utf-8")).get("facts") or {}
    geometry = tile_geometry(target)
    dtypes = {str(d.get("name")): str(d.get("dtype") or "") for d in facts.get("datapaths") or []}

    def _bytes(dtype: str, fallback: int) -> int:
        digits = "".join(c for c in dtype if c.isdigit())
        return max(1, int(digits) // 8) if digits else fallback

    return MachineBudget(
        tile_rows=geometry.rows, tile_cols=geometry.cols,
        operand_bytes=_bytes(dtypes.get("input", ""), 1),
        accum_bytes=_bytes(dtypes.get("accumulator", ""), 4),
        dram_window=None, imem_words=None, dram_base=0,
        provenance=f"rtl facts: arrays geometry ({geometry.source}) + datapath dtypes")


def tile_pass_rate(points: Sequence[MeasuredPoint], *, budget: Any) -> Any:
    """Cycles per tile pass, fitted from measured shapes rather than assumed.

    ``rate_from_observations`` needs at least two DISTINCT pass counts to separate a slope from an
    intercept; one distinct count yields a single-point extrapolation that says so, and none yields
    UNKNOWN. It never returns a default rate.
    """
    from merlin.perf.preflight import rate_from_observations  # noqa: PLC0415

    tile_elements = budget.tile_rows * budget.tile_cols
    observations = [(max(1, p.macs // max(1, tile_elements)), p.cycles) for p in points]
    return rate_from_observations(observations, note="fitted from harvested (work, cycles) points")


def preflight_shape(name: str, *, m: int, k: int, n: int, budget: Any, rate: Any = None,
                    laws: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Decide, before any oracle runs, whether this shape may be scheduled and what it would cost."""
    from merlin.perf.preflight import preflight_matmul  # noqa: PLC0415

    pf = preflight_matmul(name, m=m, k=k, n=n, budget=budget, rate=rate, laws=dict(laws or {}))
    return {"workload": name, "ok": bool(pf.ok), "tile_passes": pf.tile_passes,
            "projected_cycles": pf.projected_cycles,
            "useful_bytes": pf.useful_bytes, "footprint_bytes": pf.footprint_bytes,
            "refusals": [{"code": r.code, "detail": r.detail} for r in pf.refusals],
            "wall_seconds": {tier: getattr(est, "seconds", None)
                             for tier, est in (pf.wall or {}).items()}}


def counter_availability(target: str) -> dict[str, Any]:
    """Whether this target counts its own engine occupancy, and on what evidence.

    ``counters_for_target`` may only report ``absent`` after reading a real C header; anything else
    is ``unavailable``, so "this machine has no counters" is never inferred from failing to find one.
    """
    from merlin.perf.hw_counters import counters_for_target  # noqa: PLC0415

    try:
        record = counters_for_target(target)
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}
    return {"status": record.get("status"),
            "reason": str(record.get("reason") or record.get("detail")
                          or f"counters_for_target reported {record.get('status')!r}"),
            "header": record.get("header"), "header_sha256": record.get("header_sha256"),
            "event_codes": sorted((record.get("event_codes") or {}))[:12]}


def falsifier_evidence(identity: Any, baseline: Any, candidate: Any, *, bit_exact: bool | None,
                       negative_control: bool, invariants_held: bool | None = None) -> Any:
    """Hand the overlap verdict to the campaign record in the shape it already expects."""
    from merlin.perf.campaign import FalsifierEvidence  # noqa: PLC0415
    from merlin.perf.falsifier import ab_decision  # noqa: PLC0415

    decision = ab_decision(baseline, candidate, bit_exact=bit_exact,
                           invariants_held=invariants_held)
    return FalsifierEvidence.from_ab_decision(identity, decision,
                                              negative_control=negative_control)


# --- the capability report: every analysis module, invoked, with what it could establish -----------

class NotApplicable:
    """An analysis the target's own facts say does not apply here.

    Distinct from both outcomes it sits between: it is not a capability gap (nothing is missing) and
    it is not an established result (nothing was computed). Counting it as "derived" is the failure
    mode this repo has hit repeatedly -- a check that could not run reporting success -- so it gets
    its own status and its own column in the summary.
    """

    __slots__ = ("reason",)

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)

    def __str__(self) -> str:
        return self.reason


def _try(fn) -> dict[str, Any]:
    """Invoke one analysis and record what it established, or exactly why it could not.

    A module that cannot run on this target is not a hole in the tooling -- several are built for a
    different archetype and say so precisely. What WOULD be a hole is not knowing which, so every
    one is called and every refusal is captured with the module's own words.
    """
    try:
        value = fn()
    except Exception as exc:  # noqa: BLE001 - the refusal is the result here
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {str(exc)[:200]}"}
    if isinstance(value, NotApplicable):
        return {"status": "not_applicable", "reason": value.reason}
    return {"status": "derived", "value": value}


def capability_report(target: str, *, rtl_facts_path: Path | None = None,
                      run_root: Path | None = None,
                      interface_mlir: Path | None = None,
                      capsules: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Call every performance analysis this repo has, and report what each one can establish here.

    This exists because "is the tooling used?" was answerable only by reading imports. Now it is a
    measurement: each analysis is invoked against this target's own sources and reports ``derived``
    with a value, or ``unavailable`` with the reason its own module gave.

    Three refusals are expected and correct rather than defects, and naming them is the point:
    a command-buffer target ships no ISA definition, so the schedule-dependence analyses built for a
    self-hosted ISA cannot encode a kernel for it; a target with no vector unit has no vector term;
    and the SIMT analyses are for a SIMT cluster. Each says so in its own words.
    """
    from merlin.perf.headroom import Composition as _COMPOSITION  # noqa: PLC0415
    from merlin.perf import (bus_beat_probe as BUS, calibration as CAL,  # noqa: PLC0415
                             calibration_plan as CPLAN, command_stream_gen as CSG,
                             comparand as CMP, composer as COMP, counter_binding as CBIND,
                             depgraph as DG, handshake as HS, harvest as HV,
                             receipt_bridge as RB, roofline as RF, simt_occupancy as SIMT,
                             vector_cycles as VC, workload_gen as WG)

    facts_path = Path(rtl_facts_path) if rtl_facts_path else None
    report: dict[str, Any] = {"target": target, "analyses": {}}
    A = report["analyses"]

    A["tile_geometry"] = _try(lambda: str(WG.tile_geometry(target)))
    A["structural_ceiling"] = _try(lambda: structural_ceiling(facts_path, target))
    A["counters"] = _try(lambda: counter_availability(target))
    A["fill_drain_depth"] = _try(lambda: str(HS.measure_fill_depth(target)))
    # Asked through the derived compute units, so a systolic-only design is told it has no
    # vector unit instead of being handed an external manifest error about lane width.
    def _vector() -> Any:
        report = vector_term_for(target)
        if report["status"] == "not_applicable":
            return NotApplicable(report["reason"])
        if report["status"] != "derived":
            raise ValueError(str(report.get("reason"))[:160])
        return report["terms"]
    A["vector_term"] = _try(_vector)
    A["simt_geometry"] = _try(lambda: SIMT.geometry_for_target(target).get("status"))
    # The dependence analysis needs a machine, an op selection, a measured control-flow probe and a
    # measured settle -- each of which refuses on its own terms when absent. Called through the real
    # signature so the reported reason is the analysis's, not a mistake in this probe.
    def _dependence() -> str:
        facts = WG.machine_facts(target)
        ops = WG.candidate_ops(facts.isa)
        plan = WG.plan_matmul(facts, ops, m=16, k=16, n=16,
                              control_flow=WG.probe_control_flow(facts),
                              settle=WG.probe_settle(facts))
        return f"program of {len(plan.words)} words over {plan.tiles} tiles"
    A["schedule_dependence"] = _try(_dependence)
    A["comparand_groups"] = _try(lambda: CMP.declared_groups(list(capsules)))
    # Every one of these is INVOKED. Reading a signature would prove the module imports, which is
    # not the question -- the question is what it can establish here, and only calling it answers that.
    hw_text = ""
    if facts_path is not None:
        source = (json.loads(facts_path.read_text(encoding="utf-8")).get("facts") or {}).get("source") or {}
        candidate = Path(str(source.get("hw_mlir") or source.get("fir_path") or ""))
        if candidate.is_file():
            hw_text = candidate.read_text(encoding="utf-8", errors="replace")
    raw_facts = (json.loads(facts_path.read_text(encoding="utf-8"))
                 if facts_path is not None and facts_path.is_file() else {})
    A["counter_binding"] = _try(lambda: sorted(CBIND.extract_external_additive_counters(
        hw_text, "", top_module="", counter_module="", counter_file_module="",
        external_port_prefix="", external_base_define="", declared_unit="cycles")))
    A["bus_beat_monitors"] = _try(lambda: sorted(BUS.derive_counter_beat_monitors(hw_text, {})))
    # `build` REPORTS its problems rather than raising, so a bare call always "succeeds"; the issue
    # count is the verdict, and a non-zero one is an unavailable, not a pass.
    def _bridge() -> str:
        issues = RB.build(Path("/unreached"), Path("/unreached"), Path("/unreached"),
                          Path("/unreached"))[-1]
        if issues:
            raise ValueError(f"{issues} input(s) could not be loaded")
        return "built"
    A["receipt_bridge"] = _try(_bridge)
    A["harvest_authority"] = _try(lambda: HV.MeasurementAuthority(target=target).cycles_tier)
    A["calibration_plan"] = _try(
        lambda: CPLAN.build_calibration_plan_from_rtl(raw_facts, {}).__class__.__name__)
    A["mechanism_calibration"] = _try(
        lambda: CAL.calibrate(target=target, contract={}, traces=()).get("ran_against_traces"))
    A["corpus_composition"] = _try(lambda: COMP.compose_corpus(
        [], times={}, operator=_COMPOSITION.PARTIAL, eta=0.0).__class__.__name__)
    def _roofline() -> Any:
        if run_root is None:
            raise ValueError("no completed run was supplied to harvest observations from")
        from merlin.perf.headroom import Composition  # noqa: PLC0415
        report = empirical_roofline_report(
            Path(run_root), operand_bytes=1, composition=Composition.PARTIAL,
            composition_eta=0.16666666666666666,
            composition_provenance="measured hardware counters (EX/LD/ST)")
        if report.get("status") != "derived":
            raise ValueError(str(report.get("reason"))[:160])
        return (f"{report['resolved']}/{report['expected']} resolved, "
                f"share {report['cycle_weighted_resolved_share']:.2f}")
    A["empirical_roofline"] = _try(_roofline)
    if interface_mlir is not None:
        A["reorder_pair"] = _try(
            lambda: sorted(CSG.pair_from_interface(Path(interface_mlir).read_text(encoding="utf-8"))))
    if run_root is not None:
        A["measured_points"] = _try(lambda: len(harvest_measured_points(Path(run_root))[0]))
        A["oracle_cost_law"] = _try(
            lambda: getattr(fit_oracle_cost_law(Path(run_root), substrate="gsim").per_cycle, "value", None))

    derived = sum(1 for v in A.values() if v["status"] == "derived")
    not_applicable = sum(1 for v in A.values() if v["status"] == "not_applicable")
    report["summary"] = {"analyses": len(A), "derived": derived,
                         "not_applicable": not_applicable,
                         "unavailable": len(A) - derived - not_applicable,
                         "applicable": len(A) - not_applicable}
    return report


def render_capabilities(report: Mapping[str, Any]) -> str:
    """One line per analysis: what it established here, or the reason it could not."""
    summary = report["summary"]
    head = (f"performance analyses for {report['target']}: "
            f"{summary['derived']}/{summary['applicable']} established")
    if summary.get("not_applicable"):
        head += f" ({summary['not_applicable']} not applicable to this target)"
    out = [head]
    marks = {"derived": "[ok]", "not_applicable": "[n/a]", "unavailable": "[--]"}
    for name, row in sorted((report.get("analyses") or {}).items()):
        mark = marks.get(row["status"], "[??]")
        detail = str(row["value"])[:88] if row["status"] == "derived" else row["reason"][:88]
        out.append(f"  {mark:5} {name:24} {detail}")
    return "\n".join(out)


# --- the ISA a RoCC target derives from its own decoder --------------------------------------------

def isa_model_from_rocc_facts(target: str, rtl_facts_path: Path) -> Any:
    """This target's ISA, derived from its own decode table -- see the library implementation.

    Kept as a thin path-taking wrapper because the derivation itself belongs to every consumer of an
    IsaModel, not to the performance experiment: it lives in
    :func:`merlin.targetgen.isa_model.isa_model_from_rocc_facts`.
    """
    from merlin.targetgen.isa_model import isa_model_from_rocc_facts as _derive  # noqa: PLC0415

    return _derive(target, json.loads(Path(rtl_facts_path).read_text(encoding="utf-8")))


# --- traffic: what the emitted program actually moved, from its own decoded trace -----------------

def moved_elements_from_trace(trace: Mapping[str, Any]) -> tuple[int, int]:
    """(elements moved, instructions that declared a movement) from a decoded instruction trace.

    Generic by construction: an instruction MOVES ``rows x cols`` elements exactly when its decoded
    block states both, so nothing here matches a mnemonic, an opcode or a class name. A target whose
    decoder names its transfers differently is counted the same way, and an instruction that declares
    no extent contributes nothing rather than a guess.

    Elements, not bytes: the width belongs to the datapath the transfer touches, and a trace does not
    say which. :func:`traffic_demand` applies the operand width and says so.
    """
    elements = counted = 0
    for instruction in (trace.get("instructions") or ()):
        decoded = instruction.get("decoded") if isinstance(instruction, dict) else None
        if not isinstance(decoded, dict):
            continue
        rows, cols = decoded.get("rows"), decoded.get("cols")
        if all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (rows, cols)):
            elements += rows * cols
            counted += 1
    return elements, counted


def traffic_demand(trace: Mapping[str, Any], *, operand_bytes: int) -> Any:
    """The movement demand of one program, priced at the operand datapath width.

    A single aggregate: the trace states extents but not which datapath each transfer crossed, so
    splitting reads from writes would require assuming a direction per mnemonic -- exactly the
    name-matching this avoids. One honest number beats two invented ones.
    """
    from merlin.perf.decompose import ResourceKind  # noqa: PLC0415
    from merlin.perf.envelope import Basis, ResourceDemand  # noqa: PLC0415

    elements, counted = moved_elements_from_trace(trace)
    return ResourceDemand("movement", ResourceKind.MOVEMENT, elements * max(1, operand_bytes),
                          "bytes", Basis.MOVED, None,
                          f"{counted} decoded transfers declaring rows x cols, "
                          f"at {operand_bytes} B per element")


def empirical_roofline_report(run_root: Path, *, operand_bytes: int,
                              composition: Any = None, composition_eta: float | None = None,
                              composition_provenance: str = "") -> dict[str, Any]:
    """The measured roofline for a completed run: per workload, its bound and WHAT LIMITS IT.

    ``merlin.perf.roofline`` admits only an achievable ceiling (``n_samples >= 4`` and
    ``is_ceiling``), so a nameplate peak cannot enter here. Everything it needs is recovered from
    evidence the run already produced: cycles and work from the harvest, traffic from each program's
    own decoded trace, and both ceilings fitted across the corpus.

    Two inputs are refusals rather than defaults, and both say so precisely if absent: an explicitly
    MEASURED composition (operator + eta -- assuming ``max`` would derive overlap from nothing), and
    an explicit fixed-term set per workload, where an empty sequence means "measured zero terms" and
    an absent entry means UNKNOWN.

    The limiter is the payoff. "This shape is far from the ceiling" is a gap; "this shape is bound by
    MOVEMENT" is a lever. Measured on gemmini: every conv2d capsule resolves movement-bound at 0.076
    efficiency, which is a different instruction to the compiler than the compute-bound shapes.
    """
    from merlin.perf.decompose import ResourceKind  # noqa: PLC0415
    from merlin.perf.envelope import Basis, Peak, ResourceDemand  # noqa: PLC0415
    from merlin.perf.roofline import (EmpiricalObservation, EvidenceReceipt,  # noqa: PLC0415
                                      empirical_roofline)

    points, _skipped = harvest_measured_points(Path(run_root))
    observations, receipts, traffic, fixed = [], {}, [], {}
    sample_ids = tuple(p.capsule for p in points[:4])
    for point in points:
        trace_path = Path(point.source) / "generated" / "instruction_trace.json"
        if not trace_path.is_file():
            continue
        demand = traffic_demand(json.loads(trace_path.read_text(encoding="utf-8")),
                                operand_bytes=operand_bytes)
        if demand.amount <= 0:
            continue
        work = ResourceDemand(COMPUTE, ResourceKind.COMPUTE, point.macs, "mac", Basis.MOVED, None,
                              "compiler command buffer via merlin.perf.work_volume")
        observations.append(EmpiricalObservation(
            point.capsule, point.cycles, work, (demand,),
            "measured cycles from the functional run + traffic from its own decoded trace"))
        traffic.append((demand.amount, point.cycles))
        fixed[point.capsule] = ()          # explicit: zero MEASURED fixed terms, not "unknown"
        for key, kind in ((f"observation:{point.capsule}", "rtl_cycle_measurement"),
                          (f"work:{point.capsule}:{COMPUTE}", "compiler_ir"),
                          (f"traffic:{point.capsule}:movement", "physical_counter"),
                          (f"fixed:{point.capsule}", "calibration_fit")):
            receipts[key] = EvidenceReceipt("0" * 64, kind, sample_ids)
    if not observations:
        return {"status": "unavailable",
                "reason": "no measured point carried both priced work and decoded traffic"}
    compute_peak = achievable_ceiling(points, provenance=f"harvested from {Path(run_root).name}")
    movement_peak = Peak.observed_ceiling("movement", traffic, unit="bytes",
                                          provenance="decoded-trace traffic vs measured cycles")
    receipts[f"peak:{COMPUTE}"] = EvidenceReceipt("2" * 64, "calibration_fit", sample_ids)
    receipts["peak:movement"] = EvidenceReceipt("4" * 64, "calibration_fit", sample_ids)
    receipts["composition"] = EvidenceReceipt("5" * 64, "rtl_counter_partition", sample_ids)
    report = empirical_roofline(
        observations, peaks={COMPUTE: compute_peak, "movement": movement_peak},
        fixed_terms=fixed, evidence_receipts=receipts,
        expected_workloads=tuple(o.workload for o in observations),
        composition=composition, composition_eta=composition_eta,
        composition_provenance=composition_provenance)
    coverage = report.coverage
    rows = [{"workload": name, "measured_cycles": p.measured_cycles,
             "bound_cycles": p.bound_cycles, "efficiency": p.efficiency,
             "limiter": str(p.limiter), "margin_share": p.margin_share}
            for name, p in report.points.items() if isinstance(p.efficiency, (int, float))]
    return {"status": "derived", "observations": len(observations),
            "resolved": len(coverage.resolved), "expected": len(coverage.expected),
            "cycle_weighted_resolved_share": coverage.cycle_weighted_resolved_share,
            "compute_ceiling": float(compute_peak.value) if compute_peak.known else None,
            "movement_ceiling": float(movement_peak.value) if movement_peak.known else None,
            "rows": sorted(rows, key=lambda r: r["efficiency"])}


def compute_units_of(target: str) -> list[tuple[str, str]]:
    """This target's (unit name, unit kind) pairs, derived from its own capability manifest.

    The manifest derives from the RTL facts, so this list follows the hardware: a design that gains a
    vector lane gains the unit here without anything being re-declared, and one that loses it stops
    claiming it. That is what lets a per-unit analysis be ASKED only where the machine has the unit.
    """
    from merlin.targetgen import capability_manifests as CM, compute_units as CU  # noqa: PLC0415

    try:
        return [(u.name, u.kind) for u in CU.compute_units(CM.manifest_for(target))]
    except Exception:  # noqa: BLE001 - a target with no resolvable manifest declares no unit here
        return []


def vector_term_for(target: str, instructions: Sequence[Any] = ()) -> dict[str, Any]:
    """The vector engine's cycle term -- asked only of a target whose RTL evidences a vector unit.

    ``merlin.perf.vector_cycles`` prices a named unit and delegates lane discovery to an external
    pass, which refuses with a MANIFEST complaint ("no 256-bit VPU data row in manifest") when the
    unit is absent. That reads as a broken analysis; it is not. The applicability question belongs to
    the target's own derived compute units, so it is answered here first and the refusal becomes a
    statement about the hardware: a systolic-only design has no vector term to compute.

    Reusable by construction -- no unit name is written down. Measured 2026-09-03: atlas resolves a
    VectorTerm on its derived ``vector_unit`` (lanes=16); gemmini declares only ``systolic_mesh`` and
    is told so, rather than being handed a foreign manifest error.
    """
    from merlin.perf.vector_cycles import vector_term  # noqa: PLC0415

    units = [name for name, kind in compute_units_of(target) if kind == "vector"]
    if not units:
        declared = ", ".join(f"{n} ({k})" for n, k in compute_units_of(target)) or "none"
        return {"status": "not_applicable",
                "reason": (f"{target!r} evidences no vector compute unit in its derived units "
                           f"({declared}), so it has no vector term"),
                "units": []}
    results = {}
    for name in units:
        try:
            term = vector_term(target, list(instructions), unit=name)
            results[name] = {"cycles": term.cycles, "instructions": term.instructions,
                             "complete": term.complete,
                             "unmapped": list(getattr(term, "unmapped", ()) or ()),
                             "provenance": getattr(term, "provenance", "")}
        except Exception as exc:  # noqa: BLE001 - a unit that cannot be priced says why
            results[name] = {"status": "unavailable",
                             "reason": f"{type(exc).__name__}: {str(exc)[:180]}"}
    return {"status": "derived", "units": units, "terms": results}
