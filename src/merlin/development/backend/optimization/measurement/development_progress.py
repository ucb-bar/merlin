"""Measured search progress, bounded prefetching and achievable-rate evidence.

Functions receive the existing feedback/objective explicitly; its lifetime and
independent admission state remain owned by the public class.
"""

from __future__ import annotations

import contextvars
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.development.backend.optimization.contracts import StageGateError
from merlin.development.backend.optimization.measurement import development_execution as execution
from merlin.development.backend.optimization.measurement import feedback_metrics as FM
from merlin.development.backend.optimization.search import corpus as CORPUS


def _refuted_so_far(self, cells: Sequence[Mapping[str, Any]], index: int, total: int) -> bool:
    """Is this candidate already behind on every comparable member measured so far?

    ONE-DIRECTIONAL BY CONSTRUCTION. True only when every comparable cell measured has the
    candidate strictly slower than the baseline. It never returns True on a tie, never on an
    incomparable cell, and never before the minimum prefix -- so the sweep can be cut short only
    for a candidate that has lost everywhere it has been asked, which is the one conclusion no
    further member can overturn.

    Returns False whenever it cannot tell: too few comparable cells, any cell not comparable, or
    the last member (where stopping saves nothing). Absence of evidence never stops a sweep.
    """
    if index + 1 >= total:
        return False  # nothing left to save
    comparable = [c for c in cells if c.get("comparable")]
    if len(comparable) < self.MINIMUM_REFUTING_PREFIX or len(comparable) != len(cells):
        return False  # an incomparable member means the picture is partial
    for cell in comparable:
        delta = cell.get("candidate_minus_baseline_cycles")
        if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta <= 0:
            return False  # a tie or a win anywhere: keep measuring
    return True


def _objective_identities(self) -> tuple[set[tuple[str, str]], int]:
    """``({(family, capsule) that carry the objective}, how many declared nothing)``.

    Fails OPEN on an undeclared member and says how many there were, rather than failing closed.
    A frozen corpus is content-addressed and may predate the declaration entirely; refusing it
    would turn a schema addition into a campaign-stopping error, and dropping its members from the
    total would quietly shrink the objective instead.
    """
    objective: set[tuple[str, str]] = set()
    undeclared = 0
    # No corpus at all is the same situation one level up: nothing declares a class, so nothing
    # may be excluded from the total. Returning an empty set makes the caller fall back to every
    # comparable member and say so, which is the honest reading.
    for member in getattr(self.corpus, "capsules", None) or ():
        declared = ((member.descriptor or {}).get("performance") or {}).get("member_class")
        if not isinstance(declared, str) or not declared:
            undeclared += 1
            objective.add((member.family, member.capsule))
            continue
        if declared == self.OBJECTIVE_CLASS:
            objective.add((member.family, member.capsule))
    return objective, undeclared


def _stopping(self, cells: Sequence[Mapping[str, Any]], *, label: str, elapsed_s: float) -> dict[str, Any]:
    """Should the search stop? Every condition's answer, fired or not.

    Delegated wholesale to :mod:`merlin.analysis.perf.search.selection`, whose declared thresholds are exactly the
    question worth asking: has the search stopped moving (three consecutive queries improving
    less than 1%), and has it got close enough to what is actually attainable (within 10% of the
    achievable target)? A plateau alone is a weak signal -- a search can sit still far from the
    ceiling because the last few levers were bad -- so it is reported beside attainment rather
    than instead of it.

    The attainable target is the ACHIEVABLE ceiling, never the structural peak: stopping at 90%
    of a rate no program on this machine has ever reached would never fire, and stopping short
    because a nameplate number says there is headroom would never stop.
    """
    from merlin.analysis.perf.search import selection as SELECT  # noqa: PLC0415
    from merlin.analysis.perf.search.budget import Budget, unpriced_channel  # noqa: PLC0415

    comparable = [c for c in cells if c.get("comparable")]
    if not comparable:
        return {
            "status": "undeterminable",
            "reason": "no member is comparable, so there is no measured total to judge",
            "verdicts": [],
        }
    # THE OBJECTIVE IS SUMMED OVER THE MEMBERS THAT ARE THE OBJECTIVE, which had never been true.
    # Every comparable cell was summed, so members that exist only to fit a LAW about the machine
    # -- a reduction-depth cohort, a 2-D law over M and N -- weighed exactly as much as the
    # workload the search is meant to improve. Measured on a real campaign: the parallel-extents
    # family is 16 of 38 members and 33% of a 36.9-minute sweep, for members worth under 2% of the
    # recoverable cycles, and the search converged at 0.2%.
    #
    # The class is read from the FROZEN CORPUS rather than from the cell, deliberately. It is a
    # property of what the member is for, declared once at generation; putting it on a measurement
    # would invite it to differ between two measurements of one member. It also keeps the
    # agent-visible cell schema closed, which is what stops that document growing fields nobody
    # validates.
    objective, undeclared = _objective_identities(
        self,
    )
    judged = [c for c in comparable if (c.get("family"), c.get("capsule")) in objective] or comparable
    baseline_total = float(sum(c["baseline_gsim_cycles"] for c in judged))
    candidate_total = float(sum(c["candidate_gsim_cycles"] for c in judged))
    if self._totals is None:
        self._totals = []
    previous_best = min(self._totals) if self._totals else None
    self._totals.append(candidate_total)
    best_total = min(self._totals)

    improvements: list[float] = []
    running: float | None = None
    for total in self._totals:
        if running is not None and running > 0:
            improvements.append(max(0.0, (running - total) / running))
        running = total if running is None else min(running, total)

    # THE WORK AND THE CYCLES MUST BE OVER THE SAME MEMBERS. `attainable` is compared against
    # `best_total`, which is summed over `judged` -- the objective members. Summing the MACs over
    # every comparable cell instead put the LAW members' work in the numerator while their cycles
    # stayed out of the denominator, inflating the attainable target and letting the attainment
    # stop condition fire before the objective had actually been reached.
    attainable = SELECT.UNKNOWN
    macs = [c.get("declared_macs") for c in judged]

    def _member_rate(cell: Mapping[str, Any]) -> Any:
        # Compatibility for callers of this internal method that predate per-cell evidence.
        # Production cells always carry the key, including an explicit None when no matched
        # cohort exists; only a genuinely absent key may use the old global context value.
        return (
            cell.get("achievable_macs_per_cycle")
            if "achievable_macs_per_cycle" in cell
            else self.achievable_macs_per_cycle
        )

    unpriced = sorted(
        str(c.get("capsule"))
        for c in judged
        if not (
            isinstance(c.get("declared_macs"), int)
            and c.get("declared_macs") > 0
            and isinstance(_member_rate(c), (int, float))
            and not isinstance(_member_rate(c), bool)
            and _member_rate(c) > 0
        )
    )
    if not unpriced and macs:
        # Each member is priced at the best rate reached by a HOST-OWNED point with the same
        # reduction depth.  Summing work and dividing by the global maximum was dimensionally
        # neat and empirically wrong: it scored k=16 members against deep-k rates that fixed
        # issue/fill cost makes unreachable.
        attainable = sum(float(c["declared_macs"]) / float(_member_rate(c)) for c in judged)

    if self._spend is None:
        self._spend = []
    self._spend.append((label, max(0.0, float(elapsed_s))))
    budget = Budget(
        unit=unpriced_channel(
            "tuning_gsim_feedback",
            missing="the per-call price of a brokered tuning measurement is not measured here",
        ),
        limit_items=self.tuning_call_budget,
    )
    # One charge per measurement actually taken, with its measured wall seconds. Rebuilt from
    # the history because the budget is constructed fresh on every invocation.
    for spent_label, spent_seconds in self._spend:
        budget.charge(items=1.0, seconds=spent_seconds, label=spent_label)
    state = SELECT.SearchState(
        baseline_cycles=int(baseline_total),
        best_cycles=best_total,
        budget=budget,
        attainable_cycles=attainable,
        improvements=tuple(improvements),
    )
    verdicts = SELECT.check_stop(state)
    stops = SELECT.fired(verdicts)
    return {
        "status": "stop" if stops else "continue",
        "queries": state.queries,
        "baseline_total_cycles": baseline_total,
        "objective_members": len(judged),
        "objective_basis": (
            f"summed over the {len(judged)} member(s) declaring member_class "
            f"{self.OBJECTIVE_CLASS!r}"
            + (f"; {undeclared} member(s) declare no class and are counted" if undeclared else "")
            + ("; no member declared the class, so every comparable member was summed" if not objective else "")
        ),
        "best_total_cycles": best_total,
        "previous_best_total_cycles": previous_best,
        "attainable_total_cycles": (None if attainable is SELECT.UNKNOWN else attainable),
        # WHY attainment could not be evaluated, by NAME. One objective member whose declared work
        # carries no price disables the condition for the whole corpus, and it did so silently:
        # a reader saw `attainable_total_cycles: null` with nothing to act on, and every round of
        # every campaign reported the judge as not-fired rather than as never-able-to-speak.
        "attainment_blocked_by": unpriced,
        "share_of_attainable": (None if attainable is SELECT.UNKNOWN or not best_total else attainable / best_total),
        "budget": budget.to_dict(),
        "verdicts": [v.to_dict() for v in verdicts],
        # WHICH CONDITIONS COULD NOT BE ANSWERED AT ALL. A condition that cannot contribute
        # reports `fired: false` exactly like one that was checked and said no, so without this
        # roll-up a reader counts four judges when one of them has never been able to speak.
        "inapplicable": [v.name for v in verdicts if not v.evaluable],
    }


def _take_prefetched(self, index: int, arm: str) -> dict | None:
    """A row a wave already measured, or None. A recorded failure RAISES here, in member order,
    so a member that could not be measured is refused exactly where the serial sweep would have
    refused it -- rather than silently becoming a missing cell."""
    row = getattr(self, "_prefetched", {}).pop((index, arm), None)
    if isinstance(row, Mapping) and "__error__" in row:
        raise StageGateError(f"development GSIM {arm} measurement failed: {row['__error__']}")
    return row


def _prefetch_wave(
    self,
    wave,
    *,
    candidate: Path,
    candidate_before: str,
    call_root: Path,
    deadline: float,
    workers: int,
    measurement_plan=None,
) -> dict:
    """Measure a wave of members concurrently and return ``{(index, arm): redacted row}``.

    Concurrency is threads, not processes, because every heavy step here already runs in its own
    subprocess -- the fan-out is over waiting, not over Python. That is only safe because
    `perf_campaign.boxed_entrypoints` keeps its policy per THREAD: it used to save and restore
    two module globals, which does not compose, and the interleaving left one thread's package
    execution running with no sandbox at all.

    A member that raises is RECORDED against its slot rather than propagated. One member failing
    used to unwind the whole sweep, which is how a measurement with every member run was thrown
    away; here the caller turns a missing row into an unmeasured cell with the reason.
    """
    import concurrent.futures  # noqa: PLC0415

    def one(index: int, member: Any, arm: str) -> dict:
        decision = self.decisions.get((member.family, member.capsule))
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
        package = self.baseline if arm == "baseline" else candidate
        digest = self.baseline_sha256 if arm == "baseline" else candidate_before
        raw = execution._execute(
            self,
            arm=arm,
            package=package,
            package_sha256=digest,
            member=member,
            decision=decision,
            workspace=call_root / FM.ARM_WORKSPACE.format(index=index, arm=arm),
            timeout_s=remaining,
            **({"measurement_plan": measurement_plan} if measurement_plan is not None else {}),
        )
        result = execution._redact_execution(
            self,
            raw,
            decision,
            arm=arm,
            family=member.family,
            capsule=member.capsule,
            required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
        )
        if measurement_plan is not None:
            measurement_plan.verify(self, package=package, arm=arm, member=member)
        return result

    jobs: dict = {}
    rows: dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        # LONGEST FIRST inside the wave, so the tail of the wave is its cheap members and the
        # makespan is not set by a slow member that was submitted last.
        for index, member in sorted(wave, key=lambda im: -float(self.member_cost.get(im[1].capsule, 0.0))):
            if measurement_plan is not None or (member.family, member.capsule) not in self._baseline_cache:
                jobs[pool.submit(contextvars.copy_context().run, one, index, member, "baseline")] = (
                    index,
                    "baseline",
                )
            jobs[pool.submit(contextvars.copy_context().run, one, index, member, "candidate")] = (
                index,
                "candidate",
            )
        for future in concurrent.futures.as_completed(jobs):
            slot = jobs[future]
            try:
                rows[slot] = future.result()
            except Exception as exc:  # noqa: BLE001 - recorded against its slot, never fatal
                rows[slot] = {"__error__": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return rows


def _refresh_achievable(self) -> None:
    """Use only fresh measurements of this session's exact public baseline members."""
    from merlin.development.backend.optimization.measurement import capsule_verdict as CV  # noqa: PLC0415
    from merlin.development.backend.optimization.measurement import measured_ceilings as PMODEL  # noqa: PLC0415

    points = list(self._baseline_points.values())
    counts = {}
    for member in self.corpus.capsules:
        counts[member.capsule] = counts.get(member.capsule, 0) + 1
    self.member_cost = {
        capsule: value for (_, capsule), value in self._baseline_costs.items() if counts.get(capsule) == 1
    }
    _, self.member_cost_basis = FM.order_members_by_cost((), self.member_cost)
    self._achievable_points = tuple(points)
    if not points:
        return
    ceiling = PMODEL.achievable_ceiling(points, provenance="original public frozen baseline executions")
    if not ceiling.known:
        return  # never downgrade a known ceiling to an absent one
    self.achievable_macs_per_cycle = float(ceiling.value)
    self.achievable_dispersion = CV.ceiling_dispersion([{"macs": p.macs, "cycles": p.cycles} for p in points])
    self.achievable_basis = (
        f"best rate over {len(points)} original public baseline measurement(s); "
        "baseline arms only; no candidate measurement contributes to this ceiling"
    )


def _matched_achievable(self, member: CORPUS.PerformanceCapsule) -> tuple[float | None, str, float | None]:
    """Best measured rate for the member's exact contraction-depth signature.

    Reduction depth distinguishes contraction regimes without assuming a target mapping. A global max
    over unrelated K values is useful context, but it is not an attainable target for this
    member.  Both signatures come from declared/emitted operand geometry; absence refuses rather
    than silently falling back to the flattering global number.
    """
    from merlin.development.backend.optimization.measurement import capsule_verdict as CV  # noqa: PLC0415
    from merlin.development.backend.optimization.measurement import measured_ceilings as PMODEL  # noqa: PLC0415

    signature, signature_basis = FM.declared_reduction_depths(member.descriptor)
    if not signature:
        return None, f"member-matched achievable rate unavailable: {signature_basis}", None
    matched = [point for point in self._achievable_points if tuple(getattr(point, "reduction_depths", ())) == signature]
    if not matched:
        return (
            None,
            (
                "member-matched achievable rate unavailable: no host-owned measured point "
                f"has exact reduction-depth signature {list(signature)}"
            ),
            None,
        )
    ceiling = PMODEL.achievable_ceiling(matched, provenance=f"exact reduction-depth signature {list(signature)}")
    if not ceiling.known:
        return None, ceiling.reason, None
    dispersion = CV.ceiling_dispersion([{"macs": point.macs, "cycles": point.cycles} for point in matched])
    return (
        float(ceiling.value),
        (
            f"best rate over {len(matched)} host-owned measured point(s) with exact "
            f"reduction-depth signature {list(signature)}; {signature_basis}"
        ),
        dispersion,
    )
