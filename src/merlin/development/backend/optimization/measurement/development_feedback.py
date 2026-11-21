"""Host-owned tuning feedback with an explicit execution service, never a simulator default."""

from __future__ import annotations

import contextvars
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from merlin.analysis.perf.measurement.execution_policy import ITERATION_MAX_SECONDS
from merlin.common.digest import hash_tree
from merlin.development.backend.optimization.author_interface.prompt_inputs import (
    AUTHORING_GSIM_FLOOR_BUDGET_ENV,
    SWEEP_WORKERS_ENV,
    authoring_gsim_floor_budget,
    sweep_workers,
)
from merlin.development.backend.optimization.contracts import StageGateError, _is_sha256
from merlin.development.backend.optimization.contracts import write_json as _write_json
from merlin.development.backend.optimization.measurement import development_execution as execution
from merlin.development.backend.optimization.measurement import development_progress as progress
from merlin.development.backend.optimization.measurement import feedback_metrics as FM
from merlin.development.backend.optimization.measurement import gsim_gate as GATE
from merlin.development.backend.optimization.measurement import identities as MS
from merlin.development.backend.optimization.measurement import paired_plan as PME
from merlin.development.backend.optimization.measurement.feedback_metrics import derive_machine_bounds
from merlin.development.backend.optimization.measurement.measured_ceilings import (
    load_selected_inputs as _load_whole_model_inputs,
)
from merlin.development.backend.optimization.search import corpus as CORPUS
from merlin.development.backend.optimization.search import corpus_feedback as CF
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- development_feedback ----------------------------------------------------------------------


def _measurement_selection(value):
    from merlin.development.backend.optimization.controls.measurement_scheduling import (
        DevelopmentMeasurementPlan,
    )
    from merlin.development.backend.optimization.ordinary.ordinary_measurement_selection import (
        OrdinaryMeasurementPlan,
    )

    if type(value) not in (DevelopmentMeasurementPlan, OrdinaryMeasurementPlan):
        raise StageGateError("development measurement selection requires its live qualified plan")
    return value


@dataclass
class DevelopmentGsimFeedback:
    """Host-owned, tuning-only evaluation callable used by the broker.

    The full runner result remains outside the agent mount.  ``evaluate`` emits
    only correctness, GSIM cycles, and paired deltas after the strict GSIM gate
    has accepted every exact workload and execution.
    """

    certificate: GATE.CertificateRecord
    corpus: CORPUS.FrozenPerformanceCorpus
    baseline: Path
    baseline_sha256: str
    target_experiment: TargetExperiment
    rtl_identity: Mapping[str, Any]
    work_root: Path
    decisions: Mapping[tuple[str, str], GATE.EvaluationDecision]
    peak_macs_per_cycle: int | None = None
    peak_basis: str = ""
    #: The machine's derived bounds for each member's ROOFLINE: array geometry from the RTL facts and
    #: the memory-path widths of the exact elaboration the certified timing engine was built from
    #: (:func:`merlin.analysis.perf.bounds.capsule_roofline.machine_bounds`). None leaves every roofline UNKNOWN with
    #: that reason; it never substitutes a width.
    machine_bounds: Mapping[str, Any] | None = None
    #: The target's counter-byte binding evidence, probed once per stage the first time a counter
    #: profile runs (:func:`measurement_support.probe_counter_byte_bindings`). Passed to every counter
    #: pass so physical movement is measured whenever the target's probe proves the byte semantics,
    #: and stays UNKNOWN with the probe's own reason when it does not.
    counter_binding: Mapping[str, Any] | None = None
    #: Seconds allowed for each arm's executed-command replay; 0 disables it (the cells then say so).
    executed_profile_timeout_s: int = 120
    #: The operator's whole-model deployment record (:mod:`.whole_model_profile`); None leaves the
    #: whole-model boundary profile unavailable, and the stage advertises that.
    whole_model_inputs: Any = None
    rtl_facts_path: Path | None = None
    _whole_model_baseline: dict | None = None
    achievable_macs_per_cycle: float | None = None
    achievable_basis: str = ""
    # Spread of the achievable rate across the points that established it. It is what "already at
    # the ceiling" tolerates, and it is MEASURED so that judgement is not a constant anyone can
    # turn until the answer changes. None when fewer than two points price, which refuses.
    achievable_dispersion: float | None = None
    #: Every host-owned point currently eligible to establish a rate.  Unlike the scalar summary
    #: ceiling, per-member scoring filters these by exact contraction reduction depth.
    _achievable_points: tuple = ()
    _baseline_points: dict = field(default_factory=dict, repr=False)
    _baseline_costs: dict = field(default_factory=dict, repr=False)
    #: Median measured simulation seconds per capsule, and how the ordering was arrived at. Used to
    #: sweep cheapest-first, so a losing candidate is refuted before the corpus's slowest members
    #: are paid for. Empty means no history: the declared order stands and the basis says so.
    member_cost: Mapping[str, float] = field(default_factory=dict)
    member_cost_basis: str = ""
    tuning_call_budget: int | None = None
    #: Total candidate cycles over the comparable members, one entry per feedback invocation. The
    #: stop conditions read the SHAPE of this history, not any single measurement.
    _totals: list[float] | None = None
    #: ``(label, seconds)`` per feedback invocation -- the measurements this search actually spent.
    #: The budget ledger is rebuilt from this every call, so ``budget_exhausted`` judges real spend
    #: rather than reporting an untouched ledger.
    _spend: list[tuple[str, float]] | None = None
    executor: Callable[..., Mapping[str, Any]] | None = None
    _baseline_cache: dict[tuple[str, str], dict[str, Any]] | None = None
    _profile_baseline_cache: dict[tuple[str, str], dict[str, Any]] | None = None
    #: Rows a wave measured ahead of the loop that consumes them, keyed by (member index, arm).
    _prefetched: dict = field(default_factory=dict)
    _compiler_profile_service: object | None = field(default=None, repr=False)
    _capsule_development_scope: object | None = field(default=None, repr=False)

    def profile_witness(self) -> tuple[CORPUS.PerformanceCapsule, str]:
        """The fixed reduced witness used only to explain a global candidate's cycle change.

        Selection reads the frozen descriptor and baseline-only simulation-cost history.  It is made
        without candidate output or cycles, prefers an objective member, and stays fixed for the
        stage.  Formal promotion still evaluates the entire sealed cohort.
        """
        from merlin.analysis.perf.measurement.execution_policy import require_probe_execution  # noqa: PLC0415

        members = []
        for member in tuple(getattr(self.corpus, "capsules", ()) or ()):
            try:
                require_probe_execution(member.descriptor)
            except ValueError:
                continue
            members.append(member)
        if not members:
            raise StageGateError("the reduced occupancy profile has no frozen non-model probe")
        ordered, cost_basis = FM.order_members_by_cost(members, self.member_cost)
        objective = [
            member
            for member in ordered
            if (((member.descriptor or {}).get("performance") or {}).get("member_class") == self.OBJECTIVE_CLASS)
        ]
        member = (objective or list(ordered))[0]
        basis = (
            "first objective member under " + cost_basis
            if objective
            else "no member declares the objective class; first member under " + cost_basis
        )
        return member, basis

    def whole_model_profile(self, candidate: Path, *, round_index: int, call_index: int, timeout_s: int) -> dict:
        """Per-group cycles, gaps and rooflines of the declared whole-model programs (see that module)."""
        from merlin.development.backend.optimization.measurement import measured_ceilings as WMP  # noqa: PLC0415

        return WMP.profile_candidate(self, candidate, round_index=round_index, call_index=call_index)

    def counter_binding_evidence(self) -> Mapping[str, Any]:
        if self.counter_binding is None:
            target = str(getattr(self.target_experiment, "target", "") or "")
            try:
                self.counter_binding = MS.probe_counter_byte_bindings(self.rtl_identity, target=target)
            except Exception as exc:  # noqa: BLE001 - an unavailable probe leaves bytes UNKNOWN
                self.counter_binding = {"status": "unknown", "counter_facts": [], "why": type(exc).__name__}
        return self.counter_binding

    def profile_member(self, identity: str) -> tuple[CORPUS.PerformanceCapsule, str]:
        """The frozen tuning member ``family/capsule`` the agent asked to profile, or a refusal."""
        family, sep, capsule = str(identity or "").strip().partition("/")
        if not sep or not family or not capsule:
            raise StageGateError("a profiled member is named as family/capsule from the frozen tuning corpus")
        from merlin.analysis.perf.measurement.execution_policy import require_probe_execution  # noqa: PLC0415

        for member in tuple(getattr(self.corpus, "capsules", ()) or ()):
            if member.family == family and member.capsule == capsule:
                try:
                    require_probe_execution(member.descriptor)
                except ValueError as exc:
                    raise StageGateError(str(exc)) from exc
                return member, "named by the agent from the frozen tuning corpus"
        raise StageGateError(f"{family}/{capsule} is not a member of the frozen tuning corpus")

    def profile(
        self,
        candidate: Path,
        *,
        round_index: int,
        call_index: int,
        timeout_s: int,
        member: str | None = None,
    ) -> dict[str, Any]:
        """Warm-profile one reduced member with only occupancy/movement counters: the preselected
        witness, or the frozen tuning member the caller names (``family/capsule``)."""
        candidate = Path(candidate).resolve(strict=True)
        if member is None:
            member, selection_basis = self.profile_witness()
        else:
            member, selection_basis = self.profile_member(member)
        key = (member.family, member.capsule)
        decision = self.decisions.get(key)
        if decision is None:
            raise StageGateError(f"reduced occupancy profile decision is absent for {key}")
        root = self.work_root / f"round_{round_index:02d}" / f"profile_{call_index:03d}"
        if root.exists() or root.is_symlink():
            raise StageGateError(f"reduced occupancy profile workspace is not fresh: {root}")
        root.mkdir(parents=True)
        measured_candidate = root / "_measured_candidate"
        shutil.copytree(candidate, measured_candidate, symlinks=True)
        candidate_sha = str(hash_tree(measured_candidate)["sha256"])
        started = time.monotonic()
        deadline = started + min(timeout_s, int(ITERATION_MAX_SECONDS))
        if self._profile_baseline_cache is None:
            self._profile_baseline_cache = {}

        def execute(arm: str, package: Path, digest: str) -> dict[str, Any]:
            remaining = int(deadline - time.monotonic())
            if remaining <= 0:
                raise StageGateError("reduced occupancy profile exceeded its deterministic timeout")
            raw = execution._execute(
                self,
                arm=arm,
                package=package,
                package_sha256=digest,
                member=member,
                decision=decision,
                workspace=root / arm,
                timeout_s=remaining,
                hardware_counters=True,
            )
            return execution._redact_profile(self, raw, decision, arm=arm, member=member)

        baseline = self._profile_baseline_cache.get(key)
        if baseline is None:
            baseline = execute("baseline", self.baseline, self.baseline_sha256)
            self._profile_baseline_cache[key] = baseline
        candidate_row = execute("candidate", measured_candidate, candidate_sha)
        if str(hash_tree(measured_candidate)["sha256"]) != candidate_sha:
            raise StageGateError("reduced occupancy profile mutated the candidate snapshot")
        return {
            "schema": "host_owned_reduced_global_profile_v1",
            "purpose": (
                "calibrate occupancy, movement, and latency hiding for a complete-model "
                "plan; never a whole-model performance result"
            ),
            "witness": {
                "family": member.family,
                "capsule": member.capsule,
                "selection": selection_basis,
                "selected_before_candidate_measurement": True,
            },
            # The measured member's derived machine bound, beside its counters (see `roofline` in the
            # tuning feedback): busy time and overlap read against the floors they can be compared to.
            "roofline": FM.roofline_cell(
                member.descriptor,
                self.machine_bounds,
                baseline_cycles=baseline["total_compute_cycles"],
                candidate_cycles=candidate_row["total_compute_cycles"],
            ),
            "candidate_sha256": candidate_sha,
            "profile_contract": {
                "warmup_runs": 1,
                "measured_runs": 1,
                "primary_metric": "total_compute_cycles",
                "maximum_simulator_seconds": int(ITERATION_MAX_SECONDS),
            },
            "baseline": baseline,
            "candidate": candidate_row,
            "cycle_delta": candidate_row["total_compute_cycles"] - baseline["total_compute_cycles"],
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    #: A losing prefix must be this long before it may stop the sweep. One member is an anecdote --
    #: the cheapest member is also the one most dominated by fixed per-invocation cost, where a
    #: schedule change shows least -- so a single loss is not allowed to end a measurement.
    MINIMUM_REFUTING_PREFIX = 3

    #: What a member declares itself to be FOR. Mirrors the generator's own vocabulary; a member whose
    #: frozen descriptor predates the field is treated as OBJECTIVE and counted, because silently
    #: dropping it from the total would understate the very thing the total measures.
    OBJECTIVE_CLASS = "OBJECTIVE"

    def roofline_floor(self, member: Any) -> int | None:
        """The member's declared-work roofline floor on this machine, or None when underivable."""
        value = FM.capsule_roofline(member.descriptor, self.machine_bounds).get("roofline_cycles")
        return value if type(value) is int else None

    def evaluate(
        self,
        candidate: Path,
        *,
        round_index: int,
        call_index: int,
        timeout_s: int,
        measurement_plan=None,
        retained_revision=None,
        complete_roster: bool = False,
    ) -> dict[str, Any]:
        if type(complete_roster) is not bool or complete_roster and measurement_plan is not None:
            raise StageGateError("complete-roster RTL ranking needs the original full measurement selection")
        candidate = Path(candidate).resolve(strict=True)
        if measurement_plan is not None:
            _measurement_selection(measurement_plan)
            measurement_plan.verify(self, candidate=candidate)
        if self._baseline_cache is None:
            self._baseline_cache = {}
        call_root = self.work_root / f"round_{round_index:02d}" / f"call_{call_index:03d}"
        if call_root.exists() or call_root.is_symlink():
            raise StageGateError(f"development GSIM feedback workspace is not fresh: {call_root}")
        call_root.mkdir(parents=True)
        # MEASURE A SNAPSHOT, NOT THE AGENT'S LIVE TREE. This sweep runs for over an hour while the
        # agent keeps working, and it used to hash the live tree at both ends and refuse if the two
        # differed. Measured 2026-09-04: every one of 38 members ran and passed, and the whole
        # eighty-minute result was thrown away because the agent appended to `iteration_notes.md`
        # while the last member was finishing -- a file the prompt TELLS it to keep, which no
        # measurement reads. The check was aimed at a real property, that the cycles belong to the
        # bytes that produced them, and a snapshot gives that property outright instead of turning a
        # note into a voided campaign. Every run below reads the copy, so a concurrent edit cannot
        # change what was measured, and `candidate_sha256` names the bytes actually executed.
        if retained_revision is None:
            measured_candidate = call_root / "_measured_candidate"
            shutil.copytree(candidate, measured_candidate, symlinks=True)
        else:
            from merlin.development.backend.optimization.search.retained_revisions import executing_revision

            if executing_revision(candidate) is not retained_revision:
                raise StageGateError("retained RTL execution requires the issued fixed evaluator context")
            retained_revision.verify()
            measured_candidate = retained_revision.source
        candidate_before = str(hash_tree(measured_candidate)["sha256"])
        candidate = measured_candidate
        original_members = sorted(self.corpus.capsules, key=lambda row: (row.family, row.capsule))
        if not original_members:
            raise StageGateError("development GSIM feedback has zero frozen tuning members")
        members = original_members
        if measurement_plan is not None:
            from merlin.development.backend.optimization.contracts import document_sha256

            by_id = {document_sha256([member.family, member.capsule]): member for member in original_members}
            members = [by_id[identity] for identity in measurement_plan.selected]
        # LARGE MEMBERS NEVER BLOCK AUTHORING. Above the declared roofline-floor budget a member is
        # measured on gSIM only in the final cells (one of them alone runs past an hour); here it gets
        # its roofline cell and a stated reason, and static analysis, Spike and the reduced witnesses
        # remain the agent's tools for it.
        # An explicit complete sweep is the measured selection denominator.
        # The exploratory floor budget must not silently omit its large cases.
        from merlin.development.backend.optimization.ordinary.ordinary_measurement_selection import (
            OrdinaryMeasurementPlan,
        )

        budget = (
            None
            if complete_roster or type(measurement_plan) is OrdinaryMeasurementPlan
            else authoring_gsim_floor_budget()
        )
        deferred = []
        if budget is not None:
            floors = {(member.family, member.capsule): self.roofline_floor(member) for member in members}
            deferred = [
                m
                for m in members
                if type(floors[(m.family, m.capsule)]) is int and floors[(m.family, m.capsule)] > budget
            ]
            members = [m for m in members if m not in deferred]
        cells: list[dict[str, Any]] = []
        started = time.monotonic()
        # CHEAPEST MEASURED MEMBER FIRST. A candidate behind on every member measured so far is
        # behind; paying for the corpus's slowest members to confirm it spends the budget on a
        # conclusion already reached.
        if measurement_plan is None:
            members, _order_basis = FM.order_members_by_cost(members, self.member_cost)
        stopped_after: int | None = None
        # WAVES, so a declared fan-out does not cost the early stop its meaning. The first wave is
        # exactly the prefix the stop rule needs before it may fire, which makes wave 0's decision
        # identical to the sequential sweep's; after that the waves are the fan-out wide. What a
        # wave can waste is the members it launched past the point the sequential sweep would have
        # stopped -- bounded, and cheapest-first, so the boundary nearest the decision is the
        # cheapest members. `_prefetched` is consumed by the loop body below in place of executing.
        workers = sweep_workers()
        deadline = started + timeout_s
        self._prefetched = {}
        pending = list(enumerate(members))
        wave_sizes = [self.MINIMUM_REFUTING_PREFIX] if workers > 1 else []
        for index, member in enumerate(members):
            key = (member.family, member.capsule)
            decision = self.decisions.get(key)
            if decision is None:
                raise StageGateError(f"development GSIM decision is absent for {key}")
            remaining = timeout_s - int(time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
            if workers > 1 and (index, "candidate") not in self._prefetched and pending:
                take = wave_sizes.pop(0) if wave_sizes else workers
                wave, pending = pending[:take], pending[take:]
                self._prefetched.update(
                    progress._prefetch_wave(
                        self,
                        wave,
                        candidate=candidate,
                        candidate_before=candidate_before,
                        call_root=call_root,
                        deadline=deadline,
                        workers=workers,
                        **({"measurement_plan": measurement_plan} if measurement_plan is not None else {}),
                    )
                )
            baseline = (
                self._baseline_cache.get(key) if measurement_plan is None else None
            ) or progress._take_prefetched(self, index, "baseline")
            if baseline is None:
                raw = execution._execute(
                    self,
                    arm="baseline",
                    package=self.baseline,
                    package_sha256=self.baseline_sha256,
                    member=member,
                    decision=decision,
                    workspace=call_root / FM.ARM_WORKSPACE.format(index=index, arm="baseline"),
                    timeout_s=remaining,
                    **({"measurement_plan": measurement_plan} if measurement_plan is not None else {}),
                )
                baseline = execution._redact_execution(
                    self,
                    raw,
                    decision,
                    arm="baseline",
                    family=member.family,
                    capsule=member.capsule,
                    required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
                )
                if measurement_plan is None:
                    self._baseline_cache[key] = baseline
            if measurement_plan is not None:
                measurement_plan.verify(self, package=self.baseline, arm="baseline", member=member)
            remaining = timeout_s - int(time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
            candidate_row = progress._take_prefetched(self, index, "candidate")
            if candidate_row is None:
                raw = execution._execute(
                    self,
                    arm="candidate",
                    package=candidate,
                    package_sha256=candidate_before,
                    member=member,
                    decision=decision,
                    workspace=call_root / FM.ARM_WORKSPACE.format(index=index, arm="candidate"),
                    timeout_s=remaining,
                    **({"measurement_plan": measurement_plan} if measurement_plan is not None else {}),
                )
                candidate_row = execution._redact_execution(
                    self,
                    raw,
                    decision,
                    arm="candidate",
                    family=member.family,
                    capsule=member.capsule,
                    required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
                )
            if measurement_plan is not None:
                measurement_plan.verify(self, package=candidate, arm="candidate", member=member)
            comparable = baseline["correct"] and candidate_row["correct"]
            bcycles, ccycles = baseline["gsim_cycles"], candidate_row["gsim_cycles"]
            # UTILIZATION against a ceiling this machine's own RTL derives. Cycles alone say nothing
            # about how much of the machine a program used, and a family whose claim is a fit can be
            # satisfied by a program that uses LESS of it. Both inputs are derived and host-computed:
            # the required work from the capsule's declared operands, the peak from facts.arrays.
            # Either being underivable yields null and a reason, never an assumed number.
            spec_macs, work_basis = FM.declared_capsule_macs(member.descriptor)
            peak = self.peak_macs_per_cycle
            ideal = (spec_macs / peak) if (spec_macs and peak) else None

            def _utilization(cycles: Any) -> float | None:
                if ideal is None or not isinstance(cycles, int) or isinstance(cycles, bool):
                    return None
                return (ideal / cycles) if cycles > 0 else None

            achievable, achievable_basis, matched_dispersion = (
                progress._matched_achievable(self, member)
                if measurement_plan is None
                else (None, "selected development measurements cannot establish attainment", None)
            )
            achievable_ideal = (spec_macs / achievable) if (spec_macs and achievable) else None

            def _share(cycles: Any) -> float | None:
                if achievable_ideal is None or not isinstance(cycles, int) or isinstance(cycles, bool):
                    return None
                return (achievable_ideal / cycles) if cycles > 0 else None

            cells.append(
                {
                    "family": member.family,
                    "capsule": member.capsule,
                    "baseline_correct": baseline["correct"],
                    "candidate_correct": candidate_row["correct"],
                    "baseline_gsim_cycles": bcycles,
                    "candidate_gsim_cycles": ccycles,
                    "candidate_minus_baseline_cycles": ccycles - bcycles if comparable else None,
                    "baseline_over_candidate": bcycles / ccycles if comparable else None,
                    "comparable": comparable,
                    "declared_macs": spec_macs,
                    "declared_work_basis": work_basis,
                    "ideal_cycles_at_peak": ideal,
                    "baseline_utilization": _utilization(bcycles),
                    "candidate_utilization": _utilization(ccycles),
                    "baseline_share_of_achievable": _share(bcycles),
                    "candidate_share_of_achievable": _share(ccycles),
                    "achievable_macs_per_cycle": achievable,
                    "achievable_basis": achievable_basis,
                    **FM.capsule_verdict_fields(
                        capsule=member.capsule,
                        declared_macs=spec_macs,
                        achievable_rate=achievable,
                        baseline_cycles=bcycles,
                        candidate_cycles=ccycles if comparable else None,
                        dispersion=matched_dispersion,
                    ),
                    # WHERE EACH ARM SITS AGAINST THE MACHINE'S OWN BOUND for this member's declared
                    # work: the larger of the array's compute floor and the memory path's movement
                    # floor, both derived. Only a correct arm is positioned against it.
                    "roofline": FM.roofline_cell(
                        member.descriptor,
                        self.machine_bounds,
                        baseline_cycles=bcycles if baseline["correct"] else None,
                        candidate_cycles=ccycles if candidate_row["correct"] else None,
                    ),
                    # WHAT EACH ARM EXECUTED: accelerator commands by class, retired host
                    # instructions, and the local-memory rows its commands addressed.
                    "executed_commands": FM.executed_commands_cell(
                        baseline.get("executed"), candidate_row.get("executed")
                    ),
                    "measured": True,
                    "skip_reason": None,
                }
            )
            # STOP ONLY A LOSING CANDIDATE, NEVER PROMOTE A WINNING ONE. The rule is one-directional
            # on purpose: a candidate behind on every comparable member measured so far cannot be
            # rescued by a member it has not reached, because the objective is fewer cycles on the
            # SAME work and it is already behind on all of it. The converse is false -- a candidate
            # ahead on the cheap prefix may still lose on a member it has not paid for -- so a
            # winning prefix buys nothing and the full sweep is measured.
            # A weighted full-roster objective can recover on later members;
            # its selected complete_roster route therefore preserves all cells.
            if (
                measurement_plan is None
                and not complete_roster
                and progress._refuted_so_far(self, cells, index, len(members))
            ):
                stopped_after = index + 1
                break
        for member in members[stopped_after:] if stopped_after is not None else ():
            # RECORDED, not omitted. A missing cell and an unmeasured one are different claims.
            cells.append(
                FM.unmeasured_cell(
                    member,
                    machine=self.machine_bounds,
                    reason=(
                        "the sweep is ordered cheapest-measured-first and this candidate "
                        "was already behind on every comparable member measured before "
                        "this one; the remaining members were not paid for"
                    ),
                )
            )
        for member in deferred:
            cells.append(
                FM.unmeasured_cell(
                    member,
                    machine=self.machine_bounds,
                    reason=(
                        f"measured on gSIM only in the final measurement cells: its roofline floor exceeds "
                        f"the authoring gSIM budget of {budget} cycles"
                    ),
                )
            )
        if measurement_plan is not None:
            selected_ids = set(measurement_plan.selected)
            for member in original_members:
                if document_sha256([member.family, member.capsule]) not in selected_ids:
                    cells.append(
                        FM.unmeasured_cell(member, machine=self.machine_bounds, reason=measurement_plan.reason(member))
                    )
            measurement_plan.verify(self, candidate=candidate)
        # The snapshot must still be the bytes the runs read: nothing here may edit it, and a
        # difference now would mean the measurement mutated its own input rather than that the agent
        # kept working. That is a real defect and still refuses.
        # THE CEILING IS RE-DERIVED FROM THE BASELINES THIS SWEEP JUST MEASURED, then every field
        # that depends on it is recomputed. Doing it only at prepare time would leave round 0 -- the
        # round that sets the agent's whole plan -- scored against phase 1's corpus, which is the
        # case that was actually wrong.
        if measurement_plan is None:
            progress._refresh_achievable(
                self,
            )
        members_by_identity = {(member.family, member.capsule): member for member in original_members}
        for row in cells:
            if measurement_plan is not None:
                if row.get("measured"):
                    for field in (
                        "achievable_macs_per_cycle",
                        "baseline_share_of_achievable",
                        "candidate_share_of_achievable",
                        "factor_to_achievable",
                        "ideal_cycles_at_achievable",
                        "cycles_saved",
                        "gap_closed",
                    ):
                        row[field] = None
                    row["achievable_basis"] = "selected development measurements cannot establish attainment"
                    row["verdict"] = "undeterminable"
                    row["verdict_reason"] = row["achievable_basis"]
                continue
            if not row.get("measured") or not row.get("declared_macs"):
                continue  # an unmeasured cell keeps its nulls
            member = members_by_identity[(row["family"], row["capsule"])]
            rate, basis, dispersion = progress._matched_achievable(self, member)
            row["achievable_macs_per_cycle"] = rate
            row["achievable_basis"] = basis
            if rate:
                ideal = float(row["declared_macs"]) / rate
                row["baseline_share_of_achievable"] = ideal / row["baseline_gsim_cycles"]
                row["candidate_share_of_achievable"] = (
                    ideal / row["candidate_gsim_cycles"] if row["comparable"] else None
                )
            else:
                row["baseline_share_of_achievable"] = None
                row["candidate_share_of_achievable"] = None
            row.update(
                FM.capsule_verdict_fields(
                    capsule=row["capsule"],
                    declared_macs=row["declared_macs"],
                    achievable_rate=rate,
                    baseline_cycles=row["baseline_gsim_cycles"],
                    candidate_cycles=(row["candidate_gsim_cycles"] if row["comparable"] else None),
                    dispersion=dispersion,
                )
            )

        candidate_after = str(hash_tree(candidate)["sha256"])
        if candidate_after != candidate_before:
            raise StageGateError("development GSIM evaluation mutated the candidate snapshot")
        if retained_revision is not None:
            retained_revision.verify()
        comparable = [row for row in cells if row["comparable"]]
        return CF.validate_redacted_feedback(
            {
                "schema_version": 1,
                "kind": "host_owned_tuning_gsim_feedback",
                "round": round_index,
                "invocation": call_index,
                "tuning_corpus_sha256": self.corpus.capsules_sha256,
                "candidate_sha256": candidate_before,
                "certificate_sha256": self.certificate.sha256,
                "engine": "gsim",
                "cells": cells,
                "stopping": progress._stopping(
                    self,
                    cells,
                    label=f"round_{round_index:02d}/call_{call_index:03d}",
                    elapsed_s=time.monotonic() - started,
                )
                if measurement_plan is None
                else {
                    "status": "undeterminable",
                    "reason": "selected development measurements cannot establish convergence",
                    "verdicts": [],
                },
                "summary": {
                    "members": len(cells),
                    "comparable": len(comparable),
                    "all_correct": all(row["comparable"] for row in cells if row["measured"]),
                    "peak_macs_per_cycle": self.peak_macs_per_cycle,
                    "peak_basis": self.peak_basis,
                    "achievable_macs_per_cycle": self.achievable_macs_per_cycle if measurement_plan is None else None,
                    "achievable_basis": self.achievable_basis
                    if measurement_plan is None
                    else "selected development measurements cannot establish attainment",
                    "recoverable": FM.recoverable_cycles(cells)
                    if measurement_plan is None
                    else {
                        "status": "unavailable",
                        "reason": "selected development measurements do not price the complete original corpus",
                        "ranked": [],
                        "corpus_total_cycles": None,
                    },
                    "roofline_machine": FM.roofline_machine_summary(self.machine_bounds),
                },
            }
        )


development_executor = PME.development_executor


def prepare_development_feedback(
    *,
    contract_root: Path,
    certificate_path: Path | None,
    certificate_sha256: str | None,
    rtl_facts_path: Path | None,
    corpus: CORPUS.FrozenPerformanceCorpus,
    baseline: Path,
    baseline_sha256: str,
    target_experiment: TargetExperiment,
    work_root: Path,
    tuning_call_budget: int | None = None,
    functional_run_dir: Path | None = None,
    whole_model_inputs: Path | None = None,
    capsule_development_scope=None,
) -> DevelopmentGsimFeedback:
    """Bind one stage to a strict certificate and exact frozen tuning corpus."""
    if capsule_development_scope is not None:
        from merlin.development.backend.optimization.ordinary.capsule_ordinary_scope import (
            CapsuleOrdinaryScope,
        )

        if type(capsule_development_scope) is not CapsuleOrdinaryScope:
            raise StageGateError("compile/Spike scope requires its original capsule qualification")
        from merlin.development.backend.optimization.ordinary.capsule_ordinary_scope import verify

        verify(capsule_development_scope)
    if certificate_path is None or not _is_sha256(certificate_sha256) or rtl_facts_path is None:
        raise StageGateError(
            "development GSIM feedback certificate is unavailable: certificate SHA and RTL facts are required"
        )
    try:
        certificate = GATE.load_certificate(certificate_path, expected_sha256=str(certificate_sha256))
        if certificate.target != target_experiment.target:
            raise GATE.GsimGateError("certificate target differs from the performance target")

        rtl_identity = MS.load_rtl_identity(Path(rtl_facts_path), target_experiment.target)
        peak_macs, peak_basis = FM.derived_peak_macs_per_cycle(Path(rtl_facts_path), target_experiment.target)
        machine_bounds = derive_machine_bounds(Path(rtl_facts_path), certificate, target_experiment.target)
        # Historical run trees include hidden and final evidence. They supply
        # neither rates nor scheduling costs; only this session's admitted public
        # baseline replies can establish them after actual measurement.
        member_cost = {}
        _, member_cost_basis = FM.order_members_by_cost((), member_cost)
        achievable_macs, achievable_basis = None, "no fresh public baseline measurements"
        achievable_dispersion = None
        decisions: dict[tuple[str, str], GATE.EvaluationDecision] = {}
        for member in sorted(corpus.capsules, key=lambda row: (row.family, row.capsule)):
            if member.descriptor.get("label") != "dev":
                raise GATE.GsimGateError(f"{member.family}/{member.capsule} is not a frozen tuning member")
            workload = PME.gsim_workload(member)
            decision = GATE.plan_evaluation(certificate, workload, phase="development_correctness", gsim_available=True)
            # THE GATE GRANTS A DEVELOPMENT-ONLY REFERENCE-ENGINE FALLBACK HERE AND THIS REFUSES IT
            # ANYWAY, deliberately. Taking it would put a reference-engine cycle count into the
            # agent-visible feedback document, and that document's redaction boundary forbids naming
            # that engine at all -- a cell would either carry the forbidden name or hide which engine
            # timed it, and hiding it is worse. An out-of-envelope member is therefore admitted by
            # PAYING for its certificate offline, not by relaxing what a development cell may say.
            if capsule_development_scope is None and (
                not decision.admitted
                or not decision.eligible
                or decision.selected_engine != "gsim"
                or not decision.use_gsim
            ):
                raise GATE.GsimGateError(
                    f"{member.family}/{member.capsule} is outside the exact GSIM certificate envelope"
                )
            decisions[(member.family, member.capsule)] = decision
    except Exception as exc:  # noqa: BLE001 - all qualification failures become one pre-launch refusal
        raise StageGateError(f"development GSIM feedback certificate is unavailable or invalid: {exc}") from exc
    if str(hash_tree(baseline)["sha256"]) != baseline_sha256:
        raise StageGateError("development GSIM baseline bytes differ from the functional submission")
    from merlin.development.backend.feedback.execution.profile_runtime import functional_service

    profile_service = None if functional_run_dir is None else functional_service(functional_run_dir)
    return DevelopmentGsimFeedback(
        certificate,
        corpus,
        Path(baseline),
        baseline_sha256,
        target_experiment,
        rtl_identity,
        Path(work_root),
        decisions,
        peak_macs_per_cycle=peak_macs,
        peak_basis=peak_basis,
        machine_bounds=machine_bounds,
        achievable_macs_per_cycle=achievable_macs,
        achievable_basis=achievable_basis,
        achievable_dispersion=achievable_dispersion,
        member_cost=member_cost,
        member_cost_basis=member_cost_basis,
        executor=partial(development_executor, contract_root=contract_root),
        tuning_call_budget=tuning_call_budget,
        whole_model_inputs=_load_whole_model_inputs(whole_model_inputs, target_experiment),
        rtl_facts_path=Path(rtl_facts_path),
        _compiler_profile_service=profile_service,
        _capsule_development_scope=capsule_development_scope,
    )
