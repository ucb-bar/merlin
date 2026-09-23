"""Host-owned tuning feedback with an explicit execution service, never a simulator default."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.targetgen.target_experiment import TargetExperiment

from . import corpus as CORPUS
from . import corpus_feedback as CF
from . import feedback_metrics as FM
from . import gsim_gate as GATE
from . import measurement_support as MS
from . import paired_measurement as PME
from .broker_evidence import _is_sha256
from .contracts import StageGateError
from .contracts import write_json as _write_json

SWEEP_WORKERS_ENV = "MERLIN_PERF_SWEEP_WORKERS"


def sweep_workers() -> int:
    """The declared sweep fan-out, or 1. Refuses a value it cannot read rather than guessing one."""
    raw = (os.environ.get(SWEEP_WORKERS_ENV) or "").strip()
    if not raw:
        return 1
    try:
        workers = int(raw)
    except ValueError:
        raise StageGateError(f"{SWEEP_WORKERS_ENV}={raw!r} is not an integer; refusing to guess a fan-out") from None
    if workers < 1:
        raise StageGateError(f"{SWEEP_WORKERS_ENV}={workers} is not a positive fan-out")
    return workers


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
    achievable_macs_per_cycle: float | None = None
    achievable_basis: str = ""
    # Spread of the achievable rate across the points that established it. It is what "already at
    # the ceiling" tolerates, and it is MEASURED so that judgement is not a constant anyone can
    # turn until the answer changes. None when fewer than two points price, which refuses.
    achievable_dispersion: float | None = None
    #: The phase-1 functional points the ceiling starts from, kept so it can be RE-DERIVED over a
    #: wider set once this run's own frozen-baseline measurements exist. Phase 1's corpus is not
    #: this corpus: measured 2026-09-04, its ceiling was 80.01 MACs/cycle while four performance
    #: members already ran ABOVE it at baseline, and its dispersion of 0.472 put the "already at the
    #: ceiling" line at 42.25 -- so 14 of 38 members were told they had no headroom left.
    seed_points: tuple = ()
    #: Every host-owned point currently eligible to establish a rate.  Unlike the scalar summary
    #: ceiling, per-member scoring filters these by exact contraction reduction depth.
    _achievable_points: tuple = ()
    functional_run_id: str = ""
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

    def _execute(
        self,
        *,
        arm: str,
        package: Path,
        package_sha256: str,
        member: CORPUS.PerformanceCapsule,
        decision: GATE.EvaluationDecision,
        workspace: Path,
        timeout_s: int,
        hardware_counters: bool = False,
    ) -> Mapping[str, Any]:
        """Retry an unavailable instrument once, never a verdict or certificate rejection.

        Both attempts use the same frozen bytes and original wall deadline. All raw outcomes stay
        host-private, so an infrastructure retry neither teaches the agent nor selects a faster result.
        """
        deadline = time.monotonic() + timeout_s
        attempts = []
        for index in range(2):
            remaining = timeout_s if index == 0 else int(deadline - time.monotonic())
            if remaining <= 0:
                break
            destination = workspace if index == 0 else workspace.with_name(workspace.name + "__retry_01")
            raw = self._execute_once(
                arm=arm,
                package=package,
                package_sha256=package_sha256,
                member=member,
                decision=decision,
                workspace=destination,
                timeout_s=remaining,
                hardware_counters=hardware_counters,
            )
            measurement = raw.get("measurement") or {}
            qualification = measurement.get("gsim_qualification") or {}
            outcome = (measurement.get("execution_outcome") or {}).get("gsim") or {}
            tier = outcome.get("tier_outcome") or {}
            attempts.append({"attempt": index, "workspace": str(destination), "raw": raw})
            _write_json(
                workspace.with_name(workspace.name + ".attempts.json"),
                {
                    "schema": "performance-execution-attempts.v1",
                    "package_sha256": package_sha256,
                    "max_attempts": 2,
                    "timeout_seconds": timeout_s,
                    "attempts": attempts,
                },
            )
            if not (
                qualification.get("kind") == "execution_missing"
                and measurement.get("numeric") == "pass"
                and tier.get("status") == "unavailable"
            ):
                break
        if not attempts:
            raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
        return raw

    def _execute_once(
        self,
        *,
        arm: str,
        package: Path,
        package_sha256: str,
        member: CORPUS.PerformanceCapsule,
        decision: GATE.EvaluationDecision,
        workspace: Path,
        timeout_s: int,
        hardware_counters: bool = False,
    ) -> Mapping[str, Any]:
        from merlin.perf.execution_policy import require_probe_execution  # noqa: PLC0415

        try:
            require_probe_execution(member.descriptor)
        except ValueError as exc:
            raise StageGateError(str(exc)) from exc
        if self.executor is not None:
            return self.executor(
                arm=arm,
                package=package,
                package_sha256=package_sha256,
                member=member,
                decision=decision,
                workspace=workspace,
                timeout_s=timeout_s,
                certificate=self.certificate,
                target_experiment=self.target_experiment,
                rtl_identity=self.rtl_identity,
                hardware_counters=hardware_counters,
            )
        raise StageGateError("development GSIM feedback requires an explicit host executor")

    @staticmethod
    def _tier_skipped_beyond_declared_ceiling(measurement: Mapping[str, Any], required_tiers: Sequence[str]) -> bool:
        """True when the only failure is a tier the capsule never required, skipped by its ceiling.

        Measured 2026-09-03 on every PK feedback document: PK00_k16's baseline came back
        ``correct: False`` while its numerics were EXACT (mismatch_count 0, max_abs_error 0) and every
        tier it declares -- L0, L1, L2, L3 -- passed. A fifth tier, L4, was attempted, skipped for
        being deeper than the capsule's declared oracle ceiling, and that SKIP was recorded as a
        failure, which demoted the capsule to ``screened_only``. One of four members was therefore
        non-comparable in every single measurement, costing a quarter of the family's comparison
        surface -- for a tier the capsule never required. A check that could not run is not a verdict.

        This forgives nothing else: numerics, both simulator verdicts, and every REQUIRED tier are
        still demanded by the caller, and a failure without ceiling evidence, or at a tier the capsule
        does require, remains a failure.
        """
        failure = measurement.get("failure")
        if not isinstance(failure, Mapping) or not failure.get("oracle_ceiling"):
            return False
        tier = failure.get("tier")
        return isinstance(tier, str) and bool(tier) and tier not in tuple(required_tiers)

    def _redact_execution(
        self,
        raw: Mapping[str, Any],
        decision: GATE.EvaluationDecision,
        *,
        arm: str,
        family: str,
        capsule: str,
        required_tiers: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Reduce a host result to the only feedback fields the agent may see."""
        measurement = raw.get("measurement")
        if not isinstance(measurement, Mapping):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} returned no measurement")
        qualification = measurement.get("gsim_qualification")
        if isinstance(qualification, Mapping) and qualification.get("kind") == "execution_missing":
            outcome = (measurement.get("execution_outcome") or {}).get("gsim") or {}
            raise StageGateError(
                f"development GSIM {arm}/{family}/{capsule}: no GSIM execution evidence was produced; "
                f"tier outcome: {str(outcome.get('tier_outcome'))[:300]}"
            )
        per_sim = measurement.get("per_sim")
        if not isinstance(per_sim, Mapping):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} omitted simulator rows")
        spike, gsim = per_sim.get("spike"), per_sim.get("gsim")
        qualification = measurement.get("gsim_qualification")
        if not isinstance(spike, Mapping) or not isinstance(gsim, Mapping):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} is incomplete")
        if (
            not isinstance(qualification, Mapping)
            or qualification.get("admitted") is not True
            or not isinstance(qualification.get("decision"), Mapping)
            or qualification["decision"].get("selected_engine") != "gsim"
            or qualification["decision"].get("certificate_sha256") != decision.certificate_sha256
        ):
            # SAY WHY. `run_paired_perf_bench` already captures the reason `GATE.validate_execution`
            # refused into `measurement["failure"]["detail"]`, and this raise used to discard it. The
            # agent still learns nothing (it sees only the exception type through rc=125), but the
            # HOST had to reconstruct the cause by decoding receipts and reading host_refusals --
            # measured 2026-09-06, when eleven of twelve measurement calls were refused and the run
            # was then discarded for "not measuring its final candidate bytes".
            failure = measurement.get("failure") if isinstance(measurement.get("failure"), Mapping) else {}
            reason = qualification.get("reason") if isinstance(qualification, Mapping) else None
            detail = failure.get("detail") or reason
            raise StageGateError(
                f"development GSIM {arm}/{family}/{capsule} failed strict certificate admission"
                + (f": {str(detail)[:300]}" if detail else "")
            )
        cycles = gsim.get("cycles")
        ceiling_skip = self._tier_skipped_beyond_declared_ceiling(measurement, required_tiers)
        correct = (
            measurement.get("numeric") == "pass"
            and spike.get("correct") is True
            and gsim.get("correct") is True
            and (measurement.get("status") == "pass" or (measurement.get("status") == "screened_only" and ceiling_skip))
            and (not measurement.get("failure") or ceiling_skip)
        )
        if (
            isinstance(cycles, bool)
            or not isinstance(cycles, int)
            or cycles <= 0
            or decision.selected_engine != "gsim"
            or not decision.use_gsim
        ):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} lacks a positive certified cycle count")
        return {"correct": correct, "gsim_cycles": cycles}

    def profile_witness(self) -> tuple[CORPUS.PerformanceCapsule, str]:
        """The fixed reduced witness used only to explain a global candidate's cycle change.

        Selection reads the frozen descriptor and baseline-only simulation-cost history.  It is made
        without candidate output or cycles, prefers an objective member, and stays fixed for the
        stage.  Formal promotion still evaluates the entire sealed cohort.
        """
        from merlin.perf.execution_policy import require_probe_execution  # noqa: PLC0415

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

    @staticmethod
    def _physical_profile(linked: Mapping[str, Any]) -> dict[str, Any]:
        physical = linked.get("physical_byte_counters")
        physical = physical if isinstance(physical, Mapping) else {}
        facts = physical.get("counter_facts")
        readings = physical.get("readings")
        if (
            physical.get("semantic_resolution") != "rtl_bound_physical_bytes"
            or not isinstance(facts, Sequence)
            or isinstance(facts, (str, bytes))
            or not isinstance(readings, Mapping)
        ):
            return {
                "status": "UNKNOWN",
                "total_bytes": None,
                "reason": "physical counters lack exhaustive RTL-bound byte semantics",
            }
        try:
            from merlin.perf.dma_volume import physical_volume_from_counters  # noqa: PLC0415

            volume = physical_volume_from_counters(readings, counter_facts=facts)
        except Exception as exc:  # noqa: BLE001 - a failed binding remains unknown
            return {
                "status": "UNKNOWN",
                "total_bytes": None,
                "reason": f"physical byte derivation failed ({type(exc).__name__})",
            }
        if volume.total_bytes is None:
            return {"status": "UNKNOWN", "total_bytes": None, "reason": "; ".join(volume.unresolved)}
        return {
            "status": "measured",
            "total_bytes": int(volume.total_bytes),
            "read_bytes": volume.read_bytes,
            "write_bytes": volume.write_bytes,
            "basis": "identity-linked counters with byte meanings derived from exact RTL facts",
        }

    def _redact_profile(
        self, raw: Mapping[str, Any], decision: GATE.EvaluationDecision, *, arm: str, member: CORPUS.PerformanceCapsule
    ) -> dict[str, Any]:
        cycle = self._redact_execution(
            raw,
            decision,
            arm=arm,
            family=member.family,
            capsule=member.capsule,
            required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
        )
        if cycle["correct"] is not True:
            raise StageGateError(f"reduced occupancy profile {arm} is not correct")
        measurement = raw.get("measurement")
        measurement = measurement if isinstance(measurement, Mapping) else {}
        per_sim = measurement.get("per_sim")
        per_sim = per_sim if isinstance(per_sim, Mapping) else {}
        gsim = per_sim.get("gsim")
        gsim = gsim if isinstance(gsim, Mapping) else {}
        conditions = gsim.get("measurement_conditions")
        conditions = conditions if isinstance(conditions, Mapping) else {}
        if (
            conditions.get("cache_protocol") != "one_unmeasured_predecessor"
            or conditions.get("requested_cache_condition") != "warm"
        ):
            raise StageGateError("reduced occupancy profile did not prove one unmeasured warm predecessor")
        linked = measurement.get("linked_counter_evidence")
        linked = linked if isinstance(linked, Mapping) else {}
        occupancy = linked.get("occupancy")
        occupancy = occupancy if isinstance(occupancy, Mapping) else {}
        overlap = occupancy.get("overlap")
        overlap = overlap if isinstance(overlap, Mapping) else {}
        busy = overlap.get("busy_cycles") if overlap.get("state") == "measured" else None
        busy = (
            dict(sorted((str(key), int(value)) for key, value in busy.items())) if isinstance(busy, Mapping) else None
        )
        try:
            from merlin.runtime.backends.base import get_backend  # noqa: PLC0415

            reader = getattr(get_backend(self.target_experiment.target), "counter_engine_kinds", None)
            raw_kinds = reader() if callable(reader) else None
            kinds = (
                {str(key): str(getattr(value, "value", value)) for key, value in raw_kinds.items()}
                if isinstance(raw_kinds, Mapping)
                else None
            )
        except Exception:  # noqa: BLE001 - roles remain unknown; counter values are retained
            kinds = None
        command_artifact = measurement.get("command_buffer_artifact")
        command = command_artifact.get("command_buffer") if isinstance(command_artifact, Mapping) else None
        declared_commands = None
        representation = None
        if isinstance(command, Mapping):
            from merlin.perf.command_buffer_diagnostics import representation_activity  # noqa: PLC0415
            from merlin.perf.movement_volume import movement_from_command_buffer  # noqa: PLC0415

            movement = movement_from_command_buffer(command)
            declared_commands = sum(1 for row in movement.commands if ((row.bytes_in or 0) + (row.bytes_out or 0)) > 0)
            representation = representation_activity(command)
        physical = self._physical_profile(linked)
        missing: list[str] = []
        if linked.get("status") != "linked":
            missing.append("identity-linked occupancy and physical-byte counter passes")
        if busy is None or overlap.get("state") != "measured":
            missing.append("proved joint resource occupancy and compute/movement overlap")
        if kinds is None:
            missing.append("target-declared resource roles for the measured counters")
        if physical["status"] != "measured":
            missing.append("physical movement bytes")
        missing.extend(("issued movement command count", "executed encoding transition count"))
        return {
            "correct": True,
            "total_compute_cycles": cycle["gsim_cycles"],
            "resource_busy_cycles": busy,
            "resource_kinds": kinds,
            "movement_compute_overlap_cycles": overlap.get("realised_cycles"),
            "overlap_available_cycles": overlap.get("available_cycles"),
            "latency_hiding_efficiency": overlap.get("eta"),
            "physical_movement": physical,
            "declared_movement_commands": declared_commands,
            "representation_activity": representation,
            "issued_movement_commands": None,
            "executed_encoding_transitions": None,
            "measurement_conditions": dict(conditions),
            "missing": missing,
        }

    def profile(self, candidate: Path, *, round_index: int, call_index: int, timeout_s: int) -> dict[str, Any]:
        """Warm-profile one preselected reduced witness with only occupancy/movement counters."""
        candidate = Path(candidate).resolve(strict=True)
        member, selection_basis = self.profile_witness()
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
            raw = self._execute(
                arm=arm,
                package=package,
                package_sha256=digest,
                member=member,
                decision=decision,
                workspace=root / arm,
                timeout_s=remaining,
                hardware_counters=True,
            )
            return self._redact_profile(raw, decision, arm=arm, member=member)

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

    #: What a member declares itself to be FOR. Mirrors the generator's own vocabulary; a member whose
    #: frozen descriptor predates the field is treated as OBJECTIVE and counted, because silently
    #: dropping it from the total would understate the very thing the total measures.
    OBJECTIVE_CLASS = "OBJECTIVE"

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

        Delegated wholesale to :mod:`merlin.perf.select`, whose declared thresholds are exactly the
        question worth asking: has the search stopped moving (three consecutive queries improving
        less than 1%), and has it got close enough to what is actually attainable (within 10% of the
        achievable target)? A plateau alone is a weak signal -- a search can sit still far from the
        ceiling because the last few levers were bad -- so it is reported beside attainment rather
        than instead of it.

        The attainable target is the ACHIEVABLE ceiling, never the structural peak: stopping at 90%
        of a rate no program on this machine has ever reached would never fire, and stopping short
        because a nameplate number says there is headroom would never stop.
        """
        from merlin.perf import select as SELECT  # noqa: PLC0415
        from merlin.perf.budget import Budget, unpriced_channel  # noqa: PLC0415

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
        objective, undeclared = self._objective_identities()
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
            "share_of_attainable": (
                None if attainable is SELECT.UNKNOWN or not best_total else attainable / best_total
            ),
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
        self, wave, *, candidate: Path, candidate_before: str, call_root: Path, deadline: float, workers: int
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
            raw = self._execute(
                arm=arm,
                package=package,
                package_sha256=digest,
                member=member,
                decision=decision,
                workspace=call_root / FM.ARM_WORKSPACE.format(index=index, arm=arm),
                timeout_s=remaining,
            )
            return self._redact_execution(
                raw,
                decision,
                arm=arm,
                family=member.family,
                capsule=member.capsule,
                required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
            )

        jobs: dict = {}
        rows: dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            # LONGEST FIRST inside the wave, so the tail of the wave is its cheap members and the
            # makespan is not set by a slow member that was submitted last.
            for index, member in sorted(wave, key=lambda im: -float(self.member_cost.get(im[1].capsule, 0.0))):
                if (member.family, member.capsule) not in self._baseline_cache:
                    jobs[pool.submit(one, index, member, "baseline")] = (index, "baseline")
                jobs[pool.submit(one, index, member, "candidate")] = (index, "candidate")
            for future in concurrent.futures.as_completed(jobs):
                slot = jobs[future]
                try:
                    rows[slot] = future.result()
                except Exception as exc:  # noqa: BLE001 - recorded against its slot, never fatal
                    rows[slot] = {"__error__": f"{type(exc).__name__}: {str(exc)[:300]}"}
        return rows

    def _refresh_achievable(self) -> None:
        """Re-derive the ceiling and its tolerance over every measurement the agent did not author.

        Phase 1's corpus is not phase 2's. Measured 2026-09-04: the ceiling harvested from the
        functional run alone was 80.01 MACs/cycle while four performance members already ran ABOVE
        it at baseline (PC01_k128 99.79, PR01 94.06, PR03 88.33, PR02 87.96), and that set's
        dispersion of 0.472 put "already at the ceiling" at 42.25 -- so 14 of 38 members were being
        told they had no headroom left. Over the widened set the ceiling is 99.79 and the dispersion
        0.245, and 6 members read as finished. The tolerance MUST come from the same points as the
        ceiling: a wide-set ceiling with a narrow-set spread is not a measurement of anything.

        Widening can only ever RAISE the ceiling -- the rate is a max over the points, so it is
        monotone under union, and a point with positive cycles cannot falsify `demand/rate <= busy`.
        A higher ceiling makes every consumer stricter: shares fall, `no_headroom` fires less, and
        the attainment stop condition fires less. There is no direction in which this flatters a
        candidate.

        BASELINE ARMS ONLY, and that is the entire safety argument. The baseline package is phase
        1's submission, sha-pinned before the first member runs; its cycles cannot depend on
        anything the agent wrote. Admitting candidate arms would let the agent raise its own
        ceiling, pinning its candidate near 1.0 by construction while pushing every other member
        down -- a target the agent authors is not a target.
        """
        from . import calibration as PMODEL  # noqa: PLC0415
        from . import capsule_verdict as CV  # noqa: PLC0415

        baseline_points = FM.harvest_baseline_points(self.work_root)
        points = list(self.seed_points) + list(baseline_points)
        self._achievable_points = tuple(points)
        if not points:
            return
        ceiling = PMODEL.achievable_ceiling(points, provenance="functional and frozen baseline")
        if not ceiling.known:
            return  # never downgrade a known ceiling to an absent one
        self.achievable_macs_per_cycle = float(ceiling.value)
        self.achievable_dispersion = CV.ceiling_dispersion([{"macs": p.macs, "cycles": p.cycles} for p in points])
        self.achievable_basis = (
            f"best rate over {len(points)} measured point(s): {len(self.seed_points)} from the "
            f"phase-1 functional run {self.functional_run_id} and {len(baseline_points)} from the "
            f"frozen-baseline arm of the performance corpus (baseline arms only; no candidate "
            f"measurement contributes to this ceiling)"
        )

    def _matched_achievable(self, member: CORPUS.PerformanceCapsule) -> tuple[float | None, str, float | None]:
        """Best measured rate for the member's exact contraction-depth signature.

        Reduction depth is the amortisation axis for a weight-stationary contraction.  A global max
        over unrelated K values is useful context, but it is not an attainable target for this
        member.  Both signatures come from declared/emitted operand geometry; absence refuses rather
        than silently falling back to the flattering global number.
        """
        from . import calibration as PMODEL  # noqa: PLC0415
        from . import capsule_verdict as CV  # noqa: PLC0415

        signature, signature_basis = FM.declared_reduction_depths(member.descriptor)
        if not signature:
            return None, f"member-matched achievable rate unavailable: {signature_basis}", None
        matched = [
            point for point in self._achievable_points if tuple(getattr(point, "reduction_depths", ())) == signature
        ]
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

    def evaluate(self, candidate: Path, *, round_index: int, call_index: int, timeout_s: int) -> dict[str, Any]:
        candidate = Path(candidate).resolve(strict=True)
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
        measured_candidate = call_root / "_measured_candidate"
        shutil.copytree(candidate, measured_candidate, symlinks=True)
        candidate_before = str(hash_tree(measured_candidate)["sha256"])
        candidate = measured_candidate
        members = sorted(self.corpus.capsules, key=lambda row: (row.family, row.capsule))
        if not members:
            raise StageGateError("development GSIM feedback has zero frozen tuning members")
        cells: list[dict[str, Any]] = []
        started = time.monotonic()
        # CHEAPEST MEASURED MEMBER FIRST. A candidate behind on every member measured so far is
        # behind; paying for the corpus's slowest members to confirm it spends the budget on a
        # conclusion already reached.
        members, order_basis = FM.order_members_by_cost(members, self.member_cost)
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
                    self._prefetch_wave(
                        wave,
                        candidate=candidate,
                        candidate_before=candidate_before,
                        call_root=call_root,
                        deadline=deadline,
                        workers=workers,
                    )
                )
            baseline = self._baseline_cache.get(key) or self._take_prefetched(index, "baseline")
            if baseline is None:
                raw = self._execute(
                    arm="baseline",
                    package=self.baseline,
                    package_sha256=self.baseline_sha256,
                    member=member,
                    decision=decision,
                    workspace=call_root / FM.ARM_WORKSPACE.format(index=index, arm="baseline"),
                    timeout_s=remaining,
                )
                baseline = self._redact_execution(
                    raw,
                    decision,
                    arm="baseline",
                    family=member.family,
                    capsule=member.capsule,
                    required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
                )
                self._baseline_cache[key] = baseline
            remaining = timeout_s - int(time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
            candidate_row = self._take_prefetched(index, "candidate")
            if candidate_row is None:
                raw = self._execute(
                    arm="candidate",
                    package=candidate,
                    package_sha256=candidate_before,
                    member=member,
                    decision=decision,
                    workspace=call_root / FM.ARM_WORKSPACE.format(index=index, arm="candidate"),
                    timeout_s=remaining,
                )
                candidate_row = self._redact_execution(
                    raw,
                    decision,
                    arm="candidate",
                    family=member.family,
                    capsule=member.capsule,
                    required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
                )
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

            achievable, achievable_basis, matched_dispersion = self._matched_achievable(member)
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
            if self._refuted_so_far(cells, index, len(members)):
                stopped_after = index + 1
                break
        for member in members[stopped_after:] if stopped_after is not None else ():
            # RECORDED, not omitted. A missing cell and an unmeasured one are different claims.
            cells.append(
                FM.unmeasured_cell(
                    member,
                    reason=(
                        "the sweep is ordered cheapest-measured-first and this candidate "
                        "was already behind on every comparable member measured before "
                        "this one; the remaining members were not paid for"
                    ),
                )
            )
        # The snapshot must still be the bytes the runs read: nothing here may edit it, and a
        # difference now would mean the measurement mutated its own input rather than that the agent
        # kept working. That is a real defect and still refuses.
        # THE CEILING IS RE-DERIVED FROM THE BASELINES THIS SWEEP JUST MEASURED, then every field
        # that depends on it is recomputed. Doing it only at prepare time would leave round 0 -- the
        # round that sets the agent's whole plan -- scored against phase 1's corpus, which is the
        # case that was actually wrong.
        self._refresh_achievable()
        members_by_identity = {(member.family, member.capsule): member for member in members}
        for row in cells:
            if not row.get("measured") or not row.get("declared_macs"):
                continue  # an unmeasured cell keeps its nulls
            member = members_by_identity[(row["family"], row["capsule"])]
            rate, basis, dispersion = self._matched_achievable(member)
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
                "stopping": self._stopping(
                    cells, label=f"round_{round_index:02d}/call_{call_index:03d}", elapsed_s=time.monotonic() - started
                ),
                "summary": {
                    "members": len(cells),
                    "comparable": len(comparable),
                    "all_correct": all(row["comparable"] for row in cells if row["measured"]),
                    "peak_macs_per_cycle": self.peak_macs_per_cycle,
                    "peak_basis": self.peak_basis,
                    "achievable_macs_per_cycle": self.achievable_macs_per_cycle,
                    "achievable_basis": self.achievable_basis,
                    "recoverable": FM.recoverable_cycles(cells),
                },
            }
        )


def development_executor(
    *,
    contract_root: Path,
    arm,
    package,
    package_sha256,
    member,
    decision,
    workspace,
    timeout_s,
    certificate,
    target_experiment,
    rtl_identity,
    hardware_counters=False,
):
    # The installed engine uses the package-sandboxed Spike+GSIM path;
    # the trusted caller supplies the contract resource root explicitly.

    spec = PME.ExecutionSpec(
        execution_index=0,
        pair_index=0,
        pair_id=f"{member.family}__{member.capsule}__development",
        phase="tuning",
        arm=arm,
        family=member.family,
        capsule=member.capsule,
        replicate="r000",
        package=package,
        package_sha256=package_sha256,
        member=member,
        workload=PME.gsim_workload(member),
        gsim_decision=decision,
        gsim_certificate=certificate,
    )
    return PME.run_execution(
        spec,
        workspace,
        timeout_s,
        target_experiment,
        rtl_identity,
        contract_root=contract_root,
        hardware_counters=hardware_counters,
        workers=1 if hardware_counters else sweep_workers(),
    )


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
) -> DevelopmentGsimFeedback:
    """Bind one stage to a strict certificate and exact frozen tuning corpus."""
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
        # THE ACHIEVABLE CEILING, from cycles phase 1 already paid for. The structural peak is what
        # the array could retire if nothing ever stalled; the achievable
        # ceiling is the best rate anything on this machine actually reached, and it is what the
        # agent is asked to close on. Underivable -> None with a reason, never a substituted number.
        # Ordering the sweep by measured cost needs history; absent it the declared order stands
        # and the basis says so, rather than a proxy silently standing in for a measurement.
        member_cost: dict[str, float] = FM.harvest_member_cost(
            [Path(functional_run_dir).parent] if functional_run_dir is not None else []
        )
        _, member_cost_basis = FM.order_members_by_cost((), member_cost)
        achievable_macs, achievable_basis = None, "no functional run was supplied to harvest"
        achievable_dispersion: float | None = None
        seed_points: tuple = ()
        if functional_run_dir is not None:
            try:
                from merlin_experiments.phase2 import calibration as PMODEL  # noqa: PLC0415

                points, _skipped = PMODEL.harvest_measured_points(Path(functional_run_dir))
                ceiling = PMODEL.achievable_ceiling(
                    points, provenance=f"measured cycles harvested from {Path(functional_run_dir).name}"
                )
                try:
                    from merlin_experiments.phase2 import capsule_verdict as CV  # noqa: PLC0415

                    achievable_dispersion = CV.ceiling_dispersion(
                        [{"macs": p.macs, "cycles": p.cycles} for p in points]
                    )
                except Exception:  # noqa: BLE001 - an underivable spread refuses, never defaults
                    achievable_dispersion = None
                seed_points = tuple(points)
                if ceiling.known:
                    achievable_macs = float(ceiling.value)
                    achievable_basis = (
                        f"best rate over {ceiling.n_samples} measured points in {Path(functional_run_dir).name}"
                    )
                else:
                    achievable_basis = ceiling.reason
            except Exception as exc:  # noqa: BLE001 - an unharvestable corpus is reported, not faked
                achievable_basis = f"harvest failed ({type(exc).__name__})"
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
            if (
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
        achievable_macs_per_cycle=achievable_macs,
        achievable_basis=achievable_basis,
        achievable_dispersion=achievable_dispersion,
        seed_points=seed_points,
        _achievable_points=seed_points,
        functional_run_id=(Path(functional_run_dir).name if functional_run_dir is not None else ""),
        member_cost=member_cost,
        member_cost_basis=member_cost_basis,
        executor=partial(development_executor, contract_root=contract_root),
        tuning_call_budget=tuning_call_budget,
    )
