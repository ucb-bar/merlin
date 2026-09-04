#!/usr/bin/env python3
"""Create one auditable performance candidate from an exact functional fork.

This stage deliberately does not grade or promote anything.  It gives Codex a fresh copy of a
functionally certified package and an answer-free view of the generated performance contracts.  Any
compiler/tool execution requested by the agent goes through a second, credential-free
bwrap broker.  When the bounded authoring rounds end, the candidate is copied to a read-only snapshot
and described by a content-addressed record for :mod:`run_perf_bench` to consume.

There are two distinct filesystem and credential boundaries.  The outer Codex control plane gets one
isolated ``CODEX_HOME`` plus the explicit authentication mount and the functional run's frozen authoring
grants, but no live descriptor-derived target toolchain.  The inner execution plane gets that live
descriptor-derived toolchain and the writable candidate, but has ``--clearenv``
and receives no credential bind.  Network is available in both planes and is explicitly not claimed as
an isolation property; the experiment's protection comes from exact mounts, masks, and audited routes.
"""
from __future__ import annotations

import argparse
import ast
import copy
import contextlib
import hashlib
import inspect
import json
import math
import os
import tempfile
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

import yaml

import perf_campaign as PC
import perf_gsim_gate as GATE
import perf_pk_claim as PK
import perf_prompt as PP
from merlin.benchharness import hash_tree, runs_root
from merlin.common.paths import merlin_dir, repo_root
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import toolchain as TC
from merlin.targetgen.sandbox.answer_surfaces import answer_surfaces, audit_tokens
from merlin.targetgen.target_experiment import TargetExperiment, load_target_experiment


SCHEMA_VERSION = 3
AGENT_CORPUS_MOUNT = Path("/perf-corpus")
FUNCTIONAL_BASE_MOUNT = Path("/perf-functional-base")
FUNCTIONAL_INPUT_MANIFEST_MOUNT = Path("/perf-functional-inputs/snapshot.json")
PERF_CORPUS_MANIFEST_MOUNT = Path("/perf-corpus-manifest.json")
BROKER_NAME = "/perf-control/perf_tool.py"
BROKER_RECEIPT_MOUNT = Path("/perf-control/receipts.jsonl")
DEVELOPMENT_FEEDBACK_ACTION = "tuning-gsim-feedback"
_HOST_FEEDBACK_SENTINEL = "__host_owned_tuning_gsim_feedback__"
#: A FREE, ORDERING-ONLY analysis of two emitted command buffers. The measured feedback above costs
#: ~110 s a call and was the ONLY way to judge a candidate, so every dead end was paid for at full
#: price -- measured across three trials, two excursions of +5.9% and +11.1% cost ~220 s of oracle
#: time to discover. This prices the agent's own emitted artifacts instead: no oracle, no goldens,
#: no holdout, so it can be called as often as the agent likes and can leak nothing.
ANALYSIS_ACTION = "analyze-command-buffers"

#: The verdict every command-buffer-readable ordering signal earned, and the evidence behind it.
#: Held out by workload: parameters fitted on one half, every rate below measured on the other.
#: Regenerate with `validate_ordering_signals.py`; do not edit these by hand to match a hope.
ORDERING_REFUSED = "refused_no_signal_beat_chance"
ORDERING_EVIDENCE = {
    "held_out_pairs": 478,
    "held_out_workloads": 18,
    "artifact": "out/artifacts/perf-bench/<target>/ordering_signal_validation.json",
    "agreement": {
        # signal: (agreed, decided, why it was refused)
        "depgraph_makespan": [226, 452, "0.500 -- exactly chance"],
        "command_count": [198, 421, "0.470 -- below chance"],
        "tile_pressure": [203, 279, "0.728 overall, but 0.273 on one workload with 33 decided "
                                    "pairs, where it points backwards"],
        "barrier_count": [24, 37, "0.649 on too few decided pairs, from a single slice"],
        "depgraph_critical_path": [17, 27, "0.630 on too few decided pairs, from a single slice"],
    },
}
_HOST_ANALYSIS_SENTINEL = "__host_owned_command_buffer_analysis__"
_HEX = frozenset("0123456789abcdef")
_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
TELEMETRY_TREATMENT_SOURCES = frozenset({
    "codex_binary",
    "performance_authoring_stage",
    "performance_campaign",
    "performance_gsim_gate",
    "performance_pk_claim",
    "performance_prompt",
    "codex_driver",
    "codex_model_bridge",
    "benchharness",
    "sandbox_bwrap",
    "sandbox_toolchain",
    "sandbox_answer_surfaces",
    "target_experiment_loader",
    "experiment_tokens",
    "aet_codex_normalizer",
    "aet_codex_importer",
    "aet_reconciliation",
    "aet_activity_classifier",
    "aet_canonical_logger",
})


class StageGateError(RuntimeError):
    """The candidate cannot be launched or admitted without weakening an experiment gate."""


@dataclass(frozen=True)
class AgentSandboxPolicy:
    argv: tuple[str, ...]
    answer_surface_gap: tuple[str, ...]
    network: str
    clear_environment: bool
    candidate_writable: bool
    corpus_read_only: bool


@dataclass(frozen=True)
class AgentInputSnapshot:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    content_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class PerformanceCapsule:
    """Minimal frozen-capsule view consumed by the authoring stage.

    Kept local so the authoring boundary does not require the measurement
    campaign module to expose its discovery implementation.
    """

    family: str
    capsule: str
    source_dir: Path
    source_relative_path: str
    descriptor: dict[str, Any]
    source_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class FrozenPerformanceCorpus:
    """Content-addressed performance inputs needed by the agent-only view."""

    root: Path
    capsules_root: Path
    manifest_path: Path
    manifest_sha256: str
    capsules_sha256: str
    capsules: tuple[PerformanceCapsule, ...]


@dataclass(frozen=True)
class PerformanceCorpus:
    target: str
    corpus_root: Path
    phase_root: Path
    provenance_manifest: Path
    provenance_sha256: str
    performance_generation: dict[str, Any]
    capsules: tuple[PerformanceCapsule, ...]


@dataclass(frozen=True)
class StageFunctionalRun:
    """Current campaign verdict plus the authoring inputs it did not expose."""

    run_dir: Path
    submission_dir: Path
    run_id: str
    digest: str
    public_capsules: int
    hidden_capsules: int
    public_score: dict[str, Any]
    hidden_score: dict[str, Any]
    frozen_at: str
    bundle_input_snapshot: dict[str, Any]
    model_host_lane_snapshot: dict[str, Any]
    model_host_package: Path


@dataclass(frozen=True)
class FullModelSentinel:
    capsule: str
    source_dir: Path
    descriptor: dict[str, Any]
    source_sha256: str
    n_files: int
    n_bytes: int


@dataclass(frozen=True)
class PerformanceFamilyDeclaration:
    """Frozen family facts retained even when the shared prompt API is older."""

    family: str
    claim: str
    negative_control: str
    falsifier_observation: str
    differential_basis: str
    fitted_parameters: tuple[str, ...] = ()
    acceptance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class StageHostLaneGrant:
    target: str
    package_id: str
    package_path: str
    package_sha256: str
    manifest_path: str
    integration_seam: str


@dataclass(frozen=True)
class StageE2ESentinel:
    capsule: str
    capsule_path: str
    frozen_source_path: str
    capsule_sha256: str
    required_lanes: tuple[str, ...]
    required_tiers: tuple[str, ...]


@dataclass(frozen=True)
class StagePromptInputs:
    target: str
    approach: str
    functional_run_id: str
    functional_submission_sha256: str
    frozen_functional_path: str
    frozen_functional_sha256: str
    submission_path: str
    submission_initial_sha256: str
    functional_public_capsules: int
    functional_hidden_capsules: int
    functional_bundle_snapshot_manifest: str
    functional_bundle_snapshot_manifest_sha256: str
    functional_bundle_snapshot_sha256: str
    workload_root: str
    workload_manifest: str
    workload_manifest_sha256: str
    workload_capsules_sha256: str
    expected_cells: tuple[PP.PerfCell, ...]
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    formal_claim: Mapping[str, Any]
    smoke_replicates: int
    wall_budget_seconds: int
    rounds: int
    round_timeout_seconds: int
    max_tool_calls: int
    tool_timeout_seconds: int
    families: tuple[PerformanceFamilyDeclaration, ...]
    host_lane: StageHostLaneGrant
    e2e_sentinel: StageE2ESentinel
    tools: tuple[PP.ToolGrant, ...]
    allowed_paths: tuple[str, ...]
    execution_broker_path: str
    execution_broker_command: str
    broker_receipt_path: str


@dataclass(frozen=True)
class PromptArtifact:
    source_path: Path
    text: str
    sha256: str
    n_bytes: int


@dataclass(frozen=True)
class FrozenGrant:
    declared_path: str
    destination: Path
    source: Path
    source_sha256: str


@dataclass(frozen=True)
class FrozenFunctionalInputs:
    root: Path
    marker: Path
    marker_sha256: str
    content_sha256: str
    grants: tuple[FrozenGrant, ...]


@dataclass(frozen=True)
class BrokerAction:
    name: str
    argv_template: tuple[str, ...]
    placeholders: tuple[str, ...]
    purpose: str
    required: bool

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "argv_template": list(self.argv_template),
                "placeholders": list(self.placeholders), "purpose": self.purpose,
                "required": self.required}


@dataclass
class DevelopmentGsimFeedback:
    """Host-owned, tuning-only evaluation callable used by the broker.

    The full runner result remains outside the agent mount.  ``evaluate`` emits
    only correctness, GSIM cycles, and paired deltas after the strict GSIM gate
    has accepted every exact workload and execution.
    """

    certificate: GATE.CertificateRecord
    corpus: FrozenPerformanceCorpus
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
    #: Median measured simulation seconds per capsule, and how the ordering was arrived at. Used to
    #: sweep cheapest-first, so a losing candidate is refuted before the corpus's slowest members
    #: are paid for. Empty means no history: the declared order stands and the basis says so.
    member_cost: Mapping[str, float] = field(default_factory=dict)
    member_cost_basis: str = ""
    tuning_call_budget: int | None = None
    #: Total candidate cycles over the comparable members, one entry per feedback invocation. The
    #: stop conditions read the SHAPE of this history, not any single measurement.
    _totals: "list[float] | None" = None
    #: ``(label, seconds)`` per feedback invocation -- the measurements this search actually spent.
    #: The budget ledger is rebuilt from this every call, so ``budget_exhausted`` judges real spend
    #: rather than reporting an untouched ledger.
    _spend: "list[tuple[str, float]] | None" = None
    executor: Callable[..., Mapping[str, Any]] | None = None
    _baseline_cache: dict[tuple[str, str], dict[str, Any]] | None = None

    def _execute(self, *, arm: str, package: Path, package_sha256: str,
                 member: PerformanceCapsule, decision: GATE.EvaluationDecision,
                 workspace: Path, timeout_s: int) -> Mapping[str, Any]:
        if self.executor is not None:
            return self.executor(
                arm=arm, package=package, package_sha256=package_sha256, member=member,
                decision=decision, workspace=workspace, timeout_s=timeout_s,
                certificate=self.certificate, target_experiment=self.target_experiment,
                rtl_identity=self.rtl_identity)
        # Lazy import avoids the paired runner's import of this module during
        # stage startup.  Its run_execution path is the package-sandboxed Arm4
        # Spike+GSIM path; no simulator binary or raw output reaches Codex.
        import run_paired_perf_bench as PAIR
        spec = PAIR.ExecutionSpec(
            execution_index=0, pair_index=0,
            pair_id=f"{member.family}__{member.capsule}__development", phase="tuning",
            arm=arm, family=member.family, capsule=member.capsule, replicate="r000",
            package=package, package_sha256=package_sha256,
            member=member, workload=PAIR._gsim_workload(member), gsim_decision=decision,
            gsim_certificate=self.certificate)
        return PAIR.run_execution(
            spec, workspace, timeout_s, self.target_experiment, self.rtl_identity,
            hardware_counters=False)

    @staticmethod
    @staticmethod
    def _tier_skipped_beyond_declared_ceiling(measurement: Mapping[str, Any],
                                              required_tiers: Sequence[str]) -> bool:
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

    def _redact_execution(self, raw: Mapping[str, Any], decision: GATE.EvaluationDecision, *,
                          arm: str, family: str, capsule: str,
                          required_tiers: Sequence[str] = ()) -> dict[str, Any]:
        """Reduce a host result to the only feedback fields the agent may see."""
        measurement = raw.get("measurement")
        if not isinstance(measurement, Mapping):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} returned no measurement")
        per_sim = measurement.get("per_sim")
        if not isinstance(per_sim, Mapping):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} omitted simulator rows")
        spike, gsim = per_sim.get("spike"), per_sim.get("gsim")
        qualification = measurement.get("gsim_qualification")
        if not isinstance(spike, Mapping) or not isinstance(gsim, Mapping):
            raise StageGateError(f"development GSIM {arm}/{family}/{capsule} is incomplete")
        if (not isinstance(qualification, Mapping) or qualification.get("admitted") is not True
                or not isinstance(qualification.get("decision"), Mapping)
                or qualification["decision"].get("selected_engine") != "gsim"
                or qualification["decision"].get("certificate_sha256")
                != decision.certificate_sha256):
            raise StageGateError(
                f"development GSIM {arm}/{family}/{capsule} failed strict certificate admission")
        cycles = gsim.get("cycles")
        ceiling_skip = self._tier_skipped_beyond_declared_ceiling(measurement, required_tiers)
        correct = (measurement.get("numeric") == "pass"
                   and spike.get("correct") is True and gsim.get("correct") is True
                   and (measurement.get("status") == "pass"
                        or (measurement.get("status") == "screened_only" and ceiling_skip))
                   and (not measurement.get("failure") or ceiling_skip))
        if (isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0
                or decision.selected_engine != "gsim" or not decision.use_gsim):
            raise StageGateError(
                f"development GSIM {arm}/{family}/{capsule} lacks a positive certified cycle count")
        return {"correct": correct, "gsim_cycles": cycles}


    #: A losing prefix must be this long before it may stop the sweep. One member is an anecdote --
    #: the cheapest member is also the one most dominated by fixed per-invocation cost, where a
    #: schedule change shows least -- so a single loss is not allowed to end a measurement.
    MINIMUM_REFUTING_PREFIX = 3

    def _refuted_so_far(self, cells: "Sequence[Mapping[str, Any]]", index: int, total: int) -> bool:
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
            return False                      # nothing left to save
        comparable = [c for c in cells if c.get("comparable")]
        if len(comparable) < self.MINIMUM_REFUTING_PREFIX or len(comparable) != len(cells):
            return False                      # an incomparable member means the picture is partial
        for cell in comparable:
            delta = cell.get("candidate_minus_baseline_cycles")
            if not isinstance(delta, (int, float)) or isinstance(delta, bool) or delta <= 0:
                return False                  # a tie or a win anywhere: keep measuring
        return True

    def _stopping(self, cells: Sequence[Mapping[str, Any]], *,
                  label: str, elapsed_s: float) -> dict[str, Any]:
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
            return {"status": "undeterminable",
                    "reason": "no member is comparable, so there is no measured total to judge",
                    "verdicts": []}
        baseline_total = float(sum(c["baseline_gsim_cycles"] for c in comparable))
        candidate_total = float(sum(c["candidate_gsim_cycles"] for c in comparable))
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

        attainable = SELECT.UNKNOWN
        macs = [c.get("declared_macs") for c in comparable]
        if self.achievable_macs_per_cycle and all(isinstance(m, int) and m > 0 for m in macs):
            attainable = float(sum(macs)) / float(self.achievable_macs_per_cycle)

        if self._spend is None:
            self._spend = []
        self._spend.append((label, max(0.0, float(elapsed_s))))
        budget = Budget(unit=unpriced_channel(
            "tuning_gsim_feedback",
            missing="the per-call price of a brokered tuning measurement is not measured here"),
            limit_items=self.tuning_call_budget)
        # One charge per measurement actually taken, with its measured wall seconds. Rebuilt from
        # the history because the budget is constructed fresh on every invocation.
        for spent_label, spent_seconds in self._spend:
            budget.charge(items=1.0, seconds=spent_seconds, label=spent_label)
        state = SELECT.SearchState(
            baseline_cycles=int(baseline_total), best_cycles=best_total,
            budget=budget,
            attainable_cycles=attainable,
            improvements=tuple(improvements))
        verdicts = SELECT.check_stop(state)
        stops = SELECT.fired(verdicts)
        return {
            "status": "stop" if stops else "continue",
            "queries": state.queries,
            "baseline_total_cycles": baseline_total,
            "best_total_cycles": best_total,
            "previous_best_total_cycles": previous_best,
            "attainable_total_cycles": (None if attainable is SELECT.UNKNOWN else attainable),
            "share_of_attainable": (None if attainable is SELECT.UNKNOWN or not best_total
                                    else attainable / best_total),
            "budget": budget.to_dict(),
            "verdicts": [v.to_dict() for v in verdicts],
        }

    def evaluate(self, candidate: Path, *, round_index: int, call_index: int,
                 timeout_s: int) -> dict[str, Any]:
        candidate = Path(candidate).resolve(strict=True)
        candidate_before = str(hash_tree(candidate)["sha256"])
        if self._baseline_cache is None:
            self._baseline_cache = {}
        call_root = self.work_root / f"round_{round_index:02d}" / f"call_{call_index:03d}"
        if call_root.exists() or call_root.is_symlink():
            raise StageGateError(f"development GSIM feedback workspace is not fresh: {call_root}")
        call_root.mkdir(parents=True)
        members = sorted(self.corpus.capsules, key=lambda row: (row.family, row.capsule))
        if not members:
            raise StageGateError("development GSIM feedback has zero frozen tuning members")
        cells: list[dict[str, Any]] = []
        started = time.monotonic()
        # CHEAPEST MEASURED MEMBER FIRST. A candidate behind on every member measured so far is
        # behind; paying for the corpus's slowest members to confirm it spends the budget on a
        # conclusion already reached.
        members, order_basis = order_members_by_cost(members, self.member_cost)
        stopped_after: int | None = None
        for index, member in enumerate(members):
            key = (member.family, member.capsule)
            decision = self.decisions.get(key)
            if decision is None:
                raise StageGateError(f"development GSIM decision is absent for {key}")
            remaining = timeout_s - int(time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
            baseline = self._baseline_cache.get(key)
            if baseline is None:
                raw = self._execute(
                    arm="baseline", package=self.baseline, package_sha256=self.baseline_sha256,
                    member=member, decision=decision,
                    workspace=call_root / f"m{index:03d}_baseline", timeout_s=remaining)
                baseline = self._redact_execution(
                    raw, decision, arm="baseline", family=member.family, capsule=member.capsule,
                    required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()))
                self._baseline_cache[key] = baseline
            remaining = timeout_s - int(time.monotonic() - started)
            if remaining <= 0:
                raise StageGateError("development GSIM feedback exceeded its deterministic timeout")
            raw = self._execute(
                arm="candidate", package=candidate, package_sha256=candidate_before,
                member=member, decision=decision,
                workspace=call_root / f"m{index:03d}_candidate", timeout_s=remaining)
            candidate_row = self._redact_execution(
                raw, decision, arm="candidate", family=member.family, capsule=member.capsule,
                required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()))
            comparable = baseline["correct"] and candidate_row["correct"]
            bcycles, ccycles = baseline["gsim_cycles"], candidate_row["gsim_cycles"]
            # UTILIZATION against a ceiling this machine's own RTL derives. Cycles alone say nothing
            # about how much of the machine a program used, and a family whose claim is a fit can be
            # satisfied by a program that uses LESS of it. Both inputs are derived and host-computed:
            # the required work from the capsule's declared operands, the peak from facts.arrays.
            # Either being underivable yields null and a reason, never an assumed number.
            spec_macs, work_basis = declared_capsule_macs(member.descriptor)
            peak = self.peak_macs_per_cycle
            ideal = (spec_macs / peak) if (spec_macs and peak) else None

            def _utilization(cycles: Any) -> float | None:
                if ideal is None or not isinstance(cycles, int) or isinstance(cycles, bool):
                    return None
                return (ideal / cycles) if cycles > 0 else None

            achievable = self.achievable_macs_per_cycle
            achievable_ideal = (spec_macs / achievable) if (spec_macs and achievable) else None

            def _share(cycles: Any) -> float | None:
                if achievable_ideal is None or not isinstance(cycles, int) or isinstance(cycles, bool):
                    return None
                return (achievable_ideal / cycles) if cycles > 0 else None

            cells.append({
                "family": member.family, "capsule": member.capsule,
                "baseline_correct": baseline["correct"],
                "candidate_correct": candidate_row["correct"],
                "baseline_gsim_cycles": bcycles, "candidate_gsim_cycles": ccycles,
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
                **_capsule_verdict_fields(
                    capsule=member.capsule, declared_macs=spec_macs,
                    achievable_rate=self.achievable_macs_per_cycle,
                    baseline_cycles=bcycles, candidate_cycles=ccycles if comparable else None,
                    dispersion=self.achievable_dispersion),
                "measured": True, "skip_reason": None,
            })
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
            cells.append(_unmeasured_cell(
                member, reason=("the sweep is ordered cheapest-measured-first and this candidate "
                                "was already behind on every comparable member measured before "
                                "this one; the remaining members were not paid for")))
        candidate_after = str(hash_tree(candidate)["sha256"])
        if candidate_after != candidate_before:
            raise StageGateError("development GSIM evaluation mutated the candidate compiler")
        comparable = [row for row in cells if row["comparable"]]
        return validate_redacted_feedback({
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
                cells, label=f"round_{round_index:02d}/call_{call_index:03d}",
                elapsed_s=time.monotonic() - started),
            "summary": {"members": len(cells), "comparable": len(comparable),
                        "all_correct": len(comparable) == len(cells),
                        "peak_macs_per_cycle": self.peak_macs_per_cycle,
                        "peak_basis": self.peak_basis,
                        "achievable_macs_per_cycle": self.achievable_macs_per_cycle,
                        "achievable_basis": self.achievable_basis},
        })


@dataclass(frozen=True)
class VerifiedCandidateHandoff:
    """Narrow, stable measurement boundary produced only from a validated record."""

    record_path: Path
    record_sha256: str
    candidate_path: Path
    candidate_sha256: str
    candidate_initial_sha256: str
    functional_run_id: str
    functional_submission_sha256: str
    functional_base_path: Path
    functional_bundle_snapshot_sha256: str
    functional_bundle_manifest: Path
    functional_bundle_manifest_sha256: str
    target_descriptor: Path
    target_descriptor_sha256: str
    corpus_root: Path
    corpus_manifest: Path
    corpus_manifest_sha256: str
    corpus_sha256: str
    replicates: int
    formal_replicate_identities: tuple[str, ...]
    formal_claim: dict[str, Any]
    smoke_replicates: int
    expected_cells: tuple[dict[str, str], ...]
    families: tuple[dict[str, Any], ...]
    host_lane: dict[str, Any]
    e2e_sentinel: dict[str, Any]
    prompt_sha256: str
    prompt_facts_sha256: str
    prompt_path: Path
    transcript_path: Path
    transcript_sha256: str
    transcript_audit: dict[str, Any]
    receipt_path: Path
    receipt_sha256: str
    required_actions: tuple[str, ...]
    tool_evidence: dict[str, Any]
    sandbox_evidence: dict[str, Any]
    telemetry_evidence: dict[str, Any]
    codex_binary_sha256: str
    authoring_stage_sha256: str
    telemetry_preflight_sha256: str
    telemetry_source_sha256: dict[str, str]
    agent_contract: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"content-addressed evidence is absent or linked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64 and value.lower() == value
            and all(character in _HEX for character in value))


def _canonical_json(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            + "\n").encode("utf-8")


def _document_sha256(document: object) -> str:
    return _sha256(_canonical_json(document).rstrip(b"\n"))


def _exact_tree_record(root: Path) -> dict[str, Any]:
    """Hash all path and file bytes; reject links, special files, and emptiness."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise StageGateError(f"exact input is absent or linked: {root}")
    digest = hashlib.sha256()
    n_files = n_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise StageGateError(f"exact input contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise StageGateError(f"exact input contains a special file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(relative + b"\0" + payload + b"\0")
        n_files += 1
        n_bytes += len(payload)
    if n_files <= 0:
        raise StageGateError(f"exact input contains zero files: {root}")
    return {"sha256": digest.hexdigest(), "n_files": n_files, "n_bytes": n_bytes}


def _mapping_file(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"required stage input is absent or linked: {path}")
    try:
        document = (yaml.safe_load(path.read_text(encoding="utf-8")) if yaml_file
                    else json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise StageGateError(f"stage input is unreadable at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise StageGateError(f"stage input must be a mapping: {path}")
    return document


def _safe_relative(value: object, *, label: str) -> Path:
    if not isinstance(value, str):
        raise StageGateError(f"{label} must be repository-relative")
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise StageGateError(f"{label} must be a safe repository-relative path")
    return path


def _require_read_only_tree(root: Path, *, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise StageGateError(f"{label} is absent or linked: {root}")
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            raise StageGateError(f"{label} contains a symlink: {path}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise StageGateError(f"{label} is writable: {path}")


def _functional_input_snapshot(base: PC.FunctionalRun) -> StageFunctionalRun:
    """Join the current functional gate to its sealed authoring/host snapshot."""
    environment = _mapping_file(base.run_dir / "environment.yaml", yaml_file=True)
    snapshot = environment.get("bundle_input_snapshot")
    host = environment.get("model_host_lane_snapshot")
    if (not isinstance(snapshot, Mapping) or snapshot.get("version") != 2
            or not _is_sha256(snapshot.get("content_sha256"))
            or not isinstance(host, Mapping) or host.get("run_snapshot") != dict(snapshot)):
        raise StageGateError(
            "functional verdict lacks the exact v2 authoring and host-lane snapshot")
    raw_root = snapshot.get("path")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise StageGateError("functional input snapshot path is not absolute")
    root = Path(raw_root)
    _require_read_only_tree(root, label="functional input snapshot")
    marker = _mapping_file(root / "snapshot.json")
    for field in ("content_sha256", "n_files", "n_bytes"):
        if marker.get(field) != snapshot.get(field):
            raise StageGateError("functional input snapshot marker disagrees with the run record")
    package_rel = _safe_relative(host.get("package"), label="model host-lane package")
    repo = (root / "repo").resolve(strict=True)
    package = (repo / package_rel).resolve(strict=True)
    try:
        package.relative_to(repo)
    except ValueError as exc:
        raise StageGateError("model host-lane package escapes the frozen repository") from exc
    if package.is_symlink() or not package.is_dir():
        raise StageGateError("model host-lane package is absent or linked")
    _require_read_only_tree(package, label="model host-lane package")
    if host.get("resolved_package") != str(package):
        raise StageGateError("model host-lane record names different frozen bytes")
    package_record = hash_tree(package)
    if (host.get("package_sha256") != package_record.get("sha256")
            or host.get("n_files") != package_record.get("n_files")):
        raise StageGateError("model host-lane digest disagrees with its frozen package")
    return StageFunctionalRun(
        base.run_dir, base.submission_dir, base.run_id, base.digest,
        base.public_capsules, base.hidden_capsules, base.public_score,
        base.hidden_score, base.frozen_at, dict(snapshot), dict(host), package)


def inspect_stage_functional_run(
        run_root: Path, run_id: str, expected_digest: str, *,
        waive: "frozenset[str] | tuple[str, ...] | None" = None) -> StageFunctionalRun:
    """The stage's view of the functional baseline.

    ``waive`` is passed straight through to :func:`perf_campaign.inspect_functional_run`, which decides
    what may be waived at all -- integrity predicates refuse the waiver itself. Threading it rather
    than re-deciding here keeps ONE place that knows which gaps are acceptable; a second opinion in
    this file is how the two would drift.
    """
    return _functional_input_snapshot(
        PC.inspect_functional_run(run_root, run_id, expected_digest, waive=waive))


def verify_functional_host_lane_snapshot(host: Mapping[str, Any]) -> None:
    run_snapshot = host.get("run_snapshot")
    if not isinstance(run_snapshot, Mapping):
        raise StageGateError("model host-lane record omits its run snapshot")
    raw_root = run_snapshot.get("path")
    if not isinstance(raw_root, str):
        raise StageGateError("model host-lane run snapshot has no path")
    root = Path(raw_root)
    _require_read_only_tree(root, label="functional input snapshot")
    package = Path(str(host.get("resolved_package") or ""))
    if package.is_symlink() or not package.is_dir():
        raise StageGateError("model host-lane package is absent or linked")
    if hash_tree(package).get("sha256") != host.get("package_sha256"):
        raise StageGateError("model host-lane package digest changed")


def _tree_files(root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise StageGateError(f"candidate tree contains a symlink: {path}")
        if path.is_file():
            rows[path.relative_to(root).as_posix()] = _sha256_file(path)
    if not rows:
        raise StageGateError(f"candidate tree contains zero files: {root}")
    return rows


def assert_candidate_sealable(root: Path) -> None:
    """Keep the authoring/measurement digest domains identical."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            raise StageGateError(
                f"performance candidate retains digest-excluded ephemeral state: {path}")


def candidate_delta(base: Path, candidate: Path) -> dict[str, Any]:
    """Describe changed package bytes and reject documentation-only authoring as vacuous."""
    before, after = _tree_files(base), _tree_files(candidate)
    changed = sorted(path for path in before.keys() | after.keys()
                     if before.get(path) != after.get(path))
    execution_relevant = [path for path in changed
                          if "docs" not in Path(path).parts and Path(path).suffix.lower() != ".md"]
    return {"changed_files": changed, "changed_file_count": len(changed),
            "execution_relevant_changed_files": execution_relevant,
            "execution_relevant_changed_file_count": len(execution_relevant)}


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(document))


def _safe_component(value: str, *, label: str) -> str:
    if not value or Path(value).name != value or value in (".", ".."):
        raise StageGateError(f"{label} must be a simple non-empty path component")
    return value


def _require_real_directory(path: Path, *, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or not raw.is_dir():
        raise StageGateError(f"{label} is absent or linked: {raw}")
    return raw.resolve()


def _require_executable(name_or_path: str, *, label: str) -> Path:
    found = shutil.which(name_or_path)
    path = Path(found or name_or_path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise StageGateError(f"required {label} executable is absent: {name_or_path}")
    return path.resolve()


def _selected_names(value: str | Sequence[str] | None, *, label: str) -> tuple[str, ...]:
    if value is None or value == "all":
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    names = tuple(_safe_component(str(item).strip(), label=label)
                  for item in raw if str(item).strip())
    if not names or len(names) != len(set(names)):
        raise StageGateError(f"{label} selection must contain unique names or 'all'")
    return names


def discover_performance_corpus(
        target_experiment: TargetExperiment, *, families: str | Sequence[str] | None = None,
        capsules: str | Sequence[str] | None = None) -> PerformanceCorpus:
    """Admit only generated dev capsules from the descriptor-derived phase."""
    target = str(target_experiment.target or "").strip()
    if not target:
        raise StageGateError("target experiment has no target identity")
    corpus_root = Path(target_experiment.capsule_corpus).resolve().parent
    provenance = corpus_root / "MANIFEST.yaml"
    manifest = _mapping_file(provenance, yaml_file=True)
    generations = manifest.get("performance_generation")
    generation = generations.get(target) if isinstance(generations, Mapping) else None
    if not isinstance(generation, Mapping) or generation.get("errors") != []:
        raise StageGateError(
            f"corpus provenance has no clean generated performance record for {target!r}")
    phase = generation.get("phase")
    if (not isinstance(phase, Mapping)
            or phase.get("included_in_functional_grade") is not False
            or phase.get("label") != "dev"):
        raise StageGateError("performance provenance does not prove a dev-only phase")
    category = _safe_component(str(phase.get("category") or ""),
                               label="performance phase category")
    if not category.startswith("_"):
        raise StageGateError("performance phase is not excluded from functional discovery")
    phase_root = (corpus_root / category).resolve()
    if corpus_root not in phase_root.parents or phase_root.is_symlink() or not phase_root.is_dir():
        raise StageGateError("descriptor-derived performance phase is absent or unsafe")
    if phase_root in {Path(path).resolve() for path in target_experiment.graded_roots()}:
        raise StageGateError("performance phase leaks into functional graded roots")
    generated = manifest.get("generated")
    hand_authored = manifest.get("hand_authored")
    if not isinstance(generated, list) or not isinstance(hand_authored, list):
        raise StageGateError("corpus provenance lacks generated/manual classification")
    generated_paths = {str(value) for value in generated}
    manual_paths = {str(value) for value in hand_authored}
    phase_generated = {path for path in generated_paths
                       if Path(path).parts and Path(path).parts[0] == category}
    if any(Path(path).parts and Path(path).parts[0] == category for path in manual_paths):
        raise StageGateError("performance phase contains manually classified capsules")

    found: list[PerformanceCapsule] = []
    for descriptor_path in sorted(phase_root.glob("*/capsule.yaml")):
        source = descriptor_path.parent
        relative = source.relative_to(corpus_root).as_posix()
        if relative not in phase_generated or relative in manual_paths:
            raise StageGateError(f"performance capsule lacks generator provenance: {relative}")
        descriptor = _mapping_file(descriptor_path, yaml_file=True)
        name = _safe_component(str(descriptor.get("name") or ""),
                               label="performance capsule")
        performance = descriptor.get("performance")
        if (source.name != name or descriptor.get("label") != "dev"
                or descriptor.get("source_role") != "derived_sweep"
                or not isinstance(performance, Mapping)):
            raise StageGateError(f"performance capsule {name!r} is not a generated dev member")
        family = _safe_component(str(performance.get("family") or ""),
                                 label="performance family")
        claim = performance.get("claim")
        if claim not in ("RECOVERS", "PREDICTS", "DIFFERENTIAL"):
            raise StageGateError(f"performance capsule {name!r} has no canonical claim")
        if claim == "PREDICTS" and not isinstance(performance.get("acceptance"), Mapping):
            raise StageGateError(
                f"predictive performance capsule {name!r} has no frozen acceptance contract")
        tree = _exact_tree_record(source)
        found.append(PerformanceCapsule(
            family, name, source.resolve(), relative, descriptor,
            str(tree["sha256"]), int(tree["n_files"]), int(tree["n_bytes"])))
    if not found or {row.source_relative_path for row in found} != phase_generated:
        raise StageGateError("generated performance phase is empty or stale versus provenance")
    wanted_families = set(_selected_names(families, label="performance family"))
    wanted_capsules = set(_selected_names(capsules, label="performance capsule"))
    known_families = {row.family for row in found}
    known_capsules = {row.capsule for row in found}
    if wanted_families - known_families or wanted_capsules - known_capsules:
        raise StageGateError("performance selection names an unknown generated member")
    selected = tuple(row for row in found
                     if (not wanted_families or row.family in wanted_families)
                     and (not wanted_capsules or row.capsule in wanted_capsules))
    if not selected:
        raise StageGateError("performance selection contains zero capsules")
    return PerformanceCorpus(
        target, corpus_root, phase_root, provenance, _sha256_file(provenance),
        dict(generation), selected)


def freeze_performance_corpus(
        corpus: PerformanceCorpus, snapshot_root: Path) -> FrozenPerformanceCorpus:
    snapshot_root = Path(snapshot_root).resolve()
    if snapshot_root.exists() or snapshot_root.is_symlink():
        raise StageGateError(f"performance snapshot is not fresh: {snapshot_root}")
    if _sha256_file(corpus.provenance_manifest) != corpus.provenance_sha256:
        raise StageGateError("performance provenance changed before freeze")
    capsules_root = snapshot_root / "capsules"
    capsules_root.mkdir(parents=True)
    frozen: list[PerformanceCapsule] = []
    rows: list[dict[str, Any]] = []
    for member in corpus.capsules:
        before = _exact_tree_record(member.source_dir)
        if before["sha256"] != member.source_sha256:
            raise StageGateError(f"performance capsule changed before freeze: {member.capsule}")
        destination = capsules_root / member.source_relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(member.source_dir, destination, symlinks=False)
        copied = _exact_tree_record(destination)
        if copied != before or _exact_tree_record(member.source_dir) != before:
            raise StageGateError(f"performance capsule changed during freeze: {member.capsule}")
        frozen.append(PerformanceCapsule(
            member.family, member.capsule, destination, member.source_relative_path,
            copy.deepcopy(member.descriptor), str(copied["sha256"]),
            int(copied["n_files"]), int(copied["n_bytes"])))
        rows.append({
            "family": member.family, "capsule": member.capsule,
            "source_relative_path": member.source_relative_path,
            "snapshot_relative_path": destination.relative_to(snapshot_root).as_posix(),
            "snapshot_sha256": copied["sha256"], "n_files": copied["n_files"],
            "n_bytes": copied["n_bytes"], "performance": member.descriptor["performance"],
            "performance_sha256": _document_sha256(member.descriptor["performance"]),
        })
    aggregate = _exact_tree_record(capsules_root)
    document = {
        "schema_version": 1, "target": corpus.target,
        "source": {"provenance_manifest": str(corpus.provenance_manifest),
                   "provenance_sha256": corpus.provenance_sha256,
                   "performance_generation_sha256": _document_sha256(
                       corpus.performance_generation)},
        "capsules_sha256": aggregate["sha256"], "capsules": rows,
    }
    manifest = snapshot_root / "performance_corpus_manifest.json"
    _write_json(manifest, document)
    for path in sorted(snapshot_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    snapshot_root.chmod(0o555)
    result = FrozenPerformanceCorpus(
        snapshot_root, capsules_root, manifest, _sha256_file(manifest),
        str(aggregate["sha256"]), tuple(frozen))
    verify_frozen_performance_corpus(result)
    return result


def load_frozen_performance_corpus(
        root: Path, *, manifest_sha256: str, capsules_sha256: str,
        expected_target: str | None = None) -> FrozenPerformanceCorpus:
    root = _require_real_directory(root, label="frozen performance corpus")
    manifest = root / "performance_corpus_manifest.json"
    if _sha256_file(manifest) != manifest_sha256:
        raise StageGateError("frozen performance manifest digest changed")
    document = _mapping_file(manifest)
    if (document.get("schema_version") != 1
            or document.get("capsules_sha256") != capsules_sha256
            or (expected_target is not None and document.get("target") != expected_target)):
        raise StageGateError("frozen performance manifest identity changed")
    capsules_root = root / "capsules"
    members: list[PerformanceCapsule] = []
    for row in document.get("capsules") or []:
        if not isinstance(row, Mapping):
            raise StageGateError("frozen performance manifest contains a malformed member")
        relative = Path(str(row.get("snapshot_relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise StageGateError("frozen performance member path is unsafe")
        source = (root / relative).resolve(strict=True)
        source.relative_to(capsules_root.resolve(strict=True))
        descriptor = _mapping_file(source / "capsule.yaml", yaml_file=True)
        members.append(PerformanceCapsule(
            str(row.get("family")), str(row.get("capsule")), source,
            str(row.get("source_relative_path")), descriptor,
            str(row.get("snapshot_sha256")), int(row.get("n_files") or 0),
            int(row.get("n_bytes") or 0)))
    result = FrozenPerformanceCorpus(
        root, capsules_root, manifest, manifest_sha256, capsules_sha256, tuple(members))
    verify_frozen_performance_corpus(result)
    return result


def verify_frozen_performance_corpus(corpus: FrozenPerformanceCorpus) -> None:
    if _sha256_file(corpus.manifest_path) != corpus.manifest_sha256:
        raise StageGateError("frozen performance manifest bytes changed")
    if _exact_tree_record(corpus.capsules_root)["sha256"] != corpus.capsules_sha256:
        raise StageGateError("frozen performance capsule bytes changed")
    document = _mapping_file(corpus.manifest_path)
    rows = document.get("capsules")
    if not isinstance(rows, list) or len(rows) != len(corpus.capsules) or not rows:
        raise StageGateError("frozen performance manifest has an incomplete member set")
    indexed = {(member.family, member.capsule): member for member in corpus.capsules}
    for row in rows:
        identity = (str(row.get("family")), str(row.get("capsule")))
        member = indexed.get(identity)
        if member is None:
            raise StageGateError(f"frozen performance manifest has unknown member {identity}")
        observed = _exact_tree_record(member.source_dir)
        if any(observed[key] != row.get(key) for key in ("n_files", "n_bytes")) \
                or observed["sha256"] != row.get("snapshot_sha256"):
            raise StageGateError(f"frozen performance member changed: {identity}")


def expected_perf_cells(
        capsules: Sequence[PerformanceCapsule], replicates: int,
        timing_simulator: str = "gsim") -> tuple[PP.PerfCell, ...]:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise StageGateError("performance replicate count must be positive")
    if timing_simulator not in ("gsim", "verilator"):
        raise StageGateError("performance timing simulator must be gsim or verilator")
    cells = tuple(PP.PerfCell(member.family, member.capsule, simulator, f"r{index:03d}")
                  for member in capsules for index in range(replicates)
                  for simulator in ("spike", timing_simulator))
    if not cells or len(cells) != len(set(cells)):
        raise StageGateError("performance cell schedule is empty or duplicated")
    return cells


def select_full_model_sentinel(
        functional: StageFunctionalRun,
        target_experiment: TargetExperiment) -> FullModelSentinel:
    snapshot_repo = Path(functional.bundle_input_snapshot["path"]) / "repo"
    try:
        relative = Path(target_experiment.capsule_corpus).resolve().relative_to(repo_root())
    except ValueError as exc:
        raise StageGateError("target corpus cannot be mapped into the functional snapshot") from exc
    parent = snapshot_repo / relative.parent
    candidates: list[FullModelSentinel] = []
    for descriptor_path in sorted(parent.glob("*/*/capsule.yaml")):
        source = descriptor_path.parent
        descriptor = _mapping_file(descriptor_path, yaml_file=True)
        lanes = descriptor.get("lanes")
        required = lanes.get("require") if isinstance(lanes, Mapping) else None
        tiers = descriptor.get("required_oracle_tiers")
        if (descriptor.get("kind") != "model" or descriptor.get("label") != "public"
                or not isinstance(required, list)
                or not {"on_mesh", "scalar_rvv_lane"}.issubset(required)
                or not isinstance(tiers, list) or not {"L2", "L3"}.issubset(tiers)):
            continue
        name = _safe_component(str(descriptor.get("name") or ""), label="E2E sentinel")
        if source.name != name:
            raise StageGateError("E2E sentinel directory/name mismatch")
        tree = _exact_tree_record(source)
        candidates.append(FullModelSentinel(
            name, source.resolve(), descriptor, str(tree["sha256"]),
            int(tree["n_files"]), int(tree["n_bytes"])))
    if not candidates:
        raise StageGateError("functional snapshot has no public cross-lane L2/L3 model")
    return min(candidates, key=lambda item: (item.n_bytes, item.capsule))


def load_prompt(path: Path) -> PromptArtifact:
    """Read the exact prompt artifact; no implicit default or strategy text is injected."""
    raw = Path(path)
    if raw.is_symlink() or not raw.is_file():
        raise StageGateError(f"an explicit, real performance prompt file is required: {raw}")
    path = raw.resolve()
    payload = path.read_bytes()
    if not payload or len(payload) > 2_000_000:
        raise StageGateError("performance prompt must be non-empty and at most 2 MB")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StageGateError("performance prompt must be UTF-8") from exc
    return PromptArtifact(path, text, _sha256(payload), len(payload))


def render_stage_prompt(inputs: StagePromptInputs) -> str:
    """Render through the current shared prompt, then bind richer stage facts."""
    for name, value in (
            ("replicates", inputs.replicates),
            ("smoke_replicates", inputs.smoke_replicates),
            ("wall_budget_seconds", inputs.wall_budget_seconds),
            ("rounds", inputs.rounds),
            ("round_timeout_seconds", inputs.round_timeout_seconds),
            ("max_tool_calls", inputs.max_tool_calls),
            ("tool_timeout_seconds", inputs.tool_timeout_seconds)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageGateError(f"performance prompt {name} must be a positive integer")
    if inputs.smoke_replicates >= inputs.replicates:
        raise StageGateError("smoke replicate count cannot masquerade as the formal cohort")
    expected_replicas = tuple(f"r{index:03d}" for index in range(inputs.replicates))
    if inputs.formal_replicate_identities != expected_replicas:
        raise StageGateError("formal replicate identities are not the exact canonical cohort")
    declaration = inputs.formal_claim.get("declaration")
    evidence = declaration.get("evidence") if isinstance(declaration, Mapping) else None
    timing_simulator = (evidence.get("timing_simulator")
                        if isinstance(evidence, Mapping) else None)
    try:
        supported_acceptance = PK.supported_acceptance(str(timing_simulator))
    except ValueError as exc:
        raise StageGateError(
            "performance prompt formal claim selects an unsupported timing engine") from exc
    if (_canonical_json(declaration) != _canonical_json(supported_acceptance)
            or inputs.formal_claim.get("status") != "READY"):
        raise StageGateError("performance prompt formal claim is not preflight-ready")
    if (set(inputs.e2e_sentinel.required_lanes) != {"on_mesh", "scalar_rvv_lane"}
            or "L3" not in inputs.e2e_sentinel.required_tiers):
        raise StageGateError("performance prompt E2E sentinel lacks its cross-lane L3 contract")

    base_families = tuple(PP.PerfFamily(
        family.family, family.claim, family.negative_control,
        family.falsifier_observation, family.differential_basis,
        family.fitted_parameters) for family in inputs.families)
    base_host = PP.HostLaneGrant(
        inputs.host_lane.target, inputs.host_lane.package_id,
        inputs.host_lane.package_path, inputs.host_lane.package_sha256,
        inputs.host_lane.manifest_path, inputs.host_lane.integration_seam)
    base = PP.PerfPromptInputs(
        target=inputs.target, approach=inputs.approach,
        functional_run_id=inputs.functional_run_id,
        functional_submission_sha256=inputs.functional_submission_sha256,
        frozen_functional_path=inputs.frozen_functional_path,
        frozen_functional_sha256=inputs.frozen_functional_sha256,
        submission_path=inputs.submission_path,
        submission_initial_sha256=inputs.submission_initial_sha256,
        functional_public_capsules=inputs.functional_public_capsules,
        functional_hidden_capsules=inputs.functional_hidden_capsules,
        workload_root=inputs.workload_root,
        workload_manifest=inputs.workload_manifest,
        workload_manifest_sha256=inputs.workload_manifest_sha256,
        workload_capsules_sha256=inputs.workload_capsules_sha256,
        expected_cells=inputs.expected_cells, families=base_families,
        host_lane=base_host, tools=inputs.tools,
        allowed_paths=inputs.allowed_paths,
        execution_broker_path=inputs.execution_broker_path,
        execution_broker_command=inputs.execution_broker_command,
        broker_receipt_path=inputs.broker_receipt_path)
    rendered = PP.render_initial_prompt(base).rstrip()
    family_acceptance = {
        family.family: copy.deepcopy(family.acceptance)
        for family in inputs.families if family.acceptance is not None
    }
    supplement = {
        "schema_version": 1,
        "formal_replicates": list(inputs.formal_replicate_identities),
        "formal_claim": copy.deepcopy(dict(inputs.formal_claim)),
        "smoke_replicates": inputs.smoke_replicates,
        "functional_bundle_snapshot": {
            "manifest": inputs.functional_bundle_snapshot_manifest,
            "manifest_sha256": inputs.functional_bundle_snapshot_manifest_sha256,
            "content_sha256": inputs.functional_bundle_snapshot_sha256,
        },
        "e2e_sentinel": {
            "capsule": inputs.e2e_sentinel.capsule,
            "capsule_path": inputs.e2e_sentinel.capsule_path,
            "frozen_source_path": inputs.e2e_sentinel.frozen_source_path,
            "capsule_sha256": inputs.e2e_sentinel.capsule_sha256,
            "required_lanes": list(inputs.e2e_sentinel.required_lanes),
            "required_tiers": list(inputs.e2e_sentinel.required_tiers),
        },
        "budgets": {
            "wall_budget_seconds": inputs.wall_budget_seconds,
            "rounds": inputs.rounds,
            "round_timeout_seconds": inputs.round_timeout_seconds,
            "max_tool_calls": inputs.max_tool_calls,
            "tool_timeout_seconds": inputs.tool_timeout_seconds,
        },
        "family_acceptance": family_acceptance,
    }
    return rendered + "\n\n## Sealed authoring-stage supplement\n\n" + (
        "The outer Codex control plane has only its isolated authentication mount. "
        "The inner execution plane has the live descriptor-derived toolchain, `--clearenv`, "
        "and no credentials. Network availability is not an isolation claim. The JSON below "
        "is immutable launch data; do not retune its acceptance rules after observing results.\n\n"
        "```json\n" + json.dumps(supplement, sort_keys=True, indent=2) + "\n```\n")


def materialize_canonical_prompt(inputs: StagePromptInputs,
                                 artifact_path: Path) -> PromptArtifact:
    """Render the sole accepted prompt, after every frozen launch fact is known."""
    text = render_stage_prompt(inputs)
    if not isinstance(text, str) or not text.strip():
        raise StageGateError("performance prompt renderer returned no instruction")
    payload = text.encode("utf-8")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as stream:
        stream.write(payload)
    return load_prompt(artifact_path)


def load_frozen_functional_inputs(functional: StageFunctionalRun) -> FrozenFunctionalInputs:
    """Reify the exact v2 grant table already verified by ``inspect_functional_run``."""
    root = Path(functional.bundle_input_snapshot["path"])
    marker = root / "snapshot.json"
    if root.is_symlink() or not root.is_dir() or marker.is_symlink() or not marker.is_file():
        raise StageGateError("functional v2 input snapshot or marker is absent")
    document = json.loads(marker.read_text(encoding="utf-8"))
    if (document.get("version") != 2
            or document.get("content_sha256") != functional.bundle_input_snapshot.get(
                "content_sha256")):
        raise StageGateError("functional v2 input snapshot marker identity changed")
    records = document.get("grants")
    if not isinstance(records, list) or not records:
        raise StageGateError("functional v2 input snapshot has no exact grant table")
    grants: list[FrozenGrant] = []
    resolved_root = root.resolve(strict=True)
    repo = repo_root().absolute()
    for row in records:
        if not isinstance(row, Mapping):
            raise StageGateError("functional v2 input snapshot contains a malformed grant")
        declared = str(row.get("path") or "")
        destination = Path(str(row.get("destination") or ""))
        relative = Path(str(row.get("snapshot") or ""))
        if (not declared or not destination.is_absolute() or relative.is_absolute()
                or ".." in relative.parts):
            raise StageGateError("functional v2 input snapshot contains an unsafe grant")
        allowed_destinations = {(repo / declared).absolute(),
                                (repo / "merlin" / declared).absolute()}
        if destination not in allowed_destinations:
            raise StageGateError(f"functional frozen grant has foreign destination: {declared}")
        source = root / relative
        try:
            resolved_source = source.resolve(strict=True)
        except OSError as exc:
            raise StageGateError(f"functional frozen grant is absent: {declared}") from exc
        if resolved_source != resolved_root and resolved_root not in resolved_source.parents:
            raise StageGateError(f"functional frozen grant escapes snapshot: {declared}")
        if source.is_symlink():
            raise StageGateError(f"functional frozen grant is linked: {declared}")
        digest = (_exact_tree_record(source)["sha256"] if source.is_dir()
                  else _sha256_file(source))
        grants.append(FrozenGrant(declared, destination, source, str(digest)))
    host_source = Path(functional.model_host_package)
    if host_source.is_symlink() or not host_source.is_dir() \
            or resolved_root not in host_source.resolve(strict=True).parents:
        raise StageGateError("functional frozen host lane is absent from the v2 input snapshot")
    grants.append(FrozenGrant(
        "__model_host_lane_snapshot__", host_source, host_source,
        str(_exact_tree_record(host_source)["sha256"])))
    return FrozenFunctionalInputs(root, marker, _sha256_file(marker),
                                  str(document["content_sha256"]), tuple(grants))


def frozen_grant_mounts(inputs: FrozenFunctionalInputs) -> list[str]:
    argv: list[str] = []
    for grant in inputs.grants:
        argv += ["--ro-bind", str(grant.source), str(grant.destination)]
    return argv


def _frozen_path_for_destination(inputs: FrozenFunctionalInputs, destination: Path) -> Path:
    destination = destination.absolute()
    candidates: list[tuple[int, Path]] = []
    for grant in inputs.grants:
        if destination == grant.destination or grant.destination in destination.parents:
            candidates.append((len(grant.destination.parts),
                               grant.source / destination.relative_to(grant.destination)))
    if not candidates:
        raise StageGateError(f"functional snapshot did not grant required path: {destination}")
    frozen = max(candidates, key=lambda row: row[0])[1]
    if frozen.is_symlink() or not frozen.exists():
        raise StageGateError(f"functional frozen required path is absent: {destination}")
    return frozen


def prepare_formal_pk_claim(
        capsules: Sequence[PerformanceCapsule],
        requested_replicates: int | None = None) -> dict[str, Any]:
    """Admit the exact frozen PK declaration and derive its formal result cohort."""
    descriptors = [capsule.descriptor for capsule in capsules if capsule.family == "PK"]
    preflight = PK.preflight_pk_claim(descriptors)
    if preflight.get("status") != "READY":
        reasons = preflight.get("refusal_reasons")
        detail = "; ".join(str(value) for value in reasons) if isinstance(reasons, list) else "unknown"
        raise StageGateError(f"frozen PK formal claim preflight refused: {detail}")
    declaration = preflight.get("declaration")
    if not isinstance(declaration, Mapping):
        raise StageGateError("frozen PK acceptance is not a mapping")
    evidence = declaration.get("evidence")
    timing_simulator = (evidence.get("timing_simulator")
                        if isinstance(evidence, Mapping) else None)
    try:
        supported = PK.supported_acceptance(str(timing_simulator))
    except ValueError as exc:
        raise StageGateError("frozen PK acceptance selects an unsupported timing engine") from exc
    if _canonical_json(declaration) != _canonical_json(supported):
        raise StageGateError("frozen PK acceptance differs from the supported claim contract")
    replicate_contract = declaration.get("replicates")
    if not isinstance(replicate_contract, Mapping):
        raise StageGateError("frozen PK acceptance omits its replicate contract")
    identities = replicate_contract.get("identities")
    exact_count = replicate_contract.get("exact_count")
    if (not isinstance(identities, list) or any(not isinstance(value, str) for value in identities)
            or isinstance(exact_count, bool) or not isinstance(exact_count, int)
            or exact_count != len(identities)
            or identities != [f"r{index:03d}" for index in range(exact_count)]):
        raise StageGateError("frozen PK acceptance has an invalid exact replicate cohort")
    if requested_replicates is not None and (
            isinstance(requested_replicates, bool) or not isinstance(requested_replicates, int)
            or requested_replicates != exact_count):
        raise StageGateError(
            f"formal replicate override must equal frozen PK exact_count={exact_count}")
    expected = preflight.get("expected_identities")
    if not isinstance(expected, list) or len(expected) != len(descriptors) * exact_count * 2:
        raise StageGateError("frozen PK preflight did not produce its exact L2/L3 identity cohort")
    return copy.deepcopy(preflight)


def _family_declarations(
        capsules: Sequence[PerformanceCapsule],
        formal_claim: Mapping[str, Any]) -> tuple[PerformanceFamilyDeclaration, ...]:
    rows: dict[str, PerformanceFamilyDeclaration] = {}
    for capsule in capsules:
        performance = capsule.descriptor["performance"]
        comparand, falsifier = performance["comparand"], performance["falsifier"]
        knobs = performance["emitter"]["knobs"]
        fitted: tuple[str, ...] = ()
        if performance["claim"] == "PREDICTS":
            axes = {str(value) for key, value in knobs.items()
                    if isinstance(value, str) and ("axis" in str(key) or "parameter" in str(key))}
            fitted = tuple(sorted(axes or {str(key) for key in knobs}))
        differential = json.dumps({
            "kind": comparand["kind"], "against": comparand["against"],
            "cancels": comparand["cancels"], "demand_equal": comparand["demand_equal"],
        }, sort_keys=True, separators=(",", ":"))
        family = PerformanceFamilyDeclaration(
            capsule.family, performance["claim"], str(falsifier["negative_control"]),
            str(falsifier["observation"]), differential, fitted,
            copy.deepcopy(performance.get("acceptance")))
        previous = rows.get(capsule.family)
        if previous is not None and previous != family:
            raise StageGateError(f"performance family declaration drifts: {capsule.family}")
        rows[capsule.family] = family
    pk = rows.get("PK")
    if (pk is None or _canonical_json(pk.acceptance) != _canonical_json(
            formal_claim.get("declaration"))):
        raise StageGateError("PK family declaration drifts from its formal preflight acceptance")
    return tuple(rows[name] for name in sorted(rows))


def select_e2e_sentinel(functional: StageFunctionalRun, frozen: FrozenFunctionalInputs,
                        target_experiment: TargetExperiment) -> StageE2ESentinel:
    """Select the smallest already-passed public whole-model L3 cross-lane capsule."""
    selected = select_full_model_sentinel(functional, target_experiment)
    snapshot_repo = (frozen.root / "repo").resolve(strict=True)
    try:
        relative = selected.source_dir.resolve(strict=True).relative_to(snapshot_repo)
    except ValueError as exc:
        raise StageGateError("frozen full-model sentinel is outside the functional snapshot") from exc
    destination = (repo_root() / relative).absolute()
    # Prove the prompt destination is one of the exact frozen grant views.
    if _frozen_path_for_destination(frozen, destination).resolve() != selected.source_dir.resolve():
        raise StageGateError("full-model sentinel does not map to its frozen grant destination")
    return StageE2ESentinel(
        selected.capsule, str(destination), str(selected.source_dir), selected.source_sha256,
        tuple(selected.descriptor["lanes"]["require"]),
        tuple(selected.descriptor["required_oracle_tiers"]))


_PLACEHOLDER = re.compile(r"\{([A-Za-z][A-Za-z0-9_]*)\}")



#: What a barrier count reports when the stream cannot be read. Never zero: "no barriers found" and
#: "cannot see barriers" are different claims, and only one of them is evidence.
BARRIER_UNKNOWN = "UNKNOWN"


def _demand_lower_bound(buffer: Mapping[str, Any], peak_macs_per_cycle: int | None) -> dict[str, Any]:
    """Cycles this arm cannot beat, from its own declared work and operands.

    A bound, not a prediction. Compute demand is the priced MAC count over the structural peak;
    movement demand is the operand bytes the buffer itself declares. Both are floors -- a spilling
    schedule re-fetches, so real movement is only ever larger -- which keeps the result honestly a
    lower bound rather than an estimate that could flatter a candidate.
    """
    if not peak_macs_per_cycle:
        return {"status": "unavailable", "reason": "no derived structural peak for this target"}
    from merlin.perf.work_volume import work_from_command_buffer            # noqa: PLC0415
    work = work_from_command_buffer(buffer)
    macs = int(getattr(work, "known_macs", 0) or 0)
    tensors = buffer.get("tensors")
    if not macs or not isinstance(tensors, Mapping):
        return {"status": "unavailable", "reason": "the buffer declares no work or no tensors"}
    width = {"i8": 1, "u8": 1, "i16": 2, "bf16": 2, "f16": 2, "i32": 4, "f32": 4}
    operand_bytes = 0
    for spec in tensors.values():
        if not isinstance(spec, Mapping):
            continue
        shape, dtype = spec.get("shape"), str(spec.get("dtype") or "")
        if not isinstance(shape, Sequence) or dtype not in width:
            return {"status": "unavailable",
                    "reason": f"an operand declares no shape or an unpriced dtype {dtype!r}"}
        count = 1
        for extent in shape:
            count *= int(extent)
        operand_bytes += count * width[dtype]
    return {"status": "derived",
            "compute_floor_cycles": macs / float(peak_macs_per_cycle),
            "declared_operand_bytes": operand_bytes,
            "exact": not bool(getattr(work, "is_lower_bound", False)),
            "licence": "a floor the arm cannot beat; never an estimate of what it will cost"}


def _command_events(buffer: Mapping[str, Any]) -> dict[str, float] | None:
    """Count the emitted commands by the kind a calibrated cost model prices.

    The cost model is fitted per COMMAND KIND, so what it needs is a histogram of the buffer's own
    opcodes mapped onto the event names it was calibrated against. Returns None when the buffer
    declares an opcode outside that vocabulary -- an unrecognised command means the histogram is
    incomplete, and an incomplete histogram priced as if it were whole would understate the arm.
    """
    rows = buffer.get("commands")
    if not isinstance(rows, Sequence) or not rows:
        return None
    # The ABI opcode -> calibrated event name. Both vocabularies are declared, not guessed: the
    # left side is what the emitted buffer contains, the right is what the model was fitted on.
    mapping = {"RES_PACK": "mvin2_B", "MATMUL_RESIDENT": "compute", "MATMUL": "compute",
               "COMMIT": "mvout", "EVICT": "mvout", "FENCE": "fence", "BIAS_ADD": "compute",
               "VECTOR_MAP": "compute", "VREDUCE": "compute", "CONV2D": "compute",
               "MOVEMENT": "mvin_A", "ATTENTION_QK": "compute", "ATTENTION_PV": "compute"}
    events: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        event = mapping.get(str(row.get("opcode") or ""))
        if event is None:
            return None
        events[event] = events.get(event, 0.0) + 1.0
    return events


def _calibrated_estimate(target: str, buffer: Mapping[str, Any]) -> dict[str, Any]:
    """A calibrated cycle estimate for one arm, with its measured error band -- or a refusal.

    SCREENING ONLY. This model is order-blind: it prices a histogram of commands, so it separates
    candidates that change WHAT is issued (tiling, blocking, how many barriers) and is blind to
    candidates that only change the ORDER. It may eliminate a candidate; it may never certify one,
    and its number never enters a result. The band is reported with the value because a point
    estimate quoted without its error is the over-claim this whole layer exists to prevent.
    """
    events = _command_events(buffer)
    if events is None:
        return {"status": "unavailable",
                "reason": "the buffer declares a command outside the calibrated vocabulary, so its "
                          "event histogram is incomplete and pricing it would understate this arm"}
    try:
        from merlin.cost_model.linear import LinearCostModel                # noqa: PLC0415
        model = LinearCostModel.load(_cost_model_artifact(target))
        cycles, band = model.predict_with_band(events)
    except Exception as exc:  # noqa: BLE001 - an uncalibrated target screens nothing, and says so
        return {"status": "unavailable",
                "reason": f"no calibrated cost model for {target!r}: {type(exc).__name__}"}
    return {"status": "derived", "cycles": float(cycles), "band": float(band), "events": events,
            "basis": "per-command coefficients calibrated against the cycle-accurate engine",
            "licence": "screening only; order-blind; never a certified cycle count"}


def _cost_model_artifact(target: str) -> Path | None:
    """The target's calibrated coefficients, resolved by NAME rather than hardcoded per target."""
    from merlin.common.paths import merlin_dir                              # noqa: PLC0415
    candidate = merlin_dir() / "python" / "merlin" / "cost_model" / f"{target}_cost_coeffs.json"
    return candidate if candidate.is_file() else None


def analyze_command_buffers(baseline_json: Path, candidate_json: Path, *,
                            peak_macs_per_cycle: int | None,
                            achievable_macs_per_cycle: float | None,
                            target: str = "") -> dict[str, Any]:
    """Price two emitted command buffers against each other -- ORDERING ONLY, and free.

    WHY THIS EXISTS. The measured feedback is the only judge the agent had, and it costs ~110 s a
    call, so a candidate that turns out 11% WORSE cost the same as one that wins. Across three
    trials two such excursions burned ~220 s of oracle time to learn something the emitted buffers
    already implied.

    WHAT IT MAY AND MAY NOT SAY. It reads the candidate's OWN artifacts -- no oracle, no goldens, no
    holdout -- so it cannot leak and needs no budget. In exchange it never claims a cycle count:
    ``merlin.perf.differential`` decides whether two demands are even comparable and returns EXACT,
    ORDERING_ONLY or REFUSED with its reason. An absolute magnitude for an unmeasured shape is
    exactly the over-claim the corpus-calibrated model is not licensed to make.
    """
    from merlin.perf.work_volume import work_from_command_buffer          # noqa: PLC0415

    def _load(path: Path) -> Mapping[str, Any]:
        resolved = Path(path)
        if resolved.is_symlink() or not resolved.is_file():
            raise StageGateError(f"command buffer is absent or linked: {resolved}")
        return json.loads(resolved.read_text(encoding="utf-8"))

    out: dict[str, Any] = {"schema_version": 1, "kind": "host_owned_command_buffer_analysis",
                           "basis": "emitted artifacts only; no oracle, no golden, no holdout"}
    arms: dict[str, Any] = {}
    for arm, path in (("baseline", baseline_json), ("candidate", candidate_json)):
        work = work_from_command_buffer(_load(path))
        macs = int(getattr(work, "known_macs", 0) or 0)
        # A LOWER BOUND IS NOT A TOTAL. `work_volume` prices each command it can and records a
        # refusal for each it cannot, so `is_lower_bound` means "there is unpriced work here".
        # Reporting that as a total would understate the candidate's demand and silently flatter it.
        row: dict[str, Any] = {
            "macs": macs,
            "exact": not bool(getattr(work, "is_lower_bound", False)),
            "unpriced_commands": [str(r) for r in (getattr(work, "refusals", ()) or ())][:8],
        }
        if peak_macs_per_cycle:
            row["ideal_cycles_at_peak"] = macs / float(peak_macs_per_cycle)
        if achievable_macs_per_cycle:
            row["ideal_cycles_at_achievable"] = macs / float(achievable_macs_per_cycle)
        arms[arm] = row
    out["arms"] = arms
    out["peak_macs_per_cycle"] = peak_macs_per_cycle
    out["achievable_macs_per_cycle"] = achievable_macs_per_cycle

    # THREE FREE SIGNALS, none of which may certify. Each reads the candidate's own emitted
    # artifacts, so none can leak a golden or a holdout, and none costs oracle time. Together they
    # let a bad candidate be ELIMINATED before a measurement is spent on it -- the measured rule is
    # that a cheap tier which REFUTES is sound (12/12) while one that PASSES certifies nothing.
    buffers = {arm: _load(path)
               for arm, path in (("baseline", baseline_json), ("candidate", candidate_json))}

    # NO CALIBRATED CYCLE ESTIMATE IS OFFERED, and the reason is measured, not cautious.
    #
    # The per-command cost model is accurate on absolute magnitude for in-distribution shapes
    # (MAPE 8.1%), and it is ANTI-predictive for the comparison this action exists to make.
    # Measured over 774 within-capsule ordered pairs drawn from 115 distinct emitted programs, its
    # ordering agreement with the cycle oracle is 39.3% -- materially WORSE than the 50% a coin
    # gets, and worse than spike's 46.1%. The mechanism is structural: within one capsule the work
    # is fixed, so the `compute` term never varies between two candidates, and the only terms left
    # that do vary (config, mvin, fence counts) anti-correlate with measured cycles. Reporting it
    # here would not be a weak signal, it would be a signal pointing the wrong way, and the agent
    # would follow it. Absolute magnitude is a different question from ordering; do not let a good
    # answer to the first be quoted as an answer to the second.

    # 2. synchronization: how many completion points the candidate removed
    try:
        from merlin.perf import barrier_arms as BARRIER                     # noqa: PLC0415
        out["barriers"] = BARRIER.paired_removal(buffers["baseline"], buffers["candidate"])
    except Exception as exc:  # noqa: BLE001 - an uncountable stream is UNKNOWN, never zero
        out["barriers"] = {"status": BARRIER_UNKNOWN,
                           "reason": f"barrier counting failed: {type(exc).__name__}"}

    # 3. a LOWER BOUND on cycles from declared demand alone: what this arm cannot beat
    out["lower_bound"] = {arm: _demand_lower_bound(buffer, peak_macs_per_cycle)
                          for arm, buffer in buffers.items()}

    b, c = arms["baseline"]["macs"], arms["candidate"]["macs"]
    if b and c and b != c:
        out["work_delta"] = {"candidate_over_baseline": c / b,
                             "note": ("the candidate does a DIFFERENT amount of arithmetic; a cycle "
                                      "comparison between these two is not a schedule comparison")}
    # NO DIFFERENTIAL VERDICT IS ATTEMPTED HERE, and saying so is the point.
    #
    # This previously called `differential.compare(arms["baseline"], arms["candidate"])` on the two
    # plain dicts built just above. `compare` takes two `envelope.Composed` bounds and reads
    # `.operator` off them, so on a dict it raised AttributeError on EVERY call, the bare `except`
    # swallowed it, and the action reported a hardcoded `{"basis": "REFUSED"}` -- a refusal that
    # looked like the analyzer's considered verdict but was only a type error. A stale claim that
    # reads like evidence is worse than no claim, because it gets cited.
    #
    # The honest reason is structural, not incidental: this action compares DEMAND (the work each
    # command buffer declares) and never builds a composed envelope or per-resource demands, so it
    # has nothing a cycle-level differential could be computed from. The measurement path is what
    # carries a differential verdict.
    # WHICH OF TWO ORDERINGS IS FASTER IS NOT ANSWERED HERE, and the refusal is measured. Every
    # cheap signal available to this action was scored on the exact comparison the search makes --
    # two programs for the SAME workload, which the oracle timed faster -- over held-out workloads,
    # by `validate_ordering_signals.py`. None qualified. The numbers travel with the refusal so a
    # reader can see it is a measurement rather than caution, and so a later change is checked
    # against them rather than against a memory of them.
    out["ordering_signals"] = {
        "status": ORDERING_REFUSED,
        "basis": ("held-out within-workload ordering agreement against the cycle oracle, "
                  f"{ORDERING_EVIDENCE['held_out_pairs']} pair(s) over "
                  f"{ORDERING_EVIDENCE['held_out_workloads']} workload(s)"),
        "measured": dict(ORDERING_EVIDENCE["agreement"]),
        "reason": ("no signal readable from a command buffer orders two schedules of the same "
                   "workload better than chance, so none is offered for that purpose. Use this "
                   "action to eliminate a candidate that does MORE declared work, adds completion "
                   "points, or cannot beat its own lower bound -- all of which are decidable here. "
                   "Which of two legal orderings is faster is decidable only by measurement."),
        "artifact": ORDERING_EVIDENCE["artifact"],
    }
    out["differential"] = {
        "basis": "not_attempted",
        "reason": ("this action prices declared WORK from the command buffers; a cycle-level "
                   "differential needs a composed envelope and per-resource demands per arm, "
                   "which it never builds. Read the measurement path for a differential verdict."),
    }
    return out


def build_action_registry(candidate: Path,
                          target_experiment: TargetExperiment) -> tuple[BrokerAction, ...]:
    """Create named candidate-manifest actions; no caller-selected executable is accepted."""
    manifest_path = candidate / "manifest.yaml"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise StageGateError("functional candidate has no real manifest.yaml")
    document = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    entries, commands = document.get("entrypoints"), document.get("commands")
    if not isinstance(entries, Mapping) or not isinstance(commands, Mapping) or not commands:
        raise StageGateError("functional candidate manifest has no brokerable commands")
    tool = entries.get("tool")
    if not isinstance(tool, str) or not tool:
        raise StageGateError("functional candidate manifest has no tool entrypoint")
    tool_path = candidate / tool
    if tool_path.is_symlink() or not tool_path.is_file():
        raise StageGateError("functional candidate tool entrypoint is absent or linked")
    actions: list[BrokerAction] = []
    for command_name in sorted(commands):
        row = commands[command_name]
        raw = row.get("argv") if isinstance(row, Mapping) else None
        if not isinstance(raw, list) or not raw or any(not isinstance(v, str) for v in raw):
            raise StageGateError(f"candidate manifest command {command_name!r} has malformed argv")
        argv = tuple(str(tool_path) if value == "{tool}" else value for value in raw)
        placeholders = tuple(sorted({match for value in argv for match in _PLACEHOLDER.findall(value)}))
        if "tool" in placeholders:
            raise StageGateError(f"candidate manifest command {command_name!r} has embedded tool token")
        actions.append(BrokerAction(
            f"candidate-{command_name.replace('_', '-')}", argv, placeholders,
            f"candidate manifest command {command_name}", True))
    for probe in TC.required_tool_probes(target_experiment):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", probe.label).strip("-").lower()
        if not slug:
            raise StageGateError("required target tool probe has no safe action name")
        actions.append(BrokerAction(f"probe-{slug}", ("bash", "-c", probe.cmd), (),
                                    f"descriptor-derived probe for {probe.label}", False))
    actions.append(BrokerAction(
        DEVELOPMENT_FEEDBACK_ACTION, (_HOST_FEEDBACK_SENTINEL,), (),
        "host-owned frozen-tuning correctness and certified GSIM cycle deltas; invoke after edits",
        True))
    actions.append(BrokerAction(
        ANALYSIS_ACTION, (_HOST_ANALYSIS_SENTINEL, "{baseline_json}", "{candidate_json}"),
        ("baseline_json", "candidate_json"),
        "host-owned comparison of two emitted command buffers from the buffers alone: declared work "
        "volume per arm, the derived ceilings, the change in completion points (barriers), and a "
        "cycle LOWER BOUND per arm. Costs no oracle time -- use it to ELIMINATE a candidate before "
        "spending a measurement on it. It cannot certify one: nothing here predicts which of two "
        "orderings is faster, and the block it returns says so",
        False))
    names = [action.name for action in actions]
    if len(names) != len(set(names)):
        raise StageGateError("broker action registry contains duplicate names")
    return tuple(actions)


def _capsule_verdict_fields(**kwargs: Any) -> dict[str, Any]:
    """Decide one member, and never let a failure to decide read as a decision."""
    try:
        import perf_capsule_verdict as CV                                   # noqa: PLC0415
        row = CV.capsule_verdict(**kwargs)
        return {"verdict": row.get("verdict"), "verdict_reason": row.get("reason")}
    except Exception as exc:  # noqa: BLE001 - an undecidable member is refused, never assumed
        return {"verdict": "refused",
                "verdict_reason": f"the verdict could not be computed: {type(exc).__name__}"}


def validate_redacted_feedback(document: Mapping[str, Any]) -> dict[str, Any]:
    """Exact non-answer schema returned to the authoring agent."""
    required = {"schema_version", "kind", "round", "invocation", "tuning_corpus_sha256",
                "candidate_sha256", "certificate_sha256", "engine", "cells", "summary", "stopping"}
    if not isinstance(document, Mapping) or set(document) != required:
        raise StageGateError("development GSIM feedback violates its redacted top-level schema")
    if (document.get("schema_version") != 1
            or document.get("kind") != "host_owned_tuning_gsim_feedback"
            or document.get("engine") != "gsim"):
        raise StageGateError("development feedback is not the host-owned GSIM schema")
    for field in ("round", "invocation"):
        value = document.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise StageGateError(f"development feedback {field} is invalid")
    for field in ("tuning_corpus_sha256", "candidate_sha256", "certificate_sha256"):
        if not _is_sha256(document.get(field)):
            raise StageGateError(f"development feedback {field} is not a SHA-256")
    cells = document.get("cells")
    if not isinstance(cells, list) or not cells:
        raise StageGateError("development feedback contains zero tuning cells")
    cell_fields = {"family", "capsule", "baseline_correct", "candidate_correct",
                   "baseline_gsim_cycles", "candidate_gsim_cycles",
                   "candidate_minus_baseline_cycles", "baseline_over_candidate", "comparable",
                   "declared_macs", "declared_work_basis", "ideal_cycles_at_peak",
                   "baseline_utilization", "candidate_utilization",
                   "baseline_share_of_achievable", "candidate_share_of_achievable",
                   # Whether this member is FINISHED, better, or still owes cycles. Everything
                   # above is a number the reader has to interpret; without this the cell records
                   # a measurement and states no position on it, which is how a member at 3% of
                   # the achievable rate and one at 100% came to read identically.
                   "verdict", "verdict_reason",
                   # A cell the sweep did not pay for says so, rather than being omitted. Omitting
                   # it would let a short sweep read as a complete one.
                   "measured", "skip_reason"}
    identities: set[tuple[str, str]] = set()
    for index, row in enumerate(cells):
        if not isinstance(row, Mapping) or set(row) != cell_fields:
            raise StageGateError(f"development feedback cell {index} violates the redacted schema")
        family, capsule = row.get("family"), row.get("capsule")
        if (not isinstance(family, str) or not family or not isinstance(capsule, str)
                or not capsule or (family, capsule) in identities):
            raise StageGateError(f"development feedback cell {index} has an invalid identity")
        # Utilization is derived or it is null. A ratio outside (0, 1] would mean the program beat a
        # ceiling its own RTL says is unreachable, which is a broken derivation, not a fast program.
        macs = row.get("declared_macs")
        if macs is not None and (isinstance(macs, bool) or not isinstance(macs, int) or macs <= 0):
            raise StageGateError(f"development feedback cell {index} has invalid declared work")
        if not isinstance(row.get("declared_work_basis"), str):
            raise StageGateError(f"development feedback cell {index} omits its work basis")
        for field in ("ideal_cycles_at_peak", "baseline_utilization", "candidate_utilization",
                      "baseline_share_of_achievable", "candidate_share_of_achievable"):
            value = row.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise StageGateError(
                    f"development feedback cell {index} has an invalid {field}")
            if (field.endswith("_utilization") or field.endswith("_share_of_achievable")) and value > 1:
                raise StageGateError(
                    f"development feedback cell {index} reports {field} above the derived peak")
        identities.add((family, capsule))
        if any(not isinstance(row.get(field), bool)
               for field in ("baseline_correct", "candidate_correct", "comparable")):
            raise StageGateError(f"development feedback cell {index} has invalid correctness")
        if row["comparable"] != (row["baseline_correct"] and row["candidate_correct"]):
            raise StageGateError(
                f"development feedback cell {index} has inconsistent comparability")
        for field in ("baseline_gsim_cycles", "candidate_gsim_cycles"):
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise StageGateError(f"development feedback cell {index} has invalid GSIM cycles")
        delta, ratio = row.get("candidate_minus_baseline_cycles"), row.get("baseline_over_candidate")
        if row["comparable"]:
            expected_delta = row["candidate_gsim_cycles"] - row["baseline_gsim_cycles"]
            expected_ratio = row["baseline_gsim_cycles"] / row["candidate_gsim_cycles"]
            if (isinstance(delta, bool) or not isinstance(delta, int) or delta != expected_delta
                    or isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                    or not math.isfinite(float(ratio)) or float(ratio) != expected_ratio):
                raise StageGateError(f"development feedback cell {index} has inconsistent deltas")
        elif delta is not None or ratio is not None:
            raise StageGateError(f"development feedback cell {index} exposes a failed comparison")
    summary = document.get("summary")
    if (not isinstance(summary, Mapping)
            or set(summary) != {"members", "comparable", "all_correct",
                                "peak_macs_per_cycle", "peak_basis",
                                "achievable_macs_per_cycle", "achievable_basis"}
            or summary.get("members") != len(cells)
            or summary.get("comparable") != sum(bool(row["comparable"]) for row in cells)
            or summary.get("all_correct") != (summary.get("comparable") == len(cells))):
        raise StageGateError("development feedback summary is inconsistent")
    # Exact schemas above already exclude goldens, outputs, paths, shapes, and
    # Verilator.  This serialized audit makes that boundary easy to regression-test.
    encoded = _canonical_json(document).decode("utf-8").lower()
    for forbidden in ('"golden', '"output', '"shape', '"verilator', '"elf', '"path'):
        if forbidden in encoded:
            raise StageGateError(f"development feedback leaks forbidden field {forbidden}")
    return copy.deepcopy(dict(document))


def derived_peak_macs_per_cycle(rtl_facts_path: Path, target: str) -> tuple[int | None, str]:
    """The machine's structural MAC ceiling, DERIVED from its own RTL facts.

    This is the denominator of utilization, and it must never be a literal: it comes from
    ``merlin.perf.contract``, which reads the discovered array's geometry (rows x cols x the
    multipliers per element that the mac_idiom fact states) and refuses rather than inventing a peak
    when no array grounds the unit. If more than one compute resource carries a peak, this refuses
    too -- picking one would be choosing which machine the number describes.
    """
    try:
        from merlin.perf.contract import derive_contract  # noqa: PLC0415
        facts = json.loads(Path(rtl_facts_path).read_text(encoding="utf-8"))
        contract = derive_contract(target, facts=facts)
    except Exception as exc:  # noqa: BLE001 - an underivable ceiling is reported, never guessed
        return None, f"peak is not derivable from this target's RTL facts ({type(exc).__name__})"
    peaks: list[tuple[str, int]] = []
    for resource in contract.resources:
        term = (resource.terms or {}).get("peak_macs_per_cycle")
        value = getattr(term, "value", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            peaks.append((resource.name, value))
    if not peaks:
        return None, "this target's RTL facts evidence no compute array, so it has no derived peak"
    if len(peaks) > 1:
        names = ", ".join(sorted(name for name, _ in peaks))
        return None, f"this target evidences several compute units ({names}); utilization needs one"
    name, value = peaks[0]
    return value, f"facts-derived peak of compute unit {name!r}"


def declared_capsule_macs(descriptor: Mapping[str, Any]) -> tuple[int | None, str]:
    """The MACs the capsule's own declaration REQUIRES, independent of how a compiler emits them.

    Utilization must be priced against the work the spec demands, not the work the program happens to
    perform: dividing emitted MACs by cycles would reward a candidate for doing redundant arithmetic.
    Shapes come from the capsule's declared operands, so this stays a statement about the workload.
    """
    operation = descriptor.get("operation")
    if not isinstance(operation, Mapping) or operation.get("op") != "matmul":
        return None, f"declared work is derived for matmul only, not {(operation or {}).get('op')!r}"
    attributes = operation.get("attributes")
    if not isinstance(attributes, Mapping):
        return None, "the declared operation carries no operand attributes"
    shapes: dict[str, list[int]] = {}
    for row in descriptor.get("inputs") or ():
        if not isinstance(row, Mapping):
            continue
        shape = row.get("shape")
        if (isinstance(shape, Sequence) and not isinstance(shape, (str, bytes))
                and all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in shape)):
            shapes[str(row.get("name"))] = [int(v) for v in shape]
    lhs = shapes.get(str(attributes.get("lhs")))
    weight = shapes.get(str(attributes.get("weight")))
    if lhs is None or weight is None or len(lhs) != 2 or len(weight) != 2:
        return None, "the declared matmul operands are not two rank-2 shapes"
    if lhs[1] != weight[0]:
        return None, ("the declared operand shapes do not contract: "
                      f"lhs {lhs} against weight {weight}")
    return lhs[0] * lhs[1] * weight[1], "declared matmul operand shapes (M x K x N)"


_GUARD_UNCHANGED = "unchanged"
_GUARD_CHANGED = "changed"


def _emit_pair(package: "OR.Package", interface: Path, scratch: Path, tag: str,
               timeout_s: int) -> tuple[int, str, str]:
    """One capsule's emitted artifacts under one compiler: (rc, lowered LLVM, command buffer)."""
    from merlin.targetgen import oot_runner as OR  # noqa: PLC0415
    buffer_path = scratch / f"cb_{tag}.json"
    OR.run_entrypoint(package, "emit_command_buffer", interface, buffer_path, timeout=timeout_s)
    lowered = OR.run_entrypoint(package, "lower_target_to_llvm", interface, None, timeout=timeout_s)
    return (lowered.returncode, lowered.stdout or "",
            buffer_path.read_text(encoding="utf-8") if buffer_path.is_file() else "")


def functional_emission_guard(baseline: Path, candidate: Path,
                              target_experiment: TargetExperiment, *,
                              timeout_s: int = 120) -> dict[str, Any]:
    """Prove which functional capsules the perf change CANNOT have affected, and scrutinise the rest.

    Phase 1 certified the BASELINE compiler on this corpus, and the perf stage never re-grades it. If
    the candidate emits byte-identical code for a capsule, that capsule's behaviour is unchanged by
    construction and no simulation can add information; only the capsules whose emission CHANGED carry
    functional risk. That is what makes a cheap guard sound rather than merely fast.

    Measured 2026-09-03 on perf_stage_20260903T172344Z: 21 of 48 capsules were proved unchanged for
    free, and the 27 that changed included ``A6_resident_reuse`` -- the capsule whose residency
    property blocked phase-1 convergence, and which nothing in the perf path was checking.

    Trace findings are compared DIFFERENTIALLY, never absolutely. ``trace_check.check`` documents its
    violations as advisory diagnostics that deliberately do not decide pass/fail (the oracle does), so
    gating on their presence would manufacture the false refusals this harness has already produced
    twice today. A finding the CERTIFIED baseline does not produce is a different claim: a regression
    this candidate introduced.
    """
    from merlin.targetgen import oot_runner as OR  # noqa: PLC0415
    from merlin.targetgen import trace_check as TCK  # noqa: PLC0415
    from merlin.targetgen.rocc import decode as RD  # noqa: PLC0415

    siblings = getattr(target_experiment, "corpus_siblings", None)
    if callable(siblings):
        try:
            siblings = siblings()
        except Exception:  # noqa: BLE001 - a corpus we cannot enumerate is reported as unavailable
            siblings = ()
    roots: list[Path] = []
    for value in (getattr(target_experiment, "capsule_corpus", None), *(siblings or ())):
        if value and Path(value).is_dir():
            roots.append(Path(value))
    capsules = sorted({path.parent for root in roots for path in root.rglob("capsule.yaml")})
    if not capsules:
        return {"status": "unavailable", "reason": "this target declares no functional capsule corpus",
                "offenders": [], "rows": []}

    base_pkg, cand_pkg = OR.load_package(Path(baseline)), OR.load_package(Path(candidate))
    rows: list[dict[str, Any]] = []
    offenders: list[dict[str, Any]] = []
    for capsule_dir in capsules:
        name = capsule_dir.name
        try:
            descriptor = _mapping_file(capsule_dir / "capsule.yaml", yaml_file=True)
        except Exception:  # noqa: BLE001 - an unreadable descriptor is reported, never skipped silently
            offenders.append({"capsule": name, "kind": "descriptor_unreadable"})
            continue
        interface = capsule_dir / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        if not interface.is_file():
            rows.append({"capsule": name, "status": "no_interface"})
            continue
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as raw:
            scratch = Path(raw)
            base_rc, base_llvm, base_buffer = _emit_pair(
                base_pkg, interface, scratch, "baseline", timeout_s)
            cand_rc, cand_llvm, cand_buffer = _emit_pair(
                cand_pkg, interface, scratch, "candidate", timeout_s)
        if base_rc == 0 and cand_rc != 0:
            offenders.append({"capsule": name, "kind": "lowering_regressed",
                              "baseline_rc": base_rc, "candidate_rc": cand_rc})
            continue
        if (_sha256(base_llvm.encode("utf-8")) == _sha256(cand_llvm.encode("utf-8"))
                and _sha256(base_buffer.encode("utf-8")) == _sha256(cand_buffer.encode("utf-8"))):
            rows.append({"capsule": name, "status": _GUARD_UNCHANGED})
            continue
        row: dict[str, Any] = {"capsule": name, "status": _GUARD_CHANGED}
        try:
            expected = descriptor.get("expected") or {}
            base_trace = RD.decode_text(base_llvm, source="baseline",
                                        target=target_experiment.target)
            cand_trace = RD.decode_text(cand_llvm, source="candidate",
                                        target=target_experiment.target)
            base_findings = set(TCK.check(
                base_trace, expected,
                json.loads(base_buffer) if base_buffer else None)["violations"])
            cand_findings = set(TCK.check(
                cand_trace, expected,
                json.loads(cand_buffer) if cand_buffer else None)["violations"])
            introduced = sorted(cand_findings - base_findings)
            row["introduced_findings"] = introduced
            row["drives_accelerator"] = [bool(TCK.drives_accelerator(base_trace)),
                                         bool(TCK.drives_accelerator(cand_trace))]
            if row["drives_accelerator"] == [True, False]:
                offenders.append({"capsule": name, "kind": "accelerator_dispatch_regressed"})
            if introduced:
                offenders.append({"capsule": name, "kind": "trace_findings_introduced",
                                  "findings": introduced[:8]})
        except Exception as exc:  # noqa: BLE001 - an undecodable candidate trace is absence of proof
            row["decode_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            offenders.append({"capsule": name, "kind": "trace_not_decodable"})
        rows.append(row)

    unchanged = sum(1 for row in rows if row.get("status") == _GUARD_UNCHANGED)
    changed = sum(1 for row in rows if row.get("status") == _GUARD_CHANGED)
    return {"status": "clean" if not offenders else "offending",
            "capsules": len(rows), "proved_unchanged": unchanged, "changed": changed,
            "offenders": offenders, "rows": rows}


#: Fields an unmeasured cell carries. It is the same shape as a measured one so the redacted
#: schema stays exact, with every measurement-valued field null -- an absent number is never a zero.
def _unmeasured_cell(member: Any, *, reason: str) -> dict[str, Any]:
    return {
        "family": member.family, "capsule": member.capsule,
        "baseline_correct": None, "candidate_correct": None,
        "baseline_gsim_cycles": None, "candidate_gsim_cycles": None,
        "candidate_minus_baseline_cycles": None, "baseline_over_candidate": None,
        "comparable": False,
        "declared_macs": None, "declared_work_basis": None, "ideal_cycles_at_peak": None,
        "baseline_utilization": None, "candidate_utilization": None,
        "baseline_share_of_achievable": None, "candidate_share_of_achievable": None,
        "verdict": "refused", "verdict_reason": reason,
        "measured": False, "skip_reason": reason,
    }


def harvest_member_cost(roots: "Sequence[Path]") -> dict[str, float]:
    """Median measured simulation seconds per capsule, from runs already on disk.

    ORDER THE SWEEP BY WHAT IT COSTS, and derive that from measurement rather than from a proxy.
    Declared MACs are the obvious proxy and they are WRONG here: measured on this corpus, a
    262144-MAC deep-K member simulates in 74.9 s while a 65536-MAC wide-M/N member takes 178.5 s --
    the proxy inverts on exactly the pair it would need to get right. Simulation cost tracks the
    shape of the program, not the size of its arithmetic, so it is read from prior runs.

    Absent history yields an empty table and the caller keeps the declared order, saying so. An
    empty table is never a claim that every member costs the same.
    """
    seconds: dict[str, list[float]] = {}
    for root in roots:
        if not root or not Path(root).is_dir():
            continue
        for path in Path(root).rglob("capsule_result.json"):
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - an unreadable result contributes no cost, not a zero
                continue
            capsule = str(document.get("capsule") or "")
            tier = ((document.get("tiers") or {}).get("L3") or {})
            active = ((tier.get("timing") or {}).get("sim_active_s"))
            if capsule and isinstance(active, (int, float)) and not isinstance(active, bool) and active > 0:
                seconds.setdefault(capsule, []).append(float(active))
    table: dict[str, float] = {}
    for capsule, values in seconds.items():
        values.sort()
        middle = len(values) // 2
        table[capsule] = (values[middle] if len(values) % 2
                          else 0.5 * (values[middle - 1] + values[middle]))
    return table


def order_members_by_cost(members: "Sequence[Any]", cost: Mapping[str, float]) -> tuple[tuple, str]:
    """Cheapest measured member first; unpriced members last, so absence never looks cheap.

    Returns the ordering and the basis, because an ordering nobody can account for is one nobody
    can check. A member with no recorded cost sorts AFTER every priced one: it might be the most
    expensive in the corpus, and guessing it is cheap would put the slowest member first.
    """
    if not cost:
        return tuple(members), "declared order; no measured simulation cost is on record"
    ordered = sorted(members, key=lambda m: (cost.get(m.capsule) is None,
                                             cost.get(m.capsule, 0.0), m.family, m.capsule))
    priced = sum(1 for m in members if m.capsule in cost)
    return tuple(ordered), (f"ascending median measured simulation seconds "
                            f"({priced}/{len(members)} members priced; unpriced sort last)")


def prepare_development_feedback(
        *, certificate_path: Path | None, certificate_sha256: str | None,
        rtl_facts_path: Path | None, corpus: FrozenPerformanceCorpus, baseline: Path,
        baseline_sha256: str, target_experiment: TargetExperiment,
        work_root: Path,
        tuning_call_budget: int | None = None,
        functional_run_dir: Path | None = None) -> DevelopmentGsimFeedback:
    """Bind one stage to a strict certificate and exact frozen tuning corpus."""
    if certificate_path is None or not _is_sha256(certificate_sha256) or rtl_facts_path is None:
        raise StageGateError(
            "development GSIM feedback certificate is unavailable: certificate SHA and RTL facts are required")
    try:
        certificate = GATE.load_certificate(
            certificate_path, expected_sha256=str(certificate_sha256))
        if certificate.target != target_experiment.target:
            raise GATE.GsimGateError("certificate target differs from the performance target")
        import run_paired_perf_bench as PAIR
        rtl_identity = PAIR.FIXED._load_rtl_identity(Path(rtl_facts_path), target_experiment.target)
        peak_macs, peak_basis = derived_peak_macs_per_cycle(
            Path(rtl_facts_path), target_experiment.target)
        # THE ACHIEVABLE CEILING, from cycles phase 1 already paid for. The structural peak is what
        # the array could retire if nothing ever stalled; no measured program reaches it (31.3% on
        # gemmini), so optimising toward it is chasing a number that does not exist. The achievable
        # ceiling is the best rate anything on this machine actually reached, and it is what the
        # agent is asked to close on. Underivable -> None with a reason, never a substituted number.
        # Ordering the sweep by measured cost needs history; absent it the declared order stands
        # and the basis says so, rather than a proxy silently standing in for a measurement.
        member_cost: dict[str, float] = harvest_member_cost(
            [Path(functional_run_dir).parent] if functional_run_dir is not None else [])
        _, member_cost_basis = order_members_by_cost((), member_cost)
        achievable_macs, achievable_basis = None, "no functional run was supplied to harvest"
        achievable_dispersion: float | None = None
        if functional_run_dir is not None:
            try:
                import perf_model as PMODEL  # noqa: PLC0415
                points, _skipped = PMODEL.harvest_measured_points(Path(functional_run_dir))
                ceiling = PMODEL.achievable_ceiling(
                    points, provenance=f"measured cycles harvested from {Path(functional_run_dir).name}")
                try:
                    import perf_capsule_verdict as CV                        # noqa: PLC0415
                    achievable_dispersion = CV.ceiling_dispersion(
                        [{"macs": p.macs, "cycles": p.cycles} for p in points])
                except Exception:  # noqa: BLE001 - an underivable spread refuses, never defaults
                    achievable_dispersion = None
                if ceiling.known:
                    achievable_macs = float(ceiling.value)
                    achievable_basis = (f"best rate over {ceiling.n_samples} measured points in "
                                        f"{Path(functional_run_dir).name}")
                else:
                    achievable_basis = ceiling.reason
            except Exception as exc:  # noqa: BLE001 - an unharvestable corpus is reported, not faked
                achievable_basis = f"harvest failed ({type(exc).__name__})"
        decisions: dict[tuple[str, str], GATE.EvaluationDecision] = {}
        for member in sorted(corpus.capsules, key=lambda row: (row.family, row.capsule)):
            if member.descriptor.get("label") != "dev":
                raise GATE.GsimGateError(
                    f"{member.family}/{member.capsule} is not a frozen tuning member")
            workload = PAIR._gsim_workload(member)
            decision = GATE.plan_evaluation(
                certificate, workload, phase="development_correctness", gsim_available=True)
            # THE GATE GRANTS A DEVELOPMENT-ONLY REFERENCE-ENGINE FALLBACK HERE AND THIS REFUSES IT
            # ANYWAY, deliberately. Taking it would put a reference-engine cycle count into the
            # agent-visible feedback document, and that document's redaction boundary forbids naming
            # that engine at all -- a cell would either carry the forbidden name or hide which engine
            # timed it, and hiding it is worse. An out-of-envelope member is therefore admitted by
            # PAYING for its certificate offline, not by relaxing what a development cell may say.
            if (not decision.admitted or not decision.eligible
                    or decision.selected_engine != "gsim" or not decision.use_gsim):
                raise GATE.GsimGateError(
                    f"{member.family}/{member.capsule} is outside the exact GSIM certificate envelope")
            decisions[(member.family, member.capsule)] = decision
    except Exception as exc:  # noqa: BLE001 - all qualification failures become one pre-launch refusal
        raise StageGateError(
            f"development GSIM feedback certificate is unavailable or invalid: {exc}") from exc
    if str(hash_tree(baseline)["sha256"]) != baseline_sha256:
        raise StageGateError("development GSIM baseline bytes differ from the functional submission")
    return DevelopmentGsimFeedback(
        certificate, corpus, Path(baseline), baseline_sha256, target_experiment,
        rtl_identity, Path(work_root), decisions,
        peak_macs_per_cycle=peak_macs, peak_basis=peak_basis,
        achievable_macs_per_cycle=achievable_macs, achievable_basis=achievable_basis,
        achievable_dispersion=achievable_dispersion,
        member_cost=member_cost, member_cost_basis=member_cost_basis,
        tuning_call_budget=tuning_call_budget)


def action_registry_contract(actions: Sequence[BrokerAction], candidate: Path) -> list[dict[str, Any]]:
    """Normalize only the per-round candidate root; every manifest argument remains pinned."""
    root = str(candidate)
    return [{**action.as_dict(), "argv_template": [
        value.replace(root, "{candidate}", 1) if value == root or value.startswith(root + os.sep)
        else value for value in action.argv_template]}
            for action in actions]


def actions_from_registry_contract(rows: object, candidate: Path) -> tuple[BrokerAction, ...]:
    """Rebuild only the sealed action identities needed to replay transcript admission."""
    if not isinstance(rows, list) or not rows:
        raise StageGateError("sealed broker action registry is absent")
    actions: list[BrokerAction] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StageGateError(f"sealed broker action {index} is malformed")
        name, argv = row.get("name"), row.get("argv_template")
        placeholders, purpose, required = (
            row.get("placeholders"), row.get("purpose"), row.get("required"))
        if (not isinstance(name, str) or not name or Path(name).name != name
                or not isinstance(argv, list) or not argv
                or any(not isinstance(value, str) or not value for value in argv)
                or not isinstance(placeholders, list)
                or any(not isinstance(value, str) or not value for value in placeholders)
                or len(placeholders) != len(set(placeholders))
                or not isinstance(purpose, str) or not purpose
                or not isinstance(required, bool)):
            raise StageGateError(f"sealed broker action {index} violates the registry schema")
        found_placeholders = {match for value in argv for match in _PLACEHOLDER.findall(value)}
        if found_placeholders - {"candidate"} != set(placeholders):
            raise StageGateError(f"sealed broker action {index} changes its binding contract")
        expanded = tuple(
            str(candidate) + value.removeprefix("{candidate}")
            if value == "{candidate}" or value.startswith("{candidate}/") else value
            for value in argv)
        actions.append(BrokerAction(
            name, expanded, tuple(placeholders), purpose, required))
    if len({action.name for action in actions}) != len(actions):
        raise StageGateError("sealed broker action registry repeats an action")
    return tuple(actions)


def _host_lane_grant(functional: StageFunctionalRun) -> StageHostLaneGrant:
    host = functional.model_host_lane_snapshot
    package = Path(functional.model_host_package)
    return StageHostLaneGrant(
        str(host["target"]), str(host["run_id"]), str(package), str(host["package_sha256"]),
        str(package / "manifest.yaml"),
        "host-owned capsule/model runner consumes frozen schedule+knobs for scalar/RVV islands; "
        "candidate package handles accelerator regions")


def prepare_prompt_inputs(
        functional: StageFunctionalRun, frozen_functional: FrozenFunctionalInputs,
        frozen_corpus: FrozenPerformanceCorpus, agent_inputs: AgentInputSnapshot,
        target_experiment: TargetExperiment, actions: Sequence[BrokerAction], *,
        formal_claim: Mapping[str, Any], smoke_replicates: int,
        wall_budget_seconds: int, rounds: int, round_timeout_seconds: int,
        max_tool_calls: int, tool_timeout_seconds: int,
        candidate_path: str = "submission") -> StagePromptInputs:
    declaration = formal_claim.get("declaration")
    if not isinstance(declaration, Mapping) or not isinstance(declaration.get("replicates"), Mapping):
        raise StageGateError("formal PK claim omits its frozen replicate declaration")
    formal_identities = tuple(declaration["replicates"].get("identities") or ())
    replicates = declaration["replicates"].get("exact_count")
    if (isinstance(smoke_replicates, bool) or not isinstance(smoke_replicates, int)
            or smoke_replicates <= 0 or not isinstance(replicates, int)
            or smoke_replicates >= replicates):
        raise StageGateError(
            "smoke replicates must be positive and smaller than the formal PK cohort")
    evidence = declaration.get("evidence")
    if not isinstance(evidence, Mapping):
        raise StageGateError("formal PK claim omits timing-engine evidence semantics")
    timing_simulator = str(evidence.get("timing_simulator"))
    cells = tuple(PP.PerfCell(row.family, row.capsule, row.simulator, row.replicate)
                  for row in expected_perf_cells(
                      frozen_corpus.capsules, replicates, timing_simulator))
    families = _family_declarations(frozen_corpus.capsules, formal_claim)
    sentinel = select_e2e_sentinel(functional, frozen_functional, target_experiment)
    host = _host_lane_grant(functional)
    tools = tuple(PP.ToolGrant(
        action.name, f"python3 {BROKER_NAME} {action.name}" +
        (" " + " ".join(f"{name}=PATH" for name in action.placeholders)
         if action.placeholders else ""), action.purpose, action.required) for action in actions)
    # THE AGENT SEES SANDBOX PATHS, NOT HOST PATHS. `agent_inputs.root` is bound read-only at
    # AGENT_CORPUS_MOUNT (see inner_broker_policy / outer_codex_policy), so inside bwrap the manifest
    # is at `/perf-corpus/agent_input_manifest.json`. Declaring `agent_inputs.manifest_path` here named
    # the HOST path, which is not bound -- and the prompt tells the agent to stop with NO-GO if any
    # declared path is absent. Measured 2026-09-03: the agent probed the set, found this one missing,
    # and correctly refused in 24 s ("the required declared mount is absent"), so the stage produced a
    # candidate with zero authoring rounds and no transcript to audit. Every other entry here is
    # already a mount-side path; this was the one host-side straggler.
    allowed = (str(FUNCTIONAL_BASE_MOUNT), candidate_path, str(AGENT_CORPUS_MOUNT),
               str(AGENT_CORPUS_MOUNT / agent_inputs.manifest_path.name),
               host.package_path, host.manifest_path,
               sentinel.capsule_path, BROKER_NAME, str(BROKER_RECEIPT_MOUNT),
               str(FUNCTIONAL_INPUT_MANIFEST_MOUNT), str(PERF_CORPUS_MANIFEST_MOUNT),
               *(str(grant.destination) for grant in frozen_functional.grants))
    return StagePromptInputs(
        target=target_experiment.target, approach="arm4", functional_run_id=functional.run_id,
        functional_submission_sha256=functional.digest,
        frozen_functional_path=str(FUNCTIONAL_BASE_MOUNT),
        frozen_functional_sha256=functional.digest, submission_path=candidate_path,
        submission_initial_sha256=functional.digest,
        functional_public_capsules=functional.public_capsules,
        functional_hidden_capsules=functional.hidden_capsules,
        functional_bundle_snapshot_manifest=str(FUNCTIONAL_INPUT_MANIFEST_MOUNT),
        functional_bundle_snapshot_manifest_sha256=frozen_functional.marker_sha256,
        functional_bundle_snapshot_sha256=frozen_functional.content_sha256,
        workload_root=str(AGENT_CORPUS_MOUNT), workload_manifest=str(PERF_CORPUS_MANIFEST_MOUNT),
        workload_manifest_sha256=frozen_corpus.manifest_sha256,
        workload_capsules_sha256=frozen_corpus.capsules_sha256,
        expected_cells=cells, replicates=replicates,
        formal_replicate_identities=formal_identities,
        formal_claim=copy.deepcopy(dict(formal_claim)), smoke_replicates=smoke_replicates,
        wall_budget_seconds=wall_budget_seconds,
        rounds=rounds, round_timeout_seconds=round_timeout_seconds,
        max_tool_calls=max_tool_calls, tool_timeout_seconds=tool_timeout_seconds,
        families=families, host_lane=host,
        e2e_sentinel=sentinel, tools=tools, allowed_paths=tuple(dict.fromkeys(allowed)),
        execution_broker_path=BROKER_NAME,
        execution_broker_command=f"python3 {BROKER_NAME}",
        broker_receipt_path=str(BROKER_RECEIPT_MOUNT))


def _path_is_answer(path: Path, surfaces: Sequence) -> bool:
    resolved = path.resolve()
    for surface in surfaces:
        answer = Path(surface.path).resolve()
        if resolved == answer or (surface.kind == "dir" and answer in resolved.parents):
            return True
    return False


def build_answer_free_agent_inputs(
        corpus: FrozenPerformanceCorpus, target_experiment: TargetExperiment,
        destination: Path) -> AgentInputSnapshot:
    """Copy only non-answer capsule bytes into the read-only view exposed to the agent.

    The complete frozen corpus remains host-only.  Filtering is derived from the same answer-surface
    registry used by the bwrap coverage proof; no golden filename allow/deny list is maintained here.
    """
    raw_destination = Path(destination)
    if raw_destination.exists() or raw_destination.is_symlink():
        raise StageGateError(f"agent input snapshot already exists: {raw_destination}")
    destination = raw_destination.resolve()
    surfaces = answer_surfaces(target_experiment)
    rows: list[dict[str, Any]] = []
    destination.mkdir(parents=True)
    for member in corpus.capsules:
        original = Path(target_experiment.capsule_corpus).resolve().parent / member.source_relative_path
        for frozen_file in sorted(path for path in member.source_dir.rglob("*") if path.is_file()):
            relative_in_capsule = frozen_file.relative_to(member.source_dir)
            original_file = original / relative_in_capsule
            if _path_is_answer(original_file, surfaces):
                continue
            relative = Path(member.source_relative_path) / relative_in_capsule
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = frozen_file.read_bytes()
            output.write_bytes(payload)
            rows.append({"path": relative.as_posix(), "sha256": _sha256(payload),
                         "n_bytes": len(payload)})
    if not rows:
        raise StageGateError("answer-free performance input view contains zero files")
    aggregate = hashlib.sha256()
    for row in rows:
        aggregate.update(row["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(row["sha256"].encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
    content_sha = aggregate.hexdigest()
    manifest = {
        "schema_version": 1,
        "source_performance_manifest_sha256": corpus.manifest_sha256,
        "source_performance_corpus_sha256": corpus.capsules_sha256,
        "answer_surface_registry": "merlin.targetgen.sandbox.answer_surfaces",
        "files": rows,
        "content_sha256": content_sha,
        "n_files": len(rows),
        "n_bytes": sum(int(row["n_bytes"]) for row in rows),
    }
    manifest_path = destination / "agent_input_manifest.json"
    payload = _canonical_json(manifest)
    manifest_path.write_bytes(payload)
    for path in sorted(destination.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    destination.chmod(0o555)
    return AgentInputSnapshot(destination, manifest_path, _sha256(payload), content_sha,
                              len(rows), int(manifest["n_bytes"]))


def verify_answer_free_agent_inputs(snapshot: AgentInputSnapshot) -> None:
    if _sha256(snapshot.manifest_path.read_bytes()) != snapshot.manifest_sha256:
        raise StageGateError("answer-free performance input manifest changed")
    document = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    rows = document.get("files")
    if not isinstance(rows, list) or len(rows) != snapshot.n_files or not rows:
        raise StageGateError("answer-free performance input manifest is incomplete")
    aggregate = hashlib.sha256()
    total = 0
    for row in rows:
        relative = Path(str(row.get("path") or ""))
        if relative.is_absolute() or ".." in relative.parts:
            raise StageGateError("answer-free performance input path escapes its snapshot")
        path = snapshot.root / relative
        if path.is_symlink() or not path.is_file():
            raise StageGateError(f"answer-free performance input is absent or linked: {relative}")
        payload = path.read_bytes()
        if _sha256(payload) != row.get("sha256") or len(payload) != row.get("n_bytes"):
            raise StageGateError(f"answer-free performance input changed: {relative}")
        aggregate.update(relative.as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(row["sha256"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(row["n_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
        total += len(payload)
    if aggregate.hexdigest() != snapshot.content_sha256 or total != snapshot.n_bytes:
        raise StageGateError("answer-free performance input aggregate digest changed")


def _strip_claude_home(argv: Sequence[str]) -> list[str]:
    """Remove the broad Claude credential/history bind inherited from the shared bwrap base."""
    claude_home = Path.home() / ".claude"
    output: list[str] = []
    index = 0
    while index < len(argv):
        option = argv[index]
        if option in ("--bind", "--ro-bind", "--bind-try", "--ro-bind-try") and index + 2 < len(argv):
            destination = Path(argv[index + 2])
            if destination == claude_home or claude_home in destination.parents:
                index += 3
                continue
        if option == "--tmpfs" and index + 1 < len(argv):
            destination = Path(argv[index + 1])
            if destination == claude_home or claude_home in destination.parents:
                index += 2
                continue
        output.append(option)
        index += 1
    return output


def inner_execution_policy(target_experiment: TargetExperiment, candidate: Path,
                           agent_inputs: AgentInputSnapshot,
                           frozen_functional: FrozenFunctionalInputs | None = None,
                           functional_base: Path | None = None,
                           frozen_corpus_manifest: Path | None = None) -> AgentSandboxPolicy:
    """Build the credential-free, answer-masked sandbox used by the local tool broker."""
    candidate = _require_real_directory(candidate, label="performance candidate")
    verify_answer_free_agent_inputs(agent_inputs)
    argv = _strip_claude_home(BW.base_argv(candidate, {}, _policy_test_live_inputs=True))
    argv += ["--clearenv", "--setenv", "HOME", "/tmp",
             "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += TC.toolchain_binds(target_experiment)
    if frozen_functional is not None:
        argv += frozen_grant_mounts(frozen_functional)
        argv += ["--ro-bind", str(frozen_functional.marker),
                 str(FUNCTIONAL_INPUT_MANIFEST_MOUNT)]
    if functional_base is not None:
        argv += ["--ro-bind", str(functional_base), str(FUNCTIONAL_BASE_MOUNT)]
    if frozen_corpus_manifest is not None:
        argv += ["--ro-bind", str(frozen_corpus_manifest), str(PERF_CORPUS_MANIFEST_MOUNT)]
    argv += ["--ro-bind", str(agent_inputs.root), str(AGENT_CORPUS_MOUNT)]
    surfaces = answer_surfaces(target_experiment)
    argv = BW.apply_answer_masks(argv, surfaces)
    if "--unshare-net" in argv:
        raise StageGateError("inner execution policy unexpectedly disables required network availability")
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise StageGateError(f"inner performance-tool sandbox exposes answer surfaces: {gaps}")
    joined = " ".join(argv)
    for token in (".codex/auth.json", "AWS_ACCESS_KEY_ID", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if token in joined:
            raise StageGateError(f"inner performance-tool sandbox exposes credential token {token!r}")
    return AgentSandboxPolicy(
        tuple(argv), gaps, "available_not_an_isolation_claim", True, True, True)


def outer_codex_policy(workspace: Path, agent_inputs: AgentInputSnapshot,
                       runtime_binds: Sequence[str], target_experiment: TargetExperiment,
                       frozen_functional: FrozenFunctionalInputs | None = None,
                       functional_base: Path | None = None,
                       control_dir: Path | None = None,
                       frozen_corpus_manifest: Path | None = None) -> AgentSandboxPolicy:
    """Build Codex's filesystem boundary; network/auth exceptions are explicit and recorded."""
    workspace = _require_real_directory(workspace, label="Codex round workspace")
    verify_answer_free_agent_inputs(agent_inputs)
    argv = _strip_claude_home(BW.base_argv(workspace, {}, _policy_test_live_inputs=True))
    argv += ["--clearenv", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
             "--setenv", "XDG_RUNTIME_DIR", "/tmp/.xdg"]
    argv += list(runtime_binds)
    if frozen_functional is not None:
        argv += frozen_grant_mounts(frozen_functional)
        argv += ["--ro-bind", str(frozen_functional.marker),
                 str(FUNCTIONAL_INPUT_MANIFEST_MOUNT)]
    if functional_base is not None:
        argv += ["--ro-bind", str(functional_base), str(FUNCTIONAL_BASE_MOUNT)]
    if control_dir is not None:
        argv += ["--ro-bind", str(control_dir), "/perf-control"]
    if frozen_corpus_manifest is not None:
        argv += ["--ro-bind", str(frozen_corpus_manifest), str(PERF_CORPUS_MANIFEST_MOUNT)]
    argv += ["--ro-bind", str(agent_inputs.root), str(AGENT_CORPUS_MOUNT)]
    surfaces = answer_surfaces(target_experiment)
    argv = BW.apply_answer_masks(argv, surfaces)
    if "--unshare-net" in argv:
        raise StageGateError("outer Codex policy unexpectedly disables required network availability")
    gaps = tuple(str(surface.path) for surface in BW.coverage_gap(argv, surfaces))
    if gaps:
        raise StageGateError(f"outer Codex sandbox exposes answer surfaces: {gaps}")
    return AgentSandboxPolicy(
        tuple(argv), gaps, "available_not_an_isolation_claim", True, True, True)


def inner_command(policy: AgentSandboxPolicy, target_experiment: TargetExperiment,
                  candidate: Path, argv: Sequence[str], timeout_s: int) -> list[str]:
    """Construct one shell-free payload for the inner broker."""
    if policy.network != "available_not_an_isolation_claim" or not policy.clear_environment:
        raise StageGateError("inner command requires the explicit clear-environment policy")
    if (not argv or any(not isinstance(value, str) or not value or "\0" in value for value in argv)
            or len(argv) > 256 or sum(len(value) for value in argv) > 131_072):
        raise StageGateError("inner tool argv is empty, malformed, or too large")
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, int) or timeout_s <= 0:
        raise StageGateError("inner tool timeout must be a positive integer")
    environment = TC.sandbox_env(target_experiment, candidate)
    return [*policy.argv, "--chdir", str(candidate), "bash", "-c",
            environment + 'exec "$@"', "perf-tool", *argv]


def run_required_tool_probes(policy: AgentSandboxPolicy, target_experiment: TargetExperiment,
                             candidate: Path, *, timeout_s: int = 60) -> list[dict[str, Any]]:
    probes = TC.required_tool_probes(target_experiment)
    if not probes:
        raise StageGateError("inner execution policy derives zero required tool probes")
    rows: list[dict[str, Any]] = []
    for probe in probes:
        command = [*policy.argv, "--chdir", str(candidate), "bash", "-c",
                   TC.sandbox_env(target_experiment, candidate) + probe.cmd]
        proc = subprocess.run(command, cwd=str(repo_root()), capture_output=True, text=True,
                              timeout=timeout_s)
        row = {"label": probe.label, "returncode": proc.returncode,
               "command": probe.cmd, "bind": probe.bind,
               "stdout": (proc.stdout or "")[-400:], "stderr": (proc.stderr or "")[-400:]}
        rows.append(row)
        if proc.returncode != 0:
            raise StageGateError(
                f"required inner-sandbox tool probe {probe.label!r} failed with rc={proc.returncode}")
    return rows


class _Broker:
    """A bounded localhost bridge from Codex to the credential-free inner bwrap."""

    def __init__(self, policy: AgentSandboxPolicy, target_experiment: TargetExperiment,
                 candidate: Path, actions: Sequence[BrokerAction], receipt_path: Path, *,
                 deadline: float, max_calls: int, max_tool_seconds: int,
                 feedback_evaluator: DevelopmentGsimFeedback | None = None,
                 feedback_round: int | None = None):
        self.policy = policy
        self.target_experiment = target_experiment
        self.candidate = candidate
        self.deadline = deadline
        self.max_calls = max_calls
        self.max_tool_seconds = max_tool_seconds
        self.actions = {action.name: action for action in actions}
        self.feedback_evaluator = feedback_evaluator
        self.feedback_round = feedback_round
        if not self.actions or len(self.actions) != len(actions):
            raise StageGateError("broker requires a non-empty unique action registry")
        self.receipt_path = receipt_path
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        if self.receipt_path.exists() or self.receipt_path.is_symlink():
            raise StageGateError(f"broker receipt path is not fresh: {self.receipt_path}")
        self.receipt_path.touch(mode=0o600)
        self.token = secrets.token_urlsafe(32)
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def _refuse(self, action_name: str, bindings: Mapping[str, Any],
                reason: str) -> StageGateError:
        """Record a REFUSED invocation in the ledger, then hand back the error to raise.

        A refusal the agent sees but the ledger never records is a hole in the receipts-to-transcript
        join: the transcript shows an invocation with no receipt, and `verify_broker_receipts` cannot
        tell "the broker refused this" from "a receipt went missing", so it refuses the entire run.
        Measured 2026-09-03 on perf_stage_20260903T151801Z -- the agent aimed one `output_json=` at
        /workspace (outside the mounts), was refused here before any ledger entry existed, corrected
        itself on the very next call, and a complete round was thrown away over 25 receipts against
        26 invocations.

        This ADDS evidence rather than relaxing anything: a refused row can never satisfy a required
        action, because only `state == "complete"` with returncode 0 counts, and an escape ATTEMPT is
        now visible in the ledger instead of vanishing from it.
        """
        with self._lock:
            if len(self.calls) >= self.max_calls:
                # Budget is spent, so the run ends here anyway; do not let refusals grow the ledger.
                return StageGateError("inner tool-call budget is exhausted")
            call_index = len(self.calls)
            recorded = {key: str(value) for key, value in sorted(bindings.items())}
            entry: dict[str, Any] = {
                "index": call_index,
                "action": action_name,
                "bindings": recorded,
                "argv_sha256": _sha256(_canonical_json(
                    list(self.actions[action_name].argv_template))),
                "timeout_s": 0,
                "state": "rejected",
                "returncode": 126,
                "stdout_sha256": _sha256(b""),
                "stderr_sha256": _sha256(b""),
                "rejection_reason": reason,
            }
            self.calls.append(entry)
            receipt = dict(entry)
            receipt["receipt_schema_version"] = 1
            receipt["bindings_command_sha256"] = _sha256(_canonical_json(
                [f"{key}={value}" for key, value in sorted(recorded.items())]))
            payload = _canonical_json(receipt)
            with self.receipt_path.open("ab", buffering=0) as stream:
                stream.write(payload)
                os.fsync(stream.fileno())
        return StageGateError(reason)

    def execute(self, request: Mapping[str, Any]) -> dict[str, Any]:
        action_name, bindings = request.get("action"), request.get("bindings", {})
        if not isinstance(action_name, str) or action_name not in self.actions:
            raise StageGateError("broker request names an undeclared action")
        if not isinstance(bindings, Mapping):
            raise StageGateError("broker request bindings must be a mapping")
        action = self.actions[action_name]
        if set(bindings) != set(action.placeholders):
            raise self._refuse(action_name, bindings,
                               f"broker action {action_name!r} requires exact bindings "
                               f"{action.placeholders}")
        rendered: dict[str, str] = {}
        for name, value in bindings.items():
            if not isinstance(value, str) or not value or "\0" in value or len(value) > 8192:
                raise self._refuse(action_name, bindings,
                                   f"broker binding {name!r} is malformed")
            path = Path(value)
            if path.is_absolute():
                allowed_roots = (self.candidate, AGENT_CORPUS_MOUNT)
                if not any(path == root or root in path.parents for root in allowed_roots):
                    raise self._refuse(action_name, bindings,
                                       f"broker binding {name!r} escapes declared inputs")
            elif ".." in path.parts:
                raise self._refuse(action_name, bindings,
                                   f"broker binding {name!r} escapes the candidate")
            if name.startswith("output") and path.is_absolute() \
                    and not (path == self.candidate or self.candidate in path.parents):
                raise self._refuse(action_name, bindings,
                                   f"broker output binding {name!r} is not candidate-scoped")
            rendered[name] = value
        raw_argv = [
            _PLACEHOLDER.sub(lambda match: rendered[match.group(1)], value)
            for value in action.argv_template
        ]
        with self._lock:
            if len(self.calls) >= self.max_calls:
                raise StageGateError("inner tool-call budget is exhausted")
            remaining = int(self.deadline - time.monotonic())
            requested = request.get("timeout_s", self.max_tool_seconds)
            if isinstance(requested, bool) or not isinstance(requested, int):
                raise StageGateError("broker timeout must be an integer")
            timeout_s = min(requested, self.max_tool_seconds, remaining)
            if timeout_s <= 0:
                raise StageGateError("performance stage wall-clock budget is exhausted")
            call_index = len(self.calls)
            self.calls.append({"index": call_index, "action": action_name,
                               "bindings": dict(sorted(rendered.items())),
                               "argv_sha256": _sha256(_canonical_json(raw_argv)),
                               "timeout_s": timeout_s, "state": "running"})
        started = time.monotonic()
        try:
            return self._execute_allocated(request, action_name, rendered, raw_argv,
                                           call_index, timeout_s, started)
        except StageGateError:
            # EVERY ALLOCATED INDEX GETS A RECEIPT. The index is taken before the inner command is
            # built, and building it can refuse (clear-environment policy, malformed argv, a
            # non-positive timeout). A refusal there used to leave the index allocated with nothing
            # written, so the receipt stream skipped a number and `verify_broker_receipts` rejected
            # the NEXT row with "violates the action schema" -- which killed a trial on 2026-09-03
            # (perf_agentic_20260903T212924Z__trial_01, receipt 6) after 42 clean invocations. The
            # ledger has to be gapless for the join to mean anything.
            with self._lock:
                # The allocated entry is already marked "running"; only a row that never
                # reached "complete" needs the refusal receipt written for it.
                if (len(self.calls) > call_index
                        and self.calls[call_index].get("state") == "running"):
                    self.calls[call_index] = {**self.calls[call_index], "state": "rejected",
                                              "returncode": 126,
                                              "stdout_sha256": _sha256(b""),
                                              "stderr_sha256": _sha256(b""),
                                              "rejection_reason": "inner command was refused "
                                                                  "before it could run"}
                    receipt = dict(self.calls[call_index])
                    receipt["receipt_schema_version"] = 1
                    receipt["bindings_command_sha256"] = _sha256(_canonical_json(
                        [f"{k}={v}" for k, v in sorted(rendered.items())]))
                    payload = _canonical_json(receipt)
                    with self.receipt_path.open("ab", buffering=0) as stream:
                        stream.write(payload)
                        os.fsync(stream.fileno())
            raise

    def _execute_allocated(self, request: Mapping[str, Any], action_name: str,
                           rendered: dict[str, str], raw_argv: list[str], call_index: int,
                           timeout_s: int, started: float) -> dict[str, Any]:
        feedback_document: dict[str, Any] | None = None
        # set when the search reports it has converged; distinct from `refusal`, which is a NO-GO
        self.stop_verdict = getattr(self, "stop_verdict", None)
        if action_name == ANALYSIS_ACTION:
            try:
                evaluator = self.feedback_evaluator
                document = analyze_command_buffers(
                    Path(rendered["baseline_json"]), Path(rendered["candidate_json"]),
                    peak_macs_per_cycle=getattr(evaluator, "peak_macs_per_cycle", None),
                    achievable_macs_per_cycle=getattr(
                        evaluator, "achievable_macs_per_cycle", None),
                    target=str(getattr(
                        getattr(evaluator, "target_experiment", None), "target", "") or ""))
                result = {"returncode": 0,
                          "stdout": _canonical_json(document).decode("utf-8"), "stderr": "",
                          "elapsed_s": round(time.monotonic() - started, 3)}
            except Exception as exc:  # noqa: BLE001 - an unreadable buffer is a refusal, not a crash
                result = {"returncode": 125, "stdout": "",
                          "stderr": f"command-buffer analysis refused ({type(exc).__name__}: "
                                    f"{str(exc)[:200]})",
                          "elapsed_s": round(time.monotonic() - started, 3)}
        elif action_name == DEVELOPMENT_FEEDBACK_ACTION:
            try:
                if self.feedback_evaluator is None or self.feedback_round is None:
                    raise StageGateError("development GSIM feedback certificate is unavailable")
                feedback_document = validate_redacted_feedback(self.feedback_evaluator.evaluate(
                    self.candidate, round_index=self.feedback_round, call_index=call_index,
                    timeout_s=timeout_s))
                stdout = _canonical_json(feedback_document).decode("utf-8")
                # THE SEARCH'S OWN VERDICT, kept so the round loop can end on evidence. It was
                # computed and recorded and read by nothing, so a converged search looked exactly
                # like one that had merely run out of rounds.
                stopping = feedback_document.get("stopping")
                if isinstance(stopping, Mapping) and stopping.get("status") == "stop":
                    fired = [str(v.get("name")) for v in (stopping.get("verdicts") or [])
                             if isinstance(v, Mapping) and v.get("fired")]
                    self.stop_verdict = {"conditions": fired,
                                         "share_of_attainable": stopping.get("share_of_attainable"),
                                         "queries": stopping.get("queries")}
                result = {"returncode": 0, "stdout": stdout, "stderr": "",
                          "elapsed_s": round(time.monotonic() - started, 3)}
            except Exception as exc:  # noqa: BLE001 - receipt the refusal; never expose raw evaluator data
                result = {"returncode": 125, "stdout": "",
                          "stderr": ("development GSIM feedback refused by the host-owned "
                                     f"evaluator ({type(exc).__name__})"),
                          "elapsed_s": round(time.monotonic() - started, 3)}
        else:
            command = inner_command(
                self.policy, self.target_experiment, self.candidate, raw_argv, timeout_s)
            try:
                proc = subprocess.run(command, cwd=str(repo_root()), capture_output=True, text=True,
                                      timeout=timeout_s)
                result = {"returncode": proc.returncode, "stdout": (proc.stdout or "")[-1_000_000:],
                          "stderr": (proc.stderr or "")[-1_000_000:],
                          "elapsed_s": round(time.monotonic() - started, 3)}
            except subprocess.TimeoutExpired as exc:
                result = {"returncode": 124, "stdout": str(exc.stdout or "")[-1_000_000:],
                          "stderr": str(exc.stderr or "")[-1_000_000:], "timed_out": True,
                          "elapsed_s": round(time.monotonic() - started, 3)}
        with self._lock:
            self.calls[call_index].update({key: value for key, value in result.items()
                                           if key not in ("stdout", "stderr")})
            self.calls[call_index]["stdout_sha256"] = _sha256(result["stdout"].encode("utf-8"))
            self.calls[call_index]["stderr_sha256"] = _sha256(result["stderr"].encode("utf-8"))
            if feedback_document is not None and result["returncode"] == 0:
                feedback_payload = _canonical_json(feedback_document)
                feedback_sha = _sha256(feedback_payload)
                feedback_dir = self.receipt_path.parent / "feedback" / "sha256"
                feedback_dir.mkdir(parents=True, exist_ok=True)
                feedback_path = feedback_dir / f"{feedback_sha}.json"
                if feedback_path.exists() or feedback_path.is_symlink():
                    if feedback_path.is_symlink() or feedback_path.read_bytes() != feedback_payload:
                        raise StageGateError("development feedback receipt digest collision")
                else:
                    with feedback_path.open("xb") as stream:
                        stream.write(feedback_payload)
                        stream.flush()
                        os.fsync(stream.fileno())
                    feedback_path.chmod(0o444)
                self.calls[call_index]["feedback_receipt_sha256"] = feedback_sha
                self.calls[call_index]["feedback_receipt_path"] = str(feedback_path.resolve())
            self.calls[call_index]["state"] = "complete"
            receipt = dict(self.calls[call_index])
            receipt["receipt_schema_version"] = 1
            receipt["bindings_command_sha256"] = _sha256(_canonical_json(
                [f"{key}={value}" for key, value in sorted(rendered.items())]))
            payload = _canonical_json(receipt)
            with self.receipt_path.open("ab", buffering=0) as stream:
                stream.write(payload)
                os.fsync(stream.fileno())
        return result

    @contextlib.contextmanager
    def serving(self) -> Iterator[tuple[str, int]]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
                try:
                    if self.path != "/execute" or self.headers.get("X-Perf-Token") != owner.token:
                        self.send_error(403)
                        return
                    length = int(self.headers.get("Content-Length") or 0)
                    if length <= 0 or length > 1_000_000:
                        self.send_error(400)
                        return
                    request = json.loads(self.rfile.read(length))
                    if not isinstance(request, dict):
                        raise StageGateError("broker request must be a JSON mapping")
                    response = owner.execute(request)
                    payload = _canonical_json(response)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (StageGateError, ValueError, json.JSONDecodeError) as exc:
                    payload = _canonical_json({"error": f"{type(exc).__name__}: {exc}"})
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            def log_message(self, _format: str, *args: object) -> None:
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        try:
            yield "127.0.0.1", int(self._server.server_address[1])
        finally:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=5)


_BROKER_SHIM = '''#!/usr/bin/env python3
import json, pathlib, sys, urllib.error, urllib.request
cfg = json.loads((pathlib.Path(__file__).parent / ".perf_broker.json").read_text())
argv = sys.argv[1:]
if not argv or argv[0] not in cfg["actions"]:
    raise SystemExit("usage: python3 /perf-control/perf_tool.py ACTION [NAME=VALUE ...]")
action, bindings = argv[0], {}
for item in argv[1:]:
    if "=" not in item:
        raise SystemExit("broker action arguments must be exact NAME=VALUE bindings")
    name, value = item.split("=", 1)
    if not name or name in bindings:
        raise SystemExit("broker action binding is empty or repeated")
    bindings[name] = value
request = urllib.request.Request(
    cfg["url"], data=json.dumps({"action": action, "bindings": bindings,
                                 "timeout_s": cfg["tool_timeout_s"]}).encode(),
    headers={"Content-Type": "application/json", "X-Perf-Token": cfg["token"]}, method="POST")
try:
    with urllib.request.urlopen(request, timeout=cfg["tool_timeout_s"] + 10) as response:
        result = json.load(response)
except urllib.error.HTTPError as exc:
    sys.stderr.write(exc.read().decode(errors="replace"))
    raise SystemExit(125)
sys.stdout.write(result.get("stdout") or "")
sys.stderr.write(result.get("stderr") or "")
raise SystemExit(int(result.get("returncode", 125)))
'''


def stage_broker_shim(control_dir: Path, *, host: str, port: int, token: str,
                      tool_timeout_s: int, actions: Sequence[BrokerAction]) -> Path:
    if control_dir.is_symlink() or (control_dir.exists() and not control_dir.is_dir()):
        raise StageGateError(f"broker control directory is unsafe: {control_dir}")
    control_dir.mkdir(parents=True, exist_ok=True)
    unexpected = [path for path in control_dir.iterdir() if path.name != "receipts.jsonl"]
    if unexpected:
        raise StageGateError(f"broker control directory is not fresh: {control_dir}")
    shim = control_dir / Path(BROKER_NAME).name
    shim.write_text(_BROKER_SHIM, encoding="utf-8")
    shim.chmod(0o555)
    config = control_dir / ".perf_broker.json"
    _write_json(config, {
        "url": f"http://{host}:{port}/execute", "token": token,
        "tool_timeout_s": tool_timeout_s,
        "actions": sorted(action.name for action in actions),
    })
    config.chmod(0o444)
    return shim


def _heredoc_delimiter(line: str) -> str | None:
    """The delimiter word of a heredoc redirection on this line, or None.

    Structural, not pattern-matched: locate ``<<``, skip the ``<<<`` here-string form, drop an
    optional ``-`` (tab-stripping form), and take the next word with its quoting removed.
    """
    marker = line.find("<<")
    if marker < 0 or line[marker:marker + 3] == "<<<":
        return None
    rest = line[marker + 2:].lstrip()
    if rest.startswith("-"):
        rest = rest[1:].lstrip()
    fields = rest.split()
    if not fields:
        return None
    return fields[0].strip("'\"") or None


def _split_heredocs(text: str) -> tuple[list[str], str]:
    """Peel heredoc bodies out of a shell payload.

    A heredoc body is DATA to the shell but SOURCE to the interpreter that reads it, so it has to be
    audited as source -- and it must NOT be lexed as further commands. Its lines are not commands,
    and counting them as such corrupts the brokered/total ratio that decides whether a payload mixed
    brokered and unbrokered work.
    """
    bodies: list[str] = []
    kept: list[str] = []
    lines = text.replace("\r", "\n").split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        kept.append(line)
        delimiter = _heredoc_delimiter(line)
        index += 1
        if delimiter is None:
            continue
        body: list[str] = []
        while index < len(lines) and lines[index].strip() != delimiter:
            body.append(lines[index])
            index += 1
        index += 1  # consume the terminator line (past-the-end is fine: unterminated heredoc)
        bodies.append("\n".join(body))
    return bodies, "\n".join(kept)


def audit_codex_transcript(path: Path, target_experiment: TargetExperiment,
                           candidate: Path,
                           actions: Sequence[BrokerAction] = ()) -> dict[str, Any]:
    """Reject answer reconnaissance and direct execution in translated or native Codex JSONL.

    This audit is a second line of defence.  Answer bytes are absent from the outer mount table, and
    target tools are available only to the inner broker; an audit hit therefore makes the candidate
    non-consumable even if the attempted command failed.  Native Codex emits both ``item.started`` and
    ``item.completed`` for one command, so those envelopes are validated independently but counted once.
    """
    tokens = audit_tokens(target_experiment)
    answer_tokens = tuple(value.lower() for values in tokens.values() for value in values if value)
    entry_tokens: set[str] = set()
    manifest = candidate / "manifest.yaml"
    if manifest.is_file():
        document = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        entries = document.get("entrypoints") if isinstance(document, dict) else {}
        if isinstance(entries, dict):
            entry_tokens.update(Path(str(value)).name.lower() for value in entries.values()
                                if isinstance(value, str))
    entry_tokens -= {"python", "python3", "bash", "sh", "env"}
    target_tool_tokens: set[str] = set()
    ordinary_shell = {"awk", "basename", "bash", "cat", "command", "cut", "dirname", "echo",
                      "env", "false", "find", "grep", "head", "ls", "printf", "pwd", "python",
                      "python3", "readlink", "sed", "sh", "tail", "test", "true", "wc", "which"}
    for probe in TC.required_tool_probes(target_experiment):
        first = probe.cmd.strip().split(maxsplit=1)[0] if probe.cmd.strip() else ""
        if first and Path(first).name.lower() not in ordinary_shell:
            target_tool_tokens.add(Path(first).name.lower())
        label_token = re.match(r"[A-Za-z0-9+_.-]+", probe.label)
        if label_token and label_token.group(0).lower() not in ordinary_shell:
            target_tool_tokens.add(label_token.group(0).lower())
    hits: list[dict[str, str]] = []
    broker_invocations: list[dict[str, Any]] = []
    action_names = {action.name for action in actions}
    commands_seen = 0
    current_thread = "no-thread"
    native_commands: dict[tuple[str, str], str] = {}
    known_native_items = {
        "agent_message", "command_execution", "error", "file_change", "mcp_tool_call",
        "reasoning", "web_search",
    }

    def audit_command(command: str, line_number: int) -> None:
        nonlocal commands_seen
        commands_seen += 1
        lowered = command.lower()
        command_words = {
            Path(word.strip("'\";|&()[]{}<>")).name
            for word in lowered.replace("\n", " ").split()
        }
        try:
            outer_words = shlex.split(command)
        except ValueError:
            hits.append({"kind": "malformed_shell_command", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
            outer_words = []
        words: list[str] = []
        payload_text = command
        if (len(outer_words) == 3 and Path(outer_words[0]).name in ("bash", "sh")
                and outer_words[1] in ("-c", "-lc")):
            payload_text = outer_words[2]
        try:
            lexer = shlex.shlex(payload_text, posix=True, punctuation_chars=";&|<>()")
            lexer.whitespace_split = True
            lexer.commenters = ""
            words = list(lexer)
        except ValueError:
            words = []
            hits.append({"kind": "malformed_shell_command", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        # DECOMPOSE COMPOUND COMMANDS, then apply the SAME test to each simple command.
        #
        # This is a more precise parse, not a looser rule. The previous version tested the whole
        # payload as one string, which produced two false accusations against a correctly-behaving
        # agent (measured 2026-09-03, 6 hits on a 28-command round):
        #
        #  * a payload containing "\n" was rejected wholesale, so a batch of TEN one-per-line broker
        #    calls counted as zero invocations while the host wrote ten receipts -- surfacing as
        #    "host broker receipts do not exactly match transcript invocations" (12 vs 22);
        #  * `BROKER_NAME in lowered` fired whenever the broker path merely APPEARED, so
        #    `ls -l /perf-control/perf_tool.py` was an "invalid broker invocation" -- while the prompt
        #    ORDERS the agent to stop with NO-GO unless it verifies that exact file exists. The audit
        #    punished the behaviour the prompt demands.
        #
        # Splitting on shell separators keeps every guarantee: each simple command must still be a
        # clean, single, well-formed invocation with known action and unique bindings, and the
        # `$`/backtick ban still applies WITHIN each one. What changes is that a mention of the broker
        # as an ARGUMENT to another program is data, not an invocation -- unless that program could
        # execute it, which is what the obfuscation check below still catches.
        _SEPARATORS = ("&&", "||", ";", "|", "\n", "\r")
        # DENY BY DEFAULT. A simple command may NAME the broker only as an argument to one of these
        # read-only inspection verbs; anything else that mentions it is an invalid invocation. An
        # allowlist is the safe direction here: a deny-list of `eval`/`$(`/backtick lets
        # `cp <broker> /tmp/x && python3 /tmp/x ...` and
        # `python3 -c 'exec(open("<broker>").read())'` straight through, both of which are pinned as
        # must-fail by test_wrapped_broker_compound_rename_and_python_exec_forms_fail_closed.
        _READ_ONLY_VERBS = {"ls", "stat", "cat", "head", "tail", "wc", "sed", "grep", "find",
                            "file", "readlink", "test", "diff", "sha256sum", "md5sum", "cksum",
                            "du", "basename", "dirname", "realpath"}

        def _simple_commands(text: str) -> list[list[str]]:
            """Split into simple commands as TOKEN LISTS, quote-aware.

            Splitting the raw string on separators is wrong: it cuts inside quotes and leaves
            fragments the lexer then reports as `malformed_shell_command` (measured: 8 spurious hits
            on a real round). Lex each LINE first -- the lexer treats a newline as plain whitespace,
            so a one-call-per-line batch would otherwise collapse into a single run-on command -- then
            split the resulting tokens on separator TOKENS, which the lexer has already distinguished
            from separator characters appearing inside quotes.
            """
            groups: list[list[str]] = []
            for raw_line in text.replace("\r", "\n").split("\n"):
                if not raw_line.strip():
                    continue
                try:
                    line_lexer = shlex.shlex(raw_line, posix=True, punctuation_chars=";&|<>()")
                    line_lexer.whitespace_split = True
                    line_lexer.commenters = ""
                    line_words = list(line_lexer)
                except ValueError:
                    hits.append({"kind": "malformed_shell_command", "line": str(line_number),
                                 "command_sha256": _sha256(command.encode("utf-8"))})
                    continue
                # A REDIRECT TARGET IS DATA, NOT A COMMAND. Splitting on `<`/`>` as if they were
                # command separators turned `broker ... > out.mlir` into TWO simple commands: a valid
                # invocation plus a bare filename. The filename tripped the mixing rule AND resolved
                # under the candidate, so it was also reported as candidate execution outside the
                # broker. Measured 2026-09-03 on perf_agentic_20260903T184101Z__trial_00: 13 such
                # lines, 26 hits, a refused trial -- for the agent doing exactly what the prompt asks,
                # capturing emitted code to diff it. Only `;`, `&&`, `||`, `|` and `&` start a new
                # command; a redirection operator consumes its target as data.
                current: list[str] = []
                skip_target = False
                for token in line_words:
                    if skip_target:
                        skip_target = False
                        continue
                    if token and set(token) <= set("<>&") and ("<" in token or ">" in token):
                        skip_target = True
                        continue
                    if token and all(character in ";&|" for character in token):
                        if current:
                            groups.append(current)
                        current = []
                    else:
                        current.append(token)
                if current:
                    groups.append(current)
            return groups

        simple_total = 0
        simple_brokered = 0

        def _audit_simple(sub_words: list[str]) -> None:
            nonlocal simple_total, simple_brokered
            simple_total += 1
            simple = " ".join(sub_words)
            invokes = (len(sub_words) >= 2 and sub_words[0] in ("python", "python3")
                       and sub_words[1] == BROKER_NAME)
            if invokes:
                # A USAGE PROBE IS NOT AN INVOCATION. `python3 <broker> --help` names no action and
                # carries no bindings, so it executes nothing; the broker refuses it as an undeclared
                # action and now records that refusal. Treating interface discovery as tool-access
                # misuse refused a whole trial on 2026-09-03 (perf_agentic_..._trial_02) for one
                # `--help` among 32 commands, while its other 21 invocations were clean.
                if len(sub_words) == 3 and sub_words[2] in ("--help", "-h", "--usage"):
                    return
                if len(sub_words) < 3:
                    hits.append({"kind": "invalid_broker_invocation", "line": str(line_number),
                                 "command_sha256": _sha256(command.encode("utf-8"))})
                    return
                action = sub_words[2]
                bindings = sub_words[3:]
                binding_names = [value.split("=", 1)[0] for value in bindings if "=" in value]
                if (action in action_names and len(binding_names) == len(bindings)
                        and len(binding_names) == len(set(binding_names)) and all(binding_names)
                        and not any(value and all(character in ";&|<>()" for character in value)
                                    for value in sub_words)
                        # Substitution is banned across the WHOLE payload, not just this simple
                        # command: `input_mlir=$(pwd)/x.mlir` must fail closed even though the lexer
                        # may have already expanded or split it away from this group.
                        and not any(token in payload_text for token in ("`", "$"))):
                    simple_brokered += 1
                    broker_invocations.append({
                        "line": line_number, "action": action,
                        "bindings_sha256": _sha256(_canonical_json(sorted(bindings))),
                    })
                else:
                    hits.append({"kind": "invalid_broker_invocation", "line": str(line_number),
                                 "command_sha256": _sha256(command.encode("utf-8"))})
                return
            # Not an invocation. Naming the broker is allowed ONLY as an argument to a read-only
            # inspection verb -- which the prompt requires, since the agent must verify the broker
            # exists before it will proceed. Every other mention (copy, rename, link, interpreter
            # -c, unknown verb) is an invalid invocation.
            if BROKER_NAME.lower() in simple.lower():
                verb = Path(sub_words[0]).name.lower() if sub_words else ""
                if verb not in _READ_ONLY_VERBS or BROKER_NAME in sub_words[:1]:
                    hits.append({"kind": "invalid_broker_invocation", "line": str(line_number),
                                 "command_sha256": _sha256(command.encode("utf-8"))})

        # Heredoc bodies are peeled off FIRST: they are interpreter source, not further commands.
        heredoc_bodies, command_text = _split_heredocs(payload_text)
        simple_groups = _simple_commands(command_text)
        for _simple in simple_groups:
            _audit_simple(_simple)
        # `brokered` suppresses the candidate-execution and target-tool checks below, so it must mean
        # EVERY simple command was a clean broker invocation -- not merely that one of them was.
        # `python3 <broker> candidate-parse ... ; ./target-opt x.mlir` has a valid invocation AND an
        # unbrokered target-tool run; treating that as brokered would switch off exactly the check that
        # catches it (pinned by test_wrapped_broker_compound_rename_and_python_exec_forms_fail_closed).
        brokered = simple_total > 0 and simple_brokered == simple_total
        # A broker invocation must stand alone. Mixing one with unbrokered work in a single shell
        # command -- `python3 <broker> candidate-parse ... ; ./target-opt x.mlir` -- is how brokered
        # and unbrokered execution get laundered into one audited line, so it is refused even though
        # the invocation half is well formed. A batch that is ENTIRELY broker calls is not mixing and
        # stays legal, which is what lets the agent issue its probe set one call per line.
        if simple_brokered and simple_brokered != simple_total:
            hits.append({"kind": "invalid_broker_invocation", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        if any(token in lowered for token in answer_tokens):
            hits.append({"kind": "answer_reconnaissance", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})

        candidate_root = candidate.resolve(strict=True)

        def candidate_path(value: str, *, must_exist: bool) -> Path | None:
            raw = Path(value)
            possible_paths = ([raw] if raw.is_absolute()
                              else [candidate.parent / raw, candidate / raw])
            for possible in possible_paths:
                try:
                    resolved = possible.resolve(strict=must_exist)
                    resolved.relative_to(candidate_root)
                except (OSError, ValueError):
                    continue
                if not must_exist or resolved.exists():
                    return resolved
            return None

        # A candidate copied to an untracked host path and executed from there would evade the
        # ordinary "path is under candidate" check.  Reject the copy-out itself, regardless of whether
        # the later segment ran successfully.
        segments: list[list[str]] = [[]]
        for value in words:
            if value and all(character in ";&|" for character in value):
                segments.append([])
            else:
                segments[-1].append(value)
        for segment in segments:
            if not segment or Path(segment[0]).name not in ("cp", "install", "mv"):
                continue
            operands = [value for value in segment[1:] if not value.startswith("-")]
            if len(operands) < 2:
                continue
            destination = operands[-1]
            if (any(candidate_path(source, must_exist=True) is not None
                    for source in operands[:-1])
                    and candidate_path(destination, must_exist=False) is None):
                hits.append({"kind": "candidate_code_copied_outside", "line": str(line_number),
                             "command_sha256": _sha256(command.encode("utf-8"))})
                break

        # WHICH SIMPLE COMMAND OWNS THE FLAG. These checks used to run against `words` -- the
        # flattened token list of the WHOLE payload -- which mis-attributes flags across command
        # boundaries. Measured 2026-09-03 on perf_stage_20260903T151801Z: the agent's own integrity
        # self-check, `python3 - <<'PY' ... PY` followed by `stat -c '%n %s %a' <control files>`, was
        # reported as `candidate_execution_outside_broker` because `"-c" in words` found STAT's flag,
        # took `'%n %s %a'` to be the Python source, failed to parse it, and took the fail-closed
        # branch. Two such lines refused an otherwise clean 54-command round in which every one of the
        # five required broker actions had been invoked. Ownership of a flag is a property of the
        # simple command it appears in, so the test has to be applied there.
        def _python_source_reads_candidate(source: str) -> bool:
            """True if this Python source opens candidate bytes -- or cannot be cleared at all."""
            try:
                tree = ast.parse(source)
            except (SyntaxError, ValueError):
                return True  # unparseable source cannot be cleared; fail closed
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "open" and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)
                        and candidate_path(node.args[0].value, must_exist=True) is not None):
                    return True
            return False

        reads_candidate_source = False
        direct_candidate = False
        for sub_words in simple_groups:
            if not sub_words:
                continue
            verb = Path(sub_words[0]).name.lower()
            if verb in ("python", "python3"):
                # `python -c` has no script argv for the structural check below, so inspect its AST
                # and reject code that opens candidate bytes, including exec(open(...).read()).
                if "-c" in sub_words:
                    code_index = sub_words.index("-c") + 1
                    if (code_index >= len(sub_words)
                            or _python_source_reads_candidate(sub_words[code_index])):
                        reads_candidate_source = True
                # `python3 - <<EOF` takes its source from stdin, where a heredoc body is the script.
                # Auditing `-c` while ignoring `<<` would leave the same execution one keystroke away.
                if any(_python_source_reads_candidate(body) for body in heredoc_bodies):
                    reads_candidate_source = True
            execution_token: str | None = None
            if verb in ("python", "python3", "bash", "sh"):
                for value in sub_words[1:]:
                    if value.startswith("-"):
                        if value in ("-c", "-lc", "-m"):
                            break
                        continue
                    execution_token = value
                    break
            else:
                execution_token = sub_words[0]
            if execution_token:
                resolved_execution = candidate_path(execution_token, must_exist=True)
                if resolved_execution is not None and resolved_execution.is_file():
                    direct_candidate = True
                if Path(execution_token).name.lower() in entry_tokens:
                    direct_candidate = True
        if reads_candidate_source:
            hits.append({"kind": "candidate_execution_outside_broker",
                         "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        if not brokered and direct_candidate and not reads_candidate_source:
            hits.append({"kind": "candidate_execution_outside_broker",
                         "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})
        if not brokered and target_tool_tokens & command_words:
            hits.append({"kind": "target_tool_outside_broker", "line": str(line_number),
                         "command_sha256": _sha256(command.encode("utf-8"))})

    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        try:
            record = json.loads(line)
        except ValueError:
            hits.append({"kind": "malformed_transcript", "line": str(line_number)})
            continue
        if not isinstance(record, dict):
            hits.append({"kind": "malformed_transcript", "line": str(line_number)})
            continue
        event_type = record.get("type")
        if event_type == "codex_unparsed":
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
        if event_type == "codex_summary":
            unknown_types = record.get("unknown_types")
            if not isinstance(unknown_types, list):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            elif unknown_types:
                hits.append({"kind": "unknown_command_event", "line": str(line_number)})
        if event_type == "thread.started":
            thread_id = record.get("thread_id")
            if not isinstance(thread_id, str) or not thread_id:
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            else:
                current_thread = thread_id

        item = record.get("item")
        if event_type in ("item.started", "item.completed"):
            if not isinstance(item, dict):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            elif item.get("type") not in known_native_items:
                hits.append({"kind": "unknown_command_event", "line": str(line_number)})
            elif item.get("type") == "command_execution":
                item_id, command = item.get("id"), item.get("command")
                if (not isinstance(item_id, str) or not item_id
                        or not isinstance(command, str) or not command.strip()):
                    hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                else:
                    key = (current_thread, item_id)
                    previous = native_commands.get(key)
                    if previous is not None and previous != command:
                        hits.append({"kind": "conflicting_command_event", "line": str(line_number)})
                    elif previous is None:
                        native_commands[key] = command
                        audit_command(command, line_number)
        elif isinstance(item, dict) and item.get("type") == "command_execution":
            hits.append({"kind": "unknown_command_event", "line": str(line_number)})

        message = record.get("message")
        if message is not None and not isinstance(message, dict):
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            continue
        content = message.get("content") if isinstance(message, dict) else []
        if content is None:
            content = []
        if not isinstance(content, list):
            hits.append({"kind": "malformed_command_event", "line": str(line_number)})
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_input = block.get("input")
            if not isinstance(tool_input, dict):
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                continue
            command = tool_input.get("command")
            if command is None:
                # File edits and other non-command tools legitimately have no command field.
                continue
            if not isinstance(command, str) or not command.strip():
                hits.append({"kind": "malformed_command_event", "line": str(line_number)})
                continue
            audit_command(command, line_number)
    if commands_seen <= 0:
        hits.append({"kind": "no_command_evidence", "line": "0"})
    return {"clean": not hits, "hits": hits, "commands_seen": commands_seen,
            "candidate": str(candidate), "broker_required": BROKER_NAME,
            "broker_invocations": broker_invocations}


def verify_broker_receipts(path: Path, actions: Sequence[BrokerAction],
                           audit: Mapping[str, Any], *,
                           candidate_sha256: str | None = None) -> dict[str, Any]:
    """Join transcript broker invocations to host-owned append-only completion receipts."""
    if path.is_symlink() or not path.is_file():
        raise StageGateError("host-owned broker receipt stream is absent or linked")
    by_name = {action.name: action for action in actions}
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise StageGateError(f"broker receipt {line_number} is malformed") from exc
        # A refused invocation is a real, host-recorded event and belongs in the join. It can never
        # satisfy a required action: `successful` below counts returncode 0 only, and a refusal is
        # required to carry a non-zero one.
        if (not isinstance(row, dict) or row.get("receipt_schema_version") != 1
                or row.get("state") not in ("complete", "rejected")
                or (row.get("state") == "rejected" and row.get("returncode") == 0)
                or row.get("action") not in by_name
                or not isinstance(row.get("index"), int) or row["index"] != len(rows)
                or not _is_sha256(row.get("argv_sha256"))
                or not _is_sha256(row.get("bindings_command_sha256"))
                or not _is_sha256(row.get("stdout_sha256"))
                or not _is_sha256(row.get("stderr_sha256"))):
            raise StageGateError(f"broker receipt {line_number} violates the action schema")
        rows.append(row)
    invocations = audit.get("broker_invocations")
    if not isinstance(invocations, list):
        raise StageGateError("transcript audit omitted exact broker invocations")
    observed = [(row["action"], row["bindings_command_sha256"]) for row in rows]
    claimed = [(row.get("action"), row.get("bindings_sha256")) for row in invocations
               if isinstance(row, Mapping)]
    if observed != claimed:
        raise StageGateError("host broker receipts do not exactly match transcript invocations")
    successful = {str(row["action"]) for row in rows if row.get("returncode") == 0}
    required = {action.name for action in actions if action.required}
    missing = sorted(required - successful)
    if missing:
        raise StageGateError(f"required broker actions lack successful receipts: {missing}")
    feedback_rows = [row for row in rows
                     if row.get("action") == DEVELOPMENT_FEEDBACK_ACTION
                     and row.get("returncode") == 0]
    if not feedback_rows:
        raise StageGateError("mandatory tuning GSIM feedback was not successfully invoked")
    feedback_receipts: list[dict[str, Any]] = []
    for row in feedback_rows:
        receipt_path, receipt_sha = row.get("feedback_receipt_path"), row.get(
            "feedback_receipt_sha256")
        if not isinstance(receipt_path, str) or not _is_sha256(receipt_sha):
            raise StageGateError("tuning GSIM feedback lacks a content-addressed host receipt")
        receipt = Path(receipt_path)
        expected_root = path.parent.resolve()
        if (receipt.is_symlink() or not receipt.is_file()
                or expected_root not in receipt.resolve().parents
                or _sha256(receipt.read_bytes()) != receipt_sha):
            raise StageGateError("tuning GSIM feedback host receipt is absent, linked, or changed")
        document = validate_redacted_feedback(json.loads(receipt.read_text(encoding="utf-8")))
        if _sha256(_canonical_json(document)) != receipt_sha:
            raise StageGateError("tuning GSIM feedback receipt is not canonical")
        feedback_receipts.append({"path": str(receipt), "sha256": receipt_sha})
    if candidate_sha256 is not None:
        if not _is_sha256(candidate_sha256):
            raise StageGateError("final round candidate digest is not a SHA-256")
        matching = [row for row in feedback_receipts
                    if validate_redacted_feedback(json.loads(
                        Path(row["path"]).read_text(encoding="utf-8"))
                    ).get("candidate_sha256") == candidate_sha256]
        if not matching:
            raise StageGateError(
                "mandatory tuning GSIM feedback did not evaluate the final round candidate bytes")
    return {"path": str(path), "sha256": _sha256_file(path), "count": len(rows),
            "successful_actions": sorted(successful), "required_actions": sorted(required),
            "feedback_successes": len(feedback_rows),
            "feedback_receipts": feedback_receipts,
            "candidate_sha256": candidate_sha256,
            "final_candidate_feedback_verified": candidate_sha256 is not None,
            "all_required_succeeded": True}


def _make_writable(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts)):
        if path.is_symlink():
            raise StageGateError(f"candidate copy contains a symlink: {path}")
        path.chmod(0o755 if path.is_dir() else (path.stat().st_mode | stat.S_IWUSR))
    root.chmod(0o755)


def fresh_round_workspace(source_submission: Path, workspace: Path,
                          expected_sha256: str) -> Path:
    """Create a new round workspace from exactly the previous candidate bytes."""
    raw_workspace = Path(workspace)
    if raw_workspace.exists() or raw_workspace.is_symlink():
        raise StageGateError(f"round workspace is not fresh: {raw_workspace}")
    workspace = raw_workspace.resolve()
    submission = workspace / "submission"
    workspace.mkdir(parents=True)
    shutil.copytree(source_submission, submission, symlinks=False)
    _make_writable(submission)
    observed = hash_tree(submission)["sha256"]
    if observed != expected_sha256:
        raise StageGateError(
            f"fresh round candidate digest {observed} does not match its input {expected_sha256}")
    return submission


def _validate_formal_claim_facts(
        formal: object, replicates: object, formal_replicate_identities: object,
        smoke_replicates: object, cells: object, families: object) -> None:
    if (not isinstance(formal, Mapping) or formal.get("schema_version") != 1
            or formal.get("family") != "PK" or formal.get("claim") != "PREDICTS"
            or formal.get("status") != "READY" or formal.get("refusal_reasons") != []):
        raise StageGateError("performance candidate omits a READY frozen PK formal claim")
    declaration = formal.get("declaration")
    evidence = declaration.get("evidence") if isinstance(declaration, Mapping) else None
    timing_simulator = (evidence.get("timing_simulator")
                        if isinstance(evidence, Mapping) else None)
    try:
        supported_acceptance = PK.supported_acceptance(str(timing_simulator))
    except ValueError as exc:
        raise StageGateError(
            "performance candidate selects an unsupported PK timing engine") from exc
    if _canonical_json(declaration) != _canonical_json(supported_acceptance):
        raise StageGateError("performance candidate PK acceptance contract drifted")
    assert isinstance(declaration, Mapping)
    replicate_contract = declaration.get("replicates")
    if not isinstance(replicate_contract, Mapping):
        raise StageGateError("performance candidate PK acceptance omits replicates")
    identities = replicate_contract.get("identities")
    exact_count = replicate_contract.get("exact_count")
    if (not isinstance(identities, list) or formal_replicate_identities != identities
            or isinstance(exact_count, bool) or not isinstance(exact_count, int)
            or replicates != exact_count or exact_count != len(identities)):
        raise StageGateError("performance candidate formal replicates drift from PK acceptance")
    if (isinstance(smoke_replicates, bool) or not isinstance(smoke_replicates, int)
            or smoke_replicates <= 0 or smoke_replicates >= exact_count):
        raise StageGateError("performance candidate smoke replicas could masquerade as formal evidence")
    if not isinstance(families, list):
        raise StageGateError("performance candidate formal families are malformed")
    pk_families = [row for row in families
                   if isinstance(row, Mapping) and row.get("family") == "PK"]
    if (len(pk_families) != 1
            or _canonical_json(pk_families[0].get("acceptance")) != _canonical_json(declaration)):
        raise StageGateError("performance candidate family omits its exact PK acceptance")
    cohort = formal.get("cohort")
    expected = formal.get("expected_identities")
    if (not isinstance(cohort, Mapping) or cohort.get("replicates") != identities
            or not isinstance(expected, list) or not expected):
        raise StageGateError("performance candidate PK preflight omits its exact cohort")
    expected_cells: list[dict[str, str]] = []
    for row in expected:
        if not isinstance(row, Mapping):
            raise StageGateError("performance candidate PK preflight has a malformed identity")
        simulator, tier = row.get("simulator"), row.get("tier")
        if ((simulator, tier) not in (("spike", "L2"), (timing_simulator, "L3"))
                or row.get("family") != "PK"):
            raise StageGateError("performance candidate PK preflight changes L2/L3 semantics")
        expected_cells.append({key: str(row.get(key))
                               for key in ("family", "capsule", "simulator", "replicate")})
    if len({tuple(row.items()) for row in expected_cells}) != len(expected_cells):
        raise StageGateError("performance candidate PK preflight repeats a formal identity")
    if not isinstance(cells, list):
        raise StageGateError("performance candidate formal cells are malformed")
    recorded_pk = [row for row in cells
                   if isinstance(row, Mapping) and row.get("family") == "PK"]
    if sorted(expected_cells, key=_canonical_json) != sorted(
            (dict(row) for row in recorded_pk), key=_canonical_json):
        raise StageGateError("performance candidate PK formal identities drift from expected cells")


def validate_candidate_record(document: Mapping[str, Any], *, require_consumable: bool = True) -> dict:
    """Pure schema/boundary validator intended for ``run_perf_bench`` integration."""
    if not isinstance(document, Mapping) or document.get("schema_version") != SCHEMA_VERSION:
        raise StageGateError("performance candidate record has an unsupported schema")
    if document.get("kind") != "arm4_performance_candidate":
        raise StageGateError("performance candidate record has a foreign kind")
    target = document.get("target")
    base = document.get("base_functional")
    candidate = document.get("candidate")
    prompt = document.get("prompt")
    corpus = document.get("performance_corpus")
    sandbox = document.get("sandbox")
    broker = document.get("broker")
    development_feedback = document.get("development_feedback")
    agent = document.get("agent")
    telemetry = document.get("telemetry")
    admission = document.get("admission")
    if not all(isinstance(value, Mapping)
               for value in (target, base, candidate, prompt, corpus, sandbox, broker,
                             development_feedback, agent, telemetry,
                             admission)):
        raise StageGateError("performance candidate record omits a required evidence mapping")
    for label, value in (
            ("functional submission", base.get("submission_sha256")),
            ("candidate initial", candidate.get("initial_sha256")),
            ("candidate final", candidate.get("sha256")),
            ("prompt", prompt.get("sha256")),
            ("prompt facts", prompt.get("facts_sha256")),
            ("prompt renderer", prompt.get("renderer_sha256")),
            ("performance manifest", corpus.get("manifest_sha256")),
            ("performance corpus", corpus.get("capsules_sha256")),
            ("agent-input manifest", corpus.get("agent_input_manifest_sha256")),
            ("agent-input content", corpus.get("agent_input_sha256")),
            ("target descriptor", target.get("descriptor_sha256")),
            ("Codex binary", agent.get("codex_binary_sha256")),
            ("transcript", agent.get("transcript_sha256")),
            ("broker registry", broker.get("registry_sha256")),
            ("broker receipt manifest", broker.get("receipt_manifest_sha256")),
            ("development GSIM certificate",
             (development_feedback.get("certificate") or {}).get("sha256"))):
        if not _is_sha256(value):
            raise StageGateError(f"performance candidate record has no valid {label} SHA-256")
    if candidate.get("initial_sha256") != base.get("submission_sha256"):
        raise StageGateError("performance candidate was not forked byte-for-byte from functional")
    if (not isinstance(target.get("name"), str) or not target.get("name")
            or not isinstance(target.get("descriptor"), str) or not target.get("descriptor")):
        raise StageGateError("performance candidate record omits its target descriptor identity")
    _safe_component(str(base.get("run_id") or ""), label="base functional run id")
    bundle_snapshot = base.get("bundle_input_snapshot")
    host_lane = base.get("model_host_lane")
    sentinel = base.get("e2e_sentinel")
    if (not isinstance(bundle_snapshot, Mapping) or not _is_sha256(
            bundle_snapshot.get("content_sha256")) or not _is_sha256(
                bundle_snapshot.get("manifest_sha256"))
            or not isinstance(bundle_snapshot.get("grants"), list)
            or not bundle_snapshot["grants"]):
        raise StageGateError("performance candidate omits the frozen functional grant snapshot")
    if (not isinstance(host_lane, Mapping) or not _is_sha256(host_lane.get("package_sha256"))
            or not isinstance(host_lane.get("integration_seam"), str)):
        raise StageGateError("performance candidate omits its exact functional host lane")
    if (not isinstance(sentinel, Mapping) or not _is_sha256(sentinel.get("capsule_sha256"))
            or set(sentinel.get("required_lanes") or []) != {"on_mesh", "scalar_rvv_lane"}
            or "L3" not in (sentinel.get("required_tiers") or [])):
        raise StageGateError("performance candidate omits its frozen full-model E2E sentinel")
    for evidence, label in ((candidate.get("path"), "sealed candidate path"),
                            (prompt.get("staged_path"), "staged prompt path"),
                            (corpus.get("path"), "frozen performance corpus path"),
                            (agent.get("codex_binary"), "Codex binary path"),
                            (agent.get("transcript"), "combined transcript path")):
        if not isinstance(evidence, str) or not evidence:
            raise StageGateError(f"performance candidate record omits the {label}")
    for field in ("agent_input_files", "agent_input_bytes"):
        value = corpus.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise StageGateError(f"performance candidate record has no non-vacuous {field}")
    replicates = corpus.get("replicates")
    cells, families = corpus.get("expected_cells"), corpus.get("families")
    formal_claim = corpus.get("formal_claim")
    formal_replicate_identities = corpus.get("formal_replicate_identities")
    smoke_replicates = corpus.get("smoke_replicates")
    prompt_facts = prompt.get("facts")
    if (isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0
            or not isinstance(cells, list) or not cells or not isinstance(families, list)
            or not families or not isinstance(prompt_facts, Mapping)
            or prompt_facts.get("expected_cells") != cells
            or prompt_facts.get("families") != families
            or prompt_facts.get("replicates") != replicates
            or prompt_facts.get("formal_claim") != formal_claim
            or prompt_facts.get("formal_replicate_identities") != formal_replicate_identities
            or prompt_facts.get("smoke_replicates") != smoke_replicates):
        raise StageGateError("performance candidate omits exact prompt cells/families/replicates")
    _validate_formal_claim_facts(
        formal_claim, replicates, formal_replicate_identities, smoke_replicates, cells, families)
    if (candidate.get("read_only") is not True
            or candidate.get("base_submission_overwritten") is not False):
        raise StageGateError("performance candidate record does not prove a separate read-only snapshot")
    delta = candidate.get("delta")
    if (not isinstance(delta, Mapping) or not isinstance(delta.get("changed_files"), list)
            or not isinstance(delta.get("execution_relevant_changed_files"), list)
            or delta.get("changed_file_count") != len(delta["changed_files"])
            or delta.get("execution_relevant_changed_file_count") != len(
                delta["execution_relevant_changed_files"])):
        raise StageGateError("performance candidate record has no structured candidate delta")
    outer = sandbox.get("outer_codex_control_plane")
    inner = sandbox.get("inner_execution_plane")
    if not isinstance(outer, Mapping) or not isinstance(inner, Mapping):
        raise StageGateError("performance candidate record collapses its two sandbox boundaries")
    network_record = "available_not_an_isolation_claim"
    if (outer.get("network") != network_record or outer.get("clear_environment") is not True
            or outer.get("auth_exception") != "isolated_codex_home_explicit_auth_mount"
            or outer.get("session_history_mounted") is not False
            or outer.get("live_target_toolchain_mounted") is not False
            or outer.get("frozen_functional_grants_mounted") is not True
            or outer.get("frozen_grant_manifest_sha256") != bundle_snapshot.get("manifest_sha256")
            or outer.get("answer_surface_gap") != []):
        raise StageGateError("outer Codex control-plane evidence is incomplete")
    if not isinstance(outer.get("bwrap_binary"), str) or not outer.get("bwrap_binary"):
        raise StageGateError("outer Codex control-plane evidence omits the bwrap binary path")
    for evidence, label in ((outer.get("bwrap_binary_sha256"), "bwrap binary"),
                            (outer.get("policy_sha256"), "outer bwrap policy"),
                            (inner.get("policy_sha256"), "inner bwrap policy")):
        if not _is_sha256(evidence):
            raise StageGateError(f"performance candidate record omits the {label} digest")
    if (inner.get("network") != network_record or inner.get("clear_environment") is not True
            or inner.get("credentials") != "none" or inner.get("answer_surface_gap") != []
            or inner.get("candidate_writable") is not True
            or inner.get("corpus_read_only") is not True
            or inner.get("frozen_functional_grants_mounted") is not True
            or inner.get("frozen_grant_manifest_sha256") != bundle_snapshot.get(
                "manifest_sha256")):
        raise StageGateError("inner execution-plane evidence is not credential-free and answer-masked")
    probes = inner.get("tool_probe_results")
    probe_recheck = inner.get("tool_probe_recheck_results")
    if (not isinstance(probes, list) or not probes
            or any(not isinstance(row, Mapping) or row.get("returncode") != 0
                   or not isinstance(row.get("label"), str) or not row.get("label")
                   or not isinstance(row.get("command"), str) or not row.get("command")
                   for row in probes)):
        raise StageGateError("performance candidate record lacks passing inner tool probes")
    if probe_recheck != probes:
        raise StageGateError("performance candidate tool evidence changed between preflight and recheck")
    registry, receipt_rows = broker.get("registry"), broker.get("round_receipts")
    if (not isinstance(registry, list) or not registry
            or _sha256(_canonical_json(registry)) != broker.get("registry_sha256")
            or not isinstance(receipt_rows, list)
            or any(not isinstance(row, Mapping) or not isinstance(row.get("path"), str)
                   or not _is_sha256(row.get("sha256"))
                   or row.get("all_required_succeeded") is not True
                   or not _is_sha256(row.get("candidate_sha256"))
                   or row.get("final_candidate_feedback_verified") is not True
                   or not isinstance(row.get("feedback_successes"), int)
                   or row.get("feedback_successes") < 1
                   or not isinstance(row.get("feedback_receipts"), list)
                   or not row.get("feedback_receipts") for row in receipt_rows)
            or broker.get("control_owned_by_harness") is not True
            or broker.get("control_writable_by_agent") is not False
            or not isinstance(broker.get("receipt_manifest"), str)
            or not broker.get("receipt_manifest")):
        raise StageGateError("performance candidate lacks immutable broker registry/receipts")
    required_actions = sorted(str(row.get("name")) for row in registry
                              if isinstance(row, Mapping) and row.get("required") is True)
    if not required_actions or broker.get("required_actions") != required_actions:
        raise StageGateError("performance candidate broker required-action contract is incomplete")
    feedback_actions = [row for row in registry if isinstance(row, Mapping)
                        and row.get("name") == DEVELOPMENT_FEEDBACK_ACTION]
    if (len(feedback_actions) != 1 or feedback_actions[0].get("required") is not True
            or feedback_actions[0].get("placeholders") != []
            or feedback_actions[0].get("argv_template") != [_HOST_FEEDBACK_SENTINEL]
            or DEVELOPMENT_FEEDBACK_ACTION not in required_actions):
        raise StageGateError("mandatory tuning GSIM feedback action is absent or drifted")
    certificate = development_feedback.get("certificate")
    rtl_identity = development_feedback.get("rtl_identity")
    recorded_feedback_receipts = development_feedback.get("round_receipts")
    if (development_feedback.get("action") != DEVELOPMENT_FEEDBACK_ACTION
            or development_feedback.get("required_per_round") is not True
            or development_feedback.get("scope") != "frozen_tuning_corpus_only"
            or development_feedback.get("engine") != "gsim"
            or development_feedback.get("redaction")
            != "correctness_gsim_cycles_and_paired_deltas_only"
            or not isinstance(certificate, Mapping)
            or certificate.get("target") != target.get("name")
            or certificate.get("fidelity") != GATE.FIDELITY
            or not isinstance(rtl_identity, Mapping)
            or not isinstance(recorded_feedback_receipts, list)
            or recorded_feedback_receipts != [row.get("feedback_receipts") for row in receipt_rows]):
        raise StageGateError("development GSIM feedback evidence is incomplete or drifted")
    budget_facts = prompt_facts.get("budgets")
    expected_budget_facts = {
        "wall_budget_seconds": agent.get("wall_budget_seconds"),
        "rounds": agent.get("rounds_requested"),
        "round_timeout_seconds": agent.get("round_timeout_seconds"),
        "max_tool_calls": agent.get("max_tool_calls"),
        "tool_timeout_seconds": agent.get("tool_timeout_seconds"),
    }
    prompt_host = prompt_facts.get("host_lane")
    host_fact_keys = {"target", "package_id", "package_path", "package_sha256",
                      "manifest_path", "integration_seam"}
    if (budget_facts != expected_budget_facts or prompt_facts.get("tools") != registry
            or prompt_facts.get("e2e_sentinel") != sentinel
            or not isinstance(prompt_host, Mapping)
            or set(prompt_host) != host_fact_keys
            or any(prompt_host.get(key) != host_lane.get(key) for key in host_fact_keys)
            or prompt_facts.get("mount_destinations") != outer.get("mount_destinations")):
        raise StageGateError("canonical prompt facts drift from recorded launch enforcement")
    round_rows = agent.get("rounds")
    rounds_requested = agent.get("rounds_requested")
    bounded_agent_values = (
        agent.get("wall_budget_seconds"), agent.get("rounds_requested"),
        agent.get("round_timeout_seconds"), agent.get("max_tool_calls"),
        agent.get("tool_timeout_seconds"))
    if (agent.get("driver") != "codex"
            or any(not isinstance(agent.get(field), str) or not agent.get(field)
                   for field in ("model", "resolved_model", "effort"))
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in bounded_agent_values)
            or not isinstance(round_rows, list) or not round_rows
            or isinstance(rounds_requested, bool) or not isinstance(rounds_requested, int)
            or rounds_requested < len(round_rows)
            or any(not isinstance(row, Mapping) or isinstance(row.get("agent_exit_code"), bool)
                   or not isinstance(row.get("agent_exit_code"), int)
                   or not isinstance(row.get("transcript"), str) or not row.get("transcript")
                   or not isinstance(row.get("audit"), Mapping)
                   or not isinstance(row["audit"].get("clean"), bool)
                   or not isinstance(row["audit"].get("hits"), list)
                   or row["audit"].get("broker_required") != BROKER_NAME
                   or not isinstance(row["audit"].get("broker_invocations"), list)
                   or not isinstance(row.get("telemetry"), Mapping)
                   or row["telemetry"].get("event_count", 0) <= 0
                   or (row["telemetry"].get("summary") or {}).get("usage_complete") is not True
                   or (row["telemetry"].get("accounting") or {}).get("available") is not True
                   or (row["telemetry"].get("accounting") or {}).get("usage_complete") is not True
                   or isinstance(row["audit"].get("commands_seen"), bool)
                   or not isinstance(row["audit"].get("commands_seen"), int)
                   or not _is_sha256(row.get("transcript_sha256")) for row in round_rows)):
        raise StageGateError("performance candidate was not produced by the bounded Codex driver")
    telemetry_artifacts = telemetry.get("artifacts")
    telemetry_accounting = telemetry.get("accounting")
    telemetry_reconciliation = telemetry.get("aet_reconciliation")
    activity_share = telemetry.get("activity_share")
    activity_seconds = ((activity_share or {}).get("seconds_by_category") or {})
    activity_shares = ((activity_share or {}).get("share_by_category") or {})
    activity_wall = ((activity_share or {}).get("trajectory_wall_seconds"))
    activity_occupancy = ((activity_share or {}).get("classified_span_occupancy_ratio"))
    activity_semantics_valid = (
        isinstance(activity_seconds, Mapping) and bool(activity_seconds)
        and isinstance(activity_shares, Mapping) and bool(activity_shares)
        and set(activity_seconds) == set(activity_shares)
        and all(isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0
                for value in (*activity_seconds.values(), *activity_shares.values()))
        and math.isclose(sum(float(value) for value in activity_shares.values()), 1.0,
                         rel_tol=1e-6, abs_tol=1e-6)
        and isinstance((activity_share or {}).get("classified_seconds"), (int, float))
        and not isinstance((activity_share or {}).get("classified_seconds"), bool)
        and float(activity_share["classified_seconds"]) > 0
        and math.isclose(sum(float(value) for value in activity_seconds.values()),
                         float(activity_share["classified_seconds"]),
                         rel_tol=1e-6, abs_tol=1e-6)
        and activity_share.get("schema_version") == 2
        and activity_share.get("denominator")
        == "sum_of_classified_tool_span_seconds_including_overlap"
        and activity_share.get("is_wall_time_partition") is False
        and activity_share.get("overlapping_tool_spans_allowed") is True
        and activity_share.get("occupancy_ratio_may_exceed_one") is True
        and activity_share.get("subagent_tool_calls_tracked") is False
        and isinstance(activity_wall, (int, float)) and not isinstance(activity_wall, bool)
        and float(activity_wall) > 0
        and isinstance(activity_occupancy, (int, float))
        and not isinstance(activity_occupancy, bool) and float(activity_occupancy) > 0
        and math.isclose(float(activity_occupancy),
                         float(activity_share["classified_seconds"]) / float(activity_wall),
                         rel_tol=1e-6, abs_tol=1e-6))
    required_telemetry_artifacts = {
        "combined_raw", "trajectory", "reconciliation", "token_ledger", "tool_ledger",
        "cost_time_toolcalls", "activity_share", "preflight", "aet_metrics_log"}
    if (telemetry.get("required") is not True or telemetry.get("driver") != "codex"
            or telemetry.get("billing_mode") != "subscription_notional"
            or telemetry.get("rounds_with_complete_usage") != len(round_rows)
            or not isinstance(telemetry.get("raw_event_count"), int)
            or telemetry.get("raw_event_count", 0) <= 0
            or not isinstance(telemetry.get("tool_call_count"), int)
            or telemetry.get("tool_call_count", 0) <= 0
            or telemetry.get("subagent_tool_calls_tracked") is not False
            or not _is_sha256(telemetry.get("preflight_sha256"))
            or not isinstance(telemetry_accounting, Mapping)
            or telemetry_accounting.get("available") is not True
            or telemetry_accounting.get("usage_complete") is not True
            or telemetry_accounting.get("billing_mode") != "subscription_notional"
            or telemetry_accounting.get("estimated_cost_usd") is not None
            or not isinstance(telemetry_accounting.get("subscription_notional_usd"), (int, float))
            or telemetry_accounting.get("subscription_notional_usd", 0) <= 0
            or not isinstance(telemetry_accounting.get("tokens_total"), int)
            or telemetry_accounting.get("tokens_total", 0) <= 0
            or telemetry_accounting.get("tool_calls") != telemetry.get("tool_call_count")
            or not isinstance(telemetry_reconciliation, Mapping)
            or telemetry_reconciliation.get("ok") is not True
            or (telemetry_reconciliation.get("raw_events") or {}).get("reconciled") is not True
            or (telemetry_reconciliation.get("token_ledger") or {}).get("all_match") is not True
            or not isinstance(activity_share, Mapping)
            or activity_share.get("basis") != "aet_native_codex_structured_tool_spans"
            or not activity_semantics_valid
            or not isinstance(telemetry_artifacts, Mapping)
            or set(telemetry_artifacts) != required_telemetry_artifacts
            or any(not isinstance(value, Mapping) or not isinstance(value.get("path"), str)
                   or not value.get("path") or not _is_sha256(value.get("sha256"))
                   for value in telemetry_artifacts.values())):
        raise StageGateError("performance candidate lacks complete raw/AET/cost/activity telemetry")
    if (len(receipt_rows) != len(round_rows)
            or any(receipt.get("candidate_sha256") != round_row.get("candidate_sha256")
                   for receipt, round_row in zip(receipt_rows, round_rows))):
        raise StageGateError(
            "per-round tuning GSIM feedback is not bound to the authored candidate bytes")
    audit = agent.get("audit")
    if (not isinstance(audit, Mapping) or not isinstance(audit.get("clean"), bool)
            or not isinstance(audit.get("hits"), list)
            or audit.get("broker_required") != BROKER_NAME
            or not isinstance(audit.get("broker_invocations"), list)
            or isinstance(audit.get("commands_seen"), bool)
            or not isinstance(audit.get("commands_seen"), int)):
        raise StageGateError("performance candidate lacks structured agent audit evidence")
    if (admission.get("evaluation_performed_by_stage") is not False
            or admission.get("development_feedback_performed_by_stage") is not True
            or admission.get("success_declared_by_stage") is not False
            or admission.get("consumer") != "run_perf_bench.py"):
        raise StageGateError("performance authoring stage crossed the evaluation boundary")
    if require_consumable:
        if (admission.get("consumable") is not True or document.get("state") != "sealed"
                or rounds_requested != len(round_rows) or audit.get("clean") is not True
                or audit.get("hits") != []
                or audit.get("commands_seen", 0) <= 0
                or any(row["audit"].get("commands_seen", 0) <= 0
                       or row["audit"].get("clean") is not True for row in round_rows)
                or delta.get("execution_relevant_changed_file_count", 0) <= 0
                or broker.get("all_required_succeeded") is not True
                or any(row.get("feedback_successes", 0) < 1 for row in receipt_rows)
                or len(receipt_rows) != rounds_requested
                or any(row.get("agent_exit_code") != 0 for row in round_rows)):
            raise StageGateError(f"performance candidate is not consumable: {admission.get('refusal')}")
    return dict(document)


def verify_candidate_record(path: Path, *, require_consumable: bool = True,
                            verify_authoring_tools: bool = False,
                            target_experiment: TargetExperiment | None = None) -> dict:
    """Re-hash immutable run artifacts; optionally require the live authoring tools to remain pinned."""
    raw_path = Path(path)
    if raw_path.is_symlink() or not raw_path.is_file():
        raise StageGateError(f"performance candidate record is absent or linked: {raw_path}")
    path = raw_path.resolve()
    document = validate_candidate_record(json.loads(path.read_text(encoding="utf-8")),
                                         require_consumable=require_consumable)
    if verify_authoring_tools:
        for binary_path, expected, label in (
                (document["target"]["descriptor"], document["target"]["descriptor_sha256"],
                 "target descriptor"),
                (document["agent"]["codex_binary"], document["agent"]["codex_binary_sha256"],
                 "Codex binary"),
                (document["sandbox"]["outer_codex_control_plane"]["bwrap_binary"],
                 document["sandbox"]["outer_codex_control_plane"]["bwrap_binary_sha256"],
                 "bwrap binary")):
            binary = Path(binary_path)
            if binary.is_symlink() or not binary.is_file() or _sha256_file(binary) != expected:
                raise StageGateError(f"{label} bytes do not match the performance candidate record")
    candidate = Path(document["candidate"]["path"])
    if candidate.is_symlink() or not candidate.is_dir():
        raise StageGateError("sealed performance candidate path is absent or linked")
    if hash_tree(candidate)["sha256"] != document["candidate"]["sha256"]:
        raise StageGateError("sealed performance candidate bytes do not match their record")
    for member in (candidate, *candidate.rglob("*")):
        if member.is_symlink():
            raise StageGateError(f"sealed performance candidate contains a symlink: {member}")
        if member.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise StageGateError(f"sealed performance candidate is writable: {member}")
    transcript = Path(document["agent"]["transcript"])
    if transcript.is_symlink() or not transcript.is_file():
        raise StageGateError("performance candidate transcript is absent or linked")
    if _sha256(transcript.read_bytes()) != document["agent"]["transcript_sha256"]:
        raise StageGateError("performance candidate transcript bytes do not match their record")
    if target_experiment is not None:
        recorded_actions = actions_from_registry_contract(
            document["broker"]["registry"], candidate)
        observed_audit = audit_codex_transcript(
            transcript, target_experiment, candidate, recorded_actions)
        recorded_audit = document["agent"]["audit"]
        for field in ("clean", "hits", "commands_seen", "broker_required"):
            if observed_audit.get(field) != recorded_audit.get(field):
                raise StageGateError(
                    f"combined Codex transcript audit field {field!r} does not match its record")
    for row in document["agent"]["rounds"]:
        round_transcript = Path(str(row.get("transcript") or ""))
        if (round_transcript.is_symlink() or not round_transcript.is_file()
                or _sha256(round_transcript.read_bytes()) != row["transcript_sha256"]):
            raise StageGateError(
                f"performance candidate round {row.get('round')} transcript bytes changed")
        for label, artifact in row["telemetry"]["artifacts"].items():
            evidence_path = Path(str(artifact.get("path") or ""))
            if (evidence_path.is_symlink() or not evidence_path.is_file()
                    or _sha256_file(evidence_path) != artifact.get("sha256")):
                raise StageGateError(
                    f"performance candidate round {row.get('round')} {label} telemetry changed")
    for label, artifact in document["telemetry"]["artifacts"].items():
        evidence_path = Path(str(artifact.get("path") or ""))
        if (evidence_path.is_symlink() or not evidence_path.is_file()
                or _sha256_file(evidence_path) != artifact.get("sha256")):
            raise StageGateError(f"performance candidate {label} telemetry changed")
    preflight_doc = json.loads(Path(document["telemetry"]["artifacts"]["preflight"]["path"])
                               .read_text(encoding="utf-8"))
    if _sha256(_canonical_json(preflight_doc)) != document["telemetry"]["preflight_sha256"]:
        raise StageGateError("performance candidate telemetry preflight declaration changed")
    staged_prompt = Path(document["prompt"]["staged_path"])
    if (staged_prompt.is_symlink() or not staged_prompt.is_file()
            or _sha256(staged_prompt.read_bytes()) != document["prompt"]["sha256"]):
        raise StageGateError("staged performance prompt bytes do not match their record")
    if (_sha256(_canonical_json(document["prompt"]["facts"]))
            != document["prompt"]["facts_sha256"]):
        raise StageGateError("canonical performance prompt facts changed")
    renderer = Path(document["prompt"]["renderer_path"])
    if (renderer.is_symlink() or not renderer.is_file()
            or _sha256_file(renderer) != document["prompt"]["renderer_sha256"]):
        raise StageGateError("performance prompt renderer bytes changed")
    corpus = document["performance_corpus"]
    frozen_manifest = Path(corpus["manifest"])
    if (frozen_manifest.is_symlink() or not frozen_manifest.is_file()
            or _sha256(frozen_manifest.read_bytes()) != corpus["manifest_sha256"]):
        raise StageGateError("frozen performance manifest bytes do not match their record")
    frozen_root = Path(corpus["path"])
    try:
        frozen_loaded = load_frozen_performance_corpus(
            frozen_root, manifest_sha256=str(corpus["manifest_sha256"]),
            capsules_sha256=str(corpus["capsules_sha256"]),
            expected_target=str(document["target"]["name"]))
    except PC.CampaignGateError as exc:
        raise StageGateError(f"frozen performance corpus verification failed: {exc}") from exc
    observed_formal_claim = prepare_formal_pk_claim(
        frozen_loaded.capsules, int(corpus["replicates"]))
    if _canonical_json(observed_formal_claim) != _canonical_json(corpus["formal_claim"]):
        raise StageGateError("frozen performance descriptors changed their formal PK preflight")
    agent_manifest = Path(corpus["agent_input_manifest"])
    if (agent_manifest.is_symlink() or not agent_manifest.is_file()
            or _sha256(agent_manifest.read_bytes()) != corpus["agent_input_manifest_sha256"]):
        raise StageGateError("answer-free agent input manifest bytes do not match their record")
    verify_answer_free_agent_inputs(AgentInputSnapshot(
        Path(corpus["agent_input_path"]), agent_manifest,
        str(corpus["agent_input_manifest_sha256"]), str(corpus["agent_input_sha256"]),
        int(corpus["agent_input_files"]), int(corpus["agent_input_bytes"])))
    base = Path(document["base_functional"]["snapshot"])
    if base.is_symlink() or not base.is_dir() or hash_tree(base)["sha256"] != document[
            "base_functional"]["submission_sha256"]:
        raise StageGateError("frozen functional base bytes do not match their record")
    bundle = document["base_functional"]["bundle_input_snapshot"]
    bundle_marker = Path(bundle["manifest"])
    if (bundle_marker.is_symlink() or not bundle_marker.is_file()
            or _sha256_file(bundle_marker) != bundle["manifest_sha256"]):
        raise StageGateError("frozen functional grant marker bytes changed")
    verify_functional_host_lane_snapshot(document["base_functional"]["model_host_lane"])
    sentinel = document["base_functional"]["e2e_sentinel"]
    sentinel_source = Path(sentinel["frozen_source_path"])
    if (sentinel_source.is_symlink() or not sentinel_source.is_dir()
            or _exact_tree_record(sentinel_source)["sha256"] != sentinel["capsule_sha256"]):
        raise StageGateError("frozen full-model E2E sentinel bytes changed")
    receipt_manifest = Path(document["broker"]["receipt_manifest"])
    if (receipt_manifest.is_symlink() or not receipt_manifest.is_file()
            or _sha256_file(receipt_manifest) != document["broker"]["receipt_manifest_sha256"]):
        raise StageGateError("broker receipt manifest bytes changed")
    receipt_document = json.loads(receipt_manifest.read_text(encoding="utf-8"))
    if (receipt_document.get("schema_version") != 1
            or receipt_document.get("rounds") != document["broker"]["round_receipts"]):
        raise StageGateError("broker receipt manifest disagrees with the candidate record")
    for round_index, row in enumerate(document["broker"]["round_receipts"]):
        receipt_path = Path(str(row.get("path") or ""))
        if (receipt_path.is_symlink() or not receipt_path.is_file()
                or not _is_sha256(row.get("sha256"))
                or _sha256_file(receipt_path) != row["sha256"]
                or row.get("all_required_succeeded") is not True):
            raise StageGateError("host-owned per-round broker receipt bytes changed")
        feedback_candidates: list[str] = []
        for feedback in row.get("feedback_receipts") or []:
            feedback_path = Path(str(feedback.get("path") or ""))
            feedback_sha = feedback.get("sha256")
            if (feedback_path.is_symlink() or not feedback_path.is_file()
                    or not _is_sha256(feedback_sha)
                    or _sha256(feedback_path.read_bytes()) != feedback_sha):
                raise StageGateError("host-owned tuning GSIM feedback receipt bytes changed")
            validated = validate_redacted_feedback(json.loads(
                feedback_path.read_text(encoding="utf-8")))
            if _sha256(_canonical_json(validated)) != feedback_sha:
                raise StageGateError("host-owned tuning GSIM feedback receipt is not canonical")
            feedback_candidates.append(str(validated["candidate_sha256"]))
        expected_candidate = document["agent"]["rounds"][round_index]["candidate_sha256"]
        if expected_candidate not in feedback_candidates:
            raise StageGateError(
                "host-owned tuning GSIM feedback did not evaluate the recorded round candidate")
    feedback_certificate = document["development_feedback"]["certificate"]
    try:
        GATE.load_certificate(
            feedback_certificate["path"], expected_sha256=feedback_certificate["sha256"])
    except GATE.GsimGateError as exc:
        raise StageGateError(f"development GSIM certificate bytes changed: {exc}") from exc
    return document


def verify_candidate_handoff(
        path: Path, *, verify_authoring_tools: bool = False,
        target_experiment: TargetExperiment | None = None) -> VerifiedCandidateHandoff:
    """Return the only normalized stage-to-measurement API after full byte verification."""
    document = verify_candidate_record(
        path, require_consumable=True, verify_authoring_tools=verify_authoring_tools,
        target_experiment=target_experiment)
    base, candidate = document["base_functional"], document["candidate"]
    corpus, prompt = document["performance_corpus"], document["prompt"]
    broker, agent = document["broker"], document["agent"]
    bundle = base["bundle_input_snapshot"]
    preflight_path = Path(document["telemetry"]["artifacts"]["preflight"]["path"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    sources = preflight.get("sources") or {}
    if (set(sources) != TELEMETRY_TREATMENT_SOURCES
            or any(not isinstance(source, Mapping)
                   or not _is_sha256(source.get("sha256"))
                   for source in sources.values())):
        raise StageGateError("performance telemetry treatment source identity is incomplete")
    source_sha256 = {str(name): str(source["sha256"])
                     for name, source in sorted(sources.items())}
    model_resolution = preflight.get("model_resolution") or {}
    if (source_sha256["codex_binary"] != agent["codex_binary_sha256"]
            or source_sha256["performance_authoring_stage"]
            != document["prompt"]["renderer_sha256"]
            or model_resolution.get("requested_model") != agent.get("model")
            or model_resolution.get("resolved_model") != agent.get("resolved_model")
            or not isinstance(model_resolution.get("codex_model_map"), str)):
        raise StageGateError(
            "performance candidate executable/source identities differ from telemetry preflight")
    treatment_identity = {
        "telemetry_preflight_sha256": str(document["telemetry"]["preflight_sha256"]),
        "codex_binary_sha256": str(agent["codex_binary_sha256"]),
        "authoring_stage_sha256": str(document["prompt"]["renderer_sha256"]),
        "telemetry_source_sha256": source_sha256,
        "requested_model": str(model_resolution["requested_model"]),
        "resolved_model": str(model_resolution["resolved_model"]),
        "codex_model_map": str(model_resolution.get("codex_model_map", "")),
    }
    agent_contract = {
        "model": str(agent.get("model") or ""),
        "resolved_model": str(agent.get("resolved_model") or ""),
        "effort": str(agent.get("effort") or ""),
        "wall_budget_seconds": agent.get("wall_budget_seconds"),
        "rounds": agent.get("rounds_requested"),
        "round_timeout_seconds": agent.get("round_timeout_seconds"),
        "max_tool_calls": agent.get("max_tool_calls"),
        "tool_timeout_seconds": agent.get("tool_timeout_seconds"),
        "smoke_replicates": corpus.get("smoke_replicates"),
        "measurement_replicates": corpus.get("replicates"),
        "functional_run_id": str(base.get("run_id") or ""),
        "functional_submission_sha256": str(base.get("submission_sha256") or ""),
        "telemetry_required": True,
        "telemetry_preflight_sha256": str(document["telemetry"]["preflight_sha256"]),
        "treatment_identity": treatment_identity,
    }
    return VerifiedCandidateHandoff(
        record_path=Path(path).resolve(), record_sha256=_sha256_file(Path(path).resolve()),
        candidate_path=Path(candidate["path"]), candidate_sha256=str(candidate["sha256"]),
        candidate_initial_sha256=str(candidate["initial_sha256"]),
        functional_run_id=str(base["run_id"]),
        functional_submission_sha256=str(base["submission_sha256"]),
        functional_base_path=Path(base["snapshot"]),
        functional_bundle_snapshot_sha256=str(bundle["content_sha256"]),
        functional_bundle_manifest=Path(bundle["manifest"]),
        functional_bundle_manifest_sha256=str(bundle["manifest_sha256"]),
        target_descriptor=Path(document["target"]["descriptor"]),
        target_descriptor_sha256=str(document["target"]["descriptor_sha256"]),
        corpus_root=Path(corpus["path"]), corpus_manifest=Path(corpus["manifest"]),
        corpus_manifest_sha256=str(corpus["manifest_sha256"]),
        corpus_sha256=str(corpus["capsules_sha256"]), replicates=int(corpus["replicates"]),
        formal_replicate_identities=tuple(
            str(value) for value in corpus["formal_replicate_identities"]),
        formal_claim=copy.deepcopy(dict(corpus["formal_claim"])),
        smoke_replicates=int(corpus["smoke_replicates"]),
        expected_cells=tuple(dict(row) for row in corpus["expected_cells"]),
        families=tuple(dict(row) for row in corpus["families"]),
        host_lane=dict(base["model_host_lane"]), e2e_sentinel=dict(base["e2e_sentinel"]),
        prompt_sha256=str(prompt["sha256"]),
        prompt_facts_sha256=str(prompt["facts_sha256"]),
        prompt_path=Path(prompt["staged_path"]),
        transcript_path=Path(agent["transcript"]),
        transcript_sha256=str(agent["transcript_sha256"]),
        transcript_audit=dict(agent["audit"]),
        receipt_path=Path(broker["receipt_manifest"]),
        receipt_sha256=str(broker["receipt_manifest_sha256"]),
        required_actions=tuple(str(value) for value in broker["required_actions"]),
        tool_evidence={"registry_sha256": broker["registry_sha256"],
                       "round_receipts": broker["round_receipts"],
                       "tool_probe_results": document["sandbox"]["inner_execution_plane"][
                           "tool_probe_results"],
                       "tool_probe_recheck_results": document["sandbox"][
                           "inner_execution_plane"]["tool_probe_recheck_results"]},
        sandbox_evidence={plane: {
            "network": document["sandbox"][plane]["network"],
            "clear_environment": document["sandbox"][plane]["clear_environment"],
            "policy_sha256": document["sandbox"][plane]["policy_sha256"],
        } for plane in ("outer_codex_control_plane", "inner_execution_plane")},
        telemetry_evidence=copy.deepcopy(dict(document["telemetry"])),
        codex_binary_sha256=str(agent["codex_binary_sha256"]),
        authoring_stage_sha256=str(document["prompt"]["renderer_sha256"]),
        telemetry_preflight_sha256=str(document["telemetry"]["preflight_sha256"]),
        telemetry_source_sha256=source_sha256,
        agent_contract=agent_contract)


def _import_codex_driver():
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import codex_agent  # noqa: PLC0415
    import run_baseline_qa_loop  # noqa: PLC0415
    return codex_agent, run_baseline_qa_loop


def _telemetry_source_record(obj: object, *, label: str) -> dict[str, str]:
    source = inspect.getsourcefile(obj)
    if not source:
        raise StageGateError(f"{label} has no inspectable source file")
    path = Path(source).resolve()
    if path.is_symlink() or not path.is_file():
        raise StageGateError(f"{label} source is absent, linked, or non-regular: {path}")
    return {"path": str(path), "sha256": _sha256_file(path)}


def _declared_price_rate(path: Path, model: str) -> tuple[float, float, float, float] | None:
    """Resolve one model in the pinned YAML rate map without any hidden fallback."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, Mapping):
        return None
    lowered = model.lower()
    matches = [(str(key).lower(), value) for key, value in document.items()
               if str(key).lower() in lowered]
    for _key, value in sorted(matches, key=lambda item: (len(item[0]), item[0]), reverse=True):
        try:
            if isinstance(value, Mapping):
                input_rate, output_rate = float(value["input"]), float(value["output"])
                cache_read = float(value.get("cache_read", input_rate * 0.10))
                cache_write = float(value.get(
                    "cache_creation", value.get("cache_write", input_rate * 1.25)))
                return input_rate, output_rate, cache_read, cache_write
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                values = [float(item) for item in value]
                if 2 <= len(values) <= 4:
                    input_rate, output_rate = values[:2]
                    cache_read = values[2] if len(values) >= 3 else input_rate * 0.10
                    cache_write = values[3] if len(values) >= 4 else input_rate * 1.25
                    return input_rate, output_rate, cache_read, cache_write
        except (KeyError, TypeError, ValueError):
            return None
    return None


def telemetry_preflight(*, model: str, price_table: Path | None = None,
                        codex_binary: str | Path | None = None) -> dict[str, Any]:
    """Prove the exact raw-Codex -> token ledger -> AET trajectory path before a paid turn.

    This is deliberately strict for the performance experiment even though the shared AET bridge is
    soft for ordinary developer runs.  It performs an in-memory parser canary and pins every parser
    implementation plus the notional price input; it neither creates a run directory nor launches an
    agent.
    """
    try:
        from merlin.targetgen import experiment_tokens as ET  # noqa: PLC0415
        from aet.trajectory.codex import CodexNormalizer  # noqa: PLC0415
        from aet.trajectory.importers.codex import build_trajectory_from_run  # noqa: PLC0415
        from aet.trajectory.classify import ActivityClassifier, ActivityConfig  # noqa: PLC0415
        from aet.trajectory.reconcile import reconcile_codex  # noqa: PLC0415
        from aet.tracking.run_logger import EvalRunLogger  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"required AET/Codex telemetry stack is unavailable: {exc}") from exc
    CA, _loop = _import_codex_driver()
    try:
        import agent_bridge as model_bridge  # noqa: PLC0415
        resolved_model = str(CA.resolve_model(model) or "").strip()
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"Codex model resolution preflight failed: {exc}") from exc
    if not resolved_model:
        raise StageGateError("Codex model resolution produced an empty model identity")
    codex_path = _require_executable(
        str(codex_binary or os.environ.get("CODEX_BIN") or "codex"), label="Codex")
    if price_table is None:
        try:
            from merlin.common.paths import _dotenv  # noqa: PLC0415
            raw_price = (os.environ.get("AET_PRICE_TABLE") or _dotenv().get("AET_PRICE_TABLE") or "")
        except Exception:  # noqa: BLE001
            raw_price = os.environ.get("AET_PRICE_TABLE", "")
        price_table = Path(raw_price) if raw_price else None
    if price_table is None:
        raise StageGateError("AET_PRICE_TABLE must explicitly pin subscription notional pricing")
    price_table = Path(price_table)
    if price_table.is_symlink() or not price_table.is_file():
        raise StageGateError(f"telemetry price table is absent, linked, or non-regular: {price_table}")
    price_table = price_table.resolve()
    rate = _declared_price_rate(price_table, resolved_model)
    if rate is None:
        raise StageGateError(
            f"telemetry price table has no exact substring rate for {resolved_model!r}")

    # A real schema canary, in memory: unknown event kinds remain reconcilable, token subset
    # arithmetic is exercised, and no filesystem side effect is needed for preflight.
    canary_events = [
        {"type": "thread.started", "thread_id": "telemetry-canary"},
        {"type": "turn.started"},
        {"type": "turn.completed", "usage": {
            "input_tokens": 17, "cached_input_tokens": 5, "cache_write_input_tokens": 2,
            "output_tokens": 7, "reasoning_output_tokens": 3}},
    ]
    normalizer = CodexNormalizer()
    normalizer.feed_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
                                 for row in canary_events))
    run = normalizer.result()
    cfg = ActivityConfig()
    trajectory = build_trajectory_from_run(
        run, run_id="telemetry-canary", classifier=ActivityClassifier(cfg),
        classifier_cfg=cfg.to_dict(), model=resolved_model,
        billing_row={"provider": "openai", "billing_mode": "subscription"}, calculated_at="")
    reconciliation = reconcile_codex(run, trajectory)
    parsed = ET._codex_usage(canary_events)  # same reader used by parse_agent_transcript
    if (not reconciliation.get("ok") or not parsed or parsed.get("usage_complete") is not True
            or parsed.get("tokens_input") != 10 or parsed.get("tokens_cached") != 5
            or parsed.get("tokens_cache_write") != 2 or parsed.get("tokens_output") != 7):
        raise StageGateError("raw Codex/AET telemetry parser canary failed")
    return {
        "schema_version": 1,
        "required": True,
        "driver": "codex",
        "raw_capture": "durable_jsonl_before_interpretation_plus_timestamp_sidecar",
        "accounting": "raw_codex_turn_usage_nonoverlapping_token_buckets",
        "activity": "aet_native_codex_structured_tool_spans",
        "aet_reconciliation_required": True,
        "billing_mode": "subscription_notional",
        "model_resolution": {
            "requested_model": model,
            "resolved_model": resolved_model,
            # resolve_model consults this before every native/bridged/default route.  Preserve the
            # exact ambient input so a resume cannot silently redirect an otherwise identical slug.
            "codex_model_map": os.environ.get("CODEX_MODEL_MAP", ""),
        },
        "price_table": {"path": str(price_table), "sha256": _sha256_file(price_table),
                        "model": resolved_model, "requested_model": model,
                        "rate_per_million": list(rate)},
        "sources": {
            "codex_binary": {"path": str(codex_path),
                             "sha256": _sha256_file(codex_path)},
            "performance_authoring_stage": _telemetry_source_record(
                telemetry_preflight, label="performance authoring stage"),
            "performance_campaign": _telemetry_source_record(
                PC, label="performance campaign helper"),
            "performance_gsim_gate": _telemetry_source_record(
                GATE, label="performance GSIM gate"),
            "performance_pk_claim": _telemetry_source_record(
                PK, label="performance PK claim"),
            "performance_prompt": _telemetry_source_record(
                PP, label="performance prompt contract"),
            "codex_driver": _telemetry_source_record(CA, label="Codex driver"),
            "codex_model_bridge": _telemetry_source_record(
                model_bridge, label="Codex model bridge"),
            "benchharness": _telemetry_source_record(
                hash_tree, label="candidate tree hashing"),
            "sandbox_bwrap": _telemetry_source_record(
                BW, label="agent bwrap policy"),
            "sandbox_toolchain": _telemetry_source_record(
                TC, label="agent toolchain policy"),
            "sandbox_answer_surfaces": _telemetry_source_record(
                answer_surfaces, label="answer-surface policy"),
            "target_experiment_loader": _telemetry_source_record(
                load_target_experiment, label="target experiment loader"),
            "experiment_tokens": _telemetry_source_record(ET, label="token accounting"),
            "aet_codex_normalizer": _telemetry_source_record(
                CodexNormalizer, label="AET Codex normalizer"),
            "aet_codex_importer": _telemetry_source_record(
                build_trajectory_from_run, label="AET Codex importer"),
            "aet_reconciliation": _telemetry_source_record(
                reconcile_codex, label="AET reconciliation"),
            "aet_activity_classifier": _telemetry_source_record(
                ActivityClassifier, label="AET activity classifier"),
            "aet_canonical_logger": _telemetry_source_record(
                EvalRunLogger, label="AET canonical logger"),
        },
    }


def _round_telemetry(stage_root: Path, round_index: int, *, model: str,
                     agent_exit_code: int) -> dict[str, Any]:
    """Validate and hash the driver's raw, timestamped and summary artifacts for one round."""
    from datetime import datetime  # noqa: PLC0415
    from merlin.targetgen import experiment_tokens as ET  # noqa: PLC0415

    rounds = stage_root / "rounds"
    paths = {
        "raw": rounds / f"round_{round_index:02d}.codex_events.raw.jsonl",
        "timestamped": rounds / f"round_{round_index:02d}.codex_events.timestamped.jsonl",
        "summary": rounds / f"round_{round_index:02d}.codex_summary.json",
        "stderr": rounds / f"round_{round_index:02d}.codex_stderr.log",
        "prompt": rounds / f"round_{round_index:02d}.prompt.txt",
        "final": rounds / f"round_{round_index:02d}.final.txt",
    }
    for label, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise StageGateError(f"Codex round {round_index} lacks real {label} telemetry: {path}")
    try:
        raw_lines = paths["raw"].read_text(encoding="utf-8").splitlines()
        stamped = [json.loads(line) for line in paths["timestamped"].read_text(
            encoding="utf-8").splitlines() if line.strip()]
    except (UnicodeError, ValueError) as exc:
        raise StageGateError(f"Codex round {round_index} telemetry is malformed: {exc}") from exc
    if not raw_lines or len(stamped) != len(raw_lines):
        raise StageGateError(f"Codex round {round_index} raw/timestamped event counts differ or are zero")
    for sequence, (raw_line, wrapper) in enumerate(zip(raw_lines, stamped, strict=True), start=1):
        if not isinstance(wrapper, Mapping) or wrapper.get("seq") != sequence:
            raise StageGateError(f"Codex round {round_index} timestamp sequence is discontinuous")
        try:
            datetime.fromisoformat(str(wrapper["arrived_at"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise StageGateError(f"Codex round {round_index} has an invalid arrival timestamp") from exc
        try:
            event = json.loads(raw_line)
        except ValueError:
            if wrapper.get("unparsed") != raw_line or "event" in wrapper:
                raise StageGateError(f"Codex round {round_index} sidecar changed raw line {sequence}")
        else:
            if wrapper.get("event") != event or "unparsed" in wrapper:
                raise StageGateError(f"Codex round {round_index} sidecar changed raw event {sequence}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if (not isinstance(summary, Mapping) or summary.get("billing_mode") != "subscription_notional"
            or summary.get("exit_code") != agent_exit_code
            or summary.get("usage_complete") is not True
            or summary.get("timed_out") is not False
            or not isinstance(summary.get("wall_s"), (int, float))
            or float(summary["wall_s"]) <= 0):
        raise StageGateError(f"Codex round {round_index} usage/timing summary is incomplete")
    accounting = ET.parse_agent_transcript(
        paths["raw"], driver="codex", model=model,
        billing_mode=ET.SUBSCRIPTION_NOTIONAL)
    if (accounting.get("available") is not True or accounting.get("usage_complete") is not True
            or not isinstance(accounting.get("tokens_total"), int)
            or accounting["tokens_total"] <= 0):
        raise StageGateError(f"Codex round {round_index} token accounting is incomplete")
    for path in paths.values():
        path.chmod(0o444)
    return {
        "event_count": len(raw_lines), "summary": dict(summary), "accounting": accounting,
        "artifacts": {label: {"path": str(path), "sha256": _sha256_file(path),
                              "bytes": path.stat().st_size}
                      for label, path in paths.items()},
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")


def finalize_agent_telemetry(stage_root: Path, round_records: Sequence[Mapping[str, Any]], *,
                             model: str, run_id: str,
                             preflight_record: Mapping[str, Any]) -> dict[str, Any]:
    """Produce fail-closed AET, cost, activity, token-ledger and tool-ledger artifacts."""
    from merlin.targetgen import experiment_tokens as ET  # noqa: PLC0415
    price_record = preflight_record.get("price_table") or {}
    price_path = Path(str(price_record.get("path") or ""))
    if (price_path.is_symlink() or not price_path.is_file()
            or _sha256_file(price_path) != price_record.get("sha256")):
        raise StageGateError("pinned telemetry price table changed before final accounting")
    for label, source in (preflight_record.get("sources") or {}).items():
        source_path = Path(str((source or {}).get("path") or ""))
        if (source_path.is_symlink() or not source_path.is_file()
                or _sha256_file(source_path) != (source or {}).get("sha256")):
            raise StageGateError(f"pinned telemetry implementation changed: {label}")
    os.environ["AET_PRICE_TABLE"] = str(price_path)
    ET._OVERRIDES = None
    try:
        from aet.trajectory.importers.codex import import_codex_run  # noqa: PLC0415
        from aet.trajectory.reconcile import (  # noqa: PLC0415
            reconcile_codex, token_ledger_rows, tool_ledger_rows)
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"required AET telemetry finalizer is unavailable: {exc}") from exc

    root = stage_root / "telemetry"
    raw_dir, ts_dir = root / "raw", root / "timestamped"
    raw_dir.mkdir(parents=True, exist_ok=False)
    ts_dir.mkdir(parents=True, exist_ok=False)
    combined_raw = root / "codex_events.raw.jsonl"
    total_wall = 0.0
    with combined_raw.open("wb") as combined:
        for row in round_records:
            index = int(row["round"])
            evidence = row.get("telemetry") or {}
            artifacts = evidence.get("artifacts") or {}
            raw = Path(str((artifacts.get("raw") or {}).get("path") or ""))
            stamped_path = Path(str((artifacts.get("timestamped") or {}).get("path") or ""))
            if (_sha256_file(raw) != (artifacts.get("raw") or {}).get("sha256")
                    or _sha256_file(stamped_path) != (artifacts.get("timestamped") or {}).get("sha256")):
                raise StageGateError(f"Codex round {index} telemetry changed before AET import")
            raw_payload = raw.read_bytes()
            combined.write(raw_payload)
            if raw_payload and not raw_payload.endswith(b"\n"):
                combined.write(b"\n")
            destination = raw_dir / f"events.{index:02d}.jsonl"
            destination.write_bytes(raw_payload)
            raw_lines = raw.read_text(encoding="utf-8").splitlines()
            wrappers = [json.loads(line) for line in stamped_path.read_text(
                encoding="utf-8").splitlines() if line.strip()]
            _write_jsonl(ts_dir / destination.name, [
                {"ts": wrapper["arrived_at"], "line": line}
                for line, wrapper in zip(raw_lines, wrappers, strict=True)])
            total_wall += float((evidence.get("summary") or {}).get("wall_s", 0.0))

    trajectory, normalized = import_codex_run(
        raw_dir, timestamped=ts_dir, run_id=run_id, model=model,
        billing_mode="subscription", provider="openai", calculated_at="")
    reconciliation = reconcile_codex(normalized, trajectory, admin_usd=None)
    expected_raw_events = sum(int((row.get("telemetry") or {}).get("event_count", 0))
                              for row in round_records)
    if (reconciliation.get("ok") is not True
            or reconciliation["raw_events"].get("raw_event_count") != expected_raw_events
            or normalized.raw_event_count != expected_raw_events
            or reconciliation["token_ledger"].get("num_turns") < len(round_records)):
        raise StageGateError("AET failed to reconcile the complete raw Codex event stream")
    tool_rows = tool_ledger_rows(normalized)
    if not tool_rows:
        raise StageGateError("AET found zero structured tool calls in the performance agent run")
    trajectory_path = stage_root / "metrics" / "trajectory.json"
    trajectory.to_json(trajectory_path)
    reconciliation_path = root / "aet_reconciliation.json"
    _write_json(reconciliation_path, reconciliation)
    token_ledger = stage_root / "metrics" / "token_ledger.jsonl"
    tool_ledger = stage_root / "agent" / "tools.jsonl"
    _write_jsonl(token_ledger, token_ledger_rows(normalized))
    _write_jsonl(tool_ledger, tool_rows)

    accounting = ET.parse_agent_transcript(
        combined_raw, driver="codex", model=model, billing_mode=ET.SUBSCRIPTION_NOTIONAL)
    if accounting.get("available") is not True or accounting.get("usage_complete") is not True:
        raise StageGateError("combined raw Codex token/cost accounting is incomplete")
    if (not isinstance(accounting.get("subscription_notional_usd"), (int, float))
            or accounting["subscription_notional_usd"] <= 0):
        raise StageGateError("combined raw Codex usage lacks pinned subscription-notional cost")
    accounting["tool_calls"] = len(tool_rows)
    accounting["subagent_tool_calls_tracked"] = False
    cost_path = stage_root / "cost_time_toolcalls.yaml"
    ET.write_cost_yaml(accounting, cost_path, wall_time_seconds=round(total_wall, 3),
                       model=model, exit_code=0)

    # Native AET run-store rows make the trial discoverable by `aet spend`.  A ChatGPT seat has
    # real spend 0 here; the separately named notional metric cannot be summed into a billed budget.
    try:
        from aet.tracking.run_logger import EvalRunLogger  # noqa: PLC0415
        logger = EvalRunLogger.start(
            project="merlin", suite="gemmini-perf-bench", target="gemmini",
            method="agentic_perf_trial", seed=0, run_id=run_id,
            run_path=stage_root, tracking_mode="local")
        logger.log_token_usage(
            input_tokens=int(accounting.get("tokens_input", 0)),
            output_tokens=int(accounting.get("tokens_output", 0)),
            cache_creation_tokens=int(accounting.get("tokens_cache_write", 0)),
            cache_read_tokens=int(accounting.get("tokens_cached", 0)), model=model)
        logger.log_cost(0.0, model=model)
        logger.log_param("billing_mode", "subscription_notional")
        if accounting.get("subscription_notional_usd") is not None:
            logger.log_metric("cost.subscription_notional_usd",
                              float(accounting["subscription_notional_usd"]))
        logger.log_agent_turns(len(normalized.turns))
        logger.close()
    except Exception as exc:  # noqa: BLE001
        raise StageGateError(f"AET canonical run logger failed: {exc}") from exc
    metrics_log = stage_root / "logs" / "metrics.jsonl"
    if metrics_log.is_symlink() or not metrics_log.is_file() or metrics_log.stat().st_size <= 0:
        raise StageGateError("AET canonical metrics log was not materialized")

    durations: dict[str, float] = {}
    for band in trajectory.bands:
        durations[band.category] = durations.get(band.category, 0.0) + band.duration_s
    classified = sum(durations.values())
    if classified <= 0 or trajectory.duration_s <= 0:
        raise StageGateError("AET produced no duration-bearing classified activity spans")
    activity = {
        "schema_version": 2,
        "basis": "aet_native_codex_structured_tool_spans",
        "denominator": "sum_of_classified_tool_span_seconds_including_overlap",
        "is_wall_time_partition": False,
        "overlapping_tool_spans_allowed": True,
        "classified_seconds": round(classified, 6),
        "trajectory_wall_seconds": trajectory.duration_s,
        "agent_round_wall_seconds": round(total_wall, 3),
        "classified_span_occupancy_ratio": (round(classified / trajectory.duration_s, 6)
                                            if trajectory.duration_s > 0 else None),
        "occupancy_ratio_may_exceed_one": True,
        "subagent_tool_calls_tracked": False,
        "seconds_by_category": {key: round(value, 6) for key, value in sorted(durations.items())},
        "share_by_category": ({key: round(value / classified, 8)
                               for key, value in sorted(durations.items())}
                              if classified else {}),
        "note": ("shares partition classified tool-span seconds, not wall time; simultaneous tool "
                 "spans each contribute their full duration, so classified_span_occupancy_ratio may "
                 "exceed 1; unclassified wall is not relabeled as thinking"),
    }
    activity_path = root / "activity_share.json"
    _write_json(activity_path, activity)
    preflight_path = root / "preflight.json"
    _write_json(preflight_path, dict(preflight_record))
    artifact_paths = {
        "combined_raw": combined_raw, "trajectory": trajectory_path,
        "reconciliation": reconciliation_path, "token_ledger": token_ledger,
        "tool_ledger": tool_ledger, "cost_time_toolcalls": cost_path,
        "activity_share": activity_path, "preflight": preflight_path,
        "aet_metrics_log": metrics_log,
    }
    for path in (*raw_dir.iterdir(), *ts_dir.iterdir(), *artifact_paths.values()):
        path.chmod(0o444)
    return {
        "required": True, "driver": "codex", "raw_event_count": normalized.raw_event_count,
        "tool_call_count": len(tool_rows), "rounds_with_complete_usage": len(round_records),
        "subagent_tool_calls_tracked": False,
        "billing_mode": "subscription_notional", "accounting": accounting,
        "activity_share": activity,
        "aet_reconciliation": reconciliation,
        "preflight_sha256": _sha256(_canonical_json(preflight_record)),
        "artifacts": {name: {"path": str(path), "sha256": _sha256_file(path)}
                      for name, path in artifact_paths.items()},
    }


def _codex_round(
        workspace: Path, stage_root: Path, prompt: PromptArtifact, target_experiment: TargetExperiment,
        agent_inputs: AgentInputSnapshot, frozen_functional: FrozenFunctionalInputs,
        functional_base: Path, frozen_corpus_manifest: Path, control_dir: Path, *,
        model: str, resolved_model: str, effort: str, round_index: int,
        timeout_s: int, codex_binary: Path) -> tuple[int, Path, AgentSandboxPolicy]:
    CA, loop = _import_codex_driver()
    from merlin.common import artifacts as artifact_paths  # noqa: PLC0415

    original_bwrap = loop.bwrap_cmd
    original_cache_dir = artifact_paths.cache_dir
    original_codex_bin = os.environ.get("CODEX_BIN")
    captured: dict[str, AgentSandboxPolicy] = {}

    def stage_local_cache(namespace: str) -> Path:
        if namespace == "codex_home":
            path = stage_root / "codex_homes"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return original_cache_dir(namespace)

    def exact_bwrap(inner: str, ws: Path, _bundle: dict,
                    extra_binds: list[str] | None = None) -> str:
        policy = outer_codex_policy(
            ws, agent_inputs, extra_binds or (), target_experiment, frozen_functional,
            functional_base, control_dir, frozen_corpus_manifest)
        captured["policy"] = policy
        # ``inner`` is already shell-quoted by codex_agent.  Quote only the outer payload boundary.
        return (" ".join(policy.argv) + " bash -c '" + inner.replace("'", "'\\''") + "'")

    loop.bwrap_cmd = exact_bwrap
    artifact_paths.cache_dir = stage_local_cache
    os.environ["CODEX_BIN"] = str(codex_binary)
    try:
        rc, transcript = CA.run_round(
            workspace, stage_root, model, {}, target_experiment, "bwrap", round_index, timeout_s,
            effort=effort, prompt=prompt.text, effective_model=resolved_model)
    finally:
        loop.bwrap_cmd = original_bwrap
        artifact_paths.cache_dir = original_cache_dir
        if original_codex_bin is None:
            os.environ.pop("CODEX_BIN", None)
        else:
            os.environ["CODEX_BIN"] = original_codex_bin
    policy = captured.get("policy")
    if policy is None:
        raise StageGateError("Codex driver did not construct the required outer bwrap policy")
    return rc, transcript, policy


def run_stage(
        *, functional_runs_root: Path, functional_run_id: str, functional_submission_sha256: str,
        target_experiment: TargetExperiment, stage_root: Path,
        model: str, effort: str, wall_budget_seconds: int, rounds: int, round_timeout_seconds: int,
        max_tool_calls: int, tool_timeout_seconds: int, replicates: int | None = None,
        smoke_replicates: int = 1, families: str = "all",
        capsules: str = "all", codex_binary: str = "codex",
        gsim_certificate: Path | None = None,
        gsim_certificate_sha256: str | None = None,
        rtl_facts: Path | None = None,
        telemetry_price_table: Path | None = None) -> Path:
    """Run bounded authoring rounds and return the sealed candidate-record path."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in
           (wall_budget_seconds, rounds, round_timeout_seconds, max_tool_calls, tool_timeout_seconds,
            smoke_replicates)):
        raise StageGateError("all performance stage budgets must be positive integers")
    if not model.strip():
        raise StageGateError("an explicit Codex model is required")
    bwrap_binary = _require_executable("bwrap", label="bwrap")
    codex_path = _require_executable(codex_binary, label="Codex")
    descriptor_path = Path(target_experiment.path).resolve()
    bwrap_sha256 = _sha256_file(bwrap_binary)
    codex_sha256 = _sha256_file(codex_path)
    descriptor_sha256 = _sha256_file(descriptor_path)
    telemetry_preflight_record = telemetry_preflight(
        model=model, price_table=telemetry_price_table, codex_binary=codex_path)
    resolved_model = str((telemetry_preflight_record.get("model_resolution") or {}).get(
        "resolved_model") or "")
    if not resolved_model:
        raise StageGateError("telemetry preflight omitted the resolved Codex model")
    if bwrap_binary.name != "bwrap":
        raise StageGateError("the sandbox executable does not resolve to bwrap")
    raw_stage_root = Path(stage_root)
    if raw_stage_root.exists() or raw_stage_root.is_symlink():
        raise StageGateError(f"performance agent stage must use a fresh directory: {raw_stage_root}")
    stage_root = raw_stage_root.resolve()

    functional = inspect_stage_functional_run(
        functional_runs_root, functional_run_id, functional_submission_sha256)
    discovered = discover_performance_corpus(
        target_experiment, families=families, capsules=capsules)
    stage_root.mkdir(parents=True)
    base = PC.materialize_perf_workspace(functional, stage_root / "_frozen_functional")
    frozen_corpus = freeze_performance_corpus(discovered, stage_root / "_frozen_corpus")
    formal_claim = prepare_formal_pk_claim(frozen_corpus.capsules, replicates)
    replicate_contract = formal_claim["declaration"]["replicates"]
    replicates = int(replicate_contract["exact_count"])
    agent_inputs = build_answer_free_agent_inputs(
        frozen_corpus, target_experiment, stage_root / "_agent_inputs")
    frozen_functional = load_frozen_functional_inputs(functional)
    prepared_actions = build_action_registry(base, target_experiment)
    prepared_action_contract = action_registry_contract(prepared_actions, base)
    minimum_calls = rounds * sum(action.required for action in prepared_actions)
    if max_tool_calls < minimum_calls:
        raise StageGateError(
            "inner tool-call budget cannot cover every required broker action in every round: "
            f"need at least {minimum_calls}, got {max_tool_calls}")
    feedback = prepare_development_feedback(
        certificate_path=gsim_certificate, certificate_sha256=gsim_certificate_sha256,
        rtl_facts_path=rtl_facts, corpus=frozen_corpus, baseline=base,
        baseline_sha256=functional.digest, target_experiment=target_experiment,
        work_root=stage_root / "_development_feedback",
        # Every tuning measurement spends one inner tool call, so the run's own call budget is the
        # real bound on how many the search can take. Declaring it makes ``budget_exhausted`` report
        # true remaining spend instead of "unbounded".
        tuning_call_budget=max_tool_calls,
        functional_run_dir=functional.run_dir)
    prompt_inputs = prepare_prompt_inputs(
        functional, frozen_functional, frozen_corpus, agent_inputs, target_experiment,
        prepared_actions, formal_claim=formal_claim, smoke_replicates=smoke_replicates,
        wall_budget_seconds=wall_budget_seconds,
        rounds=rounds, round_timeout_seconds=round_timeout_seconds,
        max_tool_calls=max_tool_calls, tool_timeout_seconds=tool_timeout_seconds)
    staged_prompt = stage_root / "prompt.txt"
    prompt = materialize_canonical_prompt(prompt_inputs, staged_prompt)
    staged_prompt.chmod(0o444)
    fork = PC.functional_fork(functional)
    fork_check = PC.check_fork(fork, base)
    if fork_check.ok is not True:
        raise StageGateError(f"functional fork is invalid before performance authoring: {fork_check.reason}")

    deadline = time.monotonic() + wall_budget_seconds
    previous_submission = base
    previous_digest = functional.digest
    round_records: list[dict[str, Any]] = []
    transcript_paths: list[Path] = []
    probe_results: list[dict[str, Any]] | None = None
    last_outer: AgentSandboxPolicy | None = None
    refusal: str | None = None
    stopped_by: dict[str, Any] | None = None
    total_calls = 0
    receipt_records: list[dict[str, Any]] = []
    last_actions: tuple[BrokerAction, ...] = ()
    last_inner: AgentSandboxPolicy | None = None
    for round_index in range(rounds):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            refusal = "performance stage wall-clock budget expired before the next round"
            break
        workspace = stage_root / "agent_workspaces" / f"round_{round_index:02d}"
        candidate = fresh_round_workspace(previous_submission, workspace, previous_digest)
        (workspace / "TASK.md").write_text(prompt.text, encoding="utf-8")
        _write_json(workspace / "STAGE_CONTEXT.json", {
            "functional_run_id": functional.run_id,
            "functional_submission_sha256": functional.digest,
            "performance_manifest_sha256": frozen_corpus.manifest_sha256,
            "performance_corpus_sha256": frozen_corpus.capsules_sha256,
            "agent_corpus_mount": str(AGENT_CORPUS_MOUNT),
            "candidate": "submission",
            "tool_command": f"python3 {BROKER_NAME} ACTION [NAME=VALUE ...]",
            "broker_actions": [action.name for action in prepared_actions],
            "formal_replicates": replicates,
            "formal_replicate_identities": list(
                prompt_inputs.formal_replicate_identities),
            "smoke_replicates": smoke_replicates,
            "round": round_index,
            "rounds": rounds,
            "remaining_wall_budget_seconds": remaining,
        })
        actions = build_action_registry(candidate, target_experiment)
        if action_registry_contract(actions, candidate) != prepared_action_contract:
            refusal = f"candidate manifest action contract drifted before round {round_index}"
            break
        inner = inner_execution_policy(
            target_experiment, candidate, agent_inputs, frozen_functional, base,
            frozen_corpus.manifest_path)
        if probe_results is None:
            probe_results = run_required_tool_probes(inner, target_experiment, candidate)
        control_dir = stage_root / "control" / f"round_{round_index:02d}"
        receipt_path = control_dir / "receipts.jsonl"
        broker = _Broker(inner, target_experiment, candidate, actions, receipt_path,
                         deadline=deadline, max_calls=max_tool_calls - total_calls,
                         max_tool_seconds=tool_timeout_seconds,
                         feedback_evaluator=feedback, feedback_round=round_index)
        round_timeout = min(round_timeout_seconds, remaining)
        try:
            with broker.serving() as (host, port):
                stage_broker_shim(control_dir, host=host, port=port, token=broker.token,
                                  tool_timeout_s=tool_timeout_seconds, actions=actions)
                rc, transcript, outer = _codex_round(
                    workspace, stage_root, prompt, target_experiment, agent_inputs,
                    frozen_functional, base, frozen_corpus.manifest_path, control_dir,
                    model=model, resolved_model=resolved_model,
                    effort=effort, round_index=round_index, timeout_s=round_timeout,
                    codex_binary=codex_path)
        finally:
            broker_config = control_dir / ".perf_broker.json"
            if broker_config.is_file() and not broker_config.is_symlink():
                broker_config.chmod(0o600)
                broker_config.unlink()
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt_path.chmod(0o444)
        total_calls += len(broker.calls)
        transcript_paths.append(transcript)
        round_telemetry = _round_telemetry(
            stage_root, round_index, model=resolved_model, agent_exit_code=rc)
        # A DESTRUCTIVE refusal loses the very evidence needed to diagnose it. Measured 2026-09-03:
        # perf_stage_20260903T163936Z did five complete optimisation iterations and produced a full
        # measurement trace, then raised here over bytecode caches and wrote NO candidate record at
        # all -- while the transcript audit, which merely RECORDS its refusal, is what made every
        # other defect diagnosable. Ephemeral state makes a candidate unconsumable; it is not an
        # integrity violation, so it is recorded and the run still lands its evidence.
        try:
            assert_candidate_sealable(candidate)
        except StageGateError as exc:
            refusal = refusal or f"round {round_index} candidate is not sealable: {exc}"
        observed = hash_tree(candidate)["sha256"]
        audit = audit_codex_transcript(transcript, target_experiment, candidate, actions)
        try:
            receipt_evidence = verify_broker_receipts(
                receipt_path, actions, audit, candidate_sha256=observed)
        except StageGateError as exc:
            receipt_evidence = {"path": str(receipt_path), "error": str(exc),
                                "all_required_succeeded": False}
            refusal = f"Codex round {round_index} failed broker receipt enforcement: {exc}"
        if not any(call.get("action") == DEVELOPMENT_FEEDBACK_ACTION
                   and call.get("returncode") == 0 for call in broker.calls):
            refusal = (f"Codex round {round_index} did not successfully invoke mandatory "
                       "tuning GSIM feedback")
        receipt_records.append(receipt_evidence)
        audit_path = stage_root / "rounds" / f"round_{round_index:02d}.audit.json"
        _write_json(audit_path, audit)
        round_record = {
            "round": round_index, "workspace": str(workspace), "candidate_sha256": observed,
            "agent_exit_code": rc, "transcript": str(transcript),
            "transcript_sha256": _sha256(transcript.read_bytes()), "audit": audit,
            "telemetry": round_telemetry,
            "broker_calls": broker.calls, "broker_receipts": receipt_evidence,
            "broker_registry_sha256": _sha256(_canonical_json(
                [action.as_dict() for action in actions])), "budget_seconds": round_timeout,
        }
        round_records.append(round_record)
        last_outer = outer
        last_inner = inner
        last_actions = actions
        previous_submission, previous_digest = candidate, observed
        if rc != 0:
            refusal = f"Codex round {round_index} exited with rc={rc}"
            break
        if refusal is not None:
            break
        if not audit["clean"]:
            refusal = f"Codex round {round_index} failed the answer/tool-access audit"
            break
        # STOPPING ON EVIDENCE IS A SUCCESS, NOT A REFUSAL. `refusal` is the NO-GO channel: anything
        # placed in it makes the run unconsumable and returns 2. A converged search must therefore
        # end through its own variable, and the consumability tests below have to admit a run that
        # ended early because it was finished rather than because it broke.
        if getattr(broker, "stop_verdict", None) is not None:
            stopped_by = dict(broker.stop_verdict)
            stopped_by["round"] = round_index
            break

    if (not round_records or not transcript_paths or last_outer is None or last_inner is None
            or probe_results is None or not last_actions):
        raise StageGateError(refusal or "performance stage completed no auditable Codex round")
    probe_recheck_results: list[dict[str, Any]] = []
    try:
        probe_recheck_results = run_required_tool_probes(
            last_inner, target_experiment, previous_submission)
        if probe_recheck_results != probe_results:
            refusal = "inner sandbox tool probe evidence changed during performance authoring"
    except StageGateError as exc:
        refusal = f"inner sandbox tool recheck failed: {exc}"
    if (_sha256_file(bwrap_binary) != bwrap_sha256 or _sha256_file(codex_path) != codex_sha256
            or _sha256_file(descriptor_path) != descriptor_sha256):
        refusal = "descriptor or agent/sandbox executable bytes changed during performance authoring"
    verify_frozen_performance_corpus(frozen_corpus)
    verify_answer_free_agent_inputs(agent_inputs)
    sealed = stage_root / "sealed_candidate" / "submission"
    try:
        assert_candidate_sealable(previous_submission)
    except StageGateError as exc:
        refusal = refusal or f"final candidate is not sealable: {exc}"
    sealed_sha = PC.materialize_readonly_tree(previous_submission, sealed)
    if sealed_sha != previous_digest:
        raise StageGateError("sealed performance candidate changed during final copy")
    after = PC.check_fork(fork, base)
    if after.ok is not True:
        refusal = f"functional base fork changed during authoring: {after.reason}"
    exits_clean = all(row["agent_exit_code"] == 0 for row in round_records)
    delta = candidate_delta(base, previous_submission)
    if not delta["execution_relevant_changed_files"]:
        refusal = "Codex produced no execution-relevant candidate change"
    combined_transcript = stage_root / "rounds" / "combined.transcript.jsonl"
    with combined_transcript.open("wb") as stream:
        for path in transcript_paths:
            payload = path.read_bytes()
            stream.write(payload)
            if payload and not payload.endswith(b"\n"):
                stream.write(b"\n")
    combined_transcript.chmod(0o444)
    telemetry_record = finalize_agent_telemetry(
        stage_root, round_records, model=resolved_model, run_id=stage_root.name,
        preflight_record=telemetry_preflight_record)
    combined_audit = audit_codex_transcript(
        combined_transcript, target_experiment, previous_submission, prepared_actions)
    round_audits_clean = all(
        row["audit"]["clean"] and row["audit"]["commands_seen"] > 0 for row in round_records)
    audits_clean = round_audits_clean and combined_audit["clean"]
    if combined_audit["commands_seen"] <= 0:
        refusal = "combined Codex transcript contains zero command evidence"
    elif not combined_audit["clean"]:
        refusal = "combined Codex transcript failed the answer/tool-access audit"
    # A converged run has FEWER round records than `rounds`, by design. Requiring exact equality
    # would mark the search unconsumable for having finished early, which is the outcome the stop
    # rule exists to produce.
    expected_rounds = len(round_records) if stopped_by is not None else rounds
    receipts_clean = (len(receipt_records) == expected_rounds and all(
        row.get("all_required_succeeded") is True and row.get("feedback_successes", 0) >= 1
        for row in receipt_records))
    if not receipts_clean:
        refusal = refusal or "required host-owned broker receipt evidence is incomplete"
    # THE FUNCTIONAL GUARD. Phase 1 certified the baseline on the functional corpus and this stage
    # never re-grades it, so without this a candidate could pass every performance cell while breaking
    # capsules nothing here executes. It is cheap because it is a proof, not a sample: a capsule whose
    # emitted code is byte-identical cannot have changed behaviour.
    try:
        functional_guard = functional_emission_guard(base, previous_submission, target_experiment)
    except Exception as exc:  # noqa: BLE001 - an unrunnable guard is absence of proof, not a pass
        functional_guard = {"status": "unavailable",
                            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                            "offenders": [], "rows": []}
    if functional_guard.get("status") != "clean":
        kinds = sorted({str(row.get("kind")) for row in functional_guard.get("offenders") or ()})
        refusal = refusal or (
            "performance candidate did not clear the certified functional emission guard "
            f"({functional_guard.get('status')}"
            + (f": {', '.join(kinds)}" if kinds else "") + ")")
    functional_guard_clean = functional_guard.get("status") == "clean"
    consumable = (refusal is None and audits_clean and exits_clean and receipts_clean
                  and functional_guard_clean and len(round_records) == expected_rounds
                  and len(round_records) >= 1)
    receipt_manifest = stage_root / "control" / "receipt_manifest.json"
    _write_json(receipt_manifest, {"schema_version": 1, "rounds": receipt_records})
    receipt_manifest.chmod(0o444)
    expected_cells = [{"family": cell.family, "capsule": cell.capsule,
                       "simulator": cell.simulator, "replicate": cell.replicate}
                      for cell in prompt_inputs.expected_cells]
    family_facts = [{"family": family.family, "claim": family.claim,
                     "negative_control": family.negative_control,
                     "falsifier_observation": family.falsifier_observation,
                     "differential_basis": family.differential_basis,
                     "fitted_parameters": list(family.fitted_parameters),
                     "acceptance": copy.deepcopy(family.acceptance)}
                    for family in prompt_inputs.families]
    host_lane = {
        "target": prompt_inputs.host_lane.target,
        "package_id": prompt_inputs.host_lane.package_id,
        "package_path": prompt_inputs.host_lane.package_path,
        "package_sha256": prompt_inputs.host_lane.package_sha256,
        "manifest_path": prompt_inputs.host_lane.manifest_path,
        "integration_seam": prompt_inputs.host_lane.integration_seam,
    }
    model_host_record = dict(functional.model_host_lane_snapshot)
    model_host_record.update(host_lane)
    e2e_sentinel = {
        "capsule": prompt_inputs.e2e_sentinel.capsule,
        "capsule_path": prompt_inputs.e2e_sentinel.capsule_path,
        "frozen_source_path": prompt_inputs.e2e_sentinel.frozen_source_path,
        "capsule_sha256": prompt_inputs.e2e_sentinel.capsule_sha256,
        "required_lanes": list(prompt_inputs.e2e_sentinel.required_lanes),
        "required_tiers": list(prompt_inputs.e2e_sentinel.required_tiers),
        "purpose": "functional_L2_L3_admission_not_performance_measurement",
    }
    frozen_grants = [{"declared_path": grant.declared_path,
                      "destination": str(grant.destination), "source": str(grant.source),
                      "source_sha256": grant.source_sha256}
                     for grant in frozen_functional.grants]
    prompt_facts = {
        "replicates": replicates,
        "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
        "formal_claim": copy.deepcopy(formal_claim),
        "smoke_replicates": smoke_replicates,
        "expected_cells": expected_cells,
        "budgets": {"wall_budget_seconds": wall_budget_seconds, "rounds": rounds,
                    "round_timeout_seconds": round_timeout_seconds,
                    "max_tool_calls": max_tool_calls,
                    "tool_timeout_seconds": tool_timeout_seconds},
        "families": family_facts, "host_lane": host_lane, "e2e_sentinel": e2e_sentinel,
        "tools": prepared_action_contract,
        "mount_destinations": list(prompt_inputs.allowed_paths),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "kind": "arm4_performance_candidate",
        "state": "sealed" if consumable else "refused",
        # Per-capsule evidence that the certified functional emission survived this candidate: which
        # capsules are PROVED unchanged (byte-identical emission), which changed, and what the changed
        # ones introduced relative to the certified baseline.
        "functional_guard": functional_guard,
        "target": {
            "name": target_experiment.target,
            "descriptor": str(descriptor_path),
            "descriptor_sha256": descriptor_sha256,
        },
        "base_functional": {
            "run_id": functional.run_id, "submission_sha256": functional.digest,
            "snapshot": str(base), "fork_before": fork_check.to_dict(), "fork_after": after.to_dict(),
            "bundle_input_snapshot": {
                "path": str(frozen_functional.root),
                "content_sha256": frozen_functional.content_sha256,
                "manifest": str(frozen_functional.marker),
                "manifest_sha256": frozen_functional.marker_sha256,
                "grants": frozen_grants,
            },
            "model_host_lane": model_host_record,
            "e2e_sentinel": e2e_sentinel,
        },
        "candidate": {
            "path": str(sealed), "initial_sha256": functional.digest, "sha256": sealed_sha,
            "rounds_completed": len(round_records), "read_only": True,
            "base_submission_overwritten": False,
            "delta": delta,
        },
        "prompt": {"renderer_path": str(Path(__file__).resolve()),
                   "renderer_sha256": _sha256_file(Path(__file__).resolve()),
                   "staged_path": str(staged_prompt), "sha256": prompt.sha256,
                   "n_bytes": prompt.n_bytes, "facts": prompt_facts,
                   "facts_sha256": _sha256(_canonical_json(prompt_facts))},
        "performance_corpus": {
            "path": str(frozen_corpus.root), "manifest": str(frozen_corpus.manifest_path),
            "manifest_sha256": frozen_corpus.manifest_sha256,
            "capsules_sha256": frozen_corpus.capsules_sha256,
            "agent_input_path": str(agent_inputs.root),
            "agent_input_manifest": str(agent_inputs.manifest_path),
            "agent_input_manifest_sha256": agent_inputs.manifest_sha256,
            "agent_input_sha256": agent_inputs.content_sha256,
            "agent_input_files": agent_inputs.n_files,
            "agent_input_bytes": agent_inputs.n_bytes,
            "replicates": replicates,
            "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
            "formal_claim": copy.deepcopy(formal_claim),
            "smoke_replicates": smoke_replicates,
            "expected_cells": expected_cells,
            "families": family_facts,
        },
        "development_feedback": {
            "action": DEVELOPMENT_FEEDBACK_ACTION,
            "required_per_round": True,
            "scope": "frozen_tuning_corpus_only",
            "engine": "gsim",
            "certificate": feedback.certificate.to_dict(),
            "rtl_identity": copy.deepcopy(dict(feedback.rtl_identity)),
            "redaction": "correctness_gsim_cycles_and_paired_deltas_only",
            "round_receipts": [row.get("feedback_receipts", []) for row in receipt_records],
        },
        "sandbox": {
            "outer_codex_control_plane": {
                "engine": "bwrap", "network": last_outer.network,
                "clear_environment": last_outer.clear_environment,
                "auth_exception": "isolated_codex_home_explicit_auth_mount",
                "session_history_mounted": False,
                "live_target_toolchain_mounted": False,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": frozen_functional.marker_sha256,
                "mount_destinations": list(prompt_inputs.allowed_paths),
                "answer_surface_gap": list(last_outer.answer_surface_gap),
                "bwrap_binary": str(bwrap_binary),
                "bwrap_binary_sha256": bwrap_sha256,
                "policy_sha256": _sha256(_canonical_json(list(last_outer.argv))),
            },
            "inner_execution_plane": {
                "engine": "bwrap", "network": last_inner.network,
                "clear_environment": last_inner.clear_environment, "credentials": "none",
                "candidate_writable": last_inner.candidate_writable,
                "corpus_read_only": last_inner.corpus_read_only,
                "answer_surface_gap": list(last_inner.answer_surface_gap),
                "required_tools": [probe.label for probe in TC.required_tool_probes(target_experiment)],
                "tool_probe_results": probe_results,
                "tool_probe_recheck_results": probe_recheck_results,
                "broker_calls": total_calls,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": frozen_functional.marker_sha256,
                "policy_sha256": _sha256(_canonical_json(list(last_inner.argv))),
            },
        },
        "broker": {
            "shim_mount": BROKER_NAME, "shim_sha256": _sha256(
                _BROKER_SHIM.encode("utf-8")),
            "registry": prepared_action_contract,
            "registry_sha256": _sha256(_canonical_json(prepared_action_contract)),
            "receipt_manifest": str(receipt_manifest),
            "receipt_manifest_sha256": _sha256_file(receipt_manifest),
            "round_receipts": receipt_records,
            "required_actions": sorted(action.name for action in prepared_actions if action.required),
            "all_required_succeeded": receipts_clean,
            "control_owned_by_harness": True,
            "control_writable_by_agent": False,
        },
        "agent": {
            "driver": "codex", "model": model, "resolved_model": resolved_model,
            "effort": effort,
            "codex_binary": str(codex_path), "codex_binary_sha256": codex_sha256,
            "wall_budget_seconds": wall_budget_seconds, "round_timeout_seconds": round_timeout_seconds,
            "max_tool_calls": max_tool_calls, "tool_timeout_seconds": tool_timeout_seconds,
            "rounds_requested": rounds, "rounds": round_records,
            "transcript": str(combined_transcript),
            "transcript_sha256": _sha256(combined_transcript.read_bytes()),
            "audit": combined_audit,
        },
        "telemetry": telemetry_record,
        "admission": {
            "consumable": consumable,
            "refusal": refusal,
            # Distinct from `refusal` on purpose: this names a run that ended because the search
            # reported it was finished, which must not be readable as a failure.
            "stopped_by": stopped_by,
            "development_feedback_performed_by_stage": True,
            "evaluation_performed_by_stage": False,
            "success_declared_by_stage": False,
            "consumer": "run_perf_bench.py",
        },
    }
    record_path = stage_root / "performance_candidate.json"
    _write_json(record_path, record)
    record_path.chmod(0o444)
    validate_candidate_record(record, require_consumable=consumable)
    if consumable:
        verify_candidate_record(
            record_path, verify_authoring_tools=True, target_experiment=target_experiment)
    return record_path


def main(argv: list[str] | None = None) -> int:
    # The sandbox gets this through sandbox_env, but the host lane imports candidate modules too --
    # the development-feedback evaluator runs capsule lowerings in-process, and that wrote a second
    # batch of caches into submission/mlir_oot/lowering/ nine minutes after the first. Set it here so
    # this process and every child it spawns inherit it, sandboxed or not.
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--functional-run-id", required=True)
    parser.add_argument("--functional-submission-sha256", required=True)
    parser.add_argument("--run-id", required=True,
                        help="fresh directory name under the Arm4 performance run root")
    parser.add_argument("--model", required=True, help="explicit Codex model slug")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--wall-budget-seconds", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument(
        "--replicates", type=int, default=None,
        help="optional assertion; must equal the frozen formal acceptance exact_count")
    parser.add_argument(
        "--smoke-replicates", type=int, default=1,
        help="non-claim diagnostic count; must be smaller than the formal cohort")
    parser.add_argument("--round-timeout-seconds", type=int, default=3600)
    parser.add_argument("--max-tool-calls", type=int, default=100)
    parser.add_argument("--tool-timeout-seconds", type=int, default=900)
    parser.add_argument("--families", default="all")
    parser.add_argument("--capsules", default="all")
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--gsim-certificate", type=Path, required=True)
    parser.add_argument("--gsim-certificate-sha256", required=True)
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--telemetry-price-table", type=Path, required=True)
    parser.add_argument("--descriptor", type=Path, default=(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"))
    args = parser.parse_args(argv)
    run_id = _safe_component(args.run_id, label="performance stage run id")
    target_experiment = load_target_experiment(args.descriptor)
    functional_root = runs_root(target_experiment.target, "capsule-bench")
    stage_root = runs_root(target_experiment.target, "perf-bench") / "agent_stages" / run_id
    try:
        record = run_stage(
            functional_runs_root=functional_root,
            functional_run_id=args.functional_run_id,
            functional_submission_sha256=args.functional_submission_sha256,
            target_experiment=target_experiment, stage_root=stage_root,
            model=args.model, effort=args.effort, wall_budget_seconds=args.wall_budget_seconds,
            rounds=args.rounds, round_timeout_seconds=args.round_timeout_seconds,
            replicates=args.replicates,
            smoke_replicates=args.smoke_replicates,
            max_tool_calls=args.max_tool_calls, tool_timeout_seconds=args.tool_timeout_seconds,
            families=args.families, capsules=args.capsules, codex_binary=args.codex_binary,
            gsim_certificate=args.gsim_certificate,
            gsim_certificate_sha256=args.gsim_certificate_sha256,
            rtl_facts=args.rtl_facts,
            telemetry_price_table=args.telemetry_price_table)
    except (StageGateError, PC.CampaignGateError) as exc:
        print(f"NO-GO: {exc}", file=sys.stderr)
        return 2
    document = json.loads(record.read_text(encoding="utf-8"))
    if document["admission"]["consumable"] is not True:
        print(f"NO-GO: {document['admission']['refusal']}\nrecord: {record}", file=sys.stderr)
        return 2
    print(f"SEALED: {record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
