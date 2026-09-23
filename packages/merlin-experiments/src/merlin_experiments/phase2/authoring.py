#!/usr/bin/env python3
"""Create one auditable performance candidate from an exact functional fork.

This stage deliberately does not grade or promote anything.  It gives Codex a fresh copy of a
functionally certified package and an answer-free view of the generated performance contracts.  Any
compiler/tool execution requested by the agent goes through a second, credential-free
bwrap broker.  When the bounded authoring rounds end, the candidate is copied to a read-only snapshot
and described by a content-addressed record for :mod:`run_paired_perf_bench` to consume.

There are two distinct filesystem and credential boundaries.  The outer Codex control plane gets one
isolated ``CODEX_HOME`` plus the explicit authentication mount and the functional run's frozen authoring
grants, but no live descriptor-derived target toolchain.  The inner execution plane gets that live
descriptor-derived toolchain and the writable candidate, but has ``--clearenv``
and receives no credential bind.  Network is available in both planes and is explicitly not claimed as
an isolation property; the experiment's protection comes from exact mounts, masks, and audited routes.
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes as _sha256
from merlin.perf.agent_guidance import inspect_compiler_package
from merlin.perf.execution_policy import (
    FIRESIM_LIFECYCLE,
    ITERATION_MAX_SECONDS,
)
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox import preflight as SANDBOX_PREFLIGHT
from merlin.targetgen.sandbox.answer_surfaces import dropped_declarations
from merlin.targetgen.target_experiment import TargetExperiment
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import broker as PB
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import candidate_record as RECORD
from merlin_experiments.phase2 import candidate_verification as VERIFY
from merlin_experiments.phase2 import contracts as CONTRACTS
from merlin_experiments.phase2 import corpus as CORPUS
from merlin_experiments.phase2 import development_feedback as DF
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import emission_diagnostics as ED
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2 import stage_prompt as SP
from merlin_experiments.phase2 import telemetry as TEL
from merlin_experiments.phase2.broker import (
    AGENT_CORPUS_MOUNT,
    BROKER_NAME,
    BrokerAction,
    action_registry_contract,
    stage_broker_shim,
)
from merlin_experiments.phase2.contracts import (
    ROUND_DEADLINE_EXIT,
    StageGateError,
)
from merlin_experiments.phase2.contracts import (
    canonical_json as _canonical_json,
)
from merlin_experiments.phase2.contracts import (
    require_executable as _require_executable,
)
from merlin_experiments.phase2.contracts import (
    sha256_file as _sha256_file,
)
from merlin_experiments.phase2.contracts import (
    write_json as _write_json,
)
from merlin_experiments.phase2.transcript_audit import audit_codex_transcript as audit_codex_transcript


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


def _require_stage_containment(bwrap_binary: Path, target_experiment: TargetExperiment):
    """Refuse a launch before output creation unless both containment claims hold."""
    try:
        sandbox_probe = SANDBOX_PREFLIGHT.require_working_sandbox(
            bwrap_binary, context="performance authoring bwrap operability preflight"
        )
    except SANDBOX_PREFLIGHT.SandboxUnavailable as exc:
        raise StageGateError(
            "the isolation sandbox cannot be built on this host, so no authoring round can run: " + exc.probe.describe()
        ) from exc
    declaration_drops = dropped_declarations(target_experiment)
    if declaration_drops:
        raise StageGateError(
            "declared answer-surface rules matched nothing and masked nothing: "
            + "; ".join(drop.describe() for drop in declaration_drops)
        )
    return sandbox_probe, declaration_drops


def assert_candidate_sealable(root: Path) -> None:
    """Keep the authoring/measurement digest domains identical."""
    excluded = {"build", "__pycache__", ".git"}
    for path in root.rglob("*"):
        if excluded & set(path.relative_to(root).parts):
            raise StageGateError(f"performance candidate retains digest-excluded ephemeral state: {path}")


def candidate_delta(base: Path, candidate: Path) -> dict[str, Any]:
    """Describe changed package bytes and reject documentation-only authoring as vacuous."""
    before, after = _tree_files(base), _tree_files(candidate)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    execution_relevant = [
        path for path in changed if "docs" not in Path(path).parts and Path(path).suffix.lower() != ".md"
    ]
    return {
        "changed_files": changed,
        "changed_file_count": len(changed),
        "execution_relevant_changed_files": execution_relevant,
        "execution_relevant_changed_file_count": len(execution_relevant),
    }


_GUARD_UNCHANGED = "unchanged"
_GUARD_CHANGED = "changed"


def functional_emission_guard(
    baseline: Path,
    candidate: Path,
    target_experiment: TargetExperiment,
    *,
    frozen_functional: FI.FrozenFunctionalInputs,
    timeout_s: int = 120,
) -> dict[str, Any]:
    """Prove which functional capsules the perf change CANNOT have affected, and scrutinise the rest.

    Phase 1 certified the BASELINE compiler on this corpus, and the perf stage never re-grades it. If
    the candidate emits byte-identical code for a capsule, that capsule's behaviour is unchanged by
    construction and no simulation can add information; only the capsules whose emission CHANGED carry
    functional risk. That is what makes a cheap guard sound rather than merely fast.

    Trace findings are compared DIFFERENTIALLY, never absolutely. ``trace_check.check`` documents its
    violations as advisory diagnostics that deliberately do not decide pass/fail (the oracle does), so
    gating on their presence would manufacture the false refusals this harness has already produced
    twice today. A finding the CERTIFIED baseline does not produce is a different claim: a regression
    this candidate introduced.
    """
    from merlin.targetgen import oot_runner as OR  # noqa: PLC0415
    from merlin.targetgen import trace_check as TCK  # noqa: PLC0415
    from merlin.targetgen.rocc import decode as RD  # noqa: PLC0415
    from merlin_experiments.phase1 import corpus_inputs

    if frozen_functional.host_provenance is None:
        raise StageGateError("functional emission guard requires admitted V4 host provenance")
    verified = FI._verify_private_functional_provenance(frozen_functional.host_provenance)
    environment_path = Path(frozen_functional.host_provenance["environment"])
    environment = CONTRACTS.mapping_file(environment_path, yaml_file=True)
    # The grading view is a declared host input, not an agent-visible grant. Reuse
    # Phase 1's frozen resolver; never rediscover live descriptor siblings or mix
    # hidden answers and performance-development corpora into its public cohort.
    try:
        view = corpus_inputs.resolve(
            verified["workspace"],
            verified["bundle"],
            environment.get("public_corpus_input"),
            repo=verified["repo"],
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise StageGateError(f"functional public corpus is unavailable: {exc}") from exc
    if _sha256_file(environment_path) != frozen_functional.host_provenance["environment_sha256"]:
        raise StageGateError("functional host provenance changed during corpus selection")
    capsules = sorted(path.parent for path in view.public.rglob("capsule.yaml"))
    if not capsules:
        raise StageGateError("functional frozen public corpus contains no capsules")

    base_pkg, cand_pkg = OR.load_package(Path(baseline)), OR.load_package(Path(candidate))
    rows: list[dict[str, Any]] = []
    offenders: list[dict[str, Any]] = []
    for capsule_dir in capsules:
        name = capsule_dir.name
        try:
            descriptor = CONTRACTS.mapping_file(capsule_dir / "capsule.yaml", yaml_file=True)
        except Exception:  # noqa: BLE001 - an unreadable descriptor is reported, never skipped silently
            offenders.append({"capsule": name, "kind": "descriptor_unreadable"})
            continue
        interface = capsule_dir / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
        if not interface.is_file():
            rows.append({"capsule": name, "status": "no_interface"})
            offenders.append({"capsule": name, "kind": "interface_missing"})
            continue
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or None) as raw:
            scratch = Path(raw)
            base_rc, base_llvm, base_buffer = EA.emit_pair(base_pkg, interface, scratch, "baseline", timeout_s)
            cand_rc, cand_llvm, cand_buffer = EA.emit_pair(cand_pkg, interface, scratch, "candidate", timeout_s)
        if base_rc != 0:
            offenders.append(
                {"capsule": name, "kind": "baseline_emission_failed", "baseline_rc": base_rc, "candidate_rc": cand_rc}
            )
            continue
        if base_rc == 0 and cand_rc != 0:
            offenders.append(
                {"capsule": name, "kind": "lowering_regressed", "baseline_rc": base_rc, "candidate_rc": cand_rc}
            )
            continue
        if _sha256(base_llvm.encode("utf-8")) == _sha256(cand_llvm.encode("utf-8")) and _sha256(
            base_buffer.encode("utf-8")
        ) == _sha256(cand_buffer.encode("utf-8")):
            rows.append({"capsule": name, "status": _GUARD_UNCHANGED})
            continue
        row: dict[str, Any] = {"capsule": name, "status": _GUARD_CHANGED}
        try:
            expected = descriptor.get("expected") or {}
            base_trace = RD.decode_text(base_llvm, source="baseline", target=target_experiment.target)
            cand_trace = RD.decode_text(cand_llvm, source="candidate", target=target_experiment.target)
            base_findings = set(
                TCK.check(base_trace, expected, json.loads(base_buffer) if base_buffer else None)["violations"]
            )
            cand_findings = set(
                TCK.check(cand_trace, expected, json.loads(cand_buffer) if cand_buffer else None)["violations"]
            )
            introduced = sorted(cand_findings - base_findings)
            row["introduced_findings"] = introduced
            row["drives_accelerator"] = [
                bool(TCK.drives_accelerator(base_trace)),
                bool(TCK.drives_accelerator(cand_trace)),
            ]
            if row["drives_accelerator"] == [True, False]:
                offenders.append({"capsule": name, "kind": "accelerator_dispatch_regressed"})
            if introduced:
                offenders.append({"capsule": name, "kind": "trace_findings_introduced", "findings": introduced[:8]})
        except Exception as exc:  # noqa: BLE001 - an undecodable candidate trace is absence of proof
            row["decode_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            offenders.append({"capsule": name, "kind": "trace_not_decodable"})
        rows.append(row)

    unchanged = sum(1 for row in rows if row.get("status") == _GUARD_UNCHANGED)
    changed = sum(1 for row in rows if row.get("status") == _GUARD_CHANGED)
    return {
        "status": "clean" if not offenders else "offending",
        "capsules": len(rows),
        "proved_unchanged": unchanged,
        "changed": changed,
        "offenders": offenders,
        "rows": rows,
    }


#: Fields an unmeasured cell carries. It is the same shape as a measured one so the redacted
#: schema stays exact, with every measurement-valued field null -- an absent number is never a zero.


def _import_codex_driver():
    from merlin_experiments.phase1.providers import codex_agent  # noqa: PLC0415

    return codex_agent


def authored_round_status(*, agent_exit_code: int, audit_clean: object, refusals: Sequence[str]) -> dict[str, Any]:
    """Whether one authoring round counts as authored, and WHY -- phase 1's rule, stated once.

    THE DEFECT THIS REPLACES. The global campaign admitted a round only on ``rc == 0``, so a round
    the harness killed at its own declared deadline was ``refused``; the sequence then raised, the
    checkpoint was re-consumed, and **the round's compiler edits were discarded**. Measured on the
    v15 run: 2 of 3 paid rounds retained nothing and both restarted from the initial seed candidate.
    Phase 1 has never behaved that way -- it admits the same exit as ``stopped_by`` on exactly the
    evidence a clean round already had to produce (``perf_agent_stage`` round loop, ``rc ==
    ROUND_DEADLINE_EXIT and refusal is None and audit["clean"]``).

    A budget is not a crash. The deadline exit is admitted ONLY alongside a clean audit and no
    refusals, so every other non-zero exit stays refused: a segfault, an out-of-memory kill and a
    driver error are all still failures, and none of them is a declared budget being spent.

    Returns the status and the reason rather than a bare boolean, because "refused" and "authored at
    the deadline" are different results and a report that shows them as the same one is how the last
    campaign's three rounds looked identical.
    """
    clean = audit_clean is True
    reasons = list(refusals or ())
    exit_code = int(agent_exit_code)
    if exit_code == 0 and clean and not reasons:
        return {
            "status": "authored",
            "stopped_by": None,
            "why": "the round completed with a clean audit and no refusals",
        }
    if exit_code == ROUND_DEADLINE_EXIT and clean and not reasons:
        return {
            "status": "authored",
            "stopped_by": "round_deadline",
            "why": (
                "the round reached its declared deadline with a clean audit and no "
                "refusals; the budget is the run's declared size and spending it is the "
                "expected end of a search that did not converge first"
            ),
        }
    return {
        "status": "refused",
        "stopped_by": None,
        "why": (f"the round exited {exit_code} with audit_clean={clean} and {len(reasons)} refusal(s)"),
    }


def _codex_round(
    workspace: Path,
    stage_root: Path,
    prompt: SP.PromptArtifact,
    target_experiment: TargetExperiment,
    agent_inputs: AW.AgentInputSnapshot,
    frozen_functional: FI.FrozenFunctionalInputs,
    functional_base: Path,
    frozen_corpus_manifest: Path,
    control_dir: Path,
    *,
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs,
    model: str,
    resolved_model: str,
    effort: str,
    round_index: int,
    timeout_s: int,
    codex_binary: Path,
) -> tuple[int, Path, AW.AgentSandboxPolicy]:
    if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
        raise StageGateError("Codex authoring requires an explicit sandbox input selection")
    CA = _import_codex_driver()
    captured: dict[str, AW.AgentSandboxPolicy] = {}

    def exact_bwrap(inner: str, ws: Path, _bundle: dict, extra_binds: list[str] | None = None) -> str:
        policy = AW.outer_codex_policy(
            ws,
            agent_inputs,
            extra_binds or (),
            target_experiment,
            frozen_functional,
            functional_base,
            control_dir,
            frozen_corpus_manifest,
            inputs=sandbox_inputs,
        )
        captured["policy"] = policy
        # ``inner`` is already shell-quoted by codex_agent.  Quote only the outer payload boundary.
        return BW.compose_command(list(policy.argv), " bash -c '" + inner.replace("'", "'\\''") + "'", ws)

    rc, transcript = CA.run_round(
        workspace,
        stage_root,
        model,
        {},
        target_experiment,
        "bwrap",
        round_index,
        timeout_s,
        effort=effort,
        prompt=prompt.text,
        effective_model=resolved_model,
        sandbox_command=exact_bwrap,
        codex_binary=codex_binary,
        codex_home_root=stage_root / "codex_homes",
    )
    policy = captured.get("policy")
    if policy is None:
        raise StageGateError("Codex driver did not construct the required outer bwrap policy")
    return rc, transcript, policy


def run_stage(
    *,
    suite: str,
    contract_root: Path,
    source_root: Path,
    functional_runs_root: Path,
    functional_run_id: str,
    functional_submission_sha256: str,
    target_experiment: TargetExperiment,
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs,
    stage_root: Path,
    model: str,
    effort: str,
    wall_budget_seconds: int,
    rounds: int,
    round_timeout_seconds: int,
    max_tool_calls: int,
    tool_timeout_seconds: int,
    replicates: int | None = None,
    smoke_replicates: int = 1,
    families: str = "all",
    capsules: str = "all",
    codex_binary: str = "codex",
    gsim_certificate: Path | None = None,
    gsim_certificate_sha256: str | None = None,
    rtl_facts: Path | None = None,
    telemetry_price_table: Path | None = None,
    waive_functional_gate: tuple[str, ...] = (),
) -> Path:
    """Run bounded authoring rounds and return the sealed candidate-record path."""
    if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
        raise StageGateError("performance authoring requires an explicit sandbox input selection")
    sandbox_inputs = copy.deepcopy(sandbox_inputs)
    if not isinstance(suite, str) or not suite.strip():
        raise StageGateError("an explicit telemetry suite is required")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (
            wall_budget_seconds,
            rounds,
            round_timeout_seconds,
            max_tool_calls,
            tool_timeout_seconds,
            smoke_replicates,
        )
    ):
        raise StageGateError("all performance stage budgets must be positive integers")
    if tool_timeout_seconds > ITERATION_MAX_SECONDS:
        raise StageGateError(
            f"tool_timeout_seconds exceeds the {ITERATION_MAX_SECONDS:g}s reduced-witness "
            "iteration limit; reduce the witness"
        )
    if not model.strip():
        raise StageGateError("an explicit Codex model is required")
    bwrap_binary = _require_executable("bwrap", label="bwrap")
    sandbox_probe, declaration_drops = _require_stage_containment(bwrap_binary, target_experiment)
    codex_path = _require_executable(codex_binary, label="Codex")
    descriptor_path = Path(target_experiment.path).resolve()
    bwrap_sha256 = _sha256_file(bwrap_binary)
    codex_sha256 = _sha256_file(codex_path)
    descriptor_sha256 = _sha256_file(descriptor_path)
    telemetry_preflight_record = TEL.prepare(
        authoring_stage=Path(__file__).resolve(),
        model=model,
        price_table=telemetry_price_table,
        codex_binary=codex_path,
    )
    resolved_model = str((telemetry_preflight_record.get("model_resolution") or {}).get("resolved_model") or "")
    if not resolved_model:
        raise StageGateError("telemetry preflight omitted the resolved Codex model")
    if bwrap_binary.name != "bwrap":
        raise StageGateError("the sandbox executable does not resolve to bwrap")
    raw_stage_root = Path(stage_root)
    if raw_stage_root.exists() or raw_stage_root.is_symlink():
        raise StageGateError(f"performance agent stage must use a fresh directory: {raw_stage_root}")
    stage_root = raw_stage_root.resolve()

    # THE SAME WAIVERS THE COORDINATOR APPLIED. Without this the coordinator admits the functional
    # baseline and each trial then re-checks it un-waived and refuses -- the campaign dies at the
    # first stage having already passed its own preflight. Measured 2026-09-06: all 3 trials exited
    # rc=2 on the nine completeness predicates the launch had explicitly waived. `waive` is passed
    # through to `perf_campaign.inspect_functional_run`, which alone decides what is waivable at
    # all -- an integrity predicate refuses the waiver itself, here exactly as in the coordinator.
    functional = FI.inspect_stage_functional_run(
        functional_runs_root,
        functional_run_id,
        functional_submission_sha256,
        waive=frozenset(waive_functional_gate or ()),
    )
    discovered = CORPUS.discover_performance_corpus(target_experiment, families=families, capsules=capsules)
    stage_root.mkdir(parents=True)
    base = PC.materialize_perf_workspace(functional, stage_root / "_frozen_functional")
    frozen_corpus = CORPUS.freeze_performance_corpus(discovered, stage_root / "_frozen_corpus")
    formal_claim = RECORD.prepare_formal_claim(frozen_corpus.capsules, replicates)
    replicates = len(RECORD.preflight_cohort(formal_claim))
    agent_inputs = AW.build_answer_free_agent_inputs(frozen_corpus, target_experiment, stage_root / "_agent_inputs")
    frozen_functional = FI.load_frozen_functional_inputs(
        functional, public_manifest_path=stage_root / "_public_functional_snapshot.json"
    )
    prepared_actions = BP.action_registry(BP.CORPUS_FEEDBACK_V1, base, target_experiment)
    prepared_action_contract = action_registry_contract(prepared_actions, base)
    minimum_calls = rounds * sum(action.required for action in prepared_actions)
    if max_tool_calls < minimum_calls:
        raise StageGateError(
            "inner tool-call budget cannot cover every required broker action in every round: "
            f"need at least {minimum_calls}, got {max_tool_calls}"
        )
    feedback = DF.prepare_development_feedback(
        contract_root=contract_root,
        certificate_path=gsim_certificate,
        certificate_sha256=gsim_certificate_sha256,
        rtl_facts_path=rtl_facts,
        corpus=frozen_corpus,
        baseline=base,
        baseline_sha256=functional.digest,
        target_experiment=target_experiment,
        work_root=stage_root / "_development_feedback",
        # Every tuning measurement spends one inner tool call, so the run's own call budget is the
        # real bound on how many the search can take. Declaring it makes ``budget_exhausted`` report
        # true remaining spend instead of "unbounded".
        tuning_call_budget=max_tool_calls,
        functional_run_dir=functional.run_dir,
    )
    prompt_inputs = INPUTS.prepare_prompt_inputs(
        functional,
        frozen_functional,
        frozen_corpus,
        agent_inputs,
        target_experiment,
        prepared_actions,
        source_root=source_root,
        formal_claim=formal_claim,
        smoke_replicates=smoke_replicates,
        wall_budget_seconds=wall_budget_seconds,
        rounds=rounds,
        round_timeout_seconds=round_timeout_seconds,
        max_tool_calls=max_tool_calls,
        tool_timeout_seconds=tool_timeout_seconds,
    )
    staged_prompt = stage_root / "prompt.txt"
    prompt = SP.materialize_canonical_prompt(prompt_inputs, staged_prompt)
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
    last_outer: AW.AgentSandboxPolicy | None = None
    refusal: str | None = None
    stopped_by: dict[str, Any] | None = None
    total_calls = 0
    receipt_records: list[dict[str, Any]] = []
    last_actions: tuple[BrokerAction, ...] = ()
    last_inner: AW.AgentSandboxPolicy | None = None
    for round_index in range(rounds):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            # Spending the declared wall budget is the run finishing, not failing. This wrote into
            # the refusal channel, so the launcher's claim that the wall check "ends the run
            # cleanly" was true for no value of `rounds`. A round that already sealed a candidate
            # keeps it; with nothing sealed yet there is nothing to admit and it stays a refusal.
            if round_records:
                stopped_by = {
                    "conditions": ["wall_budget"],
                    "round": round_index,
                    "reason": "the stage spent its declared wall-clock budget",
                }
            else:
                refusal = "performance stage wall-clock budget expired before the first round"
            break
        workspace = stage_root / "agent_workspaces" / f"round_{round_index:02d}"
        candidate = AW.fresh_round_workspace(previous_submission, workspace, previous_digest)
        (workspace / "TASK.md").write_text(prompt.text, encoding="utf-8")
        optimization_inventory = inspect_compiler_package(candidate, contract=contract_root).to_dict()
        initial_whole_model_analysis = EA.analyze_whole_model_emission(
            base,
            candidate,
            prompt_inputs.e2e_sentinel,
            contract_root=contract_root,
            timeout_s=min(tool_timeout_seconds, remaining),
            peak_macs_per_cycle=getattr(feedback, "peak_macs_per_cycle", None),
            achievable_macs_per_cycle=getattr(feedback, "achievable_macs_per_cycle", None),
            target=str(getattr(target_experiment, "target", "") or ""),
        )
        profile_selector = getattr(feedback, "profile_witness", None)
        if callable(profile_selector):
            profile_member, profile_selection = profile_selector()
            reduced_global_profile = {
                "status": "ready",
                "broker_action": BP.OCCUPANCY_PROFILE_ACTION,
                "family": profile_member.family,
                "capsule": profile_member.capsule,
                "selection": profile_selection,
                "selected_before_candidate_measurement": True,
                "purpose": (
                    "calibrate complete-model occupancy, movement, and latency hiding; "
                    "never a measured whole-model performance result"
                ),
            }
        else:
            # Compatibility for test/dry-run feedback providers. Production preparation always
            # supplies DF.DevelopmentGsimFeedback and therefore a frozen selector.
            reduced_global_profile = {
                "status": "UNKNOWN",
                "broker_action": BP.OCCUPANCY_PROFILE_ACTION,
                "family": None,
                "capsule": None,
                "selection": None,
                "selected_before_candidate_measurement": None,
                "purpose": "feedback provider exposes no fixed reduced global witness",
            }
        _write_json(
            workspace / "STAGE_CONTEXT.json",
            {
                "functional_run_id": functional.run_id,
                "functional_submission_sha256": functional.digest,
                "performance_manifest_sha256": frozen_corpus.manifest_sha256,
                "performance_corpus_sha256": frozen_corpus.capsules_sha256,
                "agent_corpus_mount": str(AGENT_CORPUS_MOUNT),
                "candidate": "submission",
                "tool_command": f"python3 {BROKER_NAME} ACTION [NAME=VALUE ...]",
                "broker_actions": [action.name for action in prepared_actions],
                "formal_replicates": replicates,
                "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
                "smoke_replicates": smoke_replicates,
                "round": round_index,
                "rounds": rounds,
                "remaining_wall_budget_seconds": remaining,
                "automatic_optimization_inventory": optimization_inventory,
                "initial_whole_model_analysis": initial_whole_model_analysis,
                "reduced_global_profile": reduced_global_profile,
                "iteration_measurement_contract": {
                    "maximum_simulator_seconds": int(ITERATION_MAX_SECONDS),
                    "warmup_runs": 1,
                    "measured_runs": 1,
                    "primary_metric": "total_compute_cycles",
                    "allowed_explanatory_metrics": [
                        "resource_busy_cycles",
                        "movement_bytes",
                        "movement_commands",
                        "movement_compute_overlap_cycles",
                        "overlap_available_cycles",
                        "latency_hiding_efficiency",
                        "encoding_transitions",
                    ],
                    "full_size_execution": "optional_post_freeze_validation_not_phase2_required",
                    "firesim_required": False,
                    "firesim_queue_operation": "runworkload-full",
                    "firesim_lifecycle": [" ".join(command) for command in FIRESIM_LIFECYCLE],
                },
                "expensive_measurement_budget": {
                    "scope": "per_round",
                    "tuning_gsim_feedback_calls": BP.EXPENSIVE_ACTION_LIMITS[BP.DEVELOPMENT_FEEDBACK_ACTION],
                    "exploratory_tuning_calls": 1,
                    "reserved_final_byte_tuning_calls": 1,
                    "reduced_occupancy_profile_calls": BP.EXPENSIVE_ACTION_LIMITS[BP.OCCUPANCY_PROFILE_ACTION],
                    "firesim_calls": 0,
                    "free_iteration_actions": [BP.E2E_ANALYSIS_ACTION, BP.ANALYSIS_ACTION, BP.INVENTORY_ACTION],
                },
            },
        )
        actions = BP.action_registry(BP.CORPUS_FEEDBACK_V1, candidate, target_experiment)
        if action_registry_contract(actions, candidate) != prepared_action_contract:
            refusal = f"candidate manifest action contract drifted before round {round_index}"
            break
        inner = AW.inner_execution_policy(
            target_experiment,
            candidate,
            agent_inputs,
            frozen_functional,
            base,
            frozen_corpus.manifest_path,
            inputs=sandbox_inputs,
        )
        if probe_results is None:
            probe_results = AW.run_required_tool_probes(inner, target_experiment, candidate)
        control_dir = stage_root / "control" / f"round_{round_index:02d}"
        receipt_path = control_dir / "receipts.jsonl"
        broker = PB.Broker(
            inner,
            target_experiment,
            candidate,
            actions,
            receipt_path,
            deadline=deadline,
            max_calls=max_tool_calls - total_calls,
            max_tool_seconds=tool_timeout_seconds,
            workflow=BP.select_workflow(
                BP.CORPUS_FEEDBACK_V1,
                candidate=candidate,
                target_experiment=target_experiment,
                receipt_path=receipt_path,
                services=BP.BrokerServices(
                    whole_model_analysis=partial(EA.analyze_whole_model_emission, contract_root=contract_root),
                    command_buffer_analysis=ED.analyze_command_buffers,
                ),
                feedback_evaluator=feedback,
                feedback_round=round_index,
                functional_base=base,
                e2e_sentinel=prompt_inputs.e2e_sentinel,
            ),
        )
        round_timeout = min(round_timeout_seconds, remaining)
        try:
            with broker.serving() as (host, port):
                stage_broker_shim(
                    control_dir,
                    host=host,
                    port=port,
                    token=broker.token,
                    tool_timeout_s=tool_timeout_seconds,
                    actions=actions,
                )
                rc, transcript, outer = _codex_round(
                    workspace,
                    stage_root,
                    prompt,
                    target_experiment,
                    agent_inputs,
                    frozen_functional,
                    base,
                    frozen_corpus.manifest_path,
                    control_dir,
                    sandbox_inputs=sandbox_inputs,
                    model=model,
                    resolved_model=resolved_model,
                    effort=effort,
                    round_index=round_index,
                    timeout_s=round_timeout,
                    codex_binary=codex_path,
                )
        finally:
            broker_config = control_dir / ".perf_broker.json"
            if broker_config.is_file() and not broker_config.is_symlink():
                broker_config.chmod(0o600)
                broker_config.unlink()
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt_path.chmod(0o444)
        total_calls += len(broker.calls)
        transcript_paths.append(transcript)
        round_telemetry = TEL.collect_round(
            stage_root,
            round_index,
            model=resolved_model,
            agent_exit_code=rc,
            preflight_record=telemetry_preflight_record,
        )
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
            receipt_evidence = broker.workflow.verify_receipts(receipt_path, actions, audit, candidate_sha256=observed)
        except StageGateError as exc:
            receipt_evidence = {"path": str(receipt_path), "error": str(exc), "all_required_succeeded": False}
            refusal = f"Codex round {round_index} failed broker receipt enforcement: {exc}"
        if not any(
            call.get("action") == BP.DEVELOPMENT_FEEDBACK_ACTION and call.get("returncode") == 0
            for call in broker.calls
        ):
            refusal = f"Codex round {round_index} did not successfully invoke mandatory tuning GSIM feedback"
        receipt_records.append(receipt_evidence)
        audit_path = stage_root / "rounds" / f"round_{round_index:02d}.audit.json"
        _write_json(audit_path, audit)
        round_record = {
            "round": round_index,
            "workspace": str(workspace),
            "candidate_sha256": observed,
            "agent_exit_code": rc,
            "transcript": str(transcript),
            "transcript_sha256": _sha256(transcript.read_bytes()),
            "audit": audit,
            "telemetry": round_telemetry,
            "broker_calls": broker.calls,
            "broker_receipts": receipt_evidence,
            "broker_registry_sha256": _sha256(_canonical_json([action.as_dict() for action in actions])),
            "budget_seconds": round_timeout,
        }
        round_records.append(round_record)
        last_outer = outer
        last_inner = inner
        last_actions = actions
        previous_submission, previous_digest = candidate, observed
        # A ROUND THAT SPENT ITS BUDGET IS FINISHED, NOT BROKEN -- provided it left the evidence a
        # finished round leaves. Treating the deadline as a crash discards everything: a session
        # that ran eleven hours, measured the corpus repeatedly and sealed a valid candidate was
        # thrown away whole, because the same branch handled a spawn failure, a crash and an expiry.
        # The budget is the run's declared size; reaching it is the expected end of a search that
        # did not converge first, and the post-freeze formal grade still decides the verdict.
        #
        # It is admitted ONLY on the evidence a clean round already had to produce: a clean
        # answer/tool-access audit, a sealable candidate, and at least one successful mandatory
        # feedback call -- all three are checked above and any of them failing has already written
        # `refusal`. Anything else non-zero stays a refusal, because a crash is not a budget.
        if rc == ROUND_DEADLINE_EXIT and refusal is None and audit["clean"]:
            stopped_by = {
                "conditions": ["round_deadline"],
                "round": round_index,
                "reason": (
                    "the round reached its declared deadline with a clean audit "
                    "and a sealed candidate; the search did not converge first"
                ),
            }
            break
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
        if getattr(broker.workflow, "stop_verdict", None) is not None:
            stopped_by = dict(broker.workflow.stop_verdict)
            stopped_by["round"] = round_index
            break

    if (
        not round_records
        or not transcript_paths
        or last_outer is None
        or last_inner is None
        or probe_results is None
        or not last_actions
    ):
        raise StageGateError(refusal or "performance stage completed no auditable Codex round")
    probe_recheck_results: list[dict[str, Any]] = []
    try:
        probe_recheck_results = AW.run_required_tool_probes(last_inner, target_experiment, previous_submission)
        if probe_recheck_results != probe_results:
            refusal = "inner sandbox tool probe evidence changed during performance authoring"
    except StageGateError as exc:
        refusal = f"inner sandbox tool recheck failed: {exc}"
    if (
        _sha256_file(bwrap_binary) != bwrap_sha256
        or _sha256_file(codex_path) != codex_sha256
        or _sha256_file(descriptor_path) != descriptor_sha256
    ):
        refusal = "descriptor or agent/sandbox executable bytes changed during performance authoring"
    CORPUS.verify_frozen_performance_corpus(frozen_corpus)
    AW.verify_answer_free_agent_inputs(agent_inputs)
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
    # A round that reached its declared deadline counts as a clean exit only when the loop above
    # classified it that way -- which it does only with a clean audit and a sealed candidate, and
    # which it records in `stopped_by`. Without that, the deadline is still a refusal.
    _admitted_exits = {0, ROUND_DEADLINE_EXIT} if (stopped_by or {}).get("conditions") == ["round_deadline"] else {0}
    exits_clean = all(row["agent_exit_code"] in _admitted_exits for row in round_records)
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
    telemetry_record = TEL.finalize(
        stage_root,
        round_records,
        model=resolved_model,
        target=target_experiment.target,
        suite=suite,
        run_id=stage_root.name,
        preflight_record=telemetry_preflight_record,
    )
    combined_audit = audit_codex_transcript(
        combined_transcript, target_experiment, previous_submission, prepared_actions
    )
    round_audits_clean = all(row["audit"]["clean"] and row["audit"]["commands_seen"] > 0 for row in round_records)
    audits_clean = round_audits_clean and combined_audit["clean"]
    if combined_audit["commands_seen"] <= 0:
        refusal = "combined Codex transcript contains zero command evidence"
    elif not combined_audit["clean"]:
        refusal = "combined Codex transcript failed the answer/tool-access audit"
    # A converged run has FEWER round records than `rounds`, by design. Requiring exact equality
    # would mark the search unconsumable for having finished early, which is the outcome the stop
    # rule exists to produce.
    expected_rounds = len(round_records) if stopped_by is not None else rounds
    receipts_clean = len(receipt_records) == expected_rounds and all(
        row.get("all_required_succeeded") is True and row.get("feedback_successes", 0) >= 1 for row in receipt_records
    )
    if not receipts_clean:
        refusal = refusal or "required host-owned broker receipt evidence is incomplete"
    # THE FUNCTIONAL GUARD. Phase 1 certified the baseline on the functional corpus and this stage
    # never re-grades it, so without this a candidate could pass every performance cell while breaking
    # capsules nothing here executes. It is cheap because it is a proof, not a sample: a capsule whose
    # emitted code is byte-identical cannot have changed behaviour.
    try:
        functional_guard = functional_emission_guard(
            base, previous_submission, target_experiment, frozen_functional=frozen_functional
        )
    except Exception as exc:  # noqa: BLE001 - an unrunnable guard is absence of proof, not a pass
        functional_guard = {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {str(exc)[:200]}",
            "offenders": [],
            "rows": [],
        }
    if functional_guard.get("status") != "clean":
        kinds = sorted({str(row.get("kind")) for row in functional_guard.get("offenders") or ()})
        refusal = refusal or (
            "performance candidate did not clear the certified functional emission guard "
            f"({functional_guard.get('status')}" + (f": {', '.join(kinds)}" if kinds else "") + ")"
        )
    functional_guard_clean = functional_guard.get("status") == "clean"
    consumable = (
        refusal is None
        and audits_clean
        and exits_clean
        and receipts_clean
        and functional_guard_clean
        and len(round_records) == expected_rounds
        and len(round_records) >= 1
    )
    receipt_manifest = stage_root / "control" / "receipt_manifest.json"
    _write_json(receipt_manifest, {"schema_version": 1, "rounds": receipt_records})
    receipt_manifest.chmod(0o444)
    expected_cells = [
        {"family": cell.family, "capsule": cell.capsule, "simulator": cell.simulator, "replicate": cell.replicate}
        for cell in prompt_inputs.expected_cells
    ]
    family_facts = [
        {
            "family": family.family,
            "claim": family.claim,
            "negative_control": family.negative_control,
            "falsifier_observation": family.falsifier_observation,
            "differential_basis": family.differential_basis,
            "fitted_parameters": list(family.fitted_parameters),
            "acceptance": copy.deepcopy(family.acceptance),
        }
        for family in prompt_inputs.families
    ]
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
    frozen_grants = [
        {
            "declared_path": grant.declared_path,
            "destination": str(grant.destination),
            "source": str(grant.source),
            "source_sha256": grant.source_sha256,
        }
        for grant in frozen_functional.grants
    ]
    prompt_facts = {
        "replicates": replicates,
        "formal_replicate_identities": list(prompt_inputs.formal_replicate_identities),
        "formal_claim": copy.deepcopy(formal_claim),
        "smoke_replicates": smoke_replicates,
        "expected_cells": expected_cells,
        "budgets": {
            "wall_budget_seconds": wall_budget_seconds,
            "rounds": rounds,
            "round_timeout_seconds": round_timeout_seconds,
            "max_tool_calls": max_tool_calls,
            "tool_timeout_seconds": tool_timeout_seconds,
        },
        "families": family_facts,
        "host_lane": host_lane,
        "e2e_sentinel": e2e_sentinel,
        "tools": prepared_action_contract,
        "mount_destinations": list(prompt_inputs.allowed_paths),
    }
    record = {
        "schema_version": RECORD.SCHEMA_VERSION,
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
            "run_id": functional.run_id,
            "submission_sha256": functional.digest,
            "snapshot": str(base),
            "fork_before": fork_check.to_dict(),
            "fork_after": after.to_dict(),
            "bundle_input_snapshot": {
                "path": str(frozen_functional.root),
                "content_sha256": frozen_functional.content_sha256,
                "manifest": str(frozen_functional.marker),
                "manifest_sha256": frozen_functional.marker_sha256,
                "grants": frozen_grants,
                **(
                    {
                        "host_provenance": frozen_functional.host_provenance,
                        "public_manifest": str(frozen_functional.public_marker),
                        "public_manifest_sha256": frozen_functional.public_marker_sha256,
                        "public_content_sha256": frozen_functional.public_content_sha256,
                    }
                    if frozen_functional.host_provenance is not None
                    else {}
                ),
            },
            "model_host_lane": model_host_record,
            "e2e_sentinel": e2e_sentinel,
        },
        "candidate": {
            "path": str(sealed),
            "initial_sha256": functional.digest,
            "sha256": sealed_sha,
            "rounds_completed": len(round_records),
            "read_only": True,
            "base_submission_overwritten": False,
            "delta": delta,
        },
        "prompt": {
            "renderer_path": str(Path(__file__).resolve()),
            "renderer_sha256": _sha256_file(Path(__file__).resolve()),
            "staged_path": str(staged_prompt),
            "sha256": prompt.sha256,
            "n_bytes": prompt.n_bytes,
            "facts": prompt_facts,
            "facts_sha256": _sha256(_canonical_json(prompt_facts)),
        },
        "performance_corpus": {
            "path": str(frozen_corpus.root),
            "manifest": str(frozen_corpus.manifest_path),
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
            "action": BP.DEVELOPMENT_FEEDBACK_ACTION,
            "required_per_round": True,
            "scope": "frozen_tuning_corpus_only",
            "engine": "gsim",
            "certificate": feedback.certificate.to_dict(),
            "rtl_identity": copy.deepcopy(dict(feedback.rtl_identity)),
            "redaction": "correctness_gsim_cycles_and_paired_deltas_only",
            "round_receipts": [row.get("feedback_receipts", []) for row in receipt_records],
        },
        "sandbox": {
            "preflight": sandbox_probe.as_record(),
            "dropped_declarations": [drop.as_record() for drop in declaration_drops],
            "outer_codex_control_plane": {
                "engine": "bwrap",
                "network": last_outer.network,
                "clear_environment": last_outer.clear_environment,
                "auth_exception": "isolated_codex_home_explicit_auth_mount",
                "session_history_mounted": False,
                "live_target_toolchain_mounted": False,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": (
                    frozen_functional.public_marker_sha256 or frozen_functional.marker_sha256
                ),
                "mount_destinations": list(prompt_inputs.allowed_paths),
                "answer_surface_gap": list(last_outer.answer_surface_gap),
                "bwrap_binary": str(bwrap_binary),
                "bwrap_binary_sha256": bwrap_sha256,
                "policy_sha256": _sha256(_canonical_json(list(last_outer.argv))),
            },
            "inner_execution_plane": {
                "engine": "bwrap",
                "network": last_inner.network,
                "clear_environment": last_inner.clear_environment,
                "credentials": "none",
                "candidate_writable": last_inner.candidate_writable,
                "corpus_read_only": last_inner.corpus_read_only,
                "answer_surface_gap": list(last_inner.answer_surface_gap),
                "required_tools": [probe.label for probe in last_inner.required_tools],
                "tool_probe_results": probe_results,
                "tool_probe_recheck_results": probe_recheck_results,
                "broker_calls": total_calls,
                "frozen_functional_grants_mounted": True,
                "frozen_grant_manifest_sha256": (
                    frozen_functional.public_marker_sha256 or frozen_functional.marker_sha256
                ),
                "policy_sha256": _sha256(_canonical_json(list(last_inner.argv))),
            },
        },
        "broker": {
            "shim_mount": BROKER_NAME,
            "shim_sha256": _sha256(PB._BROKER_SHIM.encode("utf-8")),
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
            "driver": "codex",
            "model": model,
            "resolved_model": resolved_model,
            "effort": effort,
            "codex_binary": str(codex_path),
            "codex_binary_sha256": codex_sha256,
            "wall_budget_seconds": wall_budget_seconds,
            "round_timeout_seconds": round_timeout_seconds,
            "max_tool_calls": max_tool_calls,
            "tool_timeout_seconds": tool_timeout_seconds,
            "rounds_requested": rounds,
            "rounds": round_records,
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
            "consumer": RECORD.MEASUREMENT_CONSUMER,
        },
    }
    RECORD.verify_sandbox_containment_evidence(record["sandbox"])
    record_path = stage_root / "performance_candidate.json"
    _write_json(record_path, record)
    record_path.chmod(0o444)
    RECORD.validate_candidate_record(record, require_consumable=consumable)
    if consumable:
        VERIFY.verify_candidate_record(record_path, verify_authoring_tools=True, target_experiment=target_experiment)
    return record_path
