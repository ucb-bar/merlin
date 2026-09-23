"""Concrete portfolio authoring rounds and checkpoint-driven continuation.

Scientific admission remains with the experiment's installed evidence owners.
Codex transport, broker serving and telemetry reuse their existing implementations.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.perf.agent_guidance import inspect_compiler_package
from merlin.perf.execution_policy import GLOBAL_AUTHORING_ROUND_MAX_SECONDS, ITERATION_MAX_SECONDS
from merlin.targetgen.target_experiment import TargetExperiment

from . import agent_view as AV
from . import agent_workspace as AW
from . import authoring as AUTHORING
from . import broker as PB
from . import broker_policy as BP
from . import campaign as PC
from . import contracts as P2_CONTRACTS
from . import emission_analysis as EA
from . import functional_inputs as FI
from . import host_policy as HP
from . import portfolio_checkpoint as CHECKPOINT
from . import stage_prompt as SP
from . import telemetry as TEL
from . import transcript_audit as TA
from .global_experiment import GlobalPerfExperiment
from .portfolio_sandbox import PortfolioSandboxFactory


def _retryable_capacity_failure(stage_root: Path, round_index: int, terminal: Mapping[str, Any]) -> bool:
    """Recognize a host-recorded capacity refusal that did no candidate work.

    This is a continuation decision, never an acceptance: the next round starts from the last
    consumed checkpoint. Keep the classifier narrow so arbitrary agent/tool failures cannot bypass
    the required-action gate.
    """
    if (
        terminal.get("agent_exit_code", 0) == 0
        or terminal.get("audit", {}).get("clean") is not True
        or terminal.get("audit", {}).get("broker_invocations") not in (None, [])
    ):
        return False
    path = stage_root / "rounds" / f"round_{round_index:02d}.codex_summary.json"
    if path.is_symlink() or not path.is_file():
        return False
    try:
        summary = P2_CONTRACTS.mapping_file(path)
    except (OSError, ValueError):
        return False
    if (
        summary.get("exit_code") != terminal.get("agent_exit_code")
        or summary.get("timed_out") is not False
        or summary.get("turns_started") != 1
        or summary.get("turns_usage_reported") != 0
    ):
        return False
    errors = summary.get("errors")
    return (
        isinstance(errors, list)
        and bool(errors)
        and all(isinstance(value, str) and "selected model is at capacity" in value.lower() for value in errors)
    )


def _retryable_unchanged_round_failure(terminal: Mapping[str, Any], *, checkpoint_sha256: str) -> bool:
    """Continue after a clean no-change round whose mandatory edit analysis never ran.

    This admits no candidate: the next round is reconstructed from the already consumed checkpoint.
    It only prevents a correctly reverted late proposal from terminating the whole bounded sequence.
    A host-recorded timeout (124) is eligible only with a clean transcript; missing mandatory
    analysis remains a refused round, not fabricated evidence for a new candidate. The sequence
    separately revalidates its source policy and consumes the previous checkpoint before resuming.
    """
    broker = terminal.get("broker_evidence") or {}
    audit = terminal.get("audit") or {}
    reason = broker.get("reason")
    hits = audit.get("hits")
    clean_or_recorded_command_error = audit.get("clean") is True or (
        isinstance(hits, list)
        and bool(hits)
        and all(isinstance(row, Mapping) and row.get("kind") == "invalid_broker_invocation" for row in hits)
        and broker.get("all_required_succeeded") is True
    )
    broker_failure_is_bounded = broker.get("all_required_succeeded") is True or (
        broker.get("status") == "refused"
        and isinstance(reason, str)
        and reason.startswith("global broker required actions did not complete:")
    )
    eligible_exit = terminal.get("agent_exit_code") == 0 or (
        terminal.get("agent_exit_code") == 124 and audit.get("clean") is True
    )
    return (
        eligible_exit
        and clean_or_recorded_command_error
        and terminal.get("candidate_sha256") == checkpoint_sha256
        and broker_failure_is_bounded
    )


def _ready_portfolio_members(row: Mapping[str, Any]) -> frozenset[str]:
    """Exact objective identities whose current candidate analysis is promotion-ready."""
    members = (row.get("portfolio") or {}).get("members") or ()
    return frozenset(
        str((member.get("identity") or {}).get("capsule_sha256"))
        for member in members
        if (member.get("readiness") or {}).get("status") == "ready_for_probe_admission"
    )


def _prior_round_context(stage_root: Path, round_index: int) -> dict[str, Any]:
    """Expose prior agents' own bounded summaries without treating them as host evidence."""
    rows = []
    for index in range(round_index):
        final_path = stage_root / "rounds" / f"round_{index:02d}.final.txt"
        audit_path = stage_root / "global_iterations" / f"agent_round_{index:04d}.json"
        row: dict[str, Any] = {"round": index}
        if final_path.exists() or final_path.is_symlink():
            if final_path.is_symlink() or not final_path.is_file():
                raise ValueError(f"prior round final is not a regular file: {final_path}")
            payload = final_path.read_bytes()
            row["agent_summary_sha256"] = hashlib.sha256(payload).hexdigest()
            row["agent_summary"] = payload.decode("utf-8") if len(payload) <= 64 * 1024 else None
            row["agent_summary_omitted"] = len(payload) > 64 * 1024
        if audit_path.exists() or audit_path.is_symlink():
            audit = P2_CONTRACTS.mapping_file(audit_path)
            row["host_round_audit"] = {
                key: copy.deepcopy(audit.get(key))
                for key in ("status", "candidate_sha256", "agent_exit_code", "refusal_reasons")
            }
            row["host_round_audit_sha256"] = P2_CONTRACTS.sha256_file(audit_path)
        if len(row) > 1:
            rows.append(row)
    return {
        "schema": "global_prior_round_context_v1",
        "rounds": rows,
        "interpretation": (
            "agent summaries are untrusted search memory; host round audits are "
            "status evidence; neither admits a candidate or proves a speedup"
        ),
    }


def _agent_finalization_reserve_seconds(round_timeout_s: int) -> int:
    """Close the tool broker before the outer turn deadline so Codex can emit final telemetry."""
    if round_timeout_s < 120:
        return 0
    if round_timeout_s < 300:
        return 30
    # A refused last analysis can still require a source revert, a compact evidence summary, and
    # Codex's own final telemetry flush.  Two minutes proved insufficient in a real 600-second
    # authoring round: the agent reverted at the boundary, emitted its final message, and was killed
    # before the transport wrote the final artifact.  Reserve three minutes at the maximum round
    # size; this changes only when tools close, never which candidate can be accepted.
    return min(180, max(60, round_timeout_s // 3))


class PortfolioAuthoring:
    """Own ordered authoring, exact-byte validation and budgeted recovery."""

    def __init__(
        self,
        experiment: GlobalPerfExperiment,
        *,
        target_experiment: TargetExperiment,
        stage_root: Path,
        agent_inputs: AW.AgentInputSnapshot,
        frozen_functional: FI.FrozenFunctionalInputs,
        frozen_corpus_manifest: Path,
        sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs,
        model: str,
        resolved_model: str,
        effort: str,
        codex_binary: Path,
        max_tool_calls: int,
        declared_instruction_evidence: Mapping[str, Any],
        global_probe_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_semantic_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_context_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_paired_context_provider: Callable[..., Mapping[str, Any]] | None = None,
        global_source_pair_provider: Callable[..., Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
            raise ValueError("portfolio authoring requires explicit selected sandbox inputs")
        self.experiment = experiment
        self.target_experiment = target_experiment
        self.stage_root = Path(stage_root).absolute()
        self.agent_inputs = agent_inputs
        self.frozen_functional = frozen_functional
        self.frozen_corpus_manifest = Path(frozen_corpus_manifest).absolute()
        self.model = model
        self.resolved_model = resolved_model
        self.effort = effort
        self.codex_binary = Path(codex_binary).absolute()
        self.max_tool_calls = max_tool_calls
        self.global_probe_provider = global_probe_provider
        self.global_semantic_provider = global_semantic_provider
        self.global_context_provider = global_context_provider
        self.global_paired_context_provider = global_paired_context_provider
        self.global_source_pair_provider = global_source_pair_provider
        self._sandbox_inputs = copy.deepcopy(sandbox_inputs)
        self.declared_instruction_evidence = copy.deepcopy(dict(declared_instruction_evidence))

    def configure_analysis(self) -> None:
        if self.experiment.analysis.analyzer is not EA.analyze_whole_model_emission:
            return
        PortfolioSandboxFactory(
            self.experiment.analysis,
            target_experiment=self.target_experiment,
            agent_inputs=self.agent_inputs,
            frozen_functional=self.frozen_functional,
            frozen_corpus_manifest=self.frozen_corpus_manifest,
            sandbox_inputs=self._sandbox_inputs,
        ).install_worker(output=self.stage_root / "host_analysis_workers")

    def _consume_checkpoint(self, path: Path) -> dict[str, Any]:
        inputs = self.experiment.inputs
        return CHECKPOINT.consume_round_checkpoint(
            path,
            context=CHECKPOINT.CheckpointVerificationContext(
                host_policy=HP.build_record(
                    controller_source=inputs.controller_source, contract_root=inputs.contract_root
                ),
                compiler_shared_source_root=inputs.compiler_shared_source_root,
            ),
        )

    def run_sequence(
        self,
        candidate: Path,
        *,
        max_rounds: int,
        total_authoring_seconds: int,
        round_seconds: int = 600,
        on_round_failure: str = "stop",
    ) -> dict[str, Any]:
        """Bounded sustained search: exact consumed checkpoints, no silent live-handle retries.

        Authoring budget is reserved per invocation, not reconstructed from optimistic agent
        telemetry. Setup/analysis wall time is separately retained by each iteration. A checkpoint
        certifies structural/provenance review, never a measured global improvement.
        """
        experiment = self.experiment
        stage_root = self.stage_root
        if (
            min(max_rounds, total_authoring_seconds, round_seconds) <= 0
            or round_seconds > GLOBAL_AUTHORING_ROUND_MAX_SECONDS
            or on_round_failure not in ("stop", "resume-last-checkpoint")
        ):
            raise ValueError("invalid sustained global authoring bounds or continuation policy")
        experiment.revision_session.check_inputs()
        policy = copy.deepcopy(experiment.inputs.host_policy)
        initial_analysis = experiment.analysis.analyze(
            candidate, hypothesis="Bind initial seed for safe checkpoint continuation"
        )
        if experiment.mechanism_program.work_order_binding is not None:
            experiment.analysis.record_bound_mechanism_work_order_seed(candidate, initial_analysis)
        initial_row = experiment.revision_session.current(candidate, require_ready=False)
        initial_ready = initial_row["readiness"]["status"] == "ready_for_probe_admission"
        initial_seal = (
            experiment.revision_session.seal(candidate, name="initial_seed_candidate")
            if initial_ready
            else experiment.revision_session.checkpoint_authoring(candidate, name="initial_seed_authoring")
        )
        initial = self._consume_checkpoint(
            initial_seal,
        )
        spent, failures = 0, []
        checkpoints = [
            {
                "round": -1,
                "role": (
                    "initial_verified_seed_not_an_authored_result"
                    if initial_ready
                    else "initial_blocked_authoring_seed"
                ),
                "path": str(initial_seal),
                "sha256": P2_CONTRACTS.sha256_file(initial_seal),
                "candidate_sha256": initial["candidate_sha256"],
                "checkpoint_schema": initial["schema"],
                "promotion_ready": initial_ready,
            }
        ]
        current = candidate
        for index in range(max_rounds):
            budget = min(round_seconds, total_authoring_seconds - spent)
            if budget <= 0:
                break
            experiment.revision_session.check_inputs()
            if experiment.inputs.host_policy != policy:
                raise ValueError("sustained segment source policy changed")
            if index > 0:
                checkpoint = self._consume_checkpoint(
                    Path(checkpoints[-1]["path"]),
                )
                current = AW.fresh_round_workspace(
                    Path(checkpoint["candidate_path"]),
                    stage_root / "agent_workspaces" / f"round_{index:02d}",
                    checkpoint["candidate_sha256"],
                )
            ready_before = _ready_portfolio_members(experiment.revision_session.current(current, require_ready=False))
            spent += budget
            try:
                authored = self.run_round(current, round_index=index, round_timeout_s=budget)
                if authored.get("status") != "authored":
                    raise ValueError("round runner did not return an authored audited checkpoint")
                row = experiment.revision_session.current(current, require_ready=False)
                lost = sorted(ready_before - _ready_portfolio_members(row))
                if lost:
                    raise ValueError("round regressed previously verified portfolio members: " + ", ".join(lost))
                ready = row["readiness"]["status"] == "ready_for_probe_admission"
                # A final revision whose executed program failed the model's gate is not sealable,
                # but an EARLIER passing revision of this round still is: fall through to the
                # selection below with the failed row excluded, so the round keeps its best valid
                # work instead of checkpointing a miscompile or discarding the round.
                gate_failed = bool((row.get("functional_gate") or {}).get("excludes_candidate"))
                # Seal the BEST authored revision of this campaign, not merely the last one the agent
                # happened to leave in its workspace. The next round resumes from whatever is sealed, so
                # sealing the last revision discards every improvement an agent found and then moved off
                # -- observed on every round of two runs. `seal` resolves its row by CONTENT hash and
                # copies that row's preserved snapshot, so handing it the winning snapshot seals exactly
                # the bytes that earned the analysis.
                selection = experiment.analysis.best_authored_candidate() if (ready or gate_failed) else None
                if gate_failed and selection is not None:
                    ready = True  # the earlier passing revision is what gets sealed, never `row`
                seal_from = current
                if selection is not None and selection["candidate_sha256"] != row["candidate_sha256"]:
                    seal_from = Path(selection["snapshot"])
                    experiment._write(
                        f"round_{index:04d}_selection.json",
                        {
                            "schema": "global_round_candidate_selection_v1",
                            "reason": "sealed the cheapest ready revision, not the final one",
                            "selected": selection,
                            "final_revision": {
                                "iteration": row["iteration"],
                                "candidate_sha256": row["candidate_sha256"],
                                "host_cost": experiment.analysis.authored_host_cost(row.get("analysis")),
                                "functional_gate": row.get("functional_gate"),
                            },
                            "cost_metric": "host-lane (load+store) payload bytes, then static allocations",
                        },
                    )
                sealed = (
                    experiment.revision_session.seal(seal_from, name=f"round_{index:04d}_candidate")
                    if ready
                    else experiment.revision_session.checkpoint_authoring(current, name=f"round_{index:04d}_authoring")
                )
                consumed = self._consume_checkpoint(
                    sealed,
                )
                checkpoint = {
                    "round": index,
                    "path": str(sealed),
                    "sha256": P2_CONTRACTS.sha256_file(sealed),
                    "candidate_sha256": consumed["candidate_sha256"],
                    "checkpoint_schema": consumed["schema"],
                    "promotion_ready": ready,
                }
                checkpoints.append(checkpoint)
                experiment._write(
                    f"continuation_{index:04d}.json",
                    {
                        "schema": "global_round_continuation_v1",
                        "status": "checkpoint_consumed",
                        "checkpoint": checkpoint,
                        "authoring_seconds_reserved": spent,
                        "promotion_ready": ready,
                        "global_speedup_proven": False,
                    },
                )
            except Exception as exc:
                failure = {
                    "round": index,
                    "exception": type(exc).__name__,
                    "reason": str(exc),
                    "draft_path": str(current),
                    "draft_sha256": hash_tree(current)["sha256"],
                    "last_good_checkpoint": checkpoints[-1] if checkpoints else None,
                    "authoring_seconds_reserved": spent,
                    "live_handle_restarted": False,
                }
                # Never classify arbitrary provenance/audit errors as recoverable author failures.
                # The production round writes its terminal audit before raising on a nonzero exit.
                audit_path = experiment.output / f"agent_round_{index:04d}.json"
                terminal = P2_CONTRACTS.mapping_file(audit_path) if audit_path.is_file() else {}
                capacity_retry = _retryable_capacity_failure(stage_root, index, terminal)
                unchanged_retry = bool(checkpoints) and _retryable_unchanged_round_failure(
                    terminal, checkpoint_sha256=checkpoints[-1]["candidate_sha256"]
                )
                recoverable = (
                    on_round_failure == "resume-last-checkpoint"
                    and bool(checkpoints)
                    and (
                        unchanged_retry
                        or (
                            terminal.get("agent_exit_code", 0) != 0
                            and terminal.get("audit", {}).get("clean") is True
                            and (
                                terminal.get("broker_evidence", {}).get("all_required_succeeded") is True
                                or capacity_retry
                            )
                        )
                    )
                )
                failure["retryable_capacity_failure"] = capacity_retry
                failure["retryable_unchanged_round_failure"] = unchanged_retry
                try:
                    experiment.revision_session.check_inputs()
                    if checkpoints:
                        self._consume_checkpoint(
                            Path(checkpoints[-1]["path"]),
                        )
                except Exception:
                    recoverable = False
                failure["recovery"] = "next_budgeted_round_from_consumed_checkpoint" if recoverable else "stop"
                failures.append(failure)
                experiment._write(f"continuation_failure_{index:04d}.json", failure)
                if not recoverable:
                    raise
        result = {
            "schema": "global_agent_sequence_v1",
            "status": "budget_complete",
            "authoring_seconds_reserved": spent,
            "maximum_rounds": max_rounds,
            "on_round_failure": on_round_failure,
            "checkpoints": checkpoints,
            "failures": failures,
            "candidate": str(current),
            "last_good_checkpoint": checkpoints[-1] if checkpoints else None,
            "promotion_ready": bool(checkpoints and checkpoints[-1]["promotion_ready"]),
            "host_verification_policy": policy,
            "global_speedup_proven": False,
        }
        experiment._write("agent_sequence.json", result)
        return result

    def run_round(self, candidate: Path, *, round_index: int, round_timeout_s: int) -> dict[str, Any]:
        """Run a real sandboxed authoring round with the full graph as the mandatory objective.

        The caller prepares the existing frozen answer-free grants and explicit model/budgets. This
        reuses the production Codex transport, credential-free tool sandbox, transcript audit and
        telemetry, but never routes the result to the PR/PQ/PK microbenchmark consumer.
        """
        experiment = self.experiment
        workspace = candidate.parent
        sandbox_inputs = self._sandbox_inputs
        target_experiment = self.target_experiment
        stage_root = self.stage_root
        agent_inputs = self.agent_inputs
        frozen_functional = self.frozen_functional
        frozen_corpus_manifest = self.frozen_corpus_manifest
        model = self.model
        resolved_model = self.resolved_model
        effort = self.effort
        codex_binary = self.codex_binary
        max_tool_calls = self.max_tool_calls
        global_probe_provider = self.global_probe_provider
        global_semantic_provider = self.global_semantic_provider
        global_context_provider = self.global_context_provider
        global_paired_context_provider = self.global_paired_context_provider
        global_source_pair_provider = self.global_source_pair_provider
        AW.verify_answer_free_agent_inputs(agent_inputs)
        if experiment.inputs.phase1_binding is None:
            raise ValueError("paid macro authoring requires exact existing Phase-1 qualification and waivers")
        if experiment.edit_authority.contract is None:
            raise ValueError("paid macro authoring requires a host-frozen compiler edit authority")
        experiment.revision_session.validate_candidate_scope(candidate)
        if candidate.parent.resolve() != workspace.resolve():
            raise ValueError("the macro candidate must live in its isolated agent workspace")
        if min(round_timeout_s, max_tool_calls) <= 0:
            raise ValueError("macro agent round budgets must be positive")
        # WHAT THIS RUN CAN ACTUALLY ANSWER. Derived once, from the providers this round was given, and
        # used for all three of: the registry the agent is shown, the stage context it reads, and the
        # broker that will serve it. See `BP.ACTION_PROVIDER_REQUIREMENTS` for the run in which those
        # three disagreed and the agent spent its only measured-feedback call on a refusal.
        round_providers = {
            "global_probe_provider": global_probe_provider,
            "global_semantic_provider": global_semantic_provider,
            "global_context_provider": global_context_provider,
            "global_paired_context_provider": global_paired_context_provider,
            "global_source_pair_provider": global_source_pair_provider,
        }
        unnamed = [a for _n, a, _r in BP.ACTION_PROVIDER_REQUIREMENTS if a not in round_providers]
        if unnamed:
            # LOUD, not fail-closed-and-quiet. A provider the table requires and this round never names
            # would be reported unavailable while the broker holds it -- the two would disagree, and the
            # disagreement is the defect. Refuse here, where the message names the missing wiring.
            raise ValueError(
                f"this round carries no provider slot for {unnamed}, which the broker action requirement table declares"
            )
        unavailable_actions = BP.unavailable_global_actions(round_providers)
        actions = BP.action_registry(BP.WHOLE_MODEL_V1, candidate, target_experiment, unavailable=unavailable_actions)
        inner = AW.inner_execution_policy(
            target_experiment,
            candidate,
            agent_inputs,
            frozen_functional,
            experiment.inputs.baseline,
            frozen_corpus_manifest,
            inputs=sandbox_inputs,
        )
        AW.run_required_tool_probes(inner, target_experiment, candidate)
        self.configure_analysis()
        mechanism_round_start = experiment.revision_session.begin_mechanism_round(candidate, round_index=round_index)
        initial = experiment.analysis.analyze(candidate, hypothesis="Inspect the current complete-model global plan")
        mechanism_work_order_analysis = experiment.analysis.bind_mechanism_work_order_analysis(initial)
        finalization_reserve_s = _agent_finalization_reserve_seconds(round_timeout_s)
        broker_window_s = round_timeout_s - finalization_reserve_s
        # The complete portfolio can legitimately take longer than the entire Codex round.  It cannot
        # therefore be a mandatory synchronous broker action: at a 600-second round the final-response
        # reserve closes that broker after 420 seconds.  Keep the agent and every interactive tool under
        # the declared round bound, then give the exact submitted bytes their own host-only static
        # validation phase after the Codex process exits.  This adds no executable/path authority and
        # does not change the analyzer's memory admission or no-full-model-simulation policy.
        authoring_tool_window_s = broker_window_s
        post_authoring_validation_contract = {
            "schema": "host_post_authoring_static_validation_v1",
            "maximum_seconds": experiment.analysis.timeout_s,
            "execution": "host_after_codex_process_exit",
            "candidate_binding": "host_read_only_snapshot_of_exact_submitted_bytes",
            "broker_invocation_required": False,
            "full_model_simulation_allowed": False,
            "resource_admission": "unchanged_portfolio_analysis_policy",
        }
        # Retain the old context key for readers that render budget tables.  Its zero is material: no
        # part of the separately bounded host validation is borrowed from the authoring/tool window.
        mandatory_analysis_reserve = {
            "schema": "in_round_mandatory_analysis_reserve_v2",
            "seconds": 0,
            "scope": "none; superseded by host_post_authoring_validation",
        }
        prior_round_context = _prior_round_context(stage_root, round_index)
        portfolio_names = ", ".join(member.capsule for member in experiment.inputs.portfolio_sentinels)
        text = (
            "Optimize the fixed portfolio of real full-model graphs and their global plans. "
            f"The training portfolio is: {portfolio_names}. Read STAGE_CONTEXT.json. "
            "It contains a concise initial view; INITIAL_FULL_MODEL_EVIDENCE.json contains the complete "
            "unpruned graph and immutable analysis copy when a transformation needs those details. "
            "Start with portfolio_action_digest: it resolves the primary and secondary member records "
            "into one per-model readiness, work, movement, dispatch, placement and authorized-action view. "
            "When fast_accuracy_bounded_evaluation is configured, use its per-model conservative cycle "
            "intervals, physical movement, occupancy, overlap, encoding conversion, supported-work "
            "placement, connected-region, host-island and boundary evidence. Its recommended levers name "
            "only matching host-authorized surfaces. A larger accelerator op count is not a benefit by "
            "itself: the gate requires a quality-safe global benefit and rejects added movement, boundary "
            "traffic, occupancy loss, excessive uncertainty or a regression in any portfolio member. "
            "The initial exact analysis is already available there; unchanged reanalysis reuses it. "
            "For continuation rounds, prior_round_context in STAGE_CONTEXT.json contains earlier agents' "
            "own untrusted summaries plus host refusal status. Use it as search memory and do not repeat "
            "a disproved or unfinished hypothesis; it is not correctness or performance evidence. "
            "The optimization_surfaces_schema there is the authoritative Phase-2 optional manifest "
            "extension (the frozen Phase-1 schema predates it). Scope must be flag, knob, heuristic, "
            "pass, or codegen, not cca; specify path and exact AST symbol plus every required field. "
            "The host_frozen_edit_authority is the edit permission contract: edit only its exact AST "
            "symbols and explicitly listed helper-extension directories. Candidate manifest entries "
            "describe changes but cannot authorize additional files or symbols. Preserve manifest "
            "execution controls. When host_frozen_mechanism_catalog is present it is machine enforced: "
            "every semantic compiler edit in this round must map to exactly one catalog mechanism ID, "
            "and an unchanged or formatting-only submission is recorded as a refused no-op. "
            "When host_frozen_mechanism_work_order is present it is the only executable assignment: "
            "use its exact per-member graph-local source-operation IDs and chain inventory, and preserve "
            "its candidate, catalog, portfolio, source, plan, command-buffer and lowered-artifact hashes. "
            "Do not infer, substitute or add sites; stop if the assigned evidence does not match the "
            "current full-graph analysis. The work order grants no additional edit authority. "
            "Imports in approved owning files remain subject to the masked shared "
            "dependency policy. If a needed compiler lever has no approved owner, report that specific "
            "missing surface instead of expanding your own authority. Before editing, state a compact "
            "work order using the contract's required fields: surface/source-operation IDs, current "
            "plan digest, hypothesis, expected emitted delta, semantic obligations, cheap validation "
            "and stop/revert condition. Execute exactly one coherent optimization mechanism per round; "
            "it may span the target-general paths and models required to implement that mechanism, but "
            "must not include opportunistic unrelated edits. Finish it, or record its refusal/no-op and "
            "stopping condition, before attempting another mechanism. Follow the host-owned "
            "optimization_order in portfolio_action_digest from lowest numbered tier to highest: repair "
            "regressions, then delete whole-program work and boundaries, then optimize global dataflow, "
            "representation and residency, then global issue/overlap/synchronization, and only then "
            "operator, tile, or local scalar cleanup. A higher tier may be closed only by a retained "
            "structural change or an explicit source/plan-bound refusal or no-op for the current "
            "revision. Do not choose a smaller easy rewrite while a higher-tier mechanism has a "
            "quantified dynamic extent and an authorized edit surface. analyze-whole-model remains "
            "available for optional in-round "
            "screening when the remaining broker window can cover it. The host automatically snapshots "
            "and recompiles the exact submitted bytes after the Codex process exits, with the separate "
            "full-graph static-analysis budget; do not spend the final response window waiting for it. "
            "Preserve verified emission for every member that is already ready and repair explicitly "
            "blocked members; every member must be verified before promotion or measurement. Prefer "
            "transformations that "
            "improve several model families or remove a shared global bottleneck; do not specialize a "
            "compiler rule to any capsule or model name. Compare each model only with its own prior "
            "revision and use a Pareto decision; never sum unlike models into a fabricated cycle score. "
            "Use full-graph work, movement, representation, residency, synchronization and dispatch "
            "accounting to select transformations. Preserve every source operation and dependency; "
            "capture parsing alone is not candidate emission. Unknown cost remains UNKNOWN. "
            "Short mechanism-equivalent probes may calibrate uncertain costs only through "
            "profile-reduced-global-witness with host-derived identity/equivalence admission. "
            "Exactly one warm invocation precedes exactly one measured compute-cycle invocation. "
            "profile-controlled-context preserves a bounded actual queued-load source prefix and "
            "reports controlled occupancy, never full-model equivalence or global cost calibration. "
            "compare-controlled-context compares identical bounded work between the prior and current "
            "schedule; it is optional and never projects a full-model speedup. Device profiles are "
            "not required for host-only edits or models. Follow host_memory_hotspot evidence to the "
            "exact allocation/buffer identity and declared compiler surface; source attribution may "
            "still be UNKNOWN. Host semantic qualification prioritizes a changed dequantization-to-"
            "contraction mechanism when present, then supported fanout/pointwise mechanisms. "
            "Paired measurements automatically return decision_feedback joined to the exact model "
            "region and compiler surface; use measurement_driven_next_step when present. A prior "
            "revision's measured feedback is search history, not calibration for newly edited bytes. "
            "Use qualify-changed-region for host-selected semantic witnesses of an actual changed "
            "source region when available; this tests reduced mechanisms, not a full-model rerun. "
            "For host-to-convolution lowering changes, prepare-source-convolution requires exactly "
            "comparison_arm=optimization_baseline or comparison_arm=previous. It uses cached full-source "
            "proofs, compiles both reduced source programs, and returns allowed edit surfaces. "
            "Preparation never simulates, admits a runtime, or grants a numerical pass. "
            "For a contraction implementation change, prepare-source-contraction requires comparison_arm, "
            "source_op_index, max_m, max_n and max_k (decimal source index and positive reduced bounds). "
            "It preserves the source scalar semantics and actual input/initializer ABI; use its returned "
            "preparation_sha256 with qualify-source-contraction only if the complete-source-pair provider "
            "is installed. Each action has its own total 60-second budget. The pair executes complete "
            "reduced programs, not a full layer/model; successful short outputs or cycle differences do "
            "not prove the selected full-model task changed via the same route. Respect explicit UNKNOWN "
            "route relevance and never substitute an unrelated primitive or host-chain pass. "
            "Never simulate a complete layer or model during search; FireSim belongs after freeze. "
            "There is no mandatory micro GSIM sweep or micro plateau stopping rule. Preserve the "
            "host-pinned Phase-1 baseline and its recorded waivers; do not run Phase 1 again. Do not modify "
            "harnesses or evaluators. Reuse generalized compiler algorithms and target-derived facts. "
            f"Execute compiler/tools only through python3 {PB.BROKER_NAME} ACTION [NAME=VALUE ...]. "
            "Do not run Python directly against any path in the candidate workspace, even for "
            "read-only parsing, imports, AST checks or manifest inspection; use jq, sed or rg for "
            "read-only inspection and use the declared broker action for compiler execution. "
            "Broker commands must stand alone: do not pipe them to jq, redirect, or chain them. "
            "Do not place shell or Python commands before or after a broker call in the same command. "
            "analyze-whole-model, qualify-changed-region, inspect-optimization-surfaces and "
            "profile-reduced-global-witness, profile-controlled-context and compare-controlled-context "
            "accept NO NAME=VALUE bindings; do not add HYPOTHESIS=. "
            "The response is compact and links a read-only full evidence file; inspect that file "
            "separately with jq when detailed fields are needed. The host post-authoring full-model "
            "analysis is mandatory; an agent broker invocation and individual candidate entrypoint "
            "smoke commands are optional in macro mode. "
            "At round end state the full-graph transformation, structural evidence, unknown costs, "
            "and remaining semantic/promotion blockers; do not claim measured full-model speedup. "
            f"The complete broker closes after {broker_window_s} seconds, leaving "
            f"{finalization_reserve_s} seconds for the final response. Emit that response before the "
            "round deadline; a valid intermediate edit does not make a timed-out round complete. After "
            f"a clean round exits, the host gives the submitted bytes up to {experiment.analysis.timeout_s:g} "
            "additional seconds for compile-only whole-portfolio validation. That host phase is outside "
            "the authoring and broker deadlines and cannot be invoked or redirected by the agent.\n"
        )
        declared_isa = copy.deepcopy(self.declared_instruction_evidence)
        text += AV.declared_instruction_prompt(declared_isa)
        text += AV.unavailable_action_notice(unavailable_actions)
        text += (
            f"Phase-1 qualification compiler SHA: {experiment.inputs.baseline_sha256}. "
            f"Immutable optimization comparison compiler SHA: {experiment.inputs.optimization_baseline_sha256}. "
            f"Comparison selection reason: {experiment.inputs.optimization_baseline_binding['reason']}. "
            "The comparison seed does not replace or extend Phase-1 qualification; correctness "
            "of these complete-model objectives remains UNPROVEN.\n"
        )
        if experiment.inputs.historical_reference is not None:
            text += (
                "A host-pinned public historical reference bundle is available read-only at "
                "/perf-control/historical_reference.json; its compact coverage and missing contracts "
                "are in the optimization brief. These are historical engine-relative references, "
                "not warm calibration, hardware peaks or target-cycle authority. Use them to choose "
                "relevant short probes, not to claim full-model speedup.\n"
            )
        external_objectives = []
        for member in experiment.inputs.portfolio_sentinels:
            objective_record_path = Path(member.frozen_source_path) / "objective.json"
            if objective_record_path.is_file():
                record = P2_CONTRACTS.mapping_file(objective_record_path)
                if record.get("schema") == "external_full_model_objective_v1":
                    external_objectives.append(record)
                    text += (
                        "One portfolio member is a separately host-pinned external full-model "
                        "objective, not an addition to frozen Phase-1 qualification. Its numeric "
                        f"correctness is UNPROVEN. Inspect its normalized source under {member.capsule_path}; "
                        "do not load capture weights, references or execute normalization scripts.\n"
                    )
        prompt_path = workspace / "TASK.md"
        prompt_path.write_text(text)
        prompt = SP.PromptArtifact(prompt_path, text, sha256_bytes(text.encode()), len(text.encode()))
        P2_CONTRACTS.write_json(workspace / "INITIAL_FULL_MODEL_EVIDENCE.json", initial)
        initial_view = AV.agent_analysis_view(
            initial,
            complete_evidence="INITIAL_FULL_MODEL_EVIDENCE.json",
            context_provider_installed=global_context_provider is not None,
        )
        action_digest = AV.portfolio_action_digest(
            initial,
            complete_evidence="INITIAL_FULL_MODEL_EVIDENCE.json",
            edit_contract=experiment.edit_authority.contract,
        )
        P2_CONTRACTS.write_json(
            workspace / "STAGE_CONTEXT.json",
            {
                "mode": "global_perf_experiment_v1",
                "initial_whole_model_analysis": initial_view,
                "portfolio_action_digest": action_digest,
                "prior_round_context": prior_round_context,
                "host_frozen_edit_authority": experiment.edit_authority.binding
                if experiment.edit_authority.contract is not None
                else None,
                "host_frozen_mechanism_catalog": copy.deepcopy(experiment.mechanism_program.catalog_binding),
                "host_frozen_mechanism_work_order": copy.deepcopy(experiment.mechanism_program.work_order_binding),
                "mechanism_work_order_analysis": copy.deepcopy(mechanism_work_order_analysis),
                "mechanism_round_start": copy.deepcopy(mechanism_round_start),
                "automatic_optimization_inventory": inspect_compiler_package(
                    candidate,
                    contract=experiment.inputs.contract_root,
                ).to_dict(),
                "optimization_surfaces_schema": P2_CONTRACTS.mapping_file(
                    experiment.inputs.contract_root / "schemas/manifest.schema.json"
                )["properties"]["optimization_surfaces"],
                "candidate": str(candidate),
                # ADVERTISED, not just sealed: each row carries whether this run can answer it.
                "broker_actions": [action.advertised() for action in actions],
                "phase1_sha256": experiment.inputs.baseline_sha256,
                "model_sha256": experiment.inputs.sentinel.capsule_sha256,
                "optimization_baseline_sha256": experiment.inputs.optimization_baseline_sha256,
                "optimization_baseline": experiment.inputs.optimization_baseline_binding,
                "full_model_portfolio": experiment.inputs.portfolio_identity,
                "full_model_portfolio_sha256": experiment.inputs.portfolio_identity_sha256,
                "portfolio_source_paths": [member.capsule_path for member in experiment.inputs.portfolio_sentinels],
                "objective_source_path": experiment.inputs.sentinel.capsule_path,
                "external_objective": external_objectives[0] if external_objectives else None,
                "external_objectives": external_objectives,
                "maximum_iteration_seconds": experiment.analysis.timeout_s,
                "maximum_full_graph_static_analysis_seconds": experiment.analysis.timeout_s,
                "maximum_reduced_witness_seconds": int(ITERATION_MAX_SECONDS),
                "maximum_round_seconds": round_timeout_s,
                "maximum_tool_window_seconds": broker_window_s,
                "maximum_non_analysis_tool_window_seconds": authoring_tool_window_s,
                "mandatory_analysis_reserve": mandatory_analysis_reserve,
                "host_post_authoring_validation": post_authoring_validation_contract,
                "finalization_reserve_seconds": finalization_reserve_s,
                # THE MACHINE, not just what this program emitted from it. Every `broker_actions` entry now
                # also carries `available` / `unavailable_reason` from the same table, so a reader cannot
                # find an action advertised plainly beside a flag below saying its provider is absent.
                "declared_instruction_set": declared_isa,
                "unavailable_broker_actions": dict(sorted(unavailable_actions.items())),
                # The per-iteration cycle floor under the structural dispatch delta; see
                # `ED.iteration_cost_plane`. `incomplete` here is a STATUS, never a pass.
                "cost_plane": copy.deepcopy(
                    ((initial_view.get("analysis") or {}).get("diagnostics") or {}).get("cost_plane")
                ),
                "probes_available": global_probe_provider is not None,
                "changed_region_qualification_available": global_semantic_provider is not None,
                "controlled_context_provider_installed": global_context_provider is not None,
                "controlled_context_profile_available": initial_view["controlled_context_capability"]["available"],
                "paired_fixed_work_provider_installed": global_paired_context_provider is not None,
                "complete_source_pair_provider_installed": global_source_pair_provider is not None,
                "paired_fixed_work_comparison_available": False,
                "paired_fixed_work_comparison_status": "requires_two_bound_revisions_and_same_work_projection",
                "promotion_status": "unqualified_until_semantic_and_global_cost_evidence",
            },
        )
        control = stage_root / "global_control" / f"round_{round_index:04d}"
        receipts = control / "receipts.jsonl"
        broker = PB.Broker(
            inner,
            target_experiment,
            candidate,
            actions,
            receipts,
            deadline=time.monotonic() + broker_window_s,
            max_calls=max_tool_calls,
            max_tool_seconds=experiment.analysis.timeout_s,
            mandatory_analysis_reserve_seconds=0,
            workflow=BP.select_workflow(
                BP.WHOLE_MODEL_V1,
                candidate=candidate,
                target_experiment=target_experiment,
                receipt_path=receipts,
                services=BP.BrokerServices(global_analysis_view=AV.agent_analysis_view),
                global_experiment=experiment,
                global_probe_provider=global_probe_provider,
                global_semantic_provider=global_semantic_provider,
                global_context_provider=global_context_provider,
                global_paired_context_provider=global_paired_context_provider,
                global_source_pair_provider=global_source_pair_provider,
            ),
        )
        try:
            with broker.serving() as (host, port):
                PB.stage_broker_shim(
                    control,
                    host=host,
                    port=port,
                    token=broker.token,
                    tool_timeout_s=experiment.analysis.timeout_s,
                    actions=actions,
                )
                experiment.stage_historical_reference(control, workspace=workspace)
                rc, transcript, _ = AUTHORING._codex_round(
                    workspace,
                    stage_root,
                    prompt,
                    target_experiment,
                    agent_inputs,
                    frozen_functional,
                    experiment.inputs.baseline,
                    frozen_corpus_manifest,
                    control,
                    sandbox_inputs=sandbox_inputs,
                    model=model,
                    resolved_model=resolved_model,
                    effort=effort,
                    round_index=round_index,
                    timeout_s=round_timeout_s,
                    codex_binary=codex_binary,
                )
        finally:
            config = control / ".perf_broker.json"
            if config.is_file() and not config.is_symlink():
                config.chmod(0o600)
                config.unlink()
            if receipts.is_file():
                receipts.chmod(0o444)
        audit = TA.audit_codex_transcript(transcript, target_experiment, candidate, actions)
        refusals = []
        mechanism_attribution = None
        if experiment.mechanism_program.catalog_binding is not None:
            try:
                mechanism_attribution = experiment.revision_session.finalize_mechanism_round(
                    candidate, round_index=round_index
                )
                if mechanism_attribution["status"] != "allowed":
                    refusals.append(
                        "compiler mechanism attribution refused: " + str(mechanism_attribution["violations"])
                    )
            except Exception as exc:  # noqa: BLE001 - any incomplete host gate refuses the round
                mechanism_attribution = {
                    "schema": "global_compiler_mechanism_round_failure_v1",
                    "status": "refused",
                    "round": round_index,
                    "candidate_sha256": hash_tree(candidate)["sha256"],
                    "exception": type(exc).__name__,
                    "reason": str(exc),
                }
                refusals.append(f"compiler mechanism attribution failed: {exc}")
        try:
            evidence = broker.workflow.verify_receipts(receipts, actions=actions, audit=audit)
        except ValueError as exc:
            evidence = {"status": "refused", "reason": str(exc)}
            refusals.append(str(exc))
        post_validation: dict[str, Any]
        if rc == 0 and audit.get("clean") is True and not refusals:
            validation_started = time.monotonic()
            submitted_sha256 = hash_tree(candidate)["sha256"]
            try:
                validation = experiment.analysis.analyze(
                    candidate,
                    hypothesis="Host post-authoring validation of the exact submitted candidate",
                    timeout_s=experiment.analysis.timeout_s,
                )
                if (
                    validation.get("candidate_sha256") != submitted_sha256
                    or hash_tree(candidate)["sha256"] != submitted_sha256
                ):
                    raise ValueError("post-authoring validation is not bound to the submitted candidate bytes")
                iteration_record = experiment.output / f"iteration_{validation['iteration']:04d}.json"
                post_validation = {
                    **post_authoring_validation_contract,
                    "status": "complete",
                    "candidate_sha256": submitted_sha256,
                    "iteration": validation["iteration"],
                    "iteration_record": str(iteration_record),
                    "iteration_record_sha256": P2_CONTRACTS.sha256_file(iteration_record),
                    "readiness": copy.deepcopy(validation.get("readiness")),
                    "fast_evaluation": copy.deepcopy(validation.get("fast_evaluation")),
                    "exact_analysis_reused": validation.get("exact_analysis_reused") is True,
                    "elapsed_seconds": time.monotonic() - validation_started,
                }
                if (
                    experiment.fast_evaluation.provider is not None
                    and validation.get("fast_evaluation", {}).get("status") != "retain"
                ):
                    refusals.append(
                        "host fast accuracy-bounded portfolio gate did not retain the candidate: "
                        + str(validation.get("fast_evaluation", {}).get("status"))
                    )
            except Exception as exc:  # noqa: BLE001 - failed mandatory host validation refuses the round
                post_validation = {
                    **post_authoring_validation_contract,
                    "status": "refused",
                    "candidate_sha256": submitted_sha256,
                    "exception": type(exc).__name__,
                    "reason": str(exc),
                    "elapsed_seconds": time.monotonic() - validation_started,
                }
                refusals.append(f"host post-authoring full-model validation failed: {exc}")
        else:
            post_validation = {
                **post_authoring_validation_contract,
                "status": "not_started",
                "reason": "Codex round or its audit/broker evidence was not clean",
            }
        try:
            current = experiment.revision_session.current(candidate, require_ready=False)
        except ValueError as exc:
            current = {"candidate_sha256": hash_tree(candidate)["sha256"]}
            refusals.append(str(exc))
        try:
            telemetry = TEL.collect_round(
                stage_root,
                round_index,
                model=resolved_model,
                agent_exit_code=rc,
                preflight_record=TEL.read_preflight(stage_root / "telemetry_preflight.json"),
            )
        except P2_CONTRACTS.StageGateError as exc:
            telemetry = {"complete": False, "reason": str(exc)}
            refusals.append(str(exc))
        # A SPENT BUDGET IS NOT A CRASH. The rule lives in one place (phase 1 has always used it) so
        # this driver and the stage cannot drift: the round-deadline exit is admitted alongside a clean
        # audit and no refusals, and every other non-zero exit stays refused. Admitting it here is what
        # stops a deadline-killed round from being non-authored, which raised, re-consumed the
        # checkpoint and DISCARDED the round's compiler edits -- 2 of 3 paid rounds on the last run.
        admission = AUTHORING.authored_round_status(
            agent_exit_code=rc, audit_clean=audit.get("clean"), refusals=refusals
        )
        record = {
            "schema": "global_agent_round_v1",
            "round": round_index,
            "candidate_sha256": current["candidate_sha256"],
            "agent_exit_code": rc,
            "audit": audit,
            "broker_evidence": evidence,
            "telemetry": telemetry,
            "mechanism_attribution": mechanism_attribution,
            "compiler_mechanism_work_order": copy.deepcopy(experiment.mechanism_program.work_order_binding),
            "mechanism_work_order_analysis": copy.deepcopy(experiment.mechanism_program.analysis_binding),
            "host_post_authoring_validation": post_validation,
            "authoring_readiness": copy.deepcopy(current.get("readiness")),
            "fast_evaluation": copy.deepcopy(current.get("fast_evaluation")),
            "promotion_ready": (
                current.get("readiness", {}).get("status") == "ready_for_probe_admission"
                and (
                    experiment.fast_evaluation.provider is None
                    or current.get("fast_evaluation", {}).get("status") == "retain"
                )
            ),
            "status": admission["status"],
            # Named rather than implied: "authored, and the budget ended it" and "authored, having
            # finished" are different results, and a report that shows them as one is how the last
            # campaign's three rounds looked identical.
            "stopped_by": admission["stopped_by"],
            "status_reason": admission["why"],
            "refusal_reasons": refusals,
            "global_speedup_proven": False,
        }
        experiment._write(f"agent_round_{round_index:04d}.json", record)
        if record["status"] != "authored":
            raise ValueError("macro agent round did not finish with a clean authoring audit")
        return record
