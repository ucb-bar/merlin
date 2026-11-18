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
import os
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

from merlin.analysis.optimization.guidance.agent_brief import inspect_compiler_package
from merlin.analysis.perf.measurement.execution_policy import FIRESIM_LIFECYCLE, ITERATION_MAX_SECONDS
from merlin.common.digest import hash_tree
from merlin.common.digest import sha256_bytes as _sha256
from merlin.development.backend.optimization import contracts as CONTRACTS
from merlin.development.backend.optimization.author_interface import agent_workspace as AW
from merlin.development.backend.optimization.author_interface import broker_policy as BP
from merlin.development.backend.optimization.author_interface import broker_service as PB
from merlin.development.backend.optimization.author_interface import functional_inputs as FI
from merlin.development.backend.optimization.author_interface import stage_inputs as INPUTS
from merlin.development.backend.optimization.author_interface import stage_prompt as SP
from merlin.development.backend.optimization.author_interface.broker_service import (
    AGENT_CORPUS_MOUNT,
    BROKER_NAME,
    BrokerAction,
    action_registry_contract,
)
from merlin.development.backend.optimization.authoring import authoring_selections as SELECTIONS
from merlin.development.backend.optimization.authoring import candidate_claims as RECORD
from merlin.development.backend.optimization.authoring import candidate_record as _merlin_candidate_record_4
from merlin.development.backend.optimization.authoring import candidate_verification as VERIFY
from merlin.development.backend.optimization.authoring import telemetry as _merlin_telemetry_1
from merlin.development.backend.optimization.authoring import telemetry_accounting as TEL
from merlin.development.backend.optimization.authoring.authoring_artifacts import (
    assert_candidate_sealable,
    authored_round_status,
    candidate_delta,
)
from merlin.development.backend.optimization.authoring.authoring_artifacts import (
    development_feedback_performed as _development_feedback_performed,
)
from merlin.development.backend.optimization.authoring.stage_preparation import prepare_stage_inputs
from merlin.development.backend.optimization.authoring.stage_sealing import seal_stage_record
from merlin.development.backend.optimization.authoring.transcript_audit import audit_codex_transcript
from merlin.development.backend.optimization.campaign import compiler_libraries as LIBRARIES
from merlin.development.backend.optimization.campaign import compiler_libraries as LIBRARY_GUIDANCE
from merlin.development.backend.optimization.campaign import compiler_library_state as LIBRARY
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.campaign import normal_resource_launch as NR
from merlin.development.backend.optimization.campaign import package_sandbox as _merlin_package_sandbox_2
from merlin.development.backend.optimization.campaign import pass_preparation as PREPARATION
from merlin.development.backend.optimization.contracts import (
    ROUND_DEADLINE_EXIT,
    StageGateError,
)
from merlin.development.backend.optimization.contracts import (
    canonical_json as _canonical_json,
)
from merlin.development.backend.optimization.contracts import (
    require_executable as _require_executable,
)
from merlin.development.backend.optimization.contracts import (
    sha256_file as _sha256_file,
)
from merlin.development.backend.optimization.contracts import (
    write_json as _write_json,
)
from merlin.development.backend.optimization.diagnostics import emission_cache as EA
from merlin.development.backend.optimization.diagnostics import emission_diagnostics as ED
from merlin.development.backend.optimization.diagnostics import (
    whole_model_emission as _merlin_whole_model_emission_3,
)
from merlin.development.backend.optimization.measurement import development_feedback as DF
from merlin.development.backend.optimization.ordinary import capsule_ordinary_scope as CAPSULE_SCOPE
from merlin.development.backend.optimization.ordinary import capsule_source_development as CAPSULE_DEV
from merlin.development.targetgen.authoring.library_admission import active_compiler_library
from merlin.development.targetgen.sandbox import preflight as SANDBOX_PREFLIGHT
from merlin.development.targetgen.sandbox.answer_surfaces import absent_declaration_coverage, dropped_declarations
from merlin.target.registry.experiment_descriptor import TargetExperiment

# ---- authoring ---------------------------------------------------------------------------------


def _require_stage_containment(bwrap_binary: Path, target_experiment: TargetExperiment, *, external_boundary=None):
    """Refuse a launch before output creation unless both containment claims hold."""
    return NR.require_stage_containment(
        bwrap_binary,
        target_experiment,
        external_boundary=external_boundary,
        sandbox_preflight=SANDBOX_PREFLIGHT,
        dropped_declarations=dropped_declarations,
    )


_GUARD_UNCHANGED = "unchanged"

_GUARD_CHANGED = "changed"


@LIBRARIES.paired_functional_guard
def functional_emission_guard(
    baseline: Path,
    candidate: Path,
    target_experiment: TargetExperiment,
    *,
    frozen_functional: FI.FrozenFunctionalInputs,
    timeout_s: int = 120,
    grade_root: Path | None = None,
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs | None = None,
) -> dict[str, Any]:
    """Apply the paired library guard before independent emission validation."""
    from merlin.development.backend.optimization.authoring.functional_emission import (
        functional_emission_guard as guard,
    )

    return guard(
        baseline,
        candidate,
        target_experiment,
        frozen_functional=frozen_functional,
        timeout_s=timeout_s,
        grade_root=grade_root,
        sandbox_inputs=sandbox_inputs,
        active_compiler_library=active_compiler_library,
    )


def queued_author_stop(workflow, broker, deadline):
    """Read the still-live original queue; a finished model turn is not a stop."""
    from merlin.development.backend.optimization.authoring.author_runtime import queued_author_stop as stop

    return stop(workflow, broker, deadline)


def _import_codex_driver():
    from merlin.development.backend.providers import codex_launch_codex_agent as codex_agent  # noqa: PLC0415

    return codex_agent


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
    max_concurrent_subagents: int | None = None,
    external_boundary=None,
    continue_session: bool = False,
    continuation_stop=None,
    session_control=None,
) -> tuple[int, Path, AW.AgentSandboxPolicy]:
    if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
        raise StageGateError("Codex authoring requires an explicit sandbox input selection")
    arguments = dict(locals())
    from merlin.development.backend.optimization.authoring.author_runtime import dispatch_author_round

    return dispatch_author_round(driver=_import_codex_driver(), **arguments)


def admit_authoring_workflow(workflow_id: str, *, component_launch=None) -> None:
    """Keep the broker action profile distinct from fresh-session launch authority."""
    return SELECTIONS.admit_workflow(workflow_id, component_launch=component_launch)


@LIBRARIES.stage_selection
def run_stage(
    *,
    suite: str,
    contract_root: Path,
    source_root: Path,
    functional_runs_root: Path,
    functional_run_id: str,
    functional_submission_sha256: str,
    published_compiler_root: Path | None = None,
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
    workflow_id: str = BP.CORPUS_FEEDBACK_V1,
    component_launch=None,
    whole_model_profile_inputs: Path | None = None,
    search_configuration: Path | None = None,
    compiler_edit_contract: Path | None = None,
    external_resource_boundary=None,
    compiler_library_selection=None,
    diagnostic_library_stage: bool = False,
    archive_screen_selection=None,
    compiler_pass_selection=None,
    compiler_pass_preparation=None,
    capsule_phase1_owner=None,
    capsule_development=None,
    required_cohort_declaration=None,
    checked_source_diagnostic=None,
    broker_transport: str = "tcp",
    author_guidance: str | None = None,
    continue_queued_author: bool = False,
) -> Path:
    """Run bounded authoring rounds and return the sealed candidate-record path.

    ``whole_model_profile_inputs`` is the operator's optional whole-model deployment record
    (:mod:`.whole_model_profile`); without it the whole-model boundary profile is advertised as
    unavailable."""
    guidance_record = SP.freeze_author_guidance(author_guidance)
    if type(continue_queued_author) is not bool or (
        continue_queued_author and (search_configuration is None or rounds != 1)
    ):
        raise StageGateError("continuous queued authoring requires one explicit queued search round")
    if type(broker_transport) is not str or broker_transport not in {"tcp", "files.v1"}:
        raise StageGateError("authoring requires an explicit supported broker transport")
    if broker_transport != "tcp" and workflow_id != BP.CORPUS_FEEDBACK_V1:
        raise StageGateError("file broker transport is scoped to ordinary corpus authoring")
    if broker_transport == "files.v1":
        from merlin.development.backend.optimization.author_interface.broker_transports import (
            validate_call_budget,
        )

        validate_call_budget(max_tool_calls)
    if external_resource_boundary is not None:
        from merlin.common.paths import out_dir

        temporary = os.environ.get("TMPDIR")
        if not temporary:
            raise StageGateError("hard normal authoring requires an explicit bounded TMPDIR")
        with NR.selected_boundary(
            external_resource_boundary, writable_roots=(Path(stage_root), out_dir(), Path(temporary))
        ):
            pass
        if workflow_id != BP.CORPUS_FEEDBACK_V1:
            raise StageGateError("normal resource authoring does not admit another launch workflow")
    admit_authoring_workflow(workflow_id, component_launch=component_launch)
    SELECTIONS.validate(workflow_id, search_configuration, compiler_library_selection, archive_screen_selection)
    if workflow_id == BP.COMPONENT_ONLY_V1:
        from merlin.development.backend.optimization.components.launch import run_component_stage  # noqa: PLC0415

        if rounds != 1:
            raise StageGateError(
                "component authoring uses one fresh continuous session within the explicit wall budget"
            )
        return run_component_stage(
            component_launch,
            suite=suite,
            model=model,
            effort=effort,
            wall_budget_seconds=wall_budget_seconds,
            max_tool_calls=max_tool_calls,
            tool_timeout_seconds=tool_timeout_seconds,
        )
    if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
        raise StageGateError("performance authoring requires an explicit sandbox input selection")
    sandbox_inputs = copy.deepcopy(sandbox_inputs)
    search = None
    if search_configuration is not None:
        from merlin.development.backend.optimization.search.search_configuration import (
            read_search_configuration,
        )

        search = read_search_configuration(search_configuration)
        if rounds != 1 or compiler_edit_contract is None:
            raise StageGateError("queued corpus search uses one continuous round and an explicit edit contract")
        from merlin.development.backend.optimization.authoring.author_runtime import require_stage_budget

        require_stage_budget(search.settings, wall_budget_seconds, round_timeout_seconds)
    PREPARATION.validate_stage_selection(
        compiler_pass_selection,
        compiler_pass_preparation,
        compiler_library_selection,
        diagnostic_library_stage,
        search,
        compiler_edit_contract,
    )
    SELECTIONS.validate_stage_budgets(
        suite,
        model,
        (wall_budget_seconds, rounds, round_timeout_seconds, max_tool_calls, tool_timeout_seconds, smoke_replicates),
        tool_timeout_seconds=tool_timeout_seconds,
        iteration_max_seconds=ITERATION_MAX_SECONDS,
    )
    CAPSULE_DEV.validate_preparation(
        capsule_development,
        compiler_pass_selection,
        compiler_pass_preparation,
        compiler_library_selection,
        diagnostic_library_stage,
        search,
        compiler_edit_contract,
        checked_source_diagnostic,
    )
    (
        agent_inputs,
        base,
        bwrap_binary,
        bwrap_sha256,
        candidate_seed,
        capsule_scope,
        codex_path,
        codex_sha256,
        declaration_drops,
        descriptor_path,
        descriptor_sha256,
        feedback,
        fork,
        fork_check,
        formal_claim,
        frozen_corpus,
        frozen_functional,
        functional,
        inherited_common_checks,
        inherited_common_transforms,
        original_base,
        prepared_action_contract,
        prepared_actions,
        prepared_compilers,
        prompt,
        prompt_inputs,
        replicates,
        resolved_model,
        sandbox_probe,
        seed_digest,
        stage_root,
        staged_prompt,
        telemetry_preflight_record,
    ) = prepare_stage_inputs(
        capsule_development=capsule_development,
        capsule_phase1_owner=capsule_phase1_owner,
        capsules=capsules,
        checked_source_diagnostic=checked_source_diagnostic,
        codex_binary=codex_binary,
        compiler_library_selection=compiler_library_selection,
        compiler_pass_preparation=compiler_pass_preparation,
        compiler_pass_selection=compiler_pass_selection,
        continue_queued_author=continue_queued_author,
        contract_root=contract_root,
        external_resource_boundary=external_resource_boundary,
        families=families,
        functional_run_id=functional_run_id,
        functional_runs_root=functional_runs_root,
        functional_submission_sha256=functional_submission_sha256,
        gsim_certificate=gsim_certificate,
        gsim_certificate_sha256=gsim_certificate_sha256,
        guidance_record=guidance_record,
        max_tool_calls=max_tool_calls,
        model=model,
        published_compiler_root=published_compiler_root,
        replicates=replicates,
        required_cohort_declaration=required_cohort_declaration,
        round_timeout_seconds=round_timeout_seconds,
        rounds=rounds,
        rtl_facts=rtl_facts,
        search=search,
        smoke_replicates=smoke_replicates,
        source_root=source_root,
        stage_root=stage_root,
        target_experiment=target_experiment,
        telemetry_price_table=telemetry_price_table,
        tool_timeout_seconds=tool_timeout_seconds,
        waive_functional_gate=waive_functional_gate,
        wall_budget_seconds=wall_budget_seconds,
        whole_model_profile_inputs=whole_model_profile_inputs,
        authoring_source=Path(__file__),
    )

    from merlin.development.backend.optimization.author_interface import (
        common_transforms as COMMON_TRANSFORMS,
    )
    from merlin.development.backend.optimization.author_interface import (
        public_compile_only as PUBLIC_COMPILE_ONLY,
    )
    from merlin.development.backend.optimization.authoring.author_runtime import stage_deadline

    if search is not None:
        from merlin.development.backend.optimization.author_interface.queued_search_actions import (
            actions as search_actions,
        )

    deadline = stage_deadline(search.settings if search is not None else None, wall_budget_seconds)
    previous_submission = candidate_seed
    previous_digest = seed_digest
    round_records: list[dict[str, Any]] = []
    transcript_paths: list[Path] = []
    probe_results: list[dict[str, Any]] | None = None
    last_outer: AW.AgentSandboxPolicy | None = None
    refusal: str | None = None
    stopped_by: dict[str, Any] | None = None
    total_calls = 0
    receipt_records: list[dict[str, Any]] = []
    broker_channel_records: list[dict[str, Any]] = []
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
        initial_whole_model_analysis = CAPSULE_SCOPE.initial_analysis(
            capsule_scope,
            feedback,
            base,
            candidate,
            prompt_inputs.e2e_sentinel,
            contract_root,
            min(tool_timeout_seconds, remaining),
            target_experiment.target,
        )
        profile_selector = getattr(feedback, "profile_witness", None)
        if capsule_scope is None and callable(profile_selector):
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
                **(
                    {
                        "queued_author_continuation": {
                            "enabled": True,
                            "prompt_sha256": _sha256(SP.QUEUED_AUTHOR_CONTINUATION.encode()),
                            "stop_owner": "original live search coordinator and broker budgets",
                        }
                    }
                    if continue_queued_author
                    else {}
                ),
                **({"author_guidance": dict(guidance_record)} if guidance_record is not None else {}),
                **(
                    {"search_session": prompt_inputs.search_session} if prompt_inputs.search_session is not None else {}
                ),
                **({"broker_transport": broker_transport} if broker_transport != "tcp" else {}),
                **(
                    {"broker_control_ordering": "Serialize broker requests; evaluation jobs can run in parallel."}
                    if broker_transport == "files.v1"
                    else {}
                ),
                "functional_submission_sha256": functional.digest,
                **(
                    {"compiler_pass_preparation": prompt_inputs.compiler_pass_preparation}
                    if prepared_compilers is not None
                    else {}
                ),
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
                **LIBRARY_GUIDANCE.prompt_context(prompt_inputs),
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
                **CAPSULE_DEV.measurement_context(capsule_scope, maximum_seconds=int(ITERATION_MAX_SECONDS)),
            },
        )
        actions = BP.action_registry(BP.CORPUS_FEEDBACK_V1, candidate, target_experiment)
        if inherited_common_transforms:
            actions = (*actions, COMMON_TRANSFORMS.action())
        if inherited_common_checks:
            actions = (*actions, COMMON_TRANSFORMS.check_action())
        if inherited_common_transforms:
            actions = (*actions, PUBLIC_COMPILE_ONLY.action())
        if search is not None:
            actions = (*actions, *search_actions())
        actions = CAPSULE_SCOPE.stage_actions(capsule_scope, actions)
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
            probe_results = AW.run_required_tool_probes(
                inner,
                target_experiment,
                candidate,
                **({"external_boundary": external_resource_boundary} if external_resource_boundary is not None else {}),
            )
        control_dir = stage_root / "control" / f"round_{round_index:02d}"
        receipt_path = control_dir / "receipts.jsonl"
        common_transform, common_check = (
            COMMON_TRANSFORMS.services(
                frozen_functional,
                candidate=candidate,
                work_root=stage_root / "_common_transform_work",
            )
            if inherited_common_transforms and capsule_development is None
            else (None, None)
        )
        compile_only = (
            PUBLIC_COMPILE_ONLY.service(
                frozen_functional,
                candidate=candidate,
                corpus=frozen_corpus,
                target_experiment=target_experiment,
                contract=contract_root,
                facts_path=rtl_facts,
                work_root=stage_root / "_public_compile_work",
                sandbox_inputs=sandbox_inputs,
            )
            if inherited_common_transforms and capsule_development is None
            else None
        )
        workflow = BP.select_workflow(
            BP.CORPUS_FEEDBACK_V1,
            candidate=candidate,
            target_experiment=target_experiment,
            receipt_path=receipt_path,
            services=BP.BrokerServices(
                whole_model_analysis=partial(
                    _merlin_whole_model_emission_3.analyze_whole_model_emission, contract_root=contract_root
                ),
                command_buffer_analysis=ED.analyze_command_buffers,
                common_transform=common_transform,
                common_check=common_check,
                compile_only=compile_only,
            ),
            feedback_evaluator=feedback,
            feedback_round=round_index,
            functional_base=base,
            e2e_sentinel=prompt_inputs.e2e_sentinel,
        )
        search_pool = None
        if search is not None:
            from merlin.development.backend.optimization.ordinary.ordinary_search_stage import prepare_round

            workflow, search_pool = prepare_round(
                search,
                stage_root=stage_root,
                candidate=candidate,
                baseline=base,
                corpus=frozen_corpus,
                edit_contract=compiler_edit_contract,
                functional_runs_root=functional_runs_root,
                functional_run_id=functional_run_id,
                functional_submission_sha256=functional_submission_sha256,
                target_experiment=target_experiment,
                receipt_path=receipt_path,
                contract_root=contract_root,
                certificate_path=gsim_certificate,
                certificate_sha256=gsim_certificate_sha256,
                rtl_facts_path=rtl_facts,
                services=workflow.services,
                e2e_sentinel=prompt_inputs.e2e_sentinel,
                tuning_call_budget=max_tool_calls,
                waive_functional_gate=waive_functional_gate,
                whole_model_inputs=whole_model_profile_inputs,
                external_boundary=external_resource_boundary,
                archive_screen_selection=archive_screen_selection,
                compiler_pass_selection=compiler_pass_selection,
                compiler_pass_preparation=prepared_compilers,
                capsule_phase1_owner=capsule_phase1_owner,
                capsule_development=capsule_development,
                required_cohort_declaration=required_cohort_declaration,
                checked_source_diagnostic=checked_source_diagnostic,
            )
        broker = PB.Broker(
            inner,
            target_experiment,
            candidate,
            actions,
            receipt_path,
            deadline=deadline,
            max_calls=max_tool_calls - total_calls,
            max_tool_seconds=tool_timeout_seconds,
            workflow=workflow,
            **({"external_boundary": external_resource_boundary} if external_resource_boundary is not None else {}),
        )
        round_timeout = min(round_timeout_seconds, remaining)
        if search is not None and search.settings.continue_author_session:
            round_timeout = min(round_timeout, int(deadline - time.monotonic()) - search.settings.final_reserve_seconds)
        try:
            if round_timeout <= 0:
                raise StageGateError("campaign time is reserved for final verification")
            with broker.serving_author(
                transport=broker_transport,
                workspace=workspace,
                control_dir=control_dir,
                tool_timeout_s=tool_timeout_seconds,
                actions=actions,
            ) as channel:
                if channel is not None:
                    broker_channel_records.append({"round": round_index, **channel})
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
                    **(
                        {
                            "continue_session": True,
                            "continuation_stop": lambda: queued_author_stop(workflow, broker, deadline),
                        }
                        if continue_queued_author
                        and not (search is not None and search.settings.continue_author_session)
                        else {}
                    ),
                    **(
                        {"external_boundary": external_resource_boundary}
                        if external_resource_boundary is not None
                        else {}
                    ),
                    **({"max_concurrent_subagents": search.author_agents} if search is not None else {}),
                    **(
                        {"session_control": broker}
                        if search is not None and search.settings.continue_author_session
                        else {}
                    ),
                )
        finally:
            if search_pool is not None:
                search_pool.close()
            broker_config = control_dir / ".perf_broker.json"
            if broker_config.is_file() and not broker_config.is_symlink():
                broker_config.chmod(0o600)
                broker_config.unlink()
            if receipt_path.is_file() and not receipt_path.is_symlink():
                receipt_path.chmod(0o444)
        total_calls += len(broker.calls)
        transcript_paths.append(transcript)
        round_telemetry = _merlin_telemetry_1.collect_round(
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
        if capsule_scope is None and not any(
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
        # Reopen every completed repeat before ending a target-driven stage;
        # cached target verdicts cannot survive a later slower observation.
        coordinator = getattr(broker.workflow, "search_coordinator", None)
        if coordinator is not None:
            target_status = coordinator.status().get("target", {})
            if target_status.get("achieved") is True:
                stopped_by = {
                    "conditions": ["declared_performance_target"],
                    "target": target_status,
                    "round": round_index,
                }
                break
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
        probe_recheck_results = AW.run_required_tool_probes(
            last_inner,
            target_experiment,
            previous_submission,
            **({"external_boundary": external_resource_boundary} if external_resource_boundary is not None else {}),
        )
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
    CAPSULE_DEV.revalidate_corpus(frozen_corpus)
    AW.verify_answer_free_agent_inputs(agent_inputs)
    sealed = stage_root / "sealed_candidate" / "submission"
    try:
        assert_candidate_sealable(previous_submission)
    except StageGateError as exc:
        refusal = refusal or f"final candidate is not sealable: {exc}"
    sealed_sha = _merlin_package_sandbox_2.materialize_readonly_tree(previous_submission, sealed)
    if sealed_sha != previous_digest:
        raise StageGateError("sealed performance candidate changed during final copy")
    after = _merlin_package_sandbox_2.check_fork(fork, original_base)
    if after.ok is not True:
        refusal = f"functional base fork changed during authoring: {after.reason}"
    # A round that reached its declared deadline counts as a clean exit only when the loop above
    # classified it that way -- which it does only with a clean audit and a sealed candidate, and
    # which it records in `stopped_by`. Without that, the deadline is still a refusal.
    return seal_stage_record(
        after=after,
        agent_inputs=agent_inputs,
        base=base,
        broker_channel_records=broker_channel_records,
        broker_transport=broker_transport,
        bwrap_binary=bwrap_binary,
        bwrap_sha256=bwrap_sha256,
        candidate_seed=candidate_seed,
        capsule_scope=capsule_scope,
        codex_path=codex_path,
        codex_sha256=codex_sha256,
        compiler_library_selection=compiler_library_selection,
        compiler_pass_preparation=compiler_pass_preparation,
        compiler_pass_selection=compiler_pass_selection,
        declaration_drops=declaration_drops,
        descriptor_path=descriptor_path,
        descriptor_sha256=descriptor_sha256,
        diagnostic_library_stage=diagnostic_library_stage,
        effort=effort,
        external_resource_boundary=external_resource_boundary,
        feedback=feedback,
        fork_check=fork_check,
        formal_claim=formal_claim,
        frozen_corpus=frozen_corpus,
        frozen_functional=frozen_functional,
        functional=functional,
        last_inner=last_inner,
        last_outer=last_outer,
        max_tool_calls=max_tool_calls,
        model=model,
        original_base=original_base,
        prepared_action_contract=prepared_action_contract,
        prepared_actions=prepared_actions,
        prepared_compilers=prepared_compilers,
        previous_submission=previous_submission,
        probe_recheck_results=probe_recheck_results,
        probe_results=probe_results,
        prompt=prompt,
        prompt_inputs=prompt_inputs,
        receipt_records=receipt_records,
        refusal=refusal,
        replicates=replicates,
        resolved_model=resolved_model,
        round_records=round_records,
        round_timeout_seconds=round_timeout_seconds,
        rounds=rounds,
        sandbox_inputs=sandbox_inputs,
        sandbox_probe=sandbox_probe,
        sealed=sealed,
        sealed_sha=sealed_sha,
        search=search,
        seed_digest=seed_digest,
        smoke_replicates=smoke_replicates,
        stage_root=stage_root,
        staged_prompt=staged_prompt,
        stopped_by=stopped_by,
        suite=suite,
        target_experiment=target_experiment,
        telemetry_preflight_record=telemetry_preflight_record,
        tool_timeout_seconds=tool_timeout_seconds,
        total_calls=total_calls,
        transcript_paths=transcript_paths,
        wall_budget_seconds=wall_budget_seconds,
        workflow=workflow,
        authoring_source=Path(__file__),
    )
