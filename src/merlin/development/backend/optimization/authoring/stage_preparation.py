"""Prepare stage inputs for a bounded authoring stage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from merlin.development.backend.optimization import contracts as CONTRACTS
from merlin.development.backend.optimization.author_interface import agent_workspace as AW
from merlin.development.backend.optimization.author_interface import broker_policy as BP
from merlin.development.backend.optimization.author_interface import functional_inputs as FI
from merlin.development.backend.optimization.author_interface import stage_inputs as INPUTS
from merlin.development.backend.optimization.author_interface import stage_prompt as SP
from merlin.development.backend.optimization.author_interface.broker_service import (
    action_registry_contract,
)
from merlin.development.backend.optimization.authoring import telemetry as _merlin_telemetry_1
from merlin.development.backend.optimization.campaign import compiler_libraries as LIBRARIES
from merlin.development.backend.optimization.campaign import compiler_library_state as LIBRARY
from merlin.development.backend.optimization.campaign import package_sandbox as _merlin_package_sandbox_2
from merlin.development.backend.optimization.contracts import (
    StageGateError,
)
from merlin.development.backend.optimization.contracts import (
    require_executable as _require_executable,
)
from merlin.development.backend.optimization.contracts import (
    sha256_file as _sha256_file,
)
from merlin.development.backend.optimization.ordinary import capsule_ordinary_scope as CAPSULE_SCOPE
from merlin.development.backend.optimization.ordinary import capsule_source_development as CAPSULE_DEV


def prepare_stage_inputs(
    *,
    capsule_development,
    capsule_phase1_owner,
    capsules,
    checked_source_diagnostic,
    codex_binary,
    compiler_library_selection,
    compiler_pass_preparation,
    compiler_pass_selection,
    continue_queued_author,
    contract_root,
    external_resource_boundary,
    families,
    functional_run_id,
    functional_runs_root,
    functional_submission_sha256,
    gsim_certificate,
    gsim_certificate_sha256,
    guidance_record,
    max_tool_calls,
    model,
    published_compiler_root,
    replicates,
    required_cohort_declaration,
    round_timeout_seconds,
    rounds,
    rtl_facts,
    search,
    smoke_replicates,
    source_root,
    stage_root,
    target_experiment,
    telemetry_price_table,
    tool_timeout_seconds,
    waive_functional_gate,
    wall_budget_seconds,
    whole_model_profile_inputs,
    authoring_source,
):
    from merlin.development.backend.optimization.authoring.authoring import _require_stage_containment

    bwrap_binary = _require_executable("bwrap", label="bwrap")
    sandbox_probe, declaration_drops = _require_stage_containment(
        bwrap_binary,
        target_experiment,
        **({"external_boundary": external_resource_boundary} if external_resource_boundary is not None else {}),
    )
    codex_path = _require_executable(codex_binary, label="Codex")
    descriptor_path = Path(target_experiment.path).resolve()
    bwrap_sha256 = _sha256_file(bwrap_binary)
    codex_sha256 = _sha256_file(codex_path)
    descriptor_sha256 = _sha256_file(descriptor_path)
    telemetry_preflight_record = _merlin_telemetry_1.prepare(
        authoring_stage=authoring_source.resolve(),
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

    # The default gate retains the coordinator's waiver handling. Explicit
    # capsule development selects its original independent owner without waivers.
    functional, capsule_scope = CAPSULE_SCOPE.stage_selection(
        capsule_phase1_owner,
        functional_runs_root,
        functional_run_id,
        functional_submission_sha256,
        target_experiment,
        required_cohort_declaration,
        waive_functional_gate,
        search,
        families,
        capsules,
        capsule_development,
    )
    LIBRARY.require_phase1_selection(functional, compiler_library_selection)
    discovered = CAPSULE_DEV.discover(capsule_development, target_experiment, families, capsules)
    stage_root.mkdir(parents=True)
    base = _merlin_package_sandbox_2.materialize_perf_workspace(
        functional,
        stage_root / "_frozen_functional",
        published_root=published_compiler_root,
        target=target_experiment.target,
    )
    original_base, baseline_digest, candidate_seed, seed_digest = base, functional.digest, base, functional.digest
    prepared_compilers = None
    if compiler_pass_preparation is not None:
        from merlin.development.backend.optimization.campaign.pass_preparation import prepare_pair

        prepared_compilers = prepare_pair(
            compiler_pass_selection,
            compiler_pass_preparation,
            original=original_base,
            functional_sha256=functional.digest,
            root=stage_root / "_prepared_compilers",
        )
        base, baseline_digest = Path(prepared_compilers["baseline"]["path"]), prepared_compilers["baseline"]["sha256"]
        candidate_seed = Path(prepared_compilers["candidate_seed"]["path"])
        seed_digest = prepared_compilers["candidate_seed"]["sha256"]
    frozen_corpus = CAPSULE_DEV.freeze(discovered, stage_root / "_frozen_corpus")
    formal_claim, replicates = CAPSULE_DEV.formal_claim(frozen_corpus, replicates)
    agent_inputs = AW.build_answer_free_agent_inputs(frozen_corpus, target_experiment, stage_root / "_agent_inputs")
    frozen_functional = FI.load_frozen_functional_inputs(
        functional, public_manifest_path=stage_root / "_public_functional_snapshot.json"
    )
    from merlin.development.backend.optimization.author_interface import (
        common_transforms as COMMON_TRANSFORMS,
    )
    from merlin.development.backend.optimization.author_interface import (
        public_compile_only as PUBLIC_COMPILE_ONLY,
    )

    public_projection = (
        CONTRACTS.mapping_file(frozen_functional.public_marker) if frozen_functional.public_marker else {}
    )
    inherited_common_transforms = public_projection.get("compiler_tool_profile") is not None
    inherited_common_checks = COMMON_TRANSFORMS.supports_check(public_projection.get("compiler_tool_profile"))
    prepared_actions = BP.action_registry(
        BP.CORPUS_FEEDBACK_V1,
        base,
        target_experiment,
        unavailable=(
            {BP.WHOLE_MODEL_PROFILE_ACTION: BP.WHOLE_MODEL_PROFILE_UNAVAILABLE}
            if whole_model_profile_inputs is None
            else None
        ),
    )
    if inherited_common_transforms:
        prepared_actions = (*prepared_actions, COMMON_TRANSFORMS.action())
    if inherited_common_checks:
        prepared_actions = (*prepared_actions, COMMON_TRANSFORMS.check_action())
    if inherited_common_transforms:
        prepared_actions = (*prepared_actions, PUBLIC_COMPILE_ONLY.action())
    if search is not None:
        from merlin.development.backend.optimization.author_interface.queued_search_actions import (
            actions as search_actions,
        )

        prepared_actions = (*prepared_actions, *search_actions())
    prepared_actions = CAPSULE_SCOPE.stage_actions(capsule_scope, prepared_actions)
    prepared_action_contract = action_registry_contract(prepared_actions, base)
    minimum_calls = rounds * sum(action.required for action in prepared_actions)
    if max_tool_calls < minimum_calls:
        raise StageGateError(
            "inner tool-call budget cannot cover every required broker action in every round: "
            f"need at least {minimum_calls}, got {max_tool_calls}"
        )
    feedback = CAPSULE_DEV.prepare_feedback(
        capsule_development,
        checked_source_diagnostic,
        compiler_pass_selection,
        contract_root=contract_root,
        certificate_path=gsim_certificate,
        certificate_sha256=gsim_certificate_sha256,
        rtl_facts_path=rtl_facts,
        corpus=frozen_corpus,
        baseline=base,
        baseline_sha256=baseline_digest,
        target_experiment=target_experiment,
        work_root=stage_root / "_development_feedback",
        # Every tuning measurement spends one inner tool call, so the run's own call budget is the
        # real bound on how many the search can take. Declaring it makes ``budget_exhausted`` report
        # true remaining spend instead of "unbounded".
        tuning_call_budget=max_tool_calls,
        functional_run_dir=functional.run_dir,
        whole_model_inputs=whole_model_profile_inputs,
        capsule_development_scope=capsule_scope,
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
        capsule_development_scope=CAPSULE_SCOPE.verify(capsule_scope) if capsule_scope is not None else None,
    )
    prompt_inputs = LIBRARIES.prompt_inputs(prompt_inputs)
    if continue_queued_author:
        prompt_inputs = replace(prompt_inputs, continue_queued_author=True)
    if guidance_record is not None:
        prompt_inputs = replace(prompt_inputs, author_guidance=guidance_record)
    if prepared_compilers is not None:
        from merlin.development.backend.optimization.campaign.pass_preparation import prompt_projection

        prompt_inputs = replace(
            prompt_inputs,
            frozen_functional_sha256=baseline_digest,
            submission_initial_sha256=seed_digest,
            compiler_pass_preparation=prompt_projection(prepared_compilers),
        )
    if search is not None:
        from merlin.development.backend.optimization.authoring.author_runtime import timing_projection

        prompt_inputs = replace(
            prompt_inputs, author_agents=search.author_agents, search_session=timing_projection(search.settings)
        )
    staged_prompt = stage_root / "prompt.txt"
    prompt = SP.materialize_canonical_prompt(prompt_inputs, staged_prompt)
    staged_prompt.chmod(0o444)
    fork = _merlin_package_sandbox_2.functional_fork(functional)
    fork_check = _merlin_package_sandbox_2.check_fork(fork, original_base)
    if fork_check.ok is not True:
        raise StageGateError(f"functional fork is invalid before performance authoring: {fork_check.reason}")

    return (
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
    )
