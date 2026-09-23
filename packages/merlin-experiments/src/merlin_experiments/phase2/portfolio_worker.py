"""Installed full-portfolio worker with explicit source, resource and runtime selections."""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin.perf.agent_guidance import inspect_compiler_package
from merlin.perf.functional_gate import load_functional_gate_config
from merlin.perf.gate_discrimination import require_discriminating
from merlin.runtime.backends.base import get_backend
from merlin.targetgen.target_experiment import TargetExperiment
from merlin_experiments.phase2 import agent_workspace as AW
from merlin_experiments.phase2 import authoring as AUTHORING
from merlin_experiments.phase2 import broker_evidence as BE
from merlin_experiments.phase2 import campaign as PC
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import corpus as P2_CORPUS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import functional_inputs as FI
from merlin_experiments.phase2 import portfolio_resume as RESUME
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2 import telemetry as TEL
from merlin_experiments.phase2.fast_evaluation_installation import prepare as _prepare_fast_evaluator_installation
from merlin_experiments.phase2.global_experiment import GlobalPerfExperiment
from merlin_experiments.phase2.global_inputs import FrozenPhase1, full_model_portfolio_identity
from merlin_experiments.phase2.portfolio_authoring import PortfolioAuthoring
from merlin_experiments.phase2.portfolio_options import PortfolioInvocation
from merlin_experiments.phase2.portfolio_sandbox import PortfolioSandboxFactory


@dataclass(frozen=True)
class PortfolioWorkerContext:
    snapshot_root: Path
    functional_runs_root: Path
    contract_root: Path
    compiler_shared_source_root: Path
    controller_source: Path
    prior_shared_source_relative: Path
    prior_shared_source_fallback: Path
    guidance_contract: Path | None
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs
    target_experiment: TargetExperiment
    native_verifier_layout: RESUME.NativeVerifierLayout | None = None


def configure_global_analysis(
    experiment: GlobalPerfExperiment,
    *,
    target_experiment: TargetExperiment,
    agent_inputs: AW.AgentInputSnapshot,
    frozen_functional: Any,
    frozen_corpus_manifest: Path,
    stage_root: Path,
    sandbox_inputs: PC.PackageSandboxInputs | PC.FrozenPackageSandboxInputs,
) -> None:
    """Configure analysis from declared optional completion and explicit sandbox inputs."""
    if experiment.analysis.completion_contract is None:
        try:
            backend = get_backend(experiment.inputs.target)
        except KeyError:
            backend = None
        if backend is not None:
            missing = object()
            adapter = getattr(backend, "completion_contract", missing)
            if adapter is not missing:
                if not isinstance(adapter, ModuleType) or not callable(
                    getattr(adapter, "derive_completion_contract", None)
                ):
                    raise ValueError("selected backend has malformed completion_contract capability")
                experiment.analysis.completion_contract = adapter.derive_completion_contract()
    if experiment.analysis.analyzer is EA.analyze_whole_model_emission:
        sandbox_factory = PortfolioSandboxFactory(
            experiment.analysis,
            target_experiment=target_experiment,
            agent_inputs=agent_inputs,
            frozen_functional=frozen_functional,
            frozen_corpus_manifest=frozen_corpus_manifest,
            sandbox_inputs=sandbox_inputs,
        )
        sandbox_factory.install_worker(output=stage_root / "host_analysis_workers")


def declared_instruction_set_brief(target: str) -> dict[str, Any]:
    """The instruction set this TARGET declares, derived from its own facts.

    THE AGENT'S ONLY VIEW OF THE MACHINE used to be a count of the instructions it had already
    emitted: an artifact of the live round carries zero mentions of the instruction set anywhere in
    the prompt. A compiler cannot reach for a capability nobody told it about, and the largest
    structural levers are exactly the declared instructions a lowering never selects.

    Fail-closed: an underivable table is an explicit UNKNOWN, never an empty set. An empty declared
    set would make every emitted instruction undeclared and nothing unreached, which reads as full
    coverage produced by not looking.
    """
    try:
        from merlin.perf.task_instruction_evidence import declared_instruction_set, target_instruction_facts

        return declared_instruction_set(target_instruction_facts(target))
    except Exception as exc:  # noqa: BLE001 - an underivable ISA is stated, never assumed complete
        return {
            "schema": "declared_instruction_set_v1",
            "status": "UNKNOWN",
            "reason": (
                f"this target's declared instruction set could not be derived: {type(exc).__name__}: {str(exc)[:200]}"
            ),
            "target": target,
            "custom_opcode": None,
            "declared_count": None,
            "instructions": [],
        }


def validate_optimization_baseline_resume(checkpoint: Mapping[str, Any], *, optimization_baseline_sha256: str) -> None:
    """Legacy checkpoints used Phase 1 for comparison; a new segment must retain that choice."""
    previous = checkpoint.get("optimization_baseline_sha256", checkpoint.get("baseline_sha256"))
    if not BE._is_sha256(optimization_baseline_sha256) or previous != optimization_baseline_sha256:
        raise ValueError("resume checkpoint optimization baseline differs; select a new experiment explicitly")


def run_analysis_only(
    experiment,
    candidate: Path,
    *,
    stage_root: Path,
    static_analysis_seed_checkpoint: Path | None = None,
    static_analysis_seed_sha256: str | None = None,
    **analysis_policy,
) -> int:
    """Use the real compile/static-analysis boundary without authoring, probes or a seal."""
    configure_global_analysis(experiment, stage_root=stage_root, **analysis_policy)
    if static_analysis_seed_checkpoint is not None:
        experiment.static_analysis_import.import_checkpoint(
            candidate, checkpoint=static_analysis_seed_checkpoint, checkpoint_sha256=static_analysis_seed_sha256
        )
    analysis = experiment.analysis.analyze(
        candidate, hypothesis="Host-requested full-objective compile/static preflight"
    )
    result = {
        "schema": "global_analysis_only_v1",
        "readiness": analysis["readiness"],
        "candidate_sha256": analysis["candidate_sha256"],
        "portfolio_sha256": experiment.inputs.portfolio_identity_sha256,
        "portfolio_members_ready": analysis["portfolio"]["members_ready"],
        "portfolio_members_total": analysis["portfolio"]["members_total"],
        "baseline_sha256": experiment.inputs.baseline_sha256,
        "optimization_baseline_sha256": experiment.inputs.optimization_baseline_sha256,
        "iteration_record": str(experiment.output / f"iteration_{analysis['iteration']:04d}.json"),
        "phase1_rerun": False,
        "authoring_launched": False,
        "simulators_executed": False,
        "candidate_sealed": False,
        "objective_numerical_qualification": "UNPROVEN",
        "global_speedup_proven": False,
    }
    P2_CONTRACTS.write_json(stage_root / "analysis_only.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if analysis["readiness"]["status"] == "ready_for_probe_admission" else 1


def run_authoring_with_terminal_receipt(stage_root: Path, *, configure, sequence):
    """Persist pre-round/setup exceptions too; retain the original exception and fail closed."""
    stage = "configure_global_analysis"
    try:
        configure()
        stage = "initial_analysis_or_authoring_sequence"
        return sequence()
    except Exception as exc:
        P2_CONTRACTS.write_json(
            stage_root / "terminal_failure.json",
            {
                "schema": "global_launch_terminal_failure_v1",
                "status": "failed",
                "stage": stage,
                "exception": type(exc).__name__,
                "reason": str(exc),
                "completed_round_receipts": len(list((stage_root / "global_iterations").glob("agent_round_*.json"))),
                "iteration_receipts": [
                    str(path) for path in sorted((stage_root / "global_iterations").glob("iteration_*.json"))
                ],
                "promotion_status": "unqualified",
                "global_speedup_proven": False,
                "phase1_rerun": False,
                "live_handle_restarted": False,
            },
        )
        raise


def load_host_guidance_declarations(contract_path: Path, catalog_receipt):
    """Load only the inventory explicitly pinned by the supplied host catalog receipt.

    Legacy catalogs without this pin retain manifest-only guidance. Mere proximity
    to the edit contract is not authority to load an unpinned semantic inventory.
    The controller subsequently checks every declaration against actual AST and
    component ownership and the unchanged edit contract.
    """
    digest = catalog_receipt.get("guidance_inventory_sha256")
    if digest is None:
        digest = (catalog_receipt.get("unchanged_catalog_file_sha256") or {}).get("inventory.json")
    if digest is None:
        return None
    path = contract_path.parent / "inventory.json"
    if not BE._is_sha256(digest) or path.is_symlink():
        raise ValueError("host guidance inventory pin/path is invalid")
    raw = path.read_bytes()
    if len(raw) > 4_000_000 or sha256_bytes(raw) != digest:
        raise ValueError("host guidance inventory differs from pinned catalog bytes")
    inventory = json.loads(raw)
    if not isinstance(inventory, dict) or not isinstance(inventory.get("surfaces"), list):
        raise ValueError("host guidance inventory has no surface declarations")
    return inventory["surfaces"]


def run(invocation: PortfolioInvocation, config: Mapping[str, Any], *, context: PortfolioWorkerContext) -> int:
    """Admit and run one frozen worker without native-controller or checkout discovery."""
    args = invocation.args
    resource_policy = invocation.resource_policy
    fast_evaluation_configured = invocation.fast_evaluation_configured
    total_authoring = invocation.total_authoring
    from merlin_experiments import source_snapshot as perf_snapshot

    snapshot_receipt = perf_snapshot.verify(context.snapshot_root)
    if snapshot_receipt["schema"] == perf_snapshot.SCHEMA:
        config = dict(config)
        for key in ("descriptor", "telemetry_price_table"):
            if config.get(key) is not None:
                config[key] = str(
                    perf_snapshot.remap_input(context.snapshot_root, snapshot_receipt, Path(config[key]), name=key)
                )
    from merlin_experiments.phase1.providers.agent_bridge import bind_frozen_proxy_config

    proxy_config_identity = bind_frozen_proxy_config(context.snapshot_root, snapshot_receipt, os.environ)
    target = context.target_experiment
    descriptor = Path(config["descriptor"])
    if target.path.resolve() != descriptor.resolve() or target.descriptor_sha256 != P2_CONTRACTS.sha256_file(
        descriptor
    ):
        raise ValueError("worker target descriptor differs from the frozen campaign selection")
    run_root = context.functional_runs_root
    functional = FI.inspect_stage_functional_run(
        run_root,
        str(config["functional_run_id"]),
        str(config["functional_submission_sha256"]),
        waive=tuple(config["waive_functional_gate"]),
    )
    gaps = tuple(
        sorted(str(row["capsule"]) for row in functional.public_score["per_capsule"] if row.get("status") != "pass")
    )
    # Expected Phase 1 counts are independent operator assertions, not values
    # inferred from the same receipts being checked.
    try:
        expected_passed = int(config["functional_expected_public_passed"])
        expected_total = int(config["functional_expected_public_total"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "campaign config must declare functional_expected_public_passed and "
            "functional_expected_public_total for the pinned phase-1 run"
        ) from exc
    phase1 = FrozenPhase1(
        run_root,
        functional.run_id,
        functional.digest,
        tuple(config["waive_functional_gate"]),
        expected_passed,
        expected_total,
        gaps,
    )
    phase1.verify(functional.submission_dir)
    if args.output.exists():
        raise ValueError("macro stage output must be fresh")
    stage_root = args.output.resolve()
    stage_root.mkdir(parents=True)
    base = PC.materialize_perf_workspace(functional, stage_root / "_frozen_functional")
    frozen_functional = FI.load_frozen_functional_inputs(functional)
    from merlin.perf.external_objective import load_external_objective

    primary_external = (
        load_external_objective(
            args.external_objective.resolve(),
            spec_sha256=args.external_objective_sha256,
            max_source_bytes=64 * 1024 * 1024,
        )
        if args.external_objective
        else None
    )
    portfolio_externals = [
        load_external_objective(path.resolve(), spec_sha256=digest, max_source_bytes=64 * 1024 * 1024)
        for path, digest in zip(
            args.portfolio_external_objective, args.portfolio_external_objective_sha256, strict=True
        )
    ]
    external_objectives = ([primary_external] if primary_external is not None else []) + portfolio_externals
    corpus = P2_CORPUS.freeze_performance_corpus(
        P2_CORPUS.discover_performance_corpus(target, families="all", capsules="all"), stage_root / "_frozen_corpus"
    )
    inputs = AW.build_answer_free_agent_inputs(
        corpus,
        target,
        stage_root / "_agent_inputs",
        external_objective=(primary_external if primary_external is not None and not portfolio_externals else None),
        external_objectives=(tuple(external_objectives) if portfolio_externals else ()),
    )
    sentinel = (
        INPUTS.select_external_e2e_sentinel(primary_external, inputs)
        if args.external_objective
        else INPUTS.select_e2e_sentinel(
            functional,
            frozen_functional,
            target,
            objective_capsule=args.objective_capsule,
            source_root=context.snapshot_root,
        )
    )
    portfolio_sentinels = [
        INPUTS.select_e2e_sentinel(
            functional, frozen_functional, target, objective_capsule=capsule, source_root=context.snapshot_root
        )
        for capsule in args.portfolio_capsule
    ]
    portfolio_sentinels.extend(
        INPUTS.select_external_e2e_sentinel(external, inputs) for external in portfolio_externals
    )
    portfolio_identity = full_model_portfolio_identity((sentinel, *portfolio_sentinels))
    portfolio_sha256 = P2_CONTRACTS.document_sha256(portfolio_identity)
    fast_installation, fast_installation_receipt = _prepare_fast_evaluator_installation(
        sentinels=(sentinel, *portfolio_sentinels),
        target_sha256=target.descriptor_sha256,
        stage_root=stage_root,
        configured=fast_evaluation_configured,
        calibration=args.fast_evaluation_calibration,
        calibration_sha256=args.fast_evaluation_calibration_sha256,
        quality_observer=args.fast_evaluation_quality_observer,
        quality_observer_sha256=args.fast_evaluation_quality_observer_sha256,
        quality_observer_symbol=args.fast_evaluation_quality_observer_symbol,
        classification_member_sha256=args.fast_evaluation_classification_member_sha256,
        corpora=args.fast_evaluation_held_out_corpus,
        maximum_model_seconds=args.fast_evaluation_maximum_model_seconds,
    )
    fast_experiment_kwargs = fast_installation.experiment_kwargs() if fast_installation is not None else {}
    resumed = None
    if args.resume_checkpoint:
        resumed = RESUME.verify(args.resume_checkpoint.resolve(), native_layout=context.native_verifier_layout)
        validate_optimization_baseline_resume(
            resumed,
            optimization_baseline_sha256=(
                args.optimization_baseline_sha256 if args.optimization_baseline else functional.digest
            ),
        )
        resumed_portfolio_sha256 = resumed.get("portfolio_sha256")
        resume_portfolio_matches = resumed_portfolio_sha256 == portfolio_sha256 or (
            resumed_portfolio_sha256 is None and len(portfolio_sentinels) == 0
        )
        if (
            resumed["candidate_sha256"] != hash_tree(args.candidate)["sha256"]
            or resumed["baseline_sha256"] != functional.digest
            or resumed["target_sha256"] != target.descriptor_sha256
            or resumed["capsule_sha256"] != sentinel.capsule_sha256
            or not resume_portfolio_matches
            or resumed["phase1_qualification"] != phase1.verify(base)
        ):
            raise ValueError("resume checkpoint candidate, objective, target or frozen qualification differs")
    candidate = AW.fresh_round_workspace(
        args.candidate.resolve(), stage_root / "agent_workspaces" / "round_00", hash_tree(args.candidate)["sha256"]
    )
    codex, resolved_model = None, None
    if not args.validation_only and not args.analysis_only:
        codex = P2_CONTRACTS.require_executable("codex", label="Codex")
        telemetry = TEL.prepare(
            authoring_stage=Path(AUTHORING.__file__).resolve(),
            model=str(config["model"]),
            price_table=Path(config["telemetry_price_table"]),
            codex_binary=codex,
        )
        P2_CONTRACTS.write_json(stage_root / "telemetry_preflight.json", telemetry)
        resolved_model = str(telemetry["model_resolution"]["resolved_model"])
    # Validate the gate config at launch, before any budget is spent, so a bad path or a
    # malformed expectation fails here rather than as a silent per-iteration `not_run`.
    functional_gate = load_functional_gate_config(args.functional_gate) if args.functional_gate else None
    if functional_gate is not None:
        # A gate that cannot fail is refused HERE, at launch, and not discovered after a campaign
        # cited it: a gate that checks a defect count and never how many values were compared is
        # passed by a program that compared none.
        require_discriminating(
            functional_gate.gate_spec, json.loads(Path(args.functional_gate).read_text(encoding="utf-8"))
        )
    experiment = GlobalPerfExperiment(
        contract_root=context.contract_root,
        compiler_shared_source_root=context.compiler_shared_source_root,
        controller_source=context.controller_source,
        prior_shared_source_relative=context.prior_shared_source_relative,
        prior_shared_source_fallback=context.prior_shared_source_fallback,
        guidance_contract=context.guidance_contract,
        baseline=base,
        baseline_sha256=functional.digest,
        sentinel=sentinel,
        portfolio_sentinels=portfolio_sentinels,
        target=target.target,
        target_sha256=target.descriptor_sha256,
        target_descriptor=target.path,
        phase1=phase1,
        optimization_baseline=args.optimization_baseline,
        optimization_baseline_sha256=args.optimization_baseline_sha256,
        optimization_baseline_reason=args.optimization_baseline_reason,
        historical_reference_path=args.historical_reference,
        historical_reference_sha256=args.historical_reference_sha256,
        baseline_emission_cache=args.baseline_emission_cache,
        baseline_emission_seed_runs=tuple(args.baseline_emission_cache_seed_run),
        portfolio_analysis_workers=args.portfolio_analysis_workers,
        minimum_memory_available_bytes=resource_policy.minimum_memory_available_bytes,
        source_snapshot_root=context.snapshot_root,
        source_snapshot_files_sha256=P2_CONTRACTS.document_sha256(snapshot_receipt["files"]),
        output=stage_root / "global_iterations",
        timeout_s=args.iteration_seconds,
        functional_gate=functional_gate,
        **fast_experiment_kwargs,
    )
    sandbox_inputs = context.sandbox_inputs
    from merlin_experiments.phase2.portfolio_resume import installed_dispatch

    checkpoint_verifier = installed_dispatch(
        snapshot=experiment.inputs.source_snapshot_root,
        controller_source=experiment.inputs.controller_source,
        contract_root=experiment.inputs.contract_root,
        compiler_shared_source_root=experiment.inputs.compiler_shared_source_root,
    )
    if args.analysis_only:
        P2_CONTRACTS.write_json(
            stage_root / "launch.json",
            {
                "schema": "global_agent_launch_v1",
                "mode": "analysis_only",
                "checkpoint_verifier": checkpoint_verifier,
                "historical_reference": experiment.inputs.historical_reference,
                "source_config": str(args.campaign_config.resolve()),
                "source_config_sha256": P2_CONTRACTS.sha256_file(args.campaign_config),
                "source_snapshot": str(context.snapshot_root),
                "source_snapshot_files_sha256": P2_CONTRACTS.document_sha256(snapshot_receipt["files"]),
                "proxy_config": proxy_config_identity,
                "phase1": experiment.inputs.phase1_binding,
                "baseline_sha256": experiment.inputs.baseline_sha256,
                "optimization_baseline": experiment.inputs.optimization_baseline_binding,
                "optimization_baseline_sha256": experiment.inputs.optimization_baseline_sha256,
                "candidate_source_sha256": hash_tree(args.candidate)["sha256"],
                "objective": sentinel.capsule,
                "capsule_sha256": sentinel.capsule_sha256,
                "portfolio": experiment.inputs.portfolio_identity,
                "portfolio_sha256": experiment.inputs.portfolio_identity_sha256,
                "fast_evaluation_installation": fast_installation_receipt,
                "baseline_emission_cache": experiment.analysis.baseline_emission_cache_binding,
                "baseline_emission_cache_seeds": experiment.analysis.baseline_emission_cache_seeds,
                "external_objective": primary_external.record() if primary_external is not None else None,
                "external_objectives": [external.record() for external in external_objectives],
                "external_objective_spec_sha256": args.external_objective_sha256,
                "portfolio_external_objective_spec_sha256": args.portfolio_external_objective_sha256,
                "full_model_simulation_allowed": False,
                "simulators_enabled": False,
                "functional_gate": functional_gate.to_dict() if functional_gate else None,
                "static_analysis_seed": (
                    {
                        "path": str(args.static_analysis_seed_checkpoint.resolve()),
                        "sha256": args.static_analysis_seed_sha256,
                    }
                    if args.static_analysis_seed_checkpoint
                    else None
                ),
                "authoring_launched": False,
                "iteration_seconds": args.iteration_seconds,
                "host_resource_policy": resource_policy.record(),
            },
        )
        return run_analysis_only(
            experiment,
            candidate,
            stage_root=stage_root,
            target_experiment=target,
            agent_inputs=inputs,
            frozen_functional=frozen_functional,
            frozen_corpus_manifest=corpus.manifest_path,
            sandbox_inputs=sandbox_inputs,
            static_analysis_seed_checkpoint=(
                args.static_analysis_seed_checkpoint.resolve() if args.static_analysis_seed_checkpoint else None
            ),
            static_analysis_seed_sha256=args.static_analysis_seed_sha256,
        )
    if not args.validation_only:
        from merlin.perf.agent_guidance import build_compiler_edit_contract

        source_pins = None
        host_surfaces = None
        if args.edit_contract:
            contract = P2_CONTRACTS.mapping_file(args.edit_contract)
            catalog_receipt = P2_CONTRACTS.mapping_file(args.edit_contract.parent / "receipt.json")
            if catalog_receipt.get("contract_sha256") != contract.get("sha256"):
                raise ValueError("host edit catalog receipt does not bind the supplied contract")
            source_pins = catalog_receipt["source_files"]
            host_surfaces = load_host_guidance_declarations(args.edit_contract, catalog_receipt)
        else:
            contract = build_compiler_edit_contract(inspect_compiler_package(candidate, contract=context.contract_root))
        experiment.freeze_edit_scope(
            candidate, contract, source_pins=source_pins, host_surface_declarations=host_surfaces
        )
        if args.mechanism_catalog:
            experiment.freeze_mechanism_catalog(args.mechanism_catalog, args.mechanism_catalog_sha256)
        if args.mechanism_work_order:
            experiment.freeze_mechanism_work_order(
                args.mechanism_work_order, args.mechanism_work_order_sha256, candidate=candidate
            )
    from merlin.runtime.backends.base import get_backend
    from merlin_experiments.phase2.portfolio_providers import assemble

    providers = assemble(
        target=target.target,
        backend=get_backend(target.target),
        output=stage_root,
        semantic_only=args.semantic_only,
        probe_interface=args.probe_interface,
        probe_runtime_receipt=args.probe_runtime_receipt,
        profile_counters=args.probe_profile == "occupancy",
    )
    provider = providers.provider
    semantic_provider = providers.semantic_provider
    context_provider = providers.context_provider
    paired_context_provider = providers.paired_context_provider
    source_pair_provider = providers.source_pair_provider
    P2_CONTRACTS.write_json(
        stage_root / "launch.json",
        {
            "schema": "global_agent_launch_v1",
            "checkpoint_verifier": checkpoint_verifier,
            "source_config": str(args.campaign_config.resolve()),
            "historical_reference": experiment.inputs.historical_reference,
            "source_config_sha256": P2_CONTRACTS.sha256_file(args.campaign_config),
            "source_snapshot": str(context.snapshot_root),
            "source_snapshot_files_sha256": P2_CONTRACTS.document_sha256(snapshot_receipt["files"]),
            "proxy_config": proxy_config_identity,
            "phase1": experiment.inputs.phase1_binding,
            "candidate_source": str(args.candidate.resolve()),
            "baseline_sha256": experiment.inputs.baseline_sha256,
            "optimization_baseline_sha256": experiment.inputs.optimization_baseline_sha256,
            "optimization_baseline": experiment.inputs.optimization_baseline_binding,
            "candidate_source_sha256": hash_tree(args.candidate)["sha256"],
            "host_resource_policy": resource_policy.record(),
            "objective": sentinel.capsule,
            "model": resolved_model,
            "portfolio": experiment.inputs.portfolio_identity,
            "portfolio_sha256": experiment.inputs.portfolio_identity_sha256,
            "fast_evaluation_installation": fast_installation_receipt,
            "baseline_emission_cache": experiment.analysis.baseline_emission_cache_binding,
            "baseline_emission_cache_seeds": experiment.analysis.baseline_emission_cache_seeds,
            "requested_objective_capsule": args.objective_capsule,
            "requested_portfolio_capsules": args.portfolio_capsule,
            "external_objective": primary_external.record() if primary_external is not None else None,
            "external_objectives": [external.record() for external in external_objectives],
            "external_objective_roles": (["primary"] if primary_external is not None else [])
            + ["training"] * len(portfolio_externals),
            "external_objective_spec_sha256": args.external_objective_sha256,
            "portfolio_external_objective_spec_sha256": args.portfolio_external_objective_sha256,
            "mode": "checkpoint_validation" if args.validation_only else "agent_authoring",
            "comparison_candidate": str(args.comparison_candidate.resolve()) if args.comparison_candidate else None,
            "comparison_candidate_sha256": hash_tree(args.comparison_candidate)["sha256"]
            if args.comparison_candidate
            else None,
            "simulators_enabled": not args.semantic_only and (provider is not None or context_provider is not None),
            "full_model_simulation_allowed": False,
            "functional_gate": functional_gate.to_dict() if functional_gate else None,
            "probe_interface_sha256": provider.short_interface_sha256 if provider else None,
            "probe_runtime_receipt_sha256": provider.runtime_receipt_sha256 if provider else None,
            "probe_adapter_sha256": P2_CONTRACTS.sha256_file(Path(provider.adapter.__file__)) if provider else None,
            "probe_profile_scope": "isolated_primitive_" + args.probe_profile,
            "changed_region_semantic_provider": type(semantic_provider).__name__ if semantic_provider else None,
            "changed_region_native_abi": semantic_provider.abi_provenance if semantic_provider else None,
            "controlled_context_provider": type(context_provider).__name__ if context_provider else None,
            "paired_context_provider": type(paired_context_provider).__name__ if paired_context_provider else None,
            "complete_source_pair_provider": type(source_pair_provider).__name__ if source_pair_provider else None,
            "validation_context_scope": (
                "none_semantic_only"
                if args.semantic_only
                else "controlled_fixed_work_slice"
                if args.compare_controlled_context
                else "controlled_source_prefix"
            ),
            "iteration_seconds": args.iteration_seconds,
            "round_seconds": args.round_seconds,
            "maximum_rounds": args.max_rounds,
            "total_authoring_seconds": total_authoring,
            "on_round_failure": args.on_round_failure,
            "compiler_mechanism_catalog": copy.deepcopy(experiment.mechanism_program.catalog_binding),
            "compiler_mechanism_work_order": copy.deepcopy(experiment.mechanism_program.work_order_binding),
            "static_analysis_seed": (
                {
                    "path": str(args.static_analysis_seed_checkpoint.resolve()),
                    "sha256": args.static_analysis_seed_sha256,
                    "policy": "exact_content_hit_or_cold_analysis_miss",
                }
                if args.static_analysis_seed_checkpoint
                else None
            ),
            "resume_checkpoint": (
                {
                    "path": str(args.resume_checkpoint.resolve()),
                    "sha256": P2_CONTRACTS.sha256_file(args.resume_checkpoint),
                    "candidate_sha256": resumed["candidate_sha256"],
                    "previous_host_policy": resumed["host_verification_policy"],
                    "current_host_policy": experiment.inputs.host_policy,
                    "policy_transition": (
                        "explicit_new_segment_with_exact_static_seed_or_fresh_analysis"
                        if args.static_analysis_seed_checkpoint
                        else "explicit_new_segment_with_fresh_initial_model_analysis"
                    ),
                    "old_verdict_modified": False,
                }
                if resumed
                else None
            ),
        },
    )
    print(
        f"GLOBAL {'VALIDATION' if args.validation_only else 'AUTHORING'}: {stage_root} "
        f"models={len(experiment.inputs.portfolio_sentinels)} primary={sentinel.capsule} "
        "simulation="
        f"{'bounded_probes_only' if not args.semantic_only and (provider or context_provider) else 'disabled'}",
        flush=True,
    )
    if args.validation_only:
        try:
            if semantic_provider is None or (not args.semantic_only and context_provider is None):
                raise ValueError("deterministic validation lacks its selected evidence providers")
            configure_global_analysis(
                experiment,
                target_experiment=target,
                agent_inputs=inputs,
                frozen_functional=frozen_functional,
                frozen_corpus_manifest=corpus.manifest_path,
                sandbox_inputs=sandbox_inputs,
                stage_root=stage_root,
            )
            before = experiment.analysis.analyze(
                args.comparison_candidate.resolve(), hypothesis="Preserved pre-edit whole-model compiler"
            )
            after = experiment.analysis.analyze(candidate, hypothesis="Validate preserved generalized compiler edits")
            semantics = experiment.probes.qualify_changed_region(
                candidate, provider=semantic_provider, timeout_s=args.iteration_seconds
            )
            if args.semantic_only:
                probe_context = {}
            elif args.compare_controlled_context:
                if paired_context_provider is None:
                    raise ValueError("paired controlled-context provider is unavailable")
                probe_context = experiment.probes.compare_controlled_context(
                    candidate, provider=paired_context_provider, timeout_s=60
                )
            else:
                probe_context = experiment.probes.profile_controlled_context(
                    candidate, provider=context_provider, timeout_s=60
                )
            sealed = experiment.revision_session.seal(candidate)
            validation = {
                "schema": "global_checkpoint_validation_v1",
                "mode": "no_new_authoring",
                "phase1_action": "reuse_exact_snapshot_with_declared_waivers",
                "comparison_sha256": before["candidate_sha256"],
                "candidate_sha256": after["candidate_sha256"],
                "candidate_receipt": str(sealed),
                "semantic_status": semantics["evidence"].get("status"),
                "controlled_prefix_cycles": None
                if args.semantic_only or args.compare_controlled_context
                else probe_context["execution"]["total_compute_cycles"],
                "paired_fixed_work_cycles": probe_context.get("cycles"),
                "simulation_executed": not args.semantic_only,
                "full_model_cycles": None,
                "global_speedup_proven": False,
                "promotion_status": "unqualified_candidate_for_review",
            }
            P2_CONTRACTS.write_json(stage_root / "validation.json", validation)
            print(json.dumps(validation, indent=2), flush=True)
            return 0
        except Exception as exc:
            P2_CONTRACTS.write_json(
                stage_root / "validation_failure.json",
                {
                    "schema": "global_checkpoint_validation_failure_v1",
                    "exception": type(exc).__name__,
                    "reason": str(exc),
                    "global_speedup_proven": False,
                    "phase1_rerun": False,
                },
            )
            raise

    authoring = PortfolioAuthoring(
        experiment,
        target_experiment=target,
        stage_root=stage_root,
        agent_inputs=inputs,
        frozen_functional=frozen_functional,
        frozen_corpus_manifest=corpus.manifest_path,
        model=str(config["model"]),
        resolved_model=resolved_model,
        effort=str(config["effort"]),
        codex_binary=codex,
        max_tool_calls=args.max_tool_calls,
        sandbox_inputs=sandbox_inputs,
        declared_instruction_evidence=declared_instruction_set_brief(target.target),
        global_probe_provider=provider,
        global_semantic_provider=semantic_provider,
        global_context_provider=context_provider,
        global_paired_context_provider=paired_context_provider,
        global_source_pair_provider=source_pair_provider,
    )

    def configure_and_seed():
        configure_global_analysis(
            experiment,
            target_experiment=target,
            agent_inputs=inputs,
            frozen_functional=frozen_functional,
            frozen_corpus_manifest=corpus.manifest_path,
            sandbox_inputs=sandbox_inputs,
            stage_root=stage_root,
        )
        if args.static_analysis_seed_checkpoint:
            experiment.static_analysis_import.import_checkpoint(
                candidate,
                checkpoint=args.static_analysis_seed_checkpoint.resolve(),
                checkpoint_sha256=args.static_analysis_seed_sha256,
            )

    sequence = run_authoring_with_terminal_receipt(
        stage_root,
        configure=configure_and_seed,
        sequence=lambda: authoring.run_sequence(
            candidate,
            max_rounds=args.max_rounds,
            total_authoring_seconds=total_authoring,
            round_seconds=args.round_seconds,
            on_round_failure=args.on_round_failure,
        ),
    )
    try:
        # A bounded sequence may end with an exact blocked authoring checkpoint.  Preserve that
        # evidence for a follow-on segment, but never relabel it as the promotable global seal.
        # Ready, failure-free sequences still receive the conventional final review artifact.
        sealed = (
            experiment.revision_session.seal(Path(sequence["candidate"]))
            if not sequence["failures"] and sequence.get("promotion_ready") is True
            else Path(sequence["last_good_checkpoint"]["path"])
        )
    except Exception as exc:
        P2_CONTRACTS.write_json(
            stage_root / "terminal_failure.json",
            {
                "schema": "global_launch_terminal_failure_v1",
                "stage": "seal",
                "authoring_status": sequence["status"],
                "exception": type(exc).__name__,
                "reason": str(exc),
                "candidate_sha256": hash_tree(candidate)["sha256"],
                "promotion_status": "unqualified",
                "global_speedup_proven": False,
            },
        )
        raise
    print(
        json.dumps(
            {
                "authoring": sequence["status"],
                "candidate": str(sealed),
                "promotion": (
                    "unqualified" if sequence.get("promotion_ready") is True else "blocked_authoring_checkpoint"
                ),
                "full_model_timing": "UNMEASURED",
            },
            indent=2,
        )
    )
    return 0
