"""Installed assembly of global portfolio analysis, probes and frozen authorities.

Source identities and resource selections are explicit constructor inputs. Native
compatibility defaults and sandbox/agent configuration remain at their launch edges.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.execution_policy import ITERATION_MAX_SECONDS
from merlin.perf.functional_gate import FunctionalGateConfig, FunctionalGateResult
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import stage_inputs as INPUTS
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.global_inputs import FrozenPhase1, GlobalExperimentInputs
from merlin_experiments.phase2.mechanism_program import MechanismProgram
from merlin_experiments.phase2.mechanism_rounds import MechanismRounds
from merlin_experiments.phase2.portfolio_analysis import PortfolioAnalysis
from merlin_experiments.phase2.portfolio_checkpoint import load_historical_reference
from merlin_experiments.phase2.portfolio_evaluation import FastPortfolioEvaluation
from merlin_experiments.phase2.portfolio_probes import PortfolioProbes
from merlin_experiments.phase2.revision_journal import RevisionJournal
from merlin_experiments.phase2.revision_session import RevisionSession
from merlin_experiments.phase2.static_analysis_import import StaticAnalysisImport


class GlobalPerfExperiment:
    """Host controller shared by interactive/agent search and the compile-only readiness CLI.

    ``analyzer`` and ``plan_verifier`` are host integrations, never candidate-imported entrypoints.
    A candidate can suggest a transformation but cannot assert its own equivalence or objective.
    Every novel submitted edit invokes full-model compilation, including candidates later rejected.
    Revisiting an exact earlier promotion-ready snapshot may reuse its immutable static analysis;
    the revisit still gets a new chronological iteration and never inherits probe/timing evidence.
    """

    def __init__(
        self,
        *,
        baseline: Path,
        baseline_sha256: str,
        sentinel: INPUTS.StageE2ESentinel,
        target: str,
        target_sha256: str,
        portfolio_sentinels: Sequence[INPUTS.StageE2ESentinel] = (),
        output: Path,
        timeout_s: int = 300,
        target_descriptor: Path | None = None,
        phase1: FrozenPhase1 | None = None,
        optimization_baseline: Path | None = None,
        optimization_baseline_sha256: str | None = None,
        optimization_baseline_reason: str = "host-selected immutable optimization comparison seed",
        historical_reference_path: Path | None = None,
        historical_reference_sha256: str | None = None,
        compiler_shared_source_root: Path,
        contract_root: Path,
        controller_source: Path,
        prior_shared_source_relative: Path,
        prior_shared_source_fallback: Path,
        guidance_contract: Path | None = None,
        baseline_emission_cache: Path | None = None,
        baseline_emission_seed_runs: Sequence[Path] = (),
        portfolio_analysis_workers: int = 1,
        minimum_memory_available_bytes: int = 0,
        source_snapshot_root: Path | None = None,
        source_snapshot_files_sha256: str | None = None,
        analyzer: Callable[..., Mapping[str, Any]] = EA.analyze_whole_model_emission,
        plan_verifier: Callable[..., Mapping[str, Any]] | None = None,
        fast_evaluation_provider: Callable[..., Mapping[str, Any]] | None = None,
        fast_evaluation_policy: Any | None = None,
        quality_budgets: Mapping[str, Any] | None = None,
        fast_evaluation_provider_binding: Mapping[str, Any] | None = None,
        functional_gate: FunctionalGateConfig | None = None,
        functional_gate_runner: Callable[..., FunctionalGateResult] | None = None,
    ):
        PortfolioAnalysis.validate_timeout(timeout_s)
        prepared_inputs = GlobalExperimentInputs.prepare(
            baseline=baseline,
            baseline_sha256=baseline_sha256,
            sentinel=sentinel,
            target=target,
            target_sha256=target_sha256,
            output=output,
            portfolio_sentinels=portfolio_sentinels,
            target_descriptor=target_descriptor,
            phase1=phase1,
            optimization_baseline=optimization_baseline,
            optimization_baseline_sha256=optimization_baseline_sha256,
            optimization_baseline_reason=optimization_baseline_reason,
            historical_reference_path=historical_reference_path,
            historical_reference_sha256=historical_reference_sha256,
            source_snapshot_root=source_snapshot_root,
            source_snapshot_files_sha256=source_snapshot_files_sha256,
            portfolio_analysis_workers=portfolio_analysis_workers,
            minimum_memory_available_bytes=minimum_memory_available_bytes,
            compiler_shared_source_root=compiler_shared_source_root,
            contract_root=contract_root,
            controller_source=controller_source,
        )
        self.output = output
        self.revisions = RevisionJournal(self.output)
        self.edit_authority = FrozenEditAuthority(
            self.output,
            **({"guidance_contract": guidance_contract} if guidance_contract is not None else {}),
        )
        self.output.mkdir(parents=True)
        self.inputs = prepared_inputs.materialize(self.output)
        self.mechanism_program = MechanismProgram(
            self.output,
            self.edit_authority,
            portfolio_identity=self.inputs.portfolio_identity,
            portfolio_identity_sha256=self.inputs.portfolio_identity_sha256,
        )
        self.mechanism_rounds = MechanismRounds(self.mechanism_program)
        self.fast_evaluation = FastPortfolioEvaluation(
            self.inputs.portfolio_sentinels,
            provider=fast_evaluation_provider,
            policy=fast_evaluation_policy,
            quality_budgets=quality_budgets,
            provider_binding=fast_evaluation_provider_binding,
        )
        self.revision_session = RevisionSession(
            inputs=self.inputs,
            edit_authority=self.edit_authority,
            mechanism_program=self.mechanism_program,
            mechanism_rounds=self.mechanism_rounds,
            fast_evaluation=self.fast_evaluation,
            journal=self.revisions,
        )
        self.analysis = PortfolioAnalysis(
            self.revision_session,
            timeout_s=timeout_s,
            portfolio_analysis_workers=portfolio_analysis_workers,
            minimum_memory_available_bytes=minimum_memory_available_bytes,
            analyzer=analyzer,
            plan_verifier=plan_verifier,
            functional_gate=functional_gate,
            functional_gate_runner=functional_gate_runner,
            baseline_emission_cache=baseline_emission_cache,
            baseline_emission_seed_runs=baseline_emission_seed_runs,
        )
        self.probes = PortfolioProbes(self.analysis)
        self.static_analysis_import = StaticAnalysisImport(
            self.analysis,
            prior_shared_source_relative=prior_shared_source_relative,
            prior_shared_source_fallback=prior_shared_source_fallback,
        )
        self._write(
            "experiment.json",
            {
                "schema": "global_perf_experiment_v1",
                "objective": "full_model_graph_and_global_plan",
                "historical_reference": self.inputs.historical_reference,
                "baseline_sha256": baseline_sha256,
                "target": target,
                "optimization_baseline_sha256": self.inputs.optimization_baseline_sha256,
                "optimization_baseline": self.inputs.optimization_baseline_binding,
                "baseline_compiler_dependencies": self.inputs.baseline_dependencies,
                "compiler_shared_source_root": self.inputs.baseline_dependencies["shared_source_root"],
                "target_sha256": target_sha256,
                "capsule": sentinel.capsule,
                "capsule_sha256": sentinel.capsule_sha256,
                "portfolio": self.inputs.portfolio_identity,
                "portfolio_sha256": self.inputs.portfolio_identity_sha256,
                "baseline_emission_cache": self.analysis.baseline_emission_cache_binding,
                "baseline_emission_cache_seeds": self.analysis.baseline_emission_cache_seeds,
                "portfolio_analysis_workers": self.analysis.portfolio_analysis_workers,
                "minimum_memory_available_bytes": self.analysis.minimum_memory_available_bytes,
                "fast_evaluation": copy.deepcopy(self.fast_evaluation.binding),
                "fast_evaluation_binding_sha256": self.fast_evaluation.binding_sha256,
                "maximum_iteration_seconds": timeout_s,
                "maximum_full_graph_static_analysis_seconds": timeout_s,
                "maximum_reduced_witness_seconds": int(ITERATION_MAX_SECONDS),
                "full_model_simulation_allowed": False,
                "probe_measurements_required": False,
                # Legacy v1 informational key; this selects no simulator or execution default.
                "firesim_stage": "optional_post_freeze_validation",
                "phase1_action": "reuse_exact_snapshot",
                "micro_plateau_stops_search": False,
                "launch_scope": (
                    "qualified_macro_experiment" if self.inputs.phase1_binding else "development_readiness_only"
                ),
                "phase1_qualification": self.inputs.phase1_binding,
                "host_verification_policy": self.inputs.host_policy,
                "source_snapshot": (
                    str(self.inputs.source_snapshot_root) if self.inputs.source_snapshot_root is not None else None
                ),
                "source_snapshot_files_sha256": self.inputs.source_snapshot_files_sha256,
                "machine_build_policy": self.inputs.machine_build_policy,
            },
        )

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.output / name
        payload = P2_CONTRACTS.canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return path

    def stage_historical_reference(self, control: Path, *, workspace: Path) -> None:
        """Use the existing read-only /perf-control grant, never the compiler sandbox."""
        self.revision_session.check_inputs()
        if self.inputs.historical_reference is None:
            return
        if self.inputs.historical_reference_source.resolve().is_relative_to(workspace.resolve()):
            raise ValueError("historical reference source is in the writable author workspace")
        source = Path(self.inputs.historical_reference["path"])
        raw, summary = load_historical_reference(
            source, self.inputs.historical_reference["sha256"], target=self.inputs.target, candidate_roots=(workspace,)
        )
        if summary != self.inputs.historical_reference["summary"] or control.resolve().is_relative_to(
            workspace.resolve()
        ):
            raise ValueError("historical reference summary or control boundary changed")
        path = control / "historical_reference.json"
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o444)

    def freeze_edit_scope(
        self,
        candidate: Path,
        contract: Mapping[str, Any],
        *,
        source_pins: Mapping[str, str] | None = None,
        host_surface_declarations: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Host startup only: freeze the externally approved edit authority."""
        return self.edit_authority.freeze(
            candidate,
            contract,
            has_iterations=bool(self.revisions.iterations),
            source_pins=source_pins,
            host_surface_declarations=host_surface_declarations,
        )

    def freeze_mechanism_catalog(self, source: Path, source_sha256: str) -> dict[str, Any]:
        binding = self.mechanism_program.freeze_catalog(
            source, source_sha256, has_iterations=bool(self.revisions.iterations)
        )
        self.revision_session.check_inputs()
        return binding

    def freeze_mechanism_work_order(self, source: Path, source_sha256: str, *, candidate: Path) -> dict[str, Any]:
        prepared = self.mechanism_program.prepare_work_order(
            source, source_sha256, has_iterations=bool(self.revisions.iterations)
        )
        self.revision_session.validate_candidate_scope(candidate)
        binding = self.mechanism_program.freeze_work_order(prepared, candidate_sha256=hash_tree(candidate)["sha256"])
        self.revision_session.check_inputs()
        return binding

    def run(
        self, propose: Callable[[Sequence[Mapping[str, Any]]], tuple[Path, str] | None], *, iterations: int
    ) -> tuple[dict[str, Any], ...]:
        """Drive repeated agent proposals through the mandatory full-model compilation boundary."""
        if iterations < 1:
            raise ValueError("iterations must be positive")
        for _ in range(iterations):
            proposal = propose(copy.deepcopy(tuple(self.revisions.iterations)))
            if proposal is None:
                break
            candidate, hypothesis = proposal
            self.analysis.analyze(candidate, hypothesis=hypothesis)
        return copy.deepcopy(tuple(self.revisions.iterations))
