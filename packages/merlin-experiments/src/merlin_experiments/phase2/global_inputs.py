"""Admit, retain and reverify the immutable inputs of a global experiment.

Preparation reads selected inputs without creating output. Materialization owns
retained reference bytes and the optimization-comparison snapshot. Execution
state and scientific services remain with their concrete lifecycle owners.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree

from . import broker_evidence as BE
from . import campaign as PC
from . import contracts as P2_CONTRACTS
from . import host_policy as HP
from . import stage_inputs as INPUTS
from . import static_identity as SI
from .edit_authority import FrozenEditAuthority
from .mechanism_program import MechanismProgram
from .portfolio_checkpoint import load_historical_reference
from .portfolio_evaluation import FastPortfolioEvaluation
from .stage_inputs import sentinel_identity


@dataclass(frozen=True)
class FrozenPhase1:
    """Explicit launch inputs for reusing an existing qualification, never rerunning it."""

    runs_root: Path
    run_id: str
    submission_sha256: str
    waiver_predicates: tuple[str, ...]
    expected_public_passed: int
    expected_public_total: int
    expected_gap_ids: tuple[str, ...]

    def verify(self, baseline: Path) -> dict[str, Any]:
        frozen = PC.inspect_functional_run(
            self.runs_root, self.run_id, self.submission_sha256, waive=self.waiver_predicates
        )
        if hash_tree(baseline)["sha256"] != frozen.digest:
            raise ValueError("macro baseline is not the compiler from the frozen qualification")
        score = frozen.public_score
        gaps = sorted(str(row.get("capsule")) for row in score["per_capsule"] if row.get("status") != "pass")
        if (
            (score.get("n_passed"), score.get("n_capsules"))
            != (self.expected_public_passed, self.expected_public_total)
            or gaps != sorted(self.expected_gap_ids)
            or len(gaps) != self.expected_public_total - self.expected_public_passed
        ):
            raise ValueError("frozen qualification counts or exact known functional gap IDs changed")
        files = (
            "environment.yaml",
            "qa_loop_summary.yaml",
            "freeze.json",
            "run_manifest.yaml",
            "grading_public/score_capsule.json",
            "grading_hidden/score_capsule.json",
        )
        return {
            "schema": "global_frozen_phase1_binding_v1",
            "run_id": frozen.run_id,
            "run_dir": str(frozen.run_dir),
            "submission_sha256": frozen.digest,
            "public_passed": score["n_passed"],
            "public_total": score["n_capsules"],
            "known_functional_gap_ids": gaps,
            "waiver_predicates": list(self.waiver_predicates),
            "observed_deviations": [row.to_dict() for row in frozen.deviations],
            "evidence_sha256": {name: P2_CONTRACTS.sha256_file(frozen.run_dir / name) for name in files},
            "qualification_action": "read_existing_frozen_receipts_only",
        }


def full_model_portfolio_identity(sentinels: Sequence[INPUTS.StageE2ESentinel]) -> dict[str, Any]:
    """Canonical identity shared by launch/resume checks and experiment receipts."""
    if not sentinels:
        raise ValueError("a full-model optimization portfolio cannot be empty")
    return {
        "schema": "full_model_optimization_portfolio_v1",
        "members": [
            sentinel_identity(member, role=("primary" if index == 0 else "training"))
            for index, member in enumerate(sentinels)
        ],
        "selection": "multi_model_pareto_without_invented_static_cycle_total",
        "execution": "bounded_host_admitted_analysis_with_deterministic_record_order",
        "holdout_policy": "separate_post_authoring_evaluation",
        "micro_graphs": "smoke_and_mechanism_calibration_only",
    }


def current_machine_build_policy(target: str) -> Mapping[str, Any] | None:
    """Resolve the same optional non-executing object-build policy used by analysis workers."""
    from merlin.runtime.backends.base import get_backend

    try:
        backend = get_backend(target)
    except (ImportError, KeyError, ValueError):
        return {"schema": "machine_build_policy_not_supported_v1", "target": target, "cross_run_reuse_allowed": True}
    provider = getattr(backend, "machine_artifact_policy_identity", None)
    if provider is None:
        return {"schema": "machine_build_policy_not_supported_v1", "target": target, "cross_run_reuse_allowed": True}
    try:
        return SI.portable_machine_build_policy(provider(), verify_files=True)
    except (ImportError, KeyError, OSError, ValueError) as exc:
        return {
            "schema": "machine_build_policy_identity_unavailable_v1",
            "target": target,
            "failure_type": type(exc).__name__,
            "cross_run_reuse_allowed": False,
        }


@dataclass
class GlobalExperimentInputs:
    """Complete captured input ownership, shared by analysis and probe lifecycles."""

    baseline: Path
    baseline_sha256: str
    baseline_dependencies: Mapping[str, Any]
    compiler_shared_source_root: Path
    sentinel: INPUTS.StageE2ESentinel
    portfolio_sentinels: tuple[INPUTS.StageE2ESentinel, ...]
    target: str
    target_sha256: str
    target_descriptor: Path | None
    phase1: FrozenPhase1 | None
    phase1_binding: Mapping[str, Any] | None
    contract_root: Path
    controller_source: Path
    host_policy: Mapping[str, Any]
    machine_build_policy: Mapping[str, Any] | None
    source_snapshot_root: Path | None
    source_snapshot_files_sha256: str | None
    historical_reference_source: Path | None
    historical_reference: Mapping[str, Any] | None
    _historical_reference_binding_sha256: str
    optimization_baseline: Path
    optimization_baseline_sha256: str
    optimization_baseline_binding: Mapping[str, Any]
    optimization_baseline_binding_sha256: str
    portfolio_identity: Mapping[str, Any]
    portfolio_identity_sha256: str

    @classmethod
    def prepare(
        cls,
        *,
        baseline: Path,
        baseline_sha256: str,
        sentinel: INPUTS.StageE2ESentinel,
        target: str,
        target_sha256: str,
        output: Path,
        compiler_shared_source_root: Path,
        contract_root: Path,
        controller_source: Path,
        portfolio_sentinels: Sequence[INPUTS.StageE2ESentinel] = (),
        target_descriptor: Path | None = None,
        phase1: FrozenPhase1 | None = None,
        optimization_baseline: Path | None = None,
        optimization_baseline_sha256: str | None = None,
        optimization_baseline_reason: str = "host-selected immutable optimization comparison seed",
        historical_reference_path: Path | None = None,
        historical_reference_sha256: str | None = None,
        source_snapshot_root: Path | None = None,
        source_snapshot_files_sha256: str | None = None,
        portfolio_analysis_workers: int = 1,
        minimum_memory_available_bytes: int = 0,
    ) -> _PreparedGlobalInputs:
        """Perform ordered input admission without creating any output."""
        if not BE._is_sha256(target_sha256):
            raise ValueError("the target descriptor must have an exact SHA-256 identity")
        if hash_tree(baseline)["sha256"] != baseline_sha256:
            raise ValueError("frozen Phase-1 compiler digest mismatch")
        if (optimization_baseline is None) != (optimization_baseline_sha256 is None):
            raise ValueError("optimization baseline requires both an explicit path and SHA-256")
        explicit_comparison = optimization_baseline is not None
        comparison_source = optimization_baseline if explicit_comparison else baseline
        comparison_sha256 = optimization_baseline_sha256 if explicit_comparison else baseline_sha256
        if not BE._is_sha256(comparison_sha256) or hash_tree(comparison_source)["sha256"] != comparison_sha256:
            raise ValueError("optimization baseline digest mismatch")
        if not optimization_baseline_reason.strip():
            raise ValueError("optimization baseline selection requires a reason")
        if explicit_comparison and (
            comparison_source.is_symlink() or any(path.is_symlink() for path in comparison_source.rglob("*"))
        ):
            raise ValueError("optimization baseline must contain real immutable source files")
        if output.exists():
            raise ValueError("global experiment output must be fresh")
        if (source_snapshot_root is None) != (source_snapshot_files_sha256 is None):
            raise ValueError("source snapshot requires both an explicit root and files digest")
        if source_snapshot_files_sha256 is not None and not BE._is_sha256(source_snapshot_files_sha256):
            raise ValueError("source snapshot files digest must be SHA-256")
        if (historical_reference_path is None) != (historical_reference_sha256 is None):
            raise ValueError("historical reference requires both explicit path and SHA-256")
        reference_raw, reference_brief = (
            load_historical_reference(historical_reference_path, historical_reference_sha256, target=target)
            if historical_reference_path is not None
            else (None, None)
        )
        shared_root = Path(compiler_shared_source_root).resolve()
        baseline_dependencies = SI.compiler_dependency_record(baseline, shared_source_root=shared_root)
        members = (sentinel, *tuple(portfolio_sentinels))
        member_hashes = [member.capsule_sha256 for member in members]
        if any(not BE._is_sha256(value) for value in member_hashes) or len(set(member_hashes)) != len(member_hashes):
            raise ValueError("complete-model portfolio members require distinct exact identities")
        if target_descriptor is not None and P2_CONTRACTS.sha256_file(target_descriptor) != target_sha256:
            raise ValueError("target descriptor digest mismatch")
        if portfolio_analysis_workers < 1 or minimum_memory_available_bytes < 0:
            raise ValueError("portfolio concurrency policy is invalid")
        phase1_binding = phase1.verify(baseline) if phase1 is not None else None
        resources = Path(contract_root).resolve()
        controller = Path(controller_source)
        host_policy = HP.build_record(controller_source=controller, contract_root=resources)
        machine_build_policy = current_machine_build_policy(target)
        admitted = _AdmittedGlobalInputs(
            baseline=baseline,
            baseline_sha256=baseline_sha256,
            baseline_dependencies=baseline_dependencies,
            compiler_shared_source_root=shared_root,
            sentinel=sentinel,
            portfolio_sentinels=members,
            target=target,
            target_sha256=target_sha256,
            target_descriptor=target_descriptor,
            phase1=phase1,
            phase1_binding=phase1_binding,
            contract_root=resources,
            controller_source=controller,
            host_policy=host_policy,
            machine_build_policy=machine_build_policy,
            source_snapshot_root=Path(source_snapshot_root) if source_snapshot_root is not None else None,
            source_snapshot_files_sha256=source_snapshot_files_sha256,
            historical_reference_source=historical_reference_path,
        )
        return _PreparedGlobalInputs(
            _output=Path(output).absolute(),
            _admitted=admitted,
            _comparison_source=comparison_source,
            _comparison_sha256=comparison_sha256,
            _explicit_comparison=explicit_comparison,
            _comparison_reason=optimization_baseline_reason,
            _reference_raw=reference_raw,
            _reference_brief=reference_brief,
            _reference_sha256=historical_reference_sha256,
        )

    def compiler_dependencies(self, candidate: Path) -> dict[str, Any]:
        return SI.compiler_dependency_record(candidate, shared_source_root=self.compiler_shared_source_root)

    def verify(
        self,
        *,
        edit_authority: FrozenEditAuthority,
        mechanism_program: MechanismProgram,
        fast_evaluation: FastPortfolioEvaluation,
    ) -> None:
        if (
            HP.build_record(controller_source=self.controller_source, contract_root=self.contract_root)
            != self.host_policy
        ):
            raise ValueError("host verification policy changed during the global experiment")
        if current_machine_build_policy(self.target) != self.machine_build_policy:
            raise ValueError("machine build toolchain policy changed during the global experiment")
        if P2_CONTRACTS.document_sha256(
            self.portfolio_identity
        ) != self.portfolio_identity_sha256 or self.portfolio_identity["members"] != [
            sentinel_identity(member, role=("primary" if index == 0 else "training"))
            for index, member in enumerate(self.portfolio_sentinels)
        ]:
            raise ValueError("complete-model portfolio identity changed during global search")
        if P2_CONTRACTS.document_sha256(self.historical_reference) != self._historical_reference_binding_sha256:
            raise ValueError("historical reference binding changed")
        fast_evaluation.check_integrity()
        if self.historical_reference is not None:
            path = Path(self.historical_reference["path"])
            if path.is_symlink() or P2_CONTRACTS.sha256_file(path) != self.historical_reference["sha256"]:
                raise ValueError("retained historical reference bytes changed")
        edit_authority.check_integrity()
        mechanism_program.check_integrity()
        if self.phase1 is not None and self.phase1.verify(self.baseline) != self.phase1_binding:
            raise ValueError("frozen Phase-1 qualification or waivers changed during global search")
        if (
            self.target_descriptor is not None
            and P2_CONTRACTS.sha256_file(self.target_descriptor) != self.target_sha256
        ):
            raise ValueError("target descriptor changed during global search")
        if hash_tree(self.baseline)["sha256"] != self.baseline_sha256:
            raise ValueError("frozen compiler changed during global search")
        if self.compiler_dependencies(self.baseline) != self.baseline_dependencies:
            raise ValueError("frozen baseline shared compiler dependencies changed during global search")
        if (
            P2_CONTRACTS.document_sha256(self.optimization_baseline_binding)
            != self.optimization_baseline_binding_sha256
            or hash_tree(self.optimization_baseline)["sha256"] != self.optimization_baseline_sha256
            or self.compiler_dependencies(self.optimization_baseline)
            != self.optimization_baseline_binding["compiler_dependencies"]
        ):
            raise ValueError("immutable optimization baseline or shared compiler dependencies changed")
        for member in self.portfolio_sentinels:
            source = Path(member.frozen_source_path)
            if P2_CONTRACTS.exact_tree_record(source)["sha256"] != member.capsule_sha256:
                raise ValueError(f"complete-model objective changed during global search: {member.capsule}")


@dataclass(frozen=True)
class _AdmittedGlobalInputs:
    baseline: Path
    baseline_sha256: str
    baseline_dependencies: Mapping[str, Any]
    compiler_shared_source_root: Path
    sentinel: INPUTS.StageE2ESentinel
    portfolio_sentinels: tuple[INPUTS.StageE2ESentinel, ...]
    target: str
    target_sha256: str
    target_descriptor: Path | None
    phase1: FrozenPhase1 | None
    phase1_binding: Mapping[str, Any] | None
    contract_root: Path
    controller_source: Path
    host_policy: Mapping[str, Any]
    machine_build_policy: Mapping[str, Any] | None
    source_snapshot_root: Path | None
    source_snapshot_files_sha256: str | None
    historical_reference_source: Path | None


@dataclass
class _PreparedGlobalInputs:
    """Private single-use capture; no execution authority exists before materialization."""

    _output: Path
    _admitted: _AdmittedGlobalInputs
    _comparison_source: Path
    _comparison_sha256: str
    _explicit_comparison: bool
    _comparison_reason: str
    _reference_raw: bytes | None
    _reference_brief: Mapping[str, Any] | None
    _reference_sha256: str | None
    _used: bool = False

    def materialize(self, output: Path) -> GlobalExperimentInputs:
        output = Path(output).absolute()
        if self._used:
            raise ValueError("global experiment inputs may be materialized only once")
        if output != self._output or output.is_symlink() or not output.is_dir():
            raise ValueError("global experiment inputs require their prepared output directory")
        self._used = True
        admitted = self._admitted
        historical = None
        if self._reference_raw is not None:
            reference_path = output / "historical_reference.json"
            with reference_path.open("xb") as stream:
                stream.write(self._reference_raw)
            reference_path.chmod(0o444)
            historical = {
                "path": str(reference_path.resolve()),
                "sha256": self._reference_sha256,
                "summary": self._reference_brief,
            }
        comparison = admitted.baseline
        if self._explicit_comparison:
            comparison = output / "optimization_baseline"
            shutil.copytree(self._comparison_source, comparison, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            if hash_tree(comparison)["sha256"] != self._comparison_sha256:
                raise ValueError("optimization baseline changed while capturing its immutable snapshot")
            for path in comparison.rglob("*"):
                path.chmod(path.stat().st_mode & ~0o222)
            comparison.chmod(0o555)
        binding = {
            "schema": "global_optimization_baseline_v1",
            "selection": "explicit_host_seed" if self._explicit_comparison else "frozen_phase1_compiler",
            "path": str(comparison.resolve()),
            "sha256": self._comparison_sha256,
            "compiler_dependencies": SI.compiler_dependency_record(
                comparison, shared_source_root=admitted.compiler_shared_source_root
            ),
            "reason": self._comparison_reason if self._explicit_comparison else "legacy frozen compiler comparison",
            "scope": "optimization comparison only; does not replace or extend frozen Phase-1 qualification",
            "objective_numerical_qualification": "UNPROVEN",
            "phase1_regraded": False,
        }
        portfolio = full_model_portfolio_identity(admitted.portfolio_sentinels)
        return GlobalExperimentInputs(
            baseline=admitted.baseline,
            baseline_sha256=admitted.baseline_sha256,
            baseline_dependencies=admitted.baseline_dependencies,
            compiler_shared_source_root=admitted.compiler_shared_source_root,
            sentinel=admitted.sentinel,
            portfolio_sentinels=admitted.portfolio_sentinels,
            target=admitted.target,
            target_sha256=admitted.target_sha256,
            target_descriptor=admitted.target_descriptor,
            phase1=admitted.phase1,
            phase1_binding=admitted.phase1_binding,
            contract_root=admitted.contract_root,
            controller_source=admitted.controller_source,
            host_policy=admitted.host_policy,
            machine_build_policy=admitted.machine_build_policy,
            source_snapshot_root=admitted.source_snapshot_root,
            source_snapshot_files_sha256=admitted.source_snapshot_files_sha256,
            historical_reference_source=admitted.historical_reference_source,
            historical_reference=historical,
            _historical_reference_binding_sha256=P2_CONTRACTS.document_sha256(historical),
            optimization_baseline=comparison,
            optimization_baseline_sha256=self._comparison_sha256,
            optimization_baseline_binding=binding,
            optimization_baseline_binding_sha256=P2_CONTRACTS.document_sha256(binding),
            portfolio_identity=portfolio,
            portfolio_identity_sha256=P2_CONTRACTS.document_sha256(portfolio),
        )
