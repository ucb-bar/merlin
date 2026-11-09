"""One-shot admission and publication of explicitly pinned cross-run static evidence.

Historical source ownership is verified before selecting the caller-declared shared
compiler sources. A cache hit reconstructs current sandbox policy without compilation
and reevaluates scientific readiness; it never inherits dynamic probe evidence.
"""

from __future__ import annotations

import copy
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin_experiments.phase2 import broker_evidence as BE
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import emission_analysis as EA
from merlin_experiments.phase2 import host_policy as HP
from merlin_experiments.phase2 import static_cache as SC
from merlin_experiments.phase2 import static_identity as SI
from merlin_experiments.phase2.portfolio_analysis import PortfolioAnalysis
from merlin_experiments.phase2.stage_inputs import sentinel_identity


def _verify_source_snapshot(root: Path, expected_files_sha256: str | None = None) -> dict[str, Any]:
    """Use the current trusted snapshot verifier, never code from a seed checkpoint."""
    from merlin_experiments import source_snapshot as perf_snapshot

    root = Path(root)
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or root.resolve() != root
        or root.stat().st_mode & 0o222
    ):
        raise ValueError("source snapshot root is relative, linked, mutable, or absent")
    receipt = perf_snapshot.verify(root)
    files_sha256 = P2_CONTRACTS.document_sha256(receipt.get("files"))
    if expected_files_sha256 is not None and files_sha256 != expected_files_sha256:
        raise ValueError("source snapshot files identity changed")
    return {"root": str(root), "files_sha256": files_sha256, "receipt": receipt}


def _source_snapshot_root_from_policy(record: Mapping[str, Any]) -> Path:
    """Locate the sealed snapshot that owns every absolute policy source in a pinned receipt."""
    if record.get("schema") == HP.V2_SCHEMA:
        raise ValueError("v2 host policy requires an explicit source_snapshot; locations are not ownership")
    if record.get("schema") == HP.SCHEMA:
        raise ValueError("v3 host policy requires an explicit source_snapshot; locations are not ownership")
    sources = record.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("seed host policy has no source paths")
    first = Path(next(iter(sources)))
    if not first.is_absolute():
        raise ValueError("seed host policy source is not absolute")
    for parent in first.parents:
        try:
            if len(list(parent.glob("snapshot.*.json"))) == 1 and all(
                Path(path).resolve().is_relative_to(parent.resolve()) for path in sources
            ):
                if (
                    not parent.is_absolute()
                    or parent.is_symlink()
                    or parent.resolve() != parent
                    or not parent.is_dir()
                    or parent.stat().st_mode & 0o222
                ):
                    continue
                return parent
        except OSError:
            continue
    raise ValueError("seed host policy is not owned by a verifiable source snapshot")


class StaticAnalysisImport:
    """Own one attempted seed import, including refusals and immutable miss receipts."""

    def __init__(
        self,
        analysis: PortfolioAnalysis,
        *,
        prior_shared_source_relative: Path,
        prior_shared_source_fallback: Path,
    ) -> None:
        self.analysis = analysis
        self.prior_shared_source_relative = Path(prior_shared_source_relative)
        self.prior_shared_source_fallback = Path(prior_shared_source_fallback)
        self._cross_run_seed_attempted = False

    def _record_cross_run_seed_miss(
        self, *, checkpoint: Path, checkpoint_sha256: str, current_binding: Mapping[str, Any], reason: str
    ) -> dict[str, Any]:
        receipt = {
            "schema": "global_cross_run_static_analysis_import_v1",
            "status": "miss",
            "reason": reason,
            "seed_checkpoint": {"path": str(checkpoint.resolve()), "sha256": checkpoint_sha256},
            "current_binding_sha256": current_binding["sha256"],
            "full_graph_compiler_invoked": False,
            "full_model_simulation_executed": False,
            "probe_or_timing_receipts_reused": False,
            "semantic_or_decision_feedback_reused": False,
        }
        path = SC.atomic_static_write(
            "cross_run_static_analysis_seed.json", receipt, output=self.analysis.session.journal.output
        )
        return {**receipt, "receipt": {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}}

    def _reconstruct_imported_compiler_sandboxes(
        self, submitted: Path, *, dependencies: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any] | None, dict[str, Any]]:
        """Prepare the current trusted answer masks without invoking the compiler.

        Static artifact portability does not make an old absolute bwrap command portable.  A
        production cache hit therefore asks the *current* host sandbox factory to rebuild its
        grants for the fresh immutable submission, then binds the exact resulting policies.  A
        lightweight development analyzer remains analysis-only and has no probe authority.
        """
        from merlin.perf.analysis_worker import IsolatedAnalysisWorker

        if not isinstance(self.analysis.analyzer, IsolatedAnalysisWorker):
            return None, {
                "schema": "cross_run_imported_compiler_sandbox_v1",
                "status": "unavailable_non_production_analyzer",
                "previous_probe_compilation_available": False,
                "compiler_invoked": False,
            }
        scratch = self.analysis.session.journal.output / "cross_run_imported_compiler_scratch_0000"
        if scratch.exists() or scratch.is_symlink():
            raise ValueError("imported compiler sandbox scratch must be fresh")
        scratch.mkdir(mode=0o700)
        sandboxes = self.analysis.analyzer.sandbox_factory(
            self.analysis.session.inputs.optimization_baseline, submitted, scratch
        )
        if not isinstance(sandboxes, Mapping):
            raise ValueError("current sandbox factory returned no bound compiler policies")
        expected = {
            "baseline": (
                self.analysis.session.inputs.optimization_baseline,
                self.analysis.session.inputs.optimization_baseline_binding["compiler_dependencies"],
            ),
            "candidate": (submitted, dependencies),
        }
        for arm, (package, arm_dependencies) in expected.items():
            policy = sandboxes.get(arm)
            if not isinstance(policy, Mapping):
                raise ValueError(f"current sandbox factory omitted the {arm} policy")
            prefix = policy.get("command_prefix")
            boundary = policy.get("bwrap_argv_length")
            if (
                Path(str(policy.get("package_path"))).resolve() != package.resolve()
                or Path(str(policy.get("scratch_path"))).resolve() != scratch.resolve()
                or policy.get("compiler_dependencies") != arm_dependencies
                or not isinstance(prefix, list)
                or any(not isinstance(value, str) for value in prefix)
                or type(boundary) is not int
                or not 0 < boundary < len(prefix)
                or not isinstance(policy.get("answer_surfaces"), list)
            ):
                raise ValueError(f"reconstructed {arm} sandbox has stale package, dependency, or policy identity")
            for directory, digest in (policy.get("overlay_trees") or {}).items():
                overlay = Path(directory)
                if (
                    not BE._is_sha256(digest)
                    or overlay.is_symlink()
                    or not overlay.is_dir()
                    or P2_CONTRACTS.exact_tree_record(overlay)["sha256"] != digest
                ):
                    raise ValueError(f"reconstructed {arm} sandbox dependency overlay changed")
        if any(scratch.iterdir()):
            raise ValueError("sandbox reconstruction unexpectedly populated compiler scratch")
        scratch.chmod(0o555)
        retained = copy.deepcopy(dict(sandboxes))
        policy_sha256 = P2_CONTRACTS.document_sha256(retained)
        receipt = {
            "schema": "cross_run_imported_compiler_sandbox_v1",
            "status": "prepared_from_current_trusted_factory",
            "candidate_sha256": dependencies["candidate_sha256"],
            "candidate_package": str(submitted.resolve()),
            "candidate_dependencies": copy.deepcopy(dict(dependencies)),
            "optimization_baseline_sha256": self.analysis.session.inputs.optimization_baseline_sha256,
            "optimization_baseline_package": str(self.analysis.session.inputs.optimization_baseline.resolve()),
            "optimization_baseline_dependencies": copy.deepcopy(
                self.analysis.session.inputs.optimization_baseline_binding["compiler_dependencies"]
            ),
            "policy_set_sha256": policy_sha256,
            "candidate_policy_sha256": P2_CONTRACTS.document_sha256(retained["candidate"]),
            "baseline_policy_sha256": P2_CONTRACTS.document_sha256(retained["baseline"]),
            "scratch": str(scratch.resolve()),
            "compiler_invoked": False,
            "previous_probe_compilation_available": True,
            "scope": "current trusted answer masks rebound to fresh immutable imported submission",
        }
        path = SC.atomic_static_write(
            "cross_run_imported_compiler_sandbox_0000.json", receipt, output=self.analysis.session.journal.output
        )
        receipt["receipt"] = {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}
        return retained, receipt

    def import_checkpoint(self, candidate: Path, *, checkpoint: Path, checkpoint_sha256: str) -> dict[str, Any]:
        """Import an exact explicitly pinned static checkpoint under the current verifier.

        This is deliberately not a run-directory search.  Identity mismatches are safe cache
        misses; malformed/tampered inputs fail closed.  A hit makes a fresh immutable submission,
        recomputes readiness, and carries no measurement, semantic, or decision receipt.
        """
        started = time.monotonic()
        if self._cross_run_seed_attempted or self.analysis.session.journal.iterations:
            raise ValueError("cross-run static analysis may be seeded exactly once before iterations")
        self._cross_run_seed_attempted = True
        self.analysis.session.check_inputs()
        self.analysis.session.validate_candidate_scope(candidate)
        candidate_sha256 = hash_tree(candidate)["sha256"]
        dependencies = self.analysis.session.inputs.compiler_dependencies(candidate)
        current_binding = self.analysis.cross_run_static_analysis_binding(
            candidate_sha256=candidate_sha256, compiler_dependencies=dependencies
        )
        checkpoint = Path(checkpoint)
        document = SC.load_pinned_read_only_mapping(
            checkpoint, checkpoint_sha256, label="static analysis seed checkpoint"
        )
        checkpoint = checkpoint.resolve()
        if document.get("schema") != "global_perf_candidate_v1":
            raise ValueError("static analysis seed is not a promotable global candidate checkpoint")
        if self.analysis.session.inputs.machine_build_policy.get("cross_run_reuse_allowed") is False:
            return self._record_cross_run_seed_miss(
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                current_binding=current_binding,
                reason="machine_toolchain_identity_unavailable",
            )

        prior_policy = document.get("host_verification_policy")
        if not isinstance(prior_policy, Mapping):
            raise ValueError("static analysis seed has no host verification policy")
        source_root_value = document.get("source_snapshot")
        source_root = (
            Path(source_root_value)
            if isinstance(source_root_value, str)
            else _source_snapshot_root_from_policy(prior_policy)
        )
        verified_source = _verify_source_snapshot(source_root, document.get("source_snapshot_files_sha256"))
        prior_shared_source = source_root / self.prior_shared_source_relative
        if not prior_shared_source.is_dir():
            # The caller explicitly selects compatibility sources for snapshots that do
            # not contain the declared shared-source location. No checkout is discovered here.
            prior_shared_source = self.prior_shared_source_fallback
        prior_policy_sha256 = SI.host_policy_content_sha256(prior_policy, source_root=source_root)
        # V2/V3 validate both identities in the package owner. V1 retains its exact
        # historical flat-source interpretation; versions never compare equal.
        if (
            prior_policy.get("schema") == "global_host_verification_policy_v1"
            and "location_sha256" in prior_policy
            and (
                prior_policy.get("location_sha256") != P2_CONTRACTS.document_sha256(prior_policy.get("sources"))
                or prior_policy.get("sha256") != prior_policy_sha256
            )
        ):
            raise ValueError("static analysis seed host policy contradicts its verified sources")
        if prior_policy.get("schema") != self.analysis.session.inputs.host_policy.get("schema"):
            return self._record_cross_run_seed_miss(
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                current_binding=current_binding,
                reason="host_verification_policy_version_changed",
            )
        if prior_policy_sha256 != self.analysis.session.inputs.host_policy["sha256"]:
            return self._record_cross_run_seed_miss(
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                current_binding=current_binding,
                reason="host_verification_policy_content_changed",
            )
        if self.analysis.session.inputs.source_snapshot_root is None:
            raise ValueError("cross-run static import requires the current sealed source snapshot")
        _verify_source_snapshot(
            self.analysis.session.inputs.source_snapshot_root, self.analysis.session.inputs.source_snapshot_files_sha256
        )
        if (
            SI.host_policy_content_sha256(
                self.analysis.session.inputs.host_policy, source_root=self.analysis.session.inputs.source_snapshot_root
            )
            != self.analysis.session.inputs.host_policy["sha256"]
        ):
            raise ValueError("current host policy contradicts its verified source snapshot")

        seed_binding = document.get("cross_run_static_analysis_binding")
        if not isinstance(seed_binding, Mapping):
            return self._record_cross_run_seed_miss(
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                current_binding=current_binding,
                reason="checkpoint_predates_static_analysis_bundle_v1",
            )
        seed_body = {key: value for key, value in seed_binding.items() if key not in ("schema", "sha256")}
        if (
            seed_binding.get("schema") != "global_cross_run_static_analysis_binding_v1"
            or seed_binding.get("sha256") != P2_CONTRACTS.document_sha256(seed_body)
            or seed_binding.get("host_verification_policy_content_sha256") != prior_policy_sha256
        ):
            raise ValueError("static analysis seed binding is malformed or contradicts its source")
        if seed_binding != current_binding:
            changed = sorted(
                key
                for key in set(seed_binding) | set(current_binding)
                if seed_binding.get(key) != current_binding.get(key)
            )
            return self._record_cross_run_seed_miss(
                checkpoint=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                current_binding=current_binding,
                reason="exact_content_identity_changed:" + ",".join(changed),
            )

        comparison = document.get("optimization_baseline")
        if not isinstance(comparison, Mapping):
            raise ValueError("static analysis seed lacks its optimization baseline binding")
        comparison_path_value = comparison.get("path")
        comparison_path = Path(comparison_path_value) if isinstance(comparison_path_value, str) else Path()
        if (
            not comparison_path.is_absolute()
            or comparison_path.is_symlink()
            or not comparison_path.is_dir()
            or hash_tree(comparison_path)["sha256"] != comparison.get("sha256")
            or SI.compiler_dependency_content_sha256(
                SI.compiler_dependency_record(comparison_path, shared_source_root=prior_shared_source)
            )
            != seed_binding["optimization_baseline"]["compiler_dependencies_content_sha256"]
            or SI.portable_optimization_baseline_binding(comparison) != seed_binding["optimization_baseline"]
        ):
            raise ValueError("static analysis seed optimization baseline bytes changed")

        seed_candidate_value = document.get("candidate_path")
        if not isinstance(seed_candidate_value, str):
            raise ValueError("static analysis seed candidate path is malformed")
        seed_candidate = Path(seed_candidate_value)
        if (
            not seed_candidate.is_absolute()
            or seed_candidate.is_symlink()
            or not seed_candidate.is_dir()
            or seed_candidate.parent.resolve() != checkpoint.parent
            or seed_candidate.stat().st_mode & 0o222
            or any(path.is_symlink() or path.stat().st_mode & 0o222 for path in seed_candidate.rglob("*"))
            or hash_tree(seed_candidate)["sha256"] != candidate_sha256
            or SI.compiler_dependency_content_sha256(
                SI.compiler_dependency_record(seed_candidate, shared_source_root=prior_shared_source)
            )
            != seed_binding["candidate_compiler_dependencies_content_sha256"]
        ):
            raise ValueError("static analysis seed candidate bytes or dependencies changed")
        iteration_path_value = document.get("iteration_record")
        if not isinstance(iteration_path_value, str):
            raise ValueError("static analysis seed iteration path is malformed")
        iteration_path = Path(iteration_path_value)
        if not iteration_path.is_absolute() or iteration_path.parent.resolve() != checkpoint.parent:
            raise ValueError("static analysis seed iteration escaped its experiment")
        iteration = SC.load_pinned_read_only_mapping(
            iteration_path, document.get("iteration_record_sha256"), label="static analysis seed iteration"
        )
        if (
            iteration.get("schema") != "global_perf_iteration_v1"
            or iteration.get("candidate_sha256") != candidate_sha256
            or iteration.get("cross_run_static_analysis_binding") != seed_binding
            or iteration.get("readiness", {}).get("status") != "ready_for_probe_admission"
            or P2_CONTRACTS.document_sha256(iteration.get("analysis")) != document.get("analysis_sha256")
        ):
            raise ValueError("static analysis seed iteration binding changed")

        bundle = SC.load_static_analysis_bundle(
            checkpoint_parent=checkpoint.parent,
            checkpoint_bundle_ref=document.get("static_analysis_bundle"),
            iteration_bundle_ref=iteration.get("static_analysis_bundle"),
            binding=seed_binding,
            candidate_sha256=candidate_sha256,
            portfolio_sha256=self.analysis.session.inputs.portfolio_identity_sha256,
            capsule_sha256s=[member.capsule_sha256 for member in self.analysis.session.inputs.portfolio_sentinels],
        )
        current_analyses: list[dict[str, Any]] = []
        current_artifacts: dict[str, dict[str, Any]] = {}
        for sentinel, member in zip(self.analysis.session.inputs.portfolio_sentinels, bundle, strict=True):
            analysis = member.analysis
            analysis["optimization_baseline"] = copy.deepcopy(
                self.analysis.session.inputs.optimization_baseline_binding
            )
            analysis["compiler_edit_scope"] = self.analysis.session.validate_candidate_scope(candidate)
            diagnostics = analysis.setdefault("diagnostics", {})
            diagnostics["emission_execution"] = {
                "schema": "cross_run_static_analysis_import_execution_v1",
                "source_checkpoint_sha256": checkpoint_sha256,
                "full_graph_compiler_invoked": False,
                "full_model_simulation_executed": False,
                "timing_evidence_imported": False,
            }
            readiness = EA.global_iteration_readiness(analysis)
            analysis["iteration_readiness"] = copy.deepcopy(readiness)
            if readiness["status"] != "ready_for_probe_admission":
                raise ValueError("imported static analysis fails current readiness recomputation")
            artifacts = member.decode_artifacts(analysis=analysis)
            source = Path(sentinel.frozen_source_path)
            descriptor = P2_CONTRACTS.mapping_file(source / "capsule.yaml", yaml_file=True)
            interface = source / str(descriptor.get("interface_mlir") or "capsule.interface.mlir")
            if interface.is_symlink() or not interface.is_file():
                raise ValueError("current portfolio interface is absent or linked")
            artifacts["interface"] = str(interface.resolve())
            current_analyses.append(analysis)
            current_artifacts[sentinel.capsule_sha256] = artifacts
        submitted = self.analysis.session.journal.output / "submission_0000"
        shutil.copytree(candidate, submitted, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if hash_tree(submitted)["sha256"] != candidate_sha256:
            raise ValueError("candidate changed while capturing imported static analysis submission")
        for path in submitted.rglob("*"):
            if path.is_symlink():
                raise ValueError("captured imported static analysis submission contains a link")
            path.chmod(path.stat().st_mode & ~0o222)
        submitted.chmod(0o555)
        current_scope = self.analysis.session.validate_candidate_scope(submitted)
        for analysis in current_analyses:
            analysis["compiler_edit_scope"] = copy.deepcopy(current_scope)
        reconstructed_sandboxes, sandbox_reconstruction = self._reconstruct_imported_compiler_sandboxes(
            submitted, dependencies=dependencies
        )

        primary = current_analyses[0]
        primary_readiness = EA.global_iteration_readiness(primary)
        members = [
            {
                "identity": sentinel_identity(self.analysis.session.inputs.sentinel, role="primary"),
                "status": "completed",
                "analysis_ref": "/analysis",
                "readiness": primary_readiness,
                "static_comparison_ref": "/static_comparison",
                "analysis_allocation": {
                    "schema": "portfolio_cross_run_import_allocation_v1",
                    "policy": "exact_static_import_no_compilation",
                    "allocated_seconds": 0.0,
                },
                "elapsed_seconds": 0.0,
                "timing_status": "UNMEASURED_FULL_MODEL",
            }
        ]
        for sentinel, analysis in zip(
            self.analysis.session.inputs.portfolio_sentinels[1:], current_analyses[1:], strict=True
        ):
            readiness = EA.global_iteration_readiness(analysis)
            members.append(
                {
                    "identity": sentinel_identity(sentinel, role="training"),
                    "status": "completed",
                    "analysis": analysis,
                    "readiness": readiness,
                    "static_comparison": self.analysis.compare_analyses(None, analysis, previous_iteration=None),
                    "analysis_allocation": {
                        "schema": "portfolio_cross_run_import_allocation_v1",
                        "policy": "exact_static_import_no_compilation",
                        "allocated_seconds": 0.0,
                    },
                    "elapsed_seconds": 0.0,
                    "timing_status": "UNMEASURED_FULL_MODEL",
                }
            )
        readiness = copy.deepcopy(primary_readiness)
        readiness.update(
            {
                "portfolio_sha256": self.analysis.session.inputs.portfolio_identity_sha256,
                "portfolio_members_ready": len(members),
                "portfolio_members_total": len(members),
                "selection": "multi_model_pareto_without_invented_static_cycle_total",
            }
        )
        fast_evaluation = self.analysis.session.fast_evaluation.evaluate(
            [
                (sentinel, analysis, current_artifacts[sentinel.capsule_sha256])
                for sentinel, analysis in zip(
                    self.analysis.session.inputs.portfolio_sentinels, current_analyses, strict=True
                )
            ],
            edit_contract=self.analysis.session.edit_authority.contract,
            target_descriptor=self.analysis.session.inputs.target_descriptor,
            target_sha256=self.analysis.session.inputs.target_sha256,
            portfolio_sha256=self.analysis.session.inputs.portfolio_identity_sha256,
        )
        elapsed = time.monotonic() - started
        reuse = {
            "schema": "global_exact_cross_run_static_analysis_import_v1",
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": checkpoint_sha256,
            "source_snapshot": verified_source["root"],
            "source_snapshot_files_sha256": verified_source["files_sha256"],
            "binding": copy.deepcopy(current_binding),
            "full_graph_compiler_invoked": False,
            "full_model_simulation_executed": False,
            "probe_or_timing_receipts_reused": False,
            "semantic_or_decision_feedback_reused": False,
            "compiler_sandbox_reconstruction": sandbox_reconstruction,
            "reuse_verification_elapsed_seconds": elapsed,
        }
        record = {
            "schema": "global_perf_iteration_v1",
            "iteration": 0,
            "candidate_path": str(candidate.resolve()),
            "candidate_sha256": candidate_sha256,
            "submitted_snapshot": str(submitted.resolve()),
            "compiler_dependencies": dependencies,
            "analysis_reuse_binding": self.analysis.analysis_reuse_binding(
                candidate_sha256=candidate_sha256, compiler_dependencies=dependencies
            ),
            "cross_run_static_analysis_binding": copy.deepcopy(current_binding),
            "analysis_reuse": reuse,
            "exact_analysis_reused": True,
            "baseline_sha256": self.analysis.session.inputs.baseline_sha256,
            "optimization_baseline_sha256": self.analysis.session.inputs.optimization_baseline_sha256,
            "optimization_baseline": copy.deepcopy(self.analysis.session.inputs.optimization_baseline_binding),
            "compiler_mechanism_catalog": copy.deepcopy(self.analysis.session.mechanism_program.catalog_binding),
            "compiler_mechanism_work_order": copy.deepcopy(self.analysis.session.mechanism_program.work_order_binding),
            "mechanism_work_order_analysis": copy.deepcopy(self.analysis.session.mechanism_program.analysis_binding),
            "round_mechanism_attribution": {
                "schema": "global_compiler_mechanism_seed_analysis_v1",
                "status": "initial_seed",
                "candidate_sha256": candidate_sha256,
                "mechanism_catalog_sha256": self.analysis.session.mechanism_program.catalog_binding_sha256,
            }
            if self.analysis.session.mechanism_program.catalog_binding is not None
            else None,
            "hypothesis": "Bind initial seed from exact cross-run static analysis",
            "analysis": primary,
            "readiness": readiness,
            "historical_reference": copy.deepcopy(self.analysis.session.inputs.historical_reference),
            "elapsed_seconds": elapsed,
            "timing_status": "UNMEASURED_FULL_MODEL",
            "allocated_seconds": self.analysis.timeout_s,
            "probe_receipts": [],
            "global_performance_claim": "unproven",
            "relative_semantic_evidence": {
                "status": "unavailable_cross_run_static_import",
                "numerical_equivalence": False,
            },
            "fast_evaluation": fast_evaluation,
        }
        record["static_comparison"] = self.analysis.compare(record)
        self.analysis.apply_functional_gate(
            record, current_artifacts[self.analysis.session.inputs.sentinel.capsule_sha256]
        )
        readiness = record["readiness"]
        record["portfolio"] = {
            "schema": "full_model_portfolio_iteration_v1",
            "portfolio_sha256": self.analysis.session.inputs.portfolio_identity_sha256,
            "candidate_sha256": candidate_sha256,
            "members": members,
            "members_ready": len(members),
            "members_total": len(members),
            "selection": readiness["selection"],
            "analysis_allocation_policy": "exact_cross_run_static_import_no_compilation",
            "analysis_concurrency": {
                "schema": "portfolio_cross_run_import_concurrency_v1",
                "requested_workers": self.analysis.portfolio_analysis_workers,
                "admitted_workers": 0,
                "members": len(members),
                "policy": "no_workers_admitted_for_exact_cross_run_static_import",
            },
            "full_model_simulation_allowed": False,
        }
        primary_artifacts = current_artifacts[self.analysis.session.inputs.sentinel.capsule_sha256]
        static_bundle = self.analysis.persist_static_analysis_bundle(
            record, primary_artifacts, portfolio_artifacts=current_artifacts
        )
        assert static_bundle is not None
        record["static_analysis_bundle"] = static_bundle
        self.analysis.session.journal.publish_imported_seed(
            record,
            primary_artifacts,
            current_artifacts,
            reconstructed_sandboxes=reconstructed_sandboxes,
            sandbox_digest=sandbox_reconstruction["policy_set_sha256"] if reconstructed_sandboxes is not None else None,
        )
        receipt = {
            **reuse,
            "status": "hit",
            "result_iteration": 0,
            "result_iteration_record": str((self.analysis.session.journal.output / "iteration_0000.json").resolve()),
            "result_iteration_record_sha256": P2_CONTRACTS.sha256_file(
                self.analysis.session.journal.output / "iteration_0000.json"
            ),
        }
        path = SC.atomic_static_write(
            "cross_run_static_analysis_seed.json", receipt, output=self.analysis.session.journal.output
        )
        return {**receipt, "receipt": {"path": str(path), "sha256": P2_CONTRACTS.sha256_file(path)}}
