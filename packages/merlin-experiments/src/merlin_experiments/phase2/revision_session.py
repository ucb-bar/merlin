"""Live revision admission, action boundaries and immutable checkpoint publication.

The journal owns retained evidence; this session admits candidate bytes before using
it. Nested host actions share a thread-local admitted row only until an explicit
execution-boundary refresh. Blocked authoring checkpoints never authorize probes.
"""

from __future__ import annotations

import copy
import shutil
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree
from merlin.perf.mechanism_probe import ProbeBinding
from merlin_experiments.phase2 import authoring as AUTHORING
from merlin_experiments.phase2 import contracts as P2_CONTRACTS
from merlin_experiments.phase2 import portfolio_checkpoint as CHECKPOINT
from merlin_experiments.phase2.edit_authority import FrozenEditAuthority
from merlin_experiments.phase2.global_inputs import GlobalExperimentInputs
from merlin_experiments.phase2.mechanism_program import MechanismProgram
from merlin_experiments.phase2.mechanism_rounds import MechanismRounds
from merlin_experiments.phase2.portfolio_evaluation import FastPortfolioEvaluation
from merlin_experiments.phase2.revision_journal import RevisionJournal


class RevisionSession:
    """Compose frozen authorities with live candidate and retained revision admission."""

    def __init__(
        self,
        *,
        inputs: GlobalExperimentInputs,
        edit_authority: FrozenEditAuthority,
        mechanism_program: MechanismProgram,
        mechanism_rounds: MechanismRounds,
        fast_evaluation: FastPortfolioEvaluation,
        journal: RevisionJournal,
    ) -> None:
        self.inputs = inputs
        self.edit_authority = edit_authority
        self.mechanism_program = mechanism_program
        self.mechanism_rounds = mechanism_rounds
        self.fast_evaluation = fast_evaluation
        self.journal = journal
        self._integrity_action_local = threading.local()

    def check_inputs(self) -> None:
        self.inputs.verify(
            edit_authority=self.edit_authority,
            mechanism_program=self.mechanism_program,
            fast_evaluation=self.fast_evaluation,
        )

    def begin_mechanism_round(self, candidate: Path, *, round_index: int) -> dict[str, Any] | None:
        """Capture and admit round-start bytes before authoring."""
        if not self.mechanism_rounds.configured:
            return None
        self.mechanism_rounds.validate_index(round_index)
        self.check_inputs()
        self.validate_candidate_scope(candidate)
        prepared = self.mechanism_rounds.capture(
            candidate,
            round_index=round_index,
            compiler_shared_source_root=self.inputs.compiler_shared_source_root,
        )
        self.validate_candidate_scope(prepared.snapshot)
        return self.mechanism_rounds.publish_start(prepared)

    def inspect_mechanism_round(self, candidate: Path, *, require_semantic_edit: bool) -> dict[str, Any] | None:
        if not self.mechanism_rounds.configured:
            return None
        self.check_inputs()
        return self.mechanism_rounds.inspect(
            candidate,
            require_semantic_edit=require_semantic_edit,
            compiler_shared_source_root=self.inputs.compiler_shared_source_root,
        )

    def finalize_mechanism_round(self, candidate: Path, *, round_index: int) -> dict[str, Any] | None:
        if not self.mechanism_rounds.configured:
            return None
        self.mechanism_rounds.require_open(round_index)
        self.check_inputs()
        return self.mechanism_rounds.finalize(
            candidate,
            round_index=round_index,
            compiler_shared_source_root=self.inputs.compiler_shared_source_root,
        )

    def validate_candidate_scope(self, candidate: Path) -> dict[str, Any]:
        if self.edit_authority.configured or self.edit_authority.contract is not None:
            self.check_inputs()
        return self.edit_authority.validate_candidate(candidate)

    def inspect_optimization_surfaces(self, candidate: Path) -> dict[str, Any]:
        """Expose current AST locations with host-frozen semantics, never self-granted permissions."""
        self.check_inputs()
        self.validate_candidate_scope(candidate)
        return self.edit_authority.inspect_optimization_surfaces(candidate)

    def _matching_current_after_input_check(self, candidate: Path, *, require_ready: bool) -> dict[str, Any]:
        """Resolve an analyzed revision after the experiment-wide inputs were checked."""
        self.edit_authority.validate_candidate(candidate)
        if not self.journal.iterations:
            raise ValueError("compile the complete-model graph before requesting a probe or sealing")
        digest = hash_tree(candidate)["sha256"]
        dependencies = self.inputs.compiler_dependencies(candidate)
        matching_bytes = [item for item in reversed(self.journal.iterations) if item["candidate_sha256"] == digest]
        if not matching_bytes:
            raise ValueError("candidate changed: recompile its full graph and global plan")
        row = next((item for item in matching_bytes if item["compiler_dependencies"] == dependencies), None)
        if row is None:
            raise ValueError("shared compiler dependencies changed: recompile the full graph and plan")
        if require_ready and row["readiness"]["status"] != "ready_for_probe_admission":
            raise ValueError("global iteration is not ready: " + ", ".join(row["readiness"]["blockers"]))
        return row

    def _matching_current_fresh(self, candidate: Path, *, require_ready: bool) -> dict[str, Any]:
        self.check_inputs()
        return self._matching_current_after_input_check(candidate, require_ready=require_ready)

    def current(self, candidate: Path, *, require_ready: bool = True) -> dict[str, Any]:
        """Return the exact analyzed revision, optionally requiring promotion readiness.

        A blocked analysis is still valuable authoring evidence: it binds the candidate bytes to
        every portfolio member and tells the next compiler round what remains unsupported. It is
        never sufficient for probes, execution, or the promotable global-candidate seal. Nested
        accessors in one host action reuse only the revision verified by that action; no result is
        retained across actions or across an external compiler/runtime boundary.
        """
        state = getattr(self._integrity_action_local, "state", None)
        if state is None:
            return self._matching_current_fresh(candidate, require_ready=require_ready)
        candidate_key = str(Path(candidate).resolve(strict=True))
        if candidate_key != state["candidate_key"]:
            raise ValueError("one integrity action cannot substitute another candidate")
        row = state["row"]
        if require_ready and row["readiness"]["status"] != "ready_for_probe_admission":
            raise ValueError("global iteration is not ready: " + ", ".join(row["readiness"]["blockers"]))
        return row

    @contextmanager
    def action(self, candidate: Path):
        """Deduplicate strict reads only inside one synchronous host-controlled action."""
        candidate_key = str(Path(candidate).resolve(strict=True))
        state = getattr(self._integrity_action_local, "state", None)
        if state is not None:
            if candidate_key != state["candidate_key"]:
                raise ValueError("one integrity action cannot substitute another candidate")
            state["depth"] += 1
            try:
                yield state["row"], False
            finally:
                state["depth"] -= 1
            return
        row = self._matching_current_fresh(candidate, require_ready=True)
        state = {"candidate_key": candidate_key, "row": row, "depth": 1}
        self._integrity_action_local.state = state
        try:
            yield row, True
        finally:
            del self._integrity_action_local.state

    def _accept_refreshed_integrity_action_row(self, candidate: Path, row: dict[str, Any]) -> dict[str, Any]:
        state = getattr(self._integrity_action_local, "state", None)
        if state is not None:
            if (
                str(Path(candidate).resolve(strict=True)) != state["candidate_key"]
                or row["iteration"] != state["row"]["iteration"]
                or row["candidate_sha256"] != state["row"]["candidate_sha256"]
                or row["compiler_dependencies"] != state["row"]["compiler_dependencies"]
            ):
                raise ValueError("current compiler revision changed during the integrity action")
            state["row"] = row
        return row

    def revalidate(self, candidate: Path) -> dict[str, Any]:
        """Rehash the live candidate/dependencies without repeating action-wide immutable reads."""
        row = self._matching_current_after_input_check(candidate, require_ready=True)
        return self._accept_refreshed_integrity_action_row(candidate, row)

    def refresh(self, candidate: Path) -> dict[str, Any]:
        """Recheck every immutable input and live revision before leaving an action."""
        row = self._matching_current_fresh(candidate, require_ready=True)
        return self._accept_refreshed_integrity_action_row(candidate, row)

    def current_artifacts(self, candidate: Path) -> Mapping[str, Any]:
        """Host-only artifacts retained from exactly the current full-model invocation."""
        row = self.current(candidate)
        return self.journal.current_artifacts(row)

    def current_portfolio_artifacts(self, candidate: Path) -> Mapping[str, Mapping[str, Any]]:
        """Exact retained artifacts for every member of the current analyzed portfolio."""
        row = self.current(candidate)
        return self.journal.current_portfolio_artifacts(row, sentinels=self.inputs.portfolio_sentinels)

    def current_portfolio_member_context(self, candidate: Path, *, index: int) -> dict[str, Any]:
        """Strict current analysis/artifact context for one ordered portfolio member."""
        row = self.current(candidate)
        return self.journal.current_portfolio_member_context(
            row,
            index=index,
            sentinels=self.inputs.portfolio_sentinels,
            target_sha256=self.inputs.target_sha256,
        )

    def previous_portfolio_member_context(self, candidate: Path, *, index: int) -> dict[str, Any]:
        """Strict immediately preceding analysis/artifact context for one portfolio member."""
        self.current(candidate)
        return self.journal.previous_portfolio_member_context(
            index=index,
            sentinels=self.inputs.portfolio_sentinels,
            target_sha256=self.inputs.target_sha256,
        )

    def select_changed_portfolio_member(self, candidate: Path) -> dict[str, Any]:
        """Select an emitted-changed member by exact known host-work deltas and stable order."""
        with self.action(candidate):
            contexts = [
                (
                    self.previous_portfolio_member_context(candidate, index=index),
                    self.current_portfolio_member_context(candidate, index=index),
                )
                for index in range(len(self.inputs.portfolio_sentinels))
            ]
            return CHECKPOINT.select_changed_portfolio_contexts(contexts)

    def selected_changed_portfolio_context(
        self, candidate: Path, selection: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Recompute and validate a selected member before a host qualifier consumes it."""
        with self.action(candidate):
            expected = self.select_changed_portfolio_member(candidate)
            if selection is not None and dict(selection) != expected:
                raise ValueError("changed portfolio member selection is stale or caller-substituted")
            index = expected["portfolio_index"]
            return {
                "selection": expected,
                "previous": self.previous_portfolio_member_context(candidate, index=index),
                "current": self.current_portfolio_member_context(candidate, index=index),
            }

    def current_probe_binding(self, candidate: Path) -> ProbeBinding:
        row = self.current(candidate)
        return self.journal.probe_binding(row, target_sha256=self.inputs.target_sha256)

    def previous_artifacts(self, candidate: Path) -> Mapping[str, Any]:
        """Host-only preceding submitted artifact; never infer a previous revision from input IR."""
        self.current(candidate)
        return self.journal.retained_previous_artifacts()

    def previous_probe_binding(self, candidate: Path) -> ProbeBinding:
        """Bind a paired diagnostic to the actual preceding verified submitted revision."""
        self.previous_artifacts(candidate)
        return self.journal.previous_probe_binding(target_sha256=self.inputs.target_sha256)

    def optimization_baseline_artifacts(self, candidate: Path) -> Mapping[str, Any]:
        """Exact comparison bytes with unverified plan and unproven numerical status."""
        row = self.current(candidate)
        artifacts = self.journal.baseline_artifact_bytes(
            row,
            baseline_sha256=self.inputs.optimization_baseline_sha256,
            sentinel=self.inputs.sentinel,
            target=self.inputs.target,
        )
        current_artifacts = self.current_artifacts(candidate)
        return self.journal.optimization_baseline_artifacts(
            row,
            artifacts,
            current_artifacts,
            baseline_sha256=self.inputs.optimization_baseline_sha256,
            sentinel=self.inputs.sentinel,
        )

    def optimization_baseline_artifact_binding(self, candidate: Path) -> Mapping[str, Any]:
        """Identity-only comparison binding; deliberately not a verified ``ProbeBinding``."""
        artifact = self.optimization_baseline_artifacts(candidate)
        row = self.current(candidate)
        return self.journal.optimization_baseline_artifact_binding(
            row,
            artifact,
            baseline_sha256=self.inputs.optimization_baseline_sha256,
            baseline_dependencies=self.inputs.optimization_baseline_binding["compiler_dependencies"],
            baseline_binding_sha256=self.inputs.optimization_baseline_binding_sha256,
            sentinel=self.inputs.sentinel,
            target_sha256=self.inputs.target_sha256,
        )

    def seal(self, candidate: Path, *, name: str = "global_candidate") -> Path:
        """Seal verified global artifacts, independently of microbenchmark feedback/plateaus."""
        row = self.current(candidate)
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("checkpoint name must be one safe path component")
        iteration_path, iteration_sha256 = self._admit_static_record(row)
        snapshot = self.journal.output / ("sealed_submission" if name == "global_candidate" else name + "_submission")
        submitted = Path(row["submitted_snapshot"])
        # Seal the exact host-captured source used for analysis. Authoring may leave Python caches
        # afterward; they are not compiler inputs and must not be copied into the review artifact.
        AUTHORING.assert_candidate_sealable(submitted)
        if hash_tree(submitted)["sha256"] != row["candidate_sha256"]:
            raise ValueError("analyzed global submission changed before sealing")
        shutil.copytree(submitted, snapshot)
        if hash_tree(snapshot)["sha256"] != row["candidate_sha256"]:
            raise ValueError("global candidate changed while making its sealed snapshot")
        for item in sorted(snapshot.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            item.chmod(item.stat().st_mode & ~0o222)
        snapshot.chmod(snapshot.stat().st_mode & ~0o222)
        return self._write(
            name + ".json",
            {
                "schema": "global_perf_candidate_v1",
                "candidate_sha256": row["candidate_sha256"],
                "historical_reference": copy.deepcopy(self.inputs.historical_reference),
                "candidate_path": str(snapshot.resolve()),
                "iteration": row["iteration"],
                "candidate_read_only": True,
                "phase1_qualification": self.inputs.phase1_binding,
                "host_verification_policy": self.inputs.host_policy,
                "source_snapshot": (
                    str(self.inputs.source_snapshot_root) if self.inputs.source_snapshot_root is not None else None
                ),
                "source_snapshot_files_sha256": self.inputs.source_snapshot_files_sha256,
                "machine_build_policy": copy.deepcopy(self.inputs.machine_build_policy),
                "compiler_edit_authority": (
                    self.edit_authority.binding if self.edit_authority.contract is not None else None
                ),
                "compiler_mechanism_catalog": copy.deepcopy(self.mechanism_program.catalog_binding),
                "compiler_mechanism_work_order": copy.deepcopy(self.mechanism_program.work_order_binding),
                "mechanism_work_order_analysis": copy.deepcopy(self.mechanism_program.analysis_binding),
                "round_mechanism_attribution": copy.deepcopy(row.get("round_mechanism_attribution")),
                "compiler_dependencies": row["compiler_dependencies"],
                "cross_run_static_analysis_binding": row.get("cross_run_static_analysis_binding"),
                "static_analysis_bundle": row.get("static_analysis_bundle"),
                "analysis_sha256": P2_CONTRACTS.document_sha256(row["analysis"]),
                "iteration_record": str(iteration_path.resolve()),
                "iteration_record_sha256": iteration_sha256,
                "baseline_sha256": self.inputs.baseline_sha256,
                "target_sha256": self.inputs.target_sha256,
                "optimization_baseline_sha256": self.inputs.optimization_baseline_sha256,
                "optimization_baseline": self.inputs.optimization_baseline_binding,
                "capsule_sha256": self.inputs.sentinel.capsule_sha256,
                "portfolio": copy.deepcopy(self.inputs.portfolio_identity),
                "portfolio_sha256": self.inputs.portfolio_identity_sha256,
                "portfolio_iteration_sha256": P2_CONTRACTS.document_sha256(row["portfolio"]),
                "probe_receipts": row["probe_receipts"],
                "semantic_receipts": row.get("semantic_receipts", []),
                "context_receipts": row.get("context_receipts", []),
                "paired_context_receipts": row.get("paired_context_receipts", []),
                "source_contraction_preparation_receipts": row.get("source_contraction_preparation_receipts", []),
                "source_pair_receipts": row.get("source_pair_receipts", []),
                "decision_feedback": row.get("decision_feedback"),
                "fast_evaluation": copy.deepcopy(row.get("fast_evaluation")),
                "full_model_timing_status": "UNMEASURED",
                "global_speedup_proven": False,
                "promotion_status": "unqualified_candidate_for_review",
                "promotion_blockers": [
                    *row["readiness"]["promotion_blockers"],
                    *(
                        ["fast accuracy-bounded portfolio gate did not retain this candidate"]
                        if self.fast_evaluation.provider is not None
                        and row.get("fast_evaluation", {}).get("status") != "retain"
                        else []
                    ),
                ],
                "consumer": "global_plan_review_and_optional_post_freeze_validation",
            },
        )

    def checkpoint_authoring(self, candidate: Path, *, name: str) -> Path:
        """Preserve an exact blocked portfolio revision for the next authoring round only.

        This deliberately has a different schema and consumer from :meth:`seal`.  It cannot be
        used for probes, promotion, or any performance claim; its only purpose is to let a bounded
        sequence repair a compiler that does not yet lower every training model.
        """
        row = self.current(candidate, require_ready=False)
        if row["readiness"]["status"] != "blocked":
            raise ValueError("authoring checkpoints are only for blocked portfolio revisions")
        if not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("checkpoint name must be one safe path component")
        iteration_path, iteration_sha256 = self._admit_static_record(row)
        snapshot = self.journal.output / (name + "_submission")
        submitted = Path(row["submitted_snapshot"])
        AUTHORING.assert_candidate_sealable(submitted)
        if hash_tree(submitted)["sha256"] != row["candidate_sha256"]:
            raise ValueError("analyzed authoring submission changed before checkpointing")
        shutil.copytree(submitted, snapshot)
        if hash_tree(snapshot)["sha256"] != row["candidate_sha256"]:
            raise ValueError("authoring candidate changed while making its checkpoint")
        for item in sorted(snapshot.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            item.chmod(item.stat().st_mode & ~0o222)
        snapshot.chmod(snapshot.stat().st_mode & ~0o222)
        return self._write(
            name + ".json",
            {
                "schema": "global_authoring_checkpoint_v1",
                "candidate_sha256": row["candidate_sha256"],
                "candidate_path": str(snapshot.resolve()),
                "candidate_read_only": True,
                "iteration": row["iteration"],
                "readiness": copy.deepcopy(row["readiness"]),
                "portfolio_members_ready": row["portfolio"]["members_ready"],
                "portfolio_members_total": row["portfolio"]["members_total"],
                "historical_reference": copy.deepcopy(self.inputs.historical_reference),
                "phase1_qualification": self.inputs.phase1_binding,
                "host_verification_policy": self.inputs.host_policy,
                "source_snapshot": (
                    str(self.inputs.source_snapshot_root) if self.inputs.source_snapshot_root is not None else None
                ),
                "source_snapshot_files_sha256": self.inputs.source_snapshot_files_sha256,
                "machine_build_policy": copy.deepcopy(self.inputs.machine_build_policy),
                "compiler_edit_authority": (
                    self.edit_authority.binding if self.edit_authority.contract is not None else None
                ),
                "compiler_mechanism_catalog": copy.deepcopy(self.mechanism_program.catalog_binding),
                "compiler_mechanism_work_order": copy.deepcopy(self.mechanism_program.work_order_binding),
                "mechanism_work_order_analysis": copy.deepcopy(self.mechanism_program.analysis_binding),
                "round_mechanism_attribution": copy.deepcopy(row.get("round_mechanism_attribution")),
                "compiler_dependencies": row["compiler_dependencies"],
                "cross_run_static_analysis_binding": row.get("cross_run_static_analysis_binding"),
                "static_analysis_bundle": row.get("static_analysis_bundle"),
                "analysis_sha256": P2_CONTRACTS.document_sha256(row["analysis"]),
                "iteration_record": str(iteration_path.resolve()),
                "iteration_record_sha256": iteration_sha256,
                "baseline_sha256": self.inputs.baseline_sha256,
                "target_sha256": self.inputs.target_sha256,
                "optimization_baseline_sha256": self.inputs.optimization_baseline_sha256,
                "optimization_baseline": self.inputs.optimization_baseline_binding,
                "capsule_sha256": self.inputs.sentinel.capsule_sha256,
                "portfolio": copy.deepcopy(self.inputs.portfolio_identity),
                "portfolio_sha256": self.inputs.portfolio_identity_sha256,
                "portfolio_iteration_sha256": P2_CONTRACTS.document_sha256(row["portfolio"]),
                "full_model_timing_status": "UNMEASURED",
                "full_model_cycles": None,
                "global_speedup_proven": False,
                "promotion_status": "blocked_authoring_checkpoint",
                "promotion_blockers": row["readiness"]["blockers"],
                "consumer": "next_bounded_global_authoring_round_only",
            },
        )

    def _admit_static_record(self, row: Mapping[str, Any]) -> tuple[Path, str]:
        """Keep original static identity separate from later in-memory probe evidence."""
        iteration = row["iteration"]
        path = self.journal.output / f"iteration_{iteration:04d}.json"
        original = self.journal.record_sha256.get(iteration)
        if original is None or path.is_symlink() or not path.is_file() or P2_CONTRACTS.sha256_file(path) != original:
            raise ValueError("original static iteration record changed before checkpoint publication")
        return path, original

    def _write(self, name: str, record: Mapping[str, Any]) -> Path:
        path = self.journal.output / name
        payload = P2_CONTRACTS.canonical_json(record)
        with path.open("xb") as stream:
            stream.write(payload)
        path.chmod(0o444)
        return path
