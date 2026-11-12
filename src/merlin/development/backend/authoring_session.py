"""Admitted author invocation bindings and durable checkpoint/resume state."""

from __future__ import annotations

import datetime as _dt
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from merlin.common.tree_hash import hash_tree
from merlin.development.backend import recovery as ROQ
from merlin.development.backend import treatments as T
from merlin.development.backend.audit import AnswerAudit
from merlin.development.backend.authoring_config import (
    AuthoringRuntime as AuthoringRuntime,
)
from merlin.development.backend.authoring_conformance import (
    conformance_ever as _conformance_ever,
)
from merlin.development.backend.authoring_conformance import live_launch as _live_launch
from merlin.development.backend.corpus_inputs import (
    _grading_public_root as _grading_public_root,
)
from merlin.development.backend.corpus_inputs import policy_roots as policy_roots
from merlin.development.backend.feedback import certification as CERT
from merlin.development.backend.feedback import loop_grading as LG
from merlin.development.backend.providers import execution as EX
from merlin.development.targetgen.target_experiment import (
    load_capability_manifest,
    load_target_experiment,
)


@dataclass
class AuthoringProgress:
    """One continuation shared by resume, normal rounds, repairs and completion."""

    rounds: list[dict] = field(default_factory=list)
    conformance_ever: dict[str, bool | None] = field(default_factory=dict)
    best_progress: tuple | None = None
    plateau_stall: int = 0
    rate_limit_waits_used: int = 0
    active_wall_s: float = 0.0
    rate_limit_wait_s: float = 0.0
    started_at: str = ""
    verdict: dict = field(default_factory=lambda: {"all_pass": False})
    workflow_conformant: bool = False
    round_index: int = 0
    latest_transcript: Path | None = None
    operator_seal: dict | None = None
    cost_capped: bool = False


class AuthoringSession:
    """Admitted invocation bindings and the lifetime of their checkpointed progress.

    Callbacks retain the original grading/provider selections. This object never
    chooses new source, task, treatment, library or evaluator authority.
    """

    def __init__(self, prepared: PreparedRun, runtime: AuthoringRuntime):
        self.prepared = prepared
        self.runtime = runtime
        self.options, self.treatment = prepared.request.options, prepared.request.treatment
        from merlin.development.backend import joint_control as JC

        self.joint = JC.selected() is not None
        self.arm = self.options.arm
        self.bundle_id = runtime.bundle_id
        self.context = prepared.request.context
        self.target_experiment = partial(load_target_experiment, self.context.descriptor)
        _manifest = partial(load_capability_manifest, self.context.target)
        self.audit = AnswerAudit.for_descriptor(self.target_experiment(), prepared.request.bundle_manifest.parent)
        self.resolved_tools = prepared.request.resolved_tools
        self.workspace, self.run_dir = prepared.workspace, prepared.run_dir
        self.bundle, self.environment = prepared.bundle, prepared.environment
        self.public_root, self.policy_root, self.contract_root = (
            prepared.public_root,
            prepared.policy_root,
            prepared.contract_root,
        )
        self.resuming = prepared.resuming
        self.scope_roots = prepared.scope_roots
        self.operator_errata = self.environment.get("operator_errata")
        self.run_config = prepared.request.run_config
        self.verify_inputs = prepared.verify_inputs
        self.provider = EX.ProviderConfig(
            self.options.driver,
            self.options.provider,
            self.options.subagent_model,
            self.options.background_model,
            (prepared.selected_codex_runtime if self.options.codex_binary else None),
            self.options.codex_max_subagents,
        )
        from merlin.development.targetgen.sandbox import bwrap as BW

        selected_rtl_facts = (
            BW.frozen_selected_rtl_facts(self.workspace, self.bundle, repo=self.context.repo)
            if self.options.sandbox == "bwrap"
            else None
        )
        self.execution = EX.ExecutionConfig(
            prepared.request.context,
            self.provider,
            prepared.request.resolved_tools,
            runtime.oracle_timing,
            max(0, int(self.options.sim_max_jobs)),
            selected_rtl_facts,
            sim_per_capsule_timeout=self.options.sim_per_capsule_timeout,
            model_gate_interval=EX.automatic_model_gate_interval(self.context, self.options),
        )

        public_build_service = None
        public_build_selection = None
        if self.options.public_object_build_budget_s:
            from merlin.development.backend.feedback.codegen_scalability import selected_build_service

            public_build_service, public_build_selection = selected_build_service(self.context.target)
            if public_build_selection != self.environment.get("public_object_build_selection"):
                raise RuntimeError("frozen public object-build tool/source selection changed before authoring")

        from merlin.development.backend.feedback.compiler_profile_runtime import selected_service

        compiler_profile_service = selected_service(
            workspace=self.workspace, run_dir=self.run_dir, repo=self.context.repo
        )

        self.driver_grade_receipt = [None]
        self.grading_inputs = LG.GradingInputs(
            context=prepared.request.context,
            arm=self.arm,
            public_root=_grading_public_root(
                self.context, self.public_root if self.public_root is not None else runtime.public_root
            ),
            policy_roots=policy_roots(self.context, self.policy_root),
            contract=self.contract_root,
            promotion_root=self.policy_root,
            additional_forbidden=("xdsl_dialects.lowering", "compiler/dialects/lowering", "outputs_match")
            if self.options.experiment == "realistic" and self.arm == "merlin_assisted"
            else (),
            public_build_service=public_build_service,
            public_build_selection=public_build_selection,
            public_build_budget_s=self.options.public_object_build_budget_s,
            compiler_profile_service=compiler_profile_service,
            verify_grade_inputs=self.verify_inputs,
            driver_grade_completed=lambda receipt: self.driver_grade_receipt.__setitem__(0, receipt),
            rtl_facts=selected_rtl_facts,
        )
        self.grade = T.timed(
            partial(
                LG.grade,
                inputs=self.grading_inputs,
                qa_runner=self.treatment.qa_runner,
            ),
            "qa",
            self.treatment.on_duration,
        )
        self.launch = T.timed(
            partial(
                EX.launch,
                config=self.execution,
                capsules_root=self.public_root,
                policy_root=self.policy_root,
                contract=self.contract_root,
            ),
            "agent",
            self.treatment.on_duration,
        )
        self.fast_grade = partial(LG.fast_grade, inputs=self.grading_inputs)
        self.background_grade = T.timed(
            partial(LG.grade, inputs=self.grading_inputs, qa_runner=self.treatment.qa_runner, advisory_background=True),
            "qa",
            self.treatment.on_duration,
        )
        # realistic (abc2): the self-check tool the agent runs logs each invocation here (dev-trajectory /
        # soft-failure / tool-trigger record). T0 anchors wall-offset. Inherited by every claude subprocess.
        if self.options.experiment == "realistic":
            os.environ["SELFCHECK_LOG"] = str(self.run_dir / "selfcheck_log.jsonl")
            os.environ.setdefault("SELFCHECK_T0", str(time.time()))
        self.state_path = self.run_dir / "qa_loop_state.yaml"
        self.progress = AuthoringProgress()
        try:  # endpoint_kind — drives the per-round dev-conformance flag (asm applies only to external_backend)
            from merlin.development.targetgen.generate_prompt import prompt_slots as _pslots

            self.endpoint_kind = _pslots(self.target_experiment(), _manifest()).get("endpoint_kind", "")
        except Exception:  # noqa: BLE001
            self.endpoint_kind = ""
        self.launch = _live_launch(
            self.launch,
            self.options.live_conformance_feedback,
            self.workspace,
            self.run_dir,
            self.arm,
            self.endpoint_kind,
            self.resolved_tools,
        )
        self.progress.started_at = datetime.now(_dt.UTC).isoformat()
        # Survives a --resume where the numeric/workflow checkpoint is already complete and the normal round
        # loop therefore does not launch again.  ``finalize.transcript`` is deliberately excluded: it is a
        # docs-only turn, not the authoring workflow whose mandatory tools must be evidenced.
        _saved_authoring_transcripts = sorted((self.run_dir / "rounds").glob("round_*.transcript.jsonl"))
        self.progress.latest_transcript = (
            None if self.options.seal_current else ROQ.latest_live_authoring_transcript(_saved_authoring_transcripts)
        )
        if self.resuming and self.state_path.exists():
            st = yaml.safe_load(self.state_path.read_text()) or {}
            self.progress.rounds = st.get("rounds", []) or []
            self.progress.round_index = int(st.get("next_round", 0))
            self.progress.verdict = {"all_pass": bool(st.get("converged", False))}
            self.progress.workflow_conformant = bool(st.get("workflow_conformant", False))
            # Rebuild the RUN-LEVEL accumulator the gates read. Without this a --resume starts it empty, so a
            # run that had already evidenced every mandated check is blocked by its own fresh accumulator --
            # the same shape of defect as gating on the current round, just moved into the resume path.
            # Prefer the persisted map; fall back to re-accumulating from the round records, so a checkpoint
            # written before this field existed still resumes correctly.
            _saved_ever = st.get("conformance_ever")
            if isinstance(_saved_ever, dict) and _saved_ever:
                self.progress.conformance_ever.update(_saved_ever)
            else:
                self.progress.conformance_ever.update(_conformance_ever(self.progress.rounds))
            cum = st.get("cumulative", {}) or {}
            self.progress.active_wall_s = float(cum.get("active_wall_s", 0.0))
            self.progress.rate_limit_wait_s = float(cum.get("rate_limit_wait_s", 0.0))
            self.progress.rate_limit_waits_used = int(cum.get("rl_waits_used", 0))
            self.progress.started_at = cum.get("started_at", self.progress.started_at)
            print(
                f"[resume] restored from checkpoint: next_round={self.progress.round_index} converged={self.progress.verdict['all_pass']} "
                f"workflow_conformant={self.progress.workflow_conformant} over_run={self.conformant_over_run()} "
                f"active={self.progress.active_wall_s:.0f}s rate_limit_wait={self.progress.rate_limit_wait_s:.0f}s "
                f"waits_used={self.progress.rate_limit_waits_used}"
            )

        if self.options.seal_current:
            if not self.progress.rounds:
                raise RuntimeError("--seal-current requires at least one completed, audited round")
            if not (self.workspace / "submission" / "manifest.yaml").is_file():
                raise RuntimeError("--seal-current requires a checkpointed submission with manifest.yaml")
            # An interrupted in-flight round may have a partial transcript on disk.  Conformance must be
            # derived from the last round the checkpoint actually records, never from unaudited tail bytes.
            completed_rounds = sorted(
                {
                    int(row["round"])
                    for row in self.progress.rounds
                    if isinstance(row, Mapping) and isinstance(row.get("round"), int)
                }
            )
            if not completed_rounds:
                raise RuntimeError("--seal-current checkpoint names no completed round")
            completed_path = self.run_dir / "rounds" / f"round_{completed_rounds[-1]:02d}.transcript.jsonl"
            if not completed_path.is_file():
                raise RuntimeError(f"--seal-current completed-round transcript is absent: {completed_path}")
            self.progress.latest_transcript = completed_path
            self.progress.operator_seal = {
                "version": 1,
                "requested": True,
                "reason": "operator accepted an incomplete functional baseline for named-waiver admission",
                "requested_at": datetime.now(_dt.UTC).isoformat(),
                "last_completed_round": completed_rounds[-1],
                "completed_rounds": len(self.progress.rounds),
                "checkpoint_submission_sha256": hash_tree(self.workspace / "submission")["sha256"],
            }
            print(
                f"[operator-seal] authoring stopped at completed round {completed_rounds[-1]}; "
                "continuing through official grade + immutable freeze (NOT convergence)",
                flush=True,
            )

    def conformant_over_run(self) -> bool:
        """Each applicable mandatory check must have passed in some completed round."""
        applicable = [v for v in self.progress.conformance_ever.values() if v is not None]
        return all(applicable) if applicable else True

    def authoring_complete(self) -> bool:
        """Pre-freeze gate: enough evidence to stop editing and begin the official grade."""
        return (
            False
            if self.joint
            else CERT._authoring_completion(
                bool(self.progress.verdict.get("all_pass")), self.progress.workflow_conformant
            )
        )

    def checkpoint(self, next_round: int) -> None:
        self.state_path.write_text(
            yaml.safe_dump(
                {
                    "run_id": self.options.run_id,
                    "arm": self.arm,
                    "model": self.options.model,
                    "effort": self.options.effort,
                    "rounds": self.progress.rounds,
                    "next_round": next_round,
                    "converged": self.authoring_complete(),
                    "numeric_all_pass": bool(self.progress.verdict.get("all_pass", False)),
                    "workflow_conformant": self.progress.workflow_conformant,
                    # Persist run-level evidence across resume; the per-round view records when it was earned.
                    "conformance_ever": dict(self.progress.conformance_ever),
                    "conformant_over_run": self.conformant_over_run(),
                    "cumulative": {
                        "active_wall_s": round(self.progress.active_wall_s, 3),
                        "rate_limit_wait_s": round(self.progress.rate_limit_wait_s, 3),
                        "rl_waits_used": self.progress.rate_limit_waits_used,
                        "started_at": self.progress.started_at,
                    },
                    "last_updated": datetime.now(_dt.UTC).isoformat(),
                },
                sort_keys=False,
            )
        )

    def sink_telemetry(self, where: str) -> None:
        """Publish provisional cumulative telemetry without interrupting authoring.

        The final sink remains authoritative. Until then, the bridge discovers
        per-round transcripts because the combined transcript does not yet exist.
        Its append-only scalar stream uses the last observation of each metric.
        """
        try:
            from merlin.development.targetgen import aet_bridge as AB

            if not AB.aet_sink_enabled(self.run_dir):
                return
            AB.emit_to_aet(
                run_dir=self.run_dir,
                run_id=self.options.run_id,
                method=self.arm,
                model=self.options.model,
                target=self.target_experiment().target,
                suite="capsule-bench",
                billing_mode=EX.billing_mode(self.options.model, config=self.provider),
            )
        except Exception as e:  # noqa: BLE001 - telemetry may never kill a run
            print(f"[aet-sink] {where}: skipped ({type(e).__name__}: {e})", flush=True)


if TYPE_CHECKING:
    from merlin.development.backend.session import PreparedRun
