"""Run admission and frozen input lifetime, before any candidate authoring.

The caller retains the existing workspace_session wrapper for the entire continuation.
Transport and task callbacks are trusted inventoried host code, not candidate inputs;
source attribution does not freeze arbitrary closure state or the Python interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import yaml

from merlin.common.paths import resolve_grant
from merlin.common.provenance import repo_sha
from merlin.development.backend.authoring import author_subagents as AS
from merlin.development.backend.authoring import conformance_sources as AC
from merlin.development.backend.authoring import fresh_backend as FB
from merlin.development.backend.authoring import skill_staging as SK
from merlin.development.backend.authoring.library import compiler_library as CL
from merlin.development.backend.context import InvocationContext, readback_record, verify_readback_record
from merlin.development.backend.options import RunOptions
from merlin.development.backend.providers import codex_context as CTX
from merlin.development.backend.session import preflight as _session_guidance
from merlin.development.backend.session import preflight as _session_preflight
from merlin.development.backend.session import preflight as _session_resume
from merlin.development.backend.session import treatments as T
from merlin.development.backend.session import workspaces as _oot_history
from merlin.development.backend.session.inputs import corpus_views as CI
from merlin.development.backend.session.inputs import implementation_inventory as SI
from merlin.development.backend.session.inputs import run_record as RI
from merlin.development.backend.session.workspaces import select_workspace_root
from merlin.development.targetgen.sandbox import grants as _BWS
from merlin.target.registry.descriptor_loading import load_target_experiment

# ---- session -----------------------------------------------------------------------------------

if TYPE_CHECKING:
    from merlin.development.backend.authoring.task_staging import TaskStagingConfig
    from merlin.development.targetgen.authoring.library_admission import CompilerLibrarySelection

_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"

_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"


def _public_build_selection(target: str, budget_s: int) -> dict | None:
    """Observe an operator-selected pure build capability, never candidate input."""
    if not budget_s:
        return None
    from merlin.development.backend.feedback.scalability.resource_gate import selected_build_service

    _, selection = selected_build_service(target)
    return selection


def _verify_public_build_budget(options: RunOptions, environment: dict) -> None:
    recorded = environment.get("run_config") or {}
    prior = recorded.get("public_object_build_budget_s", 0) if type(recorded) is dict else None
    if type(prior) is not int or prior != options.public_object_build_budget_s:
        raise RuntimeError("resume refused: public object-build advisory selection changed")


def _client_selection(options) -> dict | None:
    from merlin.development.backend.providers.codex_runtime import selection_record

    return selection_record(options)


def _verify_client_selection(options, environment: dict) -> None:
    prior = (environment.get("run_config") or {}).get("codex_max_subagents")
    if prior != options.codex_max_subagents:
        raise RuntimeError("explicit Codex delegation limit changed")
    AS.verify_resume(options, environment.get("run_config") or {})
    CTX.verify_resume(options, environment.get("run_config") or {})
    SK.verify_resume(options, environment.get("run_config") or {})
    from merlin.development.backend.telemetry import token_budget

    token_budget.verify_resume(options, environment.get("run_config") or {})
    recorded = (environment.get("run_config") or {}).get("codex_runtime")
    if recorded != _client_selection(options):
        raise RuntimeError("explicit client selection changed")


def _verify_phase0_handoff(corpus_review: dict | None, workload_coverage: dict | None) -> None:
    """Bind the reviewed Phase 0 handoff to the frozen corpus view.

    Compiler-produced support lowering and execution cannot be required before the
    Phase 1 agent starts. They remain obligations of the later strict qualification;
    this check admits only a reviewed, source-closed deterministic corpus.
    """
    review = corpus_review or {}
    if not (review.get("whole_workload_phase1") or {}).get("required"):
        return
    from merlin.development.testgen.evidence.coverage_commitment import (
        build_phase0_readiness,
        phase0_readiness_identity,
        require_phase0_ready,
    )

    readiness = build_phase0_readiness(workload_coverage or {})
    expected = review.get("phase0_readiness")
    observed = phase0_readiness_identity(readiness, required=True)
    if expected != observed:
        raise ValueError("frozen Phase 0 readiness differs from the reviewed corpus")
    require_phase0_ready(readiness)


@dataclass(frozen=True)
class RunRequest:
    context: InvocationContext
    options: RunOptions
    treatment: T.Treatment
    bundle_manifest: Path
    launcher_argv: tuple[str, ...]
    source_entrypoint: Path
    require_native_source: bool
    account: dict
    # Read after task staging, not while constructing the invocation: tools.txt may
    # be changed by a trusted task stager, and the receipt must bind served tools.
    resolved_tools: Callable[[], tuple[str, ...]]
    compiler_library_selection: CompilerLibrarySelection | None = None
    task_staging_config: TaskStagingConfig | None = None

    @property
    def run_config(self) -> dict:
        a = self.options
        from merlin.development.backend.telemetry import token_budget

        return {
            "schedule": a.schedule,
            "compiler_execution": a.compiler_execution,
            "sandbox_reuse": getattr(a, "sandbox_reuse", "per_call"),
            "compiler_access": a.compiler_access,
            **({"author_profile": a.author_profile} if a.author_profile else {}),
            "full_compiler_manifest_sha256": a.full_compiler_manifest_sha256,
            "reference_experiments_sha256": a.reference_experiments_sha256,
            **(
                {
                    "public_owned_host_inputs": {
                        "path": a.public_owned_host_inputs,
                        "sha256": a.public_owned_host_inputs_sha256,
                    }
                }
                if a.public_owned_host_inputs
                else {}
            ),
            **({"reference_experiments_tasks": "off"} if a.reference_experiments_tasks == "off" else {}),
            "session_mode": (
                "joint_campaign"
                if getattr(a, "joint_campaign", "")
                else "codex_canary"
                if a.codex_canary
                else "preflight_only"
                if a.preflight_only
                else "submission_qualification"
                if a.qualify_submission
                else "legacy_progress_only"
                if a.continuous
                else "certified_continuous"
                if a.schedule == "continuous"
                else "legacy_round_relaunch"
            ),
            "max_wall_s": a.max_wall_s,
            "max_rounds": a.max_rounds,
            "round_timeout_s": a.round_timeout,
            "qa_timeout_s": a.qa_timeout,
            "grade_interval_s": a.grade_interval,
            "selfcheck_protocol": 3,
            **(
                {"joint_campaign_policy": hashlib.sha256(Path(a.joint_campaign).read_bytes()).hexdigest()}
                if getattr(a, "joint_campaign", "")
                else {}
            ),
            **(
                {"joint_epoch_review": hashlib.sha256(Path(a.joint_epoch).read_bytes()).hexdigest()}
                if getattr(a, "joint_epoch", "")
                else {}
            ),
            "launcher_argv": list(self.launcher_argv),
            **({"live_conformance_feedback": True} if a.live_conformance_feedback else {}),
            **(
                {"public_object_build_budget_s": a.public_object_build_budget_s}
                if a.public_object_build_budget_s
                else {}
            ),
            **({"codex_runtime": _client_selection(a)} if a.codex_binary else {}),
            **({"codex_max_subagents": a.codex_max_subagents} if a.codex_max_subagents is not None else {}),
            **({"author_subagents": subagents} if (subagents := AS.selection(a)) is not None else {}),
            **({CTX.KEY: author_context} if (author_context := CTX.selection(a)) is not None else {}),
            **({SK.KEY: author_skills} if (author_skills := SK.selection(a)) is not None else {}),
            **({"author_token_budget": budget} if (budget := token_budget.selection(a)) is not None else {}),
            **({"readback_policy": readback_record(self.context)} if self.context.readback_policy is not None else {}),
        }

    @property
    def source_context(self) -> dict:
        return {
            "repo": self.context.repo,
            "entrypoint": self.source_entrypoint,
            "descriptor": self.context.descriptor,
            "require_native": self.require_native_source,
        }


@dataclass(frozen=True)
class AssemblyEvidence:
    denied_names: list[str]
    violations: list[str]
    copy_report: dict | None


class WorkspaceAssembler(Protocol):
    def __call__(self, bundle: dict, ws: Path, sandbox: str, *, context: InvocationContext) -> AssemblyEvidence: ...


class WorkspaceProbe(Protocol):
    def __call__(self, ws: Path, bundle: dict, sandbox: str, *, context: InvocationContext) -> dict: ...


@dataclass(frozen=True)
class WorkspaceTransport:
    assemble: WorkspaceAssembler
    probe: WorkspaceProbe


class TaskStager(Protocol):
    def __call__(
        self, arm: str, ws: Path, run_dir: Path, *, sandbox: str, task_scope: dict, policy_root: Path | None
    ) -> None: ...


@dataclass(frozen=True)
class PreparedRun:
    request: RunRequest
    run_dir: Path
    workspace: Path
    bundle_dir: Path
    bundle: dict
    environment: dict
    public_root: Path | None
    policy_root: Path | None
    contract_root: Path | None
    hidden_root: Path | None
    resuming: bool
    reviewed_roots: tuple[Path, ...] | None
    transport: WorkspaceTransport
    stage_task: TaskStager
    private_full_model_spec: Path | None = None
    instruction_selection: Path | None = None
    compiler_library_binding: bytes | None = None

    @property
    def scope_roots(self) -> dict:
        return {
            "public_roots": [self.policy_root] if self.policy_root is not None else None,
            "hidden_roots": [self.hidden_root] if self.hidden_root is not None else None,
            "contract": self.contract_root,
        }

    @property
    def selected_codex_runtime(self):
        from merlin.development.backend.providers.codex_runtime import select

        library = CL.verify(self.request, self.environment, self.compiler_library_binding)
        runtime = select(
            self.request.options,
            self.request.context,
            self.bundle,
            self.workspace,
            self.run_dir,
            compiler_library_selection=library,
        )
        observed = None if runtime is None else runtime.record()
        if observed != self.environment.get("codex_runtime_policy"):
            raise RuntimeError("explicit client permission/runtime selection changed")
        return runtime

    def verify_inputs(self) -> None:
        FB.origin(self.request.options, self.workspace, prior=self.environment)
        CL.verify(self.request, self.environment, self.compiler_library_binding)
        from merlin.development.backend.session.inputs import public_software_declaration as PSD

        PSD.verify(self.environment.get(PSD.KEY), self.workspace, self.bundle, repo=self.request.context.repo)
        from merlin.development.backend.feedback.certification import hidden_capsule_policy
        from merlin.development.backend.feedback.certification.capsule_certification import verify_scope

        verify_scope(getattr(self.request.options, "validation_scope", "full"), self.environment)

        hidden_capsule_policy.verify_record(
            hidden_capsule_policy.resolve(
                self.request.context.descriptor, getattr(self.request.options, "hidden_capsule_mode", "")
            ),
            self.environment.get("hidden_capsule_validation"),
        )
        verify_readback_record(self.request.context, self.environment.get("readback_policy"))
        _verify_public_build_budget(self.request.options, self.environment)
        AC.verify_live_selection(self.request.options, self.environment)
        _verify_client_selection(self.request.options, self.environment)
        self.selected_codex_runtime
        if self.environment.get("public_object_build_selection") != _public_build_selection(
            self.request.context.target, self.request.options.public_object_build_budget_s
        ):
            raise RuntimeError("public object-build tool/source selection changed")
        sources = self.environment["implementation_sources"]
        SI.verify(sources, **self.request.source_context)
        if T.record(self.request.treatment, sources) != self.environment["invocation_treatment"]:
            raise RuntimeError("invocation treatment changed")
        for callback in (self.transport.assemble, self.transport.probe, self.stage_task, self.request.resolved_tools):
            T.callback_reference(callback, sources, label="admission callback")
        if self.request.options.sandbox == "bwrap":
            _BWS.verify_snapshot_binding(
                self.workspace,
                self.bundle,
                self.environment.get("bundle_input_snapshot"),
                repo=self.request.context.repo,
            )
        private = self.environment.get("private_full_model_spec")
        if private is not None:
            from merlin.runtime.execution.model_inputs import file_sha256

            if (
                self.private_full_model_spec is None
                or str(self.private_full_model_spec) != private.get("frozen_path")
                or file_sha256(self.private_full_model_spec) != private.get("frozen_sha256")
            ):
                raise RuntimeError("frozen operator-private full-model specification changed")
            source_freeze = private.get("source_freeze")
            if source_freeze is not None:
                from merlin.development.backend.qualification.provenance import private_source_freeze

                private_source_freeze.verify(
                    self.private_full_model_spec,
                    source_freeze,
                    root=self.run_dir / "private_full_model_input" / "sources",
                    target=self.request.context.target,
                )
        instruction = self.environment.get("private_instruction_selection")
        if instruction is not None:
            from merlin.development.backend.qualification.provenance import private_instruction_declaration
            from merlin.runtime.execution.model_inputs import file_sha256

            if (
                self.instruction_selection is None
                or str(self.instruction_selection) != instruction.get("frozen_path")
                or file_sha256(self.instruction_selection) != instruction.get("sha256")
            ):
                raise RuntimeError("frozen operator-private instruction selection changed")
            private_instruction_declaration.read(
                self.instruction_selection, target=self.request.context.target
            ).verify()
        from merlin.development.backend.feedback.scalability import public_owned_host_inputs

        host_record = self.environment.get(public_owned_host_inputs.KEY)
        if (None if host_record is None else host_record.get("source")) != self.request.run_config.get(
            public_owned_host_inputs.KEY
        ):
            raise RuntimeError("selected public host inputs differ from the original invocation")
        public_owned_host_inputs.verify_record(
            host_record,
            target=self.request.context.target,
            descriptor=self.request.context.descriptor,
            run_dir=self.run_dir,
        )
        corpus_record = self.environment.get("public_corpus_input")
        if corpus_record is not None:
            view = CI.resolve(
                self.workspace,
                self.bundle,
                corpus_record,
                repo=self.request.context.repo,
                reviewed_roots=self.reviewed_roots,
            )
            _verify_phase0_handoff(self.environment.get("corpus_review"), view.workload_coverage)


def task_scope(
    te, sandbox: str, *, repo: Path, public_roots=None, hidden_roots=None, contract: Path | None = None
) -> dict:
    """Derive non-vacuous task scope using the selected policy and hidden views."""
    from merlin.development.targetgen.corpus.capsule_io import discover_capsules

    contract = contract if contract is not None else resolve_grant("merlin/contract", root=repo)
    public = discover_capsules(
        te.graded_roots() if public_roots is None else public_roots, labels={"public", "dev"}, contract=contract
    )
    excluded = set(te.effective_exclusions(cap.get("name") for cap in public))
    public = [cap for cap in public if cap.get("name") not in excluded]
    hidden = discover_capsules(
        te.hidden_roots() if hidden_roots is None else hidden_roots, labels={"hidden"}, contract=contract
    )
    if not public:
        raise RuntimeError(
            f"{te.target}: descriptor-derived public/dev task scope is empty; refusing to serve a "
            "vacuous completion target"
        )
    return {
        "target": te.target,
        "required_public_dev_capsules": len(public),
        "held_out_capsules": len(hidden),
        "sandbox": sandbox,
        "scope_source": "TargetExperiment.graded_roots + labels public,dev + formal cohort policy",
    }


def model_python_selection(
    te, *, repo: Path, public_roots=None, hidden_roots=None, contract: Path | None = None, prior: dict | None = None
) -> dict | None:
    """Bind an explicit host Torch interpreter only when the graded scope contains models.

    The core model loader already honors ``MERLIN_M2M_PYTHON``. Select that same
    operator-owned value here before authoring; a sibling checkout is not runtime
    authority for an installed Phase 1 invocation.
    """
    from merlin.development.targetgen.corpus.capsule_io import discover_capsules
    from merlin.runtime.execution.model_inputs import file_sha256

    contract = contract if contract is not None else resolve_grant("merlin/contract", root=repo)
    public = discover_capsules(
        te.graded_roots() if public_roots is None else public_roots, labels={"public", "dev"}, contract=contract
    )
    hidden = discover_capsules(
        te.hidden_roots() if hidden_roots is None else hidden_roots, labels={"hidden"}, contract=contract
    )
    excluded = set(te.effective_exclusions(cap.get("name") for cap in public))
    if not any(cap.get("kind") == "model" and cap.get("name") not in excluded for cap in public) and not any(
        cap.get("kind") == "model" for cap in hidden
    ):
        return None
    if prior is not None and "model_python_selection" not in prior:
        raise RuntimeError("resume refused: model run has no bound Python selection; start a fresh run")
    value = os.environ.get("MERLIN_M2M_PYTHON", "")
    selected = Path(value)
    if not value or not selected.is_absolute() or not selected.is_file() or not os.access(selected, os.X_OK):
        raise ValueError("graded model capsules require an explicit executable MERLIN_M2M_PYTHON")
    resolved = selected.resolve(strict=True)
    marker = "MERLIN_MODEL_PYTHON_READY"
    try:
        probe = subprocess.run(
            [str(selected), "-I", "-B", "-c", f"import torch, numpy, safetensors; print({marker!r})"],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("selected MERLIN_M2M_PYTHON could not complete the bounded Torch import probe") from exc
    if probe.returncode != 0 or probe.stdout.strip().splitlines()[-1:] != [marker]:
        raise ValueError("selected MERLIN_M2M_PYTHON cannot import Torch, NumPy and safetensors")
    return {
        "path": str(selected),
        "resolved_path": str(resolved),
        "sha256": file_sha256(selected),
        "size_bytes": selected.stat().st_size,
    }


_ORCHESTRATION_MEMBERS = frozenset({"resolved-plan.json", "orchestration.json", ".orchestration.lock"})


def orchestration_owned(run_dir: Path) -> bool:
    """A run dir holding only the orchestration record that launched this engine (no engine state)."""
    names = {member.name for member in run_dir.iterdir()} if run_dir.is_dir() else set()
    attempts = {name for name in names if name.startswith("phase1-attempt-") and name.endswith(".log")}
    return "orchestration.json" in names and names - attempts <= _ORCHESTRATION_MEMBERS


def phase_run_dir(context, arm: str, run_id: str, *, resume: bool) -> Path:
    """``out/runs/<target>/backend-generation/<run-id>/`` (``context.phase_runs``, set by ``load_context``).

    A context built without a phase root, or a resumed run that began in the legacy
    ``capsule-bench/<arm>/`` root, keeps the legacy location.
    """
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError(f"run id must be one path component: {run_id!r}")
    legacy = context.runs / arm / run_id
    if context.phase_runs is None or (resume and legacy.is_dir()):
        return legacy  # a hand-built legacy context, or a run that began in the legacy root
    if resume:
        from merlin.common.paths import phase_runs_root

        historical = phase_runs_root(context.target, 1) / run_id
        if historical.is_dir():
            return historical
    return context.phase_runs / run_id


def validate_preflight_options(a: RunOptions) -> None:
    """Pure selection checks, before native probes and when reopening preparation."""
    from merlin.development.backend.authoring.profiles import validate_options as validate_author_profile

    validate_author_profile(a)
    reference = (a.reference_experiments, a.reference_experiments_sha256)
    if any(reference):
        if (
            not all(reference)
            or len(reference[1]) != 64
            or any(char not in "0123456789abcdef" for char in reference[1])
        ):
            raise ValueError("reference experiment feedback requires its explicit selection and exact SHA256")
    if a.reference_experiments_tasks not in ("on", "off") or (
        a.reference_experiments_tasks == "off" and not a.reference_experiments
    ):
        raise ValueError("withheld reference experiment tasks require an explicit selection")
    if a.compiler_execution == "snapshot_external_v1" and a.sandbox != "bwrap":
        raise ValueError("snapshot_external_v1 requires bwrap isolation")
    full_sources = (a.full_compiler_root, a.full_compiler_manifest, a.full_compiler_manifest_sha256)
    if a.compiler_access == "full_compiler":
        if a.compiler_execution != "snapshot_external_v1" or not all(full_sources):
            raise ValueError("full_compiler requires snapshot_external_v1 and the complete public source selection")
    elif any(full_sources):
        raise ValueError("full compiler inputs require --compiler-access full_compiler")
    ambient_seal = os.environ.get("MERLIN_CORPUS_SEAL", "").strip()
    if a.corpus_seal and ambient_seal and a.corpus_seal != ambient_seal:
        raise ValueError("explicit corpus seal differs from the environment selection")
    if a.preflight_only or a.codex_canary:
        if (
            (a.preflight_only and a.codex_canary)
            or a.resume
            or a.seed_submission
            or a.qualify_submission
            or a.seal_current
            or a.continuous
            or a.no_oracle
            or a.skip_hidden
            or a.allow_unsandboxed
            or a.sandbox != "bwrap"
        ):
            raise ValueError("preflight-only requires fresh isolated readiness without bypasses or a candidate")
        seal = Path(a.corpus_seal)
        if not a.corpus_seal or not seal.is_absolute() or not seal.is_file() or seal.resolve() != seal or not a.bundle:
            raise ValueError("preflight-only requires an explicit canonical corpus seal and bundle identity")


def validate_options(a: RunOptions) -> int | None:
    """Pre-initialization refusals shared by native and installed admission."""
    validate_preflight_options(a)
    FB.validate(a)
    from merlin.development.backend.campaign import control as joint_control

    joint_control.validate(a)
    if bool(a.public_owned_host_inputs) != bool(a.public_owned_host_inputs_sha256):
        raise ValueError("public host inputs require both explicit path and SHA256")
    if a.public_owned_host_inputs:
        if a.sandbox != "bwrap" or a.private_full_model_spec:
            raise ValueError("independent public host inputs require isolation and one selected source")
        from merlin.common.digest import is_sha256

        if not is_sha256(a.public_owned_host_inputs_sha256):
            raise ValueError("public host inputs require an exact SHA256")
    if a.instruction_selection and (not a.private_full_model_spec or a.sandbox != "bwrap"):
        raise ValueError("instruction selection requires a sandboxed private full-model declaration")
    if a.native_result_format_selection and (not a.private_full_model_spec or a.sandbox != "bwrap"):
        raise ValueError("native result selection requires a sandboxed private full-model declaration")
    from merlin.development.backend.providers.codex_runtime import validate_options as validate_client

    validate_client(a)
    if a.codex_canary and not 0 < a.round_timeout <= 600:
        raise ValueError("Codex canary requires an explicit 1..600 second round timeout")
    if a.qualify_submission:
        if a.resume or a.seed_submission or a.seal_current or a.continuous:
            raise RuntimeError("unpaid qualification requires a fresh run and one selected submission")
        if a.no_oracle or a.skip_hidden or a.sandbox != "bwrap":
            raise RuntimeError("unpaid qualification requires isolated public, hidden and oracle grading")
        if not a.private_full_model_spec:
            raise RuntimeError("unpaid qualification requires an operator-private full-model specification")
    if a.resume and a.seed_submission:
        raise RuntimeError("--seed-submission cannot be combined with --resume")
    if a.resume and a.operator_errata:
        raise RuntimeError("--operator-errata cannot be combined with --resume")
    RI.validate_seal_current_request(seal_current=a.seal_current, resume=a.resume, legacy_continuous=a.continuous)
    # A real (spending) run MUST be sandboxed: without bwrap the agent can read any absolute path (incl.
    # denied /scratch* answer dirs), so the copy-workspace + post-hoc transcript audit alone do NOT isolate
    # it. Fail closed (parity with run_agent_experiment.py) — an unsandboxed run needs an explicit opt-in.
    if a.sandbox != "bwrap" and not a.allow_unsandboxed:
        print(
            "REFUSING: a real run requires --sandbox bwrap (or explicit --allow-unsandboxed). "
            "Workspace assembly alone does not hide denied absolute paths.",
            file=sys.stderr,
        )
        return 4
    if a.sandbox == "bwrap":
        # Presence is not operability: a host can have /usr/bin/bwrap yet deny its
        # user-namespace setup. Refuse before snapshotting a large corpus or
        # launching an agent, rather than reporting the agent's empty work as a
        # model failure.
        from merlin.development.targetgen.sandbox.preflight import SandboxUnavailable, require_working_sandbox

        try:
            require_working_sandbox(context="Phase 1 agent rounds run inside bwrap")
        except SandboxUnavailable as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            print("  probe it directly: python -m merlin.development.targetgen.sandbox.preflight", file=sys.stderr)
            return 4
    return None


def prepare(
    request: RunRequest, transport: WorkspaceTransport, stage_task: TaskStager, *, workspace_leases: list
) -> PreparedRun | int:
    """Close all preparation verification scopes before returning or refusing."""
    from merlin.common.process.cpu_pool import PREPARATION, cpu_priority
    from merlin.development.backend.session.preparation import _prepare

    # Preparation shares the compute pool; its leased steps queue behind author and grade work.
    with cpu_priority(PREPARATION), ExitStack() as snapshot_scopes:
        return _prepare(
            request,
            transport,
            stage_task,
            workspace_leases=workspace_leases,
            snapshot_scopes=snapshot_scopes,
        )
