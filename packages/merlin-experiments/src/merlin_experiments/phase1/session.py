"""Run admission and frozen input lifetime, before any candidate authoring.

The caller retains the existing workspace_session wrapper for the entire continuation.
Transport and task callbacks are trusted inventoried host code, not candidate inputs;
source attribution does not freeze arbitrary closure state or the Python interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import yaml

from merlin.benchharness import repo_sha
from merlin.targetgen.sandbox import bwrap as _BWS
from merlin.targetgen.target_experiment import load_target_experiment

from . import corpus_inputs as CI
from . import run_inputs as RI
from . import source_inputs as SI
from . import treatments as T
from .context import InvocationContext
from .options import RunOptions
from .workspaces import select_workspace_root

_MODEL_HOST_SNAPSHOT_ROOT_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_ROOT"
_MODEL_HOST_SNAPSHOT_REQUIRED_ENV = "MERLIN_MODEL_HOST_LANE_SNAPSHOT_REQUIRED"


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

    @property
    def run_config(self) -> dict:
        a = self.options
        return {
            "schedule": a.schedule,
            "session_mode": (
                "legacy_progress_only"
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
            "launcher_argv": list(self.launcher_argv),
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

    @property
    def scope_roots(self) -> dict:
        return {
            "public_roots": [self.policy_root] if self.policy_root is not None else None,
            "hidden_roots": [self.hidden_root] if self.hidden_root is not None else None,
            "contract": self.contract_root,
        }

    def verify_inputs(self) -> None:
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
        corpus_record = self.environment.get("public_corpus_input")
        if corpus_record is not None:
            CI.resolve(
                self.workspace,
                self.bundle,
                corpus_record,
                repo=self.request.context.repo,
                reviewed_roots=self.reviewed_roots,
            )


def task_scope(
    te, sandbox: str, *, repo: Path, public_roots=None, hidden_roots=None, contract: Path | None = None
) -> dict:
    """Derive non-vacuous task scope using the selected policy and hidden views."""
    from merlin.targetgen.capsule_common import discover_capsules

    contract = contract if contract is not None else repo / "merlin" / "contract"
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


def validate_options(a: RunOptions) -> int | None:
    """Pre-initialization refusals shared by native and installed admission."""
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
        from merlin.targetgen.sandbox.preflight import SandboxUnavailable, require_working_sandbox

        try:
            require_working_sandbox(context="Phase 1 agent rounds run inside bwrap")
        except SandboxUnavailable as exc:
            print(f"REFUSING: {exc}", file=sys.stderr)
            print("  probe it directly: python -m merlin.targetgen.sandbox.preflight", file=sys.stderr)
            return 4
    return None


def prepare(
    request: RunRequest, transport: WorkspaceTransport, stage_task: TaskStager, *, workspace_leases: list
) -> PreparedRun | int:
    """Prepare/resume a run, returning its original refusal code or admitted inputs.

    Call inside workspace_session and keep that wrapper active through authoring and
    completion. Appending the acquired lease preserves uncertain-exit protection if
    preparation or the later continuation raises. No child or paid agent starts here.
    """
    a, context, treatment = request.options, request.context, request.treatment
    refusal = validate_options(a)
    if refusal is not None:
        return refusal
    if request.bundle_manifest.name != "input_bundle_manifest.yaml":
        raise ValueError("admission requires the bundle's canonical input_bundle_manifest.yaml path")
    arm = a.arm
    _run_config = request.run_config

    def _te():
        return load_target_experiment(context.descriptor)

    bundle = yaml.safe_load(request.bundle_manifest.read_text())
    bundle_dir = request.bundle_manifest.parent

    run_dir = context.runs / arm / a.run_id
    _resuming = run_dir.exists() and a.resume
    if run_dir.exists() and not a.resume:
        print(f"run dir exists, refusing to overwrite: {run_dir}", file=sys.stderr)
        return 2
    _source_context = request.source_context
    _environment_path = run_dir / "environment.yaml"
    if _resuming:
        try:
            _environment_record = yaml.safe_load(_environment_path.read_text())
        except Exception as exc:  # noqa: BLE001 — provenance is a formal resume gate
            raise RuntimeError(f"resume refused: environment record unreadable: {exc}") from exc
        if not isinstance(_environment_record, dict):
            raise RuntimeError("resume refused: environment record is not a mapping")
        _implementation_sources = _environment_record.get("implementation_sources")
        SI.verify(_implementation_sources, **_source_context)
    else:
        _implementation_sources = SI.record(**_source_context)

    for callback in (transport.assemble, transport.probe, stage_task, request.resolved_tools):
        T.callback_reference(callback, _implementation_sources, label="admission callback")
    _invocation_treatment = T.record(treatment, _implementation_sources)
    if _resuming and _environment_record.get("invocation_treatment") != _invocation_treatment:
        raise RuntimeError("resume refused: invocation treatment changed or is absent")
    _corpus_record = None
    _public_root, _policy_root = treatment.capsules_root, None
    _contract_root = None
    _reviewed_corpus_roots = None

    # bwrap masks broad output trees, then explicitly rebinds ONLY this workspace last. New mutable
    # work belongs under generated output; recorded/legacy workspaces remain in place when resumed.
    ws_root = select_workspace_root(
        target=context.target, arm=arm, run_dir=run_dir, experiment=context.experiment, resume=_resuming
    )
    ws = ws_root / "workspace"
    # Validate before stale-workspace cleanup: an operator may be preserving a candidate from an
    # interrupted setup whose run directory was never completed.  If that source is inside the workspace
    # this invocation is about to replace, refuse before ``rmtree`` can erase the only copy.
    _seed_source_preflight = None
    if a.seed_submission:
        _seed_source_preflight = RI.validate_seed_submission_source(a.seed_submission, ws / "submission")
    # Never infer abandonment from a stale directory or a missing run record. It may contain the only
    # candidate from an interrupted setup, or still have live workers. A new run needs a new identity.
    _have_ws = _resuming and (ws / "submission").exists()
    if ws_root.exists() and not _have_ws:
        raise FileExistsError(f"workspace already exists; preserve it and choose a new run id: {ws_root}")
    from merlin.common import storage_lifecycle as _storage

    workspace_leases.append(_storage.acquire(ws_root, owner="phase1-qa-workspace"))
    run_dir.mkdir(parents=True, exist_ok=_resuming)
    _operator_errata_record = None
    if a.operator_errata:
        _operator_errata_record = RI.stage_operator_errata(run_dir, a.operator_errata)
    _archived_bundle_manifest = run_dir / "input_bundle_manifest.yaml"
    _authored_bundle_sha256 = None
    if a.sandbox == "bwrap":
        _prepared_bundle = CI.prepare_bundle(
            run_dir,
            _te(),
            bundle_dir / "input_bundle_manifest.yaml",
            bundle,
            contract=context.repo / "merlin/contract",
            capsules_root=treatment.capsules_root,
            environment=_environment_record if _resuming else None,
        )
        bundle, _corpus_record = _prepared_bundle.bundle, _prepared_bundle.corpus_record
        _authored_bundle_sha256 = _prepared_bundle.authored_sha256
    elif not _resuming:
        shutil.copy(bundle_dir / "input_bundle_manifest.yaml", _archived_bundle_manifest)
    _bundle_manifest_sha256 = RI.bundle_manifest_identity(_archived_bundle_manifest, bundle)

    if _have_ws:
        print(f"[resume] reusing existing workspace + submission at {ws}")
        if a.sandbox == "bwrap":
            _BWS.verify_snapshot_binding(
                ws, bundle, _environment_record.get("bundle_input_snapshot"), repo=context.repo
            )
            _BWS.verify_bundle_snapshot(ws, bundle, repo=context.repo)
        denied_names = [Path(d["path"]).name for d in bundle.get("denied", [])]
        viol, copy_report = [], None
    else:
        assembly = transport.assemble(bundle, ws, a.sandbox, context=context)
        denied_names, viol, copy_report = assembly.denied_names, assembly.violations, assembly.copy_report
    _seed_submission_record = None
    if a.seed_submission:
        _seed_submission_record = RI.seed_submission(ws, _seed_source_preflight, run_dir)
    _bundle_snapshot_record = None
    _model_host_lane_snapshot = None
    _hidden_snapshot_record = None
    _hidden_dir = None
    _corpus_seal = os.environ.get("MERLIN_CORPUS_SEAL", "").strip()
    _corpus_review = None
    if _corpus_seal and a.sandbox != "bwrap":
        raise RuntimeError("sealed corpus admission requires a verified native bwrap input snapshot")
    if a.sandbox == "bwrap":
        # Both fresh setup and resume arrive here. Verify before exporting the host-only pointer; every
        # in-process grade and every operator-side broker inherits it, while bwrap_cmd strips it from the
        # agent environment. The model grader independently re-verifies the aggregate before use.
        _BWS.require_snapshot_ownership(_BWS.verify_bundle_snapshot(ws, bundle, repo=context.repo))
        _snapshot_root = _BWS.bundle_snapshot_root(ws).resolve(strict=True)
        if _corpus_seal:
            from merlin_experiments.corpus.release import verify_snapshot

            _corpus_review = verify_snapshot(Path(_corpus_seal), context.descriptor, ws, bundle, repo=context.repo)
            _reviewed_corpus_roots = tuple(_te().graded_roots())
        _bundle_snapshot_record = _BWS.snapshot_record(ws)
        _corpus_view = CI.resolve(ws, bundle, _corpus_record, repo=context.repo, reviewed_roots=_reviewed_corpus_roots)
        _public_root, _policy_root = _corpus_view.public, _corpus_view.policy
        _contract_root = _corpus_view.contract
        _te_setup = _te()
        if _te_setup.numeric_profile is not None:
            from merlin_experiments.corpus.numeric_policy import (
                load_declared_numeric_policy,
                numeric_profile_path,
            )

            if _te_setup.numeric_profile not in [entry["path"] for entry in bundle.get("host_inputs", [])]:
                raise RuntimeError("declared numeric profile is not a host input; regenerate the input bundle")
            [_numeric_profile] = _BWS.snapshot_input_paths(
                ws, bundle, [numeric_profile_path(_te_setup.numeric_profile, repo=context.repo)], repo=context.repo
            )
            load_declared_numeric_policy(_te_setup, repo=context.repo, frozen_profile=_numeric_profile)
        _hidden_dir = RI.hidden_snapshot_dir(_snapshot_root, _te_setup, context.repo)
        _hidden_snapshot_record = RI.subtree_snapshot_record(_hidden_dir)
        if _hidden_snapshot_record["n_capsules"] <= 0:
            raise RuntimeError(f"hidden capsule snapshot contains no capsules: {_hidden_snapshot_record['path']}")
        if _te_setup.host_lane is not None:
            _, _model_host_lane_snapshot = _te_setup.resolve_host_lane(root=_snapshot_root / "repo")
            _model_host_lane_snapshot["run_snapshot"] = _bundle_snapshot_record
        os.environ[_MODEL_HOST_SNAPSHOT_ROOT_ENV] = str(_snapshot_root)
        os.environ["MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"] = json.dumps(_bundle_snapshot_record, sort_keys=True)
        os.environ[_MODEL_HOST_SNAPSHOT_REQUIRED_ENV] = "1"
    else:
        # Do not let an inherited pointer bind an explicitly unsandboxed diagnostic to another run.
        os.environ.pop(_MODEL_HOST_SNAPSHOT_ROOT_ENV, None)
        os.environ.pop("MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD", None)
        os.environ.pop(_MODEL_HOST_SNAPSHOT_REQUIRED_ENV, None)

    if _resuming and _environment_record.get("corpus_review") != _corpus_review:
        raise RuntimeError("resume refused: operator corpus-review identity changed or was removed")

    _scope_roots = {
        "public_roots": [_policy_root] if _policy_root is not None else None,
        "hidden_roots": [_hidden_dir] if _hidden_dir is not None else None,
        "contract": _contract_root,
    }
    _task_scope_record = task_scope(_te(), a.sandbox, repo=context.repo, **_scope_roots)

    # Stage every prompt/document before provenance is written.  On resume these bytes are NEVER rebuilt
    # from the current worktree; they must match the treatment record from the first invocation.
    if not _resuming:
        if treatment.stage_task is not None:
            treatment.stage_task(arm, ws, run_dir, sandbox=a.sandbox, bundle_dir=bundle_dir)
        else:
            stage_task(arm, ws, run_dir, sandbox=a.sandbox, task_scope=_task_scope_record, policy_root=_policy_root)
    elif not (ws / "TASK.md").is_file():
        raise RuntimeError("resume refused: sealed workspace TASK.md is missing")

    _resolved_tool_ids = request.resolved_tools()
    if (
        _hidden_snapshot_record is not None
        and _hidden_snapshot_record["n_capsules"] != _task_scope_record["held_out_capsules"]
    ):
        raise RuntimeError(
            "live descriptor scope and frozen hidden snapshot disagree at setup: "
            f"{_task_scope_record['held_out_capsules']} vs "
            f"{_hidden_snapshot_record['n_capsules']}"
        )
    if _resuming:
        _identity = {
            "run_id": a.run_id,
            "arm": arm,
            "sandbox": a.sandbox,
            "bundle_id": bundle["bundle_id"],
            "condition": bundle.get("condition", "legacy"),
        }
        if _environment_record.get("bundle_manifest_sha256") not in (None, _bundle_manifest_sha256):
            raise RuntimeError("resume refused: archived input bundle manifest identity changed")
        if bundle.get("host_inputs") and not _environment_record.get("bundle_manifest_sha256"):
            raise RuntimeError("resume refused: private-input run has no archived bundle identity")
        _expected_hidden_dir = (
            RI.hidden_snapshot_dir(_snapshot_root, _te(), context.repo) if a.sandbox == "bwrap" else None
        )
        try:
            _hidden_dir = RI.verify_persisted_run_inputs(
                _environment_record,
                identity=_identity,
                task_scope=_task_scope_record,
                ws=ws,
                run_dir=run_dir,
                bundle_dir=bundle_dir,
                resolved_tools=_resolved_tool_ids,
                expected_hidden_dir=_expected_hidden_dir,
            )
        except RuntimeError as exc:
            raise RuntimeError(f"resume refused: {exc}") from exc
        if _hidden_dir is not None:
            _hidden_snapshot_record = dict(_environment_record["hidden_capsule_snapshot"])
        # Carry the original record into every pre-launch check below.  The CLI intentionally forbids
        # supplying a new erratum on resume; the archived bytes from the fresh setup remain authoritative.
        _operator_errata_record = _environment_record.get("operator_errata")
        # Preserve the original setup record verbatim.  In particular, do not replace its start time,
        # exact task hash, or account with whichever process happens to perform the resume.
    else:
        _treatment_snapshot = RI.treatment_snapshot_record(ws, run_dir, bundle_dir, _resolved_tool_ids)
    mask = transport.probe(ws, bundle, a.sandbox, context=context)
    if not _resuming:
        _environment_record = {
            "run_id": a.run_id,
            "arm": arm,
            "model": a.model,
            "effort": a.effort,
            # driver + provider decide how a dollar figure must be READ later: a subscription run's cost
            # is notional, a bedrock run's is metered spend against the budget.
            "driver": a.driver,
            "provider": a.provider,
            "subagent_model": a.subagent_model or None,
            "background_model": a.background_model or None,
            "sandbox": a.sandbox,
            "qa_loop": True,
            "run_config": _run_config,
            "task_scope": _task_scope_record,
            "workspace_path": str(ws),
            "workspace_copy_report": copy_report,
            "seed_submission": _seed_submission_record,
            "operator_errata": _operator_errata_record,
            "bundle_input_snapshot": _bundle_snapshot_record,
            "bundle_manifest_sha256": _bundle_manifest_sha256,
            "authored_bundle_manifest_sha256": _authored_bundle_sha256,
            "public_corpus_input": _corpus_record,
            "corpus_review": _corpus_review,
            "hidden_capsule_snapshot": _hidden_snapshot_record,
            "model_host_lane_snapshot": _model_host_lane_snapshot,
            "repo_sha": repo_sha(repo=context.repo),
            "bundle_id": bundle["bundle_id"],
            "condition": bundle.get("condition", "legacy"),
            # The explicit list remains convenient for analysis; treatment_snapshot binds it to the
            # source declarations and to the exact task/docs that instructed the agent.
            "resolved_tools": list(_resolved_tool_ids),
            "treatment_snapshot": _treatment_snapshot,
            "implementation_sources": _implementation_sources,
            "invocation_treatment": _invocation_treatment,
            "started_at": datetime.now(UTC).isoformat(),
            "isolation_violations": viol,
            "denied_paths_checked": denied_names,
            "golden_mask_selftest": mask,
            "account": request.account,  # which subscription/org budget this arm drew from (provenance)
        }
        _environment_path.write_text(yaml.safe_dump(_environment_record, sort_keys=False))
    if viol:
        print(f"ISOLATION FAILURE: {viol}", file=sys.stderr)
        return 3
    if mask["pilot_golden_visible_to_agent"] == "UNPROVEN":
        print(
            "GOLDEN-MASK UNPROVEN: the probe did not establish the agent's view; "
            f"this is neither a leak nor verified isolation. {mask.get('probe_failure', '')}",
            file=sys.stderr,
        )
        return 5
    if mask["pilot_golden_visible_to_agent"] != "OK":
        print(f"GOLDEN-MASK FAILURE: agent can see golden values: {mask}", file=sys.stderr)
        return 5
    print(f"[setup] isolation ok; golden-mask: {mask}")

    # --- oracle preflight: abort a GRADEABLE run BEFORE spending if its required oracle can't run ------
    # A run graded WITHOUT its numeric oracle can only ever emit `oracle_unavailable` (the atlas 0/11 at
    # ~$43 failure): the agent gets no actionable failure plane and thrashes to timeout. Compute up-front
    # whether THIS target's required oracle is actually runnable — for an external_backend target the mlc
    # arc cosim + the model venv, for arc/chipyard the arc model / sim binaries (all routed from the
    # contract by capsule_runner.oracle_available, no target literal). If it cannot run AND the operator
    # did NOT ask for an explicit `--no-oracle` structure-only smoke, STOP here having launched no agent
    # and spent zero tokens (this is strictly before the first launch_agent in the round loop below).
    from merlin.targetgen import capsule_runner as _CRpf

    _te_pf = _te()
    _ora_ok, _ora_why = _CRpf.oracle_available(_te_pf.target, _te_pf.sim_via)
    (run_dir / "oracle_preflight.yaml").write_text(
        yaml.safe_dump(
            {
                "target": _te_pf.target,
                "sim_via": _te_pf.sim_via,
                "oracle_available": _ora_ok,
                "reason": _ora_why,
                "no_oracle": bool(a.no_oracle),
                "verdict": "GO" if (_ora_ok or a.no_oracle) else "NO_GO",
            },
            sort_keys=False,
        )
    )
    if not _ora_ok and not a.no_oracle:
        print(
            f"NO_GO: {_ora_why} — refusing to launch a gradeable run with no numeric oracle "
            f"(zero tokens spent). Re-run with --no-oracle for an explicit structure-only smoke, or "
            f"set MERLIN_MLC_DIR + build the arc model / model venv.",
            file=sys.stderr,
        )
        return 4
    if not _ora_ok:  # a.no_oracle is set — honest structure-only smoke (see qa_grade/no-oracle path)
        print(
            f"[preflight] oracle unavailable ({_ora_why}); proceeding as an EXPLICIT --no-oracle "
            f"structure-only smoke (NOT gradeable — structural tiers only)."
        )
    else:
        print(f"[preflight] oracle GO: {_ora_why}")

    # Oracle availability and code production are separate admission evidence. The
    # selected backend owns its production probe; resource paths never select it.
    # An absent optional probe returns None (not run), not a successful qualification.
    # Explicit selection failures and failed probes refuse before authoring spends tokens.
    if not a.no_oracle:
        _cg_backend = _te_pf.preflight_codegen_backend
        if _cg_backend is None:
            _cg_ok, _cg_why = _CRpf.codegen_smoke(_te_pf.target)
        else:
            _cg_ok, _cg_why = _CRpf.codegen_smoke(_te_pf.target, backend_target=_cg_backend)
        (run_dir / "codegen_smoke.yaml").write_text(
            yaml.safe_dump(
                {
                    "target": _te_pf.target,
                    "backend_target": _cg_backend,
                    "codegen_ok": _cg_ok,
                    "reason": _cg_why,
                },
                sort_keys=False,
            )
        )
        # `is False`, not falsy: None means the smoke DID NOT RUN (n/a for this emit path), which is
        # neither a pass nor a NO_GO. Gating on falsiness would refuse every target the smoke does not
        # cover; recording None as True is what let a doomed run look verified.
        if _cg_ok is False:
            print(
                f"NO_GO: codegen smoke failed — the target's emit path cannot produce a runnable "
                f"kernel: {_cg_why}. Refusing to launch (zero tokens spent).",
                file=sys.stderr,
            )
            return 4
        print(f"[preflight] codegen smoke: {_cg_why}")

    return PreparedRun(
        request,
        run_dir,
        ws,
        bundle_dir,
        bundle,
        _environment_record,
        _public_root,
        _policy_root,
        _contract_root,
        _hidden_dir,
        _resuming,
        _reviewed_corpus_roots,
        transport,
        stage_task,
    )
