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

if TYPE_CHECKING:
    from merlin.development.backend.session.admission import (
        PreparedRun,
        RunRequest,
        TaskStager,
        WorkspaceTransport,
    )

# ---- session -----------------------------------------------------------------------------------


def _prepare(
    request: RunRequest,
    transport: WorkspaceTransport,
    stage_task: TaskStager,
    *,
    workspace_leases: list,
    snapshot_scopes: ExitStack,
) -> PreparedRun | int:
    """Prepare/resume a run, returning its original refusal code or admitted inputs.

    Call inside workspace_session and keep that wrapper active through authoring and
    completion. Appending the acquired lease preserves uncertain-exit protection if
    preparation or the later continuation raises. No child or paid agent starts here.
    """
    from merlin.development.backend.session.admission import (
        _MODEL_HOST_SNAPSHOT_REQUIRED_ENV,
        _MODEL_HOST_SNAPSHOT_ROOT_ENV,
        PreparedRun,
        RunRequest,
        TaskStager,
        WorkspaceTransport,
        _public_build_selection,
        _verify_client_selection,
        _verify_phase0_handoff,
        _verify_public_build_budget,
        model_python_selection,
        orchestration_owned,
        phase_run_dir,
        task_scope,
        validate_options,
    )
    from merlin.runtime.execution.model_inputs import file_sha256

    a, context, treatment = request.options, request.context, request.treatment
    refusal = validate_options(a)
    if refusal is not None:
        return refusal
    from merlin.development.targetgen.sandbox.preflight import SnapshotGuardUnavailable, require_snapshot_guards

    try:
        _guard_preflight = require_snapshot_guards()
    except SnapshotGuardUnavailable as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        print(json.dumps(exc.record, sort_keys=True), file=sys.stderr)
        return 4
    if request.bundle_manifest.name != "input_bundle_manifest.yaml":
        raise ValueError("admission requires the bundle's canonical input_bundle_manifest.yaml path")
    arm = a.arm
    _run_config = request.run_config

    def _te():
        return load_target_experiment(context.descriptor)

    bundle = yaml.safe_load(request.bundle_manifest.read_text())
    from merlin.development.targetgen.authoring.tool_profile import verify_tool_profile

    if (a.preflight_only or a.codex_canary) and (not isinstance(bundle, dict) or bundle.get("bundle_id") != a.bundle):
        raise ValueError("preflight bundle identity differs from the explicitly selected manifest")
    if not isinstance(bundle, dict):
        raise ValueError("bundle identity requires a manifest mapping")
    from merlin.development.backend.authoring import profiles as AP

    _author_profile = AP.resolve(a.author_profile)
    if _author_profile is not None:
        AP.verify_bundle(_author_profile, bundle)
        if request.compiler_library_selection is not None and not _author_profile.seeded:
            raise ValueError("scratch author profiles cannot acquire a legacy additional SDK")
    elif bundle.get("author_profile") is not None:
        raise ValueError("versioned profile bundle requires its explicit author-profile option")
    _compiler_tool_profile = verify_tool_profile(bundle)
    bundle_dir = request.bundle_manifest.parent
    _corpus_seal = a.corpus_seal or os.environ.get("MERLIN_CORPUS_SEAL", "").strip()
    if a.qualify_submission and not _corpus_seal:
        raise RuntimeError("unpaid qualification requires an explicitly selected reviewed corpus seal")
    if _corpus_seal:
        CI.require_reviewed_bundle(_te(), request.bundle_manifest, bundle)

    run_dir = phase_run_dir(context, arm, a.run_id, resume=a.resume)
    _resuming = run_dir.exists() and a.resume
    if run_dir.exists() and not a.resume and not orchestration_owned(run_dir):
        print(f"run dir exists, refusing to overwrite: {run_dir}", file=sys.stderr)
        return 2
    from merlin.development.backend.campaign import control as joint_control

    joint_control.configure(a, run_dir, resume=_resuming)
    _source_context = request.source_context
    _environment_path = run_dir / "environment.yaml"
    from merlin.development.backend.feedback.certification import hidden_capsule_policy

    _hidden_capsule_policy = hidden_capsule_policy.resolve(context.descriptor, getattr(a, "hidden_capsule_mode", ""))
    if _resuming:
        _environment_record, _implementation_sources = _session_resume.restore(
            a,
            context,
            _environment_path=_environment_path,
            _run_config=_run_config,
            _source_context=_source_context,
            hidden_capsule_policy=hidden_capsule_policy,
            _hidden_capsule_policy=_hidden_capsule_policy,
            verify_readback_record=lambda *args: verify_readback_record(*args),
            _verify_public_build_budget=lambda *args: _verify_public_build_budget(*args),
            _verify_client_selection=lambda *args: _verify_client_selection(*args),
            _verify_authoring_selection=lambda *args: AC.verify_live_selection(*args),
            _verify_source_selection=lambda *args, **kwargs: SI.verify(*args, **kwargs),
        )
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
    _seed_source = a.qualify_submission or a.seed_submission
    if getattr(a, "joint_epoch", "") and not _resuming:
        from merlin.development.backend.campaign.joint_epochs import selected as selected_epoch

        _seed_source = selected_epoch(a)["source"]
    if _seed_source:
        _seed_source_preflight = RI.validate_seed_submission_source(_seed_source, ws / "submission")
    # Never infer abandonment from a stale directory or a missing run record. It may contain the only
    # candidate from an interrupted setup, or still have live workers. A new run needs a new identity.
    _have_ws = _resuming and (ws / "submission").exists()
    if ws_root.exists() and not _have_ws:
        raise FileExistsError(f"workspace already exists; preserve it and choose a new run id: {ws_root}")
    from merlin.common.storage import retention as _storage

    workspace_leases.append(_storage.acquire(ws_root, owner="phase1-qa-workspace"))
    run_dir.mkdir(parents=True, exist_ok=True)
    _guard_receipt = run_dir / f"snapshot_guard_preflight.{_guard_preflight['observed_at_ns']}.json"
    with _guard_receipt.open("x", encoding="utf-8") as stream:
        json.dump(_guard_preflight, stream, sort_keys=True, indent=2)
    # The compiler history lives beside the run record, outside the agent's writable workspace. Only a
    # sandboxed run keeps one: copy mode is a diagnostic that launches no tools and is never admitted.
    if a.sandbox == "bwrap":
        _oot_history.start(run_dir, sandbox_roots=(ws_root,), resuming=_resuming)
    _operator_errata_record = None
    if a.operator_errata:
        _operator_errata_record = RI.stage_operator_errata(run_dir, a.operator_errata)
    _archived_bundle_manifest = run_dir / "input_bundle_manifest.yaml"
    _authored_bundle_sha256 = None
    if a.private_full_model_spec and a.sandbox != "bwrap":
        raise RuntimeError("operator-private full-model validation requires the isolated bwrap run")
    _prepared_private_full_model_record = None
    if a.sandbox == "bwrap":
        _prepared_bundle = CI.prepare_bundle(
            run_dir,
            _te(),
            bundle_dir / "input_bundle_manifest.yaml",
            bundle,
            contract=context.repo / "src/merlin/resources/contracts",
            capsules_root=treatment.capsules_root,
            environment=_environment_record if _resuming else None,
            private_full_model_spec=Path(a.private_full_model_spec) if a.private_full_model_spec else None,
        )
        bundle, _corpus_record = _prepared_bundle.bundle, _prepared_bundle.corpus_record
        _prepared_private_full_model_record = _prepared_bundle.private_full_model_record
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
    from merlin.development.backend.session.inputs import public_software_declaration as PSD

    _public_software = (
        _environment_record.get(PSD.KEY)
        if _resuming
        else (PSD.stage(ws, bundle, repo=context.repo) if a.sandbox == "bwrap" else None)
    )
    PSD.verify(_public_software, ws, bundle, repo=context.repo)
    _seed_submission_record = None
    if getattr(a, "joint_epoch", "") and not _resuming:
        from merlin.development.backend.campaign.joint_epochs import seed as seed_epoch

        _seed_submission_record = seed_epoch(a, ws, run_dir)
    elif a.joint_campaign and a.seed_submission:
        from merlin.development.backend.campaign.seeding.joint_seed import stage as stage_reviewed_seed

        _seed_submission_record = stage_reviewed_seed(a, ws, run_dir)
    elif _seed_source:
        _seed_submission_record = RI.seed_submission(
            ws,
            _seed_source_preflight,
            run_dir,
            require_stable_source=bool(a.qualify_submission),
            require_lineage=bool(a.seed_submission),
        )
    _full_compiler_record = None
    if a.compiler_access == "full_compiler":
        from merlin.development.backend.authoring.library import compiler_workspace as FCW

        selection = FCW.select(a.full_compiler_root, a.full_compiler_manifest, a.full_compiler_manifest_sha256)
        if _author_profile is not None:
            AP.verify_sources(_author_profile, selection, bundle=bundle)
        if _resuming:
            _full_compiler_record = _environment_record.get("full_compiler_workspace")
            if (
                not isinstance(_full_compiler_record, dict)
                or _full_compiler_record.get("selection") != selection.record()
            ):
                raise RuntimeError("resume refused: full compiler baseline selection changed")
        else:
            _full_compiler_record = FCW.stage(selection, ws / "submission")
    if _author_profile is not None:
        AP.stage_record(ws, run_dir, _author_profile, selection if a.compiler_access == "full_compiler" else None)
    _bundle_snapshot_record = None
    _model_host_lane_snapshot = None
    _hidden_snapshot_record = None
    _hidden_dir = None
    _corpus_review = None
    _semantic_model_snapshot = None
    _semantic_diagnostic = None
    _selected_public_facts = None
    if a.sandbox == "bwrap":
        from merlin.common.storage.bundle_snapshots import verified_snapshot_scope

        # Assembly and seed staging are complete. The scope reuses only verified
        # immutable snapshot content within this preparation and rehashes it fully
        # on close, including every refusal/exception, before anything is returned.
        snapshot_scopes.enter_context(verified_snapshot_scope(ws, bundle, repo=context.repo))

    if _corpus_seal and a.sandbox != "bwrap":
        raise RuntimeError("sealed corpus admission requires a verified native bwrap input snapshot")
    if a.sandbox == "bwrap":
        # Both fresh setup and resume arrive here. Verify before exporting the host-only pointer; every
        # in-process grade and every operator-side broker inherits it, while bwrap_cmd strips it from the
        # agent environment. The model grader independently re-verifies the aggregate before use.
        _BWS.require_snapshot_ownership(_BWS.verify_bundle_snapshot(ws, bundle, repo=context.repo))
        _snapshot_root = _BWS.bundle_snapshot_root(ws).resolve(strict=True)
        if _corpus_seal:
            from merlin.experiments.corpus.release import verify_snapshot_for_phase1

            _reviewed = verify_snapshot_for_phase1(
                Path(_corpus_seal), context.descriptor, ws, bundle, repo=context.repo
            )
            _corpus_review = _reviewed.review
            _semantic_model_snapshot = _reviewed.private_instruction_model
            _reviewed_corpus_roots = tuple(_te().graded_roots())
        _bundle_snapshot_record = _BWS.snapshot_record(ws)
        from merlin.development.backend.session.inputs.run_record import select as select_frozen_facts

        _selected_public_facts = _BWS.frozen_selected_rtl_facts(ws, bundle, repo=context.repo)
        select_frozen_facts(ws, _selected_public_facts)
        _corpus_view = CI.resolve(ws, bundle, _corpus_record, repo=context.repo, reviewed_roots=_reviewed_corpus_roots)
        _verify_phase0_handoff(_corpus_review, _corpus_view.workload_coverage)
        _public_root, _policy_root = _corpus_view.public, _corpus_view.policy
        _contract_root = _corpus_view.contract
        from merlin.development.backend.feedback.scalability.public_scalability import (
            install as install_scalability,
        )

        install_scalability(_te(), _corpus_view.scalability)
        if _corpus_seal:
            from merlin.development.backend.telemetry import evidence as _SD

            _model_path = _semantic_model_snapshot
            if _resuming:
                # Old reviewed runs had no diagnostic. Do not mint a new
                # receipt or change their admission identity on resume.
                if "semantic_search_diagnostic" in _environment_record:
                    _SD.verify(
                        _environment_record["semantic_search_diagnostic"],
                        run_dir,
                        workspace=ws,
                        model_path=_model_path,
                        public_root=_public_root,
                    )
        _te_setup = _te()
        if _te_setup.numeric_profile is not None:
            from merlin.experiments.corpus.numeric_policy import (
                load_declared_numeric_policy,
                numeric_profile_path,
            )

            # Whole-model grading resolves the descriptor through the same frozen
            # snapshot as the numerical recipe. Refuse an incomplete bundle here,
            # before an agent spends a round producing an ungradeable compiler.
            descriptor_source = Path(_te_setup.path)
            if not descriptor_source.is_absolute():
                descriptor_source = context.repo / descriptor_source
            host_inputs = set()
            for entry in bundle.get("host_inputs", []):
                source = Path(entry["path"])
                host_inputs.add((source if source.is_absolute() else context.repo / source).absolute())
            if descriptor_source.absolute() not in host_inputs:
                raise RuntimeError("experiment descriptor is not a host input; regenerate the input bundle")
            _BWS.snapshot_input_paths(ws, bundle, [descriptor_source], repo=context.repo)
            if _te_setup.numeric_profile not in [entry["path"] for entry in bundle.get("host_inputs", [])]:
                raise RuntimeError("declared numeric profile is not a host input; regenerate the input bundle")
            [_numeric_profile] = _BWS.snapshot_input_paths(
                ws, bundle, [numeric_profile_path(_te_setup.numeric_profile, repo=context.repo)], repo=context.repo
            )
            load_declared_numeric_policy(
                _te_setup,
                repo=context.repo,
                frozen_profile=_numeric_profile,
                frozen_resolver=lambda source: _BWS.snapshot_input_paths(
                    ws,
                    bundle,
                    [source],
                    repo=context.repo,
                )[0],
            )
        _hidden_dir = RI.hidden_snapshot_dir(_snapshot_root, _te_setup, context.repo)
        _hidden_snapshot_record = RI.subtree_snapshot_record(_hidden_dir)
        if _hidden_snapshot_record["n_capsules"] <= 0:
            raise RuntimeError(f"hidden capsule snapshot contains no capsules: {_hidden_snapshot_record['path']}")
        if _corpus_seal and not _resuming:
            from merlin.development.backend.session.preflight import require_ready as require_trace_ready
            from merlin.development.testgen.evidence.coverage_commitment import admitted_source_roots

            require_trace_ready(
                admitted_source_roots(_policy_root, _public_root, contract=_contract_root),
                target=context.target,
                contract=_contract_root,
                output=run_dir / "trace_readiness.json",
                hidden_roots=_hidden_dir,
                expected_hidden_capsules=_hidden_snapshot_record["n_capsules"],
            )
        if _te_setup.host_lane is not None:
            _, _model_host_lane_snapshot = _te_setup.resolve_host_lane(root=_snapshot_root / "repo")
            _model_host_lane_snapshot["run_snapshot"] = _bundle_snapshot_record
        os.environ[_MODEL_HOST_SNAPSHOT_ROOT_ENV] = str(_snapshot_root)
        os.environ["MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD"] = json.dumps(_bundle_snapshot_record, sort_keys=True)
        os.environ[_MODEL_HOST_SNAPSHOT_REQUIRED_ENV] = "1"
    else:
        # Do not let an inherited pointer bind an explicitly unsandboxed diagnostic to another run.
        os.environ.pop("MERLIN_RTL_FACTS", None)
        os.environ.pop(_MODEL_HOST_SNAPSHOT_ROOT_ENV, None)
        os.environ.pop("MERLIN_MODEL_HOST_LANE_SNAPSHOT_RECORD", None)
        os.environ.pop(_MODEL_HOST_SNAPSHOT_REQUIRED_ENV, None)

    _private_full_model_spec = None
    _private_full_model_record = None
    if a.private_full_model_spec:
        _private_source = Path(a.private_full_model_spec).absolute()
        if _private_source.is_symlink() or not _private_source.is_file():
            raise RuntimeError("operator-private full-model specification is absent or indirect")
        _private_full_model_record = _prepared_private_full_model_record
        if (
            not isinstance(_private_full_model_record, dict)
            or _private_full_model_record.get("source") != str(_private_source)
            or _private_full_model_record.get("source_sha256") != file_sha256(_private_source)
        ):
            raise RuntimeError("operator-private full-model source differs from the frozen host input")
        [_private_full_model_spec] = _BWS.snapshot_input_paths(
            ws, bundle, [Path(_private_full_model_record["path"])], repo=context.repo
        )
        _private_full_model_record = {
            **_private_full_model_record,
            "frozen_path": str(_private_full_model_spec),
            "frozen_sha256": file_sha256(_private_full_model_spec),
        }
    if _resuming and _environment_record.get("private_full_model_spec") != _private_full_model_record:
        raise RuntimeError("resume refused: operator-private full-model specification changed")

    from merlin.development.backend.qualification.provenance import private_instruction_declaration

    _instruction_selection, _instruction_record = private_instruction_declaration.freeze_for_run(
        a.instruction_selection,
        target=context.target,
        run_dir=run_dir,
        prior=_environment_record.get("private_instruction_selection") if _resuming else None,
        resuming=_resuming,
    )
    from merlin.development.backend.qualification.provenance import private_result_format_selection

    _, _native_result_record = private_result_format_selection.freeze_for_run(
        a.native_result_format_selection,
        target=context.target,
        run_dir=run_dir,
        prior=_environment_record.get("native_result_format_selection") if _resuming else None,
        resuming=_resuming,
    )

    # Operator-side selfcheck brokers inherit the same pinned host build inputs as formal grading.
    # Install before authoring; preparing only during formal grading is too late for selfchecks.
    from merlin.development.backend.feedback.scalability import public_owned_host_selection

    public_owned_host_selection.install_for_session(
        _private_full_model_spec,
        _instruction_selection,
        target=context.target,
        run_dir=run_dir,
        selected_facts=_selected_public_facts,
        selected_facts_sha256=file_sha256(_selected_public_facts) if _selected_public_facts is not None else None,
        descriptor=context.descriptor,
    )
    from merlin.development.backend.feedback.scalability import public_owned_host_inputs

    _public_host_selection, _public_host_record = public_owned_host_inputs.freeze_for_run(
        a.public_owned_host_inputs,
        a.public_owned_host_inputs_sha256,
        target=context.target,
        descriptor=context.descriptor,
        run_dir=run_dir,
        repo=context.repo,
        prior=_environment_record.get(public_owned_host_inputs.KEY) if _resuming else None,
        resuming=_resuming,
    )
    if _public_host_selection is not None:
        os.environ[public_owned_host_selection.ENV] = json.dumps(_public_host_selection, sort_keys=True)

    if _resuming and _environment_record.get("corpus_review") != _corpus_review:
        raise RuntimeError("resume refused: operator corpus-review identity changed or was removed")

    _scope_roots = {
        "public_roots": [_policy_root] if _policy_root is not None else None,
        "hidden_roots": [_hidden_dir] if _hidden_dir is not None else None,
        "contract": _contract_root,
    }
    _task_scope_record = task_scope(_te(), a.sandbox, repo=context.repo, **_scope_roots)
    try:
        _model_python_selection = model_python_selection(
            _te(),
            repo=context.repo,
            public_roots=_scope_roots["public_roots"],
            hidden_roots=_scope_roots["hidden_roots"],
            contract=_contract_root,
            prior=_environment_record if _resuming else None,
        )
    except (OSError, ValueError) as exc:
        print(f"NO_GO: {exc}. Refusing model authoring before grade or agent spend.", file=sys.stderr)
        return 4
    if _resuming and _environment_record.get("model_python_selection") != _model_python_selection:
        raise RuntimeError("resume refused: selected model Python path or bytes changed")
    _public_build_selection_record = _public_build_selection(context.target, a.public_object_build_budget_s)
    if _resuming and _environment_record.get("public_object_build_selection") != _public_build_selection_record:
        raise RuntimeError("resume refused: public object-build tool/source selection changed")

    _library_binding = CL.freeze_for_run(
        request, run_dir, prior=_environment_record.get(CL.KEY) if _resuming else None, resuming=_resuming
    )
    _library_identity = CL.encoded(_library_binding)

    # Stage every prompt/document before provenance is written.  On resume these bytes are NEVER rebuilt
    # from the current worktree; they must match the treatment record from the first invocation.
    _session_guidance.stage(
        request,
        a=a,
        context=context,
        treatment=treatment,
        arm=arm,
        run_dir=run_dir,
        ws=ws,
        bundle_dir=bundle_dir,
        bundle=bundle,
        _resuming=_resuming,
        stage_task=stage_task,
        _task_scope_record=_task_scope_record,
        _policy_root=_policy_root,
        _public_root=_public_root,
        _contract_root=_contract_root,
        _te=_te,
        _author_profile=_author_profile,
        _library_binding=_library_binding,
        _full_compiler_record=_full_compiler_record,
        _public_software=_public_software,
        _run_config=_run_config,
        append_library_guidance=lambda *args, **kwargs: CL.append_guidance(*args, **kwargs),
        append_software_guidance=lambda *args: PSD.append_guidance(*args),
        context_selection=lambda *args: CTX.selection(*args),
        skill_configuration=lambda configuration: configuration.get(SK.KEY),
        stage_skills=lambda *args: SK.stage(*args),
    )
    _resolved_tool_ids = request.resolved_tools()
    CL.verify(request, {CL.KEY: _library_binding}, _library_identity)
    verify_tool_profile(bundle, tools=tuple(_resolved_tool_ids))
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
            "lineage_id": (
                ((_environment_record.get("seed_submission") or {}).get("lineage") or {}).get("lineage_id") or a.run_id
            ),
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
    if _corpus_seal and not _resuming:
        # The receipt is host-private. Mint it only after all prompt staging
        # and the trusted workspace probe have finished, so neither receives
        # a run directory that already contains search history.
        from merlin.development.backend.telemetry import evidence as _SD

        _semantic_diagnostic = _SD.create(
            run_dir,
            workspace=ws,
            model_path=_semantic_model_snapshot,
            public_root=_public_root,
            contract_root=_contract_root,
        )
    if not _resuming:
        from merlin.development.backend.providers.codex_runtime import select as select_client

        _backend_origin = FB.origin(a, ws, compiler_library=request.compiler_library_selection)
        _library_snapshot = CL.verify(request, {CL.KEY: _library_binding}, _library_identity)
        _client_runtime = select_client(a, context, bundle, ws, run_dir, compiler_library_selection=_library_snapshot)
        _environment_record = {
            "run_id": a.run_id,
            "lineage_id": ((_seed_submission_record or {}).get("lineage") or {}).get("lineage_id") or a.run_id,
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
            "qa_loop": not bool(a.qualify_submission),
            "validation_scope": getattr(a, "validation_scope", "full"),
            **({"qualification_only": True} if a.qualify_submission else {}),
            "run_config": _run_config,
            **({"full_compiler_workspace": _full_compiler_record} if _full_compiler_record is not None else {}),
            **({CL.KEY: _library_binding} if _library_binding is not None else {}),
            **({PSD.KEY: _public_software} if _public_software is not None else {}),
            "hidden_capsule_validation": _hidden_capsule_policy,
            **({"codex_runtime_policy": _client_runtime.record()} if _client_runtime is not None else {}),
            **(
                {"public_object_build_selection": _public_build_selection_record}
                if _public_build_selection_record is not None
                else {}
            ),
            **({"readback_policy": readback_record(context)} if context.readback_policy is not None else {}),
            "task_scope": _task_scope_record,
            **({"model_python_selection": _model_python_selection} if _model_python_selection is not None else {}),
            "workspace_path": str(ws),
            "workspace_copy_report": copy_report,
            "seed_submission": _seed_submission_record,
            **({"backend_origin": _backend_origin} if _backend_origin is not None else {}),
            "operator_errata": _operator_errata_record,
            "bundle_input_snapshot": _bundle_snapshot_record,
            "bundle_manifest_sha256": _bundle_manifest_sha256,
            "authored_bundle_manifest_sha256": _authored_bundle_sha256,
            "public_corpus_input": _corpus_record,
            "corpus_review": _corpus_review,
            **({"semantic_search_diagnostic": _semantic_diagnostic} if _corpus_seal else {}),
            "hidden_capsule_snapshot": _hidden_snapshot_record,
            "model_host_lane_snapshot": _model_host_lane_snapshot,
            "private_full_model_spec": _private_full_model_record,
            "private_instruction_selection": _instruction_record,
            **({public_owned_host_inputs.KEY: _public_host_record} if _public_host_record is not None else {}),
            **({"native_result_format_selection": _native_result_record} if _native_result_record is not None else {}),
            "repo_sha": repo_sha(repo=context.repo),
            "bundle_id": bundle["bundle_id"],
            "condition": bundle.get("condition", "legacy"),
            # The explicit list remains convenient for analysis; treatment_snapshot binds it to the
            # source declarations and to the exact task/docs that instructed the agent.
            "resolved_tools": list(_resolved_tool_ids),
            **({"compiler_tool_profile": _compiler_tool_profile} if _compiler_tool_profile is not None else {}),
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
    refusal = _session_preflight.check(
        a,
        context,
        _te=_te,
        run_dir=run_dir,
        ws=ws,
        bundle=bundle,
        _resolved_tool_ids=_resolved_tool_ids,
        _public_root=_public_root,
        _bundle_snapshot_record=_bundle_snapshot_record,
        _bundle_manifest_sha256=_bundle_manifest_sha256,
    )
    if refusal is not None:
        return refusal

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
        _private_full_model_spec,
        _instruction_selection,
        _library_identity,
    )
