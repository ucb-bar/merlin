"""Run the independent oracle, compiler and author-tool admission before spend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from merlin.evaluation.capsule.evidence import certificates as capsule_availability
from merlin.target.isa.rocc.semantics import BoundSemantics

# ---- session_preflight -------------------------------------------------------------------------


def check(
    a,
    context,
    *,
    _te,
    run_dir,
    ws,
    bundle,
    _resolved_tool_ids,
    _public_root,
    _bundle_snapshot_record,
    _bundle_manifest_sha256,
):
    """Return the original refusal code, or None after every selected check."""

    _te_pf = _te()
    if context.readback_policy is not None and not a.no_oracle:
        from merlin.development.backend.context import readback_kwargs
        from merlin.target.contract.harness.readback_policy import MEMORY_TRANSPORTS

        if context.readback_policy in MEMORY_TRANSPORTS:
            # Construct the original full ladder before authoring; a global
            # memory override must be supported by every selected engine.
            capsule_availability.qa_checkpoint_adapters(_te_pf.target, _te_pf.sim_via, **readback_kwargs(context))
    _ora_ok, _ora_why = capsule_availability.oracle_available(_te_pf.target, _te_pf.sim_via)
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

    # Presence of a simulator is not proof that its compiler can build a kernel.
    # Exercise the selected compile-based oracle toolchain before any agent spend;
    # non-compile oracles return an explicit n/a from the shared capability check.
    if not a.no_oracle:
        from merlin.target.simulators import spike_runtime as _RBpf

        _compiler_ok, _compiler_why = _RBpf.compiler_smoke(_te_pf.sim_via)
        (run_dir / "compiler_smoke.yaml").write_text(
            yaml.safe_dump(
                {
                    "target": _te_pf.target,
                    "sim_via": _te_pf.sim_via,
                    "compiler_ok": _compiler_ok,
                    "reason": _compiler_why,
                },
                sort_keys=False,
            )
        )
        if not _compiler_ok:
            print(
                f"NO_GO: oracle compile toolchain failed: {_compiler_why}. Refusing to launch (zero tokens spent).",
                file=sys.stderr,
            )
            return 4
        print(f"[preflight] compiler smoke: {_compiler_why}")

    # Exercise the assisted tool surface using this run's verified frozen bundle.
    # The receipt is host-private, and a failed probe refuses before authoring.
    from merlin.development.backend.authoring import readiness as _TRpf

    if a.qualify_submission:
        _tooling = {
            "status": "not_required",
            "checks": [],
            "scope": "unpaid frozen-submission qualification starts no author or authoring-tool clients; "
            "formal compiler, oracle, isolation and input-integrity gates remain required",
        }
        (run_dir / "tooling_readiness.yaml").write_text(yaml.safe_dump(_tooling, sort_keys=False))
    else:
        _tooling = _TRpf.run(
            context,
            _te_pf,
            ws,
            bundle,
            tuple(_resolved_tool_ids),
            _public_root,
            run_dir,
            _bundle_snapshot_record,
            _bundle_manifest_sha256,
            without_tools=tuple(a.without_tool),
        )
    if _tooling["status"] == "no_go":
        failed = [f"{check['name']}: {check['detail']}" for check in _tooling["checks"] if not check["ok"]]
        print(
            f"NO_GO: selected authoring tools are not ready: {'; '.join(failed)}. "
            "Refusing to launch (zero tokens spent).",
            file=sys.stderr,
        )
        return 4
    print(f"[preflight] tooling readiness: {_tooling['status']}")

    # Oracle availability and code production are separate admission evidence. The
    # selected backend owns its production probe; resource paths never select it.
    # An absent optional probe returns None (not run), not a successful qualification.
    # Explicit selection failures and failed probes refuse before authoring spends tokens.
    if not a.no_oracle:
        _cg_backend = _te_pf.preflight_codegen_backend
        if _cg_backend is None:
            _cg_ok, _cg_why = capsule_availability.codegen_smoke(_te_pf.target)
        else:
            _cg_ok, _cg_why = capsule_availability.codegen_smoke(_te_pf.target, backend_target=_cg_backend)
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
    return None


# ---- session_guidance --------------------------------------------------------------------------
# Stage selected session guidance before freezing its provenance and tool view.


def stage(
    request,
    *,
    a,
    context,
    treatment,
    arm,
    run_dir,
    ws,
    bundle_dir,
    bundle,
    _resuming,
    stage_task,
    _task_scope_record,
    _policy_root,
    _public_root,
    _contract_root,
    _te,
    _author_profile,
    _library_binding,
    _full_compiler_record,
    _public_software,
    _run_config,
    append_library_guidance,
    append_software_guidance,
    context_selection,
    skill_configuration,
    stage_skills,
):
    """Retain the original prompt ordering and resolve mutable hooks at each call."""
    if not _resuming:
        if treatment.stage_task is not None:
            treatment.stage_task(arm, ws, run_dir, sandbox=a.sandbox, bundle_dir=bundle_dir)
        else:
            stage_task(arm, ws, run_dir, sandbox=a.sandbox, task_scope=_task_scope_record, policy_root=_policy_root)
        from merlin.development.backend.authoring.resource_prompt import append_resource_guidance
        from merlin.development.targetgen.execution.host.public_codegen_budget import (
            policy_from_profile,
            require_public_members,
        )

        resource_policy = policy_from_profile(bundle.get("compiler_tool_profile") or {})
        if resource_policy is not None:
            from merlin.development.targetgen.corpus.capsule_io import discover_capsules

            public_members = discover_capsules(
                [_policy_root] if _policy_root is not None else _te().graded_roots(),
                labels={"public", "dev"},
                contract=_contract_root or context.repo / "src/merlin/resources/contracts",
            )
            excluded = set(_te().effective_exclusions(row.get("name") for row in public_members))
            require_public_members(
                resource_policy, (row.get("name") for row in public_members if row.get("name") not in excluded)
            )
        if _author_profile is not None:
            from merlin.development.backend.authoring.profile_guidance import append_profile_guidance

            append_profile_guidance(
                ws,
                run_dir,
                a,
                context=context,
                public_roots=[_public_root] if _public_root is not None else _te().graded_roots(),
                contract=_contract_root or context.repo / "src/merlin/resources/contracts",
            )
        else:
            append_resource_guidance(
                ws,
                run_dir,
                a,
                **({"public_codegen_policy": resource_policy} if resource_policy is not None else {}),
                **(
                    {"task_staging_config": request.task_staging_config}
                    if request.task_staging_config is not None
                    else {}
                ),
            )
    elif not (ws / "TASK.md").is_file():
        raise RuntimeError("resume refused: sealed workspace TASK.md is missing")

    if not _resuming:
        append_library_guidance(
            ws,
            run_dir,
            _library_binding,
            request.compiler_library_selection,
            repo=context.repo,
            descriptor=context.descriptor,
            staging_record=_full_compiler_record,
        )
        append_software_guidance(_public_software, ws, run_dir)
        from merlin.development.backend.session.inputs.capability_inventory import (
            stage_selected as stage_capabilities,
        )

        capability_append = (
            stage_capabilities if _author_profile is None or _author_profile.modern else (lambda *args, **kwargs: None)
        )
        capability_append(
            ws,
            run_dir,
            context=context,
            public_roots=[_public_root] if _public_root is not None else _te().graded_roots(),
            contract=_contract_root or context.repo / "src/merlin/resources/contracts",
            excluded=_te().effective_exclusions,
        )
    if not _resuming:
        from merlin.development.backend.feedback.references.experiment_guidance import stage_guidance
        from merlin.development.backend.feedback.references.reference_map import stage as stage_reference_map

        stage_reference_map(ws, run_dir)  # the same pinned map in every arm, before the selection's section
        stage_guidance(ws, run_dir)
        _skills_record = skill_configuration(_run_config) or {}
        if (context_selection(a) or {}).get("task_skills"):
            from merlin.development.backend.authoring.task_disclosure import stage as stage_disclosure

            # after every TASK.md append, before the treatment seal; staged author skills carry the
            # feedback-file guide, so the generated one is left out
            stage_disclosure(ws, run_dir, feedback=not _skills_record.get("enabled"))
        if _skills_record.get("enabled"):
            stage_skills(ws, run_dir, _skills_record)  # beside the task-* skills, before the treatment seal


# ---- session_resume ----------------------------------------------------------------------------
# Restore and verify the original frozen session environment before admission.


def restore(
    a,
    context,
    *,
    _environment_path,
    _run_config,
    _source_context,
    hidden_capsule_policy,
    _hidden_capsule_policy,
    verify_readback_record,
    _verify_public_build_budget,
    _verify_client_selection,
    _verify_authoring_selection,
    _verify_source_selection,
):
    """Reopen the prior record and apply the caller's original verification hooks."""
    try:
        _environment_record = yaml.safe_load(_environment_path.read_text())
    except Exception as exc:  # noqa: BLE001 — provenance is a formal resume gate
        raise RuntimeError(f"resume refused: environment record unreadable: {exc}") from exc
    if not isinstance(_environment_record, dict):
        raise RuntimeError("resume refused: environment record is not a mapping")
    from merlin.development.backend.feedback.grading import worker_resources as sandbox_pool

    # A resumed run keeps the sandbox mechanism it started with (none recorded: one per call).
    _run_config["sandbox_reuse"] = sandbox_pool.adopt_recorded(_environment_record.get("run_config"))
    prior_mode = (_environment_record.get("run_config") or {}).get("compiler_execution", "profile_strict_v1")
    if prior_mode != a.compiler_execution:
        raise RuntimeError("resume refused: compiler execution treatment changed; use a new checkpoint continuation")
    if (_environment_record.get("run_config") or {}).get("author_profile", "") != a.author_profile:
        raise RuntimeError("resume refused: versioned author profile changed")
    prior_access = (_environment_record.get("run_config") or {}).get("compiler_access", "selected_library")
    prior_manifest = (_environment_record.get("run_config") or {}).get("full_compiler_manifest_sha256", "")
    if (prior_access, prior_manifest) != (a.compiler_access, a.full_compiler_manifest_sha256):
        raise RuntimeError("resume refused: editable compiler source selection changed")
    prior_reference = (_environment_record.get("run_config") or {}).get("reference_experiments_sha256", "")
    if prior_reference != a.reference_experiments_sha256:
        raise RuntimeError("resume refused: reference experiment feedback selection changed")
    prior_tasks = (_environment_record.get("run_config") or {}).get("reference_experiments_tasks", "on")
    if prior_tasks != a.reference_experiments_tasks:
        raise RuntimeError("resume refused: reference experiment task serving changed")
    from merlin.development.backend.feedback.certification.capsule_certification import verify_scope

    verify_scope(getattr(a, "validation_scope", "full"), _environment_record)
    verify_readback_record(context, _environment_record.get("readback_policy"))
    hidden_capsule_policy.verify_record(
        _hidden_capsule_policy,
        _environment_record.get("hidden_capsule_validation"),
    )
    _verify_public_build_budget(a, _environment_record)
    _verify_authoring_selection(a, _environment_record)
    _verify_client_selection(a, _environment_record)
    _implementation_sources = _environment_record.get("implementation_sources")
    _verify_source_selection(_implementation_sources, **_source_context)
    return _environment_record, _implementation_sources


# ---- trace_readiness ---------------------------------------------------------------------------
# Check selected trace evidence against the complete declared capsule requirements.
#
# This is a static input check before paid authoring. It runs no compiler or
# simulator and never invents an operand layout or turns unavailable evidence
# into a hardware capability.


def _observable_fields(program: dict) -> set[str]:
    available = set()
    for step in program.get("fields", []):
        kind = step["as"]
        if kind in {"const", "operand", "raw"}:
            known = True
        elif kind == "equals":
            known = step.get("from") in available
        elif kind == "all_equals":
            known = all(row.get("from") in available for row in step.get("predicates", []))
        else:
            known = type(step.get("offset")) is int and type(step.get("width")) is int
        if known:
            available.add(step["name"])
    return available


def inspect(capsules: list[dict], *, semantics, contract: dict) -> dict:
    """Report whether the original selected decoder can observe required modes.

    Availability is not a claim that a future compiler emits the right commands.
    Those commands and their numerics still need actual execution/trace checks.
    Legacy non-bound backends keep their existing decoder admission path.
    """
    record = {
        "schema": "merlin.phase1.trace_readiness.v1",
        "status": "ready",
        "n_capsules": len(capsules),
        "requirements": {},
        "blockers": [],
        "scope": "selected evidence availability only; no compiler or execution qualification",
    }
    if type(semantics) is not BoundSemantics:
        record["applicability"] = "legacy_decoder"
        return record
    observer = semantics.operand_observer
    isa = observer.isa_constants(semantics._target) if observer is not None else {}
    fields = {name: _observable_fields(program) for name, program in isa.get("OPERAND_FIELDS", {}).items()}
    classes = set(isa.get("FUNCT_CLASS", {}).values()) | set(isa.get("CONFIG_SUBTYPE", {}).values())
    roles = (contract.get("corpus_instruction_roles") or {}).get("by_role") or {}
    requirements = {}
    for capsule in capsules:
        if capsule.get("kind") == "model" or capsule.get("expected", {}).get("refusal"):
            continue
        for mode, required in (capsule.get("expected", {}).get("modes") or {}).items():
            if required:
                requirements.setdefault(mode, []).append(capsule["name"])
    record["requirements"] = {mode: sorted(names) for mode, names in sorted(requirements.items())}
    for mode, names in record["requirements"].items():
        reason = None
        if observer is None:
            if mode != "movement":
                reason = "missing_selected_operand_protocol"
            elif not roles.get("memory") or not roles.get("acc_readout"):
                reason = "missing_selected_movement_roles"
        else:
            required_fields = {
                "i8": {"MVOUT": {"readout"}},
                "relu": {"CONFIG_ST": {"relu"}},
                "acc_scale": {"CONFIG_ST": {"acc_scale"}},
                "k_accumulate": {"PRELOAD": {"accumulate"}},
            }.get(mode, {})
            if any(not needed <= fields.get(label, set()) for label, needed in required_fields.items()):
                reason = "unresolved_selected_operand_fields"
            if (
                reason is None
                and mode in {"relu", "acc_scale", "load_state_resident"}
                and isa.get("CONFIG_SELECTOR") is None
            ):
                reason = "unresolved_selected_config_selector"
            if mode == "resident_reuse":
                loads = {label for label in classes if label in {"MVIN", "MVIN2", "MVIN3"}}
                if not loads or any(
                    not {"spad_addr", "dram", "rows", "cols"} <= fields.get(label, set()) for label in loads
                ):
                    reason = "unresolved_selected_load_identity"
            if mode == "movement" and (not classes & {"MVIN", "MVIN2", "MVIN3"} or "MVOUT" not in classes):
                reason = "missing_selected_movement_roles"
            if mode == "load_state_resident" and observer.load_states is None:
                reason = "missing_selected_load_state_evidence"
        if reason:
            record["blockers"].append({"mode": mode, "reason": reason, "capsules": names})
    if record["blockers"]:
        record["status"] = "blocked"
    return record


def require_ready(
    roots, *, target: str, contract: Path, output: Path, hidden_roots=None, expected_hidden_capsules: int | None = None
) -> dict:
    """Use the same selected backend and capsule discovery as grading, once per admission."""
    from merlin.development.targetgen.corpus.capsule_io import discover_capsules
    from merlin.runtime.backends import registry as base

    backend = base.get_backend(target)
    capsules = discover_capsules(roots, labels={"public", "dev", "hidden"}, contract=contract)
    hidden = []
    if hidden_roots is not None:
        hidden = discover_capsules(hidden_roots, labels={"hidden"}, contract=contract)
        capsules += hidden
    if expected_hidden_capsules is not None and (
        type(expected_hidden_capsules) is not int
        or expected_hidden_capsules < 0
        or len(hidden) != expected_hidden_capsules
    ):
        raise ValueError("trace readiness hidden cohort differs from the verified hidden snapshot")
    names = [capsule["name"] for capsule in capsules]
    if len(names) != len(set(names)):
        raise ValueError("trace readiness cohort has duplicate capsule ownership")
    semantics = getattr(backend, "rocc_semantics", None)
    data = backend._data()["contract"] if type(semantics) is BoundSemantics else {}
    record = inspect(capsules, semantics=semantics, contract=data)
    output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    if record["status"] != "ready":
        modes = ", ".join(row["mode"] for row in record["blockers"])
        raise ValueError(f"Phase 1 trace evidence is unavailable for declared modes: {modes}; see {output}")
    return record
