"""Stage selected session guidance before freezing its provenance and tool view."""

from __future__ import annotations


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
        from merlin.targetgen.public_codegen_budget import policy_from_profile, require_public_members

        from .resource_prompt import append_resource_guidance

        resource_policy = policy_from_profile(bundle.get("compiler_tool_profile") or {})
        if resource_policy is not None:
            from merlin.targetgen.capsule_common import discover_capsules

            public_members = discover_capsules(
                [_policy_root] if _policy_root is not None else _te().graded_roots(),
                labels={"public", "dev"},
                contract=_contract_root or context.repo / "src/merlin/resources/contracts",
            )
            excluded = set(_te().effective_exclusions(row.get("name") for row in public_members))
            require_public_members(
                resource_policy, (row.get("name") for row in public_members if row.get("name") not in excluded)
            )
        if _author_profile is not None and not _author_profile.fresh10:
            from .profile_guidance import append_profile_guidance

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
        from .capability_inventory import stage_selected as stage_capabilities

        capability_append = (
            stage_capabilities
            if _author_profile is None or _author_profile.modern or _author_profile.fresh10
            else (lambda *args, **kwargs: None)
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
        from .construction_study import stage as stage_study
        from .feedback.reference_experiments import stage_guidance
        from .feedback.reference_map import stage as stage_reference_map

        stage_reference_map(ws, run_dir)  # the same pinned map in every arm, before the selection's section
        stage_guidance(ws, run_dir)
        stage_study(ws, run_dir, a)
        if _author_profile is not None and _author_profile.fresh10:
            from . import fresh10_disclosure

            fresh10_disclosure.stage_task(ws, run_dir, _author_profile.name, a)
        _skills_record = skill_configuration(_run_config) or {}
        if (context_selection(a) or {}).get("task_skills"):
            from .task_disclosure import stage as stage_disclosure

            # after every TASK.md append, before the treatment seal; staged author skills carry the
            # feedback-file guide, so the generated one is left out
            stage_disclosure(ws, run_dir, feedback=not _skills_record.get("enabled"))
        if _skills_record.get("enabled"):
            stage_skills(ws, run_dir, _skills_record)  # beside the task-* skills, before the treatment seal
