"""Functional-run admission and authoring with explicit operator inputs.

Native launchers select legacy defaults at their edge. This owner does not import
them, discover a checkout, provision workers or invent target resources.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from .context import InvocationContext
from .options import RunOptions
from .workspaces import workspace_session

if TYPE_CHECKING:
    from merlin.targetgen.compiler_library import CompilerLibrarySelection

    from . import treatments


@workspace_session
def run(
    context: InvocationContext,
    options: RunOptions,
    *,
    bundle_manifest: Path,
    bundle_id: str,
    oracle_timing: Path,
    launcher_argv: tuple[str, ...] = (),
    language: str = "",
    treatment: treatments.Treatment | None = None,
    public_root: Path | None = None,
    machine_defaults: Mapping[str, str] | None = None,
    library_paths: tuple[Path, ...] = (),
    base_environment: Mapping[str, str] | None = None,
    source_entrypoint: Path | None = None,
    require_native_source: bool = False,
    compiler_library_selection: CompilerLibrarySelection | None = None,
    require_shared_passes: bool = False,
    shared_pass_contract_version: int = 1,
    required_supported_shared_passes: tuple[str, ...] = (),
    _workspace_leases: list,
) -> int:
    """Keep one workspace lease and environment across real admission and execution.

    The optional native source binding preserves the legacy launcher's recorded
    identity; installed callers use this module's inventoried source. Operator
    paths, treatment and environment are trusted host inputs, never candidate data.
    Initialize context with load_context before calling; environment-sensitive
    execution owners are imported only here. Concurrent experiments must use
    separate processes.
    """
    if source_entrypoint is not None and not require_native_source:
        raise ValueError("a source entrypoint override requires native source verification")
    if type(require_shared_passes) is not bool:
        raise ValueError("shared-pass authoring requires an explicit Boolean selection")
    if type(shared_pass_contract_version) is not int or shared_pass_contract_version not in (1, 2):
        raise ValueError("shared-pass contract version must be explicitly 1 or 2")
    if not require_shared_passes and (shared_pass_contract_version != 1 or required_supported_shared_passes):
        raise ValueError("shared-pass version/required families need explicit shared-pass authoring")
    requirement = None
    requirement_bytes = None
    if require_shared_passes:
        if treatment is not None and treatment.stage_task is not None:
            raise ValueError("shared-pass authoring does not support a custom treatment task stager")
        from merlin.targetgen.generate_prompt import shared_pass_prompt_record
        from merlin.targetgen.shared_pass_contract import shared_pass_requirement

        requirement = shared_pass_requirement(
            version=shared_pass_contract_version, required_supported=required_supported_shared_passes
        )
        requirement_bytes = json.dumps(requirement, sort_keys=True)
        modules = () if compiler_library_selection is None else compiler_library_selection.contract.public_modules
        shared_pass_prompt_record(requirement, granted_tools=modules, library=compiler_library_selection)
    if options.readback_policy:
        if context.readback_policy is not None and context.readback_policy != options.readback_policy:
            raise ValueError("context and invocation readback policies differ")
        context = replace(context, readback_policy=options.readback_policy)

    from . import authoring, compiler_library, runtime_environment, session, task_staging, treatments

    refusal = session.validate_options(options)
    if refusal is not None:
        return refusal
    # --no-oracle is an explicit diagnostic bypass, never a verified oracle claim.
    if not options.no_oracle:
        from .timing import read_verified_timing, requires_chipyard_timing

        try:
            if requires_chipyard_timing(context.descriptor):
                read_verified_timing(oracle_timing, descriptor=context.descriptor, target=context.target)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 3
    from merlin.targetgen import tool_registry

    for name in (*options.with_tool, *options.without_tool):
        tool_registry.spec(name)
    task_config = task_staging.TaskStagingConfig(
        context,
        bundle_id,
        bundle_manifest.parent,
        options.experiment,
        language=language,
        author_profile=options.author_profile,
        add_tools=tuple(options.with_tool),
        drop_tools=tuple(options.without_tool),
        shared_pass_requirement=requirement,
        shared_pass_library=compiler_library_selection,
    )
    callbacks = task_staging.callbacks(task_config)
    if options.with_tool or options.without_tool:
        suffix = tool_registry.cell_suffix(tuple(options.with_tool), tuple(options.without_tool))
        print(f"ABLATION CELL: {options.arm}{suffix} -> tools {list(callbacks.resolved_tools())}")
    runtime = runtime_environment.prepare_runtime_environment(
        options,
        dict(os.environ) if base_environment is None else base_environment,
        machine_defaults={} if machine_defaults is None else machine_defaults,
        library_paths=library_paths,
    )
    if runtime.refusal is not None:
        print(f"ERROR: {runtime.message}", file=sys.stderr)
        return runtime.refusal
    from . import workspace_transport
    from .feedback.reference_experiments import invocation

    reference_scope = invocation(
        options.reference_experiments or None,
        options.reference_experiments_sha256 or None,
        tasks=options.reference_experiments_tasks != "off",
    )
    with runtime_environment.applied_environment(runtime), reference_scope:
        request = session.RunRequest(
            context=context,
            options=options,
            treatment=treatments.Treatment() if treatment is None else treatment,
            bundle_manifest=bundle_manifest,
            launcher_argv=launcher_argv,
            source_entrypoint=Path(__file__) if source_entrypoint is None else source_entrypoint,
            require_native_source=require_native_source,
            account=dict(runtime.account),
            resolved_tools=callbacks.resolved_tools,
            compiler_library_selection=compiler_library_selection,
            task_staging_config=(task_config if shared_pass_contract_version == 2 else None),
        )
        prepared = session.prepare(
            request,
            session.WorkspaceTransport(workspace_transport.assemble, workspace_transport.probe),
            callbacks.stage_task,
            workspace_leases=_workspace_leases,
        )
        if isinstance(prepared, int):
            return prepared
        if requirement is None:
            directory = getattr(prepared, "run_dir", None)
            if directory is not None:
                task_staging.verify_shared_pass_prompt(task_config, prepared.workspace, directory)
            with compiler_library.execution(prepared):
                return _execute_prepared(prepared, options, authoring, bundle_id, oracle_timing, public_root)
        if json.dumps(requirement, sort_keys=True) != requirement_bytes:
            raise ValueError("original shared-pass requirement changed during admission")
        active_requirement = task_staging.verify_shared_pass_prompt(task_config, prepared.workspace, prepared.run_dir)
        from merlin.targetgen.shared_pass_context import scoped_shared_pass_requirement

        with compiler_library.execution(prepared), scoped_shared_pass_requirement(active_requirement):
            try:
                return _execute_prepared(prepared, options, authoring, bundle_id, oracle_timing, public_root)
            finally:
                if json.dumps(requirement, sort_keys=True) != requirement_bytes:
                    raise ValueError("original shared-pass requirement changed during execution")
                task_staging.verify_shared_pass_prompt(task_config, prepared.workspace, prepared.run_dir)


def _execute_prepared(prepared, options, authoring, bundle_id, oracle_timing, public_root):
    if options.preflight_only:
        from . import preflight

        return preflight.complete(prepared)
    if options.codex_canary:
        from . import canary

        return canary.execute(prepared)
    if getattr(options, "offline_actions", ""):
        from . import construction_study

        return construction_study.execute(prepared, timing_file=oracle_timing)
    if options.qualify_submission:
        from . import qualification

        return qualification.execute(prepared)
    return authoring.execute(prepared, authoring.AuthoringRuntime(bundle_id, oracle_timing, public_root))
