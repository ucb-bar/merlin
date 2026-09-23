"""Functional-run admission and authoring with explicit operator inputs.

Native launchers select legacy defaults at their edge. This owner does not import
them, discover a checkout, provision workers or invent target resources.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .context import InvocationContext
from .options import RunOptions
from .workspaces import workspace_session

if TYPE_CHECKING:
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

    from . import authoring, runtime_environment, session, task_staging, treatments

    refusal = session.validate_options(options)
    if refusal is not None:
        return refusal
    from merlin.targetgen import tool_registry

    for name in (*options.with_tool, *options.without_tool):
        tool_registry.spec(name)
    callbacks = task_staging.callbacks(
        task_staging.TaskStagingConfig(
            context,
            bundle_id,
            bundle_manifest.parent,
            options.experiment,
            language=language,
            add_tools=tuple(options.with_tool),
            drop_tools=tuple(options.without_tool),
        )
    )
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

    with runtime_environment.applied_environment(runtime):
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
        )
        prepared = session.prepare(
            request,
            session.WorkspaceTransport(workspace_transport.assemble, workspace_transport.probe),
            callbacks.stage_task,
            workspace_leases=_workspace_leases,
        )
        if isinstance(prepared, int):
            return prepared
        return authoring.execute(prepared, authoring.AuthoringRuntime(bundle_id, oracle_timing, public_root))
