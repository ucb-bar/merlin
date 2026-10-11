"""One frozen-input admission and child command for official Phase-1 grading."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from merlin.targetgen.sandbox import bwrap
from merlin.targetgen.target_experiment import load_target_experiment

from . import run_inputs
from .context import context_argv
from .session import task_scope

if TYPE_CHECKING:
    from .session import PreparedRun


def run(
    prepared: PreparedRun,
    *,
    public_capsules: Path,
    selected_rtl_facts: Path | None,
) -> int:
    """Invoke the existing formal grader after rechecking this run's owned inputs.

    Both the authoring continuation and a fresh submission qualification use
    this exact command, child supervision and immutable private input record.
    The child, not this wrapper, decides public/hidden/private numerical gates.
    """
    request = prepared.request
    options, context = request.options, request.context
    workspace, run_dir = prepared.workspace, prepared.run_dir
    target = load_target_experiment(context.descriptor)
    if options.sandbox == "bwrap":
        bwrap.verify_bundle_snapshot(workspace, prepared.bundle, repo=context.repo)
        snapshot_root = bwrap.bundle_snapshot_root(workspace).resolve(strict=True)
        expected_hidden = run_inputs.hidden_snapshot_dir(snapshot_root, target, context.repo)
    else:
        expected_hidden = None
    hidden = run_inputs.verify_persisted_run_inputs(
        prepared.environment,
        identity={
            "run_id": options.run_id,
            "arm": options.arm,
            "sandbox": options.sandbox,
            "bundle_id": prepared.bundle["bundle_id"],
            "condition": prepared.bundle.get("condition", "legacy"),
        },
        task_scope=task_scope(target, options.sandbox, repo=context.repo, **prepared.scope_roots),
        ws=workspace,
        run_dir=run_dir,
        bundle_dir=prepared.bundle_dir,
        resolved_tools=request.resolved_tools(),
        expected_hidden_dir=expected_hidden,
    )
    prepared.verify_inputs()
    command = [
        sys.executable,
        "-m",
        "merlin_experiments.phase1.feedback.formal",
        *context_argv(context),
        "--contract",
        str(prepared.contract_root if prepared.contract_root is not None else context.repo / "merlin/contract"),
        "--run-dir",
        str(run_dir),
        "--arm",
        options.arm,
        "--model",
        options.model,
        "--qa-timeout",
        str(options.qa_timeout),
        "--capsules",
        str(public_capsules),
    ]
    if hidden is None:
        capsule = Path(target.capsule_corpus)
        hidden = (capsule if capsule.is_absolute() else context.repo / capsule).parent / "hidden"
    if hidden.is_dir():
        command += ["--hidden-capsules", str(hidden)]
    if options.no_oracle:
        command.append("--no-oracle")
    if prepared.private_full_model_spec is not None:
        command += ["--private-full-model-spec", str(prepared.private_full_model_spec)]
    instruction_selection = getattr(prepared, "instruction_selection", None)
    if instruction_selection is not None:
        command += ["--instruction-selection", str(instruction_selection)]
    if options.skip_hidden:
        command.append("--skip-hidden")
    if selected_rtl_facts is not None:
        command += ["--workspace", str(workspace), "--rtl-facts", str(selected_rtl_facts)]
    from merlin_experiments.frozen_python import inherited_python_command

    result = subprocess.run(inherited_python_command(command), cwd=str(context.repo))
    prepared.verify_inputs()
    return result.returncode
