"""Bind an explicitly selected client to the existing Phase 2 outer policy.

This is runtime transport, not authoring, target or numerical qualification.
The original client owner retains executable/companion pins and credential
isolation; the original AW policy retains every mount and answer mask.
"""

from pathlib import Path

from merlin.targetgen.sandbox import bwrap as BW
from merlin_experiments.phase1 import options
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase1.providers import codex_runtime as CR

from . import agent_workspace as AW
from . import normal_resource_launch as NR
from .contracts import StageGateError


def _read_paths(policy, *, workspace, selection):
    if (
        type(policy) is not AW.AgentSandboxPolicy
        or policy.network != "available_not_an_isolation_claim"
        or not policy.clear_environment
        or policy.answer_surface_gap
    ):
        raise StageGateError("Codex runtime needs the original answer-closed outer policy")
    argv = list(policy.argv)
    readonly = {}
    for index, flag in enumerate(argv[:-2]):
        if flag in {"--ro-bind", "--ro-bind-try", "--bind", "--bind-try"}:
            readonly[argv[index + 2]] = flag in {"--ro-bind", "--ro-bind-try"}
    mounts = BW._mounts(argv)
    home, auth, workspace = Path(selection["home_root"]), Path(selection["auth_source"]), Path(workspace)
    paths = []
    for state, _source, value in mounts:
        if state != "expose" or not readonly.get(value) or not BW.is_exposed(argv, Path(value)):
            continue
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts or path == Path("/"):
            raise StageGateError("Codex read grants cannot use an unbounded host namespace")
        if CR.overlap(path, home) or CR.overlap(path, auth):
            raise StageGateError("Codex read grants overlap private client control state")
        if CR.overlap(path, workspace):
            continue  # The original native workspace rule supplies its writable boundary.
        paths.append(str(path))
    paths.extend(selection[key] for key in ("binary", "helper_path", "code_mode_host") if key in selection)
    return tuple(dict.fromkeys(paths))


def select_runtime(*, binary, home_root, workspace, policy):
    """Select original live client inputs using the actual outer mount policy."""
    auth = (CA.real_codex_home() / "auth.json").resolve(strict=True)
    chosen = options.parse_options(
        [
            "--run-id",
            "phase2-client-runtime",
            "--driver",
            "codex",
            "--sandbox",
            "bwrap",
            "--codex-binary",
            str(binary),
            "--codex-auth-source",
            str(auth),
            "--codex-home-root",
            str(home_root),
        ],
        environ={},
    )
    selection = CR.selection_record(chosen)
    paths = _read_paths(policy, workspace=workspace, selection=selection)
    # Phase 2's original AW callback owns toolchain mounts/environment. These
    # fields are deliberately empty: this owner cannot substitute for the
    # independently checked Phase 1 execution.codex_sandbox_command contract.
    runtime = CR.SelectedCodexRuntime(tuple(sorted(selection.items())), paths, (), (), "")
    runtime.verify()
    return runtime


def verify_policy(runtime, *, workspace, policy):
    if type(runtime) is not CR.SelectedCodexRuntime:
        raise StageGateError("Codex authoring requires the original live client runtime")
    runtime.verify()
    if runtime.read_paths != _read_paths(policy, workspace=workspace, selection=dict(runtime.selection)):
        raise StageGateError("Codex author read grants changed with the outer policy")


def prepare_dispatch(
    *,
    binary,
    workspace,
    stage_root,
    agent_inputs,
    target_experiment,
    frozen_functional,
    functional_base,
    control_dir,
    frozen_corpus_manifest,
    sandbox_inputs,
    scratch,
    external_boundary=None,
):
    """Share the actual original AW callback with unpaid runtime preflight."""
    captured = {}

    def public_policy(ws):
        return AW.outer_codex_policy(
            ws,
            agent_inputs,
            (),
            target_experiment,
            frozen_functional,
            functional_base,
            control_dir,
            frozen_corpus_manifest,
            inputs=sandbox_inputs,
        )

    runtime = select_runtime(
        binary=binary,
        home_root=stage_root / "codex_homes",
        workspace=workspace,
        policy=public_policy(workspace),
    )

    def exact_bwrap(inner, ws, _bundle, extra_binds=None):
        verify_policy(runtime, workspace=ws, policy=public_policy(ws))
        policy = AW.outer_codex_policy(
            ws,
            agent_inputs,
            extra_binds or (),
            target_experiment,
            frozen_functional,
            functional_base,
            control_dir,
            frozen_corpus_manifest,
            inputs=sandbox_inputs,
        )
        captured["policy"] = policy
        actual = scratch.enter_context(
            NR.bounded_normal_scratch(policy.argv, owner=stage_root / "author_scratch", boundary=external_boundary)
        )
        return BW.compose_command(list(actual), " bash -c '" + inner.replace("'", "'\\''") + "'", ws)

    return runtime, exact_bwrap, captured
