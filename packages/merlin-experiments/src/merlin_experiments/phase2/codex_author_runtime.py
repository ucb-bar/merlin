"""Bind an explicitly selected client to the existing Phase 2 outer policy.

This is runtime transport, not authoring, target or numerical qualification.
The original client owner retains executable/companion pins and credential
isolation; the original AW policy retains every mount and answer mask.
"""

import time
from contextlib import ExitStack
from pathlib import Path
from contextlib import ExitStack

from merlin.targetgen.sandbox import bwrap as BW
from merlin_experiments.phase1 import options
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase1.providers import codex_runtime as CR

from . import agent_workspace as AW
from . import campaign as PC
from . import normal_resource_launch as NR
from . import stage_prompt as SP
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


def queued_author_stop(workflow, broker, deadline):
    """Original live stopping evidence, independent of the author's prose."""
    if time.monotonic() >= deadline:
        return "author wall budget exhausted"
    if len(broker.calls) >= broker.max_calls:
        return "author broker tool-call budget exhausted"
    coordinator = getattr(workflow, "search_coordinator", None)
    if coordinator is None:
        raise StageGateError("continuous queued author lost its original search coordinator")
    status = coordinator.status()
    if type(status) is not dict or "stop_reason" not in status:
        raise StageGateError("continuous queued author lacks original stopping evidence")
    reason = status["stop_reason"]
    if reason is not None and (type(reason) is not str or not reason.strip()):
        raise StageGateError("continuous queued author has malformed stopping evidence")
    return reason


def dispatch_author_round(
    *, driver, workspace, stage_root, prompt, target_experiment, agent_inputs,
    frozen_functional, functional_base, frozen_corpus_manifest, control_dir,
    sandbox_inputs, model, resolved_model, effort, round_index, timeout_s,
    codex_binary, max_concurrent_subagents=None, external_boundary=None,
    continue_session=False, continuation_stop=None, session_control=None,
):
    """Run the selected original driver inside the original outer policy lifetime.

    ``authoring._codex_round`` retains the public typed signature and driver
    injection seam. No runtime, boundary, scratch or callback owner changes.
    """
    if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
        raise StageGateError("Codex authoring requires an explicit sandbox input selection")
    from .search_session import driver_options
    if session_control is not None and continue_session:
        raise StageGateError("select exactly one author continuation owner")
    continuation = driver_options(session_control, stage_root=stage_root, round_index=round_index) if session_control else {}
    with NR.selected_boundary(external_boundary, writable_roots=(stage_root, workspace)), ExitStack() as scratch:
        runtime, exact_bwrap, captured = prepare_dispatch(
            binary=codex_binary,
            workspace=workspace,
            stage_root=stage_root,
            agent_inputs=agent_inputs,
            target_experiment=target_experiment,
            frozen_functional=frozen_functional,
            functional_base=functional_base,
            control_dir=control_dir,
            frozen_corpus_manifest=frozen_corpus_manifest,
            sandbox_inputs=sandbox_inputs,
            scratch=scratch,
            external_boundary=external_boundary,
        )
        rc, transcript = driver.run_round(
            workspace,
            stage_root,
            model,
            {},
            target_experiment,
            "bwrap",
            round_index,
            timeout_s,
            effort=effort,
            prompt=prompt.text,
            effective_model=resolved_model,
            sandbox_command=exact_bwrap,
            **(
                {"continue_session": True, "continuation_stop": continuation_stop,
                 "continuation_prompt": SP.QUEUED_AUTHOR_CONTINUATION}
                if continue_session else {}
            ),
            **runtime.round_kwargs(),
            **continuation,
            **({"max_concurrent_subagents": max_concurrent_subagents} if max_concurrent_subagents is not None else {}),
        )
    policy = captured.get("policy")
    if policy is None:
        raise StageGateError("Codex driver did not construct the required outer bwrap policy")
    return rc, transcript, policy


def run_author_round(
    workspace, stage_root, prompt, target_experiment, agent_inputs, frozen_functional,
    functional_base, frozen_corpus_manifest, control_dir, *, sandbox_inputs, model,
    resolved_model, effort, round_index, timeout_s, codex_binary,
    max_concurrent_subagents=None, external_boundary=None, driver=CA, session_control=None,
):
    """Retain the positional transport API with one shared dispatch owner."""
    arguments = dict(locals())
    return dispatch_author_round(**arguments)
