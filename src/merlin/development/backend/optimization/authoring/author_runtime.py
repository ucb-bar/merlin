"""Bind an explicitly selected client to the existing Phase 2 outer policy.

This is runtime transport, not authoring, target or numerical qualification.
The original client owner retains executable/companion pins and credential
isolation; the original AW policy retains every mount and answer mask.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from merlin.common.digest import hash_tree
from merlin.development.backend import options
from merlin.development.backend.optimization.author_interface import agent_workspace as AW
from merlin.development.backend.optimization.author_interface import stage_prompt as SP
from merlin.development.backend.optimization.author_interface.prompt_inputs import ADAPTIVE_SCOPE
from merlin.development.backend.optimization.authoring.transcript_audit import audit_codex_transcript
from merlin.development.backend.optimization.campaign import functional_boundary as PC
from merlin.development.backend.optimization.campaign import normal_resource_launch as NR
from merlin.development.backend.optimization.contracts import StageGateError
from merlin.development.backend.providers import codex_launch_codex_agent as CA
from merlin.development.backend.providers import codex_runtime as CR
from merlin.development.targetgen.sandbox import grants as BW

# ---- codex_author_runtime ----------------------------------------------------------------------


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
    *,
    driver,
    workspace,
    stage_root,
    prompt,
    target_experiment,
    agent_inputs,
    frozen_functional,
    functional_base,
    frozen_corpus_manifest,
    control_dir,
    sandbox_inputs,
    model,
    resolved_model,
    effort,
    round_index,
    timeout_s,
    codex_binary,
    max_concurrent_subagents=None,
    external_boundary=None,
    continue_session=False,
    continuation_stop=None,
    session_control=None,
):
    """Run the selected original driver inside the original outer policy lifetime.

    ``authoring._codex_round`` retains the public typed signature and driver
    injection seam. No runtime, boundary, scratch or callback owner changes.
    """
    if not isinstance(sandbox_inputs, (PC.PackageSandboxInputs, PC.FrozenPackageSandboxInputs)):
        raise StageGateError("Codex authoring requires an explicit sandbox input selection")

    if session_control is not None and continue_session:
        raise StageGateError("select exactly one author continuation owner")
    continuation = (
        driver_options(session_control, stage_root=stage_root, round_index=round_index) if session_control else {}
    )
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
                {
                    "continue_session": True,
                    "continuation_stop": continuation_stop,
                    "continuation_prompt": SP.QUEUED_AUTHOR_CONTINUATION,
                }
                if continue_session
                else {}
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
    workspace,
    stage_root,
    prompt,
    target_experiment,
    agent_inputs,
    frozen_functional,
    functional_base,
    frozen_corpus_manifest,
    control_dir,
    *,
    sandbox_inputs,
    model,
    resolved_model,
    effort,
    round_index,
    timeout_s,
    codex_binary,
    max_concurrent_subagents=None,
    external_boundary=None,
    driver=CA,
    session_control=None,
):
    """Retain the positional transport API with one shared dispatch owner."""
    arguments = dict(locals())
    return dispatch_author_round(**arguments)


# ---- search_session ----------------------------------------------------------------------------
# Continue the original author thread until automatic search finalization.
#
# The existing broker/coordinator owns evaluation, source selection and receipts.
# This transport control issues no grade or performance qualification.

CONTINUATION = (
    "Continue this Phase2 campaign in the same workspace and thread. Read search-status. "
    "While exploration remains open, implement and measure the next hypothesis; assign helpers "
    "and focused measurements in roofline_priorities.next order, lowest worst-repeat attainment "
    "first. Keep unknown bounds separate and preserve numerical gates. A completed turn is not "
    "campaign completion. Finish one hypothesis, frozen-submission or feedback checkpoint, await "
    "your native tool calls, then emit a concise FINAL turn message. The driver automatically "
    "resumes this same thread; returning a turn does not finalize the search. Read the fixed "
    "search_session deadlines in STAGE_CONTEXT.json and leave time to return before the author cutoff. "
    + ADAPTIVE_SCOPE
    + "When the declared target or a budget boundary stops exploration, join "
    "helpers, stop edits, invoke search-finalize, then search-select when a complete measured "
    "selection exists. Run all original required feedback and analysis on those exact bytes. "
    "Retain diagnostic-only sources and explicit incomplete results when no measured selection "
    "exists. Do not finalize early or ask an operator to choose each iteration."
)


def timing_projection(settings):
    """Describe the fixed automatic campaign cutoffs before author dispatch."""
    if not settings.continue_author_session:
        return None
    settings.verify()
    now, wall = time.monotonic(), time.time()
    hard, exploration = search_deadlines(settings, now)
    cutoffs = {
        "exploration": exploration,
        "author_return": hard - settings.final_reserve_seconds,
        "campaign": hard,
    }
    return {
        "deadlines": {
            name: {
                "unix": wall + cutoff - now,
                "utc": datetime.fromtimestamp(wall + cutoff - now, UTC).isoformat(),
                "remaining_seconds": max(0, int(cutoff - now)),
            }
            for name, cutoff in cutoffs.items()
        },
        "turn_policy": "Finish one hypothesis/submission/feedback checkpoint, await native tool calls, "
        "then emit a FINAL turn message. The driver automatically resumes the same thread. "
        "Invoke search-finalize only at the campaign target or exploration boundary; "
        "finish required final feedback and return before author_return.",
    }


def stage_deadline(settings, wall_seconds):
    """Clip the local stage to the launcher's immutable absolute campaign cap."""
    now = time.monotonic()
    limit = now + wall_seconds
    external = None if settings is None else settings.campaign_deadline_unix
    return min(limit, now + max(0, external - time.time())) if external is not None else limit


def require_stage_budget(settings, wall_seconds, round_seconds):
    required = settings.hard_seconds if settings.continue_author_session else settings.soft_seconds
    if wall_seconds < required or round_seconds < required:
        raise StageGateError("queued corpus search must retain its declared exploration and final reserve")


def search_deadlines(settings, started):
    hard = started + settings.hard_seconds
    if settings.campaign_deadline_unix is not None:
        hard = min(hard, stage_deadline(settings, settings.hard_seconds))
    reserve = settings.final_reserve_seconds
    soft = started + settings.soft_seconds - reserve
    if settings.continue_author_session:
        # One reserve for author-owned drain/selection, one for post-author checks.
        soft = min(soft, hard - 2 * reserve)
    return hard, min(soft, hard - reserve)


def driver_options(broker, *, stage_root, round_index):
    """Bind continuation to the actual original live broker, never saved status."""
    from merlin.development.backend.optimization.author_interface.broker_service import Broker
    from merlin.development.backend.optimization.search.search_coordinator import ComponentSearchCoordinator

    if type(broker) is not Broker or type(broker.workflow.search_coordinator) is not ComponentSearchCoordinator:
        raise StageGateError("automatic continuation requires the original search broker and coordinator")
    coordinator = broker.workflow.search_coordinator
    if not coordinator.settings.continue_author_session:
        raise StageGateError("automatic continuation was not selected in the frozen search settings")
    transcript = Path(stage_root) / "rounds" / f"round_{round_index:02d}.transcript.jsonl"

    def stop():
        status = coordinator.status()
        if not status["finalized"]:
            return None  # A target/budget boundary still needs an author finalization turn.
        if status["selected"] is None:
            if status["ranked"]:
                return None  # Resume to select within the original observed author lineage.
            return "search finalized without complete measured selection; campaign incomplete"
        from merlin.development.backend.optimization.search.retained_revisions import source_inventory

        store = coordinator.pool.revisions
        selected = store.get(status["selected"]["revision_id"])

        def selected_bytes():
            return (
                source_inventory(
                    broker.candidate,
                    max_bytes=store.max_bytes,
                    max_files=store.max_files,
                    exclude_generated=True,
                )
                == selected.inventory
            )

        if not selected_bytes():
            return None  # Resume to restore the measured selection before required feedback.
        actions = tuple(broker.actions.values())
        audit = audit_codex_transcript(transcript, broker.target_experiment, broker.candidate, actions)
        if not audit["clean"]:
            return "search author audit refused; campaign incomplete"
        try:
            evidence = broker.workflow.verify_receipts(
                broker.receipt_path,
                actions,
                audit,
                candidate_sha256=hash_tree(broker.candidate)["sha256"],
            )
        except StageGateError:
            return None  # Let the same author finish the original required feedback.
        return (
            "automatic search finalization complete"
            if evidence.get("all_required_succeeded") and selected_bytes()
            else None
        )

    return dict(continue_session=True, continuation_prompt=CONTINUATION, continuation_stop=stop)
