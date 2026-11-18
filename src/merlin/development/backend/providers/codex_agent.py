"""Capsule-bench agent driver for the **Codex CLI** (``codex exec --json``).

Same contract as :mod:`opencode_agent` and :mod:`bedrock_agent`: ``run_round``
drives ONE round inside the prepared workspace and returns
``(rc, transcript_path)``, writing a transcript in the Claude-stream JSONL shape
the rest of the harness already reads (trajectory, token accounting, transcript
audit). Nothing downstream needs to learn a second schema.

Four things here are deliberate and easy to get wrong.

**1. Token subsets.** Codex reports ``input_tokens`` with ``cached_input_tokens``
and ``cache_write_input_tokens`` already *inside* it, and
``reasoning_output_tokens`` already inside ``output_tokens``. The Claude stream
shape this harness consumes means the opposite by ``input_tokens`` — it is the
*uncached* part, with cache reads counted separately. So the mapping subtracts
rather than copies; copying would inflate input by the cache-hit rate, which on
a warm run is most of the prompt.

**2. Usage only arrives on ``turn.completed``.** A ``turn.failed`` carries no
usage at all, so its tokens are *unknown*, not zero. The transcript records the
turn with usage omitted and a ``codex_usage_unreported`` marker, so a spend
figure derived from it is visibly a lower bound instead of a confident total.

**3. The events carry no timestamps.** Every time in the transcript is this
reader's arrival time, recorded as the line arrives. Capture therefore tees line
by line to ``codex_events.raw.jsonl`` before interpreting anything, so a timeout
or a kill still leaves the evidence (and the token counts) on disk.

**4. Instruction parity between arms.** Codex reads ``AGENTS.md`` from the
workspace; Claude Code reads ``CLAUDE.md``/``AGENT.md``. An arm that silently
gets extra instructions is not the same arm. This driver does not author either
file — it records which instruction files the prepared workspace actually
contains into the transcript's init record, so an asymmetry is visible in the
artifact rather than discovered later.

Sandboxing: at ``--sandbox bwrap`` the whole ``codex`` process runs inside the
harness's existing bwrap wrapper (which masks goldens and hidden inputs), while
model-generated commands run inside a second, deny-by-default Codex permission
profile. The second boundary keeps the writable credential available to the
Codex parent for token refresh, but unavailable to its candidate commands.
At ``--sandbox none`` Codex's own ``workspace-write`` sandbox is used.
"""

from __future__ import annotations

import json
import os
import selectors
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from merlin.common.paths import module_source_path
from merlin.development.backend.providers import codex_capacity as _CAP  # noqa: E402
from merlin.development.backend.providers import codex_context as _CTX
from merlin.development.backend.providers.authoring_prompt import (
    authoring_prompt,
    profile_authoring_prompt,
)
from merlin.development.backend.providers.codex_context import delegation_user_prompt as delegation_user_prompt
from merlin.development.backend.providers.codex_runtime import (
    _kill_tree,
    _process_group_quiescent,
    _terminate_process_group,
)
from merlin.development.backend.providers.codex_transcript import (
    _INSTRUCTION_FILES as _INSTRUCTION_FILES,
)
from merlin.development.backend.providers.codex_transcript import (
    _TOOL_OUTPUT_CAP as _TOOL_OUTPUT_CAP,
)
from merlin.development.backend.providers.codex_transcript import (
    ITEM_COMMAND_EXECUTION as ITEM_COMMAND_EXECUTION,
)
from merlin.development.backend.providers.codex_transcript import (
    ITEM_FILE_CHANGE as ITEM_FILE_CHANGE,
)
from merlin.development.backend.providers.codex_transcript import (
    _clip_tool_output as _clip_tool_output,
)
from merlin.development.backend.providers.codex_transcript import (
    _error_text as _error_text,
)
from merlin.development.backend.providers.codex_transcript import (
    _instruction_files as _instruction_files,
)
from merlin.development.backend.providers.codex_transcript import (
    _tool_block as _tool_block,
)
from merlin.development.backend.providers.codex_transcript import (
    _Transcript as _Transcript,
)
from merlin.development.backend.providers.codex_transcript import (
    last_message_path as last_message_path,
)
from merlin.development.backend.providers.codex_usage import usage_to_claude_shape as usage_to_claude_shape
from merlin.development.backend.providers.execution import sandbox_script as _sandbox_script
from merlin.development.backend.telemetry import author_agents as _START

# ---- codex_agent -------------------------------------------------------------------------------


@_START.observed
def run_round(
    ws: Path,
    run_dir: Path,
    model: str,
    bundle: dict,
    te,
    sandbox: str,
    rnd: int,
    timeout: int,
    *,
    subagent_model: str = "",
    background_model: str = "",
    effort: str = "",
    prompt: str | None = None,
    continue_session: bool = False,
    effective_model: str | None = None,
    sandbox_command: Callable[..., str] | None = None,
    codex_binary: str | Path | None = None,
    codex_home_root: Path | None = None,
    candidate_read_paths: tuple[str, ...] | None = None,
    runtime_binds: Callable[[Path], list[str]] | None = None,
    require_fresh_home: bool = False,
    allow_failed_home_retry: bool = False,
    auth_source: Path | None = None,
    continuation_prompt: str | None = None,
    continuation_stop: Callable[[], str | None] | None = None,
    max_concurrent_subagents: int | None = None,
    subagent_capacity: int | None = None,
    helper_effort: str | None = None,
    author_context: dict | None = None,
    conformance_context: dict | None = None,
    **_ignored,
) -> tuple[int, Path]:
    """Drive ONE capsule-bench round via ``codex exec``. Returns ``(rc, transcript_path)``.

    ``codex_binary`` selects the executable without changing ``CODEX_BIN``.
    ``codex_home_root`` selects the parent of the isolated per-round home for
    bwrap execution. Omitted values retain the environment and cache defaults;
    both selections remain fixed across this round's session continuations.

    ``continue_session`` makes ONE session span the whole ``timeout`` budget. It exists because
    ``codex exec`` is one TURN, not one session: it returns as soon as the model stops, which the
    harness reads as "the agent session ended". Measured on atlas arm-4, three runs in a row ended
    that way -- the last after 83 minutes with 23890s of its 28901s budget unspent, having emitted
    exactly one ``turn.started``/``turn.completed`` pair. Continuation re-invokes
    ``codex exec resume <thread_id>``, so the SAME thread (and its accumulated understanding) keeps
    working; ``keep_launching``'s "one invocation per process" contract is preserved, because this is
    still one session -- just delivered over several processes. Off by default: in ROUNDS mode a
    round ending is how the harness re-grades and re-prompts, and continuing would break that cadence.
    ``continuation_stop`` names a trusted host's stop reason. It is polled
    between turns and at most once per second while a turn is running. An
    explicit stop ends only the owned author process group before returning an
    interrupted status to the host's normal drain/checkpoint path. Received
    logs and workspace files are retained; a partial turn earns no success.

    Legacy model-tier selections are recorded without changing spawned models.
    Explicit ``max_concurrent_subagents`` selects the native multi-agent limit
    in the frozen home; the same strict-config preflight applies on AWS. It does
    not prove that an agent delegated or change its sandbox and runtime grants.
    ``subagent_capacity`` is the run's recorded ceiling: the same config keys, no prompt change;
    ``helper_effort`` its recorded helper reasoning effort (one config key and one sentence);
    ``author_context`` its recorded context selection (:mod:`.codex_context`).
    """
    # Explicit round inputs avoid changing another concurrent round's executable
    # or credential-isolated home placement. Defaults retain historical callers.
    from merlin.development.backend.authoring import author_subagents as _AS
    from merlin.development.backend.providers.codex_launch_codex_agent import (
        _CONTINUE_MAX_TURNS,
        _CONTINUE_MIN_S,
        _CONTROL_POLL_S,
        _POLL_S,
        _TOOL_ITEMS,
        BILLING_MODE,
        EVENT_ERROR,
        EVENT_ITEM_COMPLETED,
        EVENT_ITEM_STARTED,
        EVENT_THREAD_STARTED,
        EVENT_TURN_COMPLETED,
        EVENT_TURN_FAILED,
        EVENT_TURN_STARTED,
        ITEM_AGENT_MESSAGE,
        ITEM_ERROR,
        ITEM_REASONING,
        OPERATOR_INTERRUPTED_EXIT_CODE,
        _link_for_aet,
        _now,
        _preflight_candidate_sandbox,
        _verify_frozen_config,
        build_cmd,
        build_resume_cmd,
        cli_version,
        codex_runtime_binds,
        prepare_codex_home,
        resolve_model,
    )

    capacity = _AS.effective_capacity(max_concurrent_subagents, subagent_capacity, sandbox)
    if continuation_stop is not None and (not continue_session or not callable(continuation_stop)):
        raise ValueError("host continuation stop requires an explicitly continuous session and callback")
    if codex_binary is not None and (not isinstance(codex_binary, (str, Path)) or not str(codex_binary).strip()):
        raise ValueError("explicit codex_binary must be a nonempty executable path or name")
    if auth_source is not None:
        from merlin.development.backend.providers.codex_runtime import canonical_file

        canonical_file(str(auth_source), "client credential")
        if (
            sandbox != "bwrap"
            or codex_binary is None
            or codex_home_root is None
            or not candidate_read_paths
            or runtime_binds is None
            or not require_fresh_home
        ):
            raise ValueError("explicit credential selection requires complete isolated client inputs")
    codex_bin = str(codex_binary) if codex_binary is not None else os.environ.get("CODEX_BIN", "codex")
    # A rigorous caller may preflight model resolution and pass the content-addressed result.
    # In that mode do NOT consult CODEX_MODEL_MAP a second time between the declaration and
    # the process launch: a remap landing in between would run a model the run did not
    # declare, and the transcript would name the declared one.
    resolved = str(effective_model).strip() if effective_model is not None else resolve_model(model)
    if not resolved:
        raise ValueError("effective Codex model must be non-empty")
    if sandbox == "bwrap":
        from merlin.development.backend.providers import agent_bridge as _BR

        if _BR.bridged_name(resolved, "codex"):
            # The proxy's master key belongs to the host Codex process, not the
            # candidate shell. A future broker can deliver it without a shell
            # environment leak; clearenv intentionally does not do that today.
            raise RuntimeError("bridged Codex needs a host-side proxy credential broker under bwrap")
    home_attempt = {}
    selected_home = None
    if allow_failed_home_retry and not (require_fresh_home and sandbox == "bwrap" and codex_home_root is not None):
        raise ValueError("failed home retry requires the original isolated fresh-home route")
    if require_fresh_home and sandbox == "bwrap" and codex_home_root is not None:
        selected_home = Path(codex_home_root) / f"{run_dir.name}_r{rnd:02d}"
        if allow_failed_home_retry:
            from merlin.development.backend.providers.codex_runtime import failed_round_retry

            selected_home, home_attempt = failed_round_retry(Path(codex_home_root), run_dir, rnd)
        if not home_attempt and (selected_home.exists() or selected_home.is_symlink()):
            raise ValueError("fresh authoring refuses an existing Codex home or session history")
        if selected_home.resolve().is_relative_to(ws.resolve()) or ws.resolve().is_relative_to(selected_home.resolve()):
            raise ValueError("isolated CODEX_HOME must not overlap the candidate workspace")
    rounds = run_dir / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    artifact_prefix = f"round_{rnd:02d}" + (".attempt_" + home_attempt["attempt_id"] if home_attempt else "")
    script_rounds = rounds / "attempts" / home_attempt["attempt_id"] if home_attempt else rounds
    tpath = rounds / f"{artifact_prefix}.transcript.jsonl"
    raw_path = rounds / f"{artifact_prefix}.codex_events.raw.jsonl"
    stamped_path = rounds / f"{artifact_prefix}.codex_events.timestamped.jsonl"
    stderr_path = rounds / f"{artifact_prefix}.codex_stderr.log"
    prompt_path = rounds / f"{artifact_prefix}.prompt.txt"
    final_path = rounds / f"{artifact_prefix}.final.txt"

    tr = _Transcript(tpath)
    tr.emit(
        {
            "type": "system",
            "subtype": "init",
            "driver": "codex",
            "round": rnd,
            "home_attempt": home_attempt,
            "model": resolved,
            "model_requested": model,
            "codex_bin": codex_bin,
            "cli_version": cli_version(codex_bin),
            "sandbox": sandbox,
            # Instruction parity between arms is an artifact-level fact, not a hope.
            "workspace_instruction_files": _instruction_files(ws),
            "workspace_skills": _CTX.staged_skills(ws),
            "tiering_requested_but_unsupported": bool(subagent_model or background_model),
            "started_at": _now(),
            "conformance_context": conformance_context
            if conformance_context is not None
            else {
                "schema": "merlin.phase1.conformance_execution.v1",
                "workspace": str(ws.absolute()),
                "target": getattr(te, "target", None),
                "selected_facts": None,
            },
        }
    )

    # The graded instruction. ``prompt=`` overrides it only for out-of-band uses
    # (the sandbox canary states its own task); a measured arm always gets this
    # text, so two arms cannot silently differ in what they were asked to do.
    msg = prompt if prompt is not None else authoring_prompt()
    msg = delegation_user_prompt(msg, max_concurrent_subagents)
    msg += _AS.round_guidance(capacity, max_concurrent_subagents) + _AS.helper_effort_note(helper_effort)
    msg += _CTX.round_note(author_context, capacity)
    prompt_path.write_text(msg)

    inner_final = last_message_path(ws, final_path, sandbox)
    inner_final.parent.mkdir(parents=True, exist_ok=True)
    run_cmd = build_cmd(ws, model=resolved, effort=effort, final_path=inner_final, sandbox=sandbox, codex_bin=codex_bin)
    home_info: dict = {}
    if sandbox == "bwrap":
        # The caller owns sandbox policy; a provider must not import a native controller.
        if sandbox_command is None:
            raise ValueError("bwrap provider execution requires the caller's sandbox_command")
        from merlin.common.artifacts import cache_dir

        # An isolated CODEX_HOME per run: the real ~/.codex holds every prior
        # session on this host, which a graded agent must not be able to read.
        # PURGEABLE cache, and no credential is written into it — the real
        # auth.json is bind-mounted writable for refresh, behind Codex's native
        # candidate-command sandbox (see codex_runtime_binds).
        home_root = Path(codex_home_root) if codex_home_root is not None else cache_dir("codex_home")
        codex_home = selected_home or home_root / f"{run_dir.name}_r{rnd:02d}"
        if codex_home.resolve().is_relative_to(ws.resolve()) or ws.resolve().is_relative_to(codex_home.resolve()):
            raise ValueError("isolated CODEX_HOME must not overlap the candidate workspace")
        if home_attempt:
            state = codex_home.stat()
            if (
                codex_home.is_symlink()
                or codex_home.resolve() != codex_home
                or (state.st_dev, state.st_ino) != (home_attempt["home_device"], home_attempt["home_inode"])
                or any(codex_home.iterdir())
            ):
                raise ValueError("fresh retry home changed before preparation")
        elif require_fresh_home and (codex_home.exists() or codex_home.is_symlink()):
            raise ValueError("fresh authoring refuses an existing Codex home or session history")
        home_options = {"candidate_read_paths": candidate_read_paths} if candidate_read_paths is not None else {}
        if auth_source is not None:
            home_options["auth_source"] = auth_source
        if capacity is not None:
            home_options["max_concurrent_subagents"] = capacity
        if helper_effort is not None:  # without a capacity the home refuses it rather than dropping it
            home_options["helper_effort"] = helper_effort
        if author_context is not None:
            home_options["author_context"] = author_context
        home_info = prepare_codex_home(codex_home, model=resolved, effort=effort, workspace=ws, **home_options)
        home_info["attempt"] = home_attempt
        runtime_selector = codex_runtime_binds if runtime_binds is None else runtime_binds
        _START.call("config_verify", _verify_frozen_config, codex_home, home_info["config_sha256"])
        preflight_options = {"runtime_binds": runtime_selector} if runtime_binds is not None else {}
        # Only the trusted composition owner supplies the launch-local plan.
        from merlin.development.backend.providers.execution import CodexSandboxCommand

        prepared_plan = None
        if type(sandbox_command) is CodexSandboxCommand:
            binds = _START.call("runtime_bindings", runtime_selector, codex_home)
            prepared_plan = _START.call("sandbox_assembly", sandbox_command.prepare, ws, bundle, binds)
            preflight_options["prepared_plan"] = prepared_plan
        _preflight_candidate_sandbox(
            ws, codex_home, codex_bin, bundle, sandbox_command, script_rounds, rnd, **preflight_options
        )
        inner = " ".join(shlex.quote(c) for c in run_cmd)
        cmd = _sandbox_script(
            script_rounds,
            rnd,
            0,
            prepared_plan.command(inner)
            if prepared_plan is not None
            else _START.call(
                "sandbox_assembly",
                sandbox_command,
                inner,
                ws,
                bundle,
                extra_binds=_START.call("runtime_bindings", runtime_selector, codex_home),
            ),
        )
    else:
        cmd = run_cmd

    started = time.monotonic()
    deadline = started + max(int(timeout), 1)
    turns_started = turns_reported = 0
    pending_tools: dict[str, dict] = {}
    unknown: list[str] = []
    errors: list[str] = []
    recovered_errors: set[int] = set()
    capacity_retries = capacity_consecutive = 0
    resumed_after_capacity = False
    thread_id = None
    timed_out = False
    operator_interrupted = None
    interrupted_turn = False
    seq = 0

    raw_f = open(raw_path, "wb")
    stamped_f = open(stamped_path, "w")
    #: Sent when resuming. Deliberately the SAME standing instruction, never a hint: an arm that got
    #: extra guidance mid-session would not be comparable to one that did not.
    _CONTINUE_MSG = (
        continuation_prompt
        if continuation_prompt is not None
        else profile_authoring_prompt(continuation=True)
        or (
            "Continue. Re-read qa/verdict.json for the latest grade, then keep repairing the "
            "backend under submission/. Follow TASK.md for the selected engine routing and physical "
            "qualification gates. With agent_selfcheck, check capsules an edit affects in one request with "
            "`--wait-budget 5`, keep working, and collect it with `--attach <id>`; several requests "
            "may be in flight. Check edits by naming capsules (`--capsules changed` is a slow sweep); "
            "`--capsules failing` re-checks "
            "what did not pass. Try alternatives with `--submission variants/<name>` and copy the "
            "winner into submission/. Use `--capsules all` only after focused checks improve. You "
            "may edit while a check runs; compare the verdict's source digest."
        )
    )
    _CONTINUE_MSG = delegation_user_prompt(_CONTINUE_MSG, max_concurrent_subagents)
    _CONTINUE_MSG += _AS.continuation_guidance(capacity, max_concurrent_subagents)
    _CONTINUE_MSG += _CTX.continuation_note(author_context)
    from merlin.development.backend.providers.token_admission import ManagedTokens

    token_control = ManagedTokens(run_dir, rnd, homes=(codex_home,) if sandbox == "bwrap" else ())
    turn_index = 0
    rc = 0
    active_proc: subprocess.Popen | None = None
    active_pgid: int | None = None
    control_refusal = None

    def poll_host_control():
        nonlocal control_refusal, rc
        if control_refusal is not None:
            return control_refusal
        if continuation_stop is None:
            return None
        try:
            reason = continuation_stop()
            if reason is not None and (type(reason) is not str or not reason.strip()):
                raise ValueError("continuation control must return None or a nonempty reason")
            return reason
        except Exception as exc:
            control_refusal = f"continuation control refused ({type(exc).__name__})"
            errors.append(control_refusal)
            rc = 1
            return control_refusal

    try:
        while True:
            turn_errors_start = len(errors)
            turn_failed = turn_completed = turn_progress = False
            _START.turn(turn_index)
            admission = token_control.admit(turn_index)
            if admission is not None:
                tr.emit({"type": "codex_token_admission", **admission})
                if not admission["admitted"]:
                    rc = 75
                    errors.append("author token admission: " + admission["reason"])
                    break
            cur_prompt = (
                prompt_path
                if turn_index == 0
                else prompt_path.with_name(f"{prompt_path.stem}.cont{turn_index:02d}{prompt_path.suffix}")
            )
            if not turn_index and admission is not None:
                cur_prompt.write_text(("" if token_control.phase == "integration" else msg) + token_control.prompt())
            if turn_index:
                # A previous attempt's final is not this resumed turn's answer.
                if inner_final.is_file():
                    previous_final = rounds / f"{artifact_prefix}.turn{turn_index - 1:02d}.final.txt"
                    previous_final.write_text(inner_final.read_text())
                    inner_final.unlink()
                standing = "" if token_control.phase == "integration" else _CONTINUE_MSG
                note = _CAP.RESUME_NOTE if resumed_after_capacity else ""
                cur_prompt.write_text(standing + note + token_control.prompt())
                resumed_after_capacity = False
                if sandbox == "bwrap":
                    _START.call("config_verify", _verify_frozen_config, codex_home, home_info["config_sha256"])
                resume_argv = _START.call(
                    "resume_argv",
                    build_resume_cmd,
                    ws,
                    model=resolved,
                    effort=effort,
                    final_path=inner_final,
                    sandbox=sandbox,
                    thread_id=thread_id,
                    codex_bin=codex_bin,
                )
                if sandbox == "bwrap":
                    # Re-wrap exactly as the first turn was, so the continuation runs inside the SAME
                    # sandbox with the same binds. Rewriting the wrapped string instead would have to
                    # edit shell syntax; rebuilding it cannot drift from the first turn's wrapping.
                    _inner = " ".join(shlex.quote(c) for c in resume_argv)
                    cmd = _sandbox_script(
                        rounds,
                        rnd,
                        turn_index,
                        _START.call(
                            "sandbox_assembly",
                            sandbox_command,
                            _inner,
                            ws,
                            bundle,
                            extra_binds=_START.call("runtime_bindings", runtime_selector, codex_home),
                        ),
                    )
                else:
                    cmd = resume_argv
            with open(stderr_path, "ab") as err_f, open(cur_prompt, "rb") as in_f:
                try:
                    from merlin.common.process.cpu_pool import parse_roster, selected_pool

                    with selected_pool(parse_roster(os.environ.get("MERLIN_CODEX_CPUS", ""))):
                        proc = _START.call(
                            "provider_popen",
                            subprocess.Popen,
                            cmd,
                            stdin=in_f,
                            stdout=subprocess.PIPE,
                            stderr=err_f,
                            cwd=str(ws),
                            env=dict(os.environ),
                            start_new_session=True,
                        )
                    active_proc, active_pgid = proc, proc.pid
                    token_control.provider_started()
                    # Resource telemetry is sampled OUTSIDE the graded sandbox and follows the complete
                    # bwrap/Codex/tool descendant tree. One file per continuation turn prevents ambiguous
                    # counter resets when Codex is resumed in a new process.
                    _resource_path = rounds / f"{artifact_prefix}.turn{turn_index:02d}.resource_samples.jsonl"
                    try:
                        subprocess.Popen(
                            [
                                sys.executable,
                                str(module_source_path("merlin.development.backend.providers.resource_sampler")),
                                "--pid",
                                str(proc.pid),
                                "--output",
                                str(_resource_path),
                            ],
                            cwd=str(ws),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    except OSError as _e:
                        tr.emit({"type": "resource_sampler_unavailable", "reason": str(_e), "arrived_at": _now()})
                except (OSError, ValueError) as exc:
                    # E2BIG NAMES `bash` AND NOTHING ELSE, so a spawn refused for size looks identical to
                    # a missing interpreter. execve bounds argv AND envp together, and the environment
                    # here is inherited from whatever launched the stage -- so record both sides, and the
                    # largest single contributor of each, or the next reader guesses like this one did.
                    _env = dict(os.environ)
                    _argv_bytes = sum(len(part.encode("utf-8", "replace")) + 1 for part in cmd)
                    _env_bytes = sum(
                        len(k.encode("utf-8", "replace")) + len(v.encode("utf-8", "replace")) + 2
                        for k, v in _env.items()
                    )
                    _biggest_env = sorted(
                        ((len(v.encode("utf-8", "replace")), k) for k, v in _env.items()), reverse=True
                    )[:5]
                    tr.emit(
                        {
                            "type": "result",
                            "subtype": "error",
                            "is_error": True,
                            "result": f"codex spawn failed: {type(exc).__name__}: {exc}",
                            "spawn_sizes": {
                                "argv_bytes": _argv_bytes,
                                "argv_parts": [len(part) for part in cmd],
                                "env_bytes": _env_bytes,
                                "env_count": len(_env),
                                "largest_env": [{"name": k, "bytes": n} for n, k in _biggest_env],
                                "total_bytes": _argv_bytes + _env_bytes,
                            },
                        }
                    )
                    tr.close()
                    return 127, tpath

                buf = bytearray()
                next_control_poll = time.monotonic()
                stopped_drain_deadline = None
                with selectors.DefaultSelector() as sel:
                    sel.register(proc.stdout.fileno(), selectors.EVENT_READ)
                    while True:
                        now = time.monotonic()
                        if (
                            operator_interrupted is None
                            and (continuation_stop is not None or token_control.policy is not None)
                            and now >= next_control_poll
                        ):
                            next_control_poll = now + _CONTROL_POLL_S
                            reason = poll_host_control()
                            reason = reason or token_control.poll()
                            if reason is not None:
                                if type(reason) is not str or not reason.strip():
                                    raise ValueError("host continuation stop must return a nonempty reason or None")
                                operator_interrupted = reason
                                interrupted_turn = not turn_completed
                                tr.emit(
                                    {
                                        "type": "codex_operator_interrupt",
                                        "reason": reason,
                                        "turn": turn_index + 1,
                                        "turn_incomplete": interrupted_turn,
                                        "author_process_group": proc.pid,
                                        "arrived_at": _now(),
                                    }
                                )
                                _kill_tree(proc)
                                if not _process_group_quiescent(proc.pid):
                                    raise RuntimeError("author process group remains live; refusing snapshot handoff")
                                stopped_drain_deadline = time.monotonic() + _CONTROL_POLL_S
                        if stopped_drain_deadline is not None and time.monotonic() >= stopped_drain_deadline:
                            break
                        if time.monotonic() >= deadline and operator_interrupted is None:
                            timed_out = True
                            break
                        wait = min(_POLL_S, max((stopped_drain_deadline or deadline) - time.monotonic(), 0.0))
                        if not sel.select(timeout=wait):
                            if proc.poll() is not None and not sel.select(timeout=0):
                                break
                            continue
                        try:
                            chunk = os.read(proc.stdout.fileno(), 65536)
                        except OSError:
                            break
                        if not chunk:
                            break
                        buf += chunk
                        while True:
                            nl = buf.find(b"\n")
                            if nl < 0:
                                break
                            line = bytes(buf[: nl + 1])
                            del buf[: nl + 1]
                            seq += 1
                            arrived = _now()
                            # Durable FIRST, interpreted second.
                            raw_f.write(line)
                            raw_f.flush()
                            text = line.decode("utf-8", errors="replace").rstrip("\n")
                            try:
                                event = json.loads(text)
                                if not isinstance(event, dict):
                                    event = None
                            except json.JSONDecodeError:
                                event = None
                            stamped_f.write(
                                json.dumps(
                                    {
                                        "seq": seq,
                                        "arrived_at": arrived,
                                        **({"event": event} if event is not None else {"unparsed": text}),
                                    }
                                )
                                + "\n"
                            )
                            stamped_f.flush()
                            if event is None:
                                tr.emit(
                                    {"type": "codex_unparsed", "seq": seq, "arrived_at": arrived, "line": text[:500]}
                                )
                                continue

                            etype = event.get("type")
                            if etype == EVENT_THREAD_STARTED:
                                announced = event.get("thread_id")
                                if (
                                    not isinstance(announced, str)
                                    or not announced
                                    or (thread_id is not None and announced != thread_id)
                                ):
                                    errors.append("codex session thread identity changed or is missing")
                                    _kill_tree(proc)
                                else:
                                    thread_id = announced
                                tr.emit({"type": "codex_thread", "thread_id": thread_id, "arrived_at": arrived})
                            elif etype == EVENT_TURN_STARTED:
                                turns_started += 1
                            elif etype == EVENT_TURN_COMPLETED:
                                turn_completed = True
                                shaped, reported = usage_to_claude_shape(event.get("usage") or {})
                                if reported:
                                    turns_reported += 1
                                message: dict[str, Any] = {
                                    "id": f"codex_{rnd}_{turns_started}",
                                    "model": resolved,
                                    "content": [],
                                }
                                if reported:
                                    message["usage"] = shaped
                                record = {"type": "assistant", "message": message, "arrived_at": arrived}
                                if not reported:
                                    record["codex_usage_unreported"] = True
                                tr.emit(record)
                            elif etype == EVENT_TURN_FAILED:
                                turn_failed = True
                                # No usage is carried here: unmeasured, not free.
                                errors.append(_error_text(event.get("error")))
                                tr.emit(
                                    {
                                        "type": "assistant",
                                        "message": {
                                            "id": f"codex_{rnd}_{turns_started}",
                                            "model": resolved,
                                            "content": [],
                                        },
                                        "codex_usage_unreported": True,
                                        "codex_turn_failed": True,
                                        "arrived_at": arrived,
                                    }
                                )
                            elif etype == EVENT_ERROR:
                                errors.append(_error_text(event.get("message") or event))
                            elif etype in (EVENT_ITEM_STARTED, EVENT_ITEM_COMPLETED):
                                item = event.get("item")
                                if not isinstance(item, dict):
                                    continue
                                itype = item.get("type")
                                turn_progress = turn_progress or (etype == EVENT_ITEM_COMPLETED and _CAP.progress(item))
                                item_id = str(item.get("id") or f"anon_{seq}")
                                if itype == ITEM_AGENT_MESSAGE and etype == EVENT_ITEM_COMPLETED:
                                    tr.emit(
                                        {
                                            "type": "assistant",
                                            "message": {
                                                "id": f"codex_msg_{item_id}",
                                                "model": resolved,
                                                "content": [{"type": "text", "text": item.get("text") or ""}],
                                            },
                                            "arrived_at": arrived,
                                        }
                                    )
                                elif itype == ITEM_REASONING and etype == EVENT_ITEM_COMPLETED:
                                    tr.emit(
                                        {
                                            "type": "assistant",
                                            "message": {
                                                "id": f"codex_think_{item_id}",
                                                "model": resolved,
                                                "content": [{"type": "thinking", "thinking": item.get("text") or ""}],
                                            },
                                            "arrived_at": arrived,
                                        }
                                    )
                                elif itype == ITEM_ERROR:
                                    errors.append(_error_text(item.get("message") or item))
                                elif itype in _TOOL_ITEMS:
                                    if etype == EVENT_ITEM_STARTED:
                                        pending_tools[item_id] = item
                                        tr.emit(
                                            {
                                                "type": "assistant",
                                                "message": {
                                                    "id": f"codex_tool_{item_id}",
                                                    "model": resolved,
                                                    "content": [_tool_block(item, f"codex_tool_{item_id}")],
                                                },
                                                "arrived_at": arrived,
                                            }
                                        )
                                    else:
                                        pending_tools.pop(item_id, None)
                                        tr.emit(
                                            {
                                                "type": "user",
                                                "message": {
                                                    "content": [
                                                        {
                                                            "type": "tool_result",
                                                            "tool_use_id": f"codex_tool_{item_id}",
                                                            "content": _clip_tool_output(
                                                                item.get("aggregated_output") or ""
                                                            ),
                                                            "is_error": bool(item.get("exit_code")),
                                                        }
                                                    ]
                                                },
                                                "arrived_at": arrived,
                                            }
                                        )
                                else:
                                    unknown.append(str(itype))
                            else:
                                unknown.append(str(etype))

                if buf:
                    raw_f.write(bytes(buf))
                    raw_f.flush()
                if timed_out:
                    _kill_tree(proc)
                rc = proc.wait()
                if control_refusal is not None:
                    rc = rc or 1
                # A successful leader exit is not proof that its compiler/tool descendants exited.
                _terminate_process_group(proc.pid, proc)
                active_proc, active_pgid = None, None
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            # CONTINUE only while every reason to stop is absent, and say which one stopped it. A silent
            # stop here would look identical to the defect this exists to fix.
            turn_index += 1
            remaining = deadline - time.monotonic()
            turn_errors = errors[turn_errors_start:]
            token_control.finish()
            if (
                token_control.can_resume_checkpoint(
                    operator_interrupted, thread_id=thread_id, continuous=continue_session, remaining_s=remaining
                )
                and not timed_out
                and turn_index < _CONTINUE_MAX_TURNS
                and remaining > _CONTINUE_MIN_S
                and control_refusal is None
                and poll_host_control() is None
            ):
                tr.emit(
                    {
                        "type": "codex_token_checkpoint",
                        "reason": operator_interrupted,
                        "turn_incomplete": interrupted_turn,
                        "usage_complete": False,
                    }
                )
                operator_interrupted = None
                rc = 0
                recovered_errors.update(range(turn_errors_start, len(errors)))
                continue
            host_stop = operator_interrupted or poll_host_control()
            if turn_completed or turn_progress:  # the model served this turn: earlier refusals have passed
                capacity_consecutive = 0
            backoff_s = _CAP.backoff_s(capacity_consecutive + 1)
            retry_capacity = (
                continue_session
                and host_stop is None
                and thread_id
                and turn_failed
                and not turn_completed
                and not timed_out
                and rc != 0
                and _CAP.refusal(turn_errors)
                and capacity_consecutive < _CAP.MAX_CONSECUTIVE
                and turn_index < _CONTINUE_MAX_TURNS
                and remaining > backoff_s + _CONTINUE_MIN_S
            )
            if retry_capacity:
                capacity_retries += 1
                capacity_consecutive += 1
                tr.emit(
                    {
                        "type": "codex_capacity_retry",
                        "attempt": capacity_retries,
                        "consecutive": capacity_consecutive,
                        "thread_id": thread_id,
                        "model": resolved,
                        "backoff_s": backoff_s,
                        "arrived_at": _now(),
                    }
                )
                time.sleep(backoff_s)
                recovered_errors.update(range(turn_errors_start, len(errors)))
                resumed_after_capacity = True
                continue
            control_stop = None
            if continue_session and continuation_stop is not None and not timed_out and rc == 0:
                control_stop = poll_host_control()
                remaining = deadline - time.monotonic()
            stop = (
                ("host: " + host_stop if host_stop is not None else "")
                or ("" if continue_session else "continuation not enabled")
                or ("no thread id to resume" if not thread_id else "")
                or ("wall budget spent" if remaining <= _CONTINUE_MIN_S else "")
                or ("the turn did not complete" if timed_out or rc != 0 else "")
                or (f"turn cap {_CONTINUE_MAX_TURNS}" if turn_index >= _CONTINUE_MAX_TURNS else "")
                or (control_stop or "")
            )
            tr.emit(
                {
                    "type": "codex_session_turn",
                    "turn": turn_index,
                    "continuing": not stop,
                    "stopped_because": stop,
                    "remaining_s": round(max(remaining, 0.0), 1),
                    "thread_id": thread_id,
                    "interrupted": operator_interrupted is not None,
                    "turn_incomplete": interrupted_turn,
                    "arrived_at": _now(),
                }
            )
            if stop:
                break
    finally:
        if active_proc is not None and active_pgid is not None:
            _terminate_process_group(active_pgid, active_proc)
        token_control.finish()
        for handle in (raw_f, stamped_f):
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError):
                pass
            try:
                handle.close()
            except OSError:
                pass

    # Copy the last message out of the workspace, then drop it: the agent's own final message is not
    # part of its next round's inputs, and leaving it in the workspace would make it one.
    if inner_final != final_path and inner_final.is_file():
        try:
            final_path.write_text(inner_final.read_text())
            inner_final.unlink()
        except OSError as _e:
            print(f"[codex] could not recover the final message: {_e}", file=sys.stderr)
    final_text = final_path.read_text() if final_path.is_file() else ""
    usage_complete = turns_started > 0 and turns_reported >= turns_started
    unrecovered_errors = [error for index, error in enumerate(errors) if index not in recovered_errors]
    result_rc = (
        OPERATOR_INTERRUPTED_EXIT_CODE
        if operator_interrupted is not None
        else 124
        if timed_out
        else ((rc or 1) if unrecovered_errors else rc)
    )
    summary = {
        "thread_id": thread_id,
        "turns_started": turns_started,
        "turns_usage_reported": turns_reported,
        "usage_complete": usage_complete,
        "unknown_types": sorted(set(unknown)),
        "errors": errors[:10],
        "capacity_retries": capacity_retries,
        "unrecovered_errors": unrecovered_errors[:10],
        "timed_out": timed_out,
        "operator_interrupted": operator_interrupted is not None,
        "interruption_reason": operator_interrupted,
        "turn_incomplete": interrupted_turn,
        "exit_code": result_rc,
        "process_exit_code": rc,
        # Which CLI parsed this stream. `unknown_types` above DETECTS contract drift; this says what
        # to compare a drifted run against. None means the binary could not be asked, never a guess.
        "cli_version": cli_version(codex_bin),
        "wall_s": round(time.monotonic() - started, 3),
        # ChatGPT-auth runs consume a subscription, not metered dollars. Any USD
        # figure downstream is notional and must not enter a money budget.
        "billing_mode": BILLING_MODE,
        "subagent_capacity": _AS.summary_record(capacity, max_concurrent_subagents, helper_effort),
        "author_context": _CTX.summary_record(author_context),
        "token_admission": token_control.last,
        "codex_home": home_info,
        "artifacts": {
            "raw": str(raw_path),
            "timestamped": str(stamped_path),
            "stderr": str(stderr_path),
            "prompt": str(prompt_path),
            "final": str(final_path),
        },
    }
    (rounds / f"{artifact_prefix}.codex_summary.json").write_text(json.dumps(summary, indent=2))
    tr.emit({"type": "codex_summary", **summary})
    _link_for_aet(run_dir, raw_path, rnd)

    if operator_interrupted is not None:
        tr.emit(
            {
                "type": "result",
                "subtype": "interrupted",
                "is_error": True,
                "result": "trusted host interrupted author: " + operator_interrupted,
                "turn_incomplete": interrupted_turn,
            }
        )
        tr.close()
        return result_rc, tpath
    if timed_out:
        tr.emit(
            {"type": "result", "subtype": "error", "is_error": True, "result": f"codex exec timed out after {timeout}s"}
        )
        tr.close()
        return result_rc, tpath
    if rc != 0 or unrecovered_errors:
        tr.emit(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "result": (unrecovered_errors[0] if unrecovered_errors else f"codex exited {rc}")[:500],
            }
        )
        tr.close()
        return result_rc, tpath
    tr.emit({"type": "result", "subtype": "success", "is_error": False, "result": final_text[:2000]})
    tr.close()
    return 0, tpath
