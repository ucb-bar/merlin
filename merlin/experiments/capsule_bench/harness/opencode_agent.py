"""OpenCode CLI agent driver for capsule-bench.

Drives ONE capsule-bench round through the provider-agnostic ``opencode`` CLI (https://opencode.ai) instead
of the Bedrock Converse loop (``bedrock_agent``) or the ``claude`` CLI. opencode runs its OWN agentic tool
loop (its native read/edit/bash tools), so — exactly like the claude-CLI path — we invoke the whole
``opencode run`` process (optionally wrapped by bwrap) and then shim its session ``export`` into the
claude-compatible stream-json transcript the harness grades + accounts. Feedback rides the SAME channel as
every driver: the agent reads ``ws/qa/verdict.json`` and runs the staged ``agent_selfcheck.py`` shim through
its bash tool (goldens withheld, broker-proxied outside the sandbox).

Ported (NOT imported — the harness venv has no chia/Ray dep) from ``chia/models/opencode.py``: the session
``export`` schema + the session-id/error parsing. Flags updated for opencode >= 1.18 (``--auto`` replaces the
old ``--dangerously-skip-permissions``; ``-m provider/model``; captured to a FILE because opencode truncates a
piped stream at 64 KiB and still exits 0).

⚠️ Runtime prerequisites for a LIVE run (not needed to import/wire): the ``opencode`` binary on PATH and its
Bedrock provider authenticated (opencode uses the @ai-sdk/amazon-bedrock provider — the AWS cred chain /
bearer token in the inherited env). opencode is Bun-based, so — like the ``claude`` binary — it typically
needs ``--sandbox none`` in this environment; at ``none`` the copied-workspace + post-run transcript audit
provide isolation (the same path the claude driver uses today).
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import model_tiers as _MT
from merlin.common import arrival_stamp as _AS  # the one arrival-time convention

# opencode's provider id for Amazon Bedrock (the @ai-sdk/amazon-bedrock provider). Overridable via env for a
# differently-registered provider. A model value that already carries a provider ("prov/model") is respected.
_BEDROCK_PROVIDER = os.environ.get("OPENCODE_BEDROCK_PROVIDER", "amazon-bedrock")


def _provider_model(model: str) -> str:
    """Map our alias/id to opencode's ``provider/model``. A value already carrying a provider (``/``) passes
    through; a bare Bedrock id resolves via model_tiers and gets the Bedrock provider prefix."""
    if "/" in model:
        return model
    return f"{_BEDROCK_PROVIDER}/{_MT.resolve(model)}"


def _system_prompt(te) -> str:
    """The opencode `primary` agent prompt: DRIVER MECHANICS ONLY.

    Every arm of a campaign must receive the same instructions, or the comparison is between prompts
    rather than between models. The claude driver pipes TASK.md and nothing else; the codex driver adds
    only its kickoff line and deliberately authors no instruction file. This prompt used to add ~2 KB of
    strategy on top of TASK.md, and measurement showed each addition pushed toward the failure mode the
    open models actually exhibited:

      * "a LIMITED number of tool turns" — there is no such cap anywhere in this driver; the pressure was
        invented, and a model that believes its turns are scarce stops investigating.
      * "START WRITING files ... do not over-explore" — the run that scored 20/20 spent 33 of its 45
        actions reading before its first edit. The runs told not to explore wrote early and thrashed
        (one made 114 edits inside a single 40 KB file and converged on nothing).
      * "Do NOT run the RTL-facts GENERATORS" — this arm's own bundle manifest grants them
        ("ALLOWED (CIRCT arm only): RTL-facts generators gen_isa_module/gen_rtl_digest/gen_numeric_facts")
        and TASK.md's starter plan tells the agent to use them. The submissions show the cost: the two
        runs that ran the generators shipped a byte-identical derived encoder, and the run that obeyed
        this line shipped a hand-written substitute that used 1 where the RTL-derived mesh dimension is 16.

    So the strategy is gone. What remains is what a driver legitimately owns: which of ITS tools performs
    the actions TASK.md describes. Everything about the task -- the entrypoints, the manifest contract, the
    grounding material, the integrity rules, the self-check loop -- lives in TASK.md, which every driver
    reads."""
    return (
        "You are an autonomous compiler engineer working in this workspace.\n"
        "TASK.md is your specification and is authoritative: read it first and follow it. If a prior "
        "round left `qa/verdict.json`, read that too -- it is the official grader verdict for your last "
        "submission.\n"
        "Tool mapping for this environment: use your edit/write tools to create and change files, and "
        "your bash tool to run commands (builds, the self-check, and any tooling TASK.md points you at). "
        "Nothing in TASK.md is disabled here.\n"
        "Work until the task's own success criterion is met."
    )


def _parse_session_id(stdout: str) -> str | None:
    """Pull the session id out of ``run --format json`` stdout (each event carries ``sessionID``)."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = ev.get("sessionID") or (ev.get("part") or {}).get("sessionID")
        if sid:
            return sid
    return None


def _parse_run_error(stdout: str) -> str | None:
    """First structured ``type:"error"`` event (pre-request failures like an unknown model never reach the
    export; opencode also exits 0 on many failures, so this stream error is the reliable signal)."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "error":
            err = ev.get("error") or {}
            return f"{err.get('name', 'error')}: {(err.get('data') or {}).get('message', '')}"[:300]
    return None


# A round is killed when the agent makes NO PROGRESS for this long, independently of the wall-clock
# round timeout. Measured: a stalled agent held a 4h round open having emitted nothing for 28 min.
#
# The threshold is generous ON PURPOSE. Progress is now bytes OR CPU, so anything the agent does locally
# -- building, self-checking, writing files -- keeps the round alive. The only thing left for this timer
# to catch is a model call that is genuinely wedged, and while waiting on one the process burns no CPU and
# emits nothing, which is indistinguishable from a long turn. Raising reasoning effort makes long turns
# the normal case, so a tight bound here would kill working rounds; the wall-clock round timeout still
# bounds the damage either way. Overridable via MERLIN_OPENCODE_STALL_S; 0 disables the detector.
_STALL_SECONDS = int(os.environ.get("MERLIN_OPENCODE_STALL_S", "1800"))
_POLL_SECONDS = 10
#: How often the capture file is tailed for ARRIVAL STAMPS. Separate from _POLL_SECONDS on purpose: the
#: stall/CPU bookkeeping wants a long window (a 1 s CPU delta is noise), while an activity-share plot wants
#: the finest honest resolution we can pay for. Every line seen in one drain shares that drain's timestamp,
#: so a stamp means "arrived within the last _TAIL_POLL_SECONDS" -- coarse, but MEASURED, not interpolated.
_TAIL_POLL_SECONDS = 1.0
# Progress must clear this much CPU to count. opencode block-buffers stdout when it is redirected to a
# file, so a quiet round can do real work while the byte count stands still (measured: a GLM-5 round wrote
# three source files and ran selfchecks while its transcript sat at 109 bytes for 15 min). CPU is the
# signal that separates that from a hang: a process blocked on a socket burns none, a working agent burns
# plenty. Small enough that idle poll jitter cannot reach it, large enough that real work always does.
_CPU_EPSILON_S = 1.0
_CLK_TCK = os.sysconf("SC_CLK_TCK")

# A WEDGED process defeats both signals above. MEASURED (GLM-5 relaunch, 2026-08-19): 99 tool calls, then
# 95 minutes with no new session part, two tool calls left forever in `running`, and **zero open sockets**
# -- while still burning 0.49 s of CPU per minute across 42 idle threads. That trickle clears
# _CPU_EPSILON_S at every poll, so the detector re-armed forever and the round burned its whole budget.
#
# Zero sockets is the fact that separates it: an agent between provider calls is either holding a
# connection or doing local work, and local work is not free. So a tree with NO sockets counts as making
# progress only if it is genuinely busy -- above a fraction of one core. 0.008 cores (the wedge) is not;
# a compile or a simulator run is far above.
_BUSY_CPU_FRACTION = float(os.environ.get("MERLIN_OPENCODE_BUSY_CPU", "0.10"))


def _tree_socket_count(pgid: int) -> int:
    """Open socket file descriptors across every live process in `pgid`.

    Read structurally from /proc/<pid>/fd: a socket fd's link target is ``socket:[<inode>]``. Counting
    fds (not parsing /proc/net/tcp) keeps this free of a second format to track, and the only thing the
    caller needs is whether the tree is talking to anything at all. A process that exits mid-scan simply
    contributes nothing.
    """
    total = 0
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                rest = fh.read().rpartition(")")[2].split()
            if int(rest[2]) != pgid:
                continue
        except (OSError, ValueError, IndexError):
            continue
        try:
            fd_dir = f"/proc/{pid}/fd"
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(f"{fd_dir}/{fd}").startswith("socket:"):
                        total += 1
                except OSError:
                    continue
        except OSError:
            continue
    return total


def _tree_cpu_seconds(pgid: int) -> float:
    """Total CPU (user+sys) burned by every live process in `pgid`, in seconds.

    Read structurally from /proc, never pattern-matched: the comm field is parenthesised and may itself
    contain spaces and ')', so split at the LAST ')' and index the remainder positionally (utime/stime are
    fields 14/15 one-based, i.e. offsets 11/12 after that split). A process that exits between the listdir
    and the read simply contributes nothing.
    """
    total = 0.0
    try:
        pids = [d for d in os.listdir("/proc") if d.isdigit()]
    except OSError:
        return 0.0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                rest = fh.read().rpartition(")")[2].split()
            if int(rest[2]) != pgid:          # field 5 (pgrp), 3rd after the comm split
                continue
            total += (int(rest[11]) + int(rest[12])) / _CLK_TCK
        except (OSError, ValueError, IndexError):
            continue
    return total


class AgentStalled(subprocess.TimeoutExpired):
    """The agent process is alive but has made NO PROGRESS for `stall_seconds`.

    Measured: a GLM-5 round read its task files, emitted ``step_start``, and then produced zero
    bytes for 28 minutes while the process stayed healthy. A wall-clock cap alone cannot tell that
    apart from slow progress, so with a generous round timeout the run burns the WHOLE budget
    doing nothing. Inactivity is the signal that separates the two. Subclasses TimeoutExpired so
    every existing caller keeps working; the type distinguishes a stall from an honest overrun.
    """


def _capture(cmd: list, env: dict, timeout: int, cwd: str,
             stall_seconds: int = _STALL_SECONDS,
             stamps: list | None = None) -> tuple[int, str, str]:
    """Run ``cmd`` capturing stdout to a FILE (opencode truncates a piped stream at 64 KiB and still exits 0,
    cutting the JSON mid-stream). stdin=DEVNULL because ``run`` blocks on an open stdin pipe.

    ``cmd`` is ``bash -c '<bwrap ... opencode ...>'``, so a plain ``subprocess.run(timeout=)`` would SIGKILL
    only the outer bash and leave the bwrap->opencode grandchildren alive (bwrap ``--die-with-parent`` does
    not reliably cascade across the extra shell + PID namespace); the orphaned child keeps the inherited
    stderr pipe open, hanging the parent's own post-timeout reap (observed: a stalled GLM-5 agent sat at the
    1800s cap with the driver blocked and no result). Run in a NEW SESSION and, on timeout, SIGKILL the whole
    PROCESS GROUP so the entire tree dies and the reap returns promptly; the caller maps TimeoutExpired to a
    rc=124 round result.

    ``stamps``, when given, is FILLED with one ISO-8601 UTC arrival time per stdout line, in order, as the
    lines appear in the capture file -- the same ``arrived_at`` convention :mod:`codex_agent` and
    :mod:`merlin.common.arrival_stamp` use. opencode's stream cannot be piped (see above), so the file is
    tailed instead of read; the resolution is therefore ``_TAIL_POLL_SECONDS``, not per-line."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".out", prefix="oc_out_", delete=False)
    tmp.close()
    errf = tempfile.NamedTemporaryFile(mode="w", suffix=".err", prefix="oc_err_", delete=False)
    errf.close()
    _tailf = None                      # the arrival-stamp tail handle; closed in the outer finally

    def _reap(proc):
        """Kill the whole tree: bash -> bwrap -> opencode."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()

    try:
        # stderr goes to a FILE, not a pipe: an orphaned grandchild holding a pipe open is exactly what
        # hung the post-timeout reap before, and a file also lets the poll loop below run without ever
        # risking a full-pipe deadlock.
        with open(tmp.name, "w") as out, open(errf.name, "w") as err:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 text=True, cwd=cwd, env=env, start_new_session=True)
            deadline = time.monotonic() + timeout
            pgid = os.getpgid(p.pid)
            last_size, last_change = -1, time.monotonic()
            last_poll = time.monotonic()
            last_cpu = _tree_cpu_seconds(pgid)
            # Arrival stamps. The stream goes to a FILE (it cannot be piped), so the file is TAILED:
            # every line that became visible since the previous drain gets that drain's timestamp.
            # Coarse, but measured -- nothing here interpolates a time it did not observe.
            if stamps is not None:
                _tailf = open(tmp.name, "rb")
            _pending = bytearray()

            def _drain(final: bool = False) -> None:
                if _tailf is None:
                    return
                _pending.extend(_tailf.read())
                arrived = _AS.now_iso()
                while True:
                    nl = _pending.find(b"\n")
                    if nl < 0:
                        break
                    stamps.append(arrived)
                    del _pending[:nl + 1]
                if final and _pending:              # a last line with no trailing newline
                    stamps.append(arrived)
                    _pending.clear()
            def _with_partial(exc):
                """Attach whatever the run produced before it was killed.

                The stream is this driver's ONLY record of a round: the transcript is reconstructed from
                it after the process exits, and ``finally`` below deletes the capture files. So a round
                killed on the wall clock used to report tool_calls=0 and no usage at all -- a live GLM-5
                round spent 40 minutes working and left a two-line transcript. The work is not
                recoverable, but the record of it is."""
                try:
                    _drain(final=True)          # stamp what arrived before the kill, then salvage it
                    exc.partial_stdout = Path(tmp.name).read_text(errors="replace")
                    exc.partial_stderr = Path(errf.name).read_text(errors="replace")
                except OSError:
                    exc.partial_stdout = exc.partial_stderr = ""
                return exc

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _reap(p)
                    raise _with_partial(subprocess.TimeoutExpired(cmd, timeout))
                try:
                    p.wait(timeout=min(_TAIL_POLL_SECONDS, remaining))
                    break                                   # exited on its own
                except subprocess.TimeoutExpired:
                    pass
                _drain()
                # The stall/CPU bookkeeping below keeps its ORIGINAL _POLL_SECONDS cadence: its CPU delta
                # and socket check are meaningless over a one-second window, and shortening it would
                # change what counts as a stall. Only the arrival tail runs at the finer rate.
                if time.monotonic() - last_poll < _POLL_SECONDS:
                    continue
                # Progress is WORK DONE, not liveness: a stalled agent stays healthy indefinitely. Two
                # independent signals, because either one alone is wrong: bytes miss a quiet round whose
                # output is still sitting in opencode's stdio buffer, and CPU alone would miss an agent
                # spinning without accomplishing anything. Either advancing means the round is alive.
                try:
                    size = os.path.getsize(tmp.name) + os.path.getsize(errf.name)
                except OSError:
                    size = last_size
                cpu = _tree_cpu_seconds(pgid)
                now = time.monotonic()
                # Third signal: a tree with no open sockets is only "working" if it is actually busy.
                # See _tree_socket_count -- a wedged agent trickles just enough CPU to defeat the epsilon
                # forever while talking to nobody and producing nothing.
                elapsed = max(now - last_poll, 1e-6)
                cpu_rate = (cpu - last_cpu) / elapsed
                busy = cpu - last_cpu >= _CPU_EPSILON_S
                if busy and _tree_socket_count(pgid) == 0 and cpu_rate < _BUSY_CPU_FRACTION:
                    busy = False              # alive, connected to nothing, and not doing local work
                last_poll = now
                if size != last_size or busy:
                    last_size, last_cpu, last_change = size, max(cpu, last_cpu), now
                elif stall_seconds and (now - last_change) >= stall_seconds:
                    _reap(p)
                    raise _with_partial(AgentStalled(cmd, int(now - last_change)))
            _drain(final=True)
        return p.returncode, Path(tmp.name).read_text(), Path(errf.name).read_text()
    finally:
        if _tailf is not None:
            try:
                _tailf.close()
            except OSError:
                pass
        for _f in (tmp.name, errf.name):
            try:
                os.unlink(_f)
            except OSError:
                pass


def _msg_arrived_at(info: dict) -> str | None:
    """The arrival time of an EXPORTED message, taken from opencode's own ``info.time`` (epoch ms).

    This path runs only when the live stream yielded nothing, so there is no observed arrival time to
    use. Taking opencode's own recorded time keeps the salvaged messages spread across the round instead
    of collapsed onto the single instant the salvage ran. When opencode recorded no time, this returns
    None and ``emit`` falls back to the moment the record was WRITTEN -- an upper bound, not the
    message's own time, and the same fallback every driver-authored record already gets."""
    t = info.get("time") if isinstance(info, dict) else None
    if not isinstance(t, dict):
        return None
    for key in ("completed", "created"):
        v = t.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return datetime.fromtimestamp(v / 1000.0, tz=timezone.utc).isoformat()
    return None


def _export_to_transcript(export: dict, mid: str, rnd: int, emit) -> None:
    """Map opencode's session export → claude-compatible ``assistant`` events (usage + content blocks) so
    experiment_tokens.parse_transcript / qa_grade / audit_transcript consume it like every other driver."""
    for msg in export.get("messages", []) or []:
        info = msg.get("info", {}) if isinstance(msg, dict) else {}
        if info.get("role") != "assistant":
            continue
        arrived = _msg_arrived_at(info)
        tok = info.get("tokens") or {}
        cache = tok.get("cache") or {}
        blocks = []
        tool_results = []
        for p in msg.get("parts", []) or []:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text":
                blocks.append({"type": "text", "text": p.get("text", "")})
            elif p.get("type") == "tool":
                st = p.get("state") or {}
                cid = p.get("callID") or p.get("id")            # same id links tool_use <-> tool_result
                blocks.append({"type": "tool_use", "id": cid, "name": p.get("tool", "tool"),
                               "input": st.get("input", {}) if isinstance(st, dict) else {}})
                out = st.get("output") if isinstance(st, dict) else None
                if out is not None:
                    tool_results.append({"type": "tool_result", "tool_use_id": cid,
                                         "content": out if isinstance(out, str) else json.dumps(out)})
        emit({"type": "assistant", "message": {
            "id": f"opencode_{rnd}_{info.get('id', '')}",
            "model": info.get("modelID") or mid,
            "usage": {"input_tokens": tok.get("input", 0) or 0,
                      "output_tokens": tok.get("output", 0) or 0,
                      "cache_read_input_tokens": cache.get("read", 0) or 0,
                      "cache_creation_input_tokens": cache.get("write", 0) or 0},
            "content": blocks}, **({_AS.ARRIVED_AT: arrived} if arrived else {})})
        # claude-compatible tool_result events so the transcript is self-authoritative + the mask-leak
        # audit can correlate each read's result (parity with the claude-CLI path).
        if tool_results:
            emit({"type": "user", "message": {"content": tool_results},
                  **({_AS.ARRIVED_AT: arrived} if arrived else {})})


def _parse_run_stream(stdout: str, mid: str, rnd: int, emit, stamps: list | None = None) -> int:
    """Reconstruct the transcript from opencode's ``run --format json`` STDOUT stream (always captured),
    instead of a post-hoc ``opencode export`` (fragile: a wrong sandbox isolates the session data dir, so
    the export is blind and yields 0 messages — the empty-transcript bug). Each stream line is an event with
    a ``part``: a ``text`` part is an assistant text block; a ``tool`` part is a tool_use (+ its result from
    ``state.output``); ``step-finish`` parts carry token usage. Emits claude-compatible events (assistant +
    user/tool_result) so parse_transcript / audit_transcript / conformance consume them like every driver.
    Returns the number of tool calls seen.

    ``stamps`` (from :func:`_capture`) carries one arrival time per stdout LINE, so each event emitted
    here gets the ``arrived_at`` of the line it came from -- the same field the codex driver writes, so
    one reader consumes every driver. Split on ``"\n"`` rather than ``splitlines()`` because that is
    exactly how the tail counted lines; any other split would shift the alignment silently. When no
    stamps were collected the events simply carry none (see ``emit``): an absent stamp is honest, an
    invented one is not."""
    tok = {"input": 0, "output": 0, "cread": 0, "cwrite": 0, "reasoning": 0}
    n_tools = 0

    def _stamped(obj: dict, at: str | None) -> dict:
        if at:
            obj[_AS.ARRIVED_AT] = at            # appended last: no existing field moves or changes
        return obj

    for _i, line in enumerate(stdout.split("\n")):
        arrived = stamps[_i] if (stamps is not None and _i < len(stamps)) else None
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        part = e.get("part") or {}
        pt = part.get("type")
        if pt == "text" and part.get("text"):
            emit(_stamped({"type": "assistant", "message": {"id": f"opencode_{rnd}_{part.get('id','')}", "model": mid,
                  "usage": {"input_tokens": 0, "output_tokens": 0}, "content": [{"type": "text", "text": part["text"]}]}}, arrived))
        elif pt == "tool":
            st = part.get("state") or {}
            cid = part.get("callID") or part.get("id")
            emit(_stamped({"type": "assistant", "message": {"id": f"opencode_{rnd}_{cid}", "model": mid,
                  "usage": {"input_tokens": 0, "output_tokens": 0},
                  "content": [{"type": "tool_use", "id": cid, "name": part.get("tool", "tool"),
                               "input": st.get("input", {}) if isinstance(st, dict) else {}}]}}, arrived))
            n_tools += 1
            out = st.get("output") if isinstance(st, dict) else None
            if out is not None:
                emit(_stamped({"type": "user", "message": {"content": [
                    {"type": "tool_result", "tool_use_id": cid,
                     "content": out if isinstance(out, str) else json.dumps(out)}]}}, arrived))
        elif pt == "step-finish":
            t = part.get("tokens") or {}
            c = t.get("cache") or {}
            tok["input"] += t.get("input", 0) or 0
            tok["output"] += t.get("output", 0) or 0
            tok["cread"] += c.get("read", 0) or 0
            tok["cwrite"] += c.get("write", 0) or 0
            # Reasoning is reported separately by providers that bill it separately. Bedrock bills a
            # model's thinking as ordinary OUTPUT, so this stays 0 there and the thinking is already
            # inside output_tokens -- recording it anyway is what makes that distinction visible instead
            # of a silent zero that reads as "this model did not think".
            tok["reasoning"] += t.get("reasoning", 0) or 0
    # one usage-bearing assistant event so token/cost accounting sees the round's totals. Stamped with
    # the LAST line's arrival: it is a rollup of the stream just read, not an event that arrived later.
    emit(_stamped({"type": "assistant", "message": {"id": f"opencode_{rnd}_usage", "model": mid, "content": [],
          "usage": {"input_tokens": tok["input"], "output_tokens": tok["output"],
                    "cache_read_input_tokens": tok["cread"], "cache_creation_input_tokens": tok["cwrite"],
                    "reasoning_tokens": tok["reasoning"]}}}, stamps[-1] if stamps else None))
    return n_tools


def opencode_runtime_binds(data_home: Path, config_path: Path | None = None) -> list[str]:
    """bwrap args giving opencode a WRITABLE, ISOLATED data home + a READABLE config inside the sandbox.

    The launcher resolves through the already-bound ``~/.nvm``, but opencode keeps its state under
    ``~/.local/share/opencode``, which nothing binds -- so inside the box it starts with no data dir.
    Binding the REAL one would hand the agent ``~/.local/share/opencode/storage``: every prior opencode
    session on this host, an answer-leak surface for exactly the same reason ``~/.codex/sessions`` is.

    So the run gets its own directory and ``XDG_DATA_HOME`` points at it (verified: opencode honours the
    variable and creates ``<home>/opencode`` there). These binds are passed to ``bwrap_cmd`` as
    ``extra_binds``, which are applied BEFORE the answer-mask pass, so masking still wins.

    ``config_path`` MUST be bound too. It is written with ``tempfile`` -- i.e. under ``$TMPDIR`` or
    ``/tmp`` -- and bwrap tmpfs-hides BOTH ``/tmp`` and ``/scratch*``, so inside the box the file simply
    does not exist. opencode then prints

        ! agent "merlinbench" not found. Falling back to default agent

    to stderr, which this driver does not persist, and runs with its OWN defaults. VERIFIED against every
    opencode run in this repo's history: the session logs record ``agent=build`` (plus ``title`` /
    ``explore`` / ``general``) and never once ``agent=merlinbench``. So no sandboxed opencode run has ever
    received our system prompt, our per-model ``limit.output``, our compaction settings, or our tool
    denials -- and the default agent set is where the ``task`` subagent comes from. Binding the config
    read-only is what makes any of those settings real.
    """
    data_home.mkdir(parents=True, exist_ok=True)
    binds = ["--bind", str(data_home), str(data_home),
             "--setenv", "XDG_DATA_HOME", str(data_home)]
    if config_path is not None:
        binds += ["--ro-bind", str(config_path), str(config_path),
                  "--setenv", "OPENCODE_CONFIG", str(config_path)]
    return binds


# ---------------------------------------------------------------------------------------------------
# Context-window budgeting.
#
# MEASURED (gemmini arm-4, 2026-08-19): opencode asks the provider for its registry ``limit.output``
# tokens of completion on EVERY step. For nemotron-super-3-120b that registry value is 32_000 against a
# 131_072 window, so the largest prompt the session can ever send is 99_072 tokens -- 24% of the window
# is reserved for output that never arrives (the measured completion is 200-400 tokens/step). The round
# then dies on a provider 400 ("maximum context length is 131072 ... you requested 32000 output tokens
# and your prompt contains at least 99073 input tokens") rather than compacting, because opencode's
# auto-compaction threshold is computed against ``limit.context`` and not against the usable
# ``context - output`` budget. Two rounds of the nemotron campaign and six steps of its smoke died here.
#
# Both are configuration, not code: declare a realistic per-model ``limit.output``, and give compaction
# an explicit ``reserved`` buffer at least as large as that output ask so it triggers BEFORE the
# provider refuses. ``prune`` additionally drops superseded tool output, which is where the context
# actually goes (a full 20-capsule self-check returns 20-50 KB and is re-sent every subsequent step).
#
# Values are per-model overrides keyed by the BARE provider model id. A model absent here keeps
# opencode's registry defaults -- this never invents a window, it only stops us reserving one we do not
# use. ``OPENCODE_MAX_OUTPUT_TOKENS`` overrides the output ask for a one-off.
import agent_bridge as _BR

_DEFAULT_MAX_OUTPUT = int(os.environ.get("OPENCODE_MAX_OUTPUT_TOKENS",
                                        str(_BR.DEFAULT_MAX_OUTPUT)))


def _window_config(mid: str) -> tuple[dict, dict]:
    """Return (models_override, compaction_config) for the provider/model id ``mid``.

    ``models_override`` is the ``provider.<id>.models`` fragment that re-declares ``limit`` so the
    output reservation stops eating the prompt budget. It is emitted ONLY when we can state the
    context window as a fact -- never guessed -- so an unknown model is left on registry defaults.
    """
    prov, _, bare = mid.partition("/")
    ctx = _CONTEXT_WINDOWS.get(bare)
    out = min(_DEFAULT_MAX_OUTPUT, (ctx // 8) if ctx else _DEFAULT_MAX_OUTPUT)
    # Compaction must fire while a full completion still fits, so reserve the output ask plus headroom
    # for the compaction call itself.
    compaction = {"auto": True, "prune": True, "reserved": max(out * 2, 16000)}
    if not ctx:
        return {}, compaction
    return {prov: {"models": {bare: {"limit": {"context": ctx, "output": out}}}}}, compaction


# Context windows MEASURED from the provider's own 400 response ("This model's maximum context length is
# N tokens"), not from a vendor page. Owned by agent_bridge so the opencode, codex and claude paths all
# budget against the SAME number -- a per-driver copy is how one arm ends up with a different effective
# window than another arm of the same campaign.
_CONTEXT_WINDOWS: dict[str, int] = _BR.CONTEXT_WINDOWS


def run_round(ws: Path, run_dir: Path, model: str, bundle: dict, te, sandbox: str, rnd: int,
              timeout: int, *, subagent_model: str = "", background_model: str = "",
              effort: str = "", **_ignored) -> tuple[int, Path]:
    """Drive ONE capsule-bench round via the opencode CLI. Returns (rc, transcript_path) — the same contract
    as ``launch_agent``'s claude path.

    ``effort`` is the arm's reasoning effort and MUST be threaded, for the reason the claude and codex
    paths already state: an arm that silently ran at a different reasoning effort is a different arm. It
    was previously absorbed by ``**_ignored``, so every opencode run in a campaign executed at the
    provider default while the claude and codex arms of the SAME campaign ran at the declared effort —
    a comparison between models that was partly a comparison between reasoning budgets."""
    import run_baseline_qa_loop as _R  # bwrap_cmd — the same integrity wrapper the claude path uses
    opencode_bin = os.environ.get("OPENCODE_BIN", "opencode")
    mid = _provider_model(model)
    tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tf = open(tpath, "w")

    def emit(obj: dict) -> None:
        # Driver-AUTHORED records (init, result, salvage) are stamped with the moment they are written;
        # stream-derived events arrive here already carrying the arrival time of the line they came from,
        # and setdefault must not overwrite that. Same `arrived_at` field as every other driver.
        obj.setdefault(_AS.ARRIVED_AT, _AS.now_iso())
        tf.write(json.dumps(obj) + "\n"); tf.flush()

    sub = _MT.resolve(subagent_model) if subagent_model else ""
    _delegate = _provider_model(sub) if (sub and _provider_model(sub) != mid) else None
    # What this round ACTUALLY ran with, not what the launcher intended. environment.yaml records the
    # requested effort, and for every opencode run before the effort fix it recorded `high` while the round
    # executed at the provider default -- an artifact asserting something untrue about its own run. The
    # codex driver already writes a record of this shape; matching it makes an opencode run auditable from
    # its own transcript.
    emit({"type": "system", "subtype": "init", "model": mid, "round": rnd, "driver": "opencode",
          "effort_requested": effort or None,
          "variant_passed": (effort or "").strip() or None,
          "sandbox": sandbox,
          "delegate_model": _delegate,
          "context_window": _CONTEXT_WINDOWS.get(mid.partition("/")[2]),
          "max_output_tokens": _window_config(mid)[0].get(mid.partition("/")[0], {})
                               .get("models", {}).get(mid.partition("/")[2], {})
                               .get("limit", {}).get("output"),
          "compaction": _window_config(mid)[1],
          "task_tool_offered": bool(sub),
          "workspace_instruction_files": {n: (ws / n).stat().st_size
                                          for n in ("TASK.md", "AGENTS.md", "CLAUDE.md", "AGENT.md")
                                          if (ws / n).is_file()}})

    # opencode config: our system prompt as the primary agent + allow-all permissions (external_directory is
    # NOT covered by --auto, so it must be allowed here). Tier-within-agent: register a cheaper subagent when
    # one is configured + distinct, so opencode can delegate to it via its own subagent mechanism.
    agent_name = "merlinbench"
    _models_override, _compaction = _window_config(mid)
    cfg: dict = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {agent_name: {"mode": "primary", "prompt": _system_prompt(te), "model": mid}},
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "allow", "external_directory": "allow"},
        # See _window_config: stop reserving a completion budget we never spend, and compact before the
        # provider refuses rather than after.
        "compaction": _compaction,
    }
    if _models_override:
        cfg["provider"] = _models_override
    if not sub:
        # `task` delegates to a subagent. With no delegate configured it is a tool with nothing behind it,
        # and calling it WEDGES the round: measured twice, on two different models (GLM-5 and Opus), as a
        # `task` part left in `running` forever with zero open sockets and no further session parts.
        #
        # It has to be disabled in BOTH places. opencode 1.18 marks the per-agent `tools` map
        # "@deprecated Use 'permission' field instead" and ignores it -- a run whose config carried
        # `agent.merlinbench.tools = {"task": false}` still called `task` and still wedged. The
        # TOP-LEVEL `tools` map is the live one; the per-agent entry stays for older opencode builds.
        # `permission` is the mechanism opencode 1.18 actually honours. Both `tools` maps are set too,
        # but neither works on its own: a run whose config carried the top-level AND per-agent
        # tools={"task": false} still called `task` and still wedged, because the per-agent map is
        # "@deprecated Use 'permission' field instead" and the top-level one did not take either.
        cfg["permission"]["task"] = "deny"
        cfg["agent"][agent_name]["permission"] = {"task": "deny"}
        cfg["tools"] = {"task": False}
        cfg["agent"][agent_name]["tools"] = {"task": False}
    # (resolved above, before the init record, so that record can state it)
    # A delegate is attached ONLY when the caller asks for one. It used to default to the non-Anthropic
    # tier's subagent, so a run nominally measuring one model silently had a SECOND, different model
    # available to it (qwen-coder alongside glm5/nemotron) while the codex and claude arms had none. That
    # is a third variable in a two-variable comparison; the capability stays, the default does not.
    if sub and _provider_model(sub) != mid:
        cfg["agent"]["delegate"] = {"mode": "subagent", "model": _provider_model(sub),
                                    "prompt": "Focused sub-agent: do EXACTLY the delegated task, then reply "
                                              "with a short result summary. Do not read golden/expected files."}
    cfgf = tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="oc_cfg_", delete=False)
    json.dump(cfg, cfgf)
    cfgf.close()

    env = dict(os.environ)
    env["OPENCODE_CONFIG"] = cfgf.name
    env["OPENCODE_DISABLE_PROJECT_CONFIG"] = "1"  # a stray opencode.json in cwd must not shadow our config

    msg = ("Read TASK.md and qa/verdict.json (if present) in your workspace, then build or repair the target "
           "backend under submission/ per those instructions. Run `python3 agent_selfcheck.py --submission "
           "submission --sim spike --capsules all` with your bash tool after each build to grade against the "
           "real oracle (goldens withheld), and iterate until capsules pass. Begin now.")
    run_cmd = [opencode_bin, "run", "--format", "json", "--agent", agent_name, "-m", mid,
               "--dir", str(ws)]
    # opencode spells reasoning effort `--variant` (provider-specific: high / max / minimal). Passing it is
    # what makes an opencode arm comparable to the codex and claude arms of the same campaign.
    if (effort or "").strip():
        run_cmd += ["--variant", effort.strip()]
    run_cmd += ["--auto", msg]

    # Same integrity wrapper as the claude path: bwrap when requested, else raw (cwd=ws). At bwrap the
    # run gets an isolated data home (see opencode_runtime_binds) so it neither starts without state nor
    # sees any prior session; at none, isolation is the copied workspace + the post-run transcript audit.
    if sandbox == "bwrap":
        from merlin.common.artifacts import cache_dir
        data_home = cache_dir("opencode_home") / f"{run_dir.name}_r{rnd:02d}"
        env["XDG_DATA_HOME"] = str(data_home)     # also for the outer process, so both agree
        inner = " ".join(shlex.quote(c) for c in run_cmd)
        cmd = ["bash", "-c", _R.bwrap_cmd(inner, ws, bundle,
                                          extra_binds=opencode_runtime_binds(data_home,
                                                                             Path(cfgf.name)))]
    else:
        cmd = run_cmd

    rc = 0
    #: Filled by _capture with one arrival time per stdout line (see _parse_run_stream).
    stamps: list[str] = []
    try:
        code, stdout, stderr = _capture(cmd, env, timeout, str(ws), stamps=stamps)
    except subprocess.TimeoutExpired as _te:
        _why = ("agent stalled: no output for "
                f"{getattr(_te, 'timeout', '?')}s while the process stayed alive"
                if isinstance(_te, AgentStalled) else "opencode run timed out")
        # Salvage the round's record before reporting the kill. A timed-out round did real work and its
        # actions and token usage are in the stream captured so far; discarding them reported the round as
        # though the agent had done nothing, which is both false and unbudgetable -- the tokens were spent.
        _partial = getattr(_te, "partial_stdout", "") or ""
        _recovered = _parse_run_stream(_partial, mid, rnd, emit, stamps=stamps) if _partial else 0
        if _recovered == 0 and _partial:
            _sid = _parse_session_id(_partial)
            if _sid:                                    # the session store outlives the killed process
                try:
                    _ec, _eo, _ = _capture([opencode_bin, "export", _sid], env, 120, str(ws))
                    _export_to_transcript(json.loads(_eo) if _ec == 0 else {}, mid, rnd, emit)
                except Exception:  # noqa: BLE001 — salvage is best-effort; never mask the timeout
                    pass
        emit({"type": "result", "subtype": "error", "is_error": True,
              "result": _why, "recovered_actions": _recovered})
        tf.close()
        try:
            os.unlink(cfgf.name)
        except OSError:
            pass
        return 124, tpath
    finally:
        pass

    run_err = _parse_run_error(stdout)
    sid = _parse_session_id(stdout)
    if sid is None:
        emit({"type": "result", "subtype": "error", "is_error": True,
              "result": run_err or f"opencode: no session id (rc={code}); {stderr[:200]}"})
        tf.close()
        try:
            os.unlink(cfgf.name)
        except OSError:
            pass
        return (code or 1), tpath

    # PRIMARY: reconstruct from the run's own --format json STDOUT stream (always captured; robust to the
    # sandbox/data-dir isolation that leaves `opencode export` blind). Fall back to `export` only if the
    # stream yielded no tool activity (e.g. a future format change).
    n_tools = _parse_run_stream(stdout, mid, rnd, emit, stamps=stamps)
    if n_tools == 0:
        ecode, eout, _estderr = _capture([opencode_bin, "export", sid], env, max(60, timeout // 4), str(ws))
        try:
            export = json.loads(eout) if ecode == 0 else {}
        except json.JSONDecodeError:
            export = {}
        _export_to_transcript(export, mid, rnd, emit)

    if run_err:
        emit({"type": "result", "subtype": "error", "is_error": True, "result": run_err})
        rc = 1
    else:
        emit({"type": "result", "subtype": "success", "is_error": False})
    tf.close()
    try:
        os.unlink(cfgf.name)
    except OSError:
        pass
    return rc, tpath
