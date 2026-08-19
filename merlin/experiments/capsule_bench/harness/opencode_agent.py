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
from pathlib import Path

import model_tiers as _MT

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
    """The opencode `primary` agent prompt — same ISA-grounding + framework mandate + integrity rules as the
    Converse driver, phrased for opencode's NATIVE tools (its edit/write tools for files, its bash tool to
    build/grade). Kept separate from bedrock_agent's Converse-tool-specific prompt on purpose."""
    t = te.target
    return (
        "You are an autonomous compiler engineer with a LIMITED number of tool turns. Build the target "
        "backend under `submission/` (manifest.yaml + the entrypoint tool + supporting modules) in your "
        "workspace. Read TASK.md first, then START WRITING files with your edit/write tool — do not "
        "over-explore. Get `submission/manifest.yaml` + the tool existing and PARSING as early as possible.\n"
        "MANIFEST FORMAT: manifest.yaml MUST include ALL required top-level keys — `artifact_type`, "
        "`target`, `language`, `authoring`, `integrity_exempt`, `entrypoints`, `commands` — or the whole "
        "package is REJECTED at the contract gate before any capsule grades (n_capsules=0). In particular "
        "include an `authoring:` block with `mode: agent_generated_from_rtl_facts` (it is schema-required and "
        "easy to forget). The `entrypoints.tool` field must be the BARE script path relative to the package "
        f"(e.g. `mlir_oot/{t}_opt.py`) and each command's argv references it as `{{tool}}` — do NOT put "
        "`python3` in the tool field or bake the interpreter/path into argv, or the build fails with `tool "
        "missing`. The harness runs a Python tool through the interpreter for you.\n"
        "CORRECTNESS SIGNAL: you get NO golden values. After every build, run this with your bash tool:\n"
        "  `python3 agent_selfcheck.py --submission submission --sim spike --capsules all`\n"
        "It grades your current submission against the REAL oracle and returns per-capsule pass/fail + the "
        "NUMERIC DIFF (how far off each capsule is), goldens withheld. Read the diff, fix your "
        "encoding/lowering, rebuild, self-check again — iterate until capsules pass. Between rounds also read "
        "`qa/verdict.json` in your workspace (the official grader verdict; for this arm it carries an advisory "
        "`rtl_checks` block — FileCheck over your emitted MLIR + decoded trace — to localize structural "
        "mistakes).\n"
        "GROUNDING: READ (do not regenerate) the shipped ISA spec — the green-card and isa_definition — "
        f"mounted read-only at `{t}/isa_include/` (the hwbringup set is at `{t}/`, RTL under `{t}/rtl/`, a "
        f"worked example under `{t}/example_kernel/`). Derive EVERY opcode, encoding, field-layout and the "
        "command-buffer schema from them — never invent an encoding.\n"
        "USE MERLIN'S FRAMEWORK AS AN AUTHORING AID — but SHIP A SELF-CONTAINED PACKAGE. The granted "
        "oot_starterkit is READ-ONLY REFERENCE SOURCE, never an importable dependency: it is on the "
        "path so you can READ it, and importing it from `merlin.*` at runtime is an integrity "
        "FAIL. Read its guide (`cat "
        "oot_starterkit/AGENT.md`, `cat oot_starterkit/scaffold/README.md`) and its modules — "
        "`parse_interface` (parses the fixed merlin_iface grammar — mirror it, don't hand-roll a worse "
        "parser), `CommandBufferBuilder` (emits a SCHEMA-VALID command_buffer.json — mirror it so you PASS "
        "the command_buffer_schema plane), `dialect`/`verify`/`transforms`. ⚠️ INTEGRITY (hard fail before "
        "ANY capsule grades): your SHIPPED package under `submission/` must be SELF-CONTAINED — it must NOT "
        "`import merlin` / `from merlin` (a non-exempt package that imports the harness/reference fails the "
        "integrity scan). So COPY (vendor) the small starterkit pieces you use INTO `submission/` and import "
        "them locally (e.g. `from .oot_starterkit_vendored import parse_interface`), or reimplement them — "
        f"NEVER import them from `merlin.*` at runtime. Your OWN code is the {t}-SPECIFIC lowering.\n"
        "Consult the CCA seam menu as an authoring reference (READ "
        "`merlin/python/merlin/kernels/cca_contract.py` + `action_catalog.py` — the what-to-build "
        f"checklist). Browse the full RTL at `{t}/rtl/` to confirm a datapath/encoding detail. Do NOT run the "
        "RTL-facts GENERATORS or FileCheck directly (they need masked simulator access; FileCheck runs "
        "grader-side and returns in the verdict) — use the shipped fact files. Do NOT read golden.yaml / "
        "expected_* files — they are withheld and access is logged.")


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
# Overridable per deployment via MERLIN_OPENCODE_STALL_S; 0 disables the detector.
_STALL_SECONDS = int(os.environ.get("MERLIN_OPENCODE_STALL_S", "900"))
_POLL_SECONDS = 10
# Progress must clear this much CPU to count. opencode block-buffers stdout when it is redirected to a
# file, so a quiet round can do real work while the byte count stands still (measured: a GLM-5 round wrote
# three source files and ran selfchecks while its transcript sat at 109 bytes for 15 min). CPU is the
# signal that separates that from a hang: a process blocked on a socket burns none, a working agent burns
# plenty. Small enough that idle poll jitter cannot reach it, large enough that real work always does.
_CPU_EPSILON_S = 1.0
_CLK_TCK = os.sysconf("SC_CLK_TCK")


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
             stall_seconds: int = _STALL_SECONDS) -> tuple[int, str, str]:
    """Run ``cmd`` capturing stdout to a FILE (opencode truncates a piped stream at 64 KiB and still exits 0,
    cutting the JSON mid-stream). stdin=DEVNULL because ``run`` blocks on an open stdin pipe.

    ``cmd`` is ``bash -c '<bwrap ... opencode ...>'``, so a plain ``subprocess.run(timeout=)`` would SIGKILL
    only the outer bash and leave the bwrap->opencode grandchildren alive (bwrap ``--die-with-parent`` does
    not reliably cascade across the extra shell + PID namespace); the orphaned child keeps the inherited
    stderr pipe open, hanging the parent's own post-timeout reap (observed: a stalled GLM-5 agent sat at the
    1800s cap with the driver blocked and no result). Run in a NEW SESSION and, on timeout, SIGKILL the whole
    PROCESS GROUP so the entire tree dies and the reap returns promptly; the caller maps TimeoutExpired to a
    rc=124 round result."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".out", prefix="oc_out_", delete=False)
    tmp.close()
    errf = tempfile.NamedTemporaryFile(mode="w", suffix=".err", prefix="oc_err_", delete=False)
    errf.close()

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
            last_cpu = _tree_cpu_seconds(pgid)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _reap(p)
                    raise subprocess.TimeoutExpired(cmd, timeout)
                try:
                    p.wait(timeout=min(_POLL_SECONDS, remaining))
                    break                                   # exited on its own
                except subprocess.TimeoutExpired:
                    pass
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
                if size != last_size or cpu - last_cpu >= _CPU_EPSILON_S:
                    last_size, last_cpu, last_change = size, max(cpu, last_cpu), now
                elif stall_seconds and (now - last_change) >= stall_seconds:
                    _reap(p)
                    raise AgentStalled(cmd, int(now - last_change))
        return p.returncode, Path(tmp.name).read_text(), Path(errf.name).read_text()
    finally:
        for _f in (tmp.name, errf.name):
            try:
                os.unlink(_f)
            except OSError:
                pass


def _export_to_transcript(export: dict, mid: str, rnd: int, emit) -> None:
    """Map opencode's session export → claude-compatible ``assistant`` events (usage + content blocks) so
    experiment_tokens.parse_transcript / qa_grade / audit_transcript consume it like every other driver."""
    for msg in export.get("messages", []) or []:
        info = msg.get("info", {}) if isinstance(msg, dict) else {}
        if info.get("role") != "assistant":
            continue
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
            "content": blocks}})
        # claude-compatible tool_result events so the transcript is self-authoritative + the mask-leak
        # audit can correlate each read's result (parity with the claude-CLI path).
        if tool_results:
            emit({"type": "user", "message": {"content": tool_results}})


def _parse_run_stream(stdout: str, mid: str, rnd: int, emit) -> int:
    """Reconstruct the transcript from opencode's ``run --format json`` STDOUT stream (always captured),
    instead of a post-hoc ``opencode export`` (fragile: a wrong sandbox isolates the session data dir, so
    the export is blind and yields 0 messages — the empty-transcript bug). Each stream line is an event with
    a ``part``: a ``text`` part is an assistant text block; a ``tool`` part is a tool_use (+ its result from
    ``state.output``); ``step-finish`` parts carry token usage. Emits claude-compatible events (assistant +
    user/tool_result) so parse_transcript / audit_transcript / conformance consume them like every driver.
    Returns the number of tool calls seen."""
    tok = {"input": 0, "output": 0, "cread": 0, "cwrite": 0}
    n_tools = 0
    for line in stdout.splitlines():
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
            emit({"type": "assistant", "message": {"id": f"opencode_{rnd}_{part.get('id','')}", "model": mid,
                  "usage": {"input_tokens": 0, "output_tokens": 0}, "content": [{"type": "text", "text": part["text"]}]}})
        elif pt == "tool":
            st = part.get("state") or {}
            cid = part.get("callID") or part.get("id")
            emit({"type": "assistant", "message": {"id": f"opencode_{rnd}_{cid}", "model": mid,
                  "usage": {"input_tokens": 0, "output_tokens": 0},
                  "content": [{"type": "tool_use", "id": cid, "name": part.get("tool", "tool"),
                               "input": st.get("input", {}) if isinstance(st, dict) else {}}]}})
            n_tools += 1
            out = st.get("output") if isinstance(st, dict) else None
            if out is not None:
                emit({"type": "user", "message": {"content": [
                    {"type": "tool_result", "tool_use_id": cid,
                     "content": out if isinstance(out, str) else json.dumps(out)}]}})
        elif pt == "step-finish":
            t = part.get("tokens") or {}
            c = t.get("cache") or {}
            tok["input"] += t.get("input", 0) or 0
            tok["output"] += t.get("output", 0) or 0
            tok["cread"] += c.get("read", 0) or 0
            tok["cwrite"] += c.get("write", 0) or 0
    # one usage-bearing assistant event so token/cost accounting sees the round's totals
    emit({"type": "assistant", "message": {"id": f"opencode_{rnd}_usage", "model": mid, "content": [],
          "usage": {"input_tokens": tok["input"], "output_tokens": tok["output"],
                    "cache_read_input_tokens": tok["cread"], "cache_creation_input_tokens": tok["cwrite"]}}})
    return n_tools


def opencode_runtime_binds(data_home: Path) -> list[str]:
    """bwrap args giving opencode a WRITABLE, ISOLATED data home inside the sandbox.

    The launcher resolves through the already-bound ``~/.nvm``, but opencode keeps its state under
    ``~/.local/share/opencode``, which nothing binds -- so inside the box it starts with no data dir.
    Binding the REAL one would hand the agent ``~/.local/share/opencode/storage``: every prior opencode
    session on this host, an answer-leak surface for exactly the same reason ``~/.codex/sessions`` is.

    So the run gets its own directory and ``XDG_DATA_HOME`` points at it (verified: opencode honours the
    variable and creates ``<home>/opencode`` there). These binds are passed to ``bwrap_cmd`` as
    ``extra_binds``, which are applied BEFORE the answer-mask pass, so masking still wins.
    """
    data_home.mkdir(parents=True, exist_ok=True)
    return ["--bind", str(data_home), str(data_home),
            "--setenv", "XDG_DATA_HOME", str(data_home)]


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
        tf.write(json.dumps(obj) + "\n"); tf.flush()

    emit({"type": "system", "subtype": "init", "model": mid, "round": rnd, "driver": "opencode"})

    # opencode config: our system prompt as the primary agent + allow-all permissions (external_directory is
    # NOT covered by --auto, so it must be allowed here). Tier-within-agent: register a cheaper subagent when
    # one is configured + distinct, so opencode can delegate to it via its own subagent mechanism.
    agent_name = "merlinbench"
    cfg: dict = {
        "$schema": "https://opencode.ai/config.json",
        "agent": {agent_name: {"mode": "primary", "prompt": _system_prompt(te), "model": mid}},
        "permission": {"edit": "allow", "bash": "allow", "webfetch": "allow", "external_directory": "allow"},
    }
    sub = _MT.resolve(subagent_model) if subagent_model else _MT.resolve(_MT.NON_ANTHROPIC_TIER.subagent)
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
                                          extra_binds=opencode_runtime_binds(data_home))]
    else:
        cmd = run_cmd

    rc = 0
    try:
        code, stdout, stderr = _capture(cmd, env, timeout, str(ws))
    except subprocess.TimeoutExpired as _te:
        _why = ("agent stalled: no output for "
                f"{getattr(_te, 'timeout', '?')}s while the process stayed alive"
                if isinstance(_te, AgentStalled) else "opencode run timed out")
        emit({"type": "result", "subtype": "error", "is_error": True, "result": _why})
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
    n_tools = _parse_run_stream(stdout, mid, rnd, emit)
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
