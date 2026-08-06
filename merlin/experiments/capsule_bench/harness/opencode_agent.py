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
        "oot_starterkit (on PYTHONPATH) shows the correct patterns: read its guide (`cat "
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


def _capture(cmd: list, env: dict, timeout: int, cwd: str) -> tuple[int, str, str]:
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
    try:
        with open(tmp.name, "w") as out:
            p = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.PIPE,
                                 text=True, cwd=cwd, env=env, start_new_session=True)
            try:
                _out, stderr = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)   # whole tree: bash -> bwrap -> opencode
                except (ProcessLookupError, PermissionError):
                    p.kill()
                p.wait()
                raise
        return p.returncode, Path(tmp.name).read_text(), (stderr or "")
    finally:
        try:
            os.unlink(tmp.name)
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


def run_round(ws: Path, run_dir: Path, model: str, bundle: dict, te, sandbox: str, rnd: int,
              timeout: int, *, subagent_model: str = "", background_model: str = "",
              **_ignored) -> tuple[int, Path]:
    """Drive ONE capsule-bench round via the opencode CLI. Returns (rc, transcript_path) — the same contract
    as ``launch_agent``'s claude path."""
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
               "--dir", str(ws), "--auto", msg]

    # Same integrity wrapper as the claude path: bwrap when requested, else raw (cwd=ws). opencode is
    # Bun-based, so at --sandbox bwrap it may crash exactly like the claude binary (hence sandbox=none is the
    # harness default); at none, isolation is the copied workspace + the post-run transcript audit.
    if sandbox == "bwrap":
        inner = " ".join(shlex.quote(c) for c in run_cmd)
        cmd = ["bash", "-c", _R.bwrap_cmd(inner, ws, bundle)]
    else:
        cmd = run_cmd

    rc = 0
    try:
        code, stdout, stderr = _capture(cmd, env, timeout, str(ws))
    except subprocess.TimeoutExpired:
        emit({"type": "result", "subtype": "error", "is_error": True, "result": "opencode run timed out"})
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
