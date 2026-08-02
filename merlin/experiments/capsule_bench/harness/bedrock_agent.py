"""Non-Anthropic agent backend for the capsule-bench QA loop, via the Bedrock Converse API.

The `claude` CLI (the default agent) speaks the Anthropic Messages API ONLY, so a non-Anthropic Bedrock
model (glm5 / qwen-coder / nemotron / ...) cannot drive it. This module is a drop-in alternative for one
ROUND: it runs an agentic Converse loop whose tools (`run_bash` / `write_file` / `read_file`) execute
INSIDE the SAME masked bwrap sandbox the claude agent uses (`merlin.targetgen.sandbox.bwrap.wrap`), so
answer-masking + integrity are identical. It produces the agent's `submission/` in the workspace and
writes a claude-stream-json-COMPATIBLE transcript (`assistant` events with `usage` + `tool_use` blocks),
so the driver's grading (`qa_grade`), token/cost accounting (`experiment_tokens.parse_transcript`) and
answer-access audit all work unchanged — the result is directly comparable to a claude run.

`launch_agent` routes to `run_round` when the model is a non-Anthropic alias/id (:func:`is_converse_model`).
Between rounds the driver persists the redacted verdict to `ws/qa/verdict.json`; this backend feeds that
back into the model's prompt so it iterates on real oracle feedback.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# Non-Anthropic Bedrock models usable as the agent (mirror of chia's registry; the driver venv has no
# chia). Anything NOT here is treated as an Anthropic id and left to the claude CLI.
_CONVERSE_MODELS = {
    "glm5": "zai.glm-5", "glm4.7": "zai.glm-4.7",
    "qwen-coder": "qwen.qwen3-coder-next",
    "nemotron": "nvidia.nemotron-super-3-120b",
    "kimi": "moonshotai.kimi-k2.5", "deepseek": "deepseek.v3.2",
    "nova-pro": "us.amazon.nova-pro-v1:0", "nova-lite": "us.amazon.nova-lite-v1:0",
}


def resolve(model: str) -> str:
    return _CONVERSE_MODELS.get(model, model)


def is_converse_model(model: str) -> bool:
    """True iff `model` is a non-Anthropic Bedrock model this backend drives (alias or raw id)."""
    if model in _CONVERSE_MODELS:
        return True
    return "anthropic" not in model.lower() and "." in model and model not in ("opus", "sonnet", "haiku")


_TOOLS = {"tools": [
    {"toolSpec": {"name": "run_bash", "description":
        "Run a bash command in your isolated workspace (cwd = workspace root). For build/test/inspect — "
        "NOT for writing source files (use write_file). Returns stdout+stderr.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "command": {"type": "string"}}, "required": ["command"]}}}},
    {"toolSpec": {"name": "write_file", "description":
        "Create/overwrite a text file at a path relative to the workspace root (e.g. "
        "submission/mlir_oot/atlas_opt.py). Content written verbatim — no shell escaping. Use for all "
        "source files.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}}},
    {"toolSpec": {"name": "read_file", "description":
        "Read a text file (relative to the workspace root, or an allowed contract path). Answer files "
        "are masked.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}}},
    {"toolSpec": {"name": "self_check", "description":
        "Grade your CURRENT submission/ against the REAL target oracle and get REDACTED feedback: per-tier "
        "pass/fail, decoded trace, instruction counts, and NUMERIC DIFF STATS (how far off your output is) "
        "— everything EXCEPT the withheld golden values. This is your only mid-round signal for whether the "
        "numerics are correct; call it after each build and iterate until capsules pass. `capsules` = "
        "comma-separated ids or 'all'.",
        "inputSchema": {"json": {"type": "object", "properties": {
            "capsules": {"type": "string"}}, "required": []}}}},
]}

# The delegate tool (tier-within-agent). Only added to the tool list when a distinct, cheaper subagent
# model is configured — it lets the strong primary offload well-scoped mechanical sub-tasks to the cheap
# model (the faithful Claude-Code Task->subagent pattern; there is no per-turn auto-routing).
_DELEGATE_TOOL = {"toolSpec": {"name": "delegate", "description":
    "Delegate a well-scoped, self-contained sub-task to a cheaper/faster model to save your own turns "
    "(a Task subagent). Give a precise `subtask` plus the minimal `context` it needs (paths, the encoding "
    "facts, the exact file to write). The subagent shares your workspace with write_file/read_file/run_bash "
    "but has NO self_check and cannot delegate further; it returns its result text. Use it for mechanical "
    "work (boilerplate, scaffolding, reading+summarizing a fact file) and keep the hard reasoning yourself.",
    "inputSchema": {"json": {"type": "object", "properties": {
        "subtask": {"type": "string"}, "context": {"type": "string"}}, "required": ["subtask"]}}}}

# Bedrock prompt-caching marker. Placed at the end of the STABLE prefix (system prompt + tool schemas +
# the first user message's task/ISA-facts), so the large static context is billed once and reused across
# the ~120 tool-iterations of a round instead of re-shipped uncached every call (the glm5f cached=0 waste).
_CACHE_POINT = {"cachePoint": {"type": "default"}}


def _cache_unsupported(exc: Exception) -> bool:
    """True iff a Converse error looks like the model/account rejecting cachePoint — so we self-correct to an
    uncached request rather than maintaining a per-model support allowlist (Bedrock caching support varies by
    model AND account, and changes over time). Must match the REAL AWS AccessDenied text: "You invoked an
    unsupported model or your request did not allow prompt caching." Note the word is "caching", so we stem
    to "cach" — matching on "cache" would silently MISS it (the 5th letter differs) and a non-Nova round
    would then die on its first call instead of falling back."""
    s = f"{type(exc).__name__} {exc}".lower()
    return ("cachepoint" in s or "prompt cach" in s
            or ("cach" in s and ("support" in s or "invalid" in s or "not valid" in s or "not allow" in s)))


def _strip_cachepoints(messages: list) -> None:
    """Remove every cachePoint block from message content (used when falling back to an uncached request)."""
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = [b for b in c if not (isinstance(b, dict) and "cachePoint" in b)]


def _roll_cachepoint(messages: list) -> None:
    """Move a single ROLLING cachePoint to the end of the growing conversation, so the accumulating
    tool-result history is cached incrementally too — mirroring Claude Code's rolling cache breakpoint on
    top of the static system/tools/first-message breakpoints. Without this we cache only the static header
    and re-ship the (large, growing) tool-result body uncached every iteration. Keeps total breakpoints at
    the Bedrock limit of 4 (system + tools + messages[0] + this rolling one) by first removing any prior
    rolling marker from messages[1:], leaving messages[0]'s static breakpoint intact."""
    for m in messages[1:]:
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = [b for b in c if not (isinstance(b, dict) and "cachePoint" in b)]
    if len(messages) > 1:
        last = messages[-1].get("content")
        if isinstance(last, list):
            last.append(dict(_CACHE_POINT))


def _bash_in_sandbox(te, ws: Path, bundle: dict, command: str, sandbox: str, timeout: int) -> str:
    from merlin.targetgen.sandbox import bwrap as _BW
    cmd = _BW.wrap(te, ws, command, bundle) if sandbox == "bwrap" else f"cd {ws} && {command}"
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout, cwd=str(ws))
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr else "")
    except subprocess.TimeoutExpired:
        out = f"[command timed out after {timeout}s]"
    return out[:6000]


def _write_file(ws: Path, rel: str, content: str) -> str:
    p = (ws / rel).resolve()
    if not str(p).startswith(str(ws.resolve())):
        return f"[refused] path escapes the workspace: {rel}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {rel}"


def _run_subagent(cli, sub_mid: str, ws: Path, te, bundle: dict, sandbox: str, subtask: str,
                  context: str, emit, rnd: int, tag: str, cmd_timeout: int, deadline: float,
                  *, max_iters: int = 15, max_tokens: int = 6000) -> str:
    """Run a bounded agentic sub-loop on the cheaper ``sub_mid`` model for a delegated sub-task. Shares the
    workspace + in-sandbox tool execution (so answer masking still holds); NO self_check and NO nested
    delegate. Emits assistant events tagged with ``sub_mid`` (and ``subagent: True``) so telemetry
    attributes the tokens to the delegate tier. Returns the subagent's final result text."""
    sub_tools = {"tools": [t for t in _TOOLS["tools"]
                           if t["toolSpec"]["name"] in ("run_bash", "write_file", "read_file")]}
    sys = [{"text": "You are a focused sub-agent. Do EXACTLY the delegated sub-task and nothing more, then "
            "reply with a short result summary. You have write_file/read_file/run_bash in the shared "
            "workspace (cwd = workspace root). Do NOT read golden/expected files (masked + logged)."}]
    msgs = [{"role": "user", "content": [
        {"text": f"SUBTASK:\n{subtask}\n\nCONTEXT:\n{context}".strip()}, _CACHE_POINT]}]
    final: list[str] = []
    sub_cache = True   # cache the sub-agent's system+tools+first-message prefix too (parity with the main
    #                    loop); self-correct to uncached if this cheaper model rejects cachePoint.
    for it in range(max_iters):
        if time.time() > deadline:
            break
        if sub_cache:
            _roll_cachepoint(msgs)
        _sys = sys + [_CACHE_POINT] if sub_cache else sys
        _tc = {"tools": sub_tools["tools"] + [_CACHE_POINT]} if sub_cache else sub_tools
        try:
            resp = cli.converse(modelId=sub_mid, system=_sys, messages=msgs, toolConfig=_tc,
                                inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
        except Exception as e:  # noqa: BLE001 — a delegate failure is a tool result, not a round failure
            if sub_cache and _cache_unsupported(e):
                sub_cache = False
                _strip_cachepoints(msgs)
                continue
            return f"[delegate error: {type(e).__name__}: {str(e)[:200]}]"
        u = resp.get("usage", {})
        out_msg = resp["output"]["message"]
        blocks = []
        for c in out_msg.get("content", []):
            if "text" in c:
                blocks.append({"type": "text", "text": c["text"]}); final.append(c["text"])
            elif "toolUse" in c:
                tu = c["toolUse"]
                blocks.append({"type": "tool_use", "name": tu["name"], "input": tu.get("input", {})})
        emit({"type": "assistant", "subagent": True, "message": {
            "id": f"bedrock_{rnd}_{tag}", "model": sub_mid,
            "usage": {"input_tokens": u.get("inputTokens", 0),
                      "output_tokens": u.get("outputTokens", 0),
                      "cache_read_input_tokens": u.get("cacheReadInputTokens", 0),
                      "cache_creation_input_tokens": u.get("cacheWriteInputTokens", 0)},
            "content": blocks}})
        msgs.append(out_msg)
        tus = [c["toolUse"] for c in out_msg.get("content", []) if "toolUse" in c]
        if resp.get("stopReason") != "tool_use" or not tus:
            break
        results = []
        for tu in tus:
            name, inp = tu["name"], tu.get("input", {})
            if name == "write_file":
                output = _write_file(ws, inp.get("path", ""), inp.get("content", ""))
            elif name == "read_file":
                output = _bash_in_sandbox(te, ws, bundle, f'cat "{inp.get("path", "")}"', sandbox, cmd_timeout)
            else:
                output = _bash_in_sandbox(te, ws, bundle, inp.get("command", ""), sandbox, cmd_timeout)
            results.append({"toolResult": {"toolUseId": tu["toolUseId"],
                                            "content": [{"text": output or "(no output)"}]}})
        msgs.append({"role": "user", "content": results})
    return ("\n".join(final))[:4000] or "(subagent produced no text)"


def run_round(ws: Path, run_dir: Path, model: str, bundle: dict, te, sandbox: str, rnd: int,
              timeout: int, *, subagent_model: str = "", background_model: str = "",
              max_iters: int = 120, cmd_timeout: int = 120,
              max_tokens: int = 8000) -> tuple[int, Path]:
    """Drive ONE capsule-bench round with a non-Anthropic model. Returns (rc, transcript_path) — the same
    contract as ``launch_agent``'s claude path. Writes a claude-compatible transcript so the driver grades
    + accounts it identically."""
    import boto3
    import model_tiers as _MT
    mid = resolve(model)
    # Tier-within-agent: a distinct, cheaper subagent enables the `delegate` tool (the Claude-Code Task
    # pattern). Default the delegate to the non-Anthropic subagent tier when unset; disable it if it would
    # equal the primary (delegating to yourself is pointless). background_model is accepted for API/telemetry
    # symmetry but the Converse loop generates no background chores (titles/summaries) to route to it.
    sub_mid = _MT.resolve(subagent_model) if subagent_model else _MT.resolve(_MT.NON_ANTHROPIC_TIER.subagent)
    delegate_enabled = bool(sub_mid) and sub_mid != mid
    round_tools = _TOOLS["tools"] + ([_DELEGATE_TOOL] if delegate_enabled else [])
    tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tf = open(tpath, "w")

    def emit(obj: dict) -> None:
        tf.write(json.dumps(obj) + "\n"); tf.flush()

    task = (ws / "TASK.md").read_text() if (ws / "TASK.md").is_file() else ""
    # Cross-round memory: prefer the harness-built round brief (progress log across all graded rounds +
    # the agent's own iteration_notes.md + a stale-notes nudge); fall back to the raw redacted verdict.
    brief_p = ws / "qa" / "round_brief.md"
    verdict_p = ws / "qa" / "verdict.json"
    feedback = ""
    if brief_p.is_file():
        feedback = "\n\n" + brief_p.read_text()[:8000] + "\n"
    elif verdict_p.is_file():
        feedback = ("\n\n## Previous round's official grader verdict (iterate to fix these)\n```json\n"
                    + verdict_p.read_text()[:4000] + "\n```\n")
    system = [{"text":
        "You are an autonomous compiler engineer with a LIMITED number of tool turns. Build the target "
        "backend under submission/ (manifest.yaml + the entrypoint tool + supporting modules). Look at the "
        "contract briefly, then START WRITING with write_file — do not over-explore. Get "
        "submission/manifest.yaml + the tool existing and PARSING as early as possible. "
        "MANIFEST FORMAT: the `entrypoints.tool` field must be the BARE script path relative to the "
        "package (e.g. `mlir_oot/atlas_opt.py`) and each command's argv should reference it as `{tool}` — "
        "do NOT put `python3` into the tool field or bake the interpreter/path into argv, or the build "
        "fails with `tool missing`. The harness runs a Python tool through the interpreter for you. "
        "CRITICAL: you get NO golden values, so use the self_check TOOL after every build — it grades your "
        "current submission against the real oracle and returns per-capsule pass/fail plus the NUMERIC "
        "DIFF (how far off each capsule is), goldens withheld. That is your primary correctness signal: "
        "call self_check, read the diff, fix your encoding/lowering, rebuild, self_check again — iterate "
        "until capsules pass. Between rounds you also receive the official grader verdict, which for THIS "
        "arm carries an advisory `rtl_checks` block (FileCheck over your emitted MLIR + the decoded trace, "
        "with RTL-derived bounds) — read it to catch structural/encoding mistakes the numeric diff can't "
        "localize. For grounding, READ (do not try to regenerate) the shipped ISA spec — the green-card and "
        "isa_definition — mounted read-only in your workspace at `" + te.target + "/isa_include/` (the "
        "hwbringup set is mounted as `" + te.target + "/`, with the RTL under `" + te.target + "/rtl/` "
        "and a worked example under `" + te.target + "/example_kernel/`); derive EVERY opcode, encoding, "
        "field-layout and the command-buffer schema from them with read_file — never invent an encoding. "
        "MANDATORY — BUILD ON MERLIN'S FRAMEWORK, DO NOT REINVENT IT. This is the `merlin_assisted` arm: "
        "your backend MUST be built on the granted oot_starterkit (on PYTHONPATH), NOT hand-rolled. Read "
        "its guide first (`cat oot_starterkit/AGENT.md` and `cat oot_starterkit/scaffold/README.md` — both "
        "mounted at your workspace root), then import and USE it: `from "
        "merlin.targetgen.oot_starterkit import parse_interface, CommandBufferBuilder, transforms, verify` "
        "(also `dialect`). parse_interface (iface.py) parses the fixed merlin_iface grammar — do NOT write "
        "your own interface parser. CommandBufferBuilder (cmdbuf.py) emits a SCHEMA-VALID "
        "command_buffer.json by construction — use it so you PASS the command_buffer_schema plane instead "
        "of hand-assembling JSON that fails it. dialect.py gives you the TYPED, VERIFIED xDSL IR and "
        "verify.py is the structural verifier — call them. transforms.py holds generic compiler passes. "
        "Writing your own parser/command-buffer/dialect is WRONG for this arm and is how you get "
        "parse/command_buffer_schema failures; your OWN code is only the atlas-SPECIFIC lowering that "
        "sits on top of these framework pieces. Also browse the full RTL repo mounted at `" + te.target +
        "/rtl/` (23 files) when you need to confirm a datapath/encoding detail. "
        "Use the CCA seam menu too (`from merlin.kernels import cca_contract, action_catalog` — "
        "cca_contract.check_bijection('" + te.target + "') / action_catalog.escalation_ladder) as the "
        "what-to-build checklist. Do NOT run the RTL-facts GENERATORS or FileCheck directly — the generators need "
        "RTL/simulator access that is masked in your sandbox, and FileCheck runs grader-side (its results "
        "come back to you in the verdict); use the shipped fact files instead. Do NOT attempt to read "
        "golden.yaml / expected_* files — they are withheld and access is logged."}]
    if delegate_enabled:
        system[0]["text"] += (
            " You ALSO have a `delegate` tool wired to a cheaper/faster sub-agent. DEFAULT TO DELEGATING "
            "well-scoped MECHANICAL work rather than spending your own limited turns on it: writing "
            "boilerplate/scaffolding, emitting a bulk `.word` listing from encodings you already know, or "
            "reading + summarizing a long fact/RTL file. Rule of thumb: before you hand-write more than "
            "~20 lines of mechanical output or read a large file end-to-end, delegate it with a precise "
            "`subtask` plus the minimal `context` (paths + the exact encoding facts it needs). Keep ONLY "
            "the hard reasoning — encoding/lowering choices and the halt/terminator sequence — for yourself.")
    messages = [{"role": "user", "content": [{"text": task + feedback +
                 "\n\nYour workspace is the current directory. Begin now."}, _CACHE_POINT]}]

    from botocore.config import Config as _BotoConfig
    # A large generation can exceed boto3's default 60s read timeout mid-stream (observed:
    # ReadTimeoutError truncating a round). Give converse a generous read timeout + SDK-level retries so
    # a transient network/throttle blip doesn't end an otherwise-productive round.
    cli = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                       config=_BotoConfig(read_timeout=300, connect_timeout=15,
                                          retries={"max_attempts": 5, "mode": "adaptive"}))
    emit({"type": "system", "subtype": "init", "model": mid, "round": rnd,
          "delegate_enabled": delegate_enabled, "subagent_model": sub_mid if delegate_enabled else None})
    deadline = time.time() + timeout
    rc = 0
    cache_enabled = True   # attempt Bedrock prompt caching; self-correct to uncached if the model rejects it

    def _transient(exc: Exception) -> bool:
        s = f"{type(exc).__name__} {exc}".lower()
        return any(w in s for w in ("readtimeout", "connect", "timeout", "throttl",
                                    "too many", "503", "500", "serviceunavailable"))
    try:
        for it in range(max_iters):
            if time.time() > deadline:
                rc = 124
                break
            resp = None
            if cache_enabled:
                _roll_cachepoint(messages)   # cache the accumulating tool-result body, not just the header
            for attempt in range(4):  # extra loop-level retries on transient errors the SDK didn't absorb
                if time.time() > deadline:
                    rc = 124
                    break
                try:
                    _sys = system + [_CACHE_POINT] if cache_enabled else system
                    _tc = ({"tools": round_tools + [_CACHE_POINT]} if cache_enabled
                           else {"tools": round_tools})
                    resp = cli.converse(modelId=mid, system=_sys, messages=messages, toolConfig=_tc,
                                        inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
                    break
                except Exception as e:  # noqa: BLE001
                    if cache_enabled and _cache_unsupported(e):
                        # Model rejects Bedrock prompt caching — drop the cachePoints and retry uncached
                        # (self-correcting, so there is no per-model support allowlist to maintain).
                        cache_enabled = False
                        _strip_cachepoints(messages)
                        emit({"type": "system", "subtype": "cache_disabled",
                              "error": f"{type(e).__name__}: {str(e)[:160]}"})
                        continue
                    if _transient(e) and attempt < 3:
                        emit({"type": "system", "subtype": "retry", "attempt": attempt + 1,
                              "error": f"{type(e).__name__}: {str(e)[:160]}"})
                        time.sleep(min(30, 5 * (attempt + 1)))
                        continue
                    emit({"type": "result", "subtype": "error", "is_error": True,
                          "result": f"{type(e).__name__}: {str(e)[:300]}"})
                    rc = 1
                    break
            if resp is None:  # deadline hit or non-recoverable error during the retry loop
                break
            u = resp.get("usage", {})
            out_msg = resp["output"]["message"]
            # claude-compatible assistant event (usage + content blocks) for parse_transcript + audit.
            blocks = []
            for c in out_msg.get("content", []):
                if "text" in c:
                    blocks.append({"type": "text", "text": c["text"]})
                elif "toolUse" in c:
                    tu = c["toolUse"]
                    blocks.append({"type": "tool_use", "name": tu["name"], "input": tu.get("input", {})})
            emit({"type": "assistant", "message": {
                "id": f"bedrock_{rnd}_{it}", "model": mid,
                "usage": {"input_tokens": u.get("inputTokens", 0),
                          "output_tokens": u.get("outputTokens", 0),
                          "cache_read_input_tokens": u.get("cacheReadInputTokens", 0),
                          "cache_creation_input_tokens": u.get("cacheWriteInputTokens", 0)},
                "content": blocks}})
            messages.append(out_msg)
            tool_uses = [c["toolUse"] for c in out_msg.get("content", []) if "toolUse" in c]
            if resp.get("stopReason") != "tool_use" or not tool_uses:
                emit({"type": "result", "subtype": "success", "is_error": False})
                break
            results = []
            for tu in tool_uses:
                name, inp = tu["name"], tu.get("input", {})
                if name == "write_file":
                    output = _write_file(ws, inp.get("path", ""), inp.get("content", ""))
                elif name == "read_file":
                    output = _bash_in_sandbox(te, ws, bundle, f'cat "{inp.get("path", "")}"',
                                              sandbox, cmd_timeout)
                elif name == "self_check":
                    caps = (inp.get("capsules") or "all").strip() or "all"
                    output = _bash_in_sandbox(
                        te, ws, bundle,
                        f'python3 agent_selfcheck.py --submission submission --sim spike '
                        f'--capsules "{caps}" --timeout 600', sandbox, max(cmd_timeout, 700))
                elif name == "delegate":
                    output = _run_subagent(cli, sub_mid, ws, te, bundle, sandbox,
                                           inp.get("subtask", ""), inp.get("context", ""), emit,
                                           rnd, f"{it}_deleg", cmd_timeout, deadline)
                else:
                    output = _bash_in_sandbox(te, ws, bundle, inp.get("command", ""), sandbox, cmd_timeout)
                results.append({"toolResult": {"toolUseId": tu["toolUseId"],
                                                "content": [{"text": output or "(no output)"}]}})
            messages.append({"role": "user", "content": results})
    finally:
        tf.close()
    return rc, tpath
