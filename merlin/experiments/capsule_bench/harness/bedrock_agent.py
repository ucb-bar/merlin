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


def run_round(ws: Path, run_dir: Path, model: str, bundle: dict, te, sandbox: str, rnd: int,
              timeout: int, *, max_iters: int = 120, cmd_timeout: int = 120,
              max_tokens: int = 8000) -> tuple[int, Path]:
    """Drive ONE capsule-bench round with a non-Anthropic model. Returns (rc, transcript_path) — the same
    contract as ``launch_agent``'s claude path. Writes a claude-compatible transcript so the driver grades
    + accounts it identically."""
    import boto3
    mid = resolve(model)
    tpath = run_dir / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
    tpath.parent.mkdir(parents=True, exist_ok=True)
    tf = open(tpath, "w")

    def emit(obj: dict) -> None:
        tf.write(json.dumps(obj) + "\n"); tf.flush()

    task = (ws / "TASK.md").read_text() if (ws / "TASK.md").is_file() else ""
    verdict_p = ws / "qa" / "verdict.json"
    feedback = ""
    if verdict_p.is_file():
        feedback = ("\n\n## Previous round's official grader verdict (iterate to fix these)\n```json\n"
                    + verdict_p.read_text()[:4000] + "\n```\n")
    system = [{"text":
        "You are an autonomous compiler engineer with a LIMITED number of tool turns. Build the target "
        "backend under submission/ (manifest.yaml + the entrypoint tool + supporting modules). Look at the "
        "contract briefly, then START WRITING with write_file — do not over-explore. Get "
        "submission/manifest.yaml + the tool existing and PARSING as early as possible. "
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
        "You may also import the granted in-tree tools from your OWN python (via write_file + run_bash), "
        "which are on PYTHONPATH: the CCA seam menu (`from merlin.kernels import cca_contract, "
        "action_catalog` — cca_contract.check_bijection('" + te.target + "') / "
        "action_catalog.escalation_ladder) and the oot_starterkit (`from merlin.targetgen.oot_starterkit "
        "import parse_interface, CommandBufferBuilder, transforms`) to build and self-verify command "
        "buffers. Do NOT run the RTL-facts GENERATORS or FileCheck directly — the generators need "
        "RTL/simulator access that is masked in your sandbox, and FileCheck runs grader-side (its results "
        "come back to you in the verdict); use the shipped fact files instead. Do NOT attempt to read "
        "golden.yaml / expected_* files — they are withheld and access is logged."}]
    messages = [{"role": "user", "content": [{"text": task + feedback +
                 "\n\nYour workspace is the current directory. Begin now."}]}]

    from botocore.config import Config as _BotoConfig
    # A large generation can exceed boto3's default 60s read timeout mid-stream (observed:
    # ReadTimeoutError truncating a round). Give converse a generous read timeout + SDK-level retries so
    # a transient network/throttle blip doesn't end an otherwise-productive round.
    cli = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"),
                       config=_BotoConfig(read_timeout=300, connect_timeout=15,
                                          retries={"max_attempts": 5, "mode": "adaptive"}))
    emit({"type": "system", "subtype": "init", "model": mid, "round": rnd})
    deadline = time.time() + timeout
    rc = 0

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
            for attempt in range(4):  # extra loop-level retries on transient errors the SDK didn't absorb
                if time.time() > deadline:
                    rc = 124
                    break
                try:
                    resp = cli.converse(modelId=mid, system=system, messages=messages, toolConfig=_TOOLS,
                                        inferenceConfig={"maxTokens": max_tokens, "temperature": 0})
                    break
                except Exception as e:  # noqa: BLE001
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
                else:
                    output = _bash_in_sandbox(te, ws, bundle, inp.get("command", ""), sandbox, cmd_timeout)
                results.append({"toolResult": {"toolUseId": tu["toolUseId"],
                                                "content": [{"text": output or "(no output)"}]}})
            messages.append({"role": "user", "content": results})
    finally:
        tf.close()
    return rc, tpath
