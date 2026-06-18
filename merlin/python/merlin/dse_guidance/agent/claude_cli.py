"""Dispatch the Claude Code CLI headless as an agent slot (self-contained).

Mirrors ``merlin/targetgen/agent/claude_cli.py``: invoke ``claude -p ... --output-format json``
with a cache-buster nonce, capture the transcript, return the final text. Kept self-contained so
dse_guidance does not depend on the (uncommitted) targetgen package. Local auth, no API key.

This is the *propose* side. The *dispose* side (a deterministic gate) lives in the calling slot;
the agent's text is never trusted for numbers — only for interpretation, and even then it is gated.
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable


class AgentError(RuntimeError):
    pass


def run_agent(prompt: str, *, model: str = "opus", timeout: int = 600,
              workdir: str | Path | None = None, cache_bust: bool = True) -> dict[str, Any]:
    """Run one headless Claude Code turn; return {text, usage, raw}. Raises AgentError on failure
    (including when the ``claude`` CLI is not installed — callers treat that as 'agent unavailable')."""
    if cache_bust:
        prompt = f"<!-- nonce: {uuid.uuid4().hex} -->\n{prompt}"
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              cwd=str(workdir) if workdir else None)
    except FileNotFoundError as e:
        raise AgentError("the `claude` CLI is not on PATH (agent unavailable)") from e
    except subprocess.TimeoutExpired as e:
        raise AgentError(f"claude timed out after {timeout}s") from e
    if proc.returncode != 0:
        raise AgentError(f"claude exited {proc.returncode}:\n{proc.stderr[-1200:]}")
    raw = proc.stdout
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        obj = json.loads(lines[-1]) if lines else {}
    text = obj.get("result") or obj.get("text") or ""
    if not text:
        raise AgentError(f"empty agent result; raw head:\n{raw[:1000]}")
    return {"text": text, "usage": obj.get("usage", {}), "raw": raw}


def extract_json(text: str) -> Any:
    """Pull the first fenced ```json block (or a bare JSON array/object) out of an agent reply."""
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    blob = m.group(1) if m else text
    blob = blob.strip()
    if not blob.startswith(("[", "{")):
        b = re.search(r"(\[.*\]|\{.*\})", blob, re.S)
        if not b:
            raise AgentError("agent reply contained no JSON")
        blob = b.group(1)
    try:
        return json.loads(blob)
    except json.JSONDecodeError as e:
        raise AgentError(f"agent JSON did not parse: {e}") from e


Runner = Callable[[str], dict]
