"""Dispatch the Claude Code CLI as the agent for a generation slot.

Mirrors abc-testing's runtime: invoke ``claude -p ... --model opus`` headless (local auth, no
API key), with a cache-buster nonce so repeats don't pollute each other, capture the full
transcript, and parse the final text + token usage. The LLM proposes into a typed slot; the
deterministic gate (see kernel_slot) disposes.
"""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any


class AgentError(RuntimeError):
    pass


def run_agent(prompt: str, *, model: str = "opus", timeout: int = 900,
              workdir: str | Path | None = None, save_transcript: str | Path | None = None,
              cache_bust: bool = True) -> dict[str, Any]:
    """Run one headless Claude Code turn; return {text, usage, raw, model}."""
    if cache_bust:
        prompt = f"<!-- cache-buster: {uuid.uuid4().hex} -->\n{prompt}"
    cmd = ["claude", "-p", prompt, "--model", model, "--output-format", "json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(workdir) if workdir else None)
    raw = proc.stdout
    if save_transcript:
        Path(save_transcript).write_text(raw, encoding="utf-8")
    if proc.returncode != 0:
        raise AgentError(f"claude exited {proc.returncode}:\n{proc.stderr[-1500:]}\n{raw[-1500:]}")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
        obj = json.loads(lines[-1]) if lines else {}
    text = obj.get("result") or obj.get("text") or ""
    if not text:
        raise AgentError(f"empty agent result; raw head:\n{raw[:1500]}")
    return {"text": text, "usage": obj.get("usage", {}), "raw": raw, "model": model}


def extract_code_block(text: str, lang: str = "python") -> str:
    """Pull the first fenced code block out of the agent's reply."""
    m = re.search(rf"```{lang}\s*\n(.*?)```", text, re.S)
    if not m:
        m = re.search(r"```[a-zA-Z0-9]*\s*\n(.*?)```", text, re.S)
    if not m:
        raise AgentError("agent reply contained no fenced code block")
    return m.group(1)
