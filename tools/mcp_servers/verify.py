"""Verify MCP tools — typed wrappers around `./merlin verify-output`.

Depends on:
- tools/verify/cli.py (argparse: --vmfb, --onnx flags)
- tools/verify/cli.py (stdout format: "vmfb hash:", "golden hash:",
  "PASS"/"FAIL", "max diff:" patterns — parsed by `_parse_verify_output`)

If verify-output's stdout format changes, update the regex in
`_parse_verify_output`. If new CLI flags are added, update the schema.

Returns parsed structure: pass/fail, hashes, divergence summary. Avoids
making Claude re-parse the verify-output text on every call.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOLS_DIR))
import utils  # noqa: E402

REPO_ROOT = utils.REPO_ROOT
MERLIN = REPO_ROOT / "merlin"


class ToolError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


# Parse the verify-output stdout. Format (depending on version):
#   "vmfb hash: 0xabc123..."
#   "golden hash: 0xdef456..."
#   "PASS" or "FAIL"
_HASH_RE = re.compile(r"(?P<source>\w+)\s+hash\s*[:=]\s*(?P<hash>0x[0-9a-fA-F]+|[0-9a-fA-F]{8,})")
_PASS_RE = re.compile(r"\bPASS(ED)?\b", re.IGNORECASE)
_FAIL_RE = re.compile(r"\bFAIL(ED)?\b", re.IGNORECASE)
_DIFF_RE = re.compile(r"max\s+(?:abs\s+)?(?:diff|error|delta)\s*[:=]\s*(?P<delta>[0-9.eE+\-]+)", re.IGNORECASE)


def _parse_verify_output(stdout: str) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for m in _HASH_RE.finditer(stdout):
        hashes[m.group("source").lower()] = m.group("hash")
    pass_hit = _PASS_RE.search(stdout)
    fail_hit = _FAIL_RE.search(stdout)
    deltas = [float(m.group("delta")) for m in _DIFF_RE.finditer(stdout)]
    return {
        "hashes": hashes,
        "hash_match": (None if len(hashes) < 2 else len({h.lower() for h in hashes.values()}) == 1),
        "explicit_pass": bool(pass_hit) and not bool(fail_hit),
        "explicit_fail": bool(fail_hit),
        "max_diff": max(deltas) if deltas else None,
    }


def _verify_output(args: dict[str, Any]) -> dict[str, Any]:
    """Run `./merlin verify-output` and return parsed structure.

    Return shape:
        {
          "passed": bool,
          "exit_code": int,
          "hashes": {"vmfb": "...", "golden": "..."},  # parsed from stdout
          "hash_match": bool | null,
          "max_diff": float | null,
          "command": [str, ...],
          "stdout": str,           # always included for transparency
          "stderr_tail": [str, ...] # only on failure
        }
    """
    vmfb = args.get("vmfb")
    onnx = args.get("onnx")
    extra_flags = args.get("extra_flags") or []
    if not isinstance(extra_flags, list):
        raise ToolError("extra_flags must be a list of strings")
    if not vmfb and not onnx:
        raise ToolError("provide at least one of 'vmfb' or 'onnx'.")

    cmd = [str(MERLIN), "verify-output"]
    if vmfb:
        cmd.extend(["--vmfb", str(vmfb)])
    if onnx:
        cmd.extend(["--onnx", str(onnx)])
    cmd.extend(str(f) for f in extra_flags)

    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)

    parsed = _parse_verify_output(result.stdout)
    out = {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "command": cmd,
        "stdout": result.stdout,
        **parsed,
    }
    if result.returncode != 0:
        out["stderr_tail"] = result.stderr.splitlines()[-20:]
    return out


def _verify_help(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(MERLIN), "verify-output", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"exit_code": result.returncode, "help": result.stdout}


TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        name="verify_output",
        description=(
            "Cross-hash compare a VMFB (or ONNX) output against an onnxruntime "
            "golden. Returns parsed structure: passed (bool), hashes by source, "
            "hash_match (bool), max_diff (float if reported). On failure also "
            "stderr_tail. Use for 'does <model> produce the right output', "
            "'verify <model>', or 'does this match golden'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "vmfb": {"type": "string", "description": "Path to the compiled VMFB."},
                "onnx": {"type": "string", "description": "Path to source ONNX (for golden generation)."},
                "extra_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Extra raw flags forwarded to ./merlin verify-output.",
                },
            },
            "additionalProperties": False,
        },
        handler=_verify_output,
    ),
    ToolDefinition(
        name="verify_help",
        description="Verbatim `./merlin verify-output --help`.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_verify_help,
    ),
]


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_REGISTRY)


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for tool in TOOL_REGISTRY:
        if tool.name == name:
            return tool.handler(arguments)
    raise ToolError(f"unknown tool: {name!r}; known: {[t.name for t in TOOL_REGISTRY]}")
