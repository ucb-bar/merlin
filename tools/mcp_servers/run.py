"""Run MCP tools — typed wrappers around `./merlin run`.

Depends on:
- tools/run/cli.py (mode dispatch + the `_MODE_TO_SCRIPT` dict)
- tools/run/*.py (each mode script's output format — the stdout parsers
  here look for makespan / per-instance / hash patterns)

If `run/cli.py:_MODE_TO_SCRIPT` adds/removes a mode, update `_RUN_MODES`
here. If a mode script changes its stdout format, update the relevant
regex in `_parse_run_stdout`.

Parses common post-run signals (makespan, per-instance latencies, dispatch
counts, hash) from each mode's stdout so Claude doesn't re-scrape the same
fields on every call.
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

_RUN_MODES = ("schedule", "multi-device", "het-e2e", "het-matrix", "full-loop", "roundtrip")


class ToolError(RuntimeError):
    pass


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], dict[str, Any]]


# ---- shared parsers -----------------------------------------------------

# Common patterns the run scripts emit (case-insensitive).
_MAKESPAN_RE = re.compile(r"(makespan|total\s+wall|aggregate\s+wall)[^\d]*([\d.]+)\s*(ms|us|s)", re.IGNORECASE)
_INSTANCE_LATENCY_RE = re.compile(r"^\s*(?P<job>[\w.\-]+)\s*[:=]\s*(?P<ms>[\d.]+)\s*ms", re.MULTILINE)
_HASH_RE = re.compile(r"(?:output\s+)?(?:md5|hash)\s*[:=]\s*(?P<hash>[0-9a-fA-F]{16,})", re.IGNORECASE)
_DISPATCH_COUNT_RE = re.compile(r"(\d+)\s+dispatch(?:es)?", re.IGNORECASE)


def _normalize_ms(value: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "us":
        return value / 1000.0
    if unit == "s":
        return value * 1000.0
    return value  # already ms


def _parse_run_stdout(stdout: str) -> dict[str, Any]:
    makespan_ms: float | None = None
    m = _MAKESPAN_RE.search(stdout)
    if m:
        makespan_ms = _normalize_ms(float(m.group(2)), m.group(3))
    per_instance = {m.group("job"): float(m.group("ms")) for m in _INSTANCE_LATENCY_RE.finditer(stdout)}
    hashes = [m.group("hash") for m in _HASH_RE.finditer(stdout)]
    dispatch_counts = [int(m.group(1)) for m in _DISPATCH_COUNT_RE.finditer(stdout)]
    return {
        "makespan_ms": makespan_ms,
        "per_instance_ms": per_instance or None,
        "hashes": hashes or None,
        "dispatch_counts_seen": dispatch_counts or None,
    }


# ---- tools --------------------------------------------------------------


def _list_run_modes(args: dict[str, Any]) -> dict[str, Any]:
    file_map = {
        "schedule": "schedule.py",
        "multi-device": "multi_device.py",
        "het-e2e": "het_e2e.py",
        "het-matrix": "het_matrix.py",
        "full-loop": "full_loop.py",
        "roundtrip": "roundtrip.py",
    }
    modes = []
    for mode, fname in file_map.items():
        py = REPO_ROOT / "tools" / "run" / fname
        modes.append(
            {
                "name": mode,
                "script": str(py.relative_to(REPO_ROOT)),
                "exists": py.exists(),
            }
        )
    return {"modes": modes}


def _run_help(args: dict[str, Any]) -> dict[str, Any]:
    mode = args.get("mode")
    cmd = [str(MERLIN), "run"]
    if mode:
        if mode not in _RUN_MODES:
            raise ToolError(f"unknown mode {mode!r}; known: {list(_RUN_MODES)}")
        cmd.extend([mode, "--help"])
    else:
        cmd.append("--help")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    return {"exit_code": result.returncode, "help": result.stdout, "stderr": result.stderr}


def _execute_run(args: dict[str, Any]) -> dict[str, Any]:
    """Run `./merlin run <mode> <args...>` and return parsed structure.

    Return shape:
        {
          "passed": bool,
          "exit_code": int,
          "mode": str,
          "command": [str, ...],
          "makespan_ms": float | null,
          "per_instance_ms": {job: ms, ...} | null,
          "hashes": [str, ...] | null,
          "stdout_tail": [str, ...]    # always — for transparency
          "stderr_tail": [str, ...]    # only on non-zero exit
        }
    """
    mode = args.get("mode")
    if mode not in _RUN_MODES:
        raise ToolError(f"missing or invalid mode; known: {list(_RUN_MODES)}")
    passthrough = args.get("passthrough", [])
    if not isinstance(passthrough, list):
        raise ToolError("passthrough must be a list of strings")
    cmd = [str(MERLIN), "run", mode, *[str(a) for a in passthrough]]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    parsed = _parse_run_stdout(result.stdout)
    out: dict[str, Any] = {
        "passed": result.returncode == 0,
        "exit_code": result.returncode,
        "mode": mode,
        "command": cmd,
        "stdout_tail": result.stdout.splitlines()[-30:],
        **parsed,
    }
    if result.returncode != 0:
        out["stderr_tail"] = result.stderr.splitlines()[-30:]
    return out


TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        name="run_list_modes",
        description=(
            "List the available `./merlin run` modes (schedule, multi-device, "
            "het-e2e, het-matrix, full-loop, roundtrip). Use when asked 'what "
            "board-execution flows exist?'."
        ),
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_list_run_modes,
    ),
    ToolDefinition(
        name="run_help",
        description=(
            "Get `./merlin run <mode> --help`. Use to discover the flags a "
            "specific mode accepts before calling execute_run."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": list(_RUN_MODES),
                    "description": "Run mode; omit for top-level help.",
                },
            },
            "additionalProperties": False,
        },
        handler=_run_help,
    ),
    ToolDefinition(
        name="execute_run",
        description=(
            "Execute one of the `./merlin run` modes. Returns parsed structure: "
            "exit code, makespan_ms (if reported), per_instance_ms (per job:ms), "
            "hashes (any md5/hash lines found), plus stdout_tail. Use for 'run "
            "<model> on <board>', 'execute schedule', 'profile on hardware', "
            "'round-trip a compiled model'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": list(_RUN_MODES), "description": "Which run mode to invoke."},
                "passthrough": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Arguments forwarded to the underlying script.",
                },
            },
            "required": ["mode"],
            "additionalProperties": False,
        },
        handler=_execute_run,
    ),
]


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_REGISTRY)


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for tool in TOOL_REGISTRY:
        if tool.name == name:
            return tool.handler(arguments)
    raise ToolError(f"unknown tool: {name!r}; known: {[t.name for t in TOOL_REGISTRY]}")
