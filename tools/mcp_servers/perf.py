"""Perf MCP tools — typed wrappers around `./merlin perf-decompose`.

Depends on:
- tools/perf/cli.py (argparse: --topk, --csv flags)
- tools/perf/decompose.py (CSV column schema: ord, wg_count, cycles,
  pct, kind, sym)

If perf-decompose's CSV columns change, update the column reads in
`_parse_csv`. If new flags are added, update `_perf_decompose`'s schema.

Returns parsed structured data (top-K hot dispatches, bucket-by-kind totals,
total cycles) by default. The wrapper invokes `./merlin perf-decompose` with
`--csv` to a temp file, parses the CSV, and returns structured JSON.
"""

from __future__ import annotations

import csv
import pathlib
import subprocess
import sys
import tempfile
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


def _parse_csv(csv_path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "ord": int(row["ord"]),
                    "wg_count": int(row["wg_count"]),
                    "cycles": int(row["cycles"]),
                    "pct": float(row["pct"]),
                    "kind": row["kind"],
                    "sym": row["sym"],
                }
            )
    return rows


def _perf_decompose(args: dict[str, Any]) -> dict[str, Any]:
    """Run `./merlin perf-decompose <uartlog>` and return parsed structure.

    Return shape:
        {
          "passed": bool,
          "exit_code": int,
          "total_cycles": int,
          "n_dispatches": int,
          "top_k_hot": [{"ord", "wg_count", "cycles", "pct", "kind", "sym"}, ...],
          "by_kind": {"matmul": {"cycles", "wg_count", "n_dispatches"}, ...},
          "csv_path": str,
          "stderr_tail": [str, ...]  # only on failure
        }
    """
    uartlog = args.get("uartlog")
    if not isinstance(uartlog, str) or not uartlog:
        raise ToolError("missing required arg 'uartlog' (path to FireSim uartlog).")
    topk = int(args.get("topk", 20))
    csv_out = args.get("csv")

    if csv_out:
        csv_path = pathlib.Path(csv_out)
    else:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            csv_path = pathlib.Path(tmp.name)

    cmd = [str(MERLIN), "perf-decompose", uartlog, "--topk", str(topk), "--csv", str(csv_path)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)

    rows = _parse_csv(csv_path)
    by_kind: dict[str, dict[str, float]] = {}
    for r in rows:
        b = by_kind.setdefault(r["kind"], {"cycles": 0, "wg_count": 0, "n_dispatches": 0})
        b["cycles"] += r["cycles"]
        b["wg_count"] += r["wg_count"]
        b["n_dispatches"] += 1
    total_cycles = sum(r["cycles"] for r in rows)
    top_k_hot = sorted(rows, key=lambda r: -r["cycles"])[:topk]

    out: dict[str, Any] = {
        "passed": result.returncode == 0 and bool(rows),
        "exit_code": result.returncode,
        "total_cycles": total_cycles,
        "n_dispatches": len(rows),
        "top_k_hot": top_k_hot,
        "by_kind": by_kind,
        "csv_path": str(csv_path),
        "command": cmd,
    }
    if not rows:
        out["stderr_tail"] = result.stderr.splitlines()[-10:]
        out["stdout_tail"] = result.stdout.splitlines()[-10:]
    return out


def _perf_help(args: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [str(MERLIN), "perf-decompose", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return {"exit_code": result.returncode, "help": result.stdout}


TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        name="perf_decompose",
        description=(
            "Decode a FireSim uartlog into a per-dispatch performance "
            "breakdown. Returns parsed structure: total_cycles, top-K hot "
            "dispatches (ord/cycles/pct/kind/sym), bucket-by-kind totals. "
            "Use for 'where is <model> spending time', 'hot dispatches', "
            "'profile breakdown'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "uartlog": {"type": "string", "description": "Path to FireSim uartlog file."},
                "topk": {"type": "integer", "description": "Top-K dispatches to return (default 20)."},
                "csv": {"type": "string", "description": "Optional persistent CSV output path."},
            },
            "required": ["uartlog"],
            "additionalProperties": False,
        },
        handler=_perf_decompose,
    ),
    ToolDefinition(
        name="perf_help",
        description="Verbatim `./merlin perf-decompose --help`.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler=_perf_help,
    ),
]


def list_tool_definitions() -> list[ToolDefinition]:
    return list(TOOL_REGISTRY)


def dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    for tool in TOOL_REGISTRY:
        if tool.name == name:
            return tool.handler(arguments)
    raise ToolError(f"unknown tool: {name!r}; known: {[t.name for t in TOOL_REGISTRY]}")
