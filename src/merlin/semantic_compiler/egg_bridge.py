"""Versioned, explicit subprocess boundary to Merlin's pinned `egg` bridge."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rules import RuleProgram


@dataclass(frozen=True)
class ENode:
    symbol: str
    children: tuple[int, ...]


@dataclass(frozen=True)
class Exploration:
    roots: tuple[int, ...]
    class_by_node: tuple[int, ...]
    classes: dict[int, tuple[ENode, ...]]
    stop_reason: str
    iterations: int
    node_count: int


class EGraphUnavailable(RuntimeError):
    pass


def explore(
    program: RuleProgram,
    *,
    bridge: Path,
    iterations: int = 8,
    node_limit: int = 5000,
    wall_timeout_s: int = 60,
) -> Exploration:
    """Run only the explicitly supplied Merlin bridge; no ACT discovery/fallback."""
    if not bridge.is_file():
        raise EGraphUnavailable(f"Merlin e-graph bridge is unavailable: {bridge}")
    payload = json.dumps(program.record(iterations=iterations, node_limit=node_limit), sort_keys=True).encode()
    try:
        proc = subprocess.run(
            [str(bridge)],
            input=payload,
            capture_output=True,
            check=False,
            timeout=wall_timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EGraphUnavailable(f"Merlin e-graph bridge failed: {exc}") from exc
    if proc.returncode:
        stderr = proc.stderr.decode(errors="replace")
        raise EGraphUnavailable(f"Merlin e-graph bridge exited {proc.returncode}: {stderr}")
    try:
        result: dict[str, Any] = json.loads(proc.stdout)
        if result["schema"] != "merlin.egg_result.v1":
            raise ValueError("unexpected e-graph result schema")
        classes = {
            int(row["id"]): tuple(
                ENode(str(node["symbol"]), tuple(int(child) for child in node["children"]))
                for node in row["nodes"]
            )
            for row in result["classes"]
        }
        roots = tuple(int(root) for root in result["roots"])
        if any(root not in classes for root in roots):
            raise ValueError("e-graph result has missing roots")
        return Exploration(
            roots=roots,
            class_by_node=tuple(int(value) for value in result["class_by_node"]),
            classes=classes,
            stop_reason=str(result["stop_reason"]),
            iterations=int(result["iterations"]),
            node_count=int(result["egraph_nodes"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EGraphUnavailable(f"invalid Merlin e-graph result: {exc}") from exc
