"""Extract public-safe abstraction-use witnesses from completed author commands.

These are structural shell-request witnesses, not dynamic API invocation counts.
Executable argv and executable Python fragments use the Phase 1 conformance
scanner; facts/levers/scaffolds retain its output-corroboration requirements.
ISA tools use resolved argv rather than the legacy substring-based check.

Categories overlap. An AST call site need not have executed, and a successful
outer shell does not prove that every composed subcommand succeeded. Missing
witnesses do not prove non-use. A use/progress timeline is observational evidence,
not a causal ablation. No raw command, output or source-file path is exported;
``agent_path`` must be the caller's public actor identity, not a filesystem path.
"""

from __future__ import annotations

import ast
import shlex
from pathlib import Path
from typing import Any

from merlin_experiments.phase1 import conformance


def _isa_actions(call: conformance.ToolCall) -> set[str]:
    actions = set()
    for exe, argv in conformance._resolved_commands(call.input["command"]):
        if exe in {"python", "python3"}:
            if len(argv) < 2 or Path(argv[0]).name not in {"isa_tools", "isa_tools.py"}:
                continue
            argv = argv[1:]
        elif exe not in {"isa_tools", "isa_tools.py"}:
            continue
        if argv and argv[0] in {"asm", "disasm", "lint", "debug"}:
            actions.add(argv[0])
    return actions


def _direct_facts_read(call: conformance.ToolCall) -> bool:
    """Keep direct frozen-facts inspection separate from the load_facts API."""
    for fragment in conformance._python_fragments(call):
        try:
            tree = ast.parse(fragment)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "open"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                continue
            path = Path(node.args[0].value)
            mode = node.args[1] if len(node.args) > 1 else None
            mode = next((kw.value for kw in node.keywords if kw.arg == "mode"), mode)
            read_mode = mode is None or (isinstance(mode, ast.Constant) and mode.value in {"r", "rb"})
            if path.name == "facts.json" and "rtl_facts" in path.parts and read_mode:
                return True
    for exe, argv in conformance._resolved_commands(call.input["command"]):
        if exe in {"cat", "head", "tail", "sed", "jq"} and any(
            Path(arg).name == "facts.json" and "rtl_facts" in Path(arg).parts for arg in argv
        ):
            return True
    return False


def abstraction_markers(metrics: dict[str, Any], agent_path: str) -> list[dict[str, Any]]:
    """Return category/start/end/outer-success/actor for retained completed commands.

    ``metrics`` is a per-thread metrics record from ``author_agents.threads``;
    the caller must exclude inherited parent history before passing it here.
    Failed outer commands remain visible and can be filtered by the consumer.
    Times are retained Unix wall seconds, not CPU time or provider thinking time.
    """
    markers = []
    for command in metrics.get("commands", []):
        item = command["item"]
        raw = item.get("command")
        text = shlex.join(raw) if isinstance(raw, list) else str(raw or "")
        succeeded = item.get("exit_code") == 0 and item.get("status") != "failed"
        call = conformance.ToolCall(
            "Bash", {"command": text}, str(item.get("id") or ""), True,
            succeeded, str(item.get("aggregated_output") or ""),
        )
        evidence = {
            "cca_bijection": conformance._cca_evidence(
                call, script="cca_contract.py", subcommand="check-bijection", api="check_bijection",
            ),
            "cca_escalation_ladder": conformance._cca_evidence(
                call, script="action_catalog.py", subcommand="escalation-ladder", api="escalation_ladder",
            ),
            "derived_levers": conformance._derived_levers_evidence(call),
            "rtl_facts": conformance._rtl_facts_evidence(call),
            "scaffold": conformance._scaffold_generator_evidence(call),
            "direct_facts_file_inspection": _direct_facts_read(call),
        }
        actions = _isa_actions(call)
        evidence.update({f"isa_{kind}": kind in actions for kind in ("asm", "disasm", "lint", "debug")})
        markers.extend(
            {
                "category": category,
                "start": command["start"],
                "end": command["end"],
                "outer_command_succeeded": succeeded,
                "agent_path": agent_path,
            }
            for category, present in evidence.items() if present
        )
    return markers
