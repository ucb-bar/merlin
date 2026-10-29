"""Summarize where a test case's work goes: command classes, their counts and their bytes.

    python summarize_command_buffer.py selfcheck_out/CASE/command_buffer.json      # + sibling instruction_trace.json
    python summarize_command_buffer.py selfcheck_out/CASE/instruction_trace.json   # or a saved `isa_tools.py disasm` JSON
    python summarize_command_buffer.py debug.json        # saved `isa_tools.py debug ... > debug.json`
    python summarize_command_buffer.py response.json --case CASE   # a self-check response (dynamic counts)

Nothing about the target is assumed here. Instruction classes and their funct numbers come from the
decode the runner already wrote (``instruction_trace.json`` / ``disasm``) and are checked against the
target's own ``funct_decode_table`` in ``rtl_facts/facts.json``. Roles (readout, operand load, ...) are
any list of those class names that the target contract declares under a key. Dynamic counts and bytes
come from the timing model's own ``target.class.*`` and ``*_bytes`` counters in a self-check response.
A value that none of these sources carries is reported as unknown. Stdlib only; PyYAML is used for the
contract if it is importable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOP = 8


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_upwards(start: Path, relative: str) -> Path | None:
    for folder in (start, *start.parents):
        candidate = folder / relative
        if candidate.is_file():
            return candidate
    return None


def decode_table(facts_path: Path | None) -> dict[int, str]:
    """``{funct: class}`` from the target's RTL facts (the table the decoder itself reads)."""
    if facts_path is None or not facts_path.is_file():
        return {}
    document = _load(facts_path)
    body = document.get("facts", document) if isinstance(document, dict) else {}
    for interface in body.get("interfaces") or ():
        if isinstance(interface, dict) and interface.get("name") == "funct_decode_table":
            names = interface.get("names") or {}
            return {int(key): str(value) for key, value in names.items() if str(key).lstrip("-").isdigit()}
    return {}


def declared_roles(contract_path: Path | None, classes: set[str]) -> dict[str, list[str]]:
    """``{class: [role, ...]}``: every key whose value is a list made only of known class names."""
    if contract_path is None or not contract_path.is_file() or not classes:
        return {}
    try:
        import yaml  # noqa: PLC0415 -- optional; the workspace tools already depend on it
    except ImportError:
        return {}
    try:
        document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        return {}
    roles: dict[str, list[str]] = {}

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, (*path, str(key)))
        elif isinstance(node, list):
            if node and all(isinstance(item, str) and item in classes for item in node):
                label = "/".join(path[-2:]) if len(path) > 1 else (path[-1] if path else "?")
                for item in node:
                    roles.setdefault(item, [])
                    if label not in roles[item]:
                        roles[item].append(label)
            else:
                for item in node:
                    walk(item, path)

    walk(document, ())
    return roles


def _bits(dtype: str) -> int | None:
    digits = ""
    for char in str(dtype):
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else None


def summarize_buffer(document: dict) -> dict:
    """Commands of the abstract buffer by opcode, with the bytes of the declared tensors they name."""
    tensors = document.get("tensors") or {}
    sizes = {}
    for name, spec in tensors.items():
        shape, bits = spec.get("shape") or [], _bits(spec.get("dtype", ""))
        count = 1
        for extent in shape:
            count *= int(extent)
        sizes[name] = count * bits // 8 if bits else None
    rows: dict[str, dict] = {}
    for command in document.get("commands") or ():
        row = rows.setdefault(
            command.get("opcode", "?"), {"count": 0, "tensor_bytes": 0, "handles": set(), "unknown_bytes": False}
        )
        row["count"] += 1
        for value in (command.get("operands") or {}).values():
            if isinstance(value, str) and value in sizes:
                if sizes[value] is None:
                    row["unknown_bytes"] = True
                else:
                    row["tensor_bytes"] += sizes[value]
            elif isinstance(value, str):
                row["handles"].add(value)
    for row in rows.values():
        row["handles"] = sorted(row["handles"])
    return {
        "kind": "command_buffer",
        "commands": len(document.get("commands") or ()),
        "by_opcode": rows,
        "tensor_bytes": sizes,
    }


def summarize_trace(document: dict, table: dict[int, str], roles: dict[str, list[str]]) -> dict:
    """Static instruction sites per decoded class, cross-checked against the target's decode table."""
    counts: dict[str, int] = {}
    mismatched, functs = [], {}
    for instruction in document.get("instructions") or ():
        name = instruction.get("class", "UNKNOWN")
        counts[name] = counts.get(name, 0) + 1
        funct = instruction.get("funct")
        if isinstance(funct, int):
            functs[name] = funct
            if table and table.get(funct) != name:
                mismatched.append(
                    {"index": instruction.get("index"), "class": name, "funct": funct, "table": table.get(funct)}
                )
    summary = document.get("summary") or {}
    declared = set(table.values())
    return {
        "kind": "instruction_trace",
        "multiplicity": summary.get("counted_loops", {}).get("instruction_multiplicity", "static_sites"),
        "by_class": {
            name: {"sites": count, "funct": functs.get(name), "roles": roles.get(name, [])}
            for name, count in sorted(counts.items(), key=lambda item: -item[1])
        },
        "decode_table_checked": bool(table),
        "class_mismatches": mismatched[:20],
        "declared_never_emitted": sorted(declared - set(counts)) if declared else None,
        "counted_loops": [
            {key: loop.get(key) for key in ("header_block", "trip_count", "initial", "bound", "step")}
            for function in summary.get("counted_loops", {}).get("functions", ())
            for loop in function.get("loops", ())
        ][:TOP],
    }


def summarize_debug(document: dict) -> dict:
    """Per-command hardware state from ``isa_tools.py debug``, grouped by opcode, plus the totals."""
    groups: dict[str, dict] = {}
    rows = [row for row in document.get("per_command") or () if isinstance(row, dict)]
    for row in rows:
        group = groups.setdefault(row.get("opcode", "?"), {"count": 0})
        group["count"] += 1
        for key, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                group[key] = group.get(key, 0) + value
    metrics = document.get("metrics") or {}
    cycles = metrics.get("cycles") or sum(row.get("cycles", 0) for row in rows)
    heaviest = sorted(rows, key=lambda row: -(row.get("cycles") or 0))[:TOP]
    return {
        "kind": "debug",
        "capsule": document.get("capsule"),
        "metrics": metrics,
        "by_opcode": groups,
        "heaviest_commands": [
            {**row, "share_of_cycles": round(row.get("cycles", 0) / cycles, 3) if cycles else None} for row in heaviest
        ],
        "bytes_per_cycle": round(metrics["bytes_moved"] / cycles, 3) if cycles and metrics.get("bytes_moved") else None,
        "error": document.get("error"),
    }


def summarize_response(document: dict, case: str | None) -> dict:
    """Dynamic class counts, bytes and busiest modeled resources of one self-check test case."""
    rows = document.get("per_capsule") or document.get("cases") or []
    rows = list(rows.values()) if isinstance(rows, dict) else rows
    names = [str(row.get("capsule") or row.get("name")) for row in rows if isinstance(row, dict)]
    chosen = next(
        (row for row in rows if isinstance(row, dict) and (row.get("capsule") or row.get("name")) == case), None
    )
    if chosen is None:
        return {"kind": "selfcheck_response", "error": f"pass --case; cases in this response: {names[:40]}"}
    performance = chosen.get("performance") or {}
    features = performance.get("feature_counts") or {}
    scoped = performance.get("scoped_resource_feedback") or {}
    classes = {
        key.removeprefix("target.class."): value
        for key, value in features.items()
        if key.startswith("target.class.") and value
    }
    commands = scoped.get("commands") or {}
    compute = commands.get("compute") or sum(v for k, v in classes.items() if k.startswith("compute"))
    resources = sorted(
        (r for r in scoped.get("resources") or () if isinstance(r, dict)),
        key=lambda r: -(r.get("capacity_normalized_activity") or 0),
    )[:TOP]
    brief = next(
        (
            entry
            for entry in (document.get("performance_brief") or {}).get("largest_headroom") or ()
            if entry.get("test_case") == case
        ),
        {},
    )
    return {
        "kind": "selfcheck_response",
        "case": case,
        "status": chosen.get("status"),
        "estimated_cycles": performance.get("estimated_whole_soc_cycles"),
        "commands": commands,
        "dynamic_classes": classes,
        "per_compute_command": (
            {name: round(value / compute, 3) for name, value in classes.items()} if compute else None
        ),
        "bytes": {
            key: value
            for key, value in features.items()
            if key.endswith("_bytes") and value and key.split(".", 1)[0] in ("target", "device", "modeled")
        },
        "busiest_resources": [
            {
                "name": r.get("name"),
                "activity": round(r.get("capacity_normalized_activity") or 0, 4),
                "busy_unit_cycles": r.get("busy_unit_cycles"),
            }
            for r in resources
        ],
        "brief": {
            key: brief.get(key)
            for key in (
                "bound_by",
                "roofline_bound_cycles",
                "gap_ratio",
                "array_utilization_pct",
                "dominant_modeled_resource",
                "suggested_mechanisms",
                "inspect",
            )
            if key in brief
        },
    }


def summarize(path: Path, *, facts: Path | None = None, contract: Path | None = None, case: str | None = None) -> dict:
    document = _load(path)
    if not isinstance(document, dict):
        raise ValueError(f"{path} is not a JSON object")
    facts = facts or _find_upwards(path.resolve().parent, "rtl_facts/facts.json")
    contract = contract or _find_upwards(path.resolve().parent, "target_contract.yaml")
    table = decode_table(facts)
    if "per_command" in document or ("oracle" in document and "metrics" in document):
        return summarize_debug(document)
    if "cases" in document or "per_capsule" in document:
        return summarize_response(document, case)
    roles = declared_roles(contract, set(table.values()))
    if isinstance(document.get("instructions"), list):
        return summarize_trace(document, table, roles)
    if isinstance(document.get("commands"), list):
        result = summarize_buffer(document)
        trace = path.with_name("instruction_trace.json")
        if trace.is_file():
            result["instruction_trace"] = summarize_trace(_load(trace), table, roles)
        return result
    if document.get("error"):
        return {"kind": "tool_error", "error": document["error"]}
    raise ValueError(f"{path}: not a command buffer, instruction trace, debug output or self-check response")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path)
    ap.add_argument("--facts", type=Path, default=None, help="rtl_facts/facts.json (found upwards by default)")
    ap.add_argument("--contract", type=Path, default=None, help="target_contract.yaml (found upwards by default)")
    ap.add_argument("--case", default=None, help="test case name inside a self-check response")
    a = ap.parse_args(argv)
    try:
        print(json.dumps(summarize(a.file, facts=a.facts, contract=a.contract, case=a.case), indent=1, default=str))
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
