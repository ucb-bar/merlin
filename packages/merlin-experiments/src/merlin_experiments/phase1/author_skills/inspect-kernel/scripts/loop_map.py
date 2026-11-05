#!/usr/bin/env python3
"""Rank the loops of one case's lowered kernel by the accelerator instructions they issue, and name their owners.

  python loop_map.py selfcheck_out/CASE [--top 10] [--json]

Reads the three files a self-check returns for a case. ``instruction_trace.json`` gives the counted
loops (``summary.counted_loops``: header/latch block positions, body blocks, trip counts) and one
decoded class per accelerator instruction site in text order. ``lowered.llvm.mlir`` gives, per block,
the inline-asm sites and the ownership attributes your emitter wrote (``merlin.global_task``,
``merlin.source_op_index``, ``prov.region_id``). Block ``N`` is the N-th block of its function (the
entry block is 0), which is how both files number blocks.

For each loop it reports the trip count, nesting depth, the static sites in its blocks, and a
static estimate of issued instructions: each site weighted by the product of the trip counts of
every loop containing its block. The tasks in a loop are joined to
``params.global_program_plan.tasks`` of ``command_buffer.json`` when a plan is present, otherwise to
``commands[N]`` (only meaningful if your compiler numbers tasks by command). Read-only; stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ATTRS_INT = ("merlin.global_task", "merlin.source_op_index")
REGION = "prov.region_id"


def _outside_strings(line: str) -> str:
    """The line with every double-quoted string blanked, so braces inside asm text do not count."""
    out, quoted, escaped = [], False, False
    for ch in line:
        if quoted:
            out.append(" ")
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def _values(line: str, key: str) -> list[str]:
    found, start = [], 0
    needle = key + " = "
    while True:
        at = line.find(needle, start)
        if at == -1:
            return found
        rest = line[at + len(needle):]
        if rest.startswith('"'):
            found.append(rest[1:].partition('"')[0])
        else:
            found.append(rest.split(" ", 1)[0].rstrip(",}"))
        start = at + len(needle)


def parse_lowered(text: str) -> list[dict]:
    """Functions with their blocks, in order; each block counts ops, asm sites and owner attributes."""
    functions: list[dict] = []
    current, depth, base = None, 0, 0
    for line in text.splitlines():
        body = line.strip()
        bare = _outside_strings(body)
        if current is None and (body.startswith("llvm.func ") or body.startswith('"llvm.func"')) and "@" in body:
            name = body.partition("@")[2].split("(", 1)[0].strip()
            current = {"function": name, "blocks": [_block(0)]}
            functions.append(current)
            base = depth
            if "{" not in bare:  # a declaration has no body
                current = None
        elif current is not None and body.startswith("^"):
            current["blocks"].append(_block(len(current["blocks"])))
        elif current is not None and body and not body.startswith("}"):
            block = current["blocks"][-1]
            block["ops"] += 1
            op = body.partition(" = ")[2] if body.startswith("%") and " = " in body else body
            if op.split("(", 1)[0].strip('"').endswith("inline_asm"):
                block["asm_sites"] += 1
            for key in ATTRS_INT:
                for value in _values(body, key):
                    block[key][value] += 1
            for value in _values(body, REGION):
                block[REGION][value] += 1
        depth += bare.count("{") - bare.count("}")
        if current is not None and depth <= base and body.startswith("}"):
            current = None
    return functions


def _block(index: int) -> dict:
    return {"index": index, "ops": 0, "asm_sites": 0,
            **{key: Counter() for key in (*ATTRS_INT, REGION)}}


def loop_map(case_dir: Path) -> dict:
    lowered = parse_lowered((case_dir / "lowered.llvm.mlir").read_text(encoding="utf-8"))
    trace = json.loads((case_dir / "instruction_trace.json").read_text(encoding="utf-8"))
    buffer_path = case_dir / "command_buffer.json"
    buffer = json.loads(buffer_path.read_text(encoding="utf-8")) if buffer_path.is_file() else {}
    plan = ((buffer.get("params") or {}).get("global_program_plan") or {}).get("tasks")
    commands = buffer.get("commands") or []
    classes = [str(i.get("class")) for i in trace.get("instructions") or ()]
    n_sites = sum(b["asm_sites"] for f in lowered for b in f["blocks"])
    site_classes = iter(classes if len(classes) == n_sites else ["?"] * n_sites)
    notes = [] if len(classes) == n_sites else [
        f"{len(classes)} decoded instructions vs {n_sites} inline-asm sites: classes not joined to sites"]
    for f in lowered:
        for b in f["blocks"]:
            b["classes"] = Counter(next(site_classes) for _ in range(b["asm_sites"]))
    loops_by_fn = {fn.get("function"): fn for fn in (trace.get("summary") or {}).get("counted_loops", {}).get(
        "functions") or ()}
    rows, totals = [], Counter()
    for f in lowered:
        record = loops_by_fn.get(f["function"]) or {}
        loops = [dict(loop, blocks={loop["header_block"], *loop.get("body_blocks", ())})
                 for loop in record.get("loops") or ()]
        unavailable = record.get("unavailable") or []
        if unavailable:
            notes.append(f"{f['function']}: loops without a counted trip: {unavailable}")

        def weight(index: int) -> int:
            product = 1
            for loop in loops:
                if index in loop["blocks"]:
                    product *= int(loop.get("trip_count") or 1)
            return product

        for b in f["blocks"]:
            for cls, n in b["classes"].items():
                totals[cls] += n * weight(b["index"])
        for loop in loops:
            members = [b for b in f["blocks"] if b["index"] in loop["blocks"]]
            weighted = Counter()
            for b in members:
                for cls, n in b["classes"].items():
                    weighted[cls] += n * weight(b["index"])
            tasks = Counter()
            for b in members:
                tasks.update(b["merlin.global_task"])
            rows.append({
                "function": f["function"], "header": loop["header_block"], "latch": loop.get("latch_block"),
                "trip_count": loop.get("trip_count"),
                "depth": sum(1 for other in loops if other is not loop and loop["header_block"] in other["blocks"]),
                "static_sites": sum(b["asm_sites"] for b in members),
                "issued_estimate": sum(weighted.values()), "issued_by_class": dict(weighted.most_common()),
                "tasks": {t: _owner(t, plan, commands) for t in sorted(tasks, key=lambda t: -tasks[t])},
                "regions": sorted({r for b in members for r in b[REGION]}),
                "source_ops": sorted({s for b in members for s in b["merlin.source_op_index"]}, key=_int),
            })
    rows.sort(key=lambda r: (-r["issued_estimate"], r["function"], r["header"]))
    return {"loops": rows, "issued_by_class": dict(totals.most_common()), "notes": notes,
            "basis": "static sites x enclosing trip counts; not a measured execution count"}


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return 1 << 30


def _owner(task: str, plan, commands) -> str:
    index = _int(task)
    if isinstance(plan, list):
        for row in plan:
            if isinstance(row, dict) and row.get("task_index") == index:
                return f"plan task {index}: {row.get('kind', '')} ops {row.get('source_op_indices', '')}".strip()
        return f"task {index}: not in the plan"
    if 0 <= index < len(commands) and isinstance(commands[index], dict):
        return f"commands[{index}] {commands[index].get('opcode')}"
    return f"task {index}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("case_dir", help="directory with lowered.llvm.mlir and instruction_trace.json")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    report = loop_map(Path(args.case_dir))
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"issued (estimate) by class: {report['issued_by_class']}")
    for row in report["loops"][: args.top]:
        print(f"{row['function']} ^bb{row['header']} trip={row['trip_count']} depth={row['depth']} "
              f"sites={row['static_sites']} issued~{row['issued_estimate']} {row['issued_by_class']}")
        print(f"    tasks: {row['tasks']}  regions: {row['regions'][:4]}")
    for note in report["notes"]:
        print("note:", note)
    print(f"({report['basis']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
