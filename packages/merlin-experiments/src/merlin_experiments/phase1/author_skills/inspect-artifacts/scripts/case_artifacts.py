#!/usr/bin/env python3
"""List every copy of one case's self-check artifacts with what each file proves, or diff two copies.

  python case_artifacts.py list CASE [--ws .] [--json]
  python case_artifacts.py diff DIR_A DIR_B [--json]

``list`` looks where the self-check writes (``selfcheck_out/CASE``, ``selfcheck_out/requests/RID/CASE``
and the ``model_layers`` run trees), newest first, and joins each per-request copy to its redacted
verdict row in ``.qa_channel/resp_RID.json``. ``diff`` compares two artifact directories (two
requests, or ``submission`` against a variant): file digests, command-buffer tensors and command
sequence, decoded instruction classes and counted loops, and the op histogram of the lowered IR.
Read-only: it never writes into the workspace. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

#: file name -> (kind, what it proves). Kinds: compiler output (your tool wrote it), harness decode
#: (computed from your output without running it), executed (a simulator ran your program), verdict.
KINDS = {
    "command_buffer.json": ("compiler output", "the program your command-buffer stage requested: tensors, "
                            "commands, kernel_abi. It says what was asked for, not that it ran."),
    "lowered.llvm.mlir": ("compiler output", "the target artifact your tool printed; the graded object is "
                          "translated and compiled from exactly these bytes."),
    "instruction_trace.json": ("harness decode", "a static decode of the accelerator instructions in "
                               "lowered.llvm.mlir, in text order, with counted loops. Static sites, "
                               "not an execution count."),
    "input.interface.mlir": ("harness input", "the interface program your tool received."),
    "lowered.target.mlir": ("compiler output", "your target-dialect IR (stdout of lower_interface_to_target)."),
    "numeric_report.yaml": ("executed", "per-output comparison of what a tier executed; expected values "
                            "are withheld in author-facing copies."),
    "capsule_result.json": ("verdict", "the per-case verdict: tiers, failure, numeric summary."),
    "run_manifest.yaml": ("verdict", "run identity, required tiers and per-tier status and cycles."),
}
ARTIFACTS = ("command_buffer.json", "instruction_trace.json", "lowered.llvm.mlir")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _files_root(directory: Path) -> Path:
    """A full run directory keeps its products under ``generated/``; a copied case keeps them flat."""
    return directory / "generated" if (directory / "generated").is_dir() else directory


def copies(ws: Path, case: str) -> list[dict]:
    """Every directory holding artifacts for ``case``, newest first."""
    out = ws / "selfcheck_out"
    found: list[tuple[Path, str | None, str]] = []
    if (out / case).is_dir():
        found.append((out / case, None, "latest unscoped copy (overwritten by later checks)"))
    for request in sorted((out / "requests").glob("*")) if (out / "requests").is_dir() else ():
        if (request / case).is_dir():
            found.append((request / case, request.name, "per-request copy"))
    for run in sorted(out.glob("**/model_layers/runs/*/" + case)):
        rid = run.relative_to(out).parts[1] if run.relative_to(out).parts[0] == "requests" else None
        found.append((run, rid, "model-layer run directory (functional tier only)"))
    rows = []
    for directory, rid, scope in found:
        root = _files_root(directory)
        files = [p for p in sorted(root.iterdir()) if p.is_file()] if root.is_dir() else []
        if directory != root:
            files += [p for p in sorted(directory.iterdir()) if p.is_file()]
        mtime = max((p.stat().st_mtime for p in files), default=0.0)
        rows.append({
            "path": str(directory.relative_to(ws)) if directory.is_relative_to(ws) else str(directory),
            "request_id": rid, "scope": scope, "mtime": mtime,
            "files": [{"name": p.name, "bytes": p.stat().st_size,
                       "kind": KINDS.get(p.name, ("other", ""))[0],
                       "proves": KINDS.get(p.name, ("other", ""))[1]} for p in files],
            "verdict": verdict_row(ws, rid, case) if rid else None,
        })
    return sorted(rows, key=lambda row: row["mtime"], reverse=True)


def verdict_row(ws: Path, rid: str, case: str) -> dict | None:
    """The redacted per-case row of request ``rid``, trimmed to what locates a failure."""
    response = _load_json(ws / ".qa_channel" / f"resp_{rid}.json")
    if not isinstance(response, dict):
        return None
    for row in response.get("per_capsule") or ():
        if isinstance(row, dict) and row.get("capsule") == case:
            keep = ("pass", "tiers", "barrier_tier", "barrier_status", "barrier_cycles", "declined")
            out = {key: row[key] for key in keep if key in row}
            failure = row.get("failure")
            if isinstance(failure, dict):
                out["failure"] = {k: str(v)[:300] for k, v in failure.items()}
            numeric = row.get("numeric") or {}
            out["numeric"] = {k: numeric[k] for k in ("status", "mismatch_count", "max_abs_diff", "first_mismatch")
                              if k in numeric}
            check = row.get("trace_check") or {}
            if check:
                out["trace_check"] = {k: check[k] for k in ("status", "violations") if k in check}
            out["submission"] = response.get("selfcheck_submission")
            return out
    return None


def op_histogram(text: str) -> Counter:
    """Count operation names: the first dotted token of each line, after any ``results =``."""
    counts: Counter = Counter()
    for line in text.splitlines():
        body = line.strip()
        if not body or body.startswith(("^", "}", "//", "#")):
            continue
        head, eq, rest = body.partition(" = ")
        if eq and head.startswith("%"):
            body = rest
        token = body.split(" ", 1)[0].split("(", 1)[0].strip('"')
        if "." in token and token[:1].isalpha():
            counts[token] += 1
    return counts


def functions(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        _, marker, rest = line.partition("llvm.func ")
        if marker and "@" in rest:
            names.append(rest.partition("@")[2].split("(", 1)[0].strip())
    return names


def _counter_delta(a: Counter, b: Counter) -> dict:
    return {key: [a.get(key, 0), b.get(key, 0)] for key in sorted(set(a) | set(b)) if a.get(key, 0) != b.get(key, 0)}


def diff(dir_a: Path, dir_b: Path) -> dict:
    a, b = _files_root(dir_a), _files_root(dir_b)
    report: dict = {"a": str(dir_a), "b": str(dir_b), "files": {}}
    for name in ARTIFACTS:
        pa, pb = a / name, b / name
        report["files"][name] = ("missing in a" if not pa.is_file() else "missing in b" if not pb.is_file()
                                 else "identical" if _sha(pa) == _sha(pb) else "differs")
    cba, cbb = _load_json(a / "command_buffer.json"), _load_json(b / "command_buffer.json")
    if isinstance(cba, dict) and isinstance(cbb, dict):
        ta, tb = cba.get("tensors") or {}, cbb.get("tensors") or {}
        report["tensors"] = {name: [ta.get(name), tb.get(name)] for name in sorted(set(ta) | set(tb))
                             if ta.get(name) != tb.get(name)}
        ca, cb = list(cba.get("commands") or ()), list(cbb.get("commands") or ())
        ops_a = Counter(str(c.get("opcode")) for c in ca if isinstance(c, dict))
        ops_b = Counter(str(c.get("opcode")) for c in cb if isinstance(c, dict))
        first = next((i for i, (x, y) in enumerate(zip(ca, cb)) if x != y), None)
        if first is None and len(ca) != len(cb):
            first = min(len(ca), len(cb))
        report["commands"] = {"count": [len(ca), len(cb)], "opcode_delta": _counter_delta(ops_a, ops_b),
                              "first_difference": first,
                              "at_first": [ca[first] if first is not None and first < len(ca) else None,
                                           cb[first] if first is not None and first < len(cb) else None]}
    tra, trb = _load_json(a / "instruction_trace.json"), _load_json(b / "instruction_trace.json")
    if isinstance(tra, dict) and isinstance(trb, dict):
        sa, sb = tra.get("summary") or {}, trb.get("summary") or {}
        report["trace"] = {
            "instructions": [len(tra.get("instructions") or ()), len(trb.get("instructions") or ())],
            "class_delta": _counter_delta(Counter(sa.get("class_histogram") or {}),
                                          Counter(sb.get("class_histogram") or {})),
            "loops": [_loops(sa), _loops(sb)]}
    la, lb = a / "lowered.llvm.mlir", b / "lowered.llvm.mlir"
    if la.is_file() and lb.is_file():
        xa, xb = la.read_text(encoding="utf-8"), lb.read_text(encoding="utf-8")
        report["lowered"] = {"lines": [len(xa.splitlines()), len(xb.splitlines())],
                             "functions": [functions(xa), functions(xb)],
                             "op_delta": _counter_delta(op_histogram(xa), op_histogram(xb))}
    return report


def _loops(summary: dict) -> list[str]:
    rows = []
    for fn in (summary.get("counted_loops") or {}).get("functions") or ():
        for loop in fn.get("loops") or ():
            rows.append(f"{fn.get('function')}:bb{loop.get('header_block')}x{loop.get('trip_count')}")
    return rows


def _print_list(rows: list[dict], limit: int) -> None:
    if not rows:
        print("no artifacts found for this case; run a self-check that includes it")
        return
    names = sorted({f["name"] for row in rows for f in row["files"]})
    print("what each file proves:")
    for name in names:
        kind, proves = KINDS.get(name, ("other", "not described here"))
        print(f"  {name:<24} {kind}: {proves}")
    shown = rows[:limit] if limit > 0 else rows
    for row in shown:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(row["mtime"]))
        print(f"== {row['path']}  [{row['scope']}, {stamp}]")
        print("   " + "  ".join(f"{f['name']} ({f['bytes']} B)" for f in row["files"]))
        if row["verdict"] is not None:
            print("   verdict: " + json.dumps(row["verdict"], sort_keys=True))
    if len(shown) < len(rows):
        print(f"... {len(rows) - len(shown)} older copies (raise --limit or pass 0 for all)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    lp = sub.add_parser("list", help="every copy of one case's artifacts, newest first")
    lp.add_argument("case")
    lp.add_argument("--ws", default=".", help="workspace root (default: current directory)")
    lp.add_argument("--json", action="store_true")
    lp.add_argument("--limit", type=int, default=5, help="copies to print (0 = all; --json prints all)")
    dp = sub.add_parser("diff", help="compare two artifact directories")
    dp.add_argument("a")
    dp.add_argument("b")
    dp.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        rows = copies(Path(args.ws).resolve(), args.case)
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            _print_list(rows, args.limit)
        return 0 if rows else 1
    report = diff(Path(args.a), Path(args.b))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
