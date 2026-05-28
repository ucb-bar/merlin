#!/usr/bin/env python3
"""compile_benchmark_dispatches.py — compile each per-dispatch benchmark MLIR
to its own VMFB, keyed off the parent cell's compile.log iree-compile flags.

For each cell at build/compile_audit/<model>_<hw>/:
  - reads <cell>/benchmarks/module_..._benchmark.mlir
  - extracts the parent iree-compile command from <cell>/compile.log
  - re-invokes iree-compile on each benchmark MLIR with the same flags
    (minus dump-artifacts paths to avoid recursion)
  - writes <cell>/benchmark_vmfbs/<basename>.vmfb
  - logs success/failure to <cell>/benchmark_vmfbs/compile_summary.json
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

REPO = Path("/scratch2/agustin/merlin")
AUDIT_ROOT = REPO / "build" / "compile_audit"
IREE_COMPILE = REPO / "build" / "host-merlin-release" / "tools" / "iree-compile"


def extract_flags_from_compile_log(log_path: Path) -> list[str] | None:
    """Read the iree-compile invocation from a compile.log and return its flag list."""
    if not log_path.exists():
        return None
    txt = log_path.read_text(errors="ignore")
    # The merlin compile wrapper prints `+ /path/to/iree-compile <args> -o ...`
    m = re.search(r"^\+ \S*iree-compile\s+(.+)$", txt, re.MULTILINE)
    if not m:
        return None
    raw = m.group(1)
    # Tokenize using shlex-equivalent
    import shlex

    tokens = shlex.split(raw)
    # Strip the leading positional INPUT (the model MLIR path) and the `-o <vmfb>` pair
    # and any --dump-* flags that point to the parent cell.
    flags = []
    skip_next = False
    for i, t in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        # Drop dump-artifacts dump flags (recursive dump is wasteful)
        if t.startswith("--iree-hal-dump-executable-"):
            continue
        if t == "--dump-compilation-phases-to" or t.startswith("--dump-compilation-phases-to="):
            if "=" not in t:
                skip_next = True
            continue
        # Drop the parent INPUT path (first non-flag token)
        if i == 0 and not t.startswith("-"):
            continue
        # Drop `-o <vmfb>` pair
        if t == "-o":
            skip_next = True
            continue
        if t.startswith("-o") and "=" in t:
            continue
        # Drop the positional input MLIR (mid-args; first non-flag arg after iree-compile invocation)
        flags.append(t)
    return flags


def compile_one(bench_mlir: Path, parent_flags: list[str], out_vmfb: Path) -> dict:
    """Invoke iree-compile on bench_mlir → out_vmfb, return summary dict."""
    out_vmfb.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(IREE_COMPILE),
        str(bench_mlir),
        "-o",
        str(out_vmfb),
    ] + parent_flags
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        rc = proc.returncode
        size = out_vmfb.stat().st_size if out_vmfb.exists() else 0
        return {
            "rc": rc,
            "vmfb_bytes": size,
            "stderr_tail": proc.stderr[-400:] if rc != 0 else "",
        }
    except subprocess.TimeoutExpired:
        return {"rc": -1, "vmfb_bytes": 0, "stderr_tail": "timeout"}
    except Exception as e:
        return {"rc": -2, "vmfb_bytes": 0, "stderr_tail": str(e)[:400]}


def process_cell(cell_dir: Path) -> dict:
    """Compile every benchmark MLIR in a cell. Return summary dict."""
    bench_dir = cell_dir / "benchmarks"
    if not bench_dir.is_dir():
        return {"status": "no_benchmarks_dir", "dispatches": []}
    parent_flags = extract_flags_from_compile_log(cell_dir / "compile.log")
    if parent_flags is None:
        return {"status": "no_compile_log", "dispatches": []}

    out_dir = cell_dir / "benchmark_vmfbs"
    out_dir.mkdir(exist_ok=True)
    summary = {"status": "ok", "dispatches": [], "parent_flags": parent_flags}

    mlirs = sorted(bench_dir.glob("module_*_benchmark.mlir"))
    for m in mlirs:
        stem = m.stem  # module_<sym>_..._benchmark
        vmfb = out_dir / f"{stem}.vmfb"
        res = compile_one(m, parent_flags, vmfb)
        res["dispatch"] = stem
        summary["dispatches"].append(res)

    # Write summary
    with (out_dir / "compile_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Counts for stdout
    n_ok = sum(1 for d in summary["dispatches"] if d.get("rc") == 0)
    n_total = len(summary["dispatches"])
    summary["n_ok"] = n_ok
    summary["n_total"] = n_total
    return summary


def main() -> int:
    if not AUDIT_ROOT.exists():
        print(f"ERR: {AUDIT_ROOT} missing")
        return 1
    overall = {}
    for cell_dir in sorted(AUDIT_ROOT.iterdir()):
        if not cell_dir.is_dir():
            continue
        if cell_dir.name.endswith(".tmp"):
            continue
        print(f"--- {cell_dir.name}", flush=True)
        s = process_cell(cell_dir)
        if s.get("status") == "no_benchmarks_dir":
            print("    (no benchmarks/)")
        elif s.get("status") == "no_compile_log":
            print("    (no compile.log)")
        else:
            print(f"    {s.get('n_ok', 0)}/{s.get('n_total', 0)} OK")
        overall[cell_dir.name] = {
            "status": s.get("status"),
            "n_ok": s.get("n_ok", 0),
            "n_total": s.get("n_total", 0),
        }
    # Write top-level summary
    out_top = REPO / "tmp" / "benchmark_dispatches_summary.json"
    with out_top.open("w") as f:
        json.dump(overall, f, indent=2)
    print(f"=== overall summary: {out_top}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
