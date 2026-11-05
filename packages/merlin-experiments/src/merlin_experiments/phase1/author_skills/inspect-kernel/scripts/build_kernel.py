#!/usr/bin/env python3
"""Rebuild one case's kernel object from its lowered IR and inspect it: stack frame, symbols, disassembly.

  python build_kernel.py selfcheck_out/CASE/lowered.llvm.mlir [--out DIR] [--also OTHER.o]...

The self-check returns ``lowered.llvm.mlir`` but not the object the grader compiled from it. This
script repeats the grader's object step in a scratch directory: ``mlir-translate --mlir-to-llvmir``
to ``kernel.ll``, then ``clang -c -fstack-usage`` with the triple and the ``-march/-mabi/-mcmodel/-O``
flags read from the ``toolchain`` block of ``target_contract.yaml`` (the block that also holds
``kernel_stack_frame``). It then applies the stack-frame preflight to ``kernel.su`` exactly as the
grader does (every row ``static``, exactly one row for the entry symbol, frame within
``max_static_bytes``), lists defined and undefined symbols with ``llvm-nm``, flags external symbols
defined twice across this object and any ``--also`` objects, and writes ``kernel.dis``
(``llvm-objdump -d -r``) with per-function instruction counts.

Tools are found on PATH, then under ``WS/install/bin``; ``--toolchain DIR`` overrides both.
``--triple``, ``--cflag``, ``--entry`` and ``--max-stack`` override what the contract says. Outputs go
to ``--out`` (default ``$TMPDIR/kernel-build/NAME``); nothing is written beside the input.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

KEPT_FLAG_PREFIXES = ("-march=", "-mabi=", "-mcmodel=", "-O")
KEPT_FLAGS = ("-ffreestanding", "-fno-builtin")
ALLOCATIONS = ("static", "dynamic", "dynamic,bounded")


def find_toolchain_block(node):
    """The first mapping that holds ``kernel_stack_frame`` (the object build recipe), searched depth-first."""
    if isinstance(node, dict):
        if isinstance(node.get("kernel_stack_frame"), dict):
            return node
        for value in node.values():
            hit = find_toolchain_block(value)
            if hit is not None:
                return hit
    elif isinstance(node, list):
        for value in node:
            hit = find_toolchain_block(value)
            if hit is not None:
                return hit
    return None


def recipe_from_contract(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    block = find_toolchain_block(yaml.safe_load(path.read_text(encoding="utf-8"))) or {}
    flags = [f for f in block.get("cflags") or () if str(f).startswith(KEPT_FLAG_PREFIXES) or f in KEPT_FLAGS]
    compiler = Path(str((block.get("compiler") or {}).get("path") or ""))
    # A cross compiler is named TRIPLE-gcc / TRIPLE-clang: the triple is everything before the last dash.
    triple = compiler.name.rpartition("-")[0] if "-" in compiler.name else ""
    frame = block.get("kernel_stack_frame") or {}
    return {"cflags": flags, "triple": triple, "entry": frame.get("entry_symbol"),
            "max_stack": frame.get("max_static_bytes")}


def parse_stack_usage(text: str) -> list[dict]:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"stack-usage line {number} does not have three tab-separated fields")
        identity, size, allocation = fields
        source, sep, function = identity.rpartition(":")
        if not sep or not size.isdecimal() or allocation not in ALLOCATIONS:
            raise ValueError(f"stack-usage line {number} is malformed: {line!r}")
        rows.append({"source": source, "function": function, "bytes": int(size), "allocation": allocation})
    if not rows:
        raise ValueError("stack-usage report is empty")
    return rows


def stack_verdict(rows: list[dict], *, entry: str, max_stack: int | None, source_name: str = "kernel.ll") -> dict:
    """The grader's preflight rule over parsed rows: a list of refusals, empty when it would pass."""
    problems = [f"row for {r['function']!r} names source {r['source']!r}, expected {source_name!r}"
                for r in rows if Path(r["source"]).name != source_name]
    dynamic = [r["function"] for r in rows if r["allocation"] != "static"]
    if dynamic:
        problems.append("dynamic allocation for: " + ", ".join(dynamic))
    entries = [r for r in rows if r["function"] == entry]
    if len(entries) != 1:
        problems.append(f"{len(entries)} rows for entrypoint {entry!r}; exactly one is required")
    frame = entries[0]["bytes"] if len(entries) == 1 else None
    if frame is not None and isinstance(max_stack, int) and frame > max_stack:
        problems.append(f"entrypoint {entry!r} needs {frame} static stack bytes, over the {max_stack}-byte "
                        f"budget by {frame - max_stack}")
    return {"entry": entry, "frame_bytes": frame, "max_static_bytes": max_stack,
            "headroom_bytes": None if frame is None or not isinstance(max_stack, int) else max_stack - frame,
            "rows": sorted(rows, key=lambda r: -r["bytes"]), "refusals": problems}


def parse_nm(text: str) -> dict:
    """``llvm-nm`` lines (``VALUE TYPE NAME`` or ``TYPE NAME`` for undefined) -> defined/undefined names."""
    defined, undefined = {}, []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("U", "w", "v"):
            undefined.append(parts[1])
        elif len(parts) >= 3:
            defined[parts[-1]] = parts[-2]
    return {"defined": defined, "undefined": sorted(undefined),
            "external": sorted(name for name, kind in defined.items() if kind.isupper())}


def function_sizes(disassembly: str) -> dict:
    """Instruction count per function from ``llvm-objdump -d`` text (``ADDR <name>:`` headers)."""
    counts: dict = {}
    current = None
    for line in disassembly.splitlines():
        stripped = line.strip()
        if stripped.endswith(">:") and "<" in stripped:
            label = stripped.rpartition("<")[2][:-2]
            if not label.startswith(".L"):  # assembler-local labels stay inside their function
                current = label
                counts.setdefault(current, 0)
        elif current is not None and ":" in stripped:
            address, _, rest = stripped.partition(":")
            if address.strip().isalnum() and rest.strip() and not rest.strip().startswith("R_"):  # skip relocations
                counts[current] += 1
    return counts


def _tool(names: tuple[str, ...], toolchain: Path | None, ws: Path) -> str:
    for name in names:
        for directory in ([toolchain] if toolchain else []):
            if (directory / name).is_file():
                return str(directory / name)
        found = shutil.which(name)
        if found:
            return found
        if (ws / "install" / "bin" / name).is_file():
            return str(ws / "install" / "bin" / name)
    raise SystemExit(f"none of {names} found (pass --toolchain DIR)")


def _run(argv: list[str], cwd: Path) -> subprocess.CompletedProcess:
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(argv)}\n failed rc={proc.returncode}: {proc.stderr[-1500:]}")
    return proc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("lowered", help="a lowered.llvm.mlir (or a directory holding one)")
    ap.add_argument("--ws", default=".", help="workspace root (default: current directory)")
    ap.add_argument("--out", default="", help="scratch directory (default: $TMPDIR/kernel-build/NAME)")
    ap.add_argument("--contract", default="target_contract.yaml", help="relative to --ws unless absolute")
    ap.add_argument("--toolchain", default="", help="directory holding mlir-translate, clang, llvm-nm, llvm-objdump")
    ap.add_argument("--triple", default="")
    ap.add_argument("--cflag", action="append", help="replace the contract's flags (repeatable)")
    ap.add_argument("--entry", default="")
    ap.add_argument("--max-stack", type=int, default=None)
    ap.add_argument("--also", action="append", default=[], help="another object to check for duplicate symbols")
    args = ap.parse_args(argv)
    ws = Path(args.ws).resolve()
    lowered = Path(args.lowered)
    lowered = lowered / "lowered.llvm.mlir" if lowered.is_dir() else lowered
    recipe = recipe_from_contract(ws / args.contract)
    triple = args.triple or recipe.get("triple") or ""
    flags = list(args.cflag or recipe.get("cflags") or [])
    entry = args.entry or recipe.get("entry") or ""
    max_stack = args.max_stack if args.max_stack is not None else recipe.get("max_stack")
    if not triple or not entry:
        raise SystemExit("triple or entry symbol unknown: pass --triple and --entry (contract had none)")
    toolchain = Path(args.toolchain) if args.toolchain else None
    name = lowered.resolve().parent.name
    out = Path(args.out or Path(os.environ.get("TMPDIR") or ws) / "kernel-build" / name).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(lowered, out / "kernel.llvm.mlir")
    translate = _tool(("mlir-translate",), toolchain, ws)
    clang = _tool(("clang",), toolchain, ws)
    nm, objdump = _tool(("llvm-nm",), toolchain, ws), _tool(("llvm-objdump",), toolchain, ws)
    _run([translate, "--mlir-to-llvmir", "kernel.llvm.mlir", "-o", "kernel.ll"], out)
    compile_argv = [clang, f"--target={triple}", *flags, "-fstack-usage", "-c", "kernel.ll", "-o", "kernel.o"]
    _run(compile_argv, out)
    report: dict = {"out": str(out), "compile": compile_argv}
    try:
        report["stack"] = stack_verdict(parse_stack_usage((out / "kernel.su").read_text(encoding="utf-8")),
                                        entry=entry, max_stack=max_stack)
    except (OSError, ValueError) as exc:
        report["stack"] = {"refusals": [str(exc)]}
    symbols = parse_nm(_run([nm, "kernel.o"], out).stdout)
    report["symbols"] = {"external_defined": symbols["external"], "undefined": symbols["undefined"]}
    owners: dict = {}
    for obj in [out / "kernel.o", *map(Path, args.also)]:
        for symbol in parse_nm(_run([nm, str(obj.resolve())], out).stdout)["external"]:
            owners.setdefault(symbol, []).append(obj.name)
    report["symbols"]["defined_more_than_once"] = {s: o for s, o in owners.items() if len(o) > 1}
    disassembly = _run([objdump, "-d", "-r", "kernel.o"], out).stdout
    (out / "kernel.dis").write_text(disassembly, encoding="utf-8")
    report["instructions_per_function"] = function_sizes(disassembly)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["stack"].get("refusals") or report["symbols"]["defined_more_than_once"] else 0


if __name__ == "__main__":
    sys.exit(main())
