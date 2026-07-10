#!/usr/bin/env python3
"""CLI for the Gemmini out-of-tree MLIR backend (experiment ABI v0.1).

Four subcommands, one per contract entrypoint:

    parse         <interface.mlir>                 parse + verify (nonzero exit on error)
    lower-target  <interface.mlir>                 -> gemmini-dialect MLIR on stdout
    emit-cb       <interface.mlir> <out.json>      -> schema-valid command_buffer.json
    lower-llvm    <interface.mlir>                 -> llvm.func @gemmini_kernel (RoCC) on stdout

Self-contained / integrity-clean: it uses xDSL + Merlin's interface grammar facts to author the
package, but imports no oracle (no reference/simulator, no golden).
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemmini_oot.parse_iface import parse_module, ParseError  # noqa: E402
from gemmini_oot.lower import lower_to_gemmini  # noqa: E402
from gemmini_oot.cbuf import build_command_buffer  # noqa: E402
from gemmini_oot.kernel import kernel_text  # noqa: E402


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _gemmini_text(mod) -> str:
    from xdsl.printer import Printer
    buf = io.StringIO()
    Printer(stream=buf).print_op(mod)
    return buf.getvalue() + "\n"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: gemmini_oot_tool.py <parse|lower-target|emit-cb|lower-llvm> ...",
              file=sys.stderr)
        return 2
    cmd = argv[1]
    try:
        if cmd == "parse":
            parse_module(_read(argv[2]))            # builds + verify()s; raises on error
            return 0
        if cmd == "lower-target":
            gem = lower_to_gemmini(parse_module(_read(argv[2])))
            sys.stdout.write(_gemmini_text(gem))
            return 0
        if cmd == "emit-cb":
            gem = lower_to_gemmini(parse_module(_read(argv[2])))
            cb = build_command_buffer(gem)
            with open(argv[3], "w", encoding="utf-8") as f:
                json.dump(cb, f, indent=2)
            return 0
        if cmd == "lower-llvm":
            gem = lower_to_gemmini(parse_module(_read(argv[2])))
            sys.stdout.write(kernel_text(gem))
            return 0
    except (ParseError, Exception) as e:  # fail-closed: diagnostics to stderr, nonzero exit
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"unknown command {cmd!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
