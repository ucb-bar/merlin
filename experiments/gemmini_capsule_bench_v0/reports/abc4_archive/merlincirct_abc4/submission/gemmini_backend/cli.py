#!/usr/bin/env python3
"""CLI entrypoints for the Gemmini OOT backend (experiment ABI v0.1).

Subcommands map 1:1 to the four contract entrypoints:

    parse        <interface.mlir>                 -> exit 0 / diagnostics
    lower-target <interface.mlir>                 -> gemmini-dialect MLIR (stdout)
    emit-cb      <interface.mlir> <out.json>      -> command_buffer.json
    lower-llvm   <interface.mlir>                 -> llvm-dialect MLIR (stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# allow running as a loose script (argv[0] = this file) or as a module
_HERE = Path(__file__).resolve().parent
if __package__ in (None, ""):
    sys.path.insert(0, str(_HERE.parent))
    import gemmini_backend.frontend as frontend  # type: ignore
    from gemmini_backend.passes import lower_to_gemmini
    from gemmini_backend.program import extract
    from gemmini_backend.cmdbuf import emit_command_buffer
    from gemmini_backend.rocc import emit_llvm
else:
    from . import frontend
    from .passes import lower_to_gemmini
    from .program import extract
    from .cmdbuf import emit_command_buffer
    from .rocc import emit_llvm


def _read(path: str) -> str:
    return Path(path).read_text()


def _lowered_module(text: str):
    mod = frontend.build_module(text)
    return lower_to_gemmini(mod)


def cmd_parse(args) -> int:
    try:
        frontend.build_module(_read(args[0]))
    except Exception as e:
        print(f"parse error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_lower_target(args) -> int:
    mod = _lowered_module(_read(args[0]))
    print(str(mod))
    return 0


def cmd_emit_cb(args) -> int:
    mod = _lowered_module(_read(args[0]))
    prog = extract(mod)
    cb = emit_command_buffer(prog)
    Path(args[1]).write_text(json.dumps(cb, indent=2))
    return 0


def cmd_lower_llvm(args) -> int:
    mod = _lowered_module(_read(args[0]))
    prog = extract(mod)
    sys.stdout.write(emit_llvm(prog))
    return 0


_DISPATCH = {
    "parse": cmd_parse,
    "lower-target": cmd_lower_target,
    "emit-cb": cmd_emit_cb,
    "lower-llvm": cmd_lower_llvm,
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _DISPATCH:
        print(f"usage: cli.py {{{'|'.join(_DISPATCH)}}} ...", file=sys.stderr)
        return 2
    return _DISPATCH[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
