#!/usr/bin/env python3
"""``gemmini-opt`` — the package tool. Four CLI entrypoints (subprocess boundary):

    gemmini_opt.py parse       <iface.mlir>                 -> verify, exit 0/nonzero
    gemmini_opt.py lower-iface  <iface.mlir>                -> gemmini-dialect MLIR (stdout)
    gemmini_opt.py emit-cb      <iface.mlir> <out.json>     -> command_buffer.json
    gemmini_opt.py lower-llvm   <iface.mlir>                -> llvm/RoCC MLIR (stdout)

Self-contained: imports only ``xdsl`` + stdlib (no ``merlin``).
"""
from __future__ import annotations

import io
import json
import os
import sys

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from gemmini_backend import iface_ir, passes, cmdbuf, rocc  # noqa: E402

SUPPORTED_VERSION = "0.1"


def _read(path):
    with open(path) as f:
        return f.read()


def _build_and_verify(text):
    prog = iface_ir.parse_program(text)
    if prog.version != SUPPORTED_VERSION:
        raise ValueError(
            f"unsupported merlin_iface.version {prog.version!r}; this backend "
            f"implements {SUPPORTED_VERSION!r}")
    module = passes.build_iface_module(prog)
    module.verify()
    return prog, module


def _print_module(module) -> str:
    from xdsl.printer import Printer
    s = io.StringIO()
    Printer(stream=s).print_op(module)
    return s.getvalue()


def cmd_parse(argv):
    text = _read(argv[0])
    _build_and_verify(text)
    return 0


def cmd_lower_iface(argv):
    _, _module = _build_and_verify(_read(argv[0]))
    prog = iface_ir.parse_program(_read(argv[0]))
    iface_module = passes.build_iface_module(prog)
    iface_module.verify()
    gem = passes.lower_to_gemmini(iface_module)
    gem.verify()
    sys.stdout.write(_print_module(gem))
    return 0


def cmd_emit_cb(argv):
    text = _read(argv[0])
    out_json = argv[1]
    prog, _ = _build_and_verify(text)
    cb = cmdbuf.build_command_buffer(prog)
    with open(out_json, "w") as f:
        json.dump(cb, f, indent=2)
    return 0


def cmd_lower_llvm(argv):
    text = _read(argv[0])
    prog, _ = _build_and_verify(text)
    sys.stdout.write(rocc.emit_llvm(prog))
    return 0


_COMMANDS = {
    "parse": cmd_parse,
    "lower-iface": cmd_lower_iface,
    "emit-cb": cmd_emit_cb,
    "lower-llvm": cmd_lower_llvm,
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in _COMMANDS:
        sys.stderr.write(f"usage: gemmini_opt.py {{{'|'.join(_COMMANDS)}}} <args>\n")
        return 2
    cmd, rest = argv[0], argv[1:]
    try:
        return _COMMANDS[cmd](rest)
    except Exception as e:  # noqa: BLE001 - report a diagnostic, fail closed
        sys.stderr.write(f"error[{cmd}]: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
