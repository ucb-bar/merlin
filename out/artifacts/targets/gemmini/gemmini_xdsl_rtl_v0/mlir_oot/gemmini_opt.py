"""CLI driver for all four out-of-tree backend entrypoints."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from xdsl.printer import Printer

from ir_ingest import command_buffer, extract_program, parse_verified
from targetgen.generate.llvm_artifact import emit_llvm_artifact
from transforms import ConvertIfaceToGemminiPass


def main() -> int:
    parser = argparse.ArgumentParser(prog="gemmini-opt")
    parser.add_argument("--verify-diagnostics", action="store_true")
    parser.add_argument("--convert-iface-to-gemmini", action="store_true")
    parser.add_argument("--emit-command-buffer")
    parser.add_argument("--emit-target-artifact", action="store_true")
    parser.add_argument("input_mlir")
    args = parser.parse_args()

    ctx, module = parse_verified(args.input_mlir)
    program = extract_program(module)
    if args.emit_command_buffer:
        output = Path(args.emit_command_buffer)
        output.write_text(json.dumps(command_buffer(program), indent=2) + "\n", encoding="utf-8")
        return 0
    if args.emit_target_artifact:
        print(emit_llvm_artifact(program), end="")
        return 0
    if args.convert_iface_to_gemmini:
        ConvertIfaceToGemminiPass().apply(ctx, module)
        stream = io.StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(module)
        print(stream.getvalue())
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

