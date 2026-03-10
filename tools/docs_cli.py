#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys

import benchmark
import ci
import compile as compile_tool
import patches
import setup as setup_tool

import build

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


_TOOL_MODULES = (
    ("build", build, "Configure and build Merlin and target runtimes"),
    ("compile", compile_tool, "Compile MLIR/ONNX models to target artifacts"),
    ("setup", setup_tool, "Bootstrap developer environment and toolchains"),
    ("ci", ci, "Run repository CI/lint/patch workflows"),
    ("patches", patches, "Apply/verify/refresh/drift patch stack"),
    ("benchmark", benchmark, "Run benchmark helper scripts"),
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="merlin",
        description="Unified Merlin developer command reference parser.",
    )
    subparsers = parser.add_subparsers(dest="tool", required=True)

    for name, module, help_text in _TOOL_MODULES:
        child = subparsers.add_parser(name, help=help_text, description=help_text)
        module.setup_parser(child)

    return parser


if __name__ == "__main__":
    print(get_parser().format_help())
