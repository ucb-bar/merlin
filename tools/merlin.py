#!/usr/bin/env python3
# tools/merlin.py

import argparse
import sys

import benchmark as benchmark_cmd
import chipyard as chipyard_cmd
import ci as ci_cmd
import compile as compile_cmd
import patches as patches_cmd
import ray_cmd as ray_cmd
import setup as setup_cmd
import targetgen_cmd as targetgen_cmd

import build as build_cmd

COMMANDS: tuple[tuple[str, object, str, bool], ...] = (
    ("build", build_cmd, "Configure and build Merlin and target runtimes", True),
    ("compile", compile_cmd, "Compile MLIR/ONNX models to target artifacts", True),
    ("setup", setup_cmd, "Bootstrap developer environment and toolchains", False),
    ("ci", ci_cmd, "Run repository CI/lint/patch workflows", True),
    ("patches", patches_cmd, "Verify submodule state and manage upstream patches", False),
    ("benchmark", benchmark_cmd, "Run benchmark helper scripts", True),
    ("chipyard", chipyard_cmd, "Manage Chipyard hardware backend interactions", True),
    ("ray", ray_cmd, "Manage Merlin's Ray control plane, jobs, resources, and artifacts", True),
    ("targetgen", targetgen_cmd, "Plan and orchestrate hardware-spec-driven target enablement", True),
)


def setup_parser(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name, module, help_text, supports_dry_run in COMMANDS:
        subparser = subparsers.add_parser(command_name, help=help_text, description=help_text)
        if supports_dry_run:
            subparser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
        module.setup_parser(subparser)
        subparser.set_defaults(_handler=module.main)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin", description="Unified Merlin developer command reference parser.")
    setup_parser(parser)
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
