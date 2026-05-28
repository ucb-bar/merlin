#!/usr/bin/env python3
# tools/merlin.py

import argparse
import sys

import sim as sim_cmd
import spike as spike_cmd
from coverage import cli as coverage_cmd
from mcp_servers import cli as mcp_cmd
from perf import cli as perf_cmd
from quantize import cli as quantize_cmd
from ray import cli as ray_cmd
from run import cli as run_cmd
from targetgen import cli as targetgen_cmd
from verify import cli as verify_output_cmd

import benchmark as benchmark_cmd
import ci as ci_cmd
import patches as patches_cmd
import setup as setup_cmd
from build import cli as build_cmd
from chipyard import cli as chipyard_cmd
from compile import cli as compile_cmd

COMMANDS: tuple[tuple[str, object, str, bool], ...] = (
    ("build", build_cmd, "Configure and build Merlin and target runtimes", True),
    ("compile", compile_cmd, "Compile MLIR/ONNX models to target artifacts", True),
    ("quantize", quantize_cmd, "INT8-quantize any .onnx model (QDQ, symmetric, per-tensor)", False),
    ("verify-output", verify_output_cmd, "Cross-hash check backend outputs vs onnxruntime golden", False),
    ("perf-decompose", perf_cmd, "Per-dispatch performance decomposition from FireSim uartlog", False),
    ("coverage-check", coverage_cmd, "Per-dispatch accelerator coverage report of a VMFB", False),
    ("setup", setup_cmd, "Bootstrap developer environment and toolchains", False),
    ("ci", ci_cmd, "Run repository CI/lint/patch workflows", True),
    ("patches", patches_cmd, "Verify submodule state and manage upstream patches", False),
    ("benchmark", benchmark_cmd, "Run benchmark helper scripts", True),
    ("chipyard", chipyard_cmd, "Manage Chipyard hardware backend interactions", True),
    ("ray", ray_cmd, "Manage Merlin's Ray control plane, jobs, resources, and artifacts", True),
    ("targetgen", targetgen_cmd, "Plan and orchestrate hardware-spec-driven target enablement", True),
    ("spike", spike_cmd, "Run a Gemmini kernel fixture under spike+pk for functional validation", False),
    ("sim", sim_cmd, "Run an mxGemmini fixture through the IREE pipeline + chipyard VCS simulator", False),
    ("run", run_cmd, "Execute a compiled model on a target board (schedule/multi-device/het-e2e/...)", False),
    ("mcp", mcp_cmd, "Start one of the Merlin MCP servers (build/compile/run/perf/verify)", False),
)


def setup_parser(parser: argparse.ArgumentParser):
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command_name, module, help_text, supports_dry_run in COMMANDS:
        subparser = subparsers.add_parser(command_name, help=help_text, description=help_text)
        if supports_dry_run:
            subparser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
        module.setup_parser(subparser)
        subparser.set_defaults(_handler=module.main)


def _log_usage(args: argparse.Namespace) -> None:
    """Append a one-line usage record to ~/.merlin_usage.log.

    Failure-tolerant — never raises out of this function. The log lives in
    the user's home, never inside the repo. Format:
        ISO8601_TIMESTAMP \\t <subcommand> \\t <mode>

    `mode` is best-effort: it picks up `args.mode` (used by `./merlin run`)
    or `args.subcommand` if present, else empty. Useful for an
    empirically-driven archive sweep — after a few weeks::

        awk -F'\\t' '{print $2 "\\t" $3}' ~/.merlin_usage.log \\
            | sort | uniq -c | sort -rn

    Subcommands with zero hits become archive candidates.
    """
    try:
        import datetime
        import os

        log = os.path.expanduser("~/.merlin_usage.log")
        cmd = getattr(args, "command", "") or ""
        mode = getattr(args, "mode", "") or getattr(args, "subcommand", "") or ""
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{ts}\t{cmd}\t{mode}\n")
    except Exception:
        # Never block a subcommand on logging failure.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="merlin", description="Unified Merlin developer command reference parser.")
    setup_parser(parser)
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    _log_usage(args)
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
