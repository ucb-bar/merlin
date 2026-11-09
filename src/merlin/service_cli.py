"""CLI adapters for the same public compile, runtime, and inspection calls."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path


def _print(value) -> None:
    print(json.dumps(asdict(value) if is_dataclass(value) else value, indent=2, default=str))


def compile_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from merlin.api import CompileOptions, SavedExport, Source, StagedProgram, compile

    parser = argparse.ArgumentParser(
        prog="merlin compile",
        description="Compile explicit source using one selected backend and the public Python service",
        epilog="For the retained workload/operator workflow, use merlin-compile --help.",
    )
    parser.add_argument("source", type=Path, help="MLIR file, saved .pt2 export, or compiler-stage directory")
    parser.add_argument("--backend", required=True, help="installed provider name; see merlin inspect --backends")
    parser.add_argument("--output", required=True, type=Path, help="fresh compilation output directory")
    parser.add_argument(
        "--input-kind",
        help="compiler-stage selects a compiler-only directory; "
        "defaults to torch-export for .pt2 files, otherwise mlir",
    )
    parser.add_argument(
        "--capture-output", type=Path, help="fresh export evidence workspace; defaults to OUTPUT.capture"
    )
    parser.add_argument("--system", help="hardware system selection admitted by the provider")
    parser.add_argument("--platform", help="image environment admitted by the provider, such as zephyr or baremetal")
    parser.add_argument("--harts", type=int, help="target hart count admitted by the provider")
    parser.add_argument(
        "--freeze-readonly-buffers",
        action="store_true",
        help="explicitly snapshot observed readonly BUFFER state in compiler-stage input",
    )
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAMESPACE.KEY=JSON",
        help="provider-admitted pass or optimization option with a JSON value; repeat as needed",
    )
    args = parser.parse_args(argv)
    parameters = {}
    for item in args.option:
        key, sep, value = item.partition("=")
        if not sep or key in parameters:
            parser.error(f"compiler option must be a unique KEY=JSON value: {item!r}")
        try:
            parameters[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid option JSON: {exc}")
    kind = args.input_kind or ("torch-export" if args.source.suffix.lower() == ".pt2" else "mlir")
    if args.freeze_readonly_buffers and kind != "compiler-stage":
        parser.error("--freeze-readonly-buffers requires --input-kind compiler-stage")
    if kind == "torch-export":
        source = SavedExport(
            args.source,
            capture=args.capture_output
            if args.capture_output is not None
            else args.output.with_name(args.output.name + ".capture"),
        )
    elif kind == "compiler-stage":
        if args.capture_output is not None:
            parser.error("--capture-output requires --input-kind torch-export")
        source = StagedProgram(args.source, freeze_readonly_buffers=args.freeze_readonly_buffers)
    else:
        if args.capture_output is not None:
            parser.error("--capture-output requires --input-kind torch-export")
        source = Source(args.source.read_text(encoding="utf-8"), kind=kind, origin=str(args.source.resolve()))
    result = compile(
        source,
        backend=args.backend,
        output=args.output,
        options=CompileOptions(args.system, args.platform, args.harts, parameters),
    )
    _print({"manifest": str(result.manifest), **result.to_dict()})
    return 0


def run_main(argv: list[str] | None = None) -> int:
    from merlin.api import Deployment, run
    from merlin.runtime.service import RunOptions

    parser = argparse.ArgumentParser(
        prog="merlin run", description="Execute an existing admitted ELF, or prepare external execution"
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--runner", required=True, help="runtime provider, such as merlin.linux or firesim")
    parser.add_argument(
        "--mode",
        choices=("execute", "prepare"),
        default="execute",
        help="execute with the selected runner, or prepare external execution files; FireSim requires prepare",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", default="executable")
    parser.add_argument("--entry", help="relative ELF path for an external handoff with multiple images")
    parser.add_argument(
        "--argument", action="append", default=[], help="Linux process or external guest argument; repeat as needed"
    )
    parser.add_argument("--input", action="append", default=[], metavar="RELATIVE_NAME=PATH")
    parser.add_argument("--harts", type=int, help="Linux initial affinity CPU count, or selected target hart count")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--system", help="hardware system for an ELF without a deployment manifest")
    parser.add_argument("--platform", help="image environment, such as linux, zephyr or baremetal")
    parser.add_argument("--isa", help="selected instruction-set contract")
    parser.add_argument("--abi", help="selected binary calling convention")
    parser.add_argument("--dram-base", type=lambda x: int(x, 0))
    parser.add_argument("--memory-bytes", type=lambda x: int(x, 0))
    args = parser.parse_args(argv)
    contract = None
    if any(
        value is not None
        for value in (args.system, args.platform, args.isa, args.abi, args.dram_base, args.memory_bytes)
    ):
        if any(value is None for value in (args.system, args.platform, args.isa, args.abi, args.harts)):
            parser.error("external image contract requires --system, --platform, --isa, --abi, and --harts")
        contract = Deployment(
            args.system, args.platform, args.isa, args.abi, args.harts, args.dram_base, args.memory_bytes
        )
    inputs = {}
    for value in args.input:
        name, sep, path = value.partition("=")
        if not sep or name in inputs:
            parser.error("input requires a unique RELATIVE_NAME=PATH")
        inputs[name] = Path(path)
    result = run(
        args.artifact,
        runner=args.runner,
        options=RunOptions(
            contract,
            args.harts,
            args.timeout,
            args.role,
            args.mode,
            args.output,
            tuple(args.argument),
            inputs,
            args.entry,
        ),
    )
    _print(result)
    return 1 if getattr(result, "status", None) in {"failed", "timed_out"} else 0


def inspect_main(argv: list[str] | None = None) -> int:
    from merlin.api import inspect, inspect_backends, inspect_spec

    parser = argparse.ArgumentParser(
        prog="merlin inspect", description="Inspect a compilation manifest or authored software specification"
    )
    parser.add_argument("manifest", type=Path, nargs="?")
    parser.add_argument("--spec", type=Path, help="inspect a software specification without compiling")
    parser.add_argument("--backends", action="store_true", help="list installed backend metadata without loading code")
    parser.add_argument("--expand", action="store_true", help="include expanded definitions with --spec")
    args = parser.parse_args(argv)
    if sum((args.manifest is not None, args.spec is not None, args.backends)) != 1:
        parser.error("select one compilation manifest, --spec PATH or --backends")
    if args.expand and args.spec is None:
        parser.error("--expand requires --spec")
    if args.backends:
        _print(inspect_backends())
    else:
        _print(inspect_spec(args.spec, expanded=args.expand) if args.spec is not None else inspect(args.manifest))
    return 0


def collect_main(argv: list[str] | None = None) -> int:
    from merlin.api import collect

    parser = argparse.ArgumentParser(
        prog="merlin collect", description="Collect existing external execution observations"
    )
    parser.add_argument("prepared", type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args(argv)
    _print(collect(args.prepared, results=args.results))
    return 0
