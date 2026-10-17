"""Command-line access to byte-bound Lean checks and empirical MLIR comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def _positive(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _run(args) -> int:
    from . import check

    if args.lean_command == "stage":
        result = check.check_stage(args.stage, m2m_python=args.compiler_python)
    elif args.lean_command == "typed-stage":
        result = check.check_typed_stage(args.stage, m2m_python=args.compiler_python, index_bits=args.index_bits)
    elif args.lean_command in {"support", "definedness"}:
        checker = check.check_support if args.lean_command == "support" else check.check_definedness
        result = checker(
            args.program,
            args.function,
            args.work,
            m2m_python=args.compiler_python,
            index_bits=args.index_bits,
        )
    else:
        result = check.compose_chain(
            args.stages,
            args.support,
            args.function,
            args.output,
            args.handoff,
            m2m_python=args.compiler_python,
            definedness=args.definedness,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("verdict") == "verified" else 2


def _differential(args) -> int:
    from . import differential

    # Refuse reused directories: a failed attempt must not leave an earlier matching result.
    args.work.mkdir(parents=True, exist_ok=False)
    tools = differential.Tools(args.compiler_python, args.lean_exe, args.lake, args.runner_utils)
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    result = differential.compare(
        args.program.read_text(encoding="utf-8"),
        samples,
        args.work,
        tools,
        function=args.function,
        index_bits=args.index_bits,
        arithmetic_nans=args.arithmetic_nans,
    )
    record = {
        "status": "matched_samples",
        "assumptions": ["caller independently established defined native execution for every sample"],
        "nan_comparison": "classification" if args.arithmetic_nans else "bits",
        **result,
    }
    (args.work / "comparison.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


def _dispatch(args) -> int:
    try:
        return args.lean_fn(args)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, AssertionError, subprocess.SubprocessError) as exc:
        # An incomplete check or failed comparison establishes neither a proof nor a refutation.
        print(json.dumps({"verdict": "unmeasured", "reason": f"{type(exc).__name__}: {exc}"}))
        return 2


def add_commands(subparsers) -> None:
    """Register ``merlin-verify lean`` without importing optional verification tools at startup."""
    parser = subparsers.add_parser("lean", help="check recorded shared passes or compare Lean with native MLIR")
    commands = parser.add_subparsers(dest="lean_command", required=True)
    for name in ("stage", "typed-stage", "support", "definedness", "chain", "differential"):
        command = commands.add_parser(name)
        command.add_argument("--compiler-python", type=Path, required=True, help="authoritative torch_mlir interpreter")
        command.set_defaults(fn=_dispatch, lean_fn=_differential if name == "differential" else _run)
        if name in {"stage", "typed-stage"}:
            command.add_argument(
                "stage", type=Path, help="recorded stage directory with before/after MLIR and meta.json"
            )
        if name in {"support", "definedness", "differential"}:
            command.add_argument("program", type=Path)
            command.add_argument("--work", type=Path, required=True)
        if name in {"support", "definedness", "chain", "differential"}:
            command.add_argument("--function", required=True)
        if name in {"typed-stage", "support", "definedness", "differential"}:
            command.add_argument(
                "--index-bits",
                type=_positive,
                default=64,
                choices=(32, 64) if name == "differential" else None,
            )
        if name == "chain":
            command.add_argument("stages", type=Path, nargs="+", help="stage directories in execution order")
            command.add_argument("--support", type=Path, required=True, help="source support receipt")
            command.add_argument(
                "--definedness", type=Path, help="source definedness receipt required for overall verification"
            )
            command.add_argument("--handoff", type=Path, help="recorded handoff.json")
            command.add_argument(
                "--output", type=Path, required=True, help="chain receipt; parent directory must exist"
            )
        if name == "differential":
            command.add_argument(
                "--samples", type=Path, required=True, help="JSON [sample][argument][element], integer bit patterns"
            )
            command.add_argument("--lean-exe", type=Path, required=True, help="built merlin-lean checker/printer")
            command.add_argument("--lake", type=Path, required=True)
            command.add_argument("--runner-utils", type=Path, required=True)
            command.add_argument(
                "--arithmetic-nans", action="store_true", help="compare NaNs by classification for arithmetic cases"
            )
            command.add_argument(
                "--assume-defined-inputs",
                action="store_true",
                required=True,
                help="acknowledge independent validation that every sample has defined native execution",
            )
