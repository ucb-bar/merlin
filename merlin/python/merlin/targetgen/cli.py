"""TargetGen command-line interface.

    python -m merlin.targetgen.cli build \\
        --target-name toy_npu \\
        --source-dir merlin/targets/toy_npu/docs \\
        --examples-dir merlin/targets/toy_npu/examples \\
        --out out/build/generated/merlin-target-toy-npu \\
        --emit xdsl,mlir,zephyr,llvm-plan,runtime

    python -m merlin.targetgen.cli inspect --target out/build/generated/merlin-target-toy-npu

Deterministic, no LLM calls.
"""
from __future__ import annotations

import argparse
import sys

from . import pipeline
from .validate import check_generated_target


def _cmd_build(args: argparse.Namespace) -> int:
    emit = [e for e in (args.emit or "").split(",") if e.strip()] or None
    result = pipeline.build(
        target_name=args.target_name,
        source_dir=args.source_dir,
        examples_dir=args.examples_dir,
        scala_root=args.scala_root,
        out=args.out,
        emit=emit,
    )
    print(f"target        : {result.target}")
    print(f"out           : {result.out}")
    print(f"emit          : {', '.join(result.emit) if result.emit else 'contract-only'}")
    print(f"detected      : {', '.join(result.evidence_concepts) or '(none)'}")
    print(f"files written : {len(result.written)}")
    if result.schema_problems:
        print(f"\nschema problems ({len(result.schema_problems)}):")
        for p in result.schema_problems:
            print(f"  - {p}")
        return 1
    print("schema validation: PASS")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    problems = check_generated_target(args.target)
    if problems:
        print(f"{args.target}: {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{args.target}: OK (structure + contracts valid)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merlin-targetgen", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="ingest -> synthesize -> generate a target repo")
    b.add_argument("--target-name", required=True)
    b.add_argument("--source-dir", default=None, help="local docs/source directory (not crawled)")
    b.add_argument("--examples-dir", default=None)
    b.add_argument("--scala-root", default=None)
    b.add_argument("--out", default=None, help="output directory for the generated repo")
    b.add_argument("--emit", default=",".join(pipeline.DEFAULT_EMIT),
                   help="comma list of layers: xdsl,mlir,zephyr,runtime,llvm-plan or contract-only")
    b.set_defaults(func=_cmd_build)

    i = sub.add_parser("inspect", help="validate a generated target repo")
    i.add_argument("--target", required=True, help="path to a generated target repo")
    i.set_defaults(func=_cmd_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
