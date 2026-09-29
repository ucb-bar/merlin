"""TargetGen command-line interface.

    python -m merlin.targetgen.cli build \\
        --target-name toy_npu \\
        --source-dir examples/toy_npu/target/docs \\
        --examples-dir examples/toy_npu/target/examples \\
        --out out/build/generated/merlin-target-toy-npu \\
        --emit xdsl,mlir,zephyr,llvm-plan,runtime

    python -m merlin.targetgen.cli inspect --target out/build/generated/merlin-target-toy-npu

Deterministic, no LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import pipeline
from .isa_census import derive_source_census
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


def _cmd_audit_isa(args: argparse.Namespace) -> int:
    """Write a source crosswalk without treating decode as legality evidence."""
    output = Path(args.out)
    temporary = output.with_name(output.name + ".tmp")
    try:
        census = derive_source_census(
            pattern_file=Path(args.patterns),
            decoder_file=Path(args.decoder),
            model_isa_file=Path(args.model_isa),
            rtl_revision=args.rtl_revision,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    except (OSError, SyntaxError, ValueError) as error:
        output.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        print(json.dumps({"status": "FAIL", "error": str(error), "out": str(output)}))
        return 2
    summary = census["summary"]
    problems = sum(
        len(summary[key])
        for key in (
            "patterns_not_decoded",
            "decoder_rows_without_pattern",
            "model_classes_without_compatible_pattern",
            "overlapping_patterns",
            "dma_kind_conflicts",
        )
    )
    status = "SOURCE_DISCREPANCIES" if problems else "SOURCE_CROSSWALK_ONLY"
    print(json.dumps({"status": status, "discrepancies": problems, "out": str(output)}))
    return 1 if problems else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="merlin-targetgen", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="ingest -> synthesize -> generate a target repo")
    b.add_argument("--target-name", required=True)
    b.add_argument("--source-dir", default=None, help="local docs/source directory (not crawled)")
    b.add_argument("--examples-dir", default=None)
    b.add_argument("--scala-root", default=None)
    b.add_argument("--out", default=None, help="output directory for the generated repo")
    b.add_argument(
        "--emit",
        default=",".join(pipeline.DEFAULT_EMIT),
        help="comma list of layers: xdsl,mlir,zephyr,runtime,llvm-plan or contract-only",
    )
    b.set_defaults(func=_cmd_build)

    i = sub.add_parser("inspect", help="validate a generated target repo")
    i.add_argument("--target", required=True, help="path to a generated target repo")
    i.set_defaults(func=_cmd_inspect)

    a = sub.add_parser("audit-isa", help="crosswalk pinned decoder, patterns, and model ISA sources")
    a.add_argument("--patterns", required=True, help="selected RTL Instructions.scala")
    a.add_argument("--decoder", required=True, help="selected RTL IDecode.scala")
    a.add_argument("--model-isa", required=True, help="selected Python ISA definition")
    a.add_argument("--rtl-revision", required=True, help="exact selected RTL commit")
    a.add_argument("--out", required=True, help="census JSON artifact")
    a.set_defaults(func=_cmd_audit_isa)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
