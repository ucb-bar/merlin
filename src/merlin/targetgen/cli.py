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
from dataclasses import asdict
from pathlib import Path

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


def _write_status(path: Path, report: dict[str, object]) -> None:
    """Replace a prior invocation result, including when this invocation fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cmd_native_build(args: argparse.Namespace) -> int:
    from merlin.semantic_compiler.snapshot import NativeTargetProfile, build_native_snapshot

    try:
        profile = NativeTargetProfile.from_record(json.loads(Path(args.profile).read_text()))
        snapshot = build_native_snapshot(
            profile,
            destination=Path(args.out),
            crate=Path(args.crate),
            cargo_target_dir=Path(args.cargo_target_dir),
            source_revision=args.source_revision,
        )
        report = {"status": "selection_only", "engine": args.engine, "out": str(snapshot.root),
                  "profile_sha256": snapshot.manifest["profile_sha256"]}
    except (OSError, ValueError, RuntimeError) as error:
        report = {"status": "compile_error", "engine": args.engine, "reason": str(error), "out": args.out}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "selection_only" else 2


def _cmd_native_select(args: argparse.Namespace) -> int:
    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.snapshot import open_native_snapshot

    output = Path(args.out)
    try:
        snapshot = open_native_snapshot(Path(args.snapshot))
        request = KernelRequest.from_record(json.loads(Path(args.request).read_text()))
        abi = json.loads(Path(args.abi).read_text()) if args.abi else {"fixed_inputs": {}, "fixed_outputs": None}
        if set(abi) != {"fixed_inputs", "fixed_outputs"} or not isinstance(abi["fixed_inputs"], dict):
            raise ValueError("native ABI needs fixed_inputs and fixed_outputs")
        fixed_outputs = None if abi["fixed_outputs"] is None else tuple(abi["fixed_outputs"])
        result = snapshot.select(request, fixed_inputs=abi["fixed_inputs"], fixed_outputs=fixed_outputs)
        constants = {binding.node_id: binding for binding in request.constants}
        constant_requirements = [
            {"value_id": value.id, "storage": value.storage,
             "address": result.allocation.addresses[value.id], **constants[value.source_node].record()}
            for value in (result.graph.values if result.graph else ())
            if value.kind == "constant" and result.allocation is not None
        ]
        report = {
            "schema": "merlin.native_selection_result.v1", "engine": result.engine,
            "status": result.status, "scope": "selection_only", "request_digest": result.request_digest,
            "target_digest": snapshot.profile.digest(), "snapshot": str(snapshot.root),
            "reason": result.reason, "candidate_attempts": result.candidate_attempts,
            "ordering_attempts": result.ordering_attempts, "rejected_allocation": result.rejected_allocation,
            "pruned_orders": result.pruned_orders, "check_fingerprint": result.check_fingerprint,
            "candidate_digest": result.candidate.digest() if result.candidate else None,
            "selected_graph": asdict(result.graph) if result.graph else None,
            "allocation": asdict(result.allocation) if result.allocation else None,
            "constant_requirements": constant_requirements,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        report = {"schema": "merlin.native_selection_result.v1", "engine": args.engine,
                  "status": "compile_error", "scope": "selection_only", "reason": str(error)}
    _write_status(output, report)
    print(json.dumps({"status": report["status"], "engine": args.engine, "out": str(output)}, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


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

    native_build = sub.add_parser("native-build", help="build an offline Merlin-native selection snapshot")
    native_build.add_argument("--engine", choices=("merlin_native",), required=True)
    native_build.add_argument("--profile", required=True, help="versioned native target profile JSON")
    native_build.add_argument("--crate", required=True, help="pinned Merlin egg bridge source directory")
    native_build.add_argument("--cargo-target-dir", required=True)
    native_build.add_argument("--source-revision", required=True)
    native_build.add_argument("--out", required=True, help="fresh snapshot directory")
    native_build.set_defaults(func=_cmd_native_build)

    native_select = sub.add_parser("native-select", help="select and allocate one typed kernel; no Atlas emission")
    native_select.add_argument("--engine", choices=("merlin_native",), required=True)
    native_select.add_argument("--snapshot", required=True)
    native_select.add_argument("--request", required=True, help="typed semantic kernel JSON")
    native_select.add_argument("--abi", help="optional fixed_inputs/fixed_outputs JSON; no runtime samples")
    native_select.add_argument("--out", required=True, help="selection result JSON")
    native_select.set_defaults(func=_cmd_native_select)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
