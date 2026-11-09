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
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from merlin.compiler.frontend.native import _load_native_request

from . import pipeline
from .dialect_source_scope import audit_dialect_source_scope
from .isa_census import derive_source_census
from .isa_mode_audit import audit_mode_inventory
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
            format_file=Path(args.formats) if args.formats else None,
            model_revision=args.model_revision,
            verify_revisions=args.verify_revisions,
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
            "model_classes_without_derived_layout",
            "overlapping_patterns",
            "dma_kind_conflicts",
        )
    )
    status = "SOURCE_DISCREPANCIES" if problems else "SOURCE_CROSSWALK_ONLY"
    print(json.dumps({"status": status, "discrepancies": problems, "out": str(output)}))
    return 1 if problems else 0


def _cmd_audit_dialect_modes(args: argparse.Namespace) -> int:
    """Reconcile an OOT mode ledger with pinned source rows and an optional plan."""
    import yaml

    output = Path(args.out)
    temporary = output.with_name(output.name + ".tmp")
    try:
        census_bytes = Path(args.census).read_bytes()
        inventory_bytes = Path(args.inventory).read_bytes()
        plan_bytes = Path(args.dialect_plan).read_bytes() if args.dialect_plan else None
        census = json.loads(census_bytes)
        inventory = json.loads(inventory_bytes)
        plan = yaml.safe_load(plan_bytes) if plan_bytes is not None else None
        report = audit_mode_inventory(census, inventory, dialect_plan=plan)
        report["inputs_sha256"] = {
            "census": hashlib.sha256(census_bytes).hexdigest(),
            "inventory": hashlib.sha256(inventory_bytes).hexdigest(),
            "dialect_plan": hashlib.sha256(plan_bytes).hexdigest() if plan_bytes is not None else None,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    except (OSError, ValueError, yaml.YAMLError) as error:
        output.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        print(json.dumps({"status": "FAIL", "error": str(error), "out": str(output)}))
        return 2
    criterion = "phase1_parameter_domains_ready" if args.require == "phase1-inputs" else "mode_inventory_ready"
    status = (
        "SOURCE_MISMATCH"
        if not report["source_bound"]
        else "SOURCE_DISCREPANCIES"
        if not report["source_reconciled"]
        else "PARAMETER_DOMAINS_OPEN"
        if (args.require == "phase1-inputs" and report["phase1_mode_scope_ready"] and not report[criterion])
        else "MODE_SCOPE_OPEN"
        if args.require == "phase1-inputs" and not report[criterion]
        else "PHASE1_INPUTS_READY"
        if args.require == "phase1-inputs"
        else "MODE_OBLIGATIONS_OPEN"
        if not report[criterion]
        else "MODE_INVENTORY_READY"
    )
    print(json.dumps({"status": status, "counts": report["counts"], "out": str(output)}))
    return 0 if report[criterion] else 1


def _cmd_audit_dialect_source_scope(args: argparse.Namespace) -> int:
    """Check that a mode census belongs to one reproduced selected RTL source."""
    output = Path(args.out)
    temporary = output.with_name(output.name + ".tmp")
    try:
        report = audit_dialect_source_scope(
            selection_path=Path(args.source_selection),
            census_path=Path(args.census),
            inventory_path=Path(args.inventory),
            expected_config=args.expected_config,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        output.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)
        print(json.dumps({"status": "FAIL", "error": str(error), "out": str(output)}))
        return 2
    print(json.dumps({"status": report["status"].upper(), "blockers": len(report["blockers"]), "out": str(output)}))
    return 0 if report["status"] == "ready" else 1


def _write_status(path: Path, report: dict[str, object]) -> None:
    """Replace a prior invocation result, including when this invocation fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _cmd_stage_capture(args: argparse.Namespace) -> int:
    """Publish only declared source and weight inputs for later compilation."""
    from merlin.compiler.frontend.inputs import stage_compile_inputs

    safe_status = bool(args.status_file) and not (
        Path(args.status_file).resolve().is_relative_to(Path(args.out).resolve())
        or Path(args.status_file).resolve().is_relative_to(Path(args.capture).resolve())
    )
    try:
        if args.status_file and not safe_status:
            raise ValueError("capture staging status file must be outside capture and compiler inputs")
        result = stage_compile_inputs(args.capture, args.out, precision_policy=getattr(args, "precision_policy", None))
        report = {
            "schema": "merlin.capture_staging_status.v1",
            "status": "staged",
            "out": str(Path(args.out).absolute()),
            "manifest": result,
        }
        status = 0
    except (OSError, ValueError, UnicodeError) as exc:
        report = {
            "schema": "merlin.capture_staging_status.v1",
            "status": "compile_input_error",
            "out": str(Path(args.out).absolute()),
            "error": f"{type(exc).__name__}: {exc}",
        }
        status = 2
    if safe_status:
        _write_status(Path(args.status_file), report)
    print(json.dumps(report, sort_keys=True))
    return status


def _cmd_native_build(args: argparse.Namespace) -> int:
    from merlin.semantic_compiler.snapshot import (
        NativeTargetProfile,
        build_native_snapshot,
        packaged_egg_bridge_path,
    )
    from merlin.semantic_compiler.target_binding import load_native_target_binding

    try:
        if bool(args.profile) == bool(args.support):
            raise ValueError("select exactly one native target profile or installed support provider")
        profile = (
            NativeTargetProfile.from_record(json.loads(Path(args.profile).read_text()))
            if args.profile
            else load_native_target_binding(args.support).profile()
        )
        snapshot = build_native_snapshot(
            profile,
            destination=Path(args.out),
            crate=Path(args.crate) if args.crate else packaged_egg_bridge_path(),
            cargo_target_dir=Path(args.cargo_target_dir),
            source_revision=args.source_revision,
        )
        report = {
            "status": "selection_only",
            "engine": args.engine,
            "out": str(snapshot.root),
            "profile_sha256": snapshot.manifest["profile_sha256"],
        }
    except (OSError, ValueError, RuntimeError) as error:
        report = {"status": "compile_error", "engine": args.engine, "reason": str(error), "out": args.out}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "selection_only" else 2


def _load_native_search_limits(path: str | None):
    from merlin.semantic_compiler.search import SearchLimits

    return SearchLimits.from_record(json.loads(Path(path).read_text())) if path else SearchLimits()


_SELECT_ABI_KEYS = frozenset({"fixed_inputs", "fixed_outputs"})
_RESERVATION_KEYS = frozenset({"storage", "start", "extent"})


def _load_native_select_abi(path: str | None):
    """Split a selection ABI into fixed I/O and reserved physical intervals.

    ``reservations`` is optional: each row names a storage view and an interval
    the caller already owns, so no selected value may be placed there. A
    malformed row is refused rather than dropped.
    """
    from merlin.semantic_compiler.allocate import Reservation

    abi = json.loads(Path(path).read_text()) if path else {"fixed_inputs": {}, "fixed_outputs": None}
    if (
        not isinstance(abi, dict)
        or not _SELECT_ABI_KEYS <= set(abi)
        or set(abi) - _SELECT_ABI_KEYS - {"reservations"}
        or not isinstance(abi["fixed_inputs"], dict)
    ):
        raise ValueError("native ABI needs fixed_inputs and fixed_outputs; reservations are optional")
    rows = abi.get("reservations", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) or set(row) != _RESERVATION_KEYS for row in rows):
        raise ValueError("native ABI reservations must be a list of storage/start/extent records")
    fixed_outputs = None if abi["fixed_outputs"] is None else tuple(abi["fixed_outputs"])
    return abi["fixed_inputs"], fixed_outputs, tuple(Reservation(**row) for row in rows)


def _cmd_native_select(args: argparse.Namespace) -> int:
    from merlin.compiler.frontend.layouts import bind_physical_layouts
    from merlin.compiler.frontend.native import load_selected_request
    from merlin.compiler.frontend.tiling import prepare_ordered_request
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.search import SearchAblations
    from merlin.semantic_compiler.snapshot import open_native_snapshot

    output = Path(args.out)
    try:
        snapshot = open_native_snapshot(Path(args.snapshot))
        binding = None
        if getattr(args, "support", None) and any(
            getattr(args, name, None) is not None
            for name in (
                "diagnostic_graph_policy",
                "diagnostic_region",
                "diagnostic_tile_index",
                "ordered_k_tile",
                "ordered_m_tile",
                "ordered_n_tile",
                "outer_tile_index",
            )
        ):
            from merlin.semantic_compiler.target_binding import load_native_target_binding

            binding = load_native_target_binding(args.support)
            if snapshot.profile.digest() != binding.profile().digest():
                raise ValueError("native snapshot differs from selected support profile")
        request, source_kind, source_operations, precision = load_selected_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
            load_request=_load_native_request,
        )
        request, _ = bind_physical_layouts(args, request, precision)
        ordered = prepare_ordered_request(args, binding, request, precision)
        request = ordered.request
        if request.lowering_policy != args.mode:
            raise ValueError("explicit selection mode differs from typed request")
        fixed_inputs, fixed_outputs, reservations = _load_native_select_abi(args.abi)
        ablations = SearchAblations(**{name: name in args.ablation for name in vars(SearchAblations())})
        limits = _load_native_search_limits(args.search_limits)
        result = snapshot.select(
            request,
            fixed_inputs=fixed_inputs,
            fixed_outputs=fixed_outputs,
            reservations=reservations,
            limits=limits,
            ablations=ablations,
        )
        constants = {binding.node_id: binding for binding in request.constants}
        constant_requirements = [
            {
                "value_id": value.id,
                "storage": value.storage,
                "address": result.allocation.addresses[value.id],
                **constants[value.source_node].record(),
            }
            for value in (result.graph.values if result.graph else ())
            if value.kind == "constant" and result.allocation is not None
        ]
        report = {
            "schema": "merlin.native_selection_result.v1",
            "engine": result.engine,
            "status": result.status,
            "scope": "selection_only",
            "request_digest": result.request_digest,
            "source_kind": source_kind,
            "source_identity": request.source_identity,
            "source_operations": list(source_operations),
            "qualification": "component_diagnostic" if result.diagnostic_only else "component_check",
            "diagnostic_only": result.diagnostic_only,
            "diagnostic_ablations": list(result.diagnostic_ablations),
            "target_digest": snapshot.profile.digest(),
            "snapshot": str(snapshot.root),
            "search_limits": limits.record(),
            "reason": result.reason,
            "candidate_attempts": result.candidate_attempts,
            "ordering_attempts": result.ordering_attempts,
            "rejected_allocation": result.rejected_allocation,
            "pruned_orders": result.pruned_orders,
            "check_fingerprint": result.check_fingerprint,
            "reservations": [asdict(item) for item in reservations],
            "candidate_digest": result.candidate.digest() if result.candidate else None,
            "selected_graph": asdict(result.graph) if result.graph else None,
            "allocation": asdict(result.allocation) if result.allocation else None,
            "constant_requirements": constant_requirements,
        }
        if precision is not None:
            report["precision_plan"] = precision.record()
        if ordered.mn_record is not None:
            report["ordered_mn_tiling"] = json.loads(ordered.mn_record)
        if ordered.k_record is not None:
            report["ordered_k_tiling"] = json.loads(ordered.k_record)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        from merlin.semantic_compiler.target_binding import NativeCompilationError

        report = {
            "schema": "merlin.native_selection_result.v1",
            "engine": args.engine,
            "status": (
                "unsupported_semantics"
                if isinstance(error, LinalgBridgeError)
                else error.status
                if isinstance(error, NativeCompilationError)
                else "compile_error"
            ),
            "scope": "selection_only",
            "reason": str(error),
        }
    _write_status(output, report)
    print(json.dumps({"status": report["status"], "engine": args.engine, "out": str(output)}, sort_keys=True))
    return 0 if report["status"] == "selected" else 2


def _cmd_native_compile(args: argparse.Namespace) -> int:
    """Invoke the actual builder with the historical loader/status patch hooks."""
    if getattr(args, "application", False):
        if (
            getattr(args, "diagnostic_graph_policy", None)
            or getattr(args, "physical_layouts", None)
            or any(
                getattr(args, name, None) is not None
                for name in (
                    "diagnostic_region",
                    "diagnostic_tile_index",
                    "ordered_k_tile",
                    "ordered_m_tile",
                    "ordered_n_tile",
                    "outer_tile_index",
                )
            )
        ):
            report = {
                "schema": "merlin.native_compilation_status.v1",
                "status": "compile_error",
                "engine": args.engine,
                "support": args.support,
                "out": str(args.out),
                "reason": (
                    "diagnostic graph precision cannot be combined with --application"
                    if getattr(args, "diagnostic_graph_policy", None)
                    else "physical boundary layouts cannot be combined with --application"
                    if getattr(args, "physical_layouts", None)
                    else "diagnostic tiles and ordered partitions cannot be combined with --application"
                ),
            }
            if args.status_file and not Path(args.status_file).resolve().is_relative_to(Path(args.out).resolve()):
                _write_status(Path(args.status_file), report)
            print(json.dumps(report, sort_keys=True))
            return 2
        from .native_application import compile_command

        return compile_command(args, write_status=_write_status, search_limits=_load_native_search_limits)
    from merlin.compiler.build.native import compile_report

    report = compile_report(
        args, load_request=_load_native_request, load_limits=_load_native_search_limits, write_status=_write_status
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "emitted" else 2


def _cmd_native_bridge_graph(args: argparse.Namespace) -> int:
    """Publish the checked graph boundary through the ordinary builder."""
    from merlin.compiler.build.graphbridge import publish_graph_bridge

    status, response = publish_graph_bridge(
        args.linalg,
        args.entry,
        compiled=args.compiled,
        device_symbol=args.device_symbol,
        out=args.out,
    )
    print(json.dumps(response, sort_keys=True))
    return status


def _cmd_native_verify(args: argparse.Namespace) -> int:
    if (
        getattr(args, "application", False)
        and not getattr(args, "diagnostic_graph_policy", None)
        and not getattr(args, "physical_layouts", None)
        and not any(
            getattr(args, name, None) is not None
            for name in (
                "diagnostic_region",
                "diagnostic_tile_index",
                "ordered_k_tile",
                "ordered_m_tile",
                "ordered_n_tile",
                "outer_tile_index",
            )
        )
    ):
        from .native_application_cli import verify_command

        return verify_command(args)
    from merlin.verification.native import verify_report

    report, safe_status = verify_report(args, load_request=_load_native_request)
    if safe_status:
        _write_status(Path(args.out), report)
    print(json.dumps({"status": report["status"], "engine": args.engine, "out": str(args.out)}, sort_keys=True))
    return 0 if report["status"] == "verified_artifacts" else 2


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
    a.add_argument(
        "--formats",
        help="the model's instruction-format module (default: the modules the ISA definition imports, found beside it)",
    )
    a.add_argument("--rtl-revision", required=True, help="exact selected RTL commit")
    a.add_argument("--model-revision", help="exact selected model commit")
    a.add_argument(
        "--verify-revisions",
        action="store_true",
        help="require selected RTL/model bytes to match both local Git HEADs and their commit objects",
    )
    a.add_argument("--out", required=True, help="census JSON artifact")
    a.set_defaults(func=_cmd_audit_isa)

    modes = sub.add_parser("audit-dialect-modes", help="account for every selected decoder mode in an OOT ledger")
    modes.add_argument("--census", required=True, help="output of audit-isa for the selected source bytes")
    modes.add_argument("--inventory", required=True, help="explicitly selected OOT JSON mode ledger")
    modes.add_argument("--dialect-plan", help="reviewed dialect plan YAML from the same selected target")
    modes.add_argument(
        "--require",
        choices=("phase1-inputs", "complete-dialect"),
        default="complete-dialect",
        help=(
            "phase1-inputs freezes source/mode scope and reviewed finite parameter domains; "
            "complete-dialect additionally requires typed bindings and declared admission"
        ),
    )
    modes.add_argument("--out", required=True, help="mode audit JSON artifact")
    modes.set_defaults(func=_cmd_audit_dialect_modes)

    source_scope = sub.add_parser(
        "audit-dialect-source-scope",
        help="bind a reviewed decoder-mode scope to reproduced selected RTL elaboration",
    )
    source_scope.add_argument("--source-selection", required=True, help="selected FIRRTL-to-HW source bundle")
    source_scope.add_argument("--census", required=True, help="exact source ISA census")
    source_scope.add_argument("--inventory", required=True, help="reviewed OOT mode ledger")
    source_scope.add_argument("--expected-config", required=True, help="campaign configuration symbol")
    source_scope.add_argument("--out", required=True, help="source-scope report JSON")
    source_scope.set_defaults(func=_cmd_audit_dialect_source_scope)

    stage = sub.add_parser("stage-capture", help="stage IR, weights and signature without evaluator inputs")
    stage.add_argument("--capture", required=True, help="materialized model capture directory")
    stage.add_argument("--out", required=True, help="fresh compiler-only directory")
    stage.add_argument("--status-file", help="invocation-owned machine-readable status JSON")
    stage.add_argument("--precision-policy", help="explicit source-bound numerical preparation policy")
    stage.set_defaults(func=_cmd_stage_capture)

    from .native_codegen_cli import add_arguments as add_native_arguments

    add_native_arguments(
        sub,
        {
            "_cmd_native_bridge_graph": _cmd_native_bridge_graph,
            "_cmd_native_build": _cmd_native_build,
            "_cmd_native_compile": _cmd_native_compile,
            "_cmd_native_select": _cmd_native_select,
            "_cmd_native_verify": _cmd_native_verify,
        },
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
