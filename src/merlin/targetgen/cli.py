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
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

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
    from merlin.frontends.compile_inputs import stage_compile_inputs

    safe_status = bool(args.status_file) and not (
        Path(args.status_file).resolve().is_relative_to(Path(args.out).resolve())
        or Path(args.status_file).resolve().is_relative_to(Path(args.capture).resolve())
    )
    try:
        if args.status_file and not safe_status:
            raise ValueError("capture staging status file must be outside capture and compiler inputs")
        result = stage_compile_inputs(args.capture, args.out, precision_policy=args.precision_policy)
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


def _load_native_request(args: argparse.Namespace, *, target_identity: str, mode: str):
    """Read compiler inputs only; bind parsed Linalg to the selected target."""
    from merlin.frontends.compile_inputs import load_compile_inputs, load_lifted_argument_bytes
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
    from merlin.semantic_compiler.model import KernelRequest

    if args.compile_inputs:
        if args.linalg_entry:
            raise ValueError("--linalg-entry cannot override a staged capture signature")
        staged = load_compile_inputs(args.compile_inputs)
        if staged.precision_policy is not None:
            if "matmul_regions" in staged.precision_policy:
                from merlin.semantic_compiler.precision_intake import check_source_bound_f32_matmuls

                check_source_bound_f32_matmuls(
                    staged.program,
                    entry=staged.entry,
                    policy=staged.precision_policy,
                    original_source_sha256=staged.original_source_sha256,
                )
            raise LinalgBridgeError("selected precision policy has no admitted native transformation")
        translated = translate_linalg_text(
            staged.program,
            entry=staged.entry,
            target_identity=target_identity,
            lowering_policy=mode,
            argument_constants=load_lifted_argument_bytes(args.compile_inputs, staged),
        )
        return translated.request, "staged_capture", translated.source_operations
    if args.linalg:
        if not args.linalg_entry:
            raise ValueError("--linalg-entry is required with --linalg")
        source = Path(args.linalg).read_bytes().decode("utf-8")
        translated = translate_linalg_text(
            source,
            entry=args.linalg_entry,
            target_identity=target_identity,
            lowering_policy=mode,
        )
        return translated.request, "parsed_linalg", translated.source_operations
    if args.linalg_entry:
        raise ValueError("--linalg-entry requires --linalg")
    request = KernelRequest.from_record(json.loads(Path(args.request).read_text()))
    return request, "typed_request", ()


def _load_selected_native_request(args: argparse.Namespace, *, target_identity: str, mode: str, binding=None):
    """Load a normal request or an explicit source-bound diagnostic tile."""
    from merlin.frontends.compile_inputs import load_compile_inputs
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.precision_transform import DiagnosticMatmulTile

    region_id = getattr(args, "diagnostic_region", None)
    tile_index = getattr(args, "diagnostic_tile_index", None)
    whole = getattr(args, "whole_contraction", False)
    graph_policy = getattr(args, "diagnostic_graph_policy", None)
    if graph_policy is not None:
        if whole or region_id is not None or tile_index is not None:
            raise LinalgBridgeError("graph precision policy cannot select a contraction or tile")
        if mode != "diagnostic" or binding is None:
            raise LinalgBridgeError("graph precision policy requires diagnostic mode and target support")
        prepare = getattr(binding, "prepare_precision_graph", None)
        if not callable(prepare):
            raise LinalgBridgeError("selected target support has no graph precision lowering")
        policy = json.loads(Path(graph_policy).read_text())
        staged_graph = bool(args.compile_inputs)
        if staged_graph:
            staged = load_compile_inputs(args.compile_inputs)
            if staged.precision_policy != policy or args.linalg or args.linalg_entry:
                raise LinalgBridgeError("staged graph precision must match its compiler-input policy and entry")
            plan = prepare(
                staged.program, entry=staged.entry, policy=policy, original_source_sha256=staged.original_source_sha256
            )
        else:
            if not args.linalg or not args.linalg_entry:
                raise LinalgBridgeError("graph precision policy requires --linalg/--linalg-entry or --compile-inputs")
            source = Path(args.linalg).read_text()
            plan = prepare(source, entry=args.linalg_entry, policy=policy)
        prepared_request = getattr(plan, "request", None)
        source_operations = getattr(plan, "source_operations", None)
        if (
            not isinstance(prepared_request, KernelRequest)
            or prepared_request.target_identity != target_identity
            or prepared_request.lowering_policy != "diagnostic"
            or not isinstance(source_operations, tuple)
            or not all(isinstance(operation, str) for operation in source_operations)
            or not callable(getattr(plan, "record", None))
        ):
            raise LinalgBridgeError("target graph precision lowering returned an invalid request")
        kind = "source_bound_diagnostic_staged_graph" if staged_graph else "source_bound_diagnostic_graph"
        return prepared_request, kind, source_operations, plan
    if region_id is None:
        if whole:
            raise LinalgBridgeError("--whole-contraction requires --diagnostic-region")
        if tile_index is not None:
            raise LinalgBridgeError("--diagnostic-tile-index requires --diagnostic-region")
        request, source_kind, operations = _load_native_request(
            args,
            target_identity=target_identity,
            mode=mode,
        )
        return request, source_kind, operations, None
    if not args.compile_inputs or mode != "diagnostic" or binding is None:
        raise LinalgBridgeError(
            "source-bound precision tiles require staged compiler inputs, target support, and diagnostic mode"
        )
    staged = load_compile_inputs(args.compile_inputs)
    if staged.precision_policy_document is None:
        raise LinalgBridgeError("source-bound precision tile has no selected policy document")
    prepare = getattr(binding, "prepare_precision_tile", None)
    if not callable(prepare):
        raise LinalgBridgeError("selected target support has no source-bound precision tile")
    if whole and tile_index is not None:
        raise LinalgBridgeError("whole-contraction lowering cannot select an output tile index")
    if whole:
        plan = prepare(staged, region_id, whole_contraction=True)
    elif tile_index is None:
        plan = prepare(staged, region_id)
    else:
        plan = prepare(staged, region_id, tile_index=tile_index)
    if not isinstance(plan, DiagnosticMatmulTile) or plan.request.target_identity != target_identity:
        raise LinalgBridgeError("target precision tile returned an invalid request or target identity")
    if plan.request.lowering_policy != "diagnostic":
        raise LinalgBridgeError("source-bound precision tile must stay diagnostic")
    source_op = "linalg.generic" if plan.batch_index is not None else "linalg.matmul"
    if plan.whole_contraction != whole:
        raise LinalgBridgeError("target precision lowering changed the requested contraction scope")
    kind = "source_bound_diagnostic_contraction" if whole else "source_bound_diagnostic_tile"
    return plan.request, kind, (source_op,), plan


def _bind_native_physical_layouts(args: argparse.Namespace, request, precision_plan):
    """Bind a parsed source's declared physical boundary representation."""
    path = getattr(args, "physical_layouts", None)
    if path is None:
        return request, None
    if args.request or any(
        getattr(args, name, None) is not None
        for name in ("ordered_k_tile", "ordered_m_tile", "ordered_n_tile", "outer_tile_index")
    ):
        raise ValueError("physical layouts require parsed Linalg without ordered tile transformations")
    row = json.loads(Path(path).read_text())
    fields = {
        "schema",
        "target_identity",
        "source_identity",
        "physical_extents",
        "boundary_layouts",
        "output_layouts",
    }
    schema = row.get("schema") if isinstance(row, dict) else None
    if schema not in {"merlin.physical_boundary_layouts.v1", "merlin.physical_boundary_layouts.v2"} or (
        set(row) != (fields | {"request_digest"} if schema.endswith("v2") else fields)
    ):
        raise ValueError("physical layouts need the complete versioned boundary schema")
    if precision_plan is not None and schema.endswith("v1"):
        raise ValueError("precision-transformed layouts need the exact-request v2 schema")
    if schema.endswith("v2") and row["request_digest"] != request.digest():
        raise ValueError("physical layouts differ from the precision-transformed request")
    if row["target_identity"] != request.target_identity or row["source_identity"] != request.source_identity:
        raise ValueError("physical layouts differ from the parsed source or selected target")
    if (
        not isinstance(row["physical_extents"], dict)
        or not row["physical_extents"]
        or not isinstance(row["boundary_layouts"], dict)
        or not row["boundary_layouts"]
        or not isinstance(row["output_layouts"], list)
        or not row["output_layouts"]
    ):
        raise ValueError("physical layouts need typed extents, input layouts, and ordered output layouts")
    bound = replace(
        request,
        physical_extents=tuple(sorted(row["physical_extents"].items())),
        boundary_layouts=tuple(sorted(row["boundary_layouts"].items())),
        output_layouts=tuple(row["output_layouts"]),
    )
    canonical = {
        **row,
        "physical_extents": dict(bound.physical_extents),
        "boundary_layouts": dict(bound.boundary_layouts),
        "output_layouts": list(bound.output_layouts),
    }
    return bound, (json.dumps(canonical, sort_keys=True, indent=2) + "\n").encode()


def _prepare_native_row_softmax(args: argparse.Namespace, binding, source_request):
    """Select one row group while retaining every column of its normalization."""
    from merlin.semantic_compiler.target_binding import NativeCompilationError
    from merlin.semantic_compiler.tiled_softmax import RowSoftmaxTiling

    index = getattr(args, "row_softmax_tile_index", None)
    if index is None:
        return source_request, None
    if any(
        getattr(args, name, None) is not None
        for name in (
            "ordered_k_tile",
            "ordered_m_tile",
            "ordered_n_tile",
            "outer_tile_index",
            "diagnostic_region",
            "physical_layouts",
        )
    ):
        raise ValueError("row softmax tiling cannot combine with another precision or tile transformation")
    prepare = getattr(binding, "prepare_row_softmax", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected support has no row softmax decomposition")
    plan = prepare(source_request)
    if (
        not isinstance(plan, RowSoftmaxTiling)
        or plan.source != source_request
        or type(index) is not int
        or not 0 <= index < len(plan.regions)
    ):
        raise ValueError("row softmax decomposition or row-group index differs from source")
    selected = plan.regions[index].request
    if (
        selected.target_identity != source_request.target_identity
        or selected.lowering_policy != source_request.lowering_policy
    ):
        raise ValueError("row softmax decomposition changed target or compile mode")
    record = {**plan.record(), "selected_row_index": index, "selected_request_digest": selected.digest()}
    return selected, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


def _prepare_native_ordered_k(args: argparse.Namespace, binding, source_request):
    """Ask the selected target to qualify exact ordered-K partitioning."""
    from merlin.semantic_compiler.ordered_k_tiling import OrderedKTiling
    from merlin.semantic_compiler.target_binding import NativeCompilationError

    tile_k = getattr(args, "ordered_k_tile", None)
    if tile_k is None:
        return source_request, None
    prepare = getattr(binding, "prepare_ordered_k", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected target has no qualified ordered-K tiling")
    prepared = prepare(source_request, tile_k)
    if (
        not isinstance(prepared, OrderedKTiling)
        or prepared.request.target_identity != source_request.target_identity
        or prepared.request.lowering_policy != source_request.lowering_policy
        or prepared.request.source_identity != source_request.source_identity
    ):
        raise ValueError("target returned an invalid ordered-K partition")
    record = {
        **prepared.record(),
        "source_request_digest": source_request.digest(),
        "tile_k": tile_k,
    }
    return prepared.request, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


def _prepare_native_ordered_mn(args: argparse.Namespace, binding, source_request):
    """Select one checked outer output tile from the selected target's geometry."""
    from merlin.semantic_compiler.ordered_mn_tiling import OrderedMNTiling
    from merlin.semantic_compiler.target_binding import NativeCompilationError

    tile_m = getattr(args, "ordered_m_tile", None)
    tile_n = getattr(args, "ordered_n_tile", None)
    index = getattr(args, "outer_tile_index", None)
    if tile_m is None and tile_n is None and index is None:
        return source_request, None
    if tile_m is None or tile_n is None or index is None:
        raise ValueError("ordered M/N tiling needs both geometry dimensions and an output tile index")
    prepare = getattr(binding, "prepare_ordered_mn", None) if binding is not None else None
    if not callable(prepare):
        raise NativeCompilationError("unsupported_semantics", "selected target has no qualified M/N tiling")
    prepared = prepare(source_request, tile_m, tile_n)
    if (
        not isinstance(prepared, OrderedMNTiling)
        or prepared.source.digest() != source_request.digest()
        or type(index) is not int
        or not 0 <= index < len(prepared.tiles)
    ):
        raise ValueError("target returned an invalid ordered M/N partition or tile index")
    selected = prepared.tiles[index].request
    if (
        selected.target_identity != source_request.target_identity
        or selected.lowering_policy != source_request.lowering_policy
        or selected.source_identity != source_request.source_identity
    ):
        raise ValueError("outer output tile changed source identity or compilation policy")
    record = {
        **prepared.record(),
        "selected_tile_index": index,
        "selected_request_digest": selected.digest(),
    }
    return selected, (json.dumps(record, sort_keys=True, indent=2) + "\n").encode()


def _prepared_static_inputs(binding, plan, staged_root: Path, execution_plan: dict):
    """Check the target's source-bound constant payloads before publication."""
    from merlin.semantic_compiler.precision_constants import PreparedStaticInput

    prepare = getattr(binding, "prepare_precision_constants", None)
    if not callable(prepare):
        raise ValueError("target cannot prepare source-bound static inputs")
    prepared = prepare(plan, staged_root, execution_plan)
    if not isinstance(prepared, tuple) or any(not isinstance(item, PreparedStaticInput) for item in prepared):
        raise ValueError("target returned malformed source-bound static inputs")
    roles = [item.role for item in prepared]
    allowed_roles = {conversion.source_role for conversion in plan.inputs}
    if len(set(roles)) != len(roles) or any(role not in allowed_roles for role in roles):
        raise ValueError("target returned repeated or unknown source-bound input roles")
    return prepared


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
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.search import SearchAblations
    from merlin.semantic_compiler.snapshot import open_native_snapshot
    from merlin.semantic_compiler.target_binding import NativeCompilationError, load_native_target_binding

    output = Path(args.out)
    try:
        snapshot = open_native_snapshot(Path(args.snapshot))
        binding = (
            load_native_target_binding(args.support)
            if (
                args.diagnostic_region
                or args.diagnostic_graph_policy
                or getattr(args, "row_softmax_tile_index", None) is not None
            )
            and args.support
            else None
        )
        if binding is not None and snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        request, source_kind, source_operations, precision_plan = _load_selected_native_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
        )
        request, _ = _bind_native_physical_layouts(args, request, precision_plan)
        request, softmax_record = _prepare_native_row_softmax(args, binding, request)
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
            "exploration_attempts": result.exploration_attempts,
            "candidate_attempts": result.candidate_attempts,
            "ordering_attempts": result.ordering_attempts,
            "rejected_allocation": result.rejected_allocation,
            "rejected_materialization": result.rejected_materialization,
            "pruned_orders": result.pruned_orders,
            "check_fingerprint": result.check_fingerprint,
            "reservations": [asdict(item) for item in reservations],
            "candidate_digest": result.candidate.digest() if result.candidate else None,
            "selected_graph": asdict(result.graph) if result.graph else None,
            "allocation": asdict(result.allocation) if result.allocation else None,
            "constant_requirements": constant_requirements,
        }
        if precision_plan is not None:
            report["precision_plan"] = precision_plan.record()
        if softmax_record is not None:
            report["row_softmax_tiling"] = json.loads(softmax_record)
        if result.status == "unsupported_semantics" and result.graph is None:
            report.pop("selected_graph")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
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


def _native_compile_report(args: argparse.Namespace) -> dict:
    from merlin.semantic_compiler.codegen import codegen_options, options_from_args

    with codegen_options(options_from_args(args)):
        return _native_compile_report_with_options(args)


def _native_compile_report_with_options(args: argparse.Namespace) -> dict:
    """Invoke a selected target binding with typed source and an explicit ABI."""
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.snapshot import open_native_snapshot
    from merlin.semantic_compiler.target_binding import (
        NativeCompilationError,
        load_native_target_binding,
        verify_native_publication,
    )

    try:
        if getattr(args, "application_state", None) and not getattr(args, "application", False):
            raise ValueError("--application-state requires --application")
        binding = load_native_target_binding(args.support)
        from merlin.semantic_compiler.codegen import current_codegen_options

        options = current_codegen_options()
        requested = {
            key
            for key, active in (
                ("reduction_unroll", options.reduction_unroll != "auto"),
                ("tuning_profile", options.tuning_profile is not None),
                ("program_size_limit_bytes", options.program_size_limit_bytes is not None),
            )
            if active
        }
        unsupported = requested - set(getattr(binding, "codegen_capabilities", ()))
        if unsupported:
            raise NativeCompilationError(
                "unsupported_semantics", "selected support does not accept: " + ", ".join(sorted(unsupported))
            )
        snapshot = open_native_snapshot(Path(args.snapshot))
        if snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        if getattr(args, "application", False):
            from .native_application import compile_application_report

            return compile_application_report(
                args,
                binding=binding,
                snapshot=snapshot,
                limits=_load_native_search_limits(args.search_limits),
                write_status=_write_status,
            )
        request, source_kind, source_operations, precision_plan = _load_selected_native_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
        )
        request, layout_document = _bind_native_physical_layouts(args, request, precision_plan)
        source_request = request
        request, softmax_record = _prepare_native_row_softmax(args, binding, request)
        if precision_plan is not None and (
            getattr(args, "ordered_k_tile", None) is not None
            or getattr(args, "ordered_m_tile", None) is not None
            or getattr(args, "ordered_n_tile", None) is not None
            or getattr(args, "outer_tile_index", None) is not None
        ):
            raise ValueError("source-bound precision tiles cannot also use ordered matrix partitioning")
        request, ordered_mn_record = _prepare_native_ordered_mn(args, binding, request)
        k_source_request = request
        request, ordered_k_record = _prepare_native_ordered_k(args, binding, request)
        if request.lowering_policy != args.mode:
            raise ValueError("explicit compile mode differs from typed request")
        output = Path(args.out)
        if args.status_file and Path(args.status_file).resolve().is_relative_to(output.resolve()):
            raise ValueError("native compile status file must be outside the artifact directory")
        if args.abi == "auto":
            planner = getattr(binding, "plan_boundary_abi", None)
            if planner is None:
                raise NativeCompilationError(
                    "unsupported_semantics", "selected support has no automatic boundary layout"
                )
            abi = planner(request)
        else:
            abi = json.loads(Path(args.abi).read_text())
        if (
            set(abi) != {"fixed_inputs", "fixed_outputs"}
            or not isinstance(abi["fixed_inputs"], dict)
            or (not isinstance(abi["fixed_outputs"], list))
        ):
            raise ValueError("native compile ABI needs fixed_inputs and fixed_outputs")
        if not all(type(value) is int for value in abi["fixed_inputs"].values()) or (
            not all(type(value) is int for value in abi["fixed_outputs"])
        ):
            raise ValueError("native compile ABI addresses must be integers")
        limits = _load_native_search_limits(args.search_limits)
        if output.exists():
            raise FileExistsError(f"fresh native compilation output required: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="native-compile-", dir=output.parent) as temporary:
            staged = Path(temporary) / "compilation"
            manifest = binding.compile(
                snapshot,
                request,
                fixed_inputs=abi["fixed_inputs"],
                fixed_outputs=tuple(abi["fixed_outputs"]),
                target_source=Path(args.target_source),
                destination=staged,
                limits=limits,
            )
            if softmax_record is not None:
                name = "row-softmax-tiling.json"
                tracked = manifest.get("stage_artifacts")
                if not isinstance(tracked, dict) or name in tracked or (staged / name).exists():
                    raise ValueError("target manifest cannot track row softmax decomposition")
                (staged / name).write_bytes(softmax_record)
                manifest = {
                    **manifest,
                    "softmax_source_request_digest": source_request.digest(),
                    "stage_artifacts": {**tracked, name: hashlib.sha256(softmax_record).hexdigest()},
                }
                (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            if layout_document is not None:
                tracked = manifest.get("stage_artifacts")
                name = "physical-boundary-layouts.json"
                if not isinstance(tracked, dict) or name in tracked or (staged / name).exists():
                    raise ValueError("target manifest cannot track physical boundary layouts")
                payload = layout_document
                (staged / name).write_bytes(payload)
                manifest = {
                    **manifest,
                    "stage_artifacts": {
                        **tracked,
                        name: hashlib.sha256(payload).hexdigest(),
                    },
                }
                (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            if precision_plan is not None:
                execution_plan = json.loads((staged / "execution_plan.json").read_text())
                from merlin.semantic_compiler.precision_transform import DiagnosticMatmulTile

                tile_plan = isinstance(precision_plan, DiagnosticMatmulTile)
                static_inputs = (
                    _prepared_static_inputs(binding, precision_plan, Path(args.compile_inputs), execution_plan)
                    if tile_plan
                    else ()
                )
                raw_plan = (json.dumps(precision_plan.record(), sort_keys=True, indent=2) + "\n").encode()
                (staged / "precision-plan.json").write_bytes(raw_plan)
                tracked = manifest.get("stage_artifacts")
                if (
                    not isinstance(tracked, dict)
                    or "precision-plan.json" in tracked
                    or (tile_plan and "constant_inputs" in manifest)
                ):
                    raise ValueError("target manifest cannot track the source precision plan")
                constant_artifacts = {}
                for item in static_inputs:
                    if item.payload is None:
                        continue
                    relative = item.record()["path"]
                    destination = staged / relative
                    destination.parent.mkdir(exist_ok=True)
                    if destination.exists() or destination.is_symlink():
                        raise ValueError("target constant destination already exists")
                    if relative in tracked:
                        raise ValueError("target manifest already tracks a source constant")
                    destination.write_bytes(item.payload)
                    constant_artifacts[relative] = hashlib.sha256(item.payload).hexdigest()
                manifest = {
                    **manifest,
                    "stage_artifacts": {
                        **tracked,
                        "precision-plan.json": hashlib.sha256(raw_plan).hexdigest(),
                        **constant_artifacts,
                    },
                }
                if tile_plan:
                    manifest["constant_inputs"] = [item.record() for item in static_inputs]
                (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            if ordered_k_record is not None:
                tracked = manifest.get("stage_artifacts")
                if not isinstance(tracked, dict) or "ordered-k-tiling.json" in tracked:
                    raise ValueError("target manifest cannot track the ordered-K partition")
                (staged / "ordered-k-tiling.json").write_bytes(ordered_k_record)
                manifest = {
                    **manifest,
                    "source_request_digest": k_source_request.digest(),
                    "stage_artifacts": {
                        **tracked,
                        "ordered-k-tiling.json": hashlib.sha256(ordered_k_record).hexdigest(),
                    },
                }
                (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            if ordered_mn_record is not None:
                tracked = manifest.get("stage_artifacts")
                if not isinstance(tracked, dict) or "ordered-mn-tiling.json" in tracked:
                    raise ValueError("target manifest cannot track the ordered-M/N partition")
                (staged / "ordered-mn-tiling.json").write_bytes(ordered_mn_record)
                manifest = {
                    **manifest,
                    "outer_source_request_digest": source_request.digest(),
                    "stage_artifacts": {
                        **tracked,
                        "ordered-mn-tiling.json": hashlib.sha256(ordered_mn_record).hexdigest(),
                    },
                }
                (staged / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
            verify_native_publication(
                staged,
                manifest,
                engine=args.engine,
                request_digest=request.digest(),
                target_identity=snapshot.profile.target_identity,
            )
            report = {
                "schema": "merlin.native_compilation_status.v1",
                "status": "emitted",
                "engine": args.engine,
                "support": args.support,
                "out": str(args.out),
                "request_digest": request.digest(),
                "target_identity": snapshot.profile.target_identity,
                "source_kind": source_kind,
                "source_identity": request.source_identity,
                "source_operations": list(source_operations),
                "search_limits": limits.record(),
                "artifact_kind": manifest.get("artifact_kind", "program"),
            }
            if precision_plan is not None:
                report["precision_plan_path"] = str(output / "precision-plan.json")
                report["constant_inputs"] = manifest["constant_inputs"]
            if ordered_k_record is not None:
                report["source_request_digest"] = k_source_request.digest()
                report["ordered_k_tiling_path"] = str(output / "ordered-k-tiling.json")
            if ordered_mn_record is not None:
                report["outer_source_request_digest"] = source_request.digest()
                report["ordered_mn_tiling_path"] = str(output / "ordered-mn-tiling.json")
            if manifest.get("artifact_kind") == "program_set":
                report["program_count"] = len(manifest["segments"])
                report["segment_binary_sha256"] = [row["binary_sha256"] for row in manifest["segments"]]
            else:
                report["binary_sha256"] = manifest["binary_sha256"]
            staged.rename(output)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        status = (
            "unsupported_semantics"
            if isinstance(error, LinalgBridgeError)
            else error.status
            if isinstance(error, NativeCompilationError)
            else "compile_error"
        )
        report = {
            "schema": "merlin.native_compilation_status.v1",
            "status": status,
            "engine": args.engine,
            "support": args.support,
            "out": str(args.out),
            "reason": str(error),
        }
    if args.status_file and not Path(args.status_file).resolve().is_relative_to(Path(args.out).resolve()):
        _write_status(Path(args.status_file), report)
    return report


def _cmd_native_compile(args: argparse.Namespace) -> int:
    report = _native_compile_report(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "emitted" else 2


def _cmd_native_probe_model_regions(args: argparse.Namespace) -> int:
    from .model_region_probe import probe_model_regions

    return probe_model_regions(
        args,
        compile_report=_native_compile_report,
        load_limits=_load_native_search_limits,
    )


def _cmd_native_check_host_object(args: argparse.Namespace) -> int:
    """Check the compiled host object's exact native-call roster."""
    from merlin.targetgen.contract.model_host_object import check_compiled_host_object

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        print(json.dumps({"status": "compile_error", "reason": f"fresh output required: {output}"}, sort_keys=True))
        return 2
    output.mkdir(parents=True)
    try:
        report = check_compiled_host_object(
            probe_dir=Path(args.probe),
            compile_inputs=Path(args.compile_inputs),
            lowered_dir=Path(args.lowered),
            object_path=Path(args.object),
            nm=Path(args.nm),
            mlir_opt=Path(args.mlir_opt),
        )
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, UnicodeError, subprocess.TimeoutExpired) as error:
        report = {"schema": "merlin.native_host_object_check.v1", "status": "compile_error", "reason": str(error)}
    (output / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "out": str(output)}, sort_keys=True))
    return 0 if report["status"] == "object_symbols_checked" else 2


def _cmd_native_bridge_graph(args: argparse.Namespace) -> int:
    """Write a typed host call for one exact native whole-graph artifact."""
    from merlin.semantic_compiler.target_binding import verify_native_publication
    from merlin.targetgen.contract.model_host_bridge import diagnostic_whole_graph_host_bridge

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        print(json.dumps({"status": "compile_error", "reason": "fresh output directory required"}))
        return 2
    compiled = Path(args.compiled)
    # Refuse before anything is written: the error manifest below must never
    # land inside the compiled artifact it failed to bridge.
    if output.resolve().is_relative_to(compiled.resolve()):
        print(
            json.dumps(
                {"status": "compile_error", "reason": "host bridge output must be outside the native compilation"}
            )
        )
        return 2
    try:
        manifest_path = compiled / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("native compilation manifest is linked")
        compilation = json.loads(manifest_path.read_text())
        if not isinstance(compilation, dict):
            raise ValueError("native compilation manifest must be an object")
        verify_native_publication(
            compiled,
            compilation,
            engine="merlin_native",
            request_digest=compilation.get("request_digest"),
            target_identity=compilation.get("target_identity"),
        )
        stage_artifacts = compilation.get("stage_artifacts")
        # verify_native_publication checks execution_plan.json separately by its
        # manifest digest; the other graph stages are listed in stage_artifacts.
        required = {"native-selection.json", "precision-plan.json"}
        if not isinstance(stage_artifacts, dict) or not required <= stage_artifacts.keys():
            raise ValueError("native graph stages are not tracked by the compilation manifest")
        selection = json.loads((compiled / "native-selection.json").read_text())
        precision_plan = json.loads((compiled / "precision-plan.json").read_text())
        execution_plan = json.loads((compiled / "execution_plan.json").read_text())
        layouts_name = "physical-boundary-layouts.json"
        physical_layouts = (
            json.loads((compiled / layouts_name).read_text()) if layouts_name in stage_artifacts else None
        )
        record, text = diagnostic_whole_graph_host_bridge(
            Path(args.linalg).read_text(),
            args.entry,
            device_symbol=args.device_symbol,
            compilation=compilation,
            selection=selection,
            precision_plan=precision_plan,
            execution_plan=execution_plan,
            physical_layouts=physical_layouts,
        )
        output.mkdir(parents=True)
        (output / "host-bridge.mlir").write_text(text)
        record["host_bridge_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        record["native_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (output / "manifest.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        print(json.dumps({"status": record["status"], "out": str(output)}, sort_keys=True))
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, UnicodeError) as error:
        output.mkdir(parents=True, exist_ok=True)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "schema": "merlin.diagnostic_whole_graph_host_bridge.v1",
                    "status": "compile_error",
                    "reason": str(error),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"status": "compile_error", "reason": str(error), "out": str(output)}, sort_keys=True))
        return 2


def _cmd_native_verify(args: argparse.Namespace) -> int:
    """Reopen a native package against its source and snapshot identities."""
    if getattr(args, "application", False):
        from .native_application import verify_application_command

        return verify_application_command(args, _write_status)
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError
    from merlin.semantic_compiler.snapshot import open_native_snapshot
    from merlin.semantic_compiler.target_binding import (
        NativeCompilationError,
        load_native_target_binding,
        verify_native_publication,
    )

    artifact = Path(args.artifact)
    status_path = Path(args.out)
    safe_status = not (
        status_path.resolve().is_relative_to(artifact.resolve())
        or status_path.resolve().is_relative_to(Path(args.snapshot).resolve())
    )
    try:
        if getattr(args, "application_state", None):
            raise ValueError("--application-state requires --application")
        if not safe_status:
            raise ValueError("native verification status must be outside the artifact and snapshot directories")
        snapshot = open_native_snapshot(Path(args.snapshot))
        binding = (
            load_native_target_binding(args.support)
            if args.support
            and (
                args.diagnostic_region
                or args.diagnostic_graph_policy
                or args.ordered_k_tile is not None
                or args.ordered_m_tile is not None
                or args.ordered_n_tile is not None
                or args.outer_tile_index is not None
                or args.row_softmax_tile_index is not None
            )
            else None
        )
        if binding is not None and snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        request, source_kind, _, precision_plan = _load_selected_native_request(
            args,
            target_identity=snapshot.profile.target_identity,
            mode=args.mode,
            binding=binding,
        )
        request, layout_document = _bind_native_physical_layouts(args, request, precision_plan)
        source_request = request
        request, softmax_record = _prepare_native_row_softmax(args, binding, request)
        if precision_plan is not None and (
            args.ordered_k_tile is not None
            or args.ordered_m_tile is not None
            or args.ordered_n_tile is not None
            or args.outer_tile_index is not None
        ):
            raise ValueError("source-bound precision tiles cannot also use ordered matrix partitioning")
        request, ordered_mn_record = _prepare_native_ordered_mn(args, binding, request)
        k_source_request = request
        request, ordered_k_record = _prepare_native_ordered_k(args, binding, request)
        if request.target_identity != snapshot.profile.target_identity or request.lowering_policy != args.mode:
            raise ValueError("native verification source differs from selected target or compile mode")
        manifest_bytes = (artifact / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict):
            raise ValueError("native artifact manifest must be an object")
        softmax_path = artifact / "row-softmax-tiling.json"
        if softmax_record is None:
            if softmax_path.exists() or softmax_path.is_symlink() or "softmax_source_request_digest" in manifest:
                raise ValueError("native artifact has an unrequested row softmax decomposition")
        elif (
            manifest.get("softmax_source_request_digest") != source_request.digest()
            or softmax_path.is_symlink()
            or not softmax_path.is_file()
            or softmax_path.read_bytes() != softmax_record
        ):
            raise ValueError("row softmax decomposition differs from original source")
        snapshot_digest = hashlib.sha256((snapshot.root / "manifest.json").read_bytes()).hexdigest()
        if manifest.get("native_snapshot_sha256") != snapshot_digest:
            raise ValueError("native artifact snapshot identity differs")
        if manifest.get("compile_mode") != args.mode:
            raise ValueError("native artifact compile mode differs")
        ordered_k_path = artifact / "ordered-k-tiling.json"
        if ordered_k_record is None:
            if ordered_k_path.exists() or "source_request_digest" in manifest:
                raise ValueError("native artifact has an unrequested ordered-K partition")
        elif (
            manifest.get("source_request_digest") != k_source_request.digest()
            or ordered_k_path.is_symlink()
            or not ordered_k_path.is_file()
            or ordered_k_path.read_bytes() != ordered_k_record
        ):
            raise ValueError("ordered-K partition differs from the original source")
        ordered_mn_path = artifact / "ordered-mn-tiling.json"
        if ordered_mn_record is None:
            if ordered_mn_path.exists() or "outer_source_request_digest" in manifest:
                raise ValueError("native artifact has an unrequested ordered-M/N partition")
        elif (
            manifest.get("outer_source_request_digest") != source_request.digest()
            or ordered_mn_path.is_symlink()
            or not ordered_mn_path.is_file()
            or ordered_mn_path.read_bytes() != ordered_mn_record
        ):
            raise ValueError("ordered-M/N partition differs from the original source")
        verify_native_publication(
            artifact,
            manifest,
            engine=args.engine,
            request_digest=request.digest(),
            target_identity=snapshot.profile.target_identity,
        )
        layout_path = artifact / "physical-boundary-layouts.json"
        if layout_document is not None:
            if layout_path.is_symlink() or not layout_path.is_file() or (layout_path.read_bytes() != layout_document):
                raise ValueError("physical boundary layouts differ from parsed source")
        elif layout_path.exists() or layout_path.is_symlink():
            raise ValueError("native artifact has undeclared physical boundary layouts")
        plan_path = artifact / "precision-plan.json"
        if precision_plan is not None:
            expected_plan = (json.dumps(precision_plan.record(), sort_keys=True, indent=2) + "\n").encode()
            if plan_path.is_symlink() or not plan_path.is_file() or plan_path.read_bytes() != expected_plan:
                raise ValueError("source precision plan differs from staged compiler inputs")
            execution_plan = json.loads((artifact / "execution_plan.json").read_text())
            from merlin.semantic_compiler.precision_transform import DiagnosticMatmulTile

            static_inputs = (
                _prepared_static_inputs(binding, precision_plan, Path(args.compile_inputs), execution_plan)
                if isinstance(precision_plan, DiagnosticMatmulTile)
                else ()
            )
            if isinstance(precision_plan, DiagnosticMatmulTile) and (
                manifest.get("constant_inputs") != [item.record() for item in static_inputs]
            ):
                raise ValueError("source-bound constant inputs differ from staged compiler inputs")
            for item in static_inputs:
                if item.payload is not None and (artifact / item.record()["path"]).read_bytes() != item.payload:
                    raise ValueError("source-bound constant payload differs from staged parameter")
        elif plan_path.exists() or plan_path.is_symlink():
            raise ValueError("native artifact has an undeclared source precision plan")
        stage_artifacts = manifest.get("stage_artifacts")
        selection_path = artifact / "native-selection.json"
        if manifest.get("native_check_fingerprint") is not None:
            if not isinstance(stage_artifacts, dict) or "native-selection.json" not in stage_artifacts:
                raise ValueError("native selected graph is not tracked by the artifact manifest")
            selection = json.loads(selection_path.read_text())
            if not isinstance(selection, dict) or any(
                selection.get(key) != expected
                for key, expected in (
                    ("engine", args.engine),
                    ("request_digest", request.digest()),
                    ("target_identity", snapshot.profile.target_identity),
                    ("check_fingerprint", manifest.get("native_check_fingerprint")),
                )
            ):
                raise ValueError("native selected graph identity differs from artifact manifest")
        elif selection_path.exists():
            raise ValueError("native selected graph has no checked fingerprint")
        report = {
            "schema": "merlin.native_artifact_verification.v1",
            "status": "verified_artifacts",
            "scope": "source_snapshot_declared_artifact_identity",
            "engine": args.engine,
            "artifact": str(artifact),
            "source_kind": source_kind,
            "request_digest": request.digest(),
            "target_identity": snapshot.profile.target_identity,
            "binary_sha256": manifest.get("binary_sha256"),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "checked_stage_artifacts": sorted(stage_artifacts) if isinstance(stage_artifacts, dict) else [],
            "selected_record_checked": manifest.get("native_check_fingerprint") is not None,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        report = {
            "schema": "merlin.native_artifact_verification.v1",
            "status": (
                "unsupported_semantics"
                if isinstance(error, LinalgBridgeError)
                else error.status
                if isinstance(error, NativeCompilationError)
                else "compile_error"
            ),
            "scope": "source_snapshot_declared_artifact_identity",
            "engine": args.engine,
            "artifact": str(artifact),
            "reason": str(error),
        }
    if safe_status:
        _write_status(status_path, report)
    print(json.dumps({"status": report["status"], "engine": args.engine, "out": str(status_path)}, sort_keys=True))
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
    stage.add_argument("--precision-policy", help="separately selected JSON policy bound to captured source bytes")
    stage.add_argument("--status-file", help="invocation-owned machine-readable status JSON")
    stage.set_defaults(func=_cmd_stage_capture)

    native_build = sub.add_parser("native-build", help="build an offline Merlin-native selection snapshot")
    native_build.add_argument("--engine", choices=("merlin_native",), required=True)
    native_build.add_argument("--profile", help="versioned native target profile JSON")
    native_build.add_argument("--support", help="explicit installed native target support provider")
    native_build.add_argument("--crate", help="override the packaged Merlin egg bridge source directory")
    native_build.add_argument("--cargo-target-dir", required=True)
    native_build.add_argument("--source-revision", required=True)
    native_build.add_argument("--out", required=True, help="fresh snapshot directory")
    native_build.set_defaults(func=_cmd_native_build)

    from .native_codegen_cli import add_native_codegen_commands

    add_native_codegen_commands(
        sub,
        {
            "native_select": _cmd_native_select,
            "native_compile": _cmd_native_compile,
            "native_probe_model_regions": _cmd_native_probe_model_regions,
            "native_check_host_object": _cmd_native_check_host_object,
            "native_bridge_graph": _cmd_native_bridge_graph,
            "native_verify": _cmd_native_verify,
        },
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
