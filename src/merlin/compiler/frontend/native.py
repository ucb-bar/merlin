"""Compiler-only native request intake with exact staged source binding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_native_request(args: argparse.Namespace, *, target_identity: str, mode: str):
    """Read compiler inputs only; bind parsed Linalg to the selected target."""
    from merlin.compiler.frontend.inputs import load_compile_inputs, load_lifted_argument_bytes
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
    from merlin.semantic_compiler.model import KernelRequest

    if getattr(args, "compile_inputs", None):
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


def load_selected_request(args, *, target_identity, mode, binding=None, load_request=_load_native_request):
    """Load unchanged inputs or the explicit source-bound diagnostic graph route."""
    from merlin.compiler.frontend.inputs import load_compile_inputs
    from merlin.compiler.quantization.native import prepare_graph_request, prepare_tile_request
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError

    graph_policy = getattr(args, "diagnostic_graph_policy", None)
    if graph_policy is None:
        region_id = getattr(args, "diagnostic_region", None)
        tile_index = getattr(args, "diagnostic_tile_index", None)
        whole = getattr(args, "whole_contraction", False)
        if region_id is not None:
            if not getattr(args, "compile_inputs", None) or mode != "diagnostic" or binding is None:
                raise LinalgBridgeError(
                    "source-bound precision tiles require staged compiler inputs, target support, and diagnostic mode"
                )
            if getattr(args, "linalg_entry", None):
                raise ValueError("--linalg-entry cannot override a staged capture signature")
            staged = load_compile_inputs(args.compile_inputs)
            return prepare_tile_request(
                staged, region_id, binding=binding, target_identity=target_identity, mode=mode,
                tile_index=tile_index, whole_contraction=whole,
            )
        if whole:
            raise LinalgBridgeError("--whole-contraction requires --diagnostic-region")
        if tile_index is not None:
            raise LinalgBridgeError("--diagnostic-tile-index requires --diagnostic-region")
        request, kind, operations = load_request(args, target_identity=target_identity, mode=mode)
        return request, kind, operations, None
    if (getattr(args, "whole_contraction", False) or getattr(args, "diagnostic_region", None) is not None
            or getattr(args, "diagnostic_tile_index", None) is not None):
        raise LinalgBridgeError("graph precision policy cannot select a contraction or tile")
    if mode != "diagnostic" or binding is None:
        raise LinalgBridgeError("graph precision policy requires diagnostic mode and target support")
    if not callable(getattr(binding, "prepare_precision_graph", None)):
        raise LinalgBridgeError("selected target support has no graph precision lowering")
    policy = json.loads(Path(graph_policy).read_text())
    staged_graph = bool(getattr(args, "compile_inputs", None))
    original_source_sha256 = None
    if staged_graph:
        staged = load_compile_inputs(args.compile_inputs)
        if staged.precision_policy != policy or args.linalg or args.linalg_entry:
            raise LinalgBridgeError("staged graph precision must match its compiler-input policy and entry")
        source, entry = staged.program, staged.entry
        original_source_sha256 = staged.original_source_sha256
    else:
        if not args.linalg or not args.linalg_entry:
            raise LinalgBridgeError("graph precision policy requires --linalg/--linalg-entry or --compile-inputs")
        source, entry = Path(args.linalg).read_bytes().decode("utf-8"), args.linalg_entry
    request, operations, plan = prepare_graph_request(
        source, entry=entry, policy=policy, binding=binding, target_identity=target_identity,
        mode=mode, original_source_sha256=original_source_sha256,
    )
    kind = "source_bound_diagnostic_staged_graph" if staged_graph else "source_bound_diagnostic_graph"
    return request, kind, operations, plan
