"""Source-bound model region compilation diagnostics.

This probe records isolated native tiles and source SSA gaps. It cannot issue a
whole-model execution result until a separate composition route is checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path


def _tile_count(region, first, *, batched: bool) -> int:
    """Derive the finite output grid from the selected tile boundary."""
    output = first.output
    axes = output.target_axes_from_source
    target_shape = output.target_type.shape
    batch_shape = region.batch_shape if batched else ()
    expected_batch = (0,) * len(batch_shape) if batched else None
    if (
        first.region_id != region.region_id
        or first.batch_index != expected_batch
        or output.source_type.shape != region.result_shape
        or output.source_value != region.result_source
        or output.source_batch_index != (expected_batch or ())
        or output.source_offsets != (0, 0)
        or axes not in ((0, 1), (1, 0))
        or len(target_shape) != 2
        or any(type(extent) is not int or extent <= 0 for extent in target_shape)
    ):
        raise ValueError("first source tile differs from the selected region")
    tile_m, tile_n = (target_shape[axes.index(axis)] for axis in (0, 1))
    return (math.prod(batch_shape)
            * ((region.result_shape[-2] + tile_m - 1) // tile_m)
            * ((region.result_shape[-1] + tile_n - 1) // tile_n))


def _static_inputs_status(result: dict) -> str:
    """Classify whether an emitted image's static inputs are host-callable."""
    static_inputs = result.get("constant_inputs")
    return (
        "not_emitted"
        if result["status"] != "emitted"
        else "unknown"
        if not isinstance(static_inputs, list) or any(not isinstance(item, dict) for item in static_inputs)
        else "blocked"
        if any(item.get("status") != "packed" for item in static_inputs)
        else "packed_or_runtime"
    )


def _load_abi_pool(path: str | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    raw = Path(path).read_bytes()
    pool = json.loads(raw)
    if (
        not isinstance(pool, dict) or set(pool) != {"input_slots", "output_slots"}
        or any(not isinstance(pool[key], list) or not pool[key] for key in pool)
        or any(type(slot) is not int or slot < 0 for key in pool for slot in pool[key])
        or len(set(pool["input_slots"] + pool["output_slots"]))
        != len(pool["input_slots"]) + len(pool["output_slots"])
    ):
        raise ValueError("diagnostic ABI pool needs distinct nonnegative input and output slots")
    return pool, hashlib.sha256(raw).hexdigest()


def _expected_tile_windows(
    region, first, *, batched: bool,
) -> dict[tuple[tuple[int, ...], tuple[int, int]], tuple[int, int]]:
    axes = first.output.target_axes_from_source
    shape = first.output.target_type.shape
    tile_m, tile_n = (shape[axes.index(axis)] for axis in (0, 1))
    batches = product(*(range(extent) for extent in region.batch_shape)) if batched else ((),)
    return {
        (batch, (m, n)): (min(tile_m, region.result_shape[-2] - m),
                           min(tile_n, region.result_shape[-1] - n))
        for batch in batches
        for m in range(0, region.result_shape[-2], tile_m)
        for n in range(0, region.result_shape[-1], tile_n)
    }


def _checked_emitted_tile_window(
    artifact: Path, region, first, result: dict,
    expected: dict[tuple[tuple[int, ...], tuple[int, int]], tuple[int, int]],
) -> tuple[tuple[int, ...], tuple[int, int]]:
    plan = json.loads((artifact / "precision-plan.json").read_text())
    output = plan["output"]
    key = (tuple(plan.get("batch_index", ())), tuple(output["source_offsets"]))
    if (
        plan["region_id"] != region.region_id
        or plan["request_digest"] != result.get("request_digest")
        or output.get("source_batch_index", []) != list(key[0])
        or output["source_type"]["shape"] != list(region.result_shape)
        or output["source_value"] != region.result_source.record()
        or output["target_type"]["shape"] != list(first.output.target_type.shape)
        or output["target_axes_from_source"] != list(first.output.target_axes_from_source)
        or key not in expected
        or output["source_window_shape"] != list(expected[key])
    ):
        raise ValueError("emitted tile precision plan differs from the source output grid")
    return key


def probe_model_regions(args: argparse.Namespace, *, compile_report, load_limits) -> int:
    """Compile declared source-bound tiles and retain the full SSA gap roster.

    Isolated region emission does not establish a model execution route. This
    diagnostic always reports incomplete until a separate composition stage
    binds transfers, host work, output values, and invocation state.
    """
    from merlin.common import mlir_query as mq
    from merlin.frontends.compile_inputs import load_compile_inputs
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.semantic_compiler.linalg_bridge import LinalgBridgeError, translate_linalg_text
    from merlin.semantic_compiler.model import KernelRequest
    from merlin.semantic_compiler.precision_intake import (
        check_source_bound_f32_batched_matmuls,
        check_source_bound_f32_matmuls,
        survey_source_f32_batched_matmuls,
    )
    from merlin.semantic_compiler.snapshot import open_native_snapshot
    from merlin.semantic_compiler.target_binding import NativeCompilationError, load_native_target_binding
    from merlin.targetgen.application_graph import application_graph_inventory
    from merlin.targetgen.contract.model_host_bridge import diagnostic_host_bridge
    from merlin.targetgen.contract.model_stitching import stitching_inventory

    output = Path(args.out)
    if output.exists() or output.is_symlink():
        print(json.dumps({"status": "compile_error", "reason": f"fresh output required: {output}"}, sort_keys=True))
        return 2
    output.mkdir(parents=True)
    try:
        staged = load_compile_inputs(args.compile_inputs)
        if staged.precision_policy is None:
            raise ValueError("model region probe needs an explicit source-bound precision policy")
        snapshot = open_native_snapshot(Path(args.snapshot))
        binding = load_native_target_binding(args.support)
        if snapshot.profile.digest() != binding.profile().digest():
            raise ValueError("native snapshot differs from selected support profile")
        # The pool selects every tile ABI, so an exact ABI beside it would be
        # recorded without being used.
        if bool(args.abi) == bool(getattr(args, "abi_pool", None)):
            raise ValueError("model region probe needs exactly one of --abi or --abi-pool")
        abi_sha256 = hashlib.sha256(Path(args.abi).read_bytes()).hexdigest() if args.abi else None
        abi_pool, abi_pool_sha256 = _load_abi_pool(getattr(args, "abi_pool", None))
        prepare_tile = getattr(binding, "prepare_precision_tile", None)

        def compile_tile(member: str, region_id: str, tile_index: int | None,
                         request_args: argparse.Namespace) -> tuple[dict, str | None]:
            if abi_pool is None:
                return compile_report(request_args), None
            try:
                if not callable(prepare_tile):
                    raise NativeCompilationError(
                        "unqualified_target", "selected support has no source-bound tile preparation",
                    )
                plan = (prepare_tile(staged, region_id) if tile_index is None
                        else prepare_tile(staged, region_id, tile_index=tile_index))
                roles = tuple(item.source_role for item in plan.inputs)
                outputs = len(plan.request.outputs)
                if (
                    not roles or any(type(role) is not str or not role for role in roles)
                    or len(set(roles)) != len(roles)
                    or outputs <= 0
                    or len(roles) > len(abi_pool["input_slots"])
                    or outputs > len(abi_pool["output_slots"])
                ):
                    raise NativeCompilationError(
                        "resource_limit", "source tile does not fit the diagnostic ABI pool",
                    )
                abi = {
                    "fixed_inputs": dict(zip(roles, abi_pool["input_slots"][:len(roles)], strict=True)),
                    "fixed_outputs": abi_pool["output_slots"][:outputs],
                }
                relative = f"abis/{member}.json"
                path = output / relative
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(abi, sort_keys=True) + "\n")
                request_args.abi = str(path)
            except (NativeCompilationError, OSError, ValueError, TypeError, AttributeError) as error:
                return {
                    "status": ("unsupported_semantics" if isinstance(error, LinalgBridgeError)
                               else error.status if isinstance(error, NativeCompilationError)
                               else "compile_error"),
                    "reason": str(error),
                }, None
            return compile_report(request_args), relative

        search_limits = load_limits(args.search_limits).record()
        regions = check_source_bound_f32_matmuls(
            staged.program,
            entry=staged.entry,
            policy=staged.precision_policy,
            original_source_sha256=staged.original_source_sha256,
        )
        batched_policy_selected = "batched_matmul_regions" in staged.precision_policy
        # Only a selected batched policy must cover every contraction; a route
        # listing keeps a noncanonical one as an unrouted source operation.
        if batched_policy_selected:
            batched_regions = check_source_bound_f32_batched_matmuls(
                staged.program,
                entry=staged.entry,
                policy=staged.precision_policy,
                original_source_sha256=staged.original_source_sha256,
            )
            refused_batched: tuple[tuple[int, str], ...] = ()
        else:
            batched_regions, refused_batched = survey_source_f32_batched_matmuls(
                staged.program, entry=staged.entry,
            )
        refused_reason = dict(refused_batched)
        graph = application_graph_inventory(Path(args.compile_inputs) / "program.mlir")
        operations = list(mq.walk(parse_mlir_text(staged.program)))
        rows = graph["capture_graph"]["operations"]
        if len(operations) != len(rows) or any(
            mq.op_name(operation) != row["mlir_operation"] for operation, row in zip(operations, rows, strict=True)
        ):
            raise ValueError("parsed source operations differ from the SSA inventory")
        identities = {id(operation): row["operation_id"] for operation, row in zip(operations, rows, strict=True)}
        functions = [
            operation
            for operation in operations
            if mq.op_name(operation) == "func.func" and operation.sym_name.data == staged.entry
        ]
        if len(functions) != 1 or len(functions[0].body.blocks) != 1:
            raise ValueError("model region probe needs one parsed entry block")
        top_level = tuple(functions[0].body.block.ops)
        row_by_id = {row["operation_id"]: row for row in rows}
        region_by_ordinal = {
            region.operation_ordinal: region for region in (*regions, *batched_regions)
        }
        if len(region_by_ordinal) != len(regions) + len(batched_regions):
            raise ValueError("named and batched source contractions overlap")
        source_rows = []
        for ordinal, operation in enumerate(top_level):
            source_row = row_by_id[identities[id(operation)]]
            region = region_by_ordinal.get(ordinal)
            source_rows.append(
                {
                    "operation_id": source_row["operation_id"],
                    "operation_ordinal": ordinal,
                    "mlir_operation": source_row["mlir_operation"],
                    "region_id": region.region_id if region else None,
                    "route": "native_region_candidate"
                    if region
                    else "model_output"
                    if mq.op_name(operation) == "func.return"
                    else "unrouted",
                    "operand_value_ids": [item["value_id"] for item in source_row["operands"]],
                    "result_value_ids": [item["value_id"] for item in source_row["results"]],
                    **({"unrouted_reason": refused_reason[ordinal]} if ordinal in refused_reason else {}),
                }
            )
        if sum(row["region_id"] is not None for row in source_rows) != (
            len(regions) + len(batched_regions)
        ):
            raise ValueError("source contraction roster differs from the SSA graph")
        batched_contractions = [
            {
                "region_id": region.region_id,
                "operation_id": source_rows[region.operation_ordinal]["operation_id"],
                "operation_ordinal": region.operation_ordinal,
                "batch_shape": list(region.batch_shape),
                "lhs_shape": list(region.lhs_shape),
                "rhs_shape": list(region.rhs_shape),
                "result_shape": list(region.result_shape),
                "scalar_add_order": region.scalar_add_order,
                "status": "unrouted_precision_and_tiling",
                "static_inputs_status": "not_emitted",
            }
            for region in batched_regions
        ]
        if any(source_rows[row["operation_ordinal"]]["route"] != "native_region_candidate"
               for row in batched_contractions):
            raise ValueError("batched contraction has no source candidate route")
        graph_bytes = (json.dumps(graph, sort_keys=True, indent=2) + "\n").encode()
        (output / "source-graph.json").write_bytes(graph_bytes)
        try:
            translation = translate_linalg_text(
                staged.program,
                entry=staged.entry,
                target_identity=snapshot.profile.target_identity,
                lowering_policy="diagnostic",
            )
        except LinalgBridgeError as error:
            semantic_graph = {"status": "unsupported_semantics", "reason": str(error)}
        else:
            request = translation.request
            if request.source_identity != staged.program_sha256 or (
                KernelRequest.from_record(request.record()).digest() != request.digest()
            ):
                raise ValueError("source semantic graph differs from staged source or round trip")
            if sum(node.op == "source_f32_matmul_unqualified" for node in request.nodes) != (
                len(regions) + len(batched_regions) + len(refused_batched)
            ):
                raise ValueError("source contraction inventory differs from translated semantic graph")
            semantic_bytes = (json.dumps(request.record(), sort_keys=True, indent=2) + "\n").encode()
            (output / "source-semantic-graph.json").write_bytes(semantic_bytes)
            semantic_graph = {
                "status": "translated_source_only",
                "path": "source-semantic-graph.json",
                "sha256": hashlib.sha256(semantic_bytes).hexdigest(),
                "request_digest": request.digest(),
                "node_count": len(request.nodes),
                "output_ids": list(request.outputs),
                "unqualified_node_count": sum(
                    node.op.startswith("source_") for node in request.nodes
                ),
            }
        compiled = []
        for index, region in enumerate(regions):
            member = f"region-{index:04d}"
            region_args = argparse.Namespace(
                engine=args.engine,
                support=args.support,
                snapshot=args.snapshot,
                request=None,
                linalg=None,
                compile_inputs=args.compile_inputs,
                linalg_entry=None,
                diagnostic_region=region.region_id,
                abi=args.abi,
                search_limits=args.search_limits,
                target_source=args.target_source,
                mode="diagnostic",
                out=str(output / "regions" / member),
                status_file=str(output / f"status-{member}.json"),
            )
            result, abi_path = compile_tile(member, region.region_id, None, region_args)
            static_inputs_status = _static_inputs_status(result)
            compiled.append(
                {
                    "region_id": region.region_id,
                    "operation_id": source_rows[region.operation_ordinal]["operation_id"],
                    "lhs_source": region.lhs_source.record(),
                    "rhs_source": region.rhs_source.record(),
                    "result_source": region.result_source.record(),
                    "status": result["status"],
                    "static_inputs_status": static_inputs_status,
                    "reason": result.get("reason"),
                    "artifact": f"regions/{member}" if result["status"] == "emitted" else None,
                    "request_digest": result.get("request_digest"),
                    "binary_sha256": result.get("binary_sha256"),
                    "abi_path": abi_path,
                }
            )
        named_tiles = []
        for region_index, (region, row) in enumerate(zip(regions, compiled, strict=True)):
            tile_count = 0
            try:
                if not callable(prepare_tile):
                    raise NativeCompilationError(
                        "unqualified_target", "selected support has no source-bound tile preparation",
                    )
                first = prepare_tile(staged, region.region_id, tile_index=0)
                tile_count = _tile_count(region, first, batched=False)
                if len(named_tiles) + tile_count > getattr(args, "max_named_tiles", 256):
                    raise NativeCompilationError(
                        "resource_limit", "named tile count exceeds the diagnostic work limit",
                    )
            except (NativeCompilationError, OSError, ValueError, TypeError, AttributeError) as error:
                row["tiling_status"] = error.status if isinstance(error, NativeCompilationError) else "compile_error"
                row["tiling_reason"] = str(error)
                row["tile_count"] = tile_count
                continue
            row["tile_count"] = tile_count
            if tile_count == 1:
                row["tiling_status"] = "single_tile"
                continue
            expected_windows = _expected_tile_windows(region, first, batched=False)
            seen_windows = set()
            for tile_index in range(tile_count):
                member = f"named-{region_index:04d}-tile-{tile_index:04d}"
                tile_args = argparse.Namespace(
                    engine=args.engine,
                    support=args.support,
                    snapshot=args.snapshot,
                    request=None,
                    linalg=None,
                    compile_inputs=args.compile_inputs,
                    linalg_entry=None,
                    diagnostic_region=region.region_id,
                    diagnostic_tile_index=tile_index,
                    abi=args.abi,
                    search_limits=args.search_limits,
                    target_source=args.target_source,
                    mode="diagnostic",
                    out=str(output / "regions" / member),
                    status_file=str(output / f"status-{member}.json"),
                )
                result, abi_path = compile_tile(member, region.region_id, tile_index, tile_args)
                if result["status"] == "emitted":
                    try:
                        window = _checked_emitted_tile_window(
                            output / "regions" / member, region, first, result, expected_windows,
                        )
                        if window in seen_windows:
                            raise ValueError("source output window was emitted twice")
                        seen_windows.add(window)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        result = {**result, "status": "compile_error", "reason": str(error)}
                named_tiles.append({
                    "region_id": region.region_id,
                    "operation_id": row["operation_id"],
                    "tile_index": tile_index,
                    "status": result["status"],
                    "static_inputs_status": _static_inputs_status(result),
                    "reason": result.get("reason"),
                    "artifact": f"regions/{member}" if result["status"] == "emitted" else None,
                    "request_digest": result.get("request_digest"),
                    "binary_sha256": result.get("binary_sha256"),
                    "abi_path": abi_path,
                })
            statuses = [tile["status"] for tile in named_tiles if tile["region_id"] == region.region_id]
            row["tiling_status"] = ("isolated_tiles_emitted" if all(status == "emitted" for status in statuses)
                                    and seen_windows == expected_windows.keys()
                                    else "isolated_tile_failure")
        batched_tiles = []
        if batched_policy_selected:
            for region_index, (region, row) in enumerate(zip(
                batched_regions, batched_contractions, strict=True,
            )):
                tile_count = 0
                try:
                    if not callable(prepare_tile):
                        raise NativeCompilationError(
                            "unqualified_target", "selected support has no source-bound tile preparation",
                        )
                    first = prepare_tile(staged, region.region_id, tile_index=0)
                    tile_count = _tile_count(region, first, batched=True)
                    if len(batched_tiles) + tile_count > getattr(args, "max_batched_tiles", 256):
                        raise NativeCompilationError(
                            "resource_limit", "batched tile count exceeds the diagnostic work limit",
                        )
                except (NativeCompilationError, OSError, ValueError, TypeError, AttributeError) as error:
                    row["status"] = error.status if isinstance(error, NativeCompilationError) else "compile_error"
                    row["reason"] = str(error)
                    row["tile_count"] = tile_count
                    continue
                row["tile_count"] = tile_count
                expected_windows = _expected_tile_windows(region, first, batched=True)
                seen_windows = set()
                for tile_index in range(tile_count):
                    member = f"batched-{region_index:04d}-tile-{tile_index:04d}"
                    tile_args = argparse.Namespace(
                        engine=args.engine,
                        support=args.support,
                        snapshot=args.snapshot,
                        request=None,
                        linalg=None,
                        compile_inputs=args.compile_inputs,
                        linalg_entry=None,
                        diagnostic_region=region.region_id,
                        diagnostic_tile_index=tile_index,
                        abi=args.abi,
                        search_limits=args.search_limits,
                        target_source=args.target_source,
                        mode="diagnostic",
                        out=str(output / "regions" / member),
                        status_file=str(output / f"status-{member}.json"),
                    )
                    result, abi_path = compile_tile(member, region.region_id, tile_index, tile_args)
                    if result["status"] == "emitted":
                        try:
                            window = _checked_emitted_tile_window(
                                output / "regions" / member, region, first, result, expected_windows,
                            )
                            if window in seen_windows:
                                raise ValueError("source output window was emitted twice")
                            seen_windows.add(window)
                        except (OSError, ValueError, KeyError, TypeError) as error:
                            result = {**result, "status": "compile_error", "reason": str(error)}
                    batched_tiles.append({
                        "region_id": region.region_id,
                        "operation_id": row["operation_id"],
                        "tile_index": tile_index,
                        "status": result["status"],
                        "static_inputs_status": _static_inputs_status(result),
                        "reason": result.get("reason"),
                        "artifact": f"regions/{member}" if result["status"] == "emitted" else None,
                        "request_digest": result.get("request_digest"),
                        "binary_sha256": result.get("binary_sha256"),
                        "abi_path": abi_path,
                    })
                statuses = [tile["status"] for tile in batched_tiles if tile["region_id"] == region.region_id]
                row["status"] = ("isolated_tiles_emitted" if all(status == "emitted" for status in statuses)
                                 and seen_windows == expected_windows.keys()
                                 else "isolated_tile_failure")
                if row["status"] == "isolated_tiles_emitted":
                    static_states = {
                        tile["static_inputs_status"] for tile in batched_tiles
                        if tile["region_id"] == region.region_id
                    }
                    row["static_inputs_status"] = (
                        "packed_or_runtime" if static_states == {"packed_or_runtime"}
                        else "blocked" if "blocked" in static_states else "unknown"
                    )
        counts = {
            status: sum(row["status"] == status for row in compiled)
            for status in sorted({row["status"] for row in compiled})
        }
        static_input_counts = {
            status: sum(row["static_inputs_status"] == status for row in compiled)
            for status in sorted({row["static_inputs_status"] for row in compiled})
        }
        stitching = stitching_inventory(
            parse_mlir_text(staged.program),
            staged.program_sha256,
            [
                {
                    "operation_id": row["operation_id"],
                    "request_digest": row.get("request_digest"),
                    "software_admission": {"status": "diagnostic_precision_transform"},
                    "compiler_support": (
                        f"emitted_{row['static_inputs_status']}_static_input"
                        if row["status"] == "emitted" and row["static_inputs_status"] != "packed_or_runtime"
                        else row["status"]
                    ),
                }
                for row in (*compiled, *batched_contractions)
            ],
        )
        stitching_bytes = (json.dumps(stitching, sort_keys=True, indent=2) + "\n").encode()
        (output / "stitching.json").write_bytes(stitching_bytes)
        try:
            bridge, bridge_text = diagnostic_host_bridge(
                staged.program, staged.entry, compiled, named_tiles,
                batched_regions=batched_contractions, batched_tiles=batched_tiles,
            )
        except (ValueError, RuntimeError) as error:
            bridge = {"status": "unavailable", "reason": str(error)}
        else:
            bridge_bytes = bridge_text.encode()
            (output / "host-bridge.mlir").write_bytes(bridge_bytes)
            bridge["path"] = "host-bridge.mlir"
            bridge["sha256"] = hashlib.sha256(bridge_bytes).hexdigest()
        report = {
            "schema": "merlin.native_model_region_probe.v1",
            "engine": args.engine,
            "status": "incomplete",
            "scope": "isolated_source_bound_regions",
            "mode": "diagnostic",
            "support": args.support,
            "target_identity": snapshot.profile.target_identity,
            "target_profile_digest": snapshot.profile.digest(),
            "abi_sha256": abi_sha256,
            "abi_pool_sha256": abi_pool_sha256,
            "search_limits": search_limits,
            "original_source_sha256": staged.original_source_sha256,
            "staged_program_sha256": staged.program_sha256,
            "precision_policy_sha256": staged.precision_policy_sha256,
            "source_graph_sha256": hashlib.sha256(graph_bytes).hexdigest(),
            "source_semantic_graph": semantic_graph,
            "stitching_sha256": hashlib.sha256(stitching_bytes).hexdigest(),
            "source_operation_count": len(source_rows),
            "parsed_operation_count": graph["capture_graph"]["n_operations"],
            "ssa_edge_count": len(graph["capture_graph"]["edges"]),
            "unrouted_operation_count": sum(row["route"] == "unrouted" for row in source_rows),
            "model_outputs": [row for row in source_rows if row["route"] == "model_output"],
            "source_operations": source_rows,
            "region_count": len(regions),
            "source_contraction_count": len(regions) + len(batched_contractions),
            "batched_contraction_count": len(batched_contractions),
            "batched_contractions": batched_contractions,
            "batched_tile_count": len(batched_tiles),
            "batched_tile_status_counts": {
                status: sum(tile["status"] == status for tile in batched_tiles)
                for status in sorted({tile["status"] for tile in batched_tiles})
            },
            "batched_tiles": batched_tiles,
            "named_tile_count": len(named_tiles),
            "named_tile_status_counts": {
                status: sum(tile["status"] == status for tile in named_tiles)
                for status in sorted({tile["status"] for tile in named_tiles})
            },
            "named_tiles": named_tiles,
            "region_status_counts": counts,
            "static_input_status_counts": static_input_counts,
            "boundary_crossing_count": len(stitching["candidate_boundary_crossings"]),
            "regions": compiled,
            "host_bridge": bridge,
            "composition_status": "not_constructed",
            "whole_model_executed": False,
        }
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError, UnicodeError) as error:
        report = {
            "schema": "merlin.native_model_region_probe.v1",
            "engine": args.engine,
            "status": (
                "unsupported_semantics"
                if isinstance(error, LinalgBridgeError)
                else error.status
                if isinstance(error, NativeCompilationError)
                else "compile_error"
            ),
            "scope": "isolated_source_bound_regions",
            "reason": str(error),
            "whole_model_executed": False,
        }
    (output / "manifest.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "out": str(output),
                "region_status_counts": report.get("region_status_counts", {}),
            },
            sort_keys=True,
        )
    )
    return 2
