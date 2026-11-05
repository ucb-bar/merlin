"""Bind source-bound native region results to outlined host-call symbols."""

from __future__ import annotations

import hashlib
import json
from io import StringIO

from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region
from xdsl.printer import Printer

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.application_graph import _program_graph
from merlin.xdsl_dialects.lowering.outline import externalize_dispatches, outline_dispatches


def diagnostic_whole_graph_host_bridge(
    program: str,
    entry: str,
    *,
    device_symbol: str,
    compilation: dict,
    selection: dict,
    precision_plan: dict,
    execution_plan: dict,
    physical_layouts: dict | None = None,
) -> tuple[dict, str]:
    """Outline a selected whole tensor graph as one typed host call.

    Binding v2 physical boundary layouts changes the compiled request digest
    after the precision plan was made; the tracked layout document then
    carries the exact pre-layout request digest the plan must match.

    This is a source and ABI handoff, not an implementation of the external
    call. The target package supplies its C-interface callback and must check
    physical input/output bindings before the resulting object is executable.
    """
    if (not entry or not device_symbol or entry == device_symbol
            or not (device_symbol[0].isascii() and
                    (device_symbol[0].isalpha() or device_symbol[0] == "_"))
            or any(not char.isascii() or not (char.isalnum() or char == "_")
                   for char in device_symbol)):
        raise ValueError("whole-graph bridge needs a distinct C-compatible device symbol")
    source_sha256 = hashlib.sha256(program.encode()).hexdigest()
    request_digest = compilation.get("request_digest")
    target_identity = compilation.get("target_identity")
    selected_outputs = selection.get("source_outputs")
    graph_outputs = selection.get("graph_outputs")
    conversions = precision_plan.get("boundary_conversions")
    output_addresses = execution_plan.get("output_addresses")
    execution_sha256 = hashlib.sha256(
        (json.dumps(execution_plan, sort_keys=True) + "\n").encode()
    ).hexdigest()
    tracked = compilation.get("stage_artifacts")
    layouts_tracked = isinstance(tracked, dict) and "physical-boundary-layouts.json" in tracked
    if physical_layouts is None:
        if layouts_tracked:
            raise ValueError("whole-graph bridge needs the tracked physical boundary layouts")
        plan_request_digest = request_digest
    else:
        if (not layouts_tracked or not isinstance(physical_layouts, dict)
                or physical_layouts.get("schema") != "merlin.physical_boundary_layouts.v2"
                or physical_layouts.get("target_identity") != target_identity
                or not isinstance(physical_layouts.get("request_digest"), str)
                or not physical_layouts["request_digest"]):
            raise ValueError("whole-graph physical layouts do not bind an exact precision request")
        plan_request_digest = physical_layouts["request_digest"]
    if (compilation.get("engine") != "merlin_native"
            or compilation.get("artifact_kind") != "program"
            or compilation.get("qualification_scope") != "diagnostic_only"
            or not isinstance(request_digest, str) or not request_digest
            or not isinstance(target_identity, str) or not target_identity
            or compilation.get("execution_plan_sha256") != execution_sha256
            or not isinstance(compilation.get("binary_sha256"), str)
            or selection.get("engine") != "merlin_native"
            or selection.get("request_digest") != request_digest
            or selection.get("target_identity") != target_identity
            or not isinstance(compilation.get("native_check_fingerprint"), str)
            or selection.get("check_fingerprint") != compilation.get("native_check_fingerprint")
            or precision_plan.get("request_digest") != plan_request_digest
            or precision_plan.get("source_sha256") != source_sha256
            or not isinstance(selected_outputs, list) or not selected_outputs
            or not isinstance(graph_outputs, list)
            or len(graph_outputs) != len(selected_outputs)
            or not isinstance(conversions, list)
            or not isinstance(output_addresses, list)
            or len(output_addresses) != len(selected_outputs)
            or any(type(address) is not int or address < 0 for address in output_addresses)
            or len(set(output_addresses)) != len(output_addresses)
            or any(not isinstance(row, dict) for row in conversions)
            or [row.get("source") for row in conversions[-len(selected_outputs):]] != selected_outputs):
        raise ValueError("whole-graph selection differs from the source or physical ABI")

    source = parse_mlir_text(program)
    source.verify()
    definitions = [operation for operation in source.body.block.ops
                   if isinstance(operation, func.FuncOp) and operation.sym_name.data == entry]
    if len(definitions) != 1 or not definitions[0].body.blocks:
        raise ValueError("whole-graph source needs one defined entry function")
    signature = definitions[0].function_type
    inputs = list(signature.inputs.data)
    outputs = list(signature.outputs.data)
    if len(outputs) != len(selected_outputs) or len(conversions) < len(inputs) + len(outputs):
        raise ValueError("whole-graph source results differ from the selected graph")

    declaration = func.FuncOp.external(device_symbol, inputs, outputs)
    declaration.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    block = Block(arg_types=inputs)
    call = func.CallOp(device_symbol, list(block.args), outputs)
    block.add_op(call)
    block.add_op(func.ReturnOp(*call.results))
    forward = func.FuncOp(entry, (tuple(inputs), tuple(outputs)), Region([block]))
    forward.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    module = builtin.ModuleOp([declaration, forward])
    module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(module)
    return {
        "schema": "merlin.diagnostic_whole_graph_host_bridge.v1",
        "status": "typed_declarations_only",
        "engine": "merlin_native",
        "source_sha256": source_sha256,
        "request_digest": request_digest,
        "target_identity": target_identity,
        "program_sha256": compilation["binary_sha256"],
        "entry": entry,
        "device_symbol": device_symbol,
        "c_interface_symbol": "_mlir_ciface_" + device_symbol,
        "input_types": [str(item) for item in inputs],
        "result_types": [str(item) for item in outputs],
        "source_outputs": selected_outputs,
        "output_addresses": output_addresses,
        "strict_native_eligible": False,
        "linked": False,
        "executed": False,
    }, stream.getvalue() + "\n"


def diagnostic_host_bridge(program: str, entry: str, regions: list[dict],
                           named_tiles: list[dict] | None = None, *,
                           batched_regions: list[dict] | None = None,
                           batched_tiles: list[dict] | None = None) -> tuple[dict, str]:
    """Return typed host IR with materializable native regions externalized.

    This only binds source identities and function types. It does not provide
    device call implementations, numerical conversions, or a qualified host
    route for the region bodies retained in the module.
    """
    classified = {
        "emitted", "unsupported_semantics", "compile_error", "resource_limit", "search_timeout",
        "infeasible_candidate", "tool_unavailable", "unqualified_target", "numerical_mismatch",
        "execution_failure",
    }
    static_states = {"packed_or_runtime", "blocked", "unknown", "not_emitted"}
    batch_rows = batched_regions or []
    all_rows = [*regions, *batch_rows]
    if not all_rows or any(
        row.get("status") not in classified or row.get("static_inputs_status") not in static_states
        or (row["status"] == "emitted" and row["static_inputs_status"] == "not_emitted")
        or (row["status"] != "emitted" and row["static_inputs_status"] != "not_emitted")
        for row in regions
    ):
        raise ValueError("host bridge needs classified native region results")
    if any(
        row.get("status") not in {
            "unrouted_precision_and_tiling", "isolated_tiles_emitted",
            "isolated_tile_failure", *classified,
        }
        or type(row.get("operation_ordinal")) is not int
        for row in batch_rows
    ):
        raise ValueError("host bridge needs classified batched region results")
    ids = [row.get("region_id") for row in all_rows]
    if any(not isinstance(rid, str) or not rid for rid in ids) or len(set(ids)) != len(ids):
        raise ValueError("host bridge needs unique source region IDs")
    tile_rows = named_tiles or []
    batch_tile_rows = batched_tiles or []
    if any(not isinstance(tile, dict) or tile.get("region_id") not in {
        row["region_id"] for row in regions
    } for tile in tile_rows) or any(
        not isinstance(tile, dict) or tile.get("region_id") not in {
            row["region_id"] for row in batch_rows
        } for tile in batch_tile_rows
    ):
        raise ValueError("host bridge has tiles outside its source regions")

    def selected_route(row: dict, *, batched: bool) -> str:
        if (not batched and row["status"] == "emitted"
                and row["static_inputs_status"] == "packed_or_runtime"):
            return "direct_native"
        if (row["status"] if batched else row.get("tiling_status")) != "isolated_tiles_emitted":
            return "retained_host"
        selected_tiles = [
            tile for tile in (batch_tile_rows if batched else tile_rows)
            if tile["region_id"] == row["region_id"]
        ]
        count = row.get("tile_count")
        if (type(count) is not int or count < (1 if batched else 2)
                or len(selected_tiles) != count
                or any(type(tile.get("tile_index")) is not int for tile in selected_tiles)
                or sorted(tile.get("tile_index") for tile in selected_tiles) != list(range(count))
                or any(tile.get("status") != "emitted"
                       or tile.get("operation_id") != row.get("operation_id")
                       or not isinstance(tile.get("request_digest"), str)
                       or not isinstance(tile.get("binary_sha256"), str)
                       or not isinstance(tile.get("artifact"), str)
                       for tile in selected_tiles)):
            raise ValueError("host bridge has no complete selected source tile grid")
        # Like a direct region, a grid is callable only when every tile's
        # static inputs are packed or supplied at run time.
        if any(tile.get("static_inputs_status") != "packed_or_runtime" for tile in selected_tiles):
            return "retained_host"
        return "tiled_native"

    batched_ids = {row["region_id"] for row in batch_rows}
    routes = {
        row["region_id"]: selected_route(row, batched=row["region_id"] in batched_ids)
        for row in all_rows
    }

    source = parse_mlir_text(program)
    source_graph = _program_graph(source, hashlib.sha256(program.encode()).hexdigest())
    from merlin.common import mlir_query as mq
    from merlin.semantic_compiler.precision_intake import survey_source_f32_batched_matmuls

    parsed_ops = list(mq.walk(source))
    if len(parsed_ops) != len(source_graph["operations"]):
        raise ValueError("host bridge source graph changed during parsing")
    operation_ids = {
        id(op): row["operation_id"]
        for op, row in zip(parsed_ops, source_graph["operations"], strict=True)
    }
    functions = [
        op for op in source.body.block.ops
        if isinstance(op, func.FuncOp) and op.sym_name.data == entry and op.body.blocks
    ]
    if len(functions) != 1 or len(functions[0].body.blocks) != 1:
        raise ValueError("host bridge needs one parsed entry block")
    top_level = tuple(functions[0].body.block.ops)
    source_regions: dict[str, str] = {}
    for operation in source_graph["operations"]:
        if operation["mlir_operation"] != "linalg.matmul":
            continue
        region_id = operation["provenance"].get("prov.region_id")
        if region_id in ids:
            if region_id in source_regions:
                raise ValueError(f"source region {region_id} has multiple parsed matmuls")
            source_regions[region_id] = operation["operation_id"]
    surveyed, _ = survey_source_f32_batched_matmuls(program, entry=entry)
    if {
        (item.region_id, item.operation_ordinal) for item in surveyed
    } != {
        (row["region_id"], row["operation_ordinal"]) for row in batch_rows
    }:
        raise ValueError("batched host bridge regions differ from the parsed source")
    for row in batch_rows:
        ordinal = row["operation_ordinal"]
        if not 0 <= ordinal < len(top_level) or top_level[ordinal].name != "linalg.generic":
            raise ValueError("batched host bridge has no parsed source operation")
        source_regions[row["region_id"]] = operation_ids[id(top_level[ordinal])]
    if set(source_regions) != set(ids) or any(
        row.get("operation_id") != source_regions[row["region_id"]] for row in all_rows
    ):
        raise ValueError("compiled region operation IDs differ from the exact parsed model")

    outlined = outline_dispatches(source, forward=entry)
    roots = [
        op for op in top_level
        if op.name.startswith("linalg.") and op.name not in {
            "linalg.fill", "linalg.yield", "linalg.index",
        }
    ]
    if len(roots) != len(outlined.dispatches) or any(
        op.name != dispatch.root_op
        for op, dispatch in zip(roots, outlined.dispatches, strict=True)
    ):
        raise ValueError("source roots differ from outlined dispatches")
    by_operation = {
        operation_ids[id(op)]: dispatch
        for op, dispatch in zip(roots, outlined.dispatches, strict=True)
    }
    by_region = {
        rid: by_operation[operation_id]
        for rid, operation_id in source_regions.items()
    }
    if any(
        dispatch.prov.get("prov.region_id") not in {None, rid}
        for rid, dispatch in by_region.items()
    ):
        raise ValueError("outlined source region identity changed")
    if set(by_region) != set(ids):
        raise ValueError("source regions and outlined dispatches differ")

    definitions = {
        operation.sym_name.data: operation
        for operation in outlined.module.body.block.ops
        if isinstance(operation, func.FuncOp)
    }
    ordinal_by_id = {operation_ids[id(op)]: ordinal for ordinal, op in enumerate(top_level)}
    all_rows.sort(key=lambda row: ordinal_by_id[row["operation_id"]])
    selected = [
        by_region[row["region_id"]].symbol for row in all_rows
        if routes[row["region_id"]] != "retained_host"
    ]
    bridge = externalize_dispatches(outlined, selected)
    callable_symbols = {*selected, entry}
    for operation in bridge.module.body.block.ops:
        if isinstance(operation, func.FuncOp) and operation.sym_name.data in callable_symbols:
            operation.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    bridge.module.verify()
    stream = StringIO()
    Printer(stream=stream, print_generic_format=True).print_op(bridge.module)
    bindings = []
    for row in all_rows:
        dispatch = by_region[row["region_id"]]
        signature = definitions[dispatch.symbol].function_type
        selected_tiles = batch_tile_rows if row["region_id"] in batched_ids else tile_rows
        bindings.append({
            "region_id": row["region_id"],
            "operation_id": row.get("operation_id"),
            "symbol": dispatch.symbol,
            "native_status": row["status"],
            "selected_route": routes[row["region_id"]],
            "tile_request_digests": [
                tile["request_digest"] for tile in sorted(
                (tile for tile in selected_tiles if tile["region_id"] == row["region_id"]),
                    key=lambda tile: tile["tile_index"],
                )
            ] if routes[row["region_id"]] == "tiled_native" else [],
            "static_inputs_status": row["static_inputs_status"],
            "input_types": [str(item) for item in signature.inputs.data],
            "result_types": [str(item) for item in signature.outputs.data],
            "request_digest": row.get("request_digest") if row["status"] == "emitted" else None,
        })
    return {
        "schema": "merlin.diagnostic_model_host_bridge.v1",
        "status": "typed_declarations_only",
        "entry": entry,
        "entry_c_interface_symbol": entry,
        "external_symbols": selected,
        "c_interface_symbols": selected,
        "bindings": bindings,
        "required_native_unmet": [
            row["region_id"] for row in all_rows
            if routes[row["region_id"]] == "retained_host"
        ],
        "strict_native_eligible": False,
        "linked": False,
        "executed": False,
    }, stream.getvalue() + "\n"
