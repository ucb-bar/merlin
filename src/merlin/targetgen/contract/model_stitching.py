"""Account for typed SSA crossings and logical composition around outlined kernels.

This is a diagnostic dataflow inventory.  In particular, an SSA edge does not
choose a memory placement, transfer implementation, dispatch ABI or host
lowering. The plan orders only single-block top-level operations; those missing
implementations must be supplied before a model can execute on a target.
"""

from __future__ import annotations

from merlin.targetgen.application_graph import _program_graph

SCHEMA = "merlin.model_stitching_inventory.v1"
PLAN_SCHEMA = "merlin.model_composition_plan.v1"


def _composition_plan(graph: dict, functions: list[dict], crossings: list[dict], candidates: list[dict]) -> dict:
    """Order logical host, kernel and boundary work without claiming a lowering.

    Only a single-block function has an established lexical execution order here.
    Other control flow and nested operations remain explicit obligations.
    """
    by_candidate = {candidate["operation_id"]: candidate for candidate in candidates}
    rows_by_id = {row["operation_id"]: row for row in graph["operations"]}
    planned_operation_ids = {
        step["operation_id"] for function in functions for step in function["top_level_steps"]
    }
    if not set(by_candidate).issubset(planned_operation_ids):
        raise ValueError("outlined candidate is not a direct function operation")
    by_consumer: dict[str, list[dict]] = {}
    for edge in crossings:
        by_consumer.setdefault(edge["consumer_operation_id"], []).append(edge)
    blocks_by_owner: dict[str, list[dict]] = {}
    for block in graph["blocks"]:
        blocks_by_owner.setdefault(block["owner_operation_id"], []).append(block)

    planned_functions = []
    planned_edge_ids: set[str] = set()
    obligations = [
        {"kind": "memory_placement_and_typed_transfers"},
        {"kind": "kernel_dispatch_and_pointer_abi"},
        {"kind": "whole_model_numerical_equivalence"},
    ]
    if candidates:
        obligations.append({"kind": "kernel_compilation_and_admission", "operation_ids": list(by_candidate)})
    for function in functions:
        function_id = function["operation_id"]
        blocks = blocks_by_owner.get(function_id, [])
        linear = len(blocks) == 1 and not any(
            rows_by_id[step["operation_id"]]["successor_block_ids"]
            for step in function["top_level_steps"]
        )
        steps = []
        if linear:
            for operation in function["top_level_steps"]:
                operation_id = operation["operation_id"]
                for edge in by_consumer.get(operation_id, []):
                    planned_edge_ids.add(edge["edge_id"])
                    steps.append(
                        {
                            "kind": "transfer",
                            "edge_id": edge["edge_id"],
                            "source_value_id": edge["source_value_id"],
                            "consumer_operation_id": operation_id,
                            "consumer_operand_index": edge["consumer_operand_index"],
                            "direction": edge["direction"],
                            "type": edge["type"],
                            "shape": edge["shape"],
                            "dtype": edge["dtype"],
                            "status": "unlowered",
                        }
                    )
                if operation_id in by_candidate:
                    candidate = by_candidate[operation_id]
                    steps.append(
                        {
                            "kind": "kernel",
                            "operation_id": operation_id,
                            "interface_sha256": candidate.get("interface_sha256"),
                            "request_digest": candidate.get("request_digest"),
                            "software_admission": candidate["software_admission"],
                            "compiler_support": candidate["compiler_support"],
                            "operand_value_ids": [item["value_id"] for item in operation["inputs"][:2]],
                            "result_value_id": operation["outputs"][0]["value_id"],
                            "status": "interface_uncompiled",
                        }
                    )
                elif operation["kind"] == "return_binding":
                    steps.append(
                        {
                            "kind": "return",
                            "operation_id": operation_id,
                            "value_ids": [item["value_id"] for item in operation["inputs"]],
                            "status": "unlowered",
                        }
                    )
                else:
                    steps.append(
                        {
                            "kind": "host",
                            "operation_id": operation_id,
                            "mlir_operation": operation["mlir_operation"],
                            "status": "unlowered",
                        }
                    )
        else:
            obligations.append({"kind": "control_flow_lowering", "function_operation_id": function_id})
        planned_functions.append(
            {
                "operation_id": function_id,
                "symbol": function["symbol"],
                "order": "single_block_lexical" if linear else "unestablished",
                "steps": steps,
                "unplanned_operation_ids": [] if linear else [
                    step["operation_id"] for step in function["top_level_steps"]
                ],
            }
        )
    unplanned_edges = [edge["edge_id"] for edge in crossings if edge["edge_id"] not in planned_edge_ids]
    if unplanned_edges:
        obligations.append({"kind": "boundary_order_lowering", "edge_ids": unplanned_edges})
    host_ids = [
        step["operation_id"]
        for function in functions for step in function["top_level_steps"]
        if step["kind"] == "host_unlowered"
    ]
    if host_ids:
        obligations.append({"kind": "host_operation_lowering", "operation_ids": host_ids})
    function_ids = {function["operation_id"] for function in functions}

    def inside_function(row: dict) -> bool:
        parent_id = row["parent_operation_id"]
        while parent_id is not None:
            if parent_id in function_ids:
                return True
            parent_id = rows_by_id[parent_id]["parent_operation_id"]
        return False

    outside = [
        row["operation_id"] for row in graph["operations"]
        if row["mlir_operation"] not in {"builtin.module", "func.func"} and not inside_function(row)
    ]
    if outside:
        obligations.append({"kind": "module_operation_lowering", "operation_ids": outside})
    return {
        "schema": PLAN_SCHEMA,
        "model_sha256": graph["mlir_sha256"],
        "status": "unlowered",
        "executable": False,
        "functions": planned_functions,
        "unplanned_boundary_edge_ids": unplanned_edges,
        "obligations": obligations,
    }


def stitching_inventory(module, model_sha256: str, candidates: list[dict]) -> dict:
    """Join exact outlined operation IDs to the parsed model's complete SSA graph."""
    graph = _program_graph(module, model_sha256)
    rows = graph["operations"]
    by_id = {row["operation_id"]: row for row in rows}
    selected = {candidate["operation_id"] for candidate in candidates}
    if len(selected) != len(candidates) or any(
        operation_id not in by_id or by_id[operation_id]["mlir_operation"] not in {"linalg.generic", "linalg.matmul"}
        for operation_id in selected
    ):
        raise ValueError("outlined candidates do not identify distinct parsed contractions")

    functions = []
    for function in rows:
        if function["mlir_operation"] != "func.func":
            continue
        function_id = function["operation_id"]
        steps = []
        for row in rows:
            if row["parent_operation_id"] != function_id:
                continue
            kind = "accelerator_candidate" if row["operation_id"] in selected else "host_unlowered"
            if row["mlir_operation"] == "func.return":
                kind = "return_binding"
            steps.append(
                {
                    "operation_id": row["operation_id"],
                    "ordinal": row["ordinal"],
                    "mlir_operation": row["mlir_operation"],
                    "kind": kind,
                    "source_node_ids": row["source_node_ids"],
                    "origin_node_ids": row["origin_node_ids"],
                    "inputs": row["operands"],
                    "outputs": row["results"],
                }
            )
        functions.append(
            {
                "operation_id": function_id,
                "symbol": function["properties"].get("sym_name", function["attributes"].get("sym_name")),
                "top_level_steps": steps,
                "return_values": [
                    operand for step in steps if step["kind"] == "return_binding" for operand in step["inputs"]
                ],
                "unlowered_host_operations": [
                    step["operation_id"] for step in steps if step["kind"] == "host_unlowered"
                ],
            }
        )

    crossings = []
    for edge in graph["edges"]:
        consumer_id = edge["consumer_operation_id"]
        producer_id = edge["producer_operation_id"]
        consumer = by_id[consumer_id]
        # An exact outlined contraction consumes operands 0 and 1.  Its zero
        # accumulator initializer is an implementation detail, not a transfer.
        into_kernel = consumer_id in selected and edge["operand_index"] in (0, 1)
        from_kernel = producer_id in selected
        if not into_kernel and not from_kernel:
            continue
        if into_kernel and from_kernel:
            direction = "candidate_to_candidate"
        elif into_kernel:
            direction = "host_to_candidate"
        else:
            direction = "candidate_to_host"
        crossings.append(
            {
                "edge_id": edge["id"],
                "source_value_id": edge["value_id"],
                "producer_operation_id": producer_id,
                "consumer_operation_id": consumer_id,
                "consumer_operand_index": edge["operand_index"],
                "consumer_operation": consumer["mlir_operation"],
                "type": edge["type"],
                "shape": edge["shape"],
                "dtype": edge["dtype"],
                "direction": direction,
                "status": "requires_placement_transfer_and_dispatch",
            }
        )
    return {
        "schema": SCHEMA,
        "model_sha256": model_sha256,
        "operations_accounted": graph["n_operations"],
        "candidate_operation_ids": [candidate["operation_id"] for candidate in candidates],
        "functions": functions,
        "candidate_boundary_crossings": crossings,
        "composition_plan": _composition_plan(graph, functions, crossings, candidates),
        "status": "diagnostic_unexecutable",
        "missing_for_execution": [
            "host_operation_lowering",
            "memory_placement_and_typed_transfers",
            "kernel_dispatch_and_pointer_abi",
            "whole_model_numerical_equivalence",
        ],
    }
