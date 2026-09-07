"""Count dynamic host IR work from proven constant-trip LLVM control flow.

This is an emitted-program accounting instrument, not a CPU cycle predictor.
Load/store bytes are scalar instruction payload, not DRAM traffic: caches,
vectorization, register allocation and later LLVM optimization can change costs.
Unsupported control flow makes dynamic totals UNKNOWN instead of counting a
rolled loop body once. Static allocation payload is derived from alloca types
and sizes; annotations are cross-checks only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from math import prod
from typing import Any

from xdsl.dialects.builtin import IntegerType, VectorType
from xdsl.dialects.llvm import ICmpPredicateFlag
from xdsl.ir import Block, BlockArgument, Operation

from merlin.xdsl_dialects.lowering.integer_constant_eval import constant_integer
from .host_cfg_index import PreparedHostCFG, prepare_host_cfg, require_prepared_host_cfg


_INTEGER = frozenset({"llvm.add", "llvm.sub", "llvm.mul", "llvm.sdiv", "llvm.udiv",
                      "llvm.srem", "llvm.urem", "llvm.and", "llvm.or", "llvm.xor",
                      "llvm.shl", "llvm.lshr", "llvm.ashr"})
_FLOAT = frozenset({"llvm.fadd", "llvm.fsub", "llvm.fmul", "llvm.fdiv", "llvm.frem", "llvm.fneg"})
_CAST = frozenset({"llvm.trunc", "llvm.sext", "llvm.zext", "llvm.bitcast", "llvm.fpext",
                  "llvm.fptrunc", "llvm.sitofp", "llvm.uitofp", "llvm.fptosi", "llvm.fptoui"})


def _integer_attr(attr: Any) -> int | None:
    value = getattr(getattr(attr, "value", None), "data", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _payload_bytes(ty: Any) -> int | None:
    """Instruction memory payload; does not infer aggregate layout padding."""
    if isinstance(ty, IntegerType):
        return (int(ty.width.data) + 7) // 8
    bits = getattr(ty, "bitwidth", None)
    if isinstance(bits, int):
        return (bits + 7) // 8
    if isinstance(ty, VectorType):
        if any(getattr(dim, "data", dim) for dim in ty.scalable_dims):
            return None
        element = _payload_bytes(ty.get_element_type())
        return None if element is None else prod(ty.get_shape()) * element
    return None


def _category(name: str) -> str:
    if name in _INTEGER:
        return "integer_arithmetic"
    if name in _FLOAT:
        return "floating_arithmetic"
    if name in _CAST:
        return "conversion"
    return {"llvm.load": "load", "llvm.store": "store", "llvm.icmp": "comparison",
            "llvm.fcmp": "comparison", "llvm.getelementptr": "address",
            "llvm.ptrtoint": "address", "llvm.inttoptr": "address",
            "llvm.br": "branch", "llvm.cond_br": "branch", "llvm.return": "return",
            "llvm.mlir.constant": "constant", "llvm.alloca": "allocation",
            "llvm.inline_asm": "opaque_inline_asm"}.get(name, "other")


def _constant_loop(header, latch, nodes, predecessors, positions):
    """Recognize constant induction independently of other carried SSA values.

    Accumulators need not be constant: only the tested induction and its incoming
    edges determine execution counts. Other conditional exits remain rejected by
    the enclosing CFG analysis; this does not model accumulator values.
    """
    terminal = header.last_op
    if (terminal.name != "llvm.cond_br" or not header.args
            or len(terminal.successors) != 2):
        raise ValueError("loop header is not an induction conditional branch")
    if terminal.successors[0] not in nodes or terminal.successors[1] in nodes:
        raise ValueError("loop condition does not take true into the body and false to exit")
    condition = terminal.operands[0].owner
    if (not isinstance(condition, Operation) or condition.name != "llvm.icmp"
            or len(condition.operands) != 2):
        raise ValueError("loop condition is not an integer comparison")
    try:
        predicate = ICmpPredicateFlag.from_int(_integer_attr(condition.properties.get("predicate"))).value
    except (ValueError, TypeError, AttributeError):
        raise ValueError("loop predicate is unknown") from None
    induction = condition.operands[0]
    if (predicate not in {"slt", "ult"} or induction not in header.args
            or not isinstance(induction.type, IntegerType)):
        raise ValueError("loop induction does not use supported increasing less-than comparison")
    induction_index = list(header.args).index(induction)
    bound = constant_integer(condition.operands[1])
    outside = predecessors[header] - nodes
    if len(outside) != 1 or len(predecessors[header] & nodes) != 1:
        raise ValueError("loop needs exactly one preheader and one latch")
    preheader = next(iter(outside))
    for block in (preheader, latch):
        if (block.last_op.name != "llvm.br"
                or len(block.last_op.operands) != len(header.args)
                or tuple(v.type for v in block.last_op.operands) != tuple(v.type for v in header.args)):
            raise ValueError("induction incoming edge does not match header arguments")
    initial = constant_integer(preheader.last_op.operands[induction_index])
    update = latch.last_op.operands[induction_index].owner
    if not isinstance(update, Operation) or update.name != "llvm.add" or len(update.operands) != 2:
        raise ValueError("induction update is not an addition")
    values = list(update.operands)
    def forwards_induction(value, seen=frozenset()):
        """Prove identity through all incoming edges, never through arithmetic.

        A loop body may receive its IV as a block argument instead of referring
        directly to the header value. A conflicting incoming value or an SSA
        cycle is not a proof of identity and must remain unsupported.
        """
        if value is induction:
            return True
        if (value in seen or not isinstance(value, BlockArgument)
                or value.owner not in nodes or value.owner is header):
            return False
        incoming = []
        for predecessor in predecessors[value.owner]:
            branch = predecessor.last_op
            if branch.name == "llvm.br":
                edges = [(branch.successors[0], branch.operands)]
            elif branch.name == "llvm.cond_br":
                edges = list(zip(branch.successors, (branch.then_arguments, branch.else_arguments)))
            else:
                return False
            for destination, arguments in edges:
                if destination is value.owner:
                    if len(arguments) != len(destination.args):
                        return False
                    incoming.append(arguments[value.index])
        return bool(incoming) and all(forwards_induction(arg, seen | {value}) for arg in incoming)

    induction_operands = [i for i, value in enumerate(values) if forwards_induction(value)]
    if len(induction_operands) != 1:
        raise ValueError("induction update does not consume the header argument")
    step = constant_integer(values[1 - induction_operands[0]])
    if initial is None or bound is None or step is None or step <= 0:
        raise ValueError("loop initial value, limit or positive step is not constant")
    width = int(induction.type.width.data)
    if predicate == "ult":
        initial &= (1 << width) - 1
        bound &= (1 << width) - 1
    trips = max(0, (bound - initial + step - 1) // step)
    final = initial + trips * step
    limit = (1 << width) - 1 if predicate == "ult" else (1 << (width - 1)) - 1
    if final > limit:
        raise ValueError("loop final induction update could overflow")
    for block in nodes:
        for destination in block.last_op.successors:
            if destination not in nodes and not (block is header and destination is terminal.successors[1]):
                raise ValueError("loop has an early exit")
    return {"header_block": positions[header], "latch_block": positions[latch],
            "body_blocks": sorted(positions[b] for b in nodes if b is not header),
            "initial": initial, "bound": bound, "step": step, "trip_count": trips,
            "induction_argument_index": induction_index,
            "other_carried_values": len(header.args) - 1,
            "predicate": predicate}, nodes


def analyze_host_cfg_activity(function: Any, *,
                              prepared_cfg: PreparedHostCFG | None = None) -> dict[str, Any]:
    """Account one submitted LLVM function at the emitted IR level."""
    cfg = (require_prepared_host_cfg(function, prepared_cfg)
           if prepared_cfg is not None else prepare_host_cfg(function))
    blocks = cfg.blocks
    if not blocks:
        return {"schema": "host_cfg_activity_v1", "status": "UNKNOWN", "problems": ["empty function"]}
    index = {block: i for i, block in enumerate(blocks)}
    predecessors = cfg.predecessors
    problems: list[str] = []
    for block in blocks:
        terminator = block.last_op
        if terminator is None or terminator.name not in {"llvm.br", "llvm.cond_br", "llvm.return"}:
            problems.append(f"block {index[block]} has unsupported terminator")
            continue
        for target in terminator.successors:
            if target not in cfg.block_set:
                problems.append("branch leaves analyzed function")
    reachable = set()
    pending = [blocks[0]]
    while pending:
        block = pending.pop()
        if block in reachable:
            continue
        reachable.add(block)
        if block.last_op is not None:
            pending.extend(b for b in block.last_op.successors if b in index)
    if len(reachable) != len(blocks):
        problems.append("function contains unreachable blocks")
    loops = []
    if not problems:
        dominance = cfg.dominance
        backedges = [(header, latch) for header in blocks for latch in predecessors[header]
                     if dominance.dominates(header, latch)]
        for header, latch in backedges:
            nodes = {header, latch}
            pending = [latch] if latch is not header else []
            while pending:
                block = pending.pop()
                for pred in predecessors[block]:
                    if pred not in nodes:
                        nodes.add(pred)
                        pending.append(pred)
            try:
                loops.append(_constant_loop(header, latch, nodes, predecessors, index))
            except ValueError as exc:
                problems.append(f"loop at block {index[header]}: {exc}")
        headers = {row["header_block"] for row, _ in loops}
        for block in blocks:
            if block.last_op.name == "llvm.cond_br" and index[block] not in headers:
                problems.append(f"block {index[block]} conditional execution is not proven")
        for i, (_, left) in enumerate(loops):
            if any(left & right and not (left <= right or right <= left) for _, right in loops[i+1:]):
                problems.append("natural loops overlap without nesting")
        if not any(block.last_op.name == "llvm.return" for block in blocks):
            problems.append("function has no return")
    multiplicity = {block: 1 for block in blocks}
    if not problems:
        for row, nodes in loops:
            for block in nodes:
                multiplicity[block] *= row["trip_count"] + (index[block] == row["header_block"])
    else:
        multiplicity = {block: None for block in blocks}

    static = Counter()
    dynamic = Counter()
    tasks = defaultdict(lambda: {"static": Counter(), "dynamic": Counter(),
                                 "load_payload_bytes": 0, "store_payload_bytes": 0})
    allocations = []
    buffers = defaultdict(lambda: {"load_payload_bytes": 0, "store_payload_bytes": 0,
                                   "load_operations": 0, "store_operations": 0})
    buffer_tasks = defaultdict(set)
    op_index = {op: i for i, op in enumerate(function.walk())}
    unknown_payload = []

    def pointer_root(value):
        seen = set()
        while value not in seen:
            seen.add(value)
            owner = value.owner
            if isinstance(owner, Block):
                return f"arg:{value.index}" if owner is blocks[0] else "UNKNOWN"
            if owner.name == "llvm.alloca":
                return f"alloca:{op_index[owner]}"
            if owner.name in {"llvm.getelementptr", "llvm.bitcast"}:
                value = owner.operands[0]
            else:
                return "UNKNOWN"
        return "UNKNOWN"

    for block in blocks:
        count = multiplicity[block]
        for operation in block.ops:
            category = _category(operation.name)
            owner = _integer_attr(operation.attributes.get("merlin.global_task"))
            task = tasks[str(owner) if owner is not None else "unowned"]
            static[category] += 1
            task["static"][category] += 1
            if count is not None:
                dynamic[category] += count
                task["dynamic"][category] += count
            if category in {"load", "store"}:
                ty = operation.results[0].type if category == "load" else operation.operands[0].type
                payload = _payload_bytes(ty)
                pointer = operation.operands[0] if category == "load" else operation.operands[1]
                root = pointer_root(pointer)
                buffer = buffers[root]
                buffer_tasks[root].add(str(owner) if owner is not None else "unowned")
                if payload is None:
                    unknown_payload.append(op_index[operation])
                elif count is not None:
                    task[category + "_payload_bytes"] += payload * count
                    buffer[category + "_payload_bytes"] += payload * count
                    buffer[category + "_operations"] += count
            if category == "allocation":
                elements = constant_integer(operation.operands[0])
                element_bytes = _payload_bytes(operation.properties.get("elem_type"))
                size = elements * element_bytes if elements is not None and elements >= 0 and element_bytes is not None else None
                declared = _integer_attr(operation.attributes.get("merlin.host_storage_bytes"))
                allocations.append({"operation_index": op_index[operation], "block": index[block],
                    "buffer_root": f"alloca:{op_index[operation]}",
                    "task": str(owner) if owner is not None else "unowned",
                    "elements": elements, "element_type": str(operation.properties.get("elem_type")),
                    "payload_bytes": size, "execution_count": count,
                    "declared_payload_bytes": declared,
                    "annotation_status": "absent" if declared is None else (
                        "UNKNOWN" if size is None else "matches" if declared == size else "mismatch")})
    if any(row["annotation_status"] == "mismatch" for row in allocations):
        problems.append("allocation storage-byte annotation disagrees with derived payload")
    payload_known = not problems and not unknown_payload
    task_rows = []
    for owner, row in tasks.items():
        task_rows.append({"task": owner, "static_operations": dict(row["static"]),
            "dynamic_operations": dict(row["dynamic"]) if not problems else None,
            "load_payload_bytes": row["load_payload_bytes"] if payload_known else None,
            "store_payload_bytes": row["store_payload_bytes"] if payload_known else None})
    allocation_by_root = {row["buffer_root"]: row for row in allocations}
    # Join exact SSA pointer roots to allocation operation identity, never names or size guesses.
    # Keep static allocation facts visible even when dynamic loop counts are unknown.
    hotspots = []
    for root in sorted(set(buffers) | set(allocation_by_root)):
        allocation = allocation_by_root.get(root)
        payload = buffers.get(root, {})
        hotspots.append({"buffer_root": root,
            "allocation_operation_index": allocation["operation_index"] if allocation else None,
            "allocation_task": allocation["task"] if allocation else None,
            "access_tasks": sorted(buffer_tasks[root]),
            "static_allocation_payload_bytes": allocation["payload_bytes"] if allocation else None,
            "allocation_element_type": allocation["element_type"] if allocation else None,
            "allocation_execution_count": allocation["execution_count"] if allocation else None,
            "load_payload_bytes": payload.get("load_payload_bytes", 0) if payload_known else None,
            "store_payload_bytes": payload.get("store_payload_bytes", 0) if payload_known else None,
            "pointer_identity_status": "UNKNOWN" if root == "UNKNOWN" else "derived_ssa_root",
            "source_operation_attribution": "UNKNOWN"})
    top_allocations = sorted((row for row in hotspots if row["allocation_operation_index"] is not None),
        key=lambda row: (row["static_allocation_payload_bytes"] is not None,
                         row["static_allocation_payload_bytes"] or 0), reverse=True)[:3]
    top_buffers = sorted((row for row in hotspots if row["buffer_root"] in buffers),
        key=lambda row: (row["load_payload_bytes"] is not None,
                         (row["load_payload_bytes"] or 0) + (row["store_payload_bytes"] or 0)), reverse=True)[:3]
    return {
        "schema": "host_cfg_activity_v1", "status": "derived" if not problems and not unknown_payload else "UNKNOWN",
        "problems": sorted(set(problems)), "loop_count": len(loops),
        "loops": [row for row, _ in loops],
        "blocks": [{"block": index[b], "execution_count": multiplicity[b]} for b in blocks],
        "static_operations": dict(static), "dynamic_operations": dict(dynamic) if not problems else None,
        "load_payload_bytes": sum(row["load_payload_bytes"] for row in tasks.values()) if payload_known else None,
        "store_payload_bytes": sum(row["store_payload_bytes"] for row in tasks.values()) if payload_known else None,
        "unknown_memory_operation_indices": unknown_payload,
        "tasks": task_rows, "buffer_payload": dict(buffers) if payload_known else None,
        "allocations": allocations,
        "top_allocations_by_static_payload": top_allocations,
        "top_buffers_by_scalar_memory_payload": top_buffers,
        "memory_hotspot_identity_scope": "buffer_root alloca index is function.walk operation index; exact SSA roots only",
        "static_allocation_payload_bytes": sum(row["payload_bytes"] for row in allocations)
            if all(row["payload_bytes"] is not None for row in allocations) else None,
        "cpu_cycles": None, "dram_bytes": None, "cache_traffic_bytes": None,
        "scope": "one invocation, emitted LLVM IR execution counts and scalar memory payload",
        "limitations": ["not post-optimization machine instruction counts", "not DRAM/cache traffic",
                        "integer arithmetic includes index and loop-maintenance operations",
                        "allocation payload excludes stack alignment, reuse and frame overhead",
                        "distinct SSA pointer roots do not prove physical no-alias or separate DRAM traffic",
                        "opaque inline assembly effects are accounted separately by target decoder"],
    }
