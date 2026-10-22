"""Opt-in standard TensorIR row scheduling for bounded CPU operator scratch.

Move private producers under the final consumer's row loop when schedule
legality proves the move. This preserves the operator and reduction order;
it changes production placement and buffer lifetimes, not graph numerics.
"""


def localize_private_producers(mod, tvm):
    import math
    changes, refusals = [], []

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="LocalizeCPUPrivateRows")
    def localize(function, module, context):
        del module, context
        if function.attrs and "gemmini.m" in function.attrs:
            return function
        if not isinstance(function.body, tvm.tir.BlockRealize):
            return function
        allocations = function.body.block.alloc_buffers
        if not allocations:
            return function
        external = {buffer.data for buffer in function.buffer_map.values()}
        blocks = []
        tvm.tir.stmt_functor.post_order_visit(function.body,
            lambda node: blocks.append(node) if isinstance(node, tvm.tir.Block) and node.name_hint != "root" else None)
        finals = [block for block in blocks if any(region.buffer.data in external for region in block.writes)]
        if len(finals) != 1:
            return function
        name = str(function.attrs["global_symbol"]) if function.attrs and "global_symbol" in function.attrs else finals[0].name_hint
        try:
            schedule = tvm.tir.Schedule(function, debug_mask="all")
            consumer = schedule.get_block(finals[0].name_hint)
            loops = schedule.get_loops(consumer)
            if len(loops) < 2:
                return function
            row = loops[-2]
            pending = {block.name_hint for block in blocks if block.writes and
                       all(region.buffer.data not in external for region in block.writes)}
            placed = {finals[0].name_hint}
            moved = []
            while pending:
                progress = False
                for block_name in sorted(pending):
                    block = schedule.get_block(block_name)
                    users = schedule.get_consumers(block)
                    if users and all(schedule.get(user).name_hint in placed for user in users):
                        schedule.compute_at(block, row, preserve_unit_loops=True)
                        pending.remove(block_name)
                        placed.add(block_name)
                        moved.append(block_name)
                        progress = True
                        break
                if not progress:
                    break
            if not moved:
                return function
            # Match the prerequisite order in TVM's ordinary TIR build pipeline.
            # Apply it only to the scheduled CPU function, preserving Gemmini IR.
            compact = tvm.transform.Sequential([
                tvm.tir.transform.LowerInitBlock(),
                tvm.tir.transform.PlanAndUpdateBufferAllocationLocation(),
                tvm.tir.transform.ConvertBlocksToOpaque(),
                tvm.tir.transform.CompactBufferAllocation(),
            ])(schedule.mod)["main"]
            after = []
            tvm.tir.stmt_functor.post_order_visit(compact.body,
                lambda node: after.extend(node.alloc_buffers) if isinstance(node, tvm.tir.Block) else None)
            changes.append({"function": name, "moved_blocks": moved,
                            "original_private_buffer_bytes": sum(math.prod(int(dim) for dim in buffer.shape) *
                                tvm.DataType(buffer.dtype).bits // 8 for buffer in allocations),
                            "compact_private_buffer_bytes": sum(math.prod(int(dim) for dim in buffer.shape) *
                                tvm.DataType(buffer.dtype).bits // 8 for buffer in after)})
            return compact
        except tvm.error.TVMError as error:
            refusals.append({"function": name, "reason": str(error)[-1800:]})
            return function

    return localize(mod), {"changes": changes, "refusals": refusals,
                    "API": "tir.Schedule.compute_at with all debug checks; standard LowerInitBlock/PlanAndUpdateBufferAllocationLocation/ConvertBlocksToOpaque/CompactBufferAllocation order",
                    "scope": "Private producers of single-output CPU functions; Gemmini functions untouched; illegal schedules retain original function",
                    "TVM_source_modified": False, "graph_operator_replacement": False}


def serialize_parallel_loops(mod, tvm):
    """Execute independent CPU loops serially on the declared one-hart guest."""
    count = 0

    @tvm.tir.transform.prim_func_pass(opt_level=0, name="SerializeSingleHartCPULoops")
    def serialize(function, module, context):
        del module, context
        if function.attrs and "gemmini.m" in function.attrs:
            return function

        def rewrite(node):
            nonlocal count
            if isinstance(node, tvm.tir.For) and node.kind == tvm.tir.ForKind.PARALLEL:
                count += 1
                return tvm.tir.For(node.loop_var, node.min, node.extent, tvm.tir.ForKind.SERIAL,
                                   node.body, annotations=node.annotations, span=node.span)
            return None

        return function.with_body(tvm.tir.stmt_functor.ir_transform(function.body, None, rewrite, ["tir.For"]))

    return serialize(mod), {"serialized_loops": count, "API": "prim_func_pass and stmt_functor.ir_transform",
                            "precondition": "One-hart guest, no TVM guest thread-pool runtime; parallel iterations remain ordinary serial loops",
                            "TVM_source_modified": False, "Gemmini_functions_changed": False}
