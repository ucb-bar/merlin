"""Retain integer storage normalization before every width-changing boundary.

An extension is not a homomorphism between two modular rings. Compute its
operand in the original ring, then execute the source extension on the exact
recombined storage word. This also preserves unsigned interpretation of i8.
"""


def preserve_integer_width_boundaries(module, *, mathematical_sources=()):
    from xdsl.dialects.builtin import TensorType, UnitAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, BlockArgument
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import generic, identity, point
    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_pointwise_primitives(module, integer_boundaries=True)
    arithmetic_producer = any(
        operation.name == "linalg.generic"
        and "merlin.exact_host_arithmetic" not in operation.attributes
        and any(child.name in {"arith.addi", "arith.subi", "arith.muli"} for child in operation.body.block.ops)
        and all(
            child.name in {"arith.constant", "arith.addi", "arith.subi", "arith.muli", "linalg.yield"}
            for child in operation.body.block.ops
        )
        for operation in module.walk()
    )
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        if "merlin.exact_host_arithmetic" in op.attributes:
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name not in {"arith.extsi", "arith.extui", "arith.trunci"}:
            continue
        cast = body[0]
        if cast.operands[0] not in op.body.block.args[: len(op.inputs)] or list(body[1].operands) != list(cast.results):
            continue
        source_index = list(op.body.block.args).index(cast.operands[0])
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        source = op.inputs[source_index]
        if source in mathematical_sources and mathematical_sources[source] is not None:
            source_bits = source.type.element_type.width.data
            target_bits = op.results[0].type.element_type.width.data
            mathematical_sources[op.results[0]] = (
                (1 << source_bits) - 1
                if cast.name == "arith.extui"
                else min(mathematical_sources[source], 1 << (target_bits - 1))
                if cast.name == "arith.trunci"
                else mathematical_sources[source]
            )
        external = isinstance(source, BlockArgument) and source.owner.parent_op().name == "func.func"
        if arithmetic_producer or (not external and source not in mathematical_sources):
            # Intermediate source-width producers are lowered independently.
            # Their already recombined storage requires only the original host
            # cast, never another device identity producer for each bit cast.
            op.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
            count += 1
            continue
        source_type = op.inputs[source_index].type.element_type
        typ = TensorType(source_type, op.results[0].type.get_shape())
        maps, operations = op.get_indexing_maps().data, []
        carrier = point(
            [op.inputs[source_index]],
            typ,
            lambda args, emit: args[0],
            operations,
            maps=[maps[source_index], maps[-1]],
            iterators=op.get_iterator_types(),
        )
        # The component carrier returns source-width storage bits. The original
        # cast consumes those bits after full-i32 limb recombination.
        host_block = Block(arg_types=[source_type, op.results[0].type.element_type])
        host_cast = cast.clone(value_mapper={cast.operands[0]: host_block.args[0]})
        host_block.add_ops([host_cast, L.YieldOp(host_cast.results[0])])
        rank = len(typ.get_shape())
        host = generic([carrier], op.outputs[0], host_block, [identity(rank)] * 2, op.get_iterator_types(), host=True)
        Rewriter.replace_op(op, [*operations, host], list(host.results))
        count += 1
    return count
