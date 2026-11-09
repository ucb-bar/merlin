"""Independent replay of bounded convolution windows and emitted tensor regions.

The original typed witness binds external SSA, recurrence order, complete output
coverage and loop-carried destinations. This check establishes no allocation,
backend arithmetic, runtime placement or hardware qualification.
"""

from __future__ import annotations

from merlin.compiler.ir.convolution import (
    BoundedIm2ColArm,
    BoundedIm2ColRewrite,
    BoundedIm2ColSource,
    _bounded_plain,
    _is_copy_body,
    _reassoc,
)


def verify_bounded_match(rewrite, *, expected_source=None, expected_bindings=None):
    """Reconstruct the realized affine windows, complete output coverage and reuse.

    Original pass analyses supply expected_source and expected_bindings to bind
    the witness and external SSA values to their immutable current-source context.
    These source-level tensor destinations establish neither allocation nor timing.
    """
    from xdsl.dialects import arith, builtin, scf, tensor
    from xdsl.dialects.linalg.attrs import IteratorType
    from xdsl.dialects.linalg.ops import GenericOp
    from xdsl.ir.affine import AffineExpr, AffineMap

    from merlin.common import mlir_query as mq

    def require(claim, reason):
        if not claim:
            raise ValueError("bounded im2col realization: " + reason)

    require(
        type(rewrite) is BoundedIm2ColRewrite and type(rewrite.source) is BoundedIm2ColSource,
        "requires the original typed source witness",
    )
    source = rewrite.source
    require(
        expected_bindings is None or (source.input, source.weight, source.initializer) == expected_bindings,
        "original external SSA bindings changed",
    )
    require(
        source.identity == rewrite.source_identity and (expected_source is None or expected_source == source.identity),
        "original source witness changed",
    )
    original = mq.parse(source.source_ir)
    original.verify()
    reference = next(op for op in original.walk() if type(op) is GenericOp)
    ref_fill = reference.outputs[0].owner
    n, oh, ow, channels, kh, kw, sh, sw, dh, dw, f, k, m = source.geometry
    require(
        all(type(v) is int and v > 0 for v in source.geometry) and k == channels * kh * kw and m == n * oh * ow,
        "inconsistent original geometry",
    )
    require(
        str(source.input.type) == source.input_type
        and tuple(reference.results[0].type.get_shape()) == (f, m)
        and tuple(reference.inputs[0].type.get_shape()) == (f, k)
        and tuple(reference.inputs[1].type.get_shape()) == (k, m)
        and source.weight.type == reference.inputs[0].type
        and source.initializer.type == reference.results[0].type,
        "original operand types changed",
    )
    elem = reference.inputs[1].type.element_type
    require(type(elem) is builtin.IntegerType and elem.width.data in (8, 16, 32), "unknown packing element format")
    width_bytes = elem.width.data // 8
    input_shape = tuple(source.input.type.get_shape())
    require(
        len(input_shape) == 4
        and input_shape[:2] == (n, channels)
        and (oh - 1) * sh + (kh - 1) * dh < input_shape[2]
        and (ow - 1) * sw + (kw - 1) * dw < input_shape[3]
        and source.gather_type == str(builtin.TensorType(elem, (channels, kh, kw, n, oh, ow))),
        "original window exceeds its typed input or changes element order",
    )
    require(type(rewrite.width) is int and 1 <= rewrite.width <= ow, "invalid selected panel width")
    full, tail = divmod(ow, rewrite.width)
    expected_arms = [(rewrite.width, 0, full)] + ([(tail, full * rewrite.width, 1)] if tail else [])
    require(len(rewrite.arms) == len(expected_arms), "missing or duplicated main/tail output domain")
    initial, top_operations, arm_records = source.initializer, [], []
    for arm, (width, first, per_row) in zip(rewrite.arms, expected_arms, strict=True):
        require(
            type(arm) is BoundedIm2ColArm and (arm.width, arm.first_ow, arm.panels_per_row) == (width, first, per_row),
            "changed main/tail coverage",
        )
        loop, packing = arm.loop, arm.packing_empty
        require(
            type(loop) is scf.ForOp
            and _bounded_plain(loop)
            and type(packing) is tensor.EmptyOp
            and _bounded_plain(packing)
            and not packing.operands
            and loop.parent_block() is packing.parent_block(),
            "panel destination is not outside its serial loop",
        )
        packing_type = builtin.TensorType(elem, (channels, kh, kw, 1, 1, width))
        require(
            len(loop.results) == 2
            and packing.results[0].type == packing_type
            and tuple(loop.iter_args) == (initial, packing.results[0]),
            "panel must be an explicit loop-carried destination",
        )
        bounds = tuple(v.owner for v in (loop.lb, loop.ub, loop.step))
        for op, value in zip(bounds, (0, n * oh * per_row, 1), strict=True):
            require(
                type(op) is arith.ConstantOp
                and _bounded_plain(op, ("value",))
                and isinstance(op.value, builtin.IntegerAttr)
                and op.value.value.data == value
                and type(op.results[0].type) is builtin.IndexType,
                "incorrect serial panel bounds",
            )
        top_operations.extend((packing, *bounds, loop))
        block = loop.body.block
        require(len(loop.body.blocks) == 1 and len(block.args) == 3, "unexpected loop ownership boundary")
        iv, carried, destination = block.args
        require(destination.type == packing_type and carried.type == initial.type, "changed loop-carried formats")
        matrix = arm.contraction.inputs[1].owner
        zero = arm.output_fill.inputs[0].owner
        terminator = block.last_op
        suffix = (
            arm.patch,
            arm.gather,
            matrix,
            arm.output_empty,
            zero,
            arm.output_fill,
            arm.contraction,
            arm.insert,
            terminator,
        )
        body = tuple(block.ops)
        require(body[-len(suffix) :] == suffix, "unexpected operations in panel body")
        prefix = body[: -len(suffix)]
        require(len(prefix) <= 32, "unbounded panel index work")
        visited = set()

        def expression(value):
            if value is iv:
                return ("iv",)
            op = value.owner
            require(
                op in prefix
                and op.parent_block() is block
                and not op.attributes
                and not op.regions
                and not op.successors
                and len(op.results) == 1
                and type(value.type) is builtin.IndexType,
                "unknown panel index expression",
            )
            visited.add(op)
            if type(op) is arith.ConstantOp:
                require(
                    set(op.properties) == {"value"} and isinstance(op.value, builtin.IntegerAttr),
                    "unknown index constant",
                )
                return op.value.value.data
            names = {arith.AddiOp: "add", arith.MuliOp: "mul", arith.DivUIOp: "div", arith.RemUIOp: "rem"}
            require(
                type(op) in names
                and len(op.operands) == 2
                and set(op.properties) <= {"overflowFlags"}
                and all(str(v) == "#arith.overflow<none>" for v in op.properties.values()),
                "unknown index arithmetic",
            )
            return (names[type(op)], *(expression(v) for v in op.operands))

        row, panel = ("div", ("iv",), per_row), ("rem", ("iv",), per_row)
        column = ("add", ("mul", panel, width), first)
        expected_offsets = (("div", row, oh), 0, ("mul", ("rem", row, oh), sh), ("mul", column, sw))

        def slice_check(op, offsets, sizes, source_value):
            require(
                type(op) in (tensor.ExtractSliceOp, tensor.InsertSliceOp)
                and not op.attributes
                and set(op.properties) == {"static_offsets", "static_sizes", "static_strides", "operandSegmentSizes"}
                and not op.regions
                and not op.successors,
                "unknown slice effects",
            )
            actual = iter(op.offsets)
            observed = tuple(
                v if v != builtin.DYNAMIC_INDEX else expression(next(actual)) for v in op.static_offsets.get_values()
            )
            require(
                observed == offsets
                and tuple(op.static_sizes.get_values()) == sizes
                and tuple(op.static_strides.get_values()) == (1,) * len(sizes)
                and op.operands[0] is source_value,
                "incorrect source window or output position",
            )
            require(next(actual, None) is None and not op.sizes and not op.strides, "unexpected dynamic slice extent")

        patch_shape = (1, channels, (kh - 1) * dh + 1, (width - 1) * sw + (kw - 1) * dw + 1)
        slice_check(arm.patch, expected_offsets, patch_shape, source.input)
        require(arm.patch.results[0].type == builtin.TensorType(elem, patch_shape), "changed patch shape")
        gather = arm.gather
        d = AffineExpr.dimension
        maps = (AffineMap(6, 0, (d(3), d(0), d(1) * dh, d(5) * sw + d(2) * dw)), AffineMap.identity(6))
        require(
            type(gather) is GenericOp
            and _bounded_plain(gather, ("indexing_maps", "iterator_types", "operandSegmentSizes"))
            and tuple(gather.inputs) == (arm.patch.results[0],)
            and tuple(gather.outputs) == (destination,)
            and gather.results[0].type == packing_type
            and tuple(a.data for a in gather.indexing_maps) == maps
            and tuple(t.data for t in gather.get_iterator_types()) == (IteratorType.PARALLEL,) * 6
            and _is_copy_body(gather)
            and _bounded_plain(gather.body.block.last_op),
            "gather changed copied values, coordinates or reused destination",
        )
        require(
            type(matrix) is tensor.CollapseShapeOp
            and _bounded_plain(matrix, ("reassociation",))
            and tuple(matrix.operands) == (gather.results[0],)
            and matrix.results[0].type == builtin.TensorType(elem, (k, width))
            and matrix.reassociation == _reassoc([[0, 1, 2], [3, 4, 5]]),
            "changed complete K order or matrix view",
        )
        output_type = builtin.TensorType(reference.results[0].type.element_type, (f, width))
        require(
            type(arm.output_empty) is tensor.EmptyOp
            and _bounded_plain(arm.output_empty)
            and not arm.output_empty.operands
            and arm.output_empty.results[0].type == output_type
            and type(zero) is arith.ConstantOp
            and zero.attributes == ref_fill.inputs[0].owner.attributes
            and zero.properties == ref_fill.inputs[0].owner.properties
            and zero.results[0].type == ref_fill.inputs[0].type
            and _bounded_plain(arm.output_fill, ("operandSegmentSizes",))
            and tuple(arm.output_fill.inputs) == (zero.results[0],)
            and tuple(arm.output_fill.outputs) == (arm.output_empty.results[0],)
            and arm.output_fill.body.is_structurally_equivalent(ref_fill.body),
            "changed fresh zero initialization",
        )
        inner = arm.contraction
        require(
            type(inner) is GenericOp
            and inner.attributes == reference.attributes
            and inner.properties == reference.properties
            and tuple(inner.inputs) == (source.weight, matrix.results[0])
            and tuple(inner.outputs) == (arm.output_fill.results[0],)
            and inner.results[0].type == output_type
            and inner.body.is_structurally_equivalent(reference.body),
            "changed scalar recurrence or reduction order",
        )
        slice_check(arm.insert, (0, ("add", ("mul", row, ow), column)), (f, width), inner.results[0])
        require(
            arm.insert.dest is carried
            and type(terminator) is scf.YieldOp
            and _bounded_plain(terminator)
            and tuple(terminator.operands) == (arm.insert.results[0], gather.results[0])
            and not tuple(loop.results[1].uses)
            and tuple(use.operation for use in packing.results[0].uses) == (loop,)
            and tuple(use.operation for use in destination.uses) == (gather,)
            and {use.operation for use in gather.results[0].uses} == {matrix, terminator}
            and visited == set(prefix),
            "panel ownership or completed result escaped",
        )
        initial = loop.results[0]
        arm_records.append(
            dict(
                width=width,
                first_ow=first,
                panels_per_row=per_row,
                iterations=n * oh * per_row,
                packing_shape=list(packing_type.get_shape()),
                packing_bytes=k * width * width_bytes,
                output_tile_bytes=f * width * reference.results[0].type.element_type.width.data // 8,
            )
        )
    require(
        tuple(top_operations) == rewrite.operations and rewrite.result is initial, "changed actual emitted membership"
    )
    cursor = top_operations[0]
    for op in top_operations:
        require(cursor is op, "unreported effect between panel operations")
        cursor = cursor.next_op
    return dict(
        schema="merlin.bounded_im2col.v1",
        source_identity=source.identity,
        original_packing_bytes=k * m * width_bytes,
        resident_packing_bytes=sum(a["packing_bytes"] for a in arm_records),
        gathered_elements=k * m,
        requested_gather_read_bytes=k * m * width_bytes,
        requested_gather_write_bytes=k * m * width_bytes,
        output_shape=[f, m],
        panel_invocations=sum(a["iterations"] for a in arm_records),
        arms=arm_records,
        scope="checked source windows, K order, coverage and loop-carried destinations; "
        "actual allocation unestablished",
    )
