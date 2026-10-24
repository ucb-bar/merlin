"""Frontend result storage contracts, lowered to real borrowed/owned memrefs."""

from __future__ import annotations

import inspect
import json

from .input_layout import refuse_overlapping_input_aliases, static_storage_span

ATTRIBUTE = "m2m.result_layout"


def read_layout(source: str):
    from merlin.frontends.linalg_mlir import parse_mlir_text

    if ATTRIBUTE not in source:
        return None
    from xdsl.dialects.ml_program import MLProgram

    from merlin.frontends.linalg_mlir import make_context

    ctx = make_context()
    ctx.load_dialect(MLProgram)
    module = parse_mlir_text(source, ctx)
    attr = module.attributes.get(ATTRIBUTE)
    return json.loads(attr.data) if attr is not None else None


def module_attribute(layout: dict) -> str:
    from xdsl.dialects.builtin import StringAttr

    return f" attributes {{{ATTRIBUTE} = {StringAttr(json.dumps(layout, sort_keys=True))}}}"


def _apply_result_layout(module, contract):
    from torch_mlir import ir
    from torch_mlir.dialects import arith, memref

    refuse_overlapping_input_aliases(contract)
    functions = [
        op
        for op in module.body.operations
        if op.operation.name == "func.func" and ir.StringAttr(op.attributes["sym_name"]).value == "forward"
    ]
    if len(functions) != 1:
        raise ValueError("result layout requires one forward definition")
    function = functions[0]
    block = function.regions[0].blocks[0]
    returns = [op for op in block.operations if op.operation.name == "func.return"]
    if len(returns) != 1:
        raise ValueError("result layout requires a single return")
    returned = returns[0]
    old = list(returned.operands)
    records = contract["results"]
    if len(old) != len(records) or len(block.arguments) != len(contract["inputs"]):
        raise ValueError("result layout ABI count mismatch")
    dynamic = ir.ShapedType.get_dynamic_size()
    index = ir.IndexType.get()

    def constant(value):
        return arith.ConstantOp(index, ir.IntegerAttr.get(index, value)).result

    def span(record):
        return static_storage_span(record)

    def owner_operation(value):
        owner = value.owner
        return owner.operation if isinstance(owner, ir.OpView) else owner

    def root(value):
        while isinstance(owner_operation(value), ir.Operation) and owner_operation(value).name in (
            "memref.cast",
            "memref.reinterpret_cast",
            "memref.subview",
            "memref.expand_shape",
            "memref.collapse_shape",
            "memref.transpose",
        ):
            value = owner_operation(value).operands[0]
        return value

    rewritten = []
    with ir.InsertionPoint(returned), ir.Location.unknown():
        for ordinal, (value, record) in enumerate(zip(old, records)):
            if record is None:
                rewritten.append(value)
                continue
            typ = ir.MemRefType(value.type)
            sizes = [
                memref.DimOp(value, constant(i)).result if n == dynamic else constant(n)
                for i, n in enumerate(typ.shape)
            ]

            def expression(expr, extents=None):
                extents = sizes if extents is None else extents
                if type(expr) is int and expr >= 0:
                    return constant(expr)
                if isinstance(expr, dict) and set(expr) in ({"shape_axes"}, {"shape_axes", "scale"}):
                    scale = expr.get("scale", 1)
                    if type(scale) is not int or scale < 0:
                        raise ValueError("invalid result layout scale")
                    result = constant(scale)
                    for axis in expr["shape_axes"]:
                        if type(axis) is not int or not 0 <= axis < len(extents):
                            raise ValueError("invalid result layout dimension")
                        result = arith.MulIOp(result, extents[axis]).result
                    return result
                raise ValueError("unsupported result layout expression")

            strides = record["stride"]
            offset = record["storage_offset"]
            if len(strides) != typ.rank or any(s == -1 for s in strides) or offset == -1:
                raise ValueError("unrepresentable result layout")
            aliases = record.get("alias_inputs", [])
            mutations = [
                r for r in records if r and r.get("input_index") in aliases and r.get("role") == "user_input_mutation"
            ]
            source = None
            copy = False
            if aliases:
                if record.get("role") != "user_input_mutation" and (
                    span(record) is None or list(typ.shape) != record["shape"] or dynamic in typ.shape
                ):
                    raise ValueError("cannot prove returned view fits its input storage envelope")
                base = aliases[0]
                if not 0 <= base < len(block.arguments):
                    raise ValueError("invalid result base argument")
                input_record = contract["inputs"][base]
                base_type = ir.MemRefType(block.arguments[base].type)
                base_strides, base_offset = base_type.get_strides_and_offset()
                if list(base_strides) != input_record["stride"] or base_offset != input_record["storage_offset"]:
                    raise ValueError("captured input layout differs from the compiled input ABI")
                if not mutations:
                    need, capacity = span(record), span(input_record)
                    if (
                        need is None
                        or capacity is None
                        or list(typ.shape) != record["shape"]
                        or list(base_type.shape) != input_record["shape"]
                        or dynamic in typ.shape
                        or dynamic in base_type.shape
                    ):
                        raise ValueError("cannot prove returned view fits its input storage envelope")
                    if need > capacity:
                        raise ValueError("returned view exceeds its input storage envelope")
                    source = block.arguments[base]
                    # A qualified byte-selector result must be consumed before
                    # its private allocation is released. Preserve the borrowed
                    # descriptor and write the materialized bytes into its view.
                    # Repeated destination addresses carry the same source byte
                    # under the read-only frontend alias contract.
                    copy = record.get("materialized_byte_selection") is True
                elif record.get("role") == "user_input_mutation":
                    # Reuse caller storage only with a static capacity proof.
                    need, capacity = span(record), span(contract["inputs"][base])
                    if (
                        need is not None
                        and capacity is not None
                        and need <= capacity
                        and list(typ.shape) == record["shape"]
                        and list(base_type.shape) == input_record["shape"]
                        and dynamic not in typ.shape
                        and dynamic not in base_type.shape
                    ):
                        source, copy = block.arguments[base], True
                else:
                    mutation_index = records.index(mutations[0])
                    if mutation_index >= ordinal:
                        raise ValueError("mutation storage must precede its user views")
                    source = rewritten[mutation_index]
                    # A view of a mutation result needs its own envelope proof too.
                    backing = input_record if root(source) == block.arguments[base] else mutations[0]
                    need, capacity = span(record), span(backing)
                    backing_value = root(source) if root(source) == block.arguments[base] else source
                    backing_type = ir.MemRefType(backing_value.type)
                    if (
                        need is None
                        or capacity is None
                        or list(backing_type.shape) != backing["shape"]
                        or dynamic in backing_type.shape
                    ):
                        raise ValueError("cannot prove returned view fits its input storage envelope")
                    if need > capacity:
                        raise ValueError("returned view exceeds its input storage envelope")
            elif record.get("alias_results"):
                leader = record["alias_results"][0]
                if type(leader) is not int or not 0 <= leader < ordinal:
                    raise ValueError("invalid result storage predecessor")
                source, copy = rewritten[leader], True
                allocation = root(source)
                if allocation in block.arguments:
                    base = list(block.arguments).index(allocation)
                    backing = contract["inputs"][base]
                    backing_type = ir.MemRefType(allocation.type)
                    need, capacity = span(record), span(backing)
                    if (
                        need is None
                        or capacity is None
                        or list(typ.shape) != record["shape"]
                        or dynamic in typ.shape
                        or list(backing_type.shape) != backing["shape"]
                        or dynamic in backing_type.shape
                    ):
                        raise ValueError("cannot prove returned view fits its input storage envelope")
                    if need > capacity:
                        raise ValueError("returned view exceeds its input storage envelope")
            if source is None:
                actual_strides, actual_offset = typ.get_strides_and_offset()
                peers = [
                    i
                    for i, r in enumerate(records)
                    if r and not r.get("alias_inputs") and r.get("alias_results") and r["alias_results"][0] == ordinal
                ]
                allocation = root(value)
                # Bufferization may forward a value-semantic identity directly
                # from an argument or coalesce equal results. A frontend result
                # with no alias relationship still needs its own storage class.
                prior_roots = [root(v) for v in rewritten]
                known_roots = all(
                    isinstance(v.owner, ir.Block)
                    or (isinstance(owner_operation(v), ir.Operation) and owner_operation(v).name == "memref.alloc")
                    for v in prior_roots
                )
                distinct = (
                    isinstance(owner_operation(allocation), ir.Operation)
                    and owner_operation(allocation).name == "memref.alloc"
                    and known_roots
                    and allocation not in prior_roots
                )
                if list(actual_strides) == strides and actual_offset == offset and not peers and distinct:
                    rewritten.append(value)
                    continue
                # A fresh alias class needs the union of all returned address
                # envelopes, even when its first result is only a partial view.
                # Each member copies its logical values into the shared allocation.
                envelope = constant(0)
                for member in [ordinal, *peers]:
                    other, other_type = records[member], ir.MemRefType(old[member].type)
                    dims = [
                        memref.DimOp(old[member], constant(i)).result if n == dynamic else constant(n)
                        for i, n in enumerate(other_type.shape)
                    ]
                    bound = expression(other["storage_offset"], dims)
                    empty = arith.ConstantOp(
                        ir.IntegerType.get_signless(1), ir.IntegerAttr.get(ir.IntegerType.get_signless(1), 0)
                    ).result
                    for size, stride in zip(dims, other["stride"]):
                        zero = arith.CmpIOp(0, size, constant(0)).result
                        empty = arith.OrIOp(empty, zero).result
                        term = arith.MulIOp(arith.SubIOp(size, constant(1)).result, expression(stride, dims)).result
                        bound = arith.AddIOp(bound, term).result
                    bound = arith.SelectOp(empty, constant(0), arith.AddIOp(bound, constant(1)).result).result
                    larger = arith.CmpIOp(4, bound, envelope).result
                    envelope = arith.SelectOp(larger, bound, envelope).result
                source = memref.AllocOp(ir.MemRefType.get([dynamic], typ.element_type), [envelope], []).result
                copy = True
            if ir.MemRefType(source.type).element_type != typ.element_type:
                raise ValueError("result alias changes element type")
            static_offset = offset if type(offset) is int else dynamic
            static_strides = [s if type(s) is int else dynamic for s in strides]
            target = ir.MemRefType.get(
                typ.shape, typ.element_type, layout=ir.StridedLayoutAttr.get(static_offset, static_strides)
            )
            view = memref.ReinterpretCastOp(
                target,
                source,
                [] if static_offset != dynamic else [expression(offset)],
                [s for s, n in zip(sizes, typ.shape) if n == dynamic],
                [expression(s) for s, n in zip(strides, static_strides) if n == dynamic],
                [static_offset],
                list(typ.shape),
                static_strides,
            ).result
            if copy:
                memref.CopyOp(value, view)
            rewritten.append(view)
        # Discarded owned results no longer escape. Release each allocation once,
        # retaining any allocation still backing an unchanged/aliased result.
        live_roots = [root(v) for v in rewritten]
        # Unknown view producers may still alias any discarded allocation. Without
        # complete live-root provenance, no local ownership proof can authorize a free.
        live_roots_known = all(
            isinstance(v.owner, ir.Block)
            or (isinstance(owner_operation(v), ir.Operation) and owner_operation(v).name == "memref.alloc")
            for v in live_roots
        )
        freed = []
        for value in old:
            allocation = root(value)
            if (
                live_roots_known
                and allocation not in live_roots
                and allocation not in freed
                and isinstance(owner_operation(allocation), ir.Operation)
                and owner_operation(allocation).name == "memref.alloc"
            ):
                memref.DeallocOp(allocation)
                freed.append(allocation)
    for i, value in enumerate(rewritten):
        returned.operation.operands[i] = value
    function.attributes["function_type"] = ir.TypeAttr.get(
        ir.FunctionType.get([v.type for v in block.arguments], [v.type for v in rewritten])
    )
    module.operation.verify()


def _layout_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    import json

    from torch_mlir import ir

    key = "m2m.result_layout"
    if key not in module.operation.attributes:
        return _layout_original_stages(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    contract = json.loads(ir.StringAttr(module.operation.attributes[key]).value)
    has_inputs = any(record is not None for record in contract.get("inputs", []))
    passes = _fg_split(pipeline)
    if any(p.startswith("buffer-results-to-out-params") for p in passes):
        if has_inputs:
            if not any(p.startswith("one-shot-bufferize") for p in passes):
                raise ValueError("cannot apply captured input-layout contract: pipeline lacks one-shot-bufferize")
            refuse_overlapping_input_aliases(contract)
            with ctx:
                _apply_input_layout(module, contract)
        # This ABI copies logical results into caller-owned output parameters;
        # it intentionally does not return frontend storage descriptors. Input
        # layouts still govern every load and must reach bufferization intact.
        del module.operation.attributes[key]
        return _layout_original_stages(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    buffers = [i for i, p in enumerate(passes) if p.startswith("one-shot-bufferize")]
    if not buffers:
        if has_inputs:
            raise ValueError("cannot skip captured input-layout contract: pipeline lacks one-shot-bufferize")
        return _layout_original_stages(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    refuse_overlapping_input_aliases(contract)
    with ctx:
        _apply_input_layout(module, contract)
    del module.operation.attributes[key]
    # Ownership normalization may clone borrowed returns. Reconstruct the public
    # borrowed descriptors afterwards, and reclaim those discarded owned clones.
    dealloc = [
        i for i, p in enumerate(passes) if p in ("bufferization-lower-deallocations", "convert-bufferization-to-memref")
    ]
    cut = (dealloc[-1] if dealloc else buffers[-1]) + 1
    head, tail = passes[:cut], passes[cut:]
    hoist_head = any("buffer-loop-hoisting" in p for p in head)
    late_head = any("convert-scf-to-openmp" in p for p in head)
    _layout_original_stages(
        ctx,
        module,
        ",".join(head),
        erase if hoist_head else 0,
        mid if hoist_head else (),
        late if late_head else (),
        post_openmp if late_head else (),
        pre_generalize,
    )
    with ctx:
        _apply_result_layout(module, contract)
    _layout_original_stages(
        ctx,
        module,
        ",".join(tail),
        0 if hoist_head else erase,
        () if hoist_head else mid,
        () if late_head else late,
        () if late_head else post_openmp,
        (),
    )


def inject(source: str) -> str:
    marker = "src_path, out_path, pipeline ="
    from .input_layout import (
        _apply_input_layout,
        has_overlapping_elements,
        refuse_overlapping_input_aliases,
        static_storage_span,
    )

    binding = (
        inspect.getsource(has_overlapping_elements)
        + "\n"
        + inspect.getsource(static_storage_span)
        + "\n"
        + inspect.getsource(refuse_overlapping_input_aliases)
        + "\n"
        + inspect.getsource(_apply_input_layout)
        + "\n"
        + inspect.getsource(_apply_result_layout)
        + "\n"
        + inspect.getsource(_layout_stages)
    )
    binding += "\n_layout_original_stages = _run_stages\n_run_stages = _layout_stages\n"
    return source.replace(marker, binding + marker)
