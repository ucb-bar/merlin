"""Ordered native squared-sum matching and materialization.

Native source executes in the caller-selected MLIR context; no tool discovery.
"""

SQUARED_SUM_SOURCE = r"""
def _sqs_spec(op):
    from torch_mlir import ir as _sqs_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 2
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    ancestor = op.operation
    while ancestor is not None:
        if any("strictfp" in name or "strictfp" in str(ancestor.attributes[name])
               for name in ancestor.attributes):
            return None
        ancestor = ancestor.parent
    types = []
    for value in op.operands:
        if not isinstance(value.type, _sqs_ir.RankedTensorType):
            return None
        tensor = _sqs_ir.RankedTensorType(value.type)
        if (not isinstance(tensor.element_type, _sqs_ir.F32Type)
                or any(extent < 0 for extent in tensor.shape)):
            return None
        types.append(tensor)
    if op.results[0].type != op.operands[1].type:
        return None
    allowed_attributes = {"indexing_maps", "iterator_types", "operandSegmentSizes"}
    if any(name not in allowed_attributes and not name.startswith("prov.")
           for name in op.attributes):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 2 or len(body) != 3
            or [x.operation.name for x in body]
                != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any(x.regions for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for scalar in body for name in scalar.attributes)
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>"
                   for x in body)):
        return None
    mul, add, terminator = body
    value, seed = block.arguments
    if list(mul.operands) != [value, value]:
        return None
    if list(add.operands) == [mul.results[0], seed]:
        order = [0, 1]
    elif list(add.operands) == [seed, mul.results[0]]:
        order = [1, 0]
    else:
        return None
    if list(terminator.operands) != [add.results[0]]:
        return None
    maps = [_sqs_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 2:
        return None
    dims = maps[0].n_dims
    if (dims < 1 or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
                != ["#linalg.iterator_type<parallel>"] * (dims - 1)
                   + ["#linalg.iterator_type<reduction>"]):
        return None
    extents, positions = [None] * dims, []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = []
        for expr, extent in zip(affine.results, tensor.shape):
            if not isinstance(expr, _sqs_ir.AffineDimExpr):
                return None
            position = _sqs_ir.AffineDimExpr(expr).position
            if (position in used or position >= dims
                    or (extents[position] is not None and extents[position] != extent)):
                return None
            used.append(position)
            extents[position] = extent
        positions.append(used)
    if (sorted(positions[0]) != list(range(dims))
            or sorted(positions[1]) != list(range(dims - 1))
            or any(extent is None for extent in extents)):
        return None
    return positions, extents, order


def _scalarize_squared_sums(ctx, module):
    from torch_mlir import ir as _sqs_ir
    todo = []
    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for child in list(block.operations):
                    spec = _sqs_spec(child)
                    if spec is not None:
                        todo.append((child, spec))
                    else:
                        walk(child.operation)
    walk(module.operation)
    with ctx:
        for old, (positions, extents, add_order) in todo:
            with old.location, _sqs_ir.InsertionPoint(old):
                index = _sqs_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sqs_ir.Operation.create(name, operands=list(operands),
                        results=list(results), attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sqs_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(extent) for extent in extents]
                source_mul, source_add = list(old.regions[0].blocks[0].operations)[:2]
                def output_loop(depth, current, coordinates):
                    if depth < len(extents) - 1:
                        loop = create("scf.for", [zero, limits[depth], one, current],
                                      [tensor], regions=1)
                        block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sqs_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1],
                                                  [*coordinates, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    out_indices = [coordinates[d] for d in positions[1]]
                    initial = create("tensor.extract", [current, *out_indices], [scalar]).results[0]
                    loop = create("scf.for", [zero, limits[-1], one, initial],
                                  [scalar], regions=1)
                    block = _sqs_ir.Block.create_at_start(loop.regions[0], [index, scalar])
                    with _sqs_ir.InsertionPoint(block):
                        all_indices = [*coordinates, block.arguments[0]]
                        value = create("tensor.extract", [old.operands[0],
                            *[all_indices[d] for d in positions[0]]], [scalar]).results[0]
                        product = create("arith.mulf", [value, value], [scalar],
                            {name: source_mul.attributes[name] for name in source_mul.attributes}).results[0]
                        values = [product, block.arguments[1]]
                        total = create("arith.addf", [values[d] for d in add_order], [scalar],
                            {name: source_add.attributes[name] for name in source_add.attributes}).results[0]
                        create("scf.yield", [total])
                    return create("tensor.insert", [loop.results[0], current, *out_indices],
                                  [tensor]).results[0]
                result = output_loop(0, old.operands[1], [])
                for name in old.attributes:
                    if name.startswith("prov."):
                        result.owner.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                prefix = _sqs_ir.StringAttr(previous).value + "," if previous is not None else ""
                result.owner.attributes["prov.transforms"] = _sqs_ir.StringAttr.get(
                    prefix + "scalar_squared_sum_accumulator")
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)

"""
