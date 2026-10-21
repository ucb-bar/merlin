"""Canonical native scalar host algorithms, without runner or tool discovery.

These source literals execute only in the caller's existing native MLIR context.
Ordinary lowering and the separately reviewed host library use identical bodies.
"""

CONTRACTION_SOURCE = r"""
def _scalar_contraction_spec(op):
    from torch_mlir import ir as _sc_ir
    if (op.operation.name != "linalg.generic" or len(op.operands) != 3
            or len(op.results) != 1 or len(op.regions) != 1
            or len(op.regions[0].blocks) != 1):
        return None
    types = list(v.type for v in op.operands)
    if any(not isinstance(t, _sc_ir.RankedTensorType) for t in types):
        return None
    types = [_sc_ir.RankedTensorType(t) for t in types]
    if (any(not isinstance(t.element_type, _sc_ir.F32Type) for t in types)
            or op.results[0].type != op.operands[2].type
            or any(d < 0 for t in types for d in t.shape)):
        return None
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != 3 or len(body) != 3
            or [x.operation.name for x in body] != ["arith.mulf", "arith.addf", "linalg.yield"]
            or any("fastmath" in x.attributes
                   and str(x.attributes["fastmath"]) != "#arith.fastmath<none>" for x in body)
            or any(name != "fastmath" and not name.startswith("prov.")
                   for x in body[:2] for name in x.attributes)):
        return None
    mul, add, yield_op = body
    args = list(block.arguments)
    if list(mul.operands) == args[:2]:
        mul_order = [0, 1]
    elif list(mul.operands) == args[1::-1]:
        mul_order = [1, 0]
    else:
        return None
    if list(add.operands) == [mul.results[0], args[2]]:
        add_order = [0, 1]
    elif list(add.operands) == [args[2], mul.results[0]]:
        add_order = [1, 0]
    else:
        return None
    if list(yield_op.operands) != [add.results[0]]:
        return None
    maps = [_sc_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    if len(maps) != 3:
        return None
    dims = maps[0].n_dims
    if dims < 2 or any(m.n_dims != dims or m.n_symbols for m in maps):
        return None
    iters = list(op.attributes["iterator_types"])
    # Enum attribute printer is the upstream binding's stable spelling.
    if ([str(x) for x in iters] !=
            ["#linalg.iterator_type<parallel>"] * (dims - 1) + ["#linalg.iterator_type<reduction>"]):
        return None
    positions = []
    extents = [None] * dims
    for m, t in zip(maps, types):
        if len(m.results) != len(t.shape):
            return None
        pos = []
        for expr, extent in zip(m.results, t.shape):
            if not isinstance(expr, _sc_ir.AffineDimExpr):
                return None
            d = _sc_ir.AffineDimExpr(expr).position
            if d in pos or (extents[d] is not None and extents[d] != extent):
                return None
            pos.append(d)
            extents[d] = extent
        positions.append(pos)
    if (sorted(positions[2]) != list(range(dims - 1))
            or any(dims - 1 not in pos for pos in positions[:2])
            or any(d is None for d in extents)):
        return None
    return positions, extents, mul_order, add_order


def _scalarize_tensor_contractions(
        ctx, module, outputs=1, reduction_unroll=0, rows=1, selection=None, output_guard=None):
    from torch_mlir import ir as _sc_ir
    todo = []

    def walk(op):
        for region in op.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _scalar_contraction_spec(inner)
                    eligible = (spec is not None and spec[1][-2] % outputs == 0
                                and (selection is None or any(inner.operation == item for item in selection)))
                    if eligible and rows != 1:
                        positions, extents, _, _ = spec
                        row_axis, col_axis = len(extents) - 3, len(extents) - 2
                        # A rectangular tile is a matrix schedule only when the
                        # exact projected maps prove complementary row/column
                        # reuse. No names, shapes alone or pointer facts suffice.
                        eligible = (row_axis >= 0 and extents[row_axis] % rows == 0
                                    and row_axis in positions[0] and col_axis not in positions[0]
                                    and col_axis in positions[1] and row_axis not in positions[1])
                        owner = inner.operation
                        while eligible and owner is not None:
                            if any(a in owner.attributes for a in ("strictfp", "llvm.strictfp")):
                                eligible = False
                            owner = owner.parent
                    if eligible:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)

    walk(module.operation)
    with ctx:
        for old, (positions, extents, mul_order, add_order) in todo:
            with old.location, _sc_ir.InsertionPoint(old):
                index = _sc_ir.IndexType.get()
                scalar = old.regions[0].blocks[0].arguments[0].type
                tensor = old.results[0].type
                def create(name, operands=(), results=(), attributes=None, regions=0):
                    return _sc_ir.Operation.create(name, operands=list(operands), results=list(results),
                                                   attributes=attributes or {}, regions=regions)
                def constant(value):
                    return create("arith.constant", results=[index], attributes={
                        "value": _sc_ir.IntegerAttr.get(index, value)}).results[0]
                zero, one = constant(0), constant(1)
                limits = [constant(e) for e in extents]
                step = one if outputs == 1 else constant(outputs)
                offsets = [zero, *[constant(i) for i in range(1, outputs)]]
                row_offsets = [zero, *[constant(i) for i in range(1, rows)]]
                row_step = one if rows == 1 else constant(rows)

                def output_loop(depth, current, indices):
                    if depth < len(extents) - 1:
                        stride = (step if depth == len(extents) - 2 else
                                  row_step if rows != 1 and depth == len(extents) - 3 else one)
                        loop = create("scf.for", [zero, limits[depth], stride, current], [tensor], regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, tensor])
                        with _sc_ir.InsertionPoint(block):
                            updated = output_loop(depth + 1, block.arguments[1], [*indices, block.arguments[0]])
                            create("scf.yield", [updated])
                        return loop.results[0]
                    if rows == 1:
                        lanes = [indices]
                        for offset in offsets[1:]:
                            column = create("arith.addi", [indices[-1], offset], [index]).results[0]
                            lanes.append([*indices[:-1], column])
                    else:
                        lanes = []
                        for row_offset in row_offsets:
                            row = (indices[-2] if row_offset == zero else
                                   create("arith.addi", [indices[-2], row_offset], [index]).results[0])
                            for offset in offsets:
                                column = (indices[-1] if offset == zero else
                                          create("arith.addi", [indices[-1], offset], [index]).results[0])
                                lanes.append([*indices[:-2], row, column])
                    out_indices = [[lane[d] for d in positions[2]] for lane in lanes]
                    def reduction():
                        initial = [create("tensor.extract", [current, *idx], [scalar]).results[0]
                                   for idx in out_indices]
                        # SCF -> CF forwards LLVM-typed values; CF -> LLVM keeps the key.
                        # Use the final branch property's spelling: a retained namespaced
                        # llvm.loop_annotation is ignored by the installed translator.
                        attributes = ({"loop_annotation": _sc_ir.Attribute.parse(
                            "#llvm.loop_annotation<unroll = <count = " + str(reduction_unroll) + " : i32>>")}
                            if reduction_unroll else {})
                        lane_count = rows * outputs
                        loop = create("scf.for", [zero, limits[-1], one, *initial], [scalar] * lane_count,
                                      attributes=attributes, regions=1)
                        block = _sc_ir.Block.create_at_start(loop.regions[0], [index, *([scalar] * lane_count)])
                        with _sc_ir.InsertionPoint(block):
                            accumulated, shared = [], {}
                            for lane_number, lane in enumerate(lanes):
                                all_indices = [*lane, block.arguments[0]]
                                values = []
                                for i in range(2):
                                    if rows == 1:
                                        key = i if len(extents) - 2 not in positions[i] else None
                                    else:
                                        key = (i,
                                               lane_number // outputs if len(extents) - 3 in positions[i] else 0,
                                               lane_number % outputs if len(extents) - 2 in positions[i] else 0)
                                    if key is not None and key in shared:
                                        value = shared[key]
                                    else:
                                        value = create(
                                            "tensor.extract",
                                            [old.operands[i], *[all_indices[d] for d in positions[i]]],
                                            [scalar]).results[0]
                                        if key is not None:
                                            shared[key] = value
                                    values.append(value)
                                product = create("arith.mulf", [values[d] for d in mul_order], [scalar]).results[0]
                                operands = [product, block.arguments[lane_number + 1]]
                                accumulated.append(create(
                                    "arith.addf", [operands[d] for d in add_order], [scalar]).results[0])
                            create("scf.yield", accumulated)
                        destination = current
                        for value, idx in zip(loop.results, out_indices):
                            destination = create("tensor.insert", [value, destination, *idx], [tensor]).results[0]
                        return destination
                    if output_guard is None:
                        return reduction()
                    condition = output_guard(old.operation, lanes, positions[2], create, constant)
                    guarded = create("scf.if", [condition], [tensor], regions=2)
                    for branch in range(2):
                        block = _sc_ir.Block.create_at_start(guarded.regions[branch], [])
                        with _sc_ir.InsertionPoint(block):
                            create("scf.yield", [reduction() if branch == 0 else current])
                    return guarded.results[0]

                result = output_loop(0, old.operands[2], [])
                replacement = result.owner
                for name in old.attributes:
                    if name.startswith("prov."):
                        replacement.attributes[name] = old.attributes[name]
                previous = old.attributes["prov.transforms"] if "prov.transforms" in old.attributes else None
                text = _sc_ir.StringAttr(previous).value + "," if previous is not None else ""
                suffix = ("_2x4_outputs" if rows != 1 else
                          "" if outputs == 1 else "_" + str(outputs) + "_outputs")
                transforms = text + "scalar_contraction_accumulator" + suffix
                if reduction_unroll:
                    transforms += ",unroll_scalar_contraction_reduction_by_" + str(reduction_unroll)
                replacement.attributes["prov.transforms"] = _sc_ir.StringAttr.get(transforms)
            old.results[0].replace_all_uses_with(result)
            old.operation.erase()
    module.operation.verify()
    return len(todo)


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

BROADCAST_SOURCE = r"""
def _broadcast_math_spec(op):
    from torch_mlir import ir as _bh_ir
    if any("strictfp" in name or "strictfp" in str(op.attributes[name])
           for name in op.attributes):
        return None
    if (op.operation.name != "linalg.generic" or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or len(op.operands) < 2):
        return None
    types = [value.type for value in op.operands]
    if any(not isinstance(t, _bh_ir.RankedTensorType)
           or any(extent < 0 for extent in t.shape) for t in types):
        return None
    maps = [_bh_ir.AffineMapAttr(a).value for a in op.attributes["indexing_maps"]]
    dims = maps[-1].n_dims
    if (dims < 1 or len(maps) != len(types)
            or any(m.n_dims != dims or m.n_symbols for m in maps)
            or [str(x) for x in op.attributes["iterator_types"]]
               != ["#linalg.iterator_type<parallel>"] * dims):
        return None
    # Every domain dimension is fixed by the complete output permutation.
    out_dims = []
    extents = [None] * dims
    for expr, extent in zip(maps[-1].results, types[-1].shape):
        if not isinstance(expr, _bh_ir.AffineDimExpr):
            return None
        dimension = _bh_ir.AffineDimExpr(expr).position
        if dimension in out_dims:
            return None
        out_dims.append(dimension)
        extents[dimension] = extent
    if (sorted(out_dims) != list(range(dims))
            or any(extent is None or extent <= 0 for extent in extents)):
        # A producer must never evaluate an operation absent from an empty consumer.
        return None
    dependencies = []
    for affine, tensor in zip(maps, types):
        if len(affine.results) != len(tensor.shape):
            return None
        used = set()
        for expr, extent in zip(affine.results, tensor.shape):
            if isinstance(expr, _bh_ir.AffineDimExpr):
                dimension = _bh_ir.AffineDimExpr(expr).position
                if dimension in used or extents[dimension] != extent:
                    return None
                used.add(dimension)
            elif isinstance(expr, _bh_ir.AffineConstantExpr):
                if _bh_ir.AffineConstantExpr(expr).value != 0 or extent != 1:
                    return None
            else:
                return None
        dependencies.append(frozenset(used))
    block = op.regions[0].blocks[0]
    body = list(block.operations)
    if (len(block.arguments) != len(types) or not body
            or body[-1].operation.name != "linalg.yield"
            or len(body[-1].operands) != 1):
        return None
    # These registered source operations are scalar and have no observable effects.
    # Reject an entire body containing a call/index/memory/unknown operation: moving
    # a pure chain across it could change an observed floating environment.
    allowed = {"arith.constant", "arith.addf", "arith.subf", "arith.mulf", "arith.divf",
               "arith.negf", "arith.maximumf", "arith.minimumf", "arith.cmpf", "arith.cmpi",
               "arith.select", "arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi",
               "arith.muli", "arith.shli", "arith.shrsi", "arith.shrui", "arith.bitcast",
               "arith.sitofp", "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extsi",
               "arith.extui", "arith.trunci", "arith.extf", "arith.truncf", "math.rsqrt"}
    scalar_deps = dict(zip(block.arguments, dependencies))
    defining = {}
    candidates = []
    for scalar in body[:-1]:
        if any("strictfp" in name or "strictfp" in str(scalar.attributes[name])
               for name in scalar.attributes):
            return None
        if (scalar.operation.name not in allowed or scalar.regions
                or len(scalar.results) != 1
                or isinstance(scalar.results[0].type, (_bh_ir.VectorType, _bh_ir.ShapedType))):
            return None
        if ("fastmath" in scalar.attributes
                and str(scalar.attributes["fastmath"]) != "#arith.fastmath<none>"):
            return None
        deps = frozenset().union(*(scalar_deps.get(value, frozenset())
                                  for value in scalar.operands))
        scalar_deps[scalar.results[0]] = deps
        defining[scalar.results[0]] = scalar
        if (scalar.operation.name == "math.rsqrt"
                and isinstance(scalar.results[0].type,
                               (_bh_ir.F16Type, _bh_ir.F32Type, _bh_ir.F64Type))
                and any(d not in deps and extents[d] > 1 for d in range(dims))):
            candidates.append(scalar)
    if not candidates:
        return None
    target = candidates[0]
    needed = set()
    def need(value):
        if value in needed:
            return
        needed.add(value)
        scalar = defining.get(value)
        if scalar is not None:
            for operand in scalar.operands:
                need(operand)
    need(target.results[0])
    # A dependency on an output-init block argument cannot be read by a fresh producer.
    if block.arguments[-1] in needed:
        return None
    inputs = [i for i, value in enumerate(block.arguments[:-1]) if value in needed]
    chain = [scalar for scalar in body[:-1] if scalar.results[0] in needed]
    keep = sorted(scalar_deps[target.results[0]])
    return maps, extents, body, target, inputs, chain, keep


def _hoist_broadcast_source_math(ctx, module):
    from torch_mlir import ir as _bh_ir
    todo = []
    def walk(operation):
        if any("strictfp" in name or "strictfp" in str(operation.attributes[name])
               for name in operation.attributes):
            return
        for region in operation.regions:
            for block in region.blocks:
                for inner in list(block.operations):
                    spec = _broadcast_math_spec(inner)
                    if spec is not None:
                        todo.append((inner, spec))
                    else:
                        walk(inner.operation)
    walk(module.operation)
    with ctx:
        for old, (maps, extents, body, target, inputs, chain, keep) in todo:
            old_args = list(old.regions[0].blocks[0].arguments)
            with old.location:
                reduced_type = _bh_ir.RankedTensorType.get(
                    [extents[d] for d in keep], target.results[0].type)
            def create(name, operands=(), results=(), attributes=None, regions=0):
                return _bh_ir.Operation.create(name, operands=list(operands), results=list(results),
                                              attributes=attributes or {}, regions=regions)
            def attrs(operation):
                return {name: operation.attributes[name] for name in operation.attributes}
            def clone(scalar, values):
                with scalar.location:
                    copy = create(scalar.operation.name,
                                  [values.get(v, v) for v in scalar.operands],
                                  [v.type for v in scalar.results], attrs(scalar))
                values[scalar.results[0]] = copy.results[0]
            def map_on_reduced(affine):
                results = []
                for expr in affine.results:
                    if isinstance(expr, _bh_ir.AffineDimExpr):
                        position = _bh_ir.AffineDimExpr(expr).position
                        results.append(_bh_ir.AffineDimExpr.get(keep.index(position)))
                    else:
                        results.append(_bh_ir.AffineConstantExpr.get(0))
                return _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(len(keep), 0, results))
            with old.location, _bh_ir.InsertionPoint(old):
                empty = create("tensor.empty", results=[reduced_type])
                producer_attrs = {
                    "indexing_maps": _bh_ir.ArrayAttr.get(
                        [map_on_reduced(maps[i]) for i in inputs]
                        + [_bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get_identity(len(keep)))]),
                    "iterator_types": _bh_ir.ArrayAttr.get(
                        [_bh_ir.Attribute.parse("#linalg.iterator_type<parallel>")] * len(keep)),
                    "operandSegmentSizes": _bh_ir.DenseI32ArrayAttr.get([len(inputs), 1]),
                }
                producer = create("linalg.generic", [*[old.operands[i] for i in inputs], empty.results[0]],
                                  [reduced_type], producer_attrs, 1)
                producer_block = _bh_ir.Block.create_at_start(
                    producer.regions[0], [*[old_args[i].type for i in inputs], target.results[0].type])
                with _bh_ir.InsertionPoint(producer_block):
                    values = {old_args[i]: producer_block.arguments[j] for j, i in enumerate(inputs)}
                    for scalar in chain:
                        clone(scalar, values)
                    create("linalg.yield", [values[target.results[0]]])
                consumer_attrs = attrs(old)
                consumer_attrs["indexing_maps"] = _bh_ir.ArrayAttr.get(
                    [*[_bh_ir.AffineMapAttr.get(m) for m in maps[:-1]],
                     _bh_ir.AffineMapAttr.get(_bh_ir.AffineMap.get(maps[-1].n_dims, 0,
                                               [_bh_ir.AffineDimExpr.get(d) for d in keep])),
                     _bh_ir.AffineMapAttr.get(maps[-1])])
                consumer_attrs["operandSegmentSizes"] = _bh_ir.DenseI32ArrayAttr.get([len(old.operands), 1])
                consumer = create("linalg.generic", [*old.operands[:-1], producer.results[0], old.operands[-1]],
                                  [old.results[0].type], consumer_attrs, 1)
                consumer_block = _bh_ir.Block.create_at_start(
                    consumer.regions[0], [*[v.type for v in old_args[:-1]],
                                         target.results[0].type, old_args[-1].type])
                with _bh_ir.InsertionPoint(consumer_block):
                    values = {old_args[i]: consumer_block.arguments[i] for i in range(len(old_args) - 1)}
                    values[old_args[-1]] = consumer_block.arguments[-1]
                    values[target.results[0]] = consumer_block.arguments[-2]
                    # Backward liveness preserves every other use of a chain intermediate.
                    live = set(body[-1].operands)
                    retained = []
                    for scalar in reversed(body[:-1]):
                        if scalar is target:
                            continue
                        if scalar.results[0] in live:
                            retained.append(scalar)
                            live.update(scalar.operands)
                    for scalar in reversed(retained):
                        clone(scalar, values)
                    create("linalg.yield", [values.get(v, v) for v in body[-1].operands])
            old.results[0].replace_all_uses_with(consumer.results[0])
            old.operation.erase()
    module.operation.verify()
    return len(todo)

"""
