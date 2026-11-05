"""Exact runtime tables for expensive pure math on a small integer domain.

The table is computed by the original scalar program on the executing CPU.
It contains no model data or approximated values. Source operations with an
observable floating environment, unknown effects or dynamic domains decline.
"""

SOURCE = r'''
def _memoize_finite_integer_math(ctx, module, max_bits=8, minimum_repetition=8):
    from torch_mlir import ir as ir
    import math
    pure = {
        'arith.constant', 'arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf',
        'arith.negf', 'arith.maximumf', 'arith.minimumf', 'arith.cmpf',
        'arith.cmpi', 'arith.select', 'arith.andi', 'arith.ori', 'arith.xori',
        'arith.addi', 'arith.subi', 'arith.muli', 'arith.shli', 'arith.shrsi',
        'arith.shrui', 'arith.bitcast', 'arith.sitofp', 'arith.uitofp',
        'arith.fptosi', 'arith.fptoui', 'arith.extsi', 'arith.extui',
        'arith.trunci', 'arith.extf', 'arith.truncf', 'math.exp', 'math.exp2',
        'math.sin', 'math.cos', 'math.tanh', 'math.sqrt', 'math.rsqrt',
        'math.roundeven', 'math.absf',
        'llvm.intr.roundeven',
    }
    expensive = {'math.exp', 'math.exp2', 'math.sin', 'math.cos', 'math.tanh'}
    changed = 0

    def attrs(op):
        return {name: op.attributes[name] for name in op.attributes}

    def ordinary(op):
        for name in op.attributes:
            if ('strictfp' in name or 'fenv' in name or 'strictfp' in str(op.attributes[name])
                    or name == 'fastmath' and str(op.attributes[name]) != '#arith.fastmath<none>'
                    or name == 'fastmathFlags' and str(op.attributes[name]) != '#llvm.fastmath<none>'):
                return False
        return True

    def match(op):
        if (op.operation.name != 'linalg.generic' or len(op.results) != 1
                or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
                or 'library_call' in op.attributes or not ordinary(op)):
            return None
        types = [value.type for value in op.operands]
        if (not types or any(not isinstance(typ, ir.RankedTensorType)
                or any(extent <= 0 for extent in typ.shape) for typ in types)):
            return None
        types = [ir.RankedTensorType(typ) for typ in types]
        maps = [ir.AffineMapAttr(a).value for a in op.attributes['indexing_maps']]
        rank = maps[-1].n_dims
        if (len(maps) != len(types) or maps[-1] != ir.AffineMap.get_identity(rank)
                or len(types[-1].shape) != rank or op.results[0].type != op.operands[-1].type
                or [str(x) for x in op.attributes['iterator_types']]
                    != ['#linalg.iterator_type<parallel>'] * rank):
            return None
        block = op.regions[0].blocks[0]
        body = list(block.operations)
        if (len(block.arguments) != len(types) or not body
                or body[-1].operation.name != 'linalg.yield'
                or list(block.arguments[-1].uses)):
            return None
        deps = {value: frozenset([i]) for i, value in enumerate(block.arguments)}
        defining = {}
        has_math = {}
        for scalar in body[:-1]:
            if (scalar.operation.name not in pure or scalar.regions
                    or len(scalar.results) != 1 or not ordinary(scalar)
                    or any(name not in {'value', 'fastmath', 'fastmathFlags', 'predicate'} and not name.startswith('prov.')
                           for name in scalar.attributes)):
                return None
            for operand in scalar.operands:
                if operand not in deps:
                    owner = getattr(operand.owner, 'operation', operand.owner)
                    if (not isinstance(owner, ir.Operation) or owner.name != 'arith.constant'
                            or not ordinary(owner)):
                        return None
                    deps[operand] = frozenset()
            value = scalar.results[0]
            deps[value] = frozenset().union(*(deps[v] for v in scalar.operands))
            defining[value] = scalar.operation
            has_math[value] = (scalar.operation.name in expensive
                              or any(has_math.get(v, False) for v in scalar.operands))
        # Prefer the last pure result before another independent input joins.
        # Multiple consumers retain exactly the same result through the lookup.
        for scalar in reversed(body[:-1]):
            root = scalar.results[0]
            if not has_math[root] or len(deps[root]) != 1 or not isinstance(root.type, ir.F32Type):
                continue
            position, = deps[root]
            if position >= len(types) - 1:
                continue
            integer = block.arguments[position]
            if not isinstance(integer.type, ir.IntegerType):
                continue
            bits = ir.IntegerType(integer.type).width
            if (not 1 <= bits <= max_bits
                    or math.prod(types[-1].shape) < (1 << bits) * minimum_repetition):
                continue
            needed = set()
            constants = []
            def need(value):
                if value in needed:
                    return
                needed.add(value)
                if value == integer:
                    return
                owner = defining.get(value)
                if owner is None:
                    constants.append(getattr(value.owner, 'operation', value.owner))
                else:
                    for operand in owner.operands:
                        need(operand)
            need(root)
            chain = [s.operation for s in body[:-1] if s.results[0] in needed]
            # New table entries may cover values absent from this invocation.
            # Do not speculate potentially poison-producing conversions or
            # shifts. Their original uses after the prepared result stay put.
            if any(s.name in {'arith.fptosi', 'arith.fptoui', 'arith.shli',
                              'arith.shrsi', 'arith.shrui'} for s in chain):
                continue
            return block, integer, bits, root, chain, constants
        return None

    def walk(operation):
        nonlocal changed
        if not ordinary(operation):
            return
        for region in operation.regions:
            for block in region.blocks:
                for op in list(block.operations):
                    found = match(op)
                    if found is None:
                        walk(op.operation)
                        continue
                    scalar_block, integer, bits, root, chain, constants = found
                    with ctx, op.location:
                        def create(name, operands=(), results=(), attributes=None, regions=0):
                            return ir.Operation.create(name, operands=list(operands), results=list(results),
                                attributes=attributes or {}, regions=regions)
                        table_type = ir.RankedTensorType.get([1 << bits], root.type)
                        with ir.InsertionPoint(op):
                            empty = create('tensor.empty', results=[table_type])
                            table = create('linalg.generic', [empty.results[0]], [table_type], {
                                'operandSegmentSizes': ir.DenseI32ArrayAttr.get([0, 1]),
                                'indexing_maps': ir.ArrayAttr.get([
                                    ir.AffineMapAttr.get(ir.AffineMap.get_identity(1))]),
                                'iterator_types': ir.ArrayAttr.get([
                                    ir.Attribute.parse('#linalg.iterator_type<parallel>')]),
                            }, regions=1)
                        table_block = ir.Block.create_at_start(table.regions[0], [root.type])
                        with ir.InsertionPoint(table_block):
                            index = create('linalg.index', results=[ir.IndexType.get()], attributes={
                                'dim': ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)})
                            key = create('arith.index_cast', [index.results[0]], [integer.type])
                            values = {integer: key.results[0]}
                            for constant in constants:
                                copy = create(constant.name, results=[v.type for v in constant.results],
                                              attributes=attrs(constant))
                                values.update(zip(constant.results, copy.results))
                            for scalar in chain:
                                copy = create(scalar.name, [values[v] for v in scalar.operands],
                                    [v.type for v in scalar.results], attrs(scalar))
                                values.update(zip(scalar.results, copy.results))
                            create('linalg.yield', [values[root]])
                        with ir.InsertionPoint(root.owner):
                            key = create('arith.index_castui', [integer], [ir.IndexType.get()])
                            lookup = create('tensor.extract', [table.results[0], key.results[0]], [root.type])
                        root.replace_all_uses_with(lookup.results[0])
                        for scalar in reversed(chain):
                            if not any(list(v.uses) for v in scalar.results):
                                scalar.erase()
                        changed += 1
    with ctx:
        walk(module.operation)
        module.operation.verify()
    return changed
'''


def rewrite_native_module(context, module, *, ordinary_nontrapping, exception_flags_unobserved):
    """Apply to an existing native MLIR module under explicit source effects."""
    if any(type(value) is not bool for value in (ordinary_nontrapping, exception_flags_unobserved)):
        raise ValueError("finite-domain math requires explicit Boolean effects")
    if not ordinary_nontrapping or not exception_flags_unobserved:
        return 0
    namespace = {}
    exec(compile(SOURCE, "<merlin-finite-integer-math>", "exec"), namespace)
    return namespace['_memoize_finite_integer_math'](context, module)
