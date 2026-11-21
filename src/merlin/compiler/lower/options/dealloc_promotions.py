"""Native proofs for fresh loop generations promoted to returned ownership.

Reuse upstream's ownership flag and zero-trip clone, before lowering aggregate
deallocs. No loop bound, tensor size or source behavior is specialized.
"""

PRELUDE = r"""
def _obl_copied_branch(operations, sources):
    if not operations or operations[-1].name != 'scf.yield' or len(operations[-1].operands) != len(sources):
        return False
    targets = tuple(operations[-1].operands)
    if len(set(targets)) != len(targets):
        return False
    checked = set()
    order = {op: i for i, op in enumerate(operations)}
    for target, source in zip(targets, sources):
        if not isinstance(target, ir.OpResult):
            return False
        alloc = target.owner.operation
        if alloc.name == 'bufferization.clone':
            if tuple(alloc.operands) != (source,) or len(alloc.results) != 1:
                return False
            checked.add(alloc)
            continue
        if alloc.name != 'memref.alloc':
            return False
        typ = ir.MemRefType(target.type)
        if (target.type != source.type
                or typ.layout.value != ir.AffineMap.get_identity(len(typ.shape), context=typ.context)):
            return False
        copies = [op for op in operations if op.name == 'memref.copy' and tuple(op.operands) == (source, target)]
        if len(copies) != 1 or alloc not in order or order[alloc] >= order[copies[0]]:
            return False
        dims = [axis for axis, size in enumerate(typ.shape) if size < 0]
        if len(alloc.operands) != len(dims):
            return False
        for dimension, axis in zip(alloc.operands, dims):
            if (not isinstance(dimension, ir.OpResult) or dimension.owner.name != 'memref.dim'
                    or dimension.owner.operands[0] != source or _obl_constant(dimension.owner.operands[1]) != axis):
                return False
        checked.update((alloc, copies[0]))
    for op in operations[:-1]:
        if op in checked:
            continue
        if op.name == 'arith.constant' and all(str(value.type) == 'index' for value in op.results):
            continue
        if op.name == 'memref.dim' and op.operands[0] in sources:
            continue
        return False
    return True


def _obl_promotions(loop):
    if loop.name != 'scf.for' or len(loop.regions) != 1 or len(loop.regions[0].blocks) != 1:
        return None
    body = loop.regions[0].blocks[0]
    terminal = tuple(body.operations)[-1]
    changed = [i for i, arg in enumerate(body.arguments[1:])
               if _memref_typed(arg) and terminal.operands[i] != arg]
    if not changed:
        return None
    flags = {loop.results[i] for i, value in enumerate(loop.operands[3:])
             if str(value.type) == 'i1' and _obl_constant(value) == 0
             and _obl_constant(terminal.operands[i]) == -1}
    found = {}
    for item in loop.block.operations:
        op = item.operation
        if op.name != 'scf.if' or len(op.operands) != 1 or op.operands[0] not in flags:
            continue
        if not op.results or len(op.regions) != 2 or any(len(r.blocks) != 1 for r in op.regions):
            continue
        yes, no = (tuple(child.operation for child in r.blocks[0].operations) for r in op.regions)
        if len(yes) != 1 or yes[0].name != 'scf.yield' or len(yes[0].operands) != len(op.results):
            continue
        values = tuple(yes[0].operands)
        if not _obl_copied_branch(no, values):
            continue
        slots = [[i for i in changed if loop.results[i] == value] for value in values]
        if any(len(indices) != 1 for indices in slots):
            continue
        for indices in slots:
            if indices[0] in found:
                return None
            found[indices[0]] = op
    return found if set(found) == set(changed) else None


def _obl_promoted_result(loop, index, promotion):
    yielded = tuple(promotion.regions[0].blocks[0].operations)[-1].operands
    positions = [i for i, value in enumerate(yielded) if value == loop.results[index]]
    if len(positions) != 1:
        raise ValueError('promoted owner changes its original result position')
    return promotion.results[positions[0]]


def _obl_base(value):
    seen = set()
    while isinstance(value, ir.OpResult) and value.owner.name in ('memref.cast', 'memref.extract_strided_metadata'):
        if value in seen or value != value.owner.results[0]:
            return None
        seen.add(value)
        value = value.owner.operands[0]
    return value


def _obl_promoted_loop(loop, calls, *, normalize):
    loop = loop.operation
    promotions = _obl_promotions(loop)
    if promotions is None:
        return None
    fresh = _obl_prove(loop, calls, promotion=True)
    outer = tuple(item.operation for item in loop.block.operations)
    position = {item: i for i, item in enumerate(outer)}
    seeds = {_obl_base(loop.operands[3+i]) for i in fresh}
    if any(not isinstance(seed, ir.OpResult) or seed.owner.name != 'memref.alloc'
           or seed.owner.operation.block != loop.block for seed in seeds):
        raise ValueError('promoted loop requires actual local allocation seeds')
    for seed in seeds:
        _, uses = _obl_alias_uses(seed)
        for use in uses:
            if use.operation == loop or use.name in _OBL_VIEWS or use.name == 'memref.dealloc':
                continue
            if normalize and use.name == 'bufferization.dealloc' and use.block == loop.block:
                continue
            _obl_memory_types(use)
            if not _obl_known_use(use, calls):
                raise ValueError('unknown or escaping initial-owner use: ' + use.name)
            for inner in _obl_ops(use):
                _obl_memory_types(inner)
                if inner.name not in _OBL_SCALARS:
                    raise ValueError('unknown initial-owner consumer effect: ' + inner.name)
    for i, promotion in promotions.items():
        if position[promotion] <= position[loop]:
            raise ValueError('ownership promotion must follow its actual loop')
        aliases, uses = _obl_alias_uses(loop.results[i])
        branch_ops = {op.operation for op in _obl_ops(promotion)}
        for use in uses:
            if use.operation in branch_ops or use.name in _OBL_VIEWS:
                continue
            if normalize and use.name == 'bufferization.dealloc' and use.block == loop.block:
                continue
            raise ValueError('loop result escapes its checked ownership promotion: ' + use.name)
        _, returned_uses = _obl_alias_uses(_obl_promoted_result(loop, i, promotion))
        returns = [use for use in returned_uses if use.name == 'func.return' and use.block == loop.block]
        if len(returns) != 1 or any(use.name != 'func.return' and not normalize for use in returned_uses):
            raise ValueError('promoted ownership must transfer through one complete function return')
        if any(use.name != 'func.return' and not (normalize and use.name == 'bufferization.dealloc')
               for use in returned_uses):
            raise ValueError('unknown, releasing or escaping promoted-owner use')
    if normalize:
        return fresh, promotions, seeds
    if any(any(op.operation.name == 'bufferization.clone' for op in _obl_ops(promotion))
           for promotion in promotions.values()):
        raise ValueError('returned ownership requires a materialized independent zero-trip copy')
    last = max(position[op] for op in promotions.values())
    for seed in seeds:
        aliases, uses = _obl_alias_uses(seed)
        frees = [op for op in uses if op.name == 'memref.dealloc']
        if (len(frees) != 1 or frees[0].block != loop.block or not _obl_release_value(frees[0], aliases)
                or position[frees[0].operation] <= last):
            raise ValueError('zero-trip seed must be released once after every ownership promotion')
    return fresh


def _normalize_promoted_loop_deallocs(ctx, module):
    calls, _, violations = _obl_borrowed_calls(module)
    if violations:
        raise ValueError('promoted loop borrowed ABI: ' + '; '.join(violations))
    changed = 0
    for loop in [op for op in _obl_ops(module.operation) if op.name == 'scf.for']:
        if _obl_promotions(loop) is None:
            continue
        fresh, promotions, seeds = _obl_promoted_loop(loop, calls, normalize=True)
        promoted = {_obl_promoted_result(loop, i, op) for i, op in promotions.items()}
        candidates = [op.operation for op in loop.block.operations if op.operation.name == 'bufferization.dealloc']
        for op in candidates:
            if any(tuple(result.uses) for result in op.results):
                continue
            n, nc, nr = tuple(ir.DenseI32ArrayAttr(op.attributes['operandSegmentSizes']))
            if n != nc:
                continue
            mems, conditions, retained = tuple(op.operands[:n]), tuple(op.operands[n:2*n]), tuple(op.operands[2*n:])
            if len(retained) != nr or not promoted.issubset(retained):
                continue
            # Extra retained values must be fresh, disjoint allocations/clones.
            if any(value not in promoted and (not isinstance(value, ir.OpResult)
                    or value.owner.name != 'memref.alloc'
                    or _obl_base(value) in seeds) for value in retained):
                continue
            seed_indices, final_indices = {}, {}
            for index, (mem, cond) in enumerate(zip(mems, conditions)):
                base = _obl_base(mem)
                if base in seeds and _obl_constant(cond) == -1:
                    seed_indices.setdefault(base, []).append(index)
                for i, promotion in promotions.items():
                    if base == loop.results[i] and cond == promotion.operands[0]:
                        final_indices.setdefault(i, []).append(index)
            if set(seed_indices) != seeds or any(len(ids) != 1 for ids in seed_indices.values()):
                continue
            if set(final_indices) != set(fresh) or any(len(ids) != 1 for ids in final_indices.values()):
                continue
            remove = {index for ids in (*seed_indices.values(), *final_indices.values()) for index in ids}
            # A duplicate seed/final candidate with another condition is not
            # covered by the proof; preserving it would double release owners.
            if any(index not in remove and _obl_base(mem) in seeds | {loop.results[i] for i in fresh}
                   for index, mem in enumerate(mems)):
                continue
            positions = {item.operation: j for j, item in enumerate(loop.block.operations)}
            if positions[op] <= max(positions[p] for p in promotions.values()):
                continue
            keep = [i for i in range(n) if i not in remove]
            # CloneOp itself permits aliasing. Materialize the same ranked
            # alloc/copy implementation as upstream CloneOpConversion before
            # using freshness to simplify any ownership operation.
            unique_promotions = tuple(dict.fromkeys(promotions.values()))
            clones = [child for promotion in unique_promotions for child in _obl_ops(promotion)
                      if child.name == 'bufferization.clone']
            for clone in clones:
                typ = ir.MemRefType(clone.results[0].type)
                if (typ.layout.value != ir.AffineMap.get_identity(len(typ.shape), context=ctx)
                        or clone.operands[0].type != clone.results[0].type):
                    raise ValueError('promoted copy requires actual identity-layout ranked storage')
            for promotion in unique_promotions:
                clones = [child for child in _obl_ops(promotion) if child.name == 'bufferization.clone']
                for clone in clones:
                    typ = ir.MemRefType(clone.results[0].type)
                    if (typ.layout.value != ir.AffineMap.get_identity(len(typ.shape), context=ctx)
                            or clone.operands[0].type != clone.results[0].type):
                        raise ValueError('promoted copy requires actual identity-layout ranked storage')
                    with ctx, clone.location, ir.InsertionPoint(clone):
                        dimensions = []
                        for axis, size in enumerate(typ.shape):
                            if size >= 0:
                                continue
                            index = ir.Operation.create('arith.constant', results=[ir.IndexType.get()],
                                attributes={'value': ir.IntegerAttr.get(ir.IndexType.get(), axis)})
                            dimension = ir.Operation.create('memref.dim', results=[ir.IndexType.get()],
                                operands=[clone.operands[0], index.results[0]])
                            dimensions.append(dimension.results[0])
                        allocation = ir.Operation.create('memref.alloc', results=[clone.results[0].type],
                            operands=dimensions, attributes={'operandSegmentSizes': ir.DenseI32ArrayAttr.get(
                                [len(dimensions), 0])})
                        ir.Operation.create('memref.copy', operands=[clone.operands[0], allocation.results[0]])
                    clone.results[0].replace_all_uses_with(allocation.results[0])
                    clone.erase()
            with ctx, op.location, ir.InsertionPoint(op):
                for seed in seeds:
                    ir.Operation.create('memref.dealloc', operands=[seed])
                if keep:
                    ir.Operation.create('bufferization.dealloc', operands=[
                        *(mems[i] for i in keep), *(conditions[i] for i in keep), *retained],
                        results=[result.type for result in op.results], attributes={
                            'operandSegmentSizes': ir.DenseI32ArrayAttr.get([len(keep), len(keep), nr])})
            op.erase()
            changed += len(fresh)
            break
    module.operation.verify()
    print('OK promoted_loop_ownership', changed, 'returned generations')


def _protect_promoted_seed_releases(ctx, module):
    calls, _, violations = _obl_borrowed_calls(module)
    if violations:
        raise ValueError('promoted loop borrowed ABI: ' + '; '.join(violations))
    for loop in list(_obl_ops(module.operation)):
        if loop.name != 'scf.for' or _obl_promotions(loop) is None:
            continue
        _obl_promoted_loop(loop, calls, normalize=False)
        for i in _obl_promotions(loop):
            seed = _obl_base(loop.operands[3+i])
            _, uses = _obl_alias_uses(seed)
            for release in uses:
                if release.name != 'memref.dealloc' or release.operands[0] != seed:
                    continue
                # The upstream liveness pass otherwise follows its conservative
                # loop/if alias union to func.return and moves this free after
                # the terminator. An identity view shields only this proved
                # closing release. The independent checker still follows it.
                with ctx, release.location, ir.InsertionPoint(release):
                    view = ir.Operation.create('memref.cast', operands=[seed], results=[seed.type])
                release.operands[0] = view.results[0]
"""

RUNNER = r"""
_PROMOTED_ORIGINAL_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    anchor = 'bufferization-lower-deallocations'
    if anchor not in passes:
        return _PROMOTED_ORIGINAL_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    at = passes.index(anchor)
    head, tail = passes[:at], passes[at:]
    hoisted = any('buffer-loop-hoisting' in p for p in head)
    parallel = any('convert-scf-to-openmp' in p for p in head)
    _PROMOTED_ORIGINAL_RUN_STAGES(ctx, module, ','.join(head), erase if hoisted else 0,
        mid if hoisted else (), late if parallel else (), post_openmp if parallel else (), pre_generalize)
    _normalize_promoted_loop_deallocs(ctx, module)
    return _PROMOTED_ORIGINAL_RUN_STAGES(ctx, module, ','.join(tail), 0 if hoisted else erase,
        () if hoisted else mid, () if parallel else late, () if parallel else post_openmp, ())
"""
