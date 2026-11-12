"""Expose complete loop-carried tensor destinations just before bufferization.

Upstream elementwise fusion deliberately resets unused init values to empty
tensors (RemoveOutsDependency). At the later bufferization seam, a completely
overwritten parallel output can instead retain the loop-carried value as its
destination. Arithmetic, order and value semantics stay unchanged. Native alias
analysis still decides physical reuse; no ownership or lifetime check is relaxed.
"""

import ast

FEATURE = "reuse_loop_tensor_destinations"
MARKER = "__merlin_loop_tensor_destinations__"

PRELUDE = r"""
def _ltd_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for child in tuple(block.operations):
                yield child.operation
                yield from _ltd_walk(child.operation)


def _ltd_numeric(type_):
    return isinstance(type_, (ir.IntegerType, ir.IndexType)) or str(type_) in (
        'f16', 'bf16', 'f32', 'f64',
        'f8E4M3FN', 'f8E4M3FNUZ', 'f8E5M2', 'f8E5M2FNUZ')


def _ltd_fresh_seed(value):
    if not isinstance(value, ir.OpResult):
        return False
    op = value.owner.operation
    if op.name == 'tensor.empty':
        return True
    if op.name == 'arith.constant':
        return 'value' in op.attributes and isinstance(op.attributes['value'], ir.DenseElementsAttr)
    if (op.name == 'linalg.generic' and 'library_call' not in op.attributes and len(op.results) == 1 and op.operands
            and all(_ltd_numeric(v.type) or isinstance(v.type, ir.RankedTensorType)
                    and _ltd_numeric(v.type.element_type) for v in tuple(op.operands)[:-1])
            and _ltd_fresh_seed(op.operands[-1]) and len(op.regions) == 1 and len(op.regions[0].blocks) == 1):
        segments = tuple(ir.DenseI32ArrayAttr(op.attributes['operandSegmentSizes']))
        scalar = op.regions[0].blocks[0]
        if segments != (len(op.operands) - 1, 1) or not scalar.arguments or list(scalar.arguments[-1].uses):
            return False
        with op.context:
            identity = ir.AffineMap.get_identity(len(ir.RankedTensorType(value.type).shape))
        maps = tuple(ir.ArrayAttr(op.attributes['indexing_maps']))
        return (ir.AffineMapAttr(maps[-1]).value == identity
                and all(str(it) == '#linalg.iterator_type<parallel>'
                        for it in ir.ArrayAttr(op.attributes['iterator_types']))
                # The recursively checked unused output seed is independent.
                # Tensor inputs and qualified readonly/private scalar captures
                # supply values without granting their storage ownership.
                and _ltd_scalar_body(op, fresh_seed=True))
    return (op.name == 'linalg.fill' and 'library_call' not in op.attributes and len(op.results) == 1
            and len(op.operands) == 2 and _ltd_numeric(op.operands[0].type) and _ltd_fresh_seed(op.operands[-1]))


def _ltd_readonly_input(value):
    views = ('memref.cast', 'memref.subview', 'memref.reinterpret_cast', 'memref.collapse_shape', 'memref.expand_shape')
    while isinstance(value, ir.OpResult):
        op = value.owner.operation
        if op.name in views:
            value = op.operands[0]
            continue
        if op.name != 'bufferization.to_buffer' or len(op.operands) != 1 or 'read_only' not in op.attributes:
            return False
        source = op.operands[0]
        if not isinstance(source, ir.BlockArgument) or source.owner.owner.operation.name != 'func.func':
            return False
        return source.owner == source.owner.owner.operation.regions[0].blocks[0]
    return False


def _ltd_private_root(value):
    # Called only with an independently synthesized carried seed. A live
    # private allocation is disjoint from it; no readonly/immutability fact is
    # invented for the buffer, and every original read/write remains in place.
    root = value
    while isinstance(root, ir.OpResult):
        op = root.owner.operation
        if op.name in ('memref.cast', 'memref.subview', 'memref.reinterpret_cast',
                       'memref.collapse_shape', 'memref.expand_shape', 'memref.view'):
            root = op.operands[0]
            continue
        if op.name == 'memref.alloc':
            return True
        break
    return False


def _ltd_separate_read_root(value):
    return _ltd_private_root(value) or _ltd_readonly_input(value)


def _ltd_tensor_read_root(value):
    while isinstance(value, ir.OpResult):
        op = value.owner.operation
        if op.name in ('tensor.cast', 'tensor.extract_slice', 'tensor.collapse_shape',
                       'tensor.expand_shape', 'tensor.reshape'):
            value = op.operands[0]
            continue
        if op.name in ('linalg.generic', 'linalg.fill', 'arith.constant'):
            # A checked complete value producer supplies readable tensor state
            # independently of the loop's fresh destination. Its recursively
            # closed body/init proof grants no ownership of its input buffers.
            return _ltd_fresh_seed(value)
        if op.name == 'bufferization.to_tensor':
            return (len(op.operands) == 1 and 'restrict' in op.attributes
                    and _ltd_separate_read_root(op.operands[0]))
        return False
    return (isinstance(value, ir.BlockArgument) and value.owner.owner.operation.name == 'func.func'
            and value.owner == value.owner.owner.operation.regions[0].blocks[0])


def _ltd_scalar_body(op, *, fresh_seed, private_atomics=False):
    # Readonly entry storage or an actual private allocation is disjoint from
    # a newly synthesized loop seed. This changes no capture or memory effect;
    # native bufferization still decides whether physical reuse is legal.
    for inner in _ltd_walk(op):
        name = inner.name
        if name == 'memref.generic_atomic_rmw':
            if (not private_atomics or not fresh_seed or not inner.operands
                    or not _ltd_private_root(inner.operands[0])):
                return False
            try:
                if str(inner.results[0].type) in ('f16', 'f32', 'f64'):
                    match = _word_match(inner)
                elif '_region_match' in globals():
                    match = _region_match(inner)
                else:
                    match = _cas_match(inner)
                if match is None:
                    return False
            except (ValueError, KeyError, IndexError):
                return False
            continue
        if name == 'memref.atomic_rmw':
            if (not private_atomics or not fresh_seed or len(inner.operands) < 2 or inner.regions
                    or not _ltd_private_root(inner.operands[1]) or len(inner.results) != 1
                    or (str(inner.results[0].type) not in ('i32', 'i64')
                        and (str(inner.results[0].type) not in ('f32', 'f64')
                             or ir.IntegerAttr(inner.attributes['kind']).value not in (0, 3)))
                    or inner.operands[0].type != inner.results[0].type
                    or any(not isinstance(v.type, ir.IndexType) for v in tuple(inner.operands)[2:])):
                return False
            continue
        if name in ('memref.load', 'memref.dim'):
            if (not fresh_seed or not inner.operands or not _ltd_separate_read_root(inner.operands[0])
                    or inner.regions or len(inner.results) != 1 or not _ltd_numeric(inner.results[0].type)
                    or any(not isinstance(v.type, ir.IndexType) for v in tuple(inner.operands)[1:])):
                return False
            continue
        if name in ('memref.cast', 'memref.subview', 'memref.reinterpret_cast', 'memref.collapse_shape',
                    'memref.expand_shape', 'memref.view'):
            if (not private_atomics or not fresh_seed or len(inner.results) != 1 or inner.regions
                    or not _ltd_private_root(inner.results[0])
                    or any(not _ltd_numeric(v.type) for v in tuple(inner.operands)[1:])):
                return False
            continue
        if name == 'cf.assert':
            if (not private_atomics or not fresh_seed or len(inner.operands) != 1
                    or str(inner.operands[0].type) != 'i1' or inner.results or inner.regions):
                return False
            continue
        if name == 'tensor.extract':
            if (not fresh_seed or not inner.operands
                    or not (_ltd_tensor_read_root(inner.operands[0])
                            or private_atomics and _ltd_snapshot_tensor_value(inner.operands[0]))
                    or inner.regions or len(inner.results) != 1 or not _ltd_numeric(inner.results[0].type)
                    or any(not isinstance(v.type, ir.IndexType) for v in tuple(inner.operands)[1:])):
                return False
            continue
        if name in ('tensor.empty', 'tensor.from_elements'):
            if (not private_atomics or not fresh_seed or inner.regions or len(inner.results) != 1
                    or not isinstance(inner.results[0].type, ir.RankedTensorType)):
                return False
            type_ = ir.RankedTensorType(inner.results[0].type)
            if (tuple(type_.shape) != (1,) or not _ltd_numeric(type_.element_type)
                    or (name == 'tensor.empty' and inner.operands)
                    or (name == 'tensor.from_elements' and
                        (len(inner.operands) != 1 or inner.operands[0].type != type_.element_type))):
                return False
            continue
        if name == 'scf.execute_region':
            if not private_atomics or not fresh_seed or not _ltd_nested_snapshot(inner):
                return False
            continue
        if name == 'linalg.generic':
            if not private_atomics or not fresh_seed or not _ltd_atomic_snapshot(inner):
                return False
            continue
        if name == 'scf.yield' and inner.block.owner.operation.name == 'scf.execute_region':
            if not private_atomics or not fresh_seed or not _ltd_nested_snapshot(inner.block.owner.operation):
                return False
            continue
        if name == 'scf.for':
            # Guard byte loops have no carried state. Keep their conditions,
            # effects and assertions; this recognizes values, not event motion.
            if (not private_atomics or not fresh_seed or len(inner.operands) != 3 or inner.results
                    or any(not isinstance(v.type, ir.IndexType) for v in inner.operands)
                    or len(inner.regions) != 1 or len(inner.regions[0].blocks) != 1):
                return False
            body = inner.regions[0].blocks[0]
            terminal = tuple(body.operations)[-1].operation
            if (len(body.arguments) != 1 or not isinstance(body.arguments[0].type, ir.IndexType)
                    or terminal.name != 'scf.yield' or terminal.operands):
                return False
            continue
        if name == 'scf.if':
            if (not fresh_seed or len(inner.operands) != 1 or str(inner.operands[0].type) != 'i1'
                    or len(inner.regions) != 2 or any(not _ltd_numeric(v.type) for v in inner.results)):
                return False
            for region in inner.regions:
                if len(region.blocks) != 1 or region.blocks[0].arguments:
                    return False
                terminal = tuple(region.blocks[0].operations)[-1].operation
                if (terminal.name != 'scf.yield'
                        or tuple(v.type for v in terminal.operands) != tuple(v.type for v in inner.results)):
                    return False
            continue
        if name.split('.')[0] not in ('arith', 'math') and name not in (
                'linalg.yield', 'linalg.index', 'scf.yield', 'tensor.yield', 'memref.atomic_yield'):
            return False
        if inner.regions or any(not _ltd_numeric(v.type) for v in (*inner.operands, *inner.results)):
            return False
    return True


def _ltd_nested_snapshot(op):
    if op.name != 'scf.execute_region' or len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    operations = tuple(op.regions[0].blocks[0].operations)
    return (len(operations) == 2 and _ltd_atomic_snapshot(operations[0].operation)
            and _ltd_scalar_body(operations[0].operation, fresh_seed=True, private_atomics=True))


def _ltd_snapshot_tensor_value(value):
    # Scalar extraction may consume a checked computed tensor. Captures of
    # the enclosing snapshot still pass _ltd_copy_safe's complete dependency
    # closure; this predicate grants no ownership to the tensor's storage.
    if not isinstance(value, ir.OpResult) or not isinstance(value.type, ir.RankedTensorType):
        return False
    op = value.owner.operation
    if not _ltd_numeric(ir.RankedTensorType(value.type).element_type):
        return False
    if op.name == 'scf.execute_region':
        return _ltd_nested_snapshot(op)
    return (op.name == 'linalg.generic' and 'library_call' not in op.attributes
            and len(op.results) == 1 and op.operands and _ltd_fresh_seed(op.operands[-1])
            and _ltd_scalar_body(op, fresh_seed=True, private_atomics=_ltd_atomic_snapshot(op)))


def _ltd_carried_edges(loop):
    if loop.name == 'scf.for' and len(loop.regions) == 1 and len(loop.regions[0].blocks) == 1:
        block = loop.regions[0].blocks[0]
        terminal = tuple(block.operations)[-1].operation
        if terminal.name == 'scf.yield':
            return block, tuple(zip(tuple(loop.operands)[3:], tuple(block.arguments)[1:], terminal.operands))
    if loop.name == 'scf.while' and len(loop.regions) == 2 and all(len(r.blocks) == 1 for r in loop.regions):
        before, after = (r.blocks[0] for r in loop.regions)
        condition = tuple(before.operations)[-1].operation
        terminal = tuple(after.operations)[-1].operation
        # Forward the complete before-state unchanged, including position and
        # type. A computed/permuted condition state cannot inherit seed facts.
        if (condition.name == 'scf.condition' and terminal.name == 'scf.yield'
                and tuple(condition.operands)[1:] == tuple(before.arguments)
                and len(loop.operands) == len(after.arguments) == len(terminal.operands)
                and tuple(v.type for v in loop.operands) == tuple(v.type for v in after.arguments)):
            return after, tuple(zip(loop.operands, after.arguments, terminal.operands))
    return None, ()


def _ltd_complete_generic(ctx, op, block, previous, incoming, inside):
    if (op.name != 'linalg.generic' or 'library_call' in op.attributes or op.block != block or len(op.results) != 1
            or not op.operands or previous.type != op.results[0].type or previous.type != op.operands[-1].type
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return False
    shape = tuple(ir.RankedTensorType(previous.type).shape)
    segments = tuple(ir.DenseI32ArrayAttr(op.attributes['operandSegmentSizes']))
    if segments != (len(op.operands) - 1, 1):
        return False
    maps = tuple(ir.ArrayAttr(op.attributes['indexing_maps']))
    with ctx:
        identity = ir.AffineMap.get_identity(len(shape))
    if (len(maps) != len(op.operands) or ir.AffineMapAttr(maps[-1]).value != identity
            or len(ir.ArrayAttr(op.attributes['iterator_types'])) != len(shape)
            or any(str(it) != '#linalg.iterator_type<parallel>'
                   for it in ir.ArrayAttr(op.attributes['iterator_types']))):
        return False
    scalar = op.regions[0].blocks[0]
    return (len(scalar.arguments) == len(op.operands) and not list(scalar.arguments[-1].uses)
            and _ltd_scalar_body(op, fresh_seed=_ltd_fresh_seed(incoming))
            and _ltd_copy_safe(op.results[0], incoming, inside, set()))


def _ltd_captured_values(op):
    nested = tuple(_ltd_walk(op))
    local = {op, *nested}
    return {v for child in nested for v in child.operands
            if (v.owner.operation if isinstance(v, ir.OpResult) else v.owner.owner.operation) not in local}


def _ltd_atomic_snapshot(op):
    # An exactly-once region remains at its original position. Only its already
    # computed tensor value may feed a destination copy/reuse; the event itself
    # is never cloned, retargeted or moved. An independent output seed and
    # private atomic targets exclude aliases with the loop's fresh tensor seed.
    parent = op.block.owner.operation
    if (parent.name != 'scf.execute_region' or 'no_inline' not in parent.attributes
            or len(parent.regions) != 1 or len(parent.regions[0].blocks) != 1
            or parent.regions[0].blocks[0].arguments or len(parent.results) != 1
            or op.name != 'linalg.generic' or 'library_call' in op.attributes or len(op.results) != 1
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1 or not op.operands
            or not _ltd_fresh_seed(op.operands[-1])):
        return False
    operations = tuple(parent.regions[0].blocks[0].operations)
    if (len(operations) != 2 or operations[0].operation != op or operations[-1].operation.name != 'scf.yield'
            or tuple(operations[-1].operands) != tuple(op.results)):
        return False
    scalar = op.regions[0].blocks[0]
    segments = tuple(ir.DenseI32ArrayAttr(op.attributes['operandSegmentSizes']))
    return (segments == (len(op.operands) - 1, 1) and not list(scalar.arguments[-1].uses)
            and any(inner.name in ('memref.atomic_rmw', 'memref.generic_atomic_rmw') for inner in _ltd_walk(op)))


def _ltd_copy_safe(value, incoming, inside, seen):
    if value in seen:
        return True
    seen.add(value)
    if isinstance(value, ir.BlockArgument):
        return True  # Explicit SSA inputs, not a physical no-alias premise.
    op = value.owner.operation
    if op not in inside:
        return True
    if op.name in ('memref.load', 'memref.dim'):
        return (len(op.results) == 1 and _ltd_numeric(value.type) and _ltd_fresh_seed(incoming)
                and _ltd_separate_read_root(op.operands[0])
                and all(_ltd_copy_safe(v, incoming, inside, seen) for v in tuple(op.operands)[1:]))
    if op.name == 'bufferization.to_tensor':
        return (len(op.operands) == 1 and len(op.results) == 1 and 'restrict' in op.attributes
                and _ltd_fresh_seed(incoming) and _ltd_separate_read_root(op.operands[0]))
    if op.name == 'scf.for':
        # Source-exact scalar contraction may introduce nested tensor-update
        # loops before this seam. Keep their complete computation, then copy
        # its finished value into the carried destination. Check dead effects
        # too: following only the yielded SSA dependency is insufficient.
        if (len(op.regions) != 1 or len(op.regions[0].blocks) != 1
                or len(op.operands) != len(op.results) + 3
                or any(not isinstance(v.type, ir.IndexType) for v in tuple(op.operands)[:3])):
            return False
        body = op.regions[0].blocks[0]
        operations = tuple(body.operations)
        if (not operations or operations[-1].operation.name != 'scf.yield'
                or tuple(v.type for v in operations[-1].operands) != tuple(v.type for v in op.results)
                or len(body.arguments) != len(op.operands) - 2
                or not isinstance(body.arguments[0].type, ir.IndexType)
                or tuple(v.type for v in tuple(body.arguments)[1:])
                != tuple(v.type for v in tuple(op.operands)[3:])
                or any(not _ltd_numeric(ir.RankedTensorType(v.type).element_type
                        if isinstance(v.type, ir.RankedTensorType) else v.type)
                       for v in (*op.operands, *op.results))):
            return False
        for child in _ltd_walk(op):
            if child.name in ('scf.yield', 'linalg.yield', 'tensor.yield'):
                if child.results or child.regions:
                    return False
                continue
            if child.name == 'linalg.index' and not child.operands and not child.regions:
                continue
            if not child.results or any(not _ltd_copy_safe(v, incoming, inside, seen) for v in child.results):
                return False
        return all(_ltd_copy_safe(v, incoming, inside, seen) for v in (*op.operands, *_ltd_captured_values(op)))
    if op.name == 'scf.if':
        if (len(op.operands) != 1 or str(op.operands[0].type) != 'i1' or len(op.regions) != 2
                or not _ltd_copy_safe(op.operands[0], incoming, inside, seen)):
            return False
        index = tuple(op.results).index(value)
        for region in op.regions:
            if len(region.blocks) != 1 or region.blocks[0].arguments:
                return False
            terminal = tuple(region.blocks[0].operations)[-1].operation
            if (terminal.name != 'scf.yield'
                    or tuple(v.type for v in terminal.operands) != tuple(v.type for v in op.results)
                    or not _ltd_copy_safe(terminal.operands[index], incoming, inside, seen)):
                return False
        return True
    if op.name == 'scf.execute_region':
        if (not _ltd_fresh_seed(incoming) or len(op.regions) != 1 or len(op.regions[0].blocks) != 1
                or len(op.results) != 1):
            return False
        operations = tuple(op.regions[0].blocks[0].operations)
        return (len(operations) == 2 and _ltd_atomic_snapshot(operations[0].operation)
                and _ltd_copy_safe(operations[0].results[0], incoming, inside, seen))
    def numeric(type_):
        return _ltd_numeric(ir.RankedTensorType(type_).element_type
                            if isinstance(type_, ir.RankedTensorType) else type_)
    if any(not numeric(v.type) for v in (*op.operands, *op.results)):
        return False
    tensor_values = ('tensor.empty', 'tensor.cast', 'tensor.extract', 'tensor.extract_slice', 'tensor.insert',
                     'tensor.insert_slice', 'tensor.collapse_shape', 'tensor.expand_shape',
                     'tensor.reshape', 'tensor.splat')
    if op.name.split('.')[0] in ('arith', 'math') or op.name in tensor_values:
        if op.regions:
            return False
    elif op.name.startswith('linalg.') or op.name == 'tensor.pad':
        if 'library_call' in op.attributes or not _ltd_scalar_body(op, fresh_seed=_ltd_fresh_seed(incoming),
                                                                 private_atomics=_ltd_atomic_snapshot(op)):
            return False
        for captured in _ltd_captured_values(op):
            if str(captured.type).startswith('memref'):
                if not _ltd_fresh_seed(incoming) or not _ltd_separate_read_root(captured):
                    return False
            elif not _ltd_copy_safe(captured, incoming, inside, seen):
                return False
    else:
        return False
    return all(_ltd_copy_safe(v, incoming, inside, seen) for v in op.operands)


def _ltd_late_destination(ctx, previous, producer):
    # Reuse at the producer would clobber an old value still needed by another
    # computation. An identity value copy at the selected edge keeps those
    # reads before the update. Nonidentity input maps similarly need a complete
    # computed tensor before overwriting any carried element.
    block = producer.block
    order = {op.operation: i for i, op in enumerate(block.operations)}
    scope = previous.owner.owner.operation
    for use in previous.uses:
        current = use.owner.operation
        # A loop block argument has no uses outside its owning loop. Stop
        # there: native bindings cannot ask a detached/module op for a block.
        while current is not None and current != scope:
            if current.block == block:
                if order[current] > order[producer]:
                    return True
                break
            current = current.parent
    if producer.name == 'linalg.generic':
        maps = tuple(ir.ArrayAttr(producer.attributes['indexing_maps']))
        with ctx:
            identity = ir.AffineMap.get_identity(len(ir.RankedTensorType(previous.type).shape))
        if any(v == previous and ir.AffineMapAttr(maps[i]).value != identity
               for i, v in enumerate(tuple(producer.operands)[:-1])):
            return True
    return False


def _ltd_destination_plan(ctx, value, block, terminal, position, previous, incoming, inside, claimed):
    if (value.type != previous.type or not isinstance(value.type, ir.RankedTensorType)
            or any(n <= 0 for n in ir.RankedTensorType(value.type).shape)
            or not _ltd_numeric(ir.RankedTensorType(value.type).element_type)):
        return None
    if value == previous:
        return []
    owner = value.owner.operation if isinstance(value, ir.OpResult) else value.owner.owner.operation
    if owner not in inside:
        # An immutable value captured outside the loop needs an explicit value
        # copy on this path. It is not a grant to overwrite its physical owner.
        return [('copy', terminal, position, value, previous)]
    if not isinstance(value, ir.OpResult):
        return [('copy', terminal, position, value, previous)]
    if owner.name == 'linalg.generic':
        if owner not in claimed and _ltd_complete_generic(ctx, owner, block, previous, incoming, inside):
            if _ltd_late_destination(ctx, previous, owner):
                return [('copy', terminal, position, value, previous)]
            return [] if owner.operands[-1] == previous else [('generic', owner, previous)]
    if owner.name == 'scf.if' and owner.block == block and _ltd_late_destination(ctx, previous, owner):
        return ([('copy', terminal, position, value, previous)]
                if _ltd_copy_safe(value, incoming, inside, set()) else None)
    if (owner.name != 'scf.if' or owner.block != block or len(owner.operands) != 1
            or str(owner.operands[0].type) != 'i1' or len(owner.regions) != 2):
        # Keep non-overwrite producers (including reductions/init reads)
        # intact and materialize their complete value. Close the producer DAG
        # against hidden captures/effects before using a carried destination.
        return ([('copy', terminal, position, value, previous)]
                if _ltd_copy_safe(value, incoming, inside, set()) else None)
    result_index = tuple(owner.results).index(value)
    plan = []
    for region in owner.regions:
        if len(region.blocks) != 1 or region.blocks[0].arguments:
            return None
        branch = region.blocks[0]
        yielded = tuple(branch.operations)[-1].operation
        if (yielded.name != 'scf.yield' or tuple(v.type for v in yielded.operands)
                != tuple(v.type for v in owner.results)):
            return None
        part = _ltd_destination_plan(ctx, yielded.operands[result_index], branch, yielded, result_index,
                                     previous, incoming, inside, claimed)
        if part is None:
            return None  # Plan every path before changing any path.
        plan.extend(part)
    return plan


def _ltd_identity_copy(ctx, anchor, source, destination, *, detached=False):
    type_ = ir.RankedTensorType(source.type)
    with ctx, anchor.location:
        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(type_.shape)))
        attributes = {
            'indexing_maps': ir.ArrayAttr.get([identity, identity]),
            'iterator_types': ir.ArrayAttr.get(
                [ir.Attribute.parse('#linalg.iterator_type<parallel>')] * len(type_.shape)),
            'operandSegmentSizes': ir.DenseI32ArrayAttr.get([1, 1]),
        }
        for name in ('prov.source_ids', 'prov.original_ops', 'prov.region_id'):
            if name in anchor.attributes:
                attributes[name] = anchor.attributes[name]
        attributes['prov.family'] = ir.StringAttr.get('loop-state')
        attributes['prov.op'] = ir.StringAttr.get('copy-selected-state')
        copy = ir.Operation.create('linalg.generic', operands=[source, destination],
                                   results=[source.type], attributes=attributes, regions=1,
                                   ip=False if detached else ir.InsertionPoint(anchor))
        body = ir.Block.create_at_start(copy.regions[0], [type_.element_type, type_.element_type])
        with ir.InsertionPoint(body):
            ir.Operation.create('linalg.yield', operands=[body.arguments[0]])
        return copy


def _ltd_copy_state(ctx, terminal, position, source, destination):
    terminal.operands[position] = _ltd_identity_copy(ctx, terminal, source, destination).results[0]


def _ltd_private_seed(ctx, loop, incoming):
    # A restricted import from actual private storage supplies values, not a
    # fresh carried destination. Offer an explicit independent copy, then run
    # the complete producer/capture proof against it before attaching anything.
    # Entry/unknown/escaping roots gain no ownership premise from this helper.
    if (not isinstance(incoming, ir.OpResult) or not isinstance(incoming.type, ir.RankedTensorType)
            or any(n <= 0 for n in ir.RankedTensorType(incoming.type).shape)):
        return None
    owner = incoming.owner.operation
    if (owner.name != 'bufferization.to_tensor' or len(owner.operands) != 1
            or 'restrict' not in owner.attributes or not _ltd_private_root(owner.operands[0])):
        return None
    with ctx, loop.location:
        empty = ir.Operation.create('tensor.empty', results=[incoming.type], ip=False)
        copy = _ltd_identity_copy(ctx, loop, incoming, empty.results[0], detached=True)
        copy.attributes['prov.op'] = ir.StringAttr.get('seed-private-carried-value')
    return empty, copy


def _ltd_reuse(ctx, module):
    changed, claimed = 0, set()
    for loop in tuple(_ltd_walk(module.operation)):
        block, edges = _ltd_carried_edges(loop)
        if not edges:
            continue
        terminal = tuple(block.operations)[-1].operation
        inside = {loop, *_ltd_walk(loop)}
        for position, (incoming, previous, result) in enumerate(edges):
            plan = _ltd_destination_plan(ctx, result, block, terminal, position, previous, incoming, inside, claimed)
            seed = None
            if plan is None and not _ltd_fresh_seed(incoming):
                seed = _ltd_private_seed(ctx, loop, incoming)
                if seed is not None:
                    plan = _ltd_destination_plan(ctx, result, block, terminal, position, previous,
                                                 seed[1].results[0], inside, claimed)
            if plan is None:
                if seed is not None:
                    seed[1].erase()
                    seed[0].erase()
                continue
            if seed is not None:
                point = ir.InsertionPoint(loop)
                point.insert(seed[0])
                point.insert(seed[1])
                loop.operands[position + (3 if loop.name == 'scf.for' else 0)] = seed[1].results[0]
                changed += 1
            for action in plan:
                if action[0] == 'generic':
                    op, destination = action[1:]
                    op.operands[len(op.operands) - 1] = destination
                    claimed.add(op)
                else:
                    _ltd_copy_state(ctx, *action[1:])
                changed += 1
    module.operation.verify()
    print('OK loop_tensor_destinations', changed, flush=True)
    return changed


_LTD_ORIGINAL_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    marker = '__merlin_loop_tensor_destinations__'
    if pipeline.count(marker) != 1:
        raise ValueError('loop tensor destinations require one explicit pre-bufferization marker')
    head, _, tail = pipeline.partition(marker)
    head, tail = head.rstrip(','), tail.lstrip(',')
    seam = (tail.removeprefix('__merlin_integer_compare_exchange__,')
            .removeprefix('__merlin_atomic_word_views__,')
            .removeprefix('__merlin_integer_atomic_regions__,'))
    if not seam.startswith('one-shot-bufferize{') or 'one-shot-bufferize' in head:
        raise ValueError('loop tensor destinations must immediately precede one-shot bufferization')
    if 'buffer-hoisting' in head or 'convert-scf-to' in head:
        raise ValueError('loop tensor destinations require the original tensor loop structure')
    if head:
        _LTD_ORIGINAL_RUN_STAGES(ctx, module, head, 0, (), (), (), pre_generalize)
    _ltd_reuse(ctx, module)
    return _LTD_ORIGINAL_RUN_STAGES(ctx, module, tail, erase, mid, late, post_openmp, ())
"""


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_run_stages"
    ]
    if len(calls) != 1:
        raise ValueError("loop tensor destinations require one native stage runner call")
    lines = source.splitlines(keepends=True)
    at = calls[0].lineno - 1
    return "".join(lines[:at]) + PRELUDE + "\n" + "".join(lines[at:])


def edit_pipeline(passes):
    from .atomic_compare_exchange import MARKER as CAS_MARKER
    from .atomic_integer_regions import MARKER as REGION_MARKER
    from .atomic_word_views import MARKER as WORD_MARKER

    positions = [i for i, name in enumerate(passes) if name.startswith("one-shot-bufferize{")]
    if len(positions) != 1 or MARKER in passes:
        raise ValueError("loop tensor destinations require one unmodified bufferization seam")
    at = positions[0]
    atomic = [i for i, name in enumerate(passes) if name in (CAS_MARKER, REGION_MARKER, WORD_MARKER)]
    if atomic:
        if len(atomic) != 1 or atomic[0] != at - 1:
            raise ValueError("loop tensor destinations require one contiguous atomic bufferization seam")
        # Destination reuse must see the original atomic regions. Keep the
        # atomic runner and its bufferization/ownership suffix together even
        # when registered feature edits were applied before this one.
        at = atomic[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description=(
                    "Reuse completely overwritten parallel tensor loop destinations "
                    "before native alias analysis; default off."
                ),
            )
        )
    return FEATURE
