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

PRELUDE = r'''
def _ltd_walk(op):
    for region in op.regions:
        for block in region.blocks:
            for child in tuple(block.operations):
                yield child.operation
                yield from _ltd_walk(child.operation)


def _ltd_numeric(type_):
    return isinstance(type_, (ir.IntegerType, ir.IndexType)) or str(type_) in ('f16', 'bf16', 'f32', 'f64')


def _ltd_fresh_seed(value):
    if not isinstance(value, ir.OpResult):
        return False
    op = value.owner.operation
    if op.name == 'tensor.empty':
        return True
    if op.name == 'arith.constant':
        return 'value' in op.attributes and isinstance(op.attributes['value'], ir.DenseElementsAttr)
    if (op.name == 'linalg.generic' and 'library_call' not in op.attributes and len(op.results) == 1 and op.operands
            and all(_ltd_numeric(v.type) for v in tuple(op.operands)[:-1])
            and _ltd_fresh_seed(op.operands[-1]) and len(op.regions) == 1 and len(op.regions[0].blocks) == 1):
        segments = tuple(ir.DenseI32ArrayAttr(op.attributes['operandSegmentSizes']))
        scalar = op.regions[0].blocks[0]
        if segments != (len(op.operands) - 1, 1) or not scalar.arguments or list(scalar.arguments[-1].uses):
            return False
        with op.context:
            identity = ir.AffineMap.get_identity(len(ir.RankedTensorType(value.type).shape))
        maps = tuple(ir.ArrayAttr(op.attributes['indexing_maps']))
        return (ir.AffineMapAttr(maps[-1]).value == identity
                and all(str(it) == '#linalg.iterator_type<parallel>' for it in ir.ArrayAttr(op.attributes['iterator_types']))
                and _ltd_scalar_body(op, fresh_seed=False))
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


def _ltd_scalar_body(op, *, fresh_seed):
    # The read extension admits only readonly entry storage, disjoint from a
    # newly synthesized loop seed. It changes no capture or memory effect;
    # native bufferization still decides whether physical reuse is legal.
    for inner in _ltd_walk(op):
        name = inner.name
        if name == 'memref.load':
            if (not fresh_seed or not inner.operands or not _ltd_readonly_input(inner.operands[0])
                    or inner.regions or len(inner.results) != 1 or not _ltd_numeric(inner.results[0].type)
                    or any(not isinstance(v.type, ir.IndexType) for v in tuple(inner.operands)[1:])):
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
                if terminal.name != 'scf.yield' or tuple(v.type for v in terminal.operands) != tuple(v.type for v in inner.results):
                    return False
            continue
        if name.split('.')[0] not in ('arith', 'math') and name not in ('linalg.yield', 'linalg.index', 'scf.yield'):
            return False
        if inner.regions or any(not _ltd_numeric(v.type) for v in (*inner.operands, *inner.results)):
            return False
    return True


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


def _ltd_copy_safe(value, incoming, inside, seen):
    if value in seen:
        return True
    seen.add(value)
    if isinstance(value, ir.BlockArgument):
        return True  # Explicit SSA inputs, not a physical no-alias premise.
    op = value.owner.operation
    if op not in inside:
        return True
    if op.name == 'memref.load':
        return (len(op.results) == 1 and _ltd_numeric(value.type) and _ltd_fresh_seed(incoming)
                and _ltd_readonly_input(op.operands[0])
                and all(_ltd_copy_safe(v, incoming, inside, seen) for v in tuple(op.operands)[1:]))
    def numeric(type_):
        return _ltd_numeric(ir.RankedTensorType(type_).element_type) if isinstance(type_, ir.RankedTensorType) else _ltd_numeric(type_)
    if any(not numeric(v.type) for v in (*op.operands, *op.results)):
        return False
    tensor_values = ('tensor.empty', 'tensor.cast', 'tensor.extract', 'tensor.extract_slice', 'tensor.insert',
                     'tensor.insert_slice', 'tensor.collapse_shape', 'tensor.expand_shape', 'tensor.reshape', 'tensor.splat')
    if op.name.split('.')[0] in ('arith', 'math') or op.name in tensor_values:
        if op.regions:
            return False
    elif op.name.startswith('linalg.'):
        if 'library_call' in op.attributes or not _ltd_scalar_body(op, fresh_seed=_ltd_fresh_seed(incoming)):
            return False
        for captured in _ltd_captured_values(op):
            if str(captured.type).startswith('memref'):
                if not _ltd_fresh_seed(incoming) or not _ltd_readonly_input(captured):
                    return False
            elif not _ltd_copy_safe(captured, incoming, inside, seen):
                return False
    else:
        return False
    return all(_ltd_copy_safe(v, incoming, inside, seen) for v in op.operands)


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
            return [] if owner.operands[-1] == previous else [('generic', owner, previous)]
    if (owner.name != 'scf.if' or owner.block != block or len(owner.operands) != 1
            or str(owner.operands[0].type) != 'i1' or len(owner.regions) != 2):
        # Keep non-overwrite producers (including reductions/init reads)
        # intact and materialize their complete value. Close the producer DAG
        # against hidden captures/effects before using a carried destination.
        return [('copy', terminal, position, value, previous)] if _ltd_copy_safe(value, incoming, inside, set()) else None
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


def _ltd_copy_state(ctx, terminal, position, source, destination):
    type_ = ir.RankedTensorType(source.type)
    with ctx, terminal.location, ir.InsertionPoint(terminal):
        identity = ir.AffineMapAttr.get(ir.AffineMap.get_identity(len(type_.shape)))
        attributes = {
            'indexing_maps': ir.ArrayAttr.get([identity, identity]),
            'iterator_types': ir.ArrayAttr.get([ir.Attribute.parse('#linalg.iterator_type<parallel>')] * len(type_.shape)),
            'operandSegmentSizes': ir.DenseI32ArrayAttr.get([1, 1]),
        }
        for name in ('prov.source_ids', 'prov.original_ops', 'prov.region_id'):
            if name in terminal.attributes:
                attributes[name] = terminal.attributes[name]
        attributes['prov.family'] = ir.StringAttr.get('loop-state')
        attributes['prov.op'] = ir.StringAttr.get('copy-selected-state')
        copy = ir.Operation.create('linalg.generic', operands=[source, destination],
                                   results=[source.type], attributes=attributes, regions=1)
        body = ir.Block.create_at_start(copy.regions[0], [type_.element_type, type_.element_type])
        with ir.InsertionPoint(body):
            ir.Operation.create('linalg.yield', operands=[body.arguments[0]])
        terminal.operands[position] = copy.results[0]


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
            if plan is None:
                continue
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
    if not tail.startswith('one-shot-bufferize{') or 'one-shot-bufferize' in head:
        raise ValueError('loop tensor destinations must immediately precede one-shot bufferization')
    if 'buffer-hoisting' in head or 'convert-scf-to' in head:
        raise ValueError('loop tensor destinations require the original tensor loop structure')
    if head:
        _LTD_ORIGINAL_RUN_STAGES(ctx, module, head, 0, (), (), (), pre_generalize)
    _ltd_reuse(ctx, module)
    return _LTD_ORIGINAL_RUN_STAGES(ctx, module, tail, erase, mid, late, post_openmp, ())
'''


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [node for node in ast.parse(source).body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "_run_stages"]
    if len(calls) != 1:
        raise ValueError("loop tensor destinations require one native stage runner call")
    lines = source.splitlines(keepends=True)
    at = calls[0].lineno - 1
    return "".join(lines[:at]) + PRELUDE + "\n" + "".join(lines[at:])


def edit_pipeline(passes):
    positions = [i for i, name in enumerate(passes) if name.startswith("one-shot-bufferize{")]
    if len(positions) != 1 or MARKER in passes:
        raise ValueError("loop tensor destinations require one unmodified bufferization seam")
    at = positions[0]
    return [*passes[:at], MARKER, *passes[at:]]


def ensure_registered():
    from .impr_features import ImprFeature, known, register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE, action_class="PASS", edit_pipeline=edit_pipeline,
            description="Reuse completely overwritten parallel tensor loop destinations before native alias analysis; default off."))
    return FEATURE
