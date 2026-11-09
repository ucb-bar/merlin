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


def _ltd_reuse(ctx, module):
    changed = 0
    for loop in tuple(_ltd_walk(module.operation)):
        if loop.name != 'scf.for' or len(loop.regions) != 1 or len(loop.regions[0].blocks) != 1:
            continue
        block = loop.regions[0].blocks[0]
        terminal = tuple(block.operations)[-1].operation
        if terminal.name != 'scf.yield':
            continue
        for previous, result in zip(tuple(block.arguments)[1:], terminal.operands):
            if not isinstance(result, ir.OpResult) or not isinstance(result.type, ir.RankedTensorType):
                continue
            op = result.owner.operation
            if (op.name != 'linalg.generic' or 'library_call' in op.attributes or op.block != block or len(op.results) != 1
                    or not op.operands or previous.type != result.type or previous.type != op.operands[-1].type
                    or previous == op.operands[-1] or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
                continue
            shape = tuple(ir.RankedTensorType(result.type).shape)
            if any(n <= 0 for n in shape):
                continue
            segments = tuple(ir.DenseI32ArrayAttr(op.attributes['operandSegmentSizes']))
            if segments != (len(op.operands) - 1, 1):
                continue
            maps = tuple(ir.ArrayAttr(op.attributes['indexing_maps']))
            with ctx:
                identity = ir.AffineMap.get_identity(len(shape))
            if (len(maps) != len(op.operands) or ir.AffineMapAttr(maps[-1]).value != identity
                    or len(ir.ArrayAttr(op.attributes['iterator_types'])) != len(shape)
                    or any(str(it) != '#linalg.iterator_type<parallel>'
                           for it in ir.ArrayAttr(op.attributes['iterator_types']))):
                continue
            scalar = op.regions[0].blocks[0]
            if len(scalar.arguments) != len(op.operands) or list(scalar.arguments[-1].uses):
                continue
            # Complete typed scalar region: no captured buffer, call or opaque
            # operation is hidden by the destination's unused init argument.
            legal = True
            for inner in _ltd_walk(op):
                if inner.name.split('.')[0] not in ('arith', 'math') and inner.name not in ('linalg.yield', 'linalg.index'):
                    legal = False
                    break
                if inner.regions or any(not (isinstance(v.type, (ir.IntegerType, ir.IndexType))
                        or str(v.type) in ('f16', 'bf16', 'f32', 'f64')) for v in (*inner.operands, *inner.results)):
                    legal = False
                    break
            if not legal:
                continue
            op.operands[len(op.operands) - 1] = previous
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
