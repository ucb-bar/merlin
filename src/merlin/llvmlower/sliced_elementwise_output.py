"""Restrict a pure elementwise producer to the only output slice its caller reads.

This runs on bufferized MLIR after deallocation placement. It handles one narrow,
general pattern: a full-size fresh output of an identity-mapped, parallel,
side-effect-free ``linalg.generic`` is used only by a static subview, collapse,
and copy. The generic can compute that slice directly into a smaller fresh
buffer. Every refusal leaves the original computation intact.
"""

PRELUDE = r"""
def _slice_unused_elementwise_outputs(ctx, module):
    changed = 0
    for func in module.body.operations:
        if func.operation.name != 'func.func' or not func.regions[0].blocks:
            continue
        for block in func.regions[0].blocks:
            for alloc in list(block.operations):
                if alloc.operation.name != 'memref.alloc' or len(alloc.results) != 1:
                    continue
                value = alloc.results[0]
                users = [use.owner for use in value.uses]
                if len(users) != 3:
                    continue
                generics = [op for op in users if op.operation.name == 'linalg.generic']
                subviews = [op for op in users if op.operation.name == 'memref.subview']
                deallocs = [op for op in users if op.operation.name == 'memref.dealloc']
                if len(generics) != 1 or len(subviews) != 1 or len(deallocs) != 1:
                    continue
                generic, subview, dealloc = generics[0], subviews[0], deallocs[0]
                if len(generic.operands) != 2 or generic.operands[1] != value or generic.results:
                    continue
                source = generic.operands[0]
                old_ty, src_ty = ir.MemRefType(value.type), ir.MemRefType(source.type)
                if list(old_ty.shape) != list(src_ty.shape):
                    continue
                rank = len(old_ty.shape)
                maps = generic.attributes['indexing_maps']
                if len(maps) != 2 or any(
                    m.value != ir.AffineMap.get_identity(rank, context=ctx) for m in maps
                ):
                    continue
                if any(
                    str(it) != '#linalg.iterator_type<parallel>'
                    for it in generic.attributes['iterator_types']
                ):
                    continue
                if any(
                    child.operation.name != 'linalg.yield'
                    and not child.operation.name.startswith(('arith.', 'math.'))
                    for region in generic.regions
                    for body in region.blocks
                    for child in body.operations
                ):
                    continue
                if len(subview.operands) != 1 or subview.operands[0] != value or len(subview.results) != 1:
                    continue
                offsets = list(subview.attributes['static_offsets'])
                sizes = list(subview.attributes['static_sizes'])
                steps = list(subview.attributes['static_strides'])
                if (
                    len(offsets) != rank or len(sizes) != rank or len(steps) != rank
                    or len(ir.MemRefType(subview.results[0].type).shape) != rank
                    or any(x < 0 for x in offsets)
                    or any(x <= 0 for x in sizes + steps)
                ):
                    continue
                if sizes == list(old_ty.shape) or any(
                    o + (s - 1) * step >= dim
                    for o, s, step, dim in zip(offsets, sizes, steps, old_ty.shape)
                ):
                    continue
                if any(step != 1 for step in steps):
                    continue
                sub_users = [use.owner for use in subview.results[0].uses]
                if len(sub_users) != 1 or sub_users[0].operation.name != 'memref.collapse_shape':
                    continue
                collapse = sub_users[0]
                collapse_users = [use.owner for use in collapse.results[0].uses]
                if (
                    len(collapse_users) != 1
                    or collapse_users[0].operation.name != 'memref.copy'
                    or collapse_users[0].operands[0] != collapse.results[0]
                ):
                    continue
                strides, base_off = src_ty.get_strides_and_offset()
                if base_off < 0 or any(stride < 0 for stride in strides):
                    continue
                result_strides = [stride * step for stride, step in zip(strides, steps)]
                result_off = base_off + sum(o * stride for o, stride in zip(offsets, strides))
                with ctx, alloc.location, ir.InsertionPoint(generic):
                    slice_type = ir.MemRefType.get(
                        sizes, src_ty.element_type,
                        layout=ir.StridedLayoutAttr.get(result_off, result_strides),
                    )
                    input_slice = ir.Operation.create(
                        'memref.subview', results=[slice_type], operands=[source],
                        attributes={
                            'static_offsets': ir.DenseI64ArrayAttr.get(offsets),
                            'static_sizes': ir.DenseI64ArrayAttr.get(sizes),
                            'static_strides': ir.DenseI64ArrayAttr.get(steps),
                            'operandSegmentSizes': ir.DenseI32ArrayAttr.get([1, 0, 0, 0]),
                        },
                    )
                    small_type = ir.MemRefType.get(sizes, old_ty.element_type)
                    attrs = {'alignment': alloc.attributes['alignment']} if 'alignment' in alloc.attributes else {}
                    small = ir.Operation.create('memref.alloc', results=[small_type], attributes=attrs)
                generic.operands[0] = input_slice.results[0]
                generic.operands[1] = small.results[0]
                with ctx, collapse.location, ir.InsertionPoint(collapse):
                    collapsed_ty = ir.MemRefType.get(
                        list(ir.MemRefType(collapse.results[0].type).shape), old_ty.element_type,
                    )
                    new_collapse = ir.Operation.create(
                        'memref.collapse_shape', results=[collapsed_ty],
                        operands=[small.results[0]],
                        attributes={'reassociation': collapse.attributes['reassociation']},
                    )
                collapse.results[0].replace_all_uses_with(new_collapse.results[0])
                dealloc.operands[0] = small.results[0]
                collapse.operation.erase()
                subview.operation.erase()
                alloc.operation.erase()
                changed += 1
    module.operation.verify()
    print('OK sliced_elementwise_output', changed, flush=True)
"""

RUNNER = r"""
_SLICE_ORIG_RUN_STAGES = _run_stages


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(),
                pre_generalize=()):
    passes = [p for p in pipeline.split(',') if p]
    marker = '__merlin_slice_unused_elementwise_output__'
    if marker not in passes:
        return _SLICE_ORIG_RUN_STAGES(ctx, module, pipeline, erase, mid, late,
                                      post_openmp, pre_generalize)
    if passes.count(marker) != 1:
        raise RuntimeError('output-slice marker must appear exactly once')
    at = passes.index(marker)
    head, tail = passes[:at], passes[at + 1:]
    if not any('optimize-allocation-liveness' in p for p in head):
        raise RuntimeError('output-slice rewrite requires checked deallocation sinking')
    if any('buffer-loop-hoisting' in p or 'convert-scf-to-openmp' in p for p in head):
        raise RuntimeError('output-slice rewrite moved past a stage anchor')
    _SLICE_ORIG_RUN_STAGES(ctx, module, ','.join(head), 0, (), (), (), pre_generalize)
    _slice_unused_elementwise_outputs(ctx, module)
    bad, n_alloc, n_sunk = _dealloc_placement_violations(module)
    print('OK sliced_dealloc_placement', n_alloc, 'allocations', n_sunk, 'sunk', len(bad), 'violations')
    if bad:
        raise RuntimeError('output-slice rewrite invalidated deallocation placement: '
                           + '; '.join(bad[:20]))
    if tail:
        _SLICE_ORIG_RUN_STAGES(ctx, module, ','.join(tail), erase, mid, late,
                               post_openmp, ())
"""
