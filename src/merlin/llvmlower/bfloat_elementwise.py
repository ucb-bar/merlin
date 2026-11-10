"""Explicit scalar BF16 math legalization without changing tensor storage.

Reuse upstream arith-emulate-unsupported-floats and arith-expand for the
qualified min/max/div operations and exact casts. Scalar sqrt uses the same
extend/compute/truncate boundary; BF16 has eight significand bits, so f32
precision resolves its rounding midpoint. Transport stays a separate feature.
"""

import ast

FEATURE = "emulate_bfloat_elementwise"
MARKER = "__merlin_bfloat_elementwise__"
OPERATIONS = frozenset({"arith.maxnumf","arith.minnumf","arith.maximumf","arith.minimumf","arith.divf","math.sqrt"})

SUPPORT = r'''
def _bfe_walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _bfe_walk(child.operation)


def _bf_elementwise(module):
    from torch_mlir import passmanager as _bfe_passmanager
    with module.context:
        module.operation.verify()
        bf16,f32 = ir.BF16Type.get(),ir.F32Type.get()
        operations = tuple(_bfe_walk(module.operation))
        supported = {'arith.maxnumf','arith.minnumf','arith.maximumf','arith.minimumf','arith.divf','math.sqrt'}
        transport = {'arith.constant','arith.bitcast','arith.extf','arith.truncf','arith.select',
            'memref.load','memref.store','tensor.extract','tensor.insert','tensor.yield','linalg.yield','linalg.fill',
            'func.return','scf.if','scf.for','scf.while','scf.yield','scf.condition','cf.br','cf.cond_br'}
        plans = []
        for op in operations:
            values = tuple(op.operands)+tuple(op.results)
            if any(isinstance(value.type,(ir.RankedTensorType,ir.VectorType)) and value.type.element_type == bf16
                   for value in values) and op.name.split('.')[0] in ('arith','math','vector'):
                raise ValueError('BF16 math legalization requires scalar operations')
            if not any(value.type == bf16 for value in values):
                continue
            if op.name in transport:
                continue
            if op.name not in supported:
                raise ValueError('unqualified BF16 scalar operation: '+op.name)
            arity = 1 if op.name == 'math.sqrt' else 2
            if (len(op.operands) != arity or len(op.results) != 1 or op.regions
                    or any(value.type != bf16 for value in values)):
                raise ValueError('BF16 math requires the complete same-type scalar signature')
            if ('fastmath' in op.attributes and op.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>')
                    or 'roundingmode' in op.attributes):
                raise ValueError('BF16 math requires neutral flags and default nearest-even arithmetic')
            plans.append(op)
        # Validate the entire module before changing any source operation.
        for op in plans:
            if op.name != 'math.sqrt':
                continue
            with ir.InsertionPoint(op):
                widened = ir.Operation.create('arith.extf',results=[f32],operands=[op.operands[0]],loc=op.location)
                root = ir.Operation.create('math.sqrt',results=[f32],operands=[widened.results[0]],
                    attributes=dict(op.attributes),loc=op.location)
                narrowed = ir.Operation.create('arith.truncf',results=[bf16],operands=[root.results[0]],loc=op.location)
            op.results[0].replace_all_uses_with(narrowed.results[0])
            op.erase()
        if plans:
            _bfe_passmanager.PassManager.parse(
                'builtin.module(arith-emulate-unsupported-floats{source-types=bf16 target-type=f32})').run(module.operation)
            module.operation.verify()
        return len(plans)
'''

PRELUDE = SUPPORT + r'''
_BFE_PREVIOUS_RUN_STAGES = _run_stages
_BFE_CONSUMED = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    marker = '__merlin_bfloat_elementwise__'
    if marker not in pipeline:
        return _BFE_PREVIOUS_RUN_STAGES(ctx,module,pipeline,erase,mid,late,post_openmp,pre_generalize)
    if pipeline.count(marker) != 1 or not pipeline.startswith(marker+','):
        raise ValueError('BF16 elementwise legalization requires one initial source marker')
    _bf_elementwise(module)
    global _BFE_CONSUMED
    _BFE_CONSUMED = True
    return _BFE_PREVIOUS_RUN_STAGES(ctx,module,pipeline[len(marker)+1:],erase,mid,late,post_openmp,pre_generalize)
'''


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [node for node in ast.parse(source).body if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call)
        and isinstance(node.value.func,ast.Name) and node.value.func.id == "_run_stages"]
    if len(calls) != 1:
        raise ValueError("BF16 elementwise legalization requires one native stage runner call")
    lines = source.splitlines(keepends=True)
    at,after = calls[0].lineno-1,calls[0].end_lineno
    return ("".join(lines[:at])+PRELUDE+"\n"+"".join(lines[at:after])
        +"\nif not _BFE_CONSUMED:\n    raise ValueError('BF16 elementwise marker was not consumed')\n"+"".join(lines[after:]))


def edit_pipeline(passes):
    if MARKER in passes or any(entry.startswith("arith-emulate-unsupported-floats") for entry in passes):
        raise ValueError("BF16 elementwise legalization requires an unmodified source pipeline")
    return [MARKER,*passes]


def ensure_registered():
    from .impr_features import ImprFeature,known,register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE,action_class="PASS",edit_pipeline=edit_pipeline,
            description="Explicit scalar BF16 min/max/div/sqrt via f32 arithmetic and RNE casts; storage ABI unchanged."))
    return FEATURE
