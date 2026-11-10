"""Preserve BF16 control-flow words without host floating round trips.

Arithmetic casts use upstream arith-expand. This separate native legalization
represents remaining scalar transport as i16 after complete LLVM conversion.
Public descriptors and their two-byte storage layout remain unchanged.
"""

import ast

FEATURE = "preserve_bfloat_transport"
MARKER = "__merlin_bfloat_transport__"

SUPPORT = r'''
def _bf_walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _bf_walk(child.operation)


def _bf_transport(module):
    with module.context:
        return _bf_transport_in_context(module)


def _bf_transport_in_context(module):
    module.operation.verify()
    operations = tuple(_bf_walk(module.operation))
    bf16, i16 = ir.BF16Type.get(), ir.IntegerType.get_signless(16)
    values, constants, bitcasts = [], [], []
    allowed = {'llvm.load', 'llvm.store', 'llvm.bitcast', 'llvm.mlir.constant',
               'llvm.mlir.undef', 'llvm.mlir.poison', 'llvm.select',
               'llvm.br', 'llvm.cond_br', 'llvm.freeze', 'llvm.switch'}
    # LLVM opaque pointers let ordinary loads/stores carry the exact same
    # two-byte words as integers. Leave GEP element sizes and all effects alone.
    # Reject arithmetic/calls rather than assert a floating-point equivalence.
    for op in operations:
        if op.name.split('.')[0] not in ('llvm', 'builtin'):
            raise ValueError('BF16 transport requires complete LLVM conversion')
        if op.name == 'llvm.func':
            signature = ir.TypeAttr(op.attributes['function_type']).value
            if any(type_ == bf16 for type_ in (*signature.inputs, signature.return_type)):
                raise ValueError('BF16 by-value function ABI is not qualified')
        if any(isinstance(value.type, ir.VectorType) and value.type.element_type == bf16
               for value in (*op.operands, *op.results)):
            raise ValueError('BF16 vector transport is not qualified')
        for region in op.regions:
            for block in region.blocks:
                args = [arg for arg in block.arguments if arg.type == bf16]
                if args:
                    if op.name != 'llvm.func' or block == region.blocks[0]:
                        raise ValueError('BF16 by-value function ABI is not qualified')
                    values.extend(args)
        floating = [value for value in (*op.operands, *op.results) if value.type == bf16]
        if not floating:
            continue
        if op.name not in allowed:
            raise ValueError('BF16 transport cannot reinterpret arithmetic or scalar calls: ' + op.name)
        if 'fastmathFlags' in op.attributes and op.attributes['fastmathFlags'] != ir.Attribute.parse('#llvm.fastmath<none>'):
            raise ValueError('BF16 transport requires neutral floating-point flags')
        values.extend(value for value in op.results if value.type == bf16)
        if op.name == 'llvm.mlir.constant':
            # Obtain exact APFloat bits through the registered upstream fold,
            # never through a Python float (which could quiet a signaling NaN).
            from torch_mlir import passmanager as _bf_pm
            attribute = op.attributes['value']
            probe = ir.Module.parse('module {func.func @word()->i16 {%v=arith.constant ' + str(attribute)
                + ' %b=arith.bitcast %v:bf16 to i16 return %b:i16}}')
            _bf_pm.PassManager.parse('builtin.module(canonicalize)').run(probe.operation)
            folded = [item for item in _bf_walk(probe.operation) if item.name == 'arith.constant']
            if len(folded) != 1 or folded[0].results[0].type != i16:
                raise ValueError('BF16 constant did not fold to one exact integer word')
            constants.append((op, ir.IntegerAttr(folded[0].attributes['value'])))
        if op.name == 'llvm.bitcast':
            bitcasts.append(op)
    for value in values:
        value.set_type(i16)
    for op, bits in constants:
        op.attributes['value'] = bits
    for op in bitcasts:
        if op.operands[0].type == op.results[0].type:
            op.results[0].replace_all_uses_with(op.operands[0])
            op.erase()
    module.operation.verify()
    return len(values)
'''

PRELUDE = SUPPORT + r'''
_BF_PREVIOUS_RUN_STAGES = _run_stages
_BF_CONSUMED = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    marker = '__merlin_bfloat_transport__'
    if marker not in pipeline:
        return _BF_PREVIOUS_RUN_STAGES(ctx, module, pipeline, erase, mid, late, post_openmp, pre_generalize)
    if pipeline.count(marker) != 1:
        raise ValueError('BF16 transport requires one complete LLVM seam')
    head, _, tail = pipeline.partition(marker)
    head, tail = head.rstrip(','), tail.lstrip(',')
    if tail or not all(name in head for name in ('convert-arith-to-llvm', 'convert-cf-to-llvm')):
        raise ValueError('BF16 transport must follow the complete LLVM pipeline')
    result = _BF_PREVIOUS_RUN_STAGES(ctx, module, head, erase, mid, late, post_openmp, pre_generalize)
    _bf_transport(module)
    global _BF_CONSUMED
    _BF_CONSUMED = True
    return result
'''


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [node for node in ast.parse(source).body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
             and isinstance(node.value.func, ast.Name) and node.value.func.id == "_run_stages"]
    if len(calls) != 1:
        raise ValueError("BF16 transport requires one native stage runner call")
    lines = source.splitlines(keepends=True)
    at, after = calls[0].lineno - 1, calls[0].end_lineno
    return ("".join(lines[:at]) + PRELUDE + "\n" + "".join(lines[at:after])
            + "\nif not _BF_CONSUMED:\n    raise ValueError('BF16 transport marker was not consumed')\n"
            + "".join(lines[after:]))


def edit_pipeline(passes):
    if (passes.count("convert-arith-to-llvm") != 1 or passes.count("convert-cf-to-llvm") != 1 or MARKER in passes):
        raise ValueError("BF16 transport requires one unmodified complete LLVM seam")
    return [*passes, MARKER]


def ensure_registered():
    from .impr_features import ImprFeature, known, register
    if FEATURE not in known():
        register(ImprFeature(name=FEATURE, action_class="PASS", edit_pipeline=edit_pipeline,
            description="Preserve scalar BF16 LLVM transport as exact integer words; default off."))
    return FEATURE
