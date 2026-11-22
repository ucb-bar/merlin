"""Strict FP8 scalar math using upstream arithmetic/math type emulation.

Reuse LLVM a47bddcc EmulateUnsupportedFloats and ExtendToSupportedTypes.
Keep f64 intermediates, typed storage and one neutral narrowing per source
operation. Upstream deliberately leaves FMA alone and marks introduced casts
contractible; this adapter handles FMA and removes that extra permission.
The separate exact FP8 cast features consume the resulting standard casts.
"""

import ast

FEATURE = "emulate_fp8_elementwise"
MARKER = "__merlin_fp8_elementwise__"

SUPPORT = r"""
_F8E_FORMATS = {'f8E4M3FN','f8E4M3FNUZ','f8E5M2','f8E5M2FNUZ'}


def _f8e_walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _f8e_walk(child.operation)


def _fp8_elementwise(module):
    from torch_mlir.passmanager import PassManager as _f8e_pm
    with module.context:
        module.operation.verify()
        transport = {'arith.constant','arith.bitcast','arith.extf','arith.truncf','arith.select'}
        plans, shaped = [], False
        for op in tuple(_f8e_walk(module.operation)):
            if not op.name.startswith(('arith.','math.')) or op.name in transport:
                continue
            values = tuple(op.operands)+tuple(op.results)
            def is_fp8(type_):
                scalar = type_.element_type if isinstance(type_,(ir.RankedTensorType,ir.VectorType)) else type_
                return str(scalar) in _F8E_FORMATS
            if not any(is_fp8(value.type) for value in values):
                continue
            if (op.regions or op.successors or
                    any(key not in ('fastmath','predicate') and not key.startswith('prov.') for key in op.attributes)):
                raise ValueError('FP8 math requires an ordinary typed arithmetic operation and default RTNE')
            if any(isinstance(value.type,ir.VectorType) for value in values):
                raise ValueError('FP8 math requires scalar or logical tensor operations before vector mapping')
            shaped = shaped or any(isinstance(value.type,ir.RankedTensorType) for value in values)
            plans.append(op)
        if not plans:
            return 0
        for op in plans:
            if 'fastmath' in op.attributes and op.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>'):
                # Fast-math flags permit relaxed results; they do not require
                # them. Choose the strict refinement and retain the permission
                # as intent without contracting across source FP8 rounding.
                op.attributes['prov.fp8_native_source_fastmath'] = ir.StringAttr.get(str(op.attributes['fastmath']))
                op.attributes['fastmath'] = ir.Attribute.parse('#arith.fastmath<none>')
        if shaped:
            _f8e_pm.parse('builtin.module(func.func(convert-elementwise-to-linalg))').run(module.operation)
        # Existing casts are legal and retained by both upstream passes.
        # Preserve their flags; neutralize only newly introduced conversions.
        original_casts = {op for op in _f8e_walk(module.operation) if op.name in ('arith.extf','arith.truncf')}
        f64 = ir.F64Type.get()
        i8, i1 = ir.IntegerType.get_signless(8), ir.IntegerType.get_signless(1)
        for op in tuple(_f8e_walk(module.operation)):
            if op.name not in ('arith.negf','math.absf','math.copysign') or not any(
                    str(v.type) in _F8E_FORMATS for v in op.results):
                continue
            arity = 2 if op.name == 'math.copysign' else 1
            if (len(op.operands)!=arity or len(op.results)!=1 or
                    any(v.type != op.results[0].type for v in op.operands)):
                raise ValueError('FP8 word math requires its complete same-type scalar signature')
            with ir.InsertionPoint(op), op.location:
                def emit(name,operands,type_,attributes=None):
                    return ir.Operation.create(name,results=[type_],operands=operands,
                        attributes=attributes or {}).results[0]
                def constant(v):
                    return emit('arith.constant',[],i8,{'value':ir.IntegerAttr.get(i8,v)})
                word = emit('arith.bitcast',[op.operands[0]],i8)
                if op.name == 'arith.negf':
                    result = emit('arith.xori',[word,constant(128)],i8)
                else:
                    result = emit('arith.andi',[word,constant(127)],i8)
                    if op.name == 'math.copysign':
                        second = emit('arith.bitcast',[op.operands[1]],i8)
                        sign = emit('arith.andi',[second,constant(128)],i8)
                        result = emit('arith.ori',[result,sign],i8)
                if str(op.results[0].type).endswith('FNUZ'):
                    magnitude = emit('arith.andi',[word,constant(127)],i8)
                    special = emit('arith.cmpi',[magnitude,constant(0)],i1,
                        {'predicate':ir.IntegerAttr.get(ir.IntegerType.get_signless(64),0)})
                    result = emit('arith.select',[special,word,result],i8)
                result = emit('arith.bitcast',[result],op.results[0].type)
            op.results[0].replace_all_uses_with(result)
            op.erase()
        for op in tuple(_f8e_walk(module.operation)):
            if op.name != 'math.fma' or not any(str(v.type) in _F8E_FORMATS for v in op.results):
                continue
            if (len(op.operands)!=3 or len(op.results)!=1 or
                    any(v.type != op.results[0].type for v in op.operands)):
                raise ValueError('FP8 FMA requires its complete same-type scalar signature')
            with ir.InsertionPoint(op), op.location:
                widened = [ir.Operation.create('arith.extf',results=[f64],operands=[v]).results[0] for v in op.operands]
                result = ir.Operation.create('math.fma',results=[f64],operands=widened,attributes=dict(op.attributes))
                rounded = ir.Operation.create('arith.truncf',results=[op.results[0].type],operands=[result.results[0]])
            op.results[0].replace_all_uses_with(rounded.results[0])
            op.erase()
        _f8e_pm.parse('builtin.module('
            'arith-emulate-unsupported-floats{source-types=f8E4M3FN,f8E4M3FNUZ,f8E5M2,f8E5M2FNUZ target-type=f64},'
            'math-extend-to-supported-types{extra-types=f16,bf16 target-type=f64})').run(module.operation)
        for op in _f8e_walk(module.operation):
            if op.name in ('arith.extf','arith.truncf') and op not in original_casts and 'fastmath' in op.attributes:
                op.attributes['fastmath'] = ir.Attribute.parse('#arith.fastmath<none>')
        module.operation.verify()
        return len(plans)
"""

PRELUDE = SUPPORT + r"""
_F8E_PREVIOUS_RUN_STAGES = _run_stages
_F8E_CONSUMED = False


def _run_stages(ctx,module,pipeline,erase,mid=(),late=(),post_openmp=(),pre_generalize=()):
    marker = '__merlin_fp8_elementwise__'
    if marker not in pipeline:
        return _F8E_PREVIOUS_RUN_STAGES(ctx,module,pipeline,erase,mid,late,post_openmp,pre_generalize)
    if pipeline.count(marker)!=1 or not pipeline.startswith(marker+','):
        raise ValueError('FP8 math requires one initial source marker')
    _fp8_elementwise(module)
    global _F8E_CONSUMED
    _F8E_CONSUMED = True
    return _F8E_PREVIOUS_RUN_STAGES(ctx,module,pipeline[len(marker)+1:],erase,mid,late,post_openmp,pre_generalize)
"""


def bind_runner(source, *, selected):
    if not selected:
        return source
    calls = [node for node in ast.parse(source).body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
             and node.value.func.id == "_run_stages"]
    if len(calls) != 1:
        raise ValueError("FP8 math requires one native stage runner call")
    lines = source.splitlines(keepends=True)
    at, after = calls[0].lineno - 1, calls[0].end_lineno
    return ("".join(lines[:at]) + PRELUDE + "\n" + "".join(lines[at:after])
            + "\nif not _F8E_CONSUMED:\n    raise ValueError('FP8 math marker was not consumed')\n"
            + "".join(lines[after:]))


def edit_pipeline(passes):
    if MARKER in passes:
        raise ValueError("FP8 math requires an unmodified source pipeline")
    return [MARKER, *passes]


def ensure_registered():
    from merlin.compiler.optimizations.registry import ImprFeature, known, register

    if FEATURE not in known():
        register(ImprFeature(name=FEATURE, action_class="PASS", edit_pipeline=edit_pipeline,
            implies=frozenset({"lower_fp8_narrowing", "lower_fp8_widening"}),
            description="Strict FP8 arithmetic through upstream f64 emulation and exact per-operation casts."))
    return FEATURE
