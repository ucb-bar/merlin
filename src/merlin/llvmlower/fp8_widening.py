"""Exact scalar FP8 widening without changing typed storage or portable casts.

The existing native SDK converts FP8 memory to bytes but leaves arith.extf
unresolved. Decode standard sign/exponent/fraction fields through ordinary
integer operations and exact f32 arithmetic. Every finite FP8 value fits
exactly in f16/BF16/f32/f64. Computed NaN payloads are not selected by contract.
"""

import ast

FEATURE = "lower_fp8_widening"
MARKER = "__merlin_fp8_widening__"

SUPPORT = r"""
_F8W_FORMATS = {
    'f8E4M3FN': (4, 3, 7, 'finite-nan'),
    'f8E4M3FNUZ': (4, 3, 8, 'unsigned-zero'),
    'f8E5M2': (5, 2, 15, 'ieee'),
    'f8E5M2FNUZ': (5, 2, 16, 'unsigned-zero'),
}


def _f8w_walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _f8w_walk(child.operation)


def _fp8_widen(module, *, _generalized=False):
    with module.context:
        module.operation.verify()
        plans = []
        transport = {'arith.constant', 'arith.bitcast', 'arith.select',
            'memref.load', 'memref.store', 'tensor.extract', 'tensor.insert',
            'tensor.yield', 'linalg.yield', 'linalg.fill', 'func.return',
            'scf.if', 'scf.for', 'scf.while', 'scf.yield', 'scf.condition', 'cf.br', 'cf.cond_br'}
        for op in tuple(_f8w_walk(module.operation)):
            values = tuple(op.operands)+tuple(op.results)
            if op.name == 'arith.extf' and any(
                    isinstance(value.type, (ir.RankedTensorType, ir.VectorType))
                    and str(value.type.element_type) in _F8W_FORMATS for value in values):
                raise ValueError('FP8 widening requires scalar casts')
            if not any(str(value.type) in _F8W_FORMATS for value in values):
                continue
            if op.name in transport:
                continue
            if (op.name != 'arith.extf' or len(op.operands) != 1 or len(op.results) != 1
                    or op.regions or str(op.operands[0].type) not in _F8W_FORMATS
                    or str(op.results[0].type) not in ('f16', 'bf16', 'f32', 'f64')):
                raise ValueError('unqualified FP8 scalar operation: '+op.name)
            if ('fastmath' in op.attributes and op.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>')
                    or 'roundingmode' in op.attributes):
                raise ValueError('FP8 widening requires neutral flags and exact conversion')
            if any(name != 'fastmath' and not name.startswith('prov.') for name in op.attributes):
                raise ValueError('FP8 widening has an unqualified arithmetic property')
            plans.append(op)
        # Complete validation precedes edits. This profile adds no permission
        # for narrowing, arithmetic, effects, source borrowing or target mapping.
        named = False
        for op in plans:
            parent = op.parent
            while parent is not None:
                if parent.name.startswith('linalg.') and parent.name != 'linalg.generic':
                    from torch_mlir.dialects.linalg import isa_contraction_op
                    named = named or isa_contraction_op(parent)
                parent = parent.parent
        if named:
            # Named linalg bodies must match their canonical arithmetic.
            # Preserve named contractions in portable source; only this
            # selected native legalization needs their ordinary generic form.
            if _generalized:
                raise ValueError('upstream did not generalize the FP8 cast owner')
            from torch_mlir.passmanager import PassManager
            PassManager.parse('builtin.module(func.func(linalg-generalize-named-ops))').run(module.operation)
            return _fp8_widen(module, _generalized=True)
        i1, i8, i32 = (ir.IntegerType.get_signless(n) for n in (1, 8, 32))
        f32 = ir.F32Type.get()
        for original in plans:
            exponent_bits, fraction_bits, bias, special = _F8W_FORMATS[str(original.operands[0].type)]
            annotations = {name: value for name, value in dict(original.attributes).items() if name != 'fastmath'}
            with ir.InsertionPoint(original), original.location:
                def emit(name, operands, result, attributes=None):
                    return ir.Operation.create(name, operands=operands, results=[result],
                        attributes={**annotations, **(attributes or {})}, loc=original.location).results[0]
                def constant(value):
                    return emit('arith.constant', [], i32, {'value': ir.IntegerAttr.get(i32, value)})
                def binary(name, left, right):
                    return emit('arith.'+name, [left, right], i32)
                def eq(left, right):
                    return emit('arith.cmpi', [left, right], i1,
                        {'predicate': ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 0)})
                def select(condition, yes, no):
                    return emit('arith.select', [condition, yes, no], i32)

                raw = emit('arith.bitcast', [original.operands[0]], i8)
                word = emit('arith.extui', [raw], i32)
                sign = binary('shli', binary('andi', word, constant(128)), constant(24))
                fraction = binary('andi', word, constant((1 << fraction_bits)-1))
                exponent = binary('andi', binary('shrui', word, constant(fraction_bits)), constant((1 << exponent_bits)-1))
                normal_exp = binary('shli', binary('addi', exponent, constant(127-bias)), constant(23))
                normal_bits = binary('ori', normal_exp, binary('shli', fraction, constant(23-fraction_bits)))
                # The fraction and its scale are exact in f32. Even the least
                # nonzero result is normal f32, so no host FTZ dependency is added.
                integer_fraction = emit('arith.uitofp', [fraction], f32)
                scale = emit('arith.constant', [], f32,
                    {'value': ir.FloatAttr.get(f32, 2.0**(1-bias-fraction_bits))})
                subnormal = emit('arith.mulf', [integer_fraction, scale], f32)
                subnormal_bits = emit('arith.bitcast', [subnormal], i32)
                bits = select(eq(exponent, constant(0)), subnormal_bits, normal_bits)
                if special == 'ieee':
                    nonfinite = select(eq(fraction, constant(0)), constant(0x7f800000), constant(0x7fc00000))
                    bits = select(eq(exponent, constant((1 << exponent_bits)-1)), nonfinite, bits)
                elif special == 'finite-nan':
                    bits = select(eq(binary('andi', word, constant(127)), constant(127)), constant(0x7fc00000), bits)
                bits = binary('ori', bits, sign)
                if special == 'unsigned-zero':
                    bits = select(eq(word, constant(128)), constant(0x7fc00000), bits)
                value = emit('arith.bitcast', [bits], f32)
                target = original.results[0].type
                if target != f32:
                    value = emit('arith.extf' if str(target) == 'f64' else 'arith.truncf', [value], target)
            original.results[0].replace_all_uses_with(value)
            original.erase()
        module.operation.verify()
        return len(plans)
"""

PRELUDE = (
    SUPPORT
    + r"""
_F8W_PREVIOUS_RUN_STAGES = _run_stages
_F8W_CONSUMED = False


def _run_stages(ctx, module, pipeline, erase, mid=(), late=(), post_openmp=(), pre_generalize=()):
    global _F8W_CONSUMED
    marker = '__merlin_fp8_widening__'
    if marker not in pipeline:
        return _F8W_PREVIOUS_RUN_STAGES(ctx,module,pipeline,erase,mid,late,post_openmp,pre_generalize)
    if pipeline.count(marker) != 1 or not pipeline.startswith(marker+',') or _F8W_CONSUMED:
        raise ValueError('FP8 widening requires one initial source marker')
    _fp8_widen(module)
    _F8W_CONSUMED = True
    return _F8W_PREVIOUS_RUN_STAGES(ctx,module,pipeline[len(marker)+1:],erase,mid,late,post_openmp,pre_generalize)
"""
)


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
        raise ValueError("FP8 widening requires one native stage runner call")
    lines = source.splitlines(keepends=True)
    at, after = calls[0].lineno - 1, calls[0].end_lineno
    return (
        "".join(lines[:at])
        + PRELUDE
        + "\n"
        + "".join(lines[at:after])
        + "\nif not _F8W_CONSUMED:\n    raise ValueError('FP8 widening marker was not consumed')\n"
        + "".join(lines[after:])
    )


def edit_pipeline(passes):
    if MARKER in passes:
        raise ValueError("FP8 widening requires an unmodified source pipeline")
    return [MARKER, *passes]


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description="Exact scalar FP8 widening to f16/BF16/f32/f64; storage ABI unchanged.",
            )
        )
    return FEATURE
