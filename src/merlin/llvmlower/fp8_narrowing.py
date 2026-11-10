"""Standard scalar RTNE truncation to the four FP8 interchange formats.

This implements arith.truncf semantics only. Source saturation and infinity
policies must remain executable in the input, not permissions from metadata.
The existing LLVM converter represents FP8 memory as bytes but cannot lower
the numeric truncation. Exact integer significand rounding avoids an f64
intermediate precision loss and handles target subnormals without FP FTZ.
"""

import ast

FEATURE = "lower_fp8_narrowing"
MARKER = "__merlin_fp8_narrowing__"

SUPPORT = r"""
_F8N_FORMATS = {
    'f8E4M3FN': (4, 3, 7, 126, 127, 'finite-nan'),
    'f8E4M3FNUZ': (4, 3, 8, 127, 128, 'unsigned-zero'),
    'f8E5M2': (5, 2, 15, 123, 126, 'ieee'),
    'f8E5M2FNUZ': (5, 2, 16, 127, 128, 'unsigned-zero'),
}


def _f8n_walk(op):
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _f8n_walk(child.operation)


def _fp8_narrow(module):
    with module.context:
        module.operation.verify()
        plans = []
        for op in tuple(_f8n_walk(module.operation)):
            if op.name != 'arith.truncf':
                continue
            if any(isinstance(value.type, (ir.RankedTensorType, ir.VectorType))
                   and str(value.type.element_type) in _F8N_FORMATS for value in op.results):
                raise ValueError('FP8 narrowing requires scalar casts')
            if not any(str(value.type) in _F8N_FORMATS for value in op.results):
                continue
            if (len(op.operands) != 1 or len(op.results) != 1 or op.regions
                    or str(op.operands[0].type) not in ('f16','bf16','f32','f64')):
                raise ValueError('FP8 narrowing requires a standard wider floating source')
            if any(name != 'fastmath' and not name.startswith('prov.') for name in op.attributes):
                raise ValueError('FP8 narrowing has an unqualified arithmetic property')
            if 'fastmath' in op.attributes and op.attributes['fastmath'] != ir.Attribute.parse('#arith.fastmath<none>'):
                raise ValueError('FP8 narrowing requires neutral flags and default RTNE')
            plans.append(op)
        for original in plans:
            _, target_fraction, target_bias, maximum, nan, special = _F8N_FORMATS[str(original.results[0].type)]
            annotations = {name: value for name, value in dict(original.attributes).items() if name != 'fastmath'}
            with ir.InsertionPoint(original), original.location:
                def emit(name, operands, result, attributes=None):
                    return ir.Operation.create(name, operands=operands, results=[result],
                        attributes={**annotations, **(attributes or {})}, loc=original.location).results[0]
                source = original.operands[0]
                if str(source.type) in ('f16','bf16'):
                    source = emit('arith.extf',[source],ir.F32Type.get())
                source_width, source_exponent, source_fraction, source_bias = (
                    (64,11,52,1023) if str(source.type)=='f64' else (32,8,23,127))
                word_type = ir.IntegerType.get_signless(source_width)
                i1, i8 = ir.IntegerType.get_signless(1), ir.IntegerType.get_signless(8)
                def constant(value):
                    return emit('arith.constant',[],word_type,{'value':ir.IntegerAttr.get(word_type,value)})
                def binary(name,left,right):
                    return emit('arith.'+name,[left,right],left.type)
                def compare(predicate,left,right):
                    return emit('arith.cmpi',[left,right],i1,
                        {'predicate':ir.IntegerAttr.get(ir.IntegerType.get_signless(64),predicate)})
                def select(condition,yes,no):
                    return emit('arith.select',[condition,yes,no],yes.type)
                raw = emit('arith.bitcast',[source],word_type)
                sign = binary('andi',binary('shrui',raw,constant(source_width-8)),constant(128))
                magnitude = binary('andi',raw,constant((1 << (source_width-1))-1))
                exponent = binary('shrui',magnitude,constant(source_fraction))
                fraction = binary('andi',magnitude,constant((1 << source_fraction)-1))
                significand = binary('ori',fraction,constant(1 << source_fraction))
                unbiased = binary('subi',exponent,constant(source_bias))
                delta = source_fraction-target_fraction
                shift = binary('subi',constant(delta+1-target_bias),unbiased)
                shift = binary('maxsi',shift,constant(delta))
                shift = binary('minsi',shift,constant(source_width-1))
                quotient = binary('shrui',significand,shift)
                mask = binary('subi',binary('shli',constant(1),shift),constant(1))
                remainder = binary('andi',significand,mask)
                half = binary('shli',constant(1),binary('subi',shift,constant(1)))
                above = compare(8,remainder,half)
                tie = compare(0,remainder,half)
                odd = compare(1,binary('andi',quotient,constant(1)),constant(0))
                increment = binary('ori',above,binary('andi',tie,odd))
                rounded = binary('addi',quotient,emit('arith.extui',[increment],word_type))
                normal = binary('shli',binary('addi',unbiased,constant(target_bias)),constant(target_fraction))
                normal = binary('addi',normal,binary('subi',rounded,constant(1 << target_fraction)))
                encoded = select(compare(5,unbiased,constant(1-target_bias)),normal,rounded)
                overflow = 124 if special=='ieee' else nan
                encoded = binary('minui',encoded,constant(overflow))
                # Source f32/f64 subnormals are smaller than every FP8
                # rounding midpoint. Half/BF16 sources widened exactly first.
                encoded = select(compare(0,exponent,constant(0)),constant(0),encoded)
                is_nan = compare(1,fraction,constant(0))
                nonfinite = select(is_nan,constant(nan),constant(124 if special=='ieee' else nan))
                encoded = select(compare(0,exponent,constant((1 << source_exponent)-1)),nonfinite,encoded)
                signed = binary('ori',encoded,sign)
                if special=='unsigned-zero':
                    signed = select(compare(0,encoded,constant(0)),constant(0),signed)
                bits = emit('arith.trunci',[signed],i8)
                result = emit('arith.bitcast',[bits],original.results[0].type)
            original.results[0].replace_all_uses_with(result)
            original.erase()
        module.operation.verify()
        return len(plans)
"""

PRELUDE = (
    SUPPORT
    + r"""
_F8N_PREVIOUS_RUN_STAGES = _run_stages
_F8N_CONSUMED = False


def _run_stages(ctx,module,pipeline,erase,mid=(),late=(),post_openmp=(),pre_generalize=()):
    global _F8N_CONSUMED
    marker = '__merlin_fp8_narrowing__'
    if marker not in pipeline:
        return _F8N_PREVIOUS_RUN_STAGES(ctx,module,pipeline,erase,mid,late,post_openmp,pre_generalize)
    if pipeline.count(marker)!=1 or not pipeline.startswith(marker+',') or _F8N_CONSUMED:
        raise ValueError('FP8 narrowing requires one initial source marker')
    _fp8_narrow(module)
    _F8N_CONSUMED = True
    return _F8N_PREVIOUS_RUN_STAGES(ctx,module,pipeline[len(marker)+1:],erase,mid,late,post_openmp,pre_generalize)
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
        raise ValueError("FP8 narrowing requires one native stage runner call")
    lines = source.splitlines(keepends=True)
    at, after = calls[0].lineno - 1, calls[0].end_lineno
    return (
        "".join(lines[:at])
        + PRELUDE
        + "\n"
        + "".join(lines[at:after])
        + "\nif not _F8N_CONSUMED:\n    raise ValueError('FP8 narrowing marker was not consumed')\n"
        + "".join(lines[after:])
    )


def edit_pipeline(passes):
    if MARKER in passes:
        raise ValueError("FP8 narrowing requires an unmodified source pipeline")
    return [MARKER, *passes]


def ensure_registered():
    from .impr_features import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                edit_pipeline=edit_pipeline,
                description="Exact standard scalar RTNE truncation to FP8; source policy remains executable.",
            )
        )
    return FEATURE
