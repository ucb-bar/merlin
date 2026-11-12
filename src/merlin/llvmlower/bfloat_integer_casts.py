"""Exact scalar integer-to-BF16 RTNE expansion in standard integer arithmetic.

The upstream unsupported-float emulator converts through f32, which can double
round wide integers. Round the source integer significand once instead.
"""

SUPPORT = r'''
def _bf_integer_cast(original):
    annotations = {name:value for name,value in dict(original.attributes).items() if name != 'fastmath'}
    i1,i16,i64 = (ir.IntegerType.get_signless(width) for width in (1,16,64))
    with ir.InsertionPoint(original), original.location:
        def emit(name, operands, result, attributes=None):
            return ir.Operation.create(name,operands=operands,results=[result],
                attributes={**annotations,**(attributes or {})},loc=original.location).results[0]
        def constant(value):
            return emit('arith.constant',[],i64,{'value':ir.IntegerAttr.get(i64,value)})
        def binary(name,left,right):
            return emit('arith.'+name,[left,right],left.type)
        def compare(predicate,left,right):
            return emit('arith.cmpi',[left,right],i1,{'predicate':ir.IntegerAttr.get(i64,predicate)})
        def select(condition,yes,no):
            return emit('arith.select',[condition,yes,no],yes.type)
        source = original.operands[0]
        signed = original.name == 'arith.sitofp'
        if source.type != i64:
            source = emit('arith.extsi' if signed else 'arith.extui',[source],i64)
        zero,one = constant(0),constant(1)
        negative = compare(2,source,zero) if signed else compare(1,zero,zero)
        magnitude = select(negative,binary('subi',zero,source),source)
        # Find floor(log2(magnitude)) without a floating approximation or a
        # zero-undefined count-leading-zeros operation. All shifts stay in range.
        reduced,exponent = magnitude,zero
        for shift in (32,16,8,4,2,1):
            large = compare(9,reduced,constant(1 << shift))
            reduced = select(large,binary('shrui',reduced,constant(shift)),reduced)
            exponent = binary('addi',exponent,select(large,constant(shift),zero))
        precision = constant(7)
        right = select(compare(4,exponent,precision),binary('subi',exponent,precision),one)
        quotient = binary('shrui',magnitude,right)
        mask = binary('subi',binary('shli',one,right),one)
        remainder = binary('andi',magnitude,mask)
        half = binary('shli',one,binary('subi',right,one))
        increment = binary('ori',compare(8,remainder,half),binary('andi',
            compare(0,remainder,half),compare(1,binary('andi',quotient,one),zero)))
        rounded = binary('addi',quotient,emit('arith.extui',[increment],i64))
        left = select(compare(2,exponent,precision),binary('subi',precision,exponent),zero)
        exact = binary('shli',magnitude,left)
        significand = select(compare(4,exponent,precision),rounded,exact)
        encoded = binary('addi',binary('shli',binary('addi',exponent,constant(127)),precision),
                         binary('subi',significand,constant(128)))
        encoded = select(compare(0,magnitude,zero),zero,encoded)
        encoded = binary('ori',encoded,select(negative,constant(32768),zero))
        word = emit('arith.trunci',[encoded],i16)
        result = emit('arith.bitcast',[word],ir.BF16Type.get())
    original.results[0].replace_all_uses_with(result)
    original.erase()
'''
