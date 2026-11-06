"""Exact wrapping scalar integer arithmetic on device radix-256 components.

The frontend supplies typed linear SSA and broadcast maps. Every byte, carry,
comparison and result uses DMA and mesh operations, with no host payload access.
"""
from math import prod
from quantized_float import (_scratch, _literal, _copy, _load, _store, _linear)
from rtl_isa import DIM


def _rectifier(value):
    from frontend import _attr, _int, _const, Unsupported
    owner = value.owner
    if getattr(owner, 'name', None) == 'arith.maxsi':
        for index in (0, 1):
            try:
                if _const(owner.operands[index]) == 0:
                    return owner.operands[1-index]
            except Unsupported:
                pass
    if getattr(owner, 'name', None) != 'arith.select':
        return None
    condition, yes, no = owner.operands
    comparison = condition.owner
    if comparison.name != 'arith.cmpi':
        return None
    predicate = _int(_attr(comparison, 'predicate'))
    lhs, rhs = comparison.operands
    for source, zero, pred in ((lhs, rhs, predicate), (rhs, lhs, {2:4,3:5,4:2,5:3}.get(predicate))):
        try:
            if _const(zero) != 0:
                continue
        except Unsupported:
            continue
        if (pred in (2,3) and yes is zero and no is source
                or pred in (4,5) and yes is source and no is zero):
            return source
    return None


def recognize(op):
    from frontend import _attr, _spec, _affine_expr, _const, Unsupported
    from fractions import Fraction
    from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr
    if op.name not in ('linalg.generic', 'linalg.map') or len(op.results) != 1:
        return None
    dtype = _spec(op.results[0])['dtype']
    inputs = list(op.inputs)
    if dtype not in ('i8', 'i32') or not inputs or len(inputs) > 2:
        return None
    if any(_spec(value)['dtype'] != dtype for value in inputs):
        return None
    region = op.regions[0].block
    body = list(region.ops)
    if not body or body[-1].name != 'linalg.yield':
        return None
    # Ring linearization is valid only when intermediate operations keep the
    # same width. In particular widening an already wrapped sum is different.
    allowed = {'arith.constant', 'arith.addi', 'arith.subi', 'arith.muli',
               'arith.cmpi', 'arith.select', 'arith.maxsi', 'linalg.yield'}
    if any(s.name not in allowed or s.regions for s in body):
        return None
    if any(str(v.type) not in (dtype, 'i1') for s in body for v in s.results):
        return None
    yielded = body[-1].operands[0]
    # Pure storage copies remain with movement lowering. Arithmetic identities
    # such as add(x, 0) still have a typed scalar producer and use this route.
    if yielded in region.args:
        return None
    rectified = _rectifier(yielded)
    if rectified is not None:
        yielded = rectified
    environment = {}
    for arg, source in zip(region.args, inputs):
        try:
            environment[arg] = [0]*len(region.args) + [Fraction(_const(source))]
        except Unsupported:
            pass
    try:
        expression = _affine_expr(yielded, list(region.args), environment)
    except Unsupported:
        return None
    if any(expression[len(inputs):-1]) or any(Fraction(x).denominator != 1 for x in expression):
        return None
    weights = [int(x) for x in expression[:len(inputs)]]
    # Bounds cover all byte carries and every mesh partial, not runtime data.
    if sum(abs(w) for w in weights) > 32:
        return None
    rank = len(_spec(op.results[0])['shape'])
    maps = list(_attr(op, 'indexing_maps', []))
    if op.name == 'linalg.map':
        axes = [list(range(rank)) for _ in inputs]
    else:
        if (len(maps) != len(inputs)+1 or len(_attr(op, 'iterator_types', [])) != rank
                or any('parallel' not in str(i) for i in _attr(op, 'iterator_types', []))
                or tuple(maps[-1].data.results) != tuple(AffineDimExpr(i) for i in range(rank))):
            return None
        axes = []
        for source, mapping in zip(inputs, maps):
            shape = _spec(source)['shape']
            if len(mapping.data.results) != len(shape):
                return None
            current = []
            for size, index in zip(shape, mapping.data.results):
                if isinstance(index, AffineDimExpr):
                    if size != _spec(op.results[0])['shape'][index.position]:
                        return None
                    current.append(index.position)
                elif isinstance(index, AffineConstantExpr) and index.value == 0 and size == 1:
                    current.append(-1)
                else:
                    return None
            axes.append(current)
    return inputs, {'weights': weights, 'constant': int(expression[-1]),
                    'activation': 'relu' if rectified is not None else 'none',
                    'input_axes': axes, 'output_dtype': dtype}


def _bias(e, state, value, rows):
    pointer = _scratch(e, rows*4)
    _copy(e, 0 if value == 0 else _literal(e, state, [value], 'i32'), pointer, rows, 1, 0, 4, 'i32', 'i32')
    return pointer


def _byte_sum(e, state, packed, weights, rows, bias):
    result = _scratch(e, rows*4)
    _linear(e, state, packed, weights, result, rows, len(weights), bias=bias)
    return result


def _wrap_byte(e, state, raw, rows):
    # q=floor((raw+128)/256). The odd numerator excludes RNE ties.
    quotient = _scratch(e, rows)
    _load(e, raw, rows, stride=4, dtype='i32')
    _load(e, raw, rows, stride=4, dtype='i32', add=True)
    _load(e, _literal(e, state, [1], 'i32'), rows, stride=0, dtype='i32', add=True)
    _store(e, quotient, rows, scale=1/512)
    remainder = raw
    for _ in range(2):
        remainder = _byte_sum(e, state, quotient, [-128], rows, remainder)
    return quotient, remainder


def _tile(e, state, sources, strides, destination, rows, width, weights, constant, relu):
    carry = _bias(e, state, 0, rows)
    for byte in range(width):
        packed = _scratch(e, rows*len(sources))
        for col, (source, stride) in enumerate(zip(sources, strides)):
            _copy(e, e.add(source, byte), e.add(packed, col), rows, 1, stride, len(sources))
        digit = (constant >> (8*byte)) & 255
        _load(e, carry, rows, stride=4, dtype='i32')
        _load(e, 0 if digit == 0 else _literal(e, state, [digit], 'i32'), rows, stride=0, dtype='i32', add=True)
        _store(e, carry, rows, stride=4, full=True)
        raw = _byte_sum(e, state, packed, weights, rows, carry)
        quotient, remainder = _wrap_byte(e, state, raw, rows)
        _load(e, remainder, rows, stride=4, dtype='i32')
        _store(e, e.add(destination, byte), rows, stride=width)
        if byte+1 < width:
            signs = _scratch(e, rows*len(sources))
            identity = [-int(i == j) for i in range(len(sources)) for j in range(len(sources))]
            _linear(e, state, packed, identity, signs, rows, len(sources), len(sources), flag=True)
            _copy(e, quotient, carry, rows, 1, 1, 4, 'i8', 'i32')
            carry = _byte_sum(e, state, signs, weights, rows, carry)
            negative = _scratch(e, rows)
            _load(e, remainder, rows, stride=4, dtype='i32')
            _store(e, negative, rows)
            _linear(e, state, negative, [-1], negative, rows, 1, flag=True)
            carry = _byte_sum(e, state, negative, [-1], rows, carry)
    if relu:
        _relu_tile(e, state, destination, rows, width)


def _relu_tile(e, state, pointer, rows, width):
    if width == 1:
        _load(e, pointer, rows)
        _store(e, pointer, rows, relu=True)
        return
    sign = _scratch(e, rows)
    packed = _scratch(e, rows*2)
    _copy(e, e.add(pointer, 3), sign, rows, 1, 4, 1)
    _linear(e, state, sign, [-1], sign, rows, 1, flag=True)
    _copy(e, sign, e.add(packed, 1), rows, 1, 1, 2)
    for byte in range(4):
        _copy(e, e.add(pointer, byte), packed, rows, 1, 4, 2)
        positive, negative, flag = (_scratch(e, rows*4) for _ in range(3))
        _linear(e, state, packed, [1, -128], positive, rows, 2)
        _linear(e, state, packed, [-1, -128], negative, rows, 2,
                bias=_bias(e, state, -1, rows))
        _linear(e, state, packed, [-1, -128], flag, rows, 2, flag=True)
        for source in (positive, negative):
            _load(e, source, rows, stride=4, dtype='i32')
            _store(e, source, rows, relu=True)
        merged = _scratch(e, rows*3)
        for col, source in enumerate((positive, negative, flag)):
            _copy(e, source, e.add(merged, col), rows, 1, 1, 3)
        value = _byte_sum(e, state, merged, [1, -1, -1], rows, None)
        _load(e, value, rows, stride=4, dtype='i32')
        _store(e, e.add(pointer, byte), rows, stride=4)


def lower(e, command, state):
    if command['opcode'] != 'DEVICE_INTEGER_AFFINE':
        return False
    operands, attrs = command['operands'], command['attributes']
    dst = operands['dst']
    shape = e.tensors[dst]['shape']
    sources = [operands[key] for key in ('x','y') if key in operands]
    width = 1 if e.tensors[dst]['dtype'] == 'i8' else 4
    source_strides = []
    for source, axes in zip(sources, attrs['input_axes']):
        strides = [0]*len(shape)
        step = width
        for size, axis in reversed(list(zip(e.tensors[source]['shape'], axes))):
            if axis >= 0:
                strides[axis] += step
            step *= size
        source_strides.append(strides)
    def visit(axis, pointers, output):
        if axis == len(shape)-1 or not shape:
            count = shape[-1] if shape else 1
            steps = [s[-1] if shape else 0 for s in source_strides]
            def tile(start, rows):
                pointers_at = [e.add(p, e.mul(start, s)) for p,s in zip(pointers, steps)]
                _tile(e, state, pointers_at, steps, e.add(output, e.mul(start, width)), rows, width,
                      attrs['weights'], attrs['constant'], attrs['activation'] == 'relu')
            full = count//DIM*DIM
            if full:
                e.loop(0, full, DIM, lambda start: tile(start, DIM))
            if full < count:
                tile(full, count-full)
            return
        def body(index):
            visit(axis+1, [e.add(p,e.mul(index,s[axis])) for p,s in zip(pointers,source_strides)],
                  e.add(output,e.mul(index,prod(shape[axis+1:])*width)))
        e.loop(0, shape[axis], 1, body)
    visit(0, [e.ptr(n) for n in sources], e.ptr(dst))
    return True
