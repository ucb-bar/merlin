"""Exact unary float islands under the selected declared quantized model.

The selected [-1,1] calibration domain and signed byte range give scale 2**-6.
Entry RNE predicates operate on the f32 storage bytes; bounded mesh MACs assemble
one final quantized/dequantized result. No tensor payload reaches the host.
"""
from fractions import Fraction
from math import isfinite, prod
import struct

from rtl_isa import DIM

ACC = 1 << 31
ADD = ACC | (1 << 30)
FULL = ACC | (1 << 29)


def _bits(value):
    return struct.unpack('<I', struct.pack('<f', float(value)))[0]


def _literal_fraction(value):
    value = struct.unpack('<f', struct.pack('<f', float(value)))[0]
    if not isfinite(value):
        raise ValueError('nonfinite float island literal')
    return Fraction(value)


def _combine(name, left, right):
    a, b = left
    c, d = right
    if name == 'arith.addf':
        return a + c, b + d
    if name == 'arith.subf':
        return a - c, b - d
    if name == 'arith.mulf' and not (a and c):
        return (a * d, b * d) if not c else (c * b, d * b)
    if name == 'arith.divf' and not c and d:
        return a / d, b / d
    raise ValueError('float island is not affine with finite literal divisors')


def _expression(block, inputs):
    from frontend import _const, Unsupported
    values = {}
    source = None
    for arg, tensor in zip(block.args, inputs):
        try:
            values[arg] = (Fraction(0), _literal_fraction(_const(tensor)))
        except Unsupported:
            if source is not None:
                raise ValueError('float island needs one runtime storage input')
            source = tensor
            values[arg] = (Fraction(1), Fraction(0))
    arithmetic = False
    operations = list(block.ops)
    for scalar in operations[:-1]:
        if scalar.regions or len(scalar.results) != 1 or str(scalar.results[0].type) != 'f32':
            raise ValueError('float island needs pure scalar f32 operations')
        if scalar.name == 'arith.constant':
            value = (Fraction(0), _literal_fraction(_const(scalar.results[0])))
        elif scalar.name == 'arith.negf':
            value = tuple(-x for x in values[scalar.operands[0]])
            arithmetic = True
        elif scalar.name in ('arith.addf', 'arith.subf', 'arith.mulf', 'arith.divf'):
            operands = []
            for operand in scalar.operands:
                if operand not in values:
                    values[operand] = (Fraction(0), _literal_fraction(_const(operand)))
                operands.append(values[operand])
            value = _combine(scalar.name, *operands)
            arithmetic = True
        else:
            raise ValueError('unsupported scalar float island operation')
        values[scalar.results[0]] = value
    if not arithmetic or source is None:
        raise ValueError('storage identity and constant results retain their own semantics')
    return source, values[operations[-1].operands[0]]


def recognize(op, entry_args):
    """Recognize a parallel unary affine f32 island from its typed scalar SSA."""
    from frontend import _attr, _spec, Unsupported
    from xdsl.ir.affine import AffineDimExpr
    if (op.name not in ('linalg.generic', 'linalg.map') or len(op.results) != 1
            or _spec(op.results[0])['dtype'] != 'f32'
            or len(op.regions) != 1 or len(op.regions[0].blocks) != 1):
        return None
    inputs = list(op.inputs)
    if any(_spec(value)['dtype'] != 'f32' for value in inputs):
        return None
    block = op.regions[0].block
    operations = list(block.ops)
    if not operations or operations[-1].name != 'linalg.yield' or len(operations[-1].operands) != 1:
        return None
    try:
        source, (weight, constant) = _expression(block, inputs)
    except (ValueError, KeyError, Unsupported, OverflowError):
        return None
    if source not in entry_args or _spec(source) != _spec(op.results[0]):
        return None
    if op.name == 'linalg.generic':
        rank = len(_spec(source)['shape'])
        identity = tuple(AffineDimExpr(i) for i in range(rank))
        maps = list(_attr(op, 'indexing_maps', []))
        iters = list(_attr(op, 'iterator_types', []))
        if (len(maps) != len(inputs) + 1 or len(iters) != rank
                or any('parallel' not in str(iterator) for iterator in iters)):
            return None
        for index in (inputs.index(source), len(inputs)):
            if maps[index].data.num_dims != rank or tuple(maps[index].data.results) != identity:
                return None
    attrs = {'weight_numerator': str(weight.numerator), 'weight_denominator': str(weight.denominator),
             'constant_numerator': str(constant.numerator), 'constant_denominator': str(constant.denominator),
             'entry_scale_exponent': -6, 'rounding': 'half_even', 'output_dtype': 'f32'}
    return source, attrs


def _scratch(e, size):
    name = '_float_scratch_' + str(len(e.tensors))
    while name in e.tensors:
        name += '_'
    e.tensors[name] = {'shape': [size], 'dtype': 'i8', 'role': 'intermediate'}
    return e.ptr(name)


def _literal(e, state, values, dtype='i8'):
    values = tuple(values)
    key = ('float_literal', dtype, values)
    if key not in state:
        name = '_float_literal_' + str(len(e.tensors))
        while name in e.tensors:
            name += '_'
        state[key] = e.global_data(name, values, dtype)
    return e.ptr(state[key])


def _descriptor(address, rows, cols):
    return address | (rows << 48) | (cols << 32)


def _load(e, pointer, rows, cols=1, stride=1, dtype='i8', add=False, scale=1):
    shrink = int(dtype == 'i8')
    word = (_bits(scale) << 32) | (DIM << 16) | (1 << 8) | (shrink << 2) | 1
    e.issue(0, word, stride)
    e.issue(2, pointer, _descriptor(ADD if add else ACC, rows, cols))
    e.fence()


def _store(e, pointer, rows, cols=1, stride=1, scale=1, relu=False, full=False):
    e.issue(0, 2 | (4 if relu else 0), (_bits(scale) << 32) | stride)
    e.issue(3, pointer, _descriptor(FULL if full else ACC, rows, cols))
    e.fence()


def _copy(e, source, destination, rows, cols, source_stride, dest_stride,
          input_dtype='i8', output_dtype='i8'):
    _load(e, source, rows, cols, source_stride, input_dtype)
    _store(e, destination, rows, cols, dest_stride, full=output_dtype == 'i32')


def _normalize_flag(e, pointer, rows, cols=1):
    _load(e, pointer, rows, cols, cols, scale=1 / 127)
    _store(e, pointer, rows, cols, cols)


def _linear(e, state, source, weights, destination, rows, k, n=1, bias=None, flag=False):
    """A single bounded WS mesh pass; all k <= 6, n <= 4 intermediates fit i32."""
    assert 0 < k <= DIM and 0 < n <= DIM and 0 < rows <= DIM
    weight_pointer = _literal(e, state, weights)
    e.issue(0, (_bits(1) << 32) | (1 << 16) | 4, 1 << 48)
    if bias is not None:
        _load(e, bias, rows, n, n * 4, 'i32')
    for pointer, stride, address, nr, nc in ((source, k, 0, rows, k),
                                           (weight_pointer, n, DIM, k, n)):
        e.issue(0, (_bits(1) << 32) | (DIM << 16) | (1 << 8) | 1, stride)
        e.issue(2, pointer, _descriptor(address, nr, nc))
        e.fence()
    e.issue(6, _descriptor(DIM, k, n), _descriptor(ADD if bias is not None else ACC, rows, n))
    e.issue(4, _descriptor(0, rows, k), _descriptor(0xffffffff, rows, n))
    e.fence()
    _store(e, destination, rows, n, n if flag else n * 4,
           scale=127 if flag else 1, relu=flag, full=not flag)
    if flag:
        _normalize_flag(e, destination, rows, n)


def _magnitude(e, state, source, rows):
    """Extract abs(word)>>16, low16!=0 and sign without host loads or lossy DMA."""
    signs = _scratch(e, rows * 4)
    _linear(e, state, source, [-int(i == j) for i in range(4) for j in range(4)],
            signs, rows, 4, 4, flag=True)
    high_byte = _scratch(e, rows)
    _load(e, e.add(source, 3), rows, stride=4)
    for multiplier in (127, 1):
        _load(e, e.add(signs, 3), rows, stride=4, add=True, scale=multiplier)
    _store(e, high_byte, rows)
    packed = _scratch(e, rows * 3)
    for pointer, stride, col in ((e.add(source, 2), 4, 0), (high_byte, 1, 1),
                                 (e.add(signs, 2), 4, 2)):
        _copy(e, pointer, e.add(packed, col), rows, 1, stride, 3)
    high = _scratch(e, rows * 4)
    _linear(e, state, packed, [1, 127, 127], high, rows, 3)
    for weight in (127, 2):
        _linear(e, state, packed, [0, weight, weight], high, rows, 3, bias=high)
    low = _scratch(e, rows * 4)
    _linear(e, state, source, [1, 1, 0, 0], low, rows, 4)
    for weight in (127, 127, 2):
        _linear(e, state, signs, [weight, weight, 0, 0], low, rows, 4, bias=low)
    low_nonzero = _scratch(e, rows)
    _load(e, low, rows, stride=4, dtype='i32')
    _store(e, low_nonzero, rows, scale=127, relu=True)
    _normalize_flag(e, low_nonzero, rows)
    return high, low_nonzero, e.add(signs, 3)


def _predicate(e, state, high, low, threshold, inclusive, rows, destination):
    # Entry midpoints have zero low16. Doubling high separates > from >=;
    # low16!=0 distinguishes the strict boundary's immediate successors.
    assert threshold & 65535 == 0
    _load(e, high, rows, stride=4, dtype='i32')
    _load(e, high, rows, stride=4, dtype='i32', add=True)
    shift = -2 * (threshold >> 16) + int(inclusive)
    _load(e, _literal(e, state, [shift], 'i32'), rows, stride=0, dtype='i32', add=True)
    if not inclusive:
        _load(e, low, rows, add=True)
    _store(e, destination, rows, scale=127, relu=True)
    _normalize_flag(e, destination, rows)


def _output_bytes(code, weight, constant):
    result = min(127, max(-128, round(code * weight + 64 * constant)))
    word = _bits(Fraction(result, 64))
    return [((word >> (8 * i) & 255) ^ 128) - 128 for i in range(4)]


def _transitions(weight, constant=Fraction(0)):
    """Tabulate all 256 entry codes with exact arithmetic and one output RNE."""
    previous = [_output_bytes(0, weight, constant)] * 2
    for code in range(1, 129):
        current = [_output_bytes(q, weight, constant) for q in (min(code, 127), -code)]
        deltas = [[a - b for a, b in zip(row, old)] for row, old in zip(current, previous)]
        previous = current
        if any(any(row) for row in deltas):
            yield _bits(Fraction(2 * code - 1, 128)), code % 2 == 0, deltas


def _delta_weights(deltas):
    # A byte difference lies in [-255,255]; three signed-byte pieces suffice.
    weights = []
    remaining = [value for row in deltas for value in row]
    for _ in range(3):
        piece = [min(127, max(-128, value)) for value in remaining]
        weights.extend(piece)
        remaining = [value - part for value, part in zip(remaining, piece)]
    assert not any(remaining)
    return weights


def _emit_tile(e, state, source, destination, rows, weight, constant):
    high, low, sign = _magnitude(e, state, source, rows)
    nan = _scratch(e, rows)
    _predicate(e, state, high, low, 0x7f800000, False, rows, nan)
    predicate, flags, result = (_scratch(e, rows * size) for size in (1, 6, 16))
    initial = _output_bytes(0, weight, constant)
    _copy(e, _literal(e, state, initial, 'i32'), result, rows, 4, 0, 16, 'i32', 'i32')
    for threshold, inclusive, deltas in _transitions(weight, constant):
        _predicate(e, state, high, low, threshold, inclusive, rows, predicate)
        for negative in (False, True):
            _load(e, predicate, rows)
            _load(e, sign, rows, stride=4, add=True, scale=1 if negative else -1)
            _load(e, nan, rows, add=True, scale=-1)
            if negative:
                _load(e, _literal(e, state, [-1], 'i32'), rows, stride=0, dtype='i32', add=True)
            _store(e, e.add(flags, int(negative)), rows, stride=6, relu=True)
        for col in (2, 4):
            _copy(e, flags, e.add(flags, col), rows, 2, 6, 6)
        _linear(e, state, flags, _delta_weights(deltas), result, rows, 6, 4, bias=result)
    _copy(e, result, destination, rows, 4, 16, 4, 'i32', 'i8')


def lower(e, command, state):
    if command['opcode'] != 'DEVICE_QUANTIZED_FLOAT':
        return False
    operands, attrs = command['operands'], command['attributes']
    weight = Fraction(int(attrs['weight_numerator']), int(attrs['weight_denominator']))
    constant = Fraction(int(attrs['constant_numerator']), int(attrs['constant_denominator']))
    src, dst = operands['src'], operands['dst']
    count = prod(e.tensors[dst]['shape'])
    def tile(start, rows):
        offset = e.mul(start, 4)
        _emit_tile(e, state, e.ptr(src, offset), e.ptr(dst, offset), rows, weight, constant)
    full = count // DIM * DIM
    if full:
        e.loop(0, full, DIM, lambda start: tile(start, DIM))
    if full < count:
        tile(full, count - full)
    return True
