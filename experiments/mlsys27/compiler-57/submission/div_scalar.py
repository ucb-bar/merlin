"""Device scalar division under the declared scale-2**-6 quantized model.

Exact rational tables encode entry RNE, division and one saturated byte
readout. Runtime predicates and output bytes use DMA and bounded mesh MACs;
no tensor payload is read or computed by the host.
"""
from fractions import Fraction
from math import isfinite, prod
import struct

from device import ACC, ACCUM, config_ld, config_st, descriptor, fbits, matmul, transfer


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def match_quantized_div(record):
    """Recognize a parallel f32 input divided by a finite nonidentity literal."""
    from frontend import _map_dims
    body = record.get('scalar_body')
    if not body or len(record['results']) != 1 or record['results'][0]['dtype'] != 'f32':
        return None
    if any(iterator != 'parallel' for iterator in record['iterator_types']):
        return None
    if any(operation['op'] not in ('arith.constant', 'arith.divf')
           for operation in body['operations']):
        return None
    values = {f'arg:{i}': edge.get('const_value', ('input', i))
              for i, edge in enumerate(record['ins'])}
    values.update({f'capture:{i}': edge.get('const_value')
                   for i, edge in enumerate(body['captures'])})
    division = None
    for i, operation in enumerate(body['operations']):
        if operation['regions'] or operation['effects'] or operation['results'] != ['f32']:
            return None
        if operation['op'] == 'arith.constant':
            token = operation['attributes']['value'].partition(' : ')[0]
            value = (struct.unpack('<f', struct.pack('<I', int(token, 16)))[0]
                     if token.startswith('0x') else _f32(float(token)))
        elif operation['op'] == 'arith.divf' and division is None:
            left, right = [values.get(key) for key in operation['operands']]
            if not isinstance(left, tuple) or type(right) not in (int, float):
                return None
            right = _f32(right)
            if not isfinite(right) or right in (0, 1):
                return None
            division = (left[1], float(right))
            value = ('division', i)
        else:
            return None
        values[f'op:{i}:0'] = value
    if division is None or len(body['yields']) != 1 or values.get(body['yields'][0]) != ('division', i):
        return None
    source, divisor = division
    edge, result = record['ins'][source], record['results'][0]
    if (edge['dtype'] != 'f32' or edge['shape'] != result['shape'] or
            edge['source'][0] != 'arg'):
        return None
    identity = list(range(len(result['shape'])))
    maps = record.get('indexing_maps')
    if not maps or _map_dims(maps[source]) != identity or _map_dims(maps[-1]) != identity:
        return None
    return {'input_index': source, 'divisor': divisor}



def _global(g, size):
    label = '@div_scratch_' + str(g.counter)
    g.counter += 1
    g.globals.append(f'{label} = internal global [{size} x i8] zeroinitializer, align 64')
    return label


def _literal(g, values, dtype='i8'):
    values = tuple(values)
    cache = getattr(g, '_div_literals', None)
    if cache is None:
        cache = g._div_literals = {}
    key = dtype, values
    if key not in cache:
        label = '@div_literal_' + str(g.counter)
        g.counter += 1
        body = ', '.join(f'{dtype} {value}' for value in values)
        g.globals.append(f'{label} = private constant [{len(values)} x {dtype}] [{body}], align 64')
        cache[key] = label
    return cache[key]


def _load(g, ptr, rows, cols=1, stride=1, dtype='i8', add=False, scale=1.0):
    config_ld(g, stride, dtype=dtype, scale=scale)
    g.rocc(2, g.ptrint(ptr), descriptor(g, ACC | (ACCUM if add else 0), rows, cols))
    g.fence()


def _store(g, ptr, rows, cols=1, stride=1, scale=1.0, relu=False):
    config_st(g, stride, scale, relu)
    g.rocc(3, g.ptrint(ptr), descriptor(g, ACC, rows, cols))
    g.fence()


def _normalize_flag(g, ptr, rows, cols=1):
    _load(g, ptr, rows, cols, cols, scale=1.0 / 127)
    _store(g, ptr, rows, cols, cols)


def _linear(g, source, weights, destination, rows, k, n=1, bias=None, flag=False):
    attrs = {'epilogue': ['acc_scale', 'relu'], 'acc_scale': 127.0} if flag else {}
    matmul(g, source, _literal(g, weights), destination, rows, k, n, attrs,
           'i8' if flag else 'i32', bias, n * 4 if bias else 0)
    if flag:
        _normalize_flag(g, destination, rows, n)


def _magnitude(g, source, rows):
    """Return high16(abs(f32)), low16!=0, and sign using bounded mesh sums."""
    signs = _global(g, rows * 4)
    weights = [-int(i == j) for i in range(4) for j in range(4)]
    _linear(g, source, weights, signs, rows, 4, 4, flag=True)
    high_byte = _global(g, rows)
    _load(g, g.gep(source, 3), rows, stride=4)
    _load(g, g.gep(signs, 3), rows, stride=4, add=True, scale=127)
    _load(g, g.gep(signs, 3), rows, stride=4, add=True)
    _store(g, high_byte, rows)
    packed = _global(g, rows * 3)
    for ptr, stride, col in [(g.gep(source, 2), 4, 0), (high_byte, 1, 1),
                             (g.gep(signs, 2), 4, 2)]:
        transfer(g, ptr, g.gep(packed, col), rows, 1, stride, 3)
    high = _global(g, rows * 4)
    _linear(g, packed, [1, 127, 127], high, rows, 3)
    _linear(g, packed, [0, 127, 127], high, rows, 3, bias=high)
    _linear(g, packed, [0, 2, 2], high, rows, 3, bias=high)
    low = _global(g, rows * 4)
    _linear(g, source, [1, 1, 0, 0], low, rows, 4)
    for weight in (127, 127, 2):
        _linear(g, signs, [weight, weight, 0, 0], low, rows, 4, bias=low)
    low_nonzero = _global(g, rows)
    _load(g, low, rows, stride=4, dtype='i32')
    _store(g, low_nonzero, rows, scale=127, relu=True)
    _normalize_flag(g, low_nonzero, rows)
    return high, low_nonzero, g.gep(signs, 3)


def _predicate(g, high, low_nonzero, threshold, inclusive, rows, destination):
    """Compare abs(x) to an entry midpoint, preserving both RNE tie cases."""
    assert threshold & 65535 == 0
    _load(g, high, rows, stride=4, dtype='i32')
    _load(g, high, rows, stride=4, dtype='i32', add=True)
    shift = -2 * (threshold >> 16) + int(inclusive)
    _load(g, _literal(g, [shift], 'i32'), rows, stride=0, dtype='i32', add=True)
    if not inclusive:
        _load(g, low_nonzero, rows, add=True)
    _store(g, destination, rows, scale=127, relu=True)
    _normalize_flag(g, destination, rows)


def _signed_bytes(value):
    word = fbits(value)
    return [((word >> (8 * i) & 255) ^ 128) - 128 for i in range(4)]


def _transitions(divisor):
    """Compile every declared byte code with exact division and one output RNE."""
    divisor = Fraction(divisor)
    previous = [[0] * 4, [0] * 4]
    for code in range(1, 129):
        current = []
        for q in (min(code, 127), -code):
            result = min(127, max(-128, round(Fraction(q) / divisor)))
            current.append(_signed_bytes(Fraction(result, 64)))
        deltas = [[a - b for a, b in zip(row, old)] for row, old in zip(current, previous)]
        previous = current
        if any(any(row) for row in deltas):
            yield fbits(Fraction(2 * code - 1, 128)), code % 2 == 0, deltas


def _delta_weights(deltas):
    weights = []
    for piece in range(3):
        for delta in deltas:
            for value in delta:
                for _ in range(piece):
                    value -= min(127, max(-128, value))
                weights.append(min(127, max(-128, value)))
    return weights


def _divide_tile(g, source, destination, rows, divisor):
    high, low, sign = _magnitude(g, source, rows)
    nan = _global(g, rows)
    _predicate(g, high, low, 0x7f800000, False, rows, nan)
    predicate = _global(g, rows)
    flags = _global(g, rows * 6)
    result = _global(g, rows * 16)
    transfer(g, 0, result, rows, 4, 0, 16, 'i8', 'i32')
    for threshold, inclusive, deltas in _transitions(divisor):
        _predicate(g, high, low, threshold, inclusive, rows, predicate)
        for negative in (False, True):
            _load(g, predicate, rows)
            _load(g, sign, rows, stride=4, add=True, scale=1 if negative else -1)
            _load(g, nan, rows, add=True, scale=-1)
            if negative:
                _load(g, _literal(g, [-1], 'i32'), rows, stride=0, dtype='i32', add=True)
            _store(g, g.gep(flags, int(negative)), rows, stride=6, relu=True)
        for col in (2, 4):
            transfer(g, flags, g.gep(flags, col), rows, 2, 6, 6)
        _linear(g, flags, _delta_weights(deltas), result, rows, 6, 4, bias=result)
    transfer(g, result, destination, rows, 4, 16, 4, 'i32', 'i8')


def emit_quantized_div(g, cb, command, ptrs):
    if command['opcode'] != 'DEVICE_QUANTIZED_DIV':
        return False
    operands, attrs = command['operands'], command['attributes']
    src, dst = operands['src'], operands['dst']
    count = prod(cb['tensors'][dst]['shape'])
    def tile(start, rows):
        _divide_tile(g, g.gep(ptrs[src], g.mul(start, 4)),
                     g.gep(ptrs[dst], g.mul(start, 4)), rows, attrs['divisor'])
    full, tail = divmod(count, 16)
    if full:
        g.loop(full, lambda index: tile(g.mul(index, 16), 16))
    if tail:
        tile(full * 16, tail)
    placement = {'lane': 'gemmini_mesh', 'route': 'A', 'region_id': attrs.get('source_region_id'),
                 'op': command['opcode'], 'source_op_index': attrs.get('source_op_index'),
                 'reason': 'declared quantized scalar division by device byte predicates and MAC'}
    cb['params'].setdefault('lane_placement', []).append(placement)
    return True
