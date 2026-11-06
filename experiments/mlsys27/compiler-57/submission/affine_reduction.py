"""Device lowering for exact affine maps and sums.

All tensor values cross DMA paths; host operations construct addresses and issue
commands only. Integer coefficients use signed-byte diagonal contractions, so
negative coefficients never negate the -128 input on the saturating load path.
"""
from fractions import Fraction
from itertools import product
from math import prod, gcd, lcm, isfinite
from functools import lru_cache
import struct

DIM = 16
ACC = 0x80000000
ADD = 0xC0000000
FULL = 0xA0000000
ONE = 0x3F800000


def _dtype(e, name):
    return e.tensors[name]['dtype']


def _width(dtype):
    return {'i8': 1, 'int8': 1, 'i32': 4, 'int32': 4}.get(dtype)


def _pack(addr, rows, cols):
    return addr | (cols << 32) | (rows << 48)


def _ldconfig(e, stride, dtype='i8'):
    # CONFIG_LD: byte operands widened into accumulator use shrink=1;
    # full i32 accumulator seeds must retain the unshrunk load state.
    shrink = 1 if _width(dtype) == 1 else 0
    e.issue(0, (ONE << 32) | (DIM << 16) | (1 << 8) | (shrink << 2) | 1, stride)


def _load(e, name, offset, rows, cols, *, accumulate=True, stride=0):
    dtype = _dtype(e, name)
    _ldconfig(e, stride, dtype)
    e.issue(2, e.ptr(name, e.mul(offset, _width(dtype))), _pack(ADD if accumulate else ACC, rows, cols))
    # Overlapping accumulator loads need completion, not just issue order.
    e.fence()


def _zero(e, rows, cols):
    _ldconfig(e, 0, 'i32')
    e.issue(2, 0, _pack(ACC, rows, cols))
    e.fence()


def _store(e, name, offset, rows, cols, stride, scale=Fraction(1), relu=False):
    dtype = _dtype(e, name)
    bits = struct.unpack('<I', struct.pack('<f', float(scale)))[0]
    e.issue(0, 2 | (4 if relu else 0), e.add(e.mul(bits, 1 << 32), stride))
    e.issue(3, e.ptr(name, e.mul(offset, _width(dtype))),
            _pack(FULL if _width(dtype) == 4 else ACC, rows, cols))
    e.fence()


def _scale(attrs):
    if 'scale_numerator' in attrs:
        return Fraction(int(attrs['scale_numerator']), int(attrs['scale_denominator']))
    return Fraction(str(attrs.get('readout_scale', attrs.get('acc_scale', 1))))


def _admit_readout(e, dst, scale, bound, attrs):
    width = _width(_dtype(e, dst))
    if width == 4:
        return scale == 1 and attrs.get('zero_point', 0) == 0
    if width != 1 or attrs.get('zero_point', 0) != 0:
        return False
    if attrs.get('quant_min', -128) != -128 or attrs.get('quant_max', 127) != 127:
        return False
    if attrs.get('readout_semantics') == 'f32_product_rne':
        return scale > 0 and isfinite(float(scale))
    return _prove_readout(scale, bound)



def _f32(x):
    try:
        return struct.unpack('<f', struct.pack('<f', float(x)))[0]
    except OverflowError:
        return float('inf') if x > 0 else -float('inf')


def _sat_rne(x):
    if x >= 127:
        return 127
    if x <= -128:
        return -128
    return round(x)


@lru_cache(maxsize=4096)
def _prove_readout(scale, bound):
    """Prove equality over every integer in [-bound,bound].

    Both positive-scale byte quantizers are monotone. Their only possible changes
    are the 255 boundaries between consecutive byte codes. For each exact
    rational transition, checking its neighboring integers establishes the same
    transition in the complete float32 conversion/product/RNE pipeline. Endpoints
    cover a bounded interval with no internal transition. This is a domain proof,
    never a sample of runtime tensor values.
    """
    if scale <= 0 or not isfinite(float(scale)) or bound < 0:
        return False
    multiplier = _f32(scale)
    if not isfinite(multiplier) or multiplier <= 0:
        return False
    candidates = {-bound, bound}
    for code in range(-127, 128):
        boundary = Fraction(2 * code - 1, 2) / scale
        # At a tie, an even destination code owns the boundary.
        first = -((-boundary.numerator) // boundary.denominator) if code % 2 == 0 else boundary.numerator // boundary.denominator + 1
        if -bound <= first <= bound:
            candidates.add(first)
        if -bound <= first - 1 <= bound:
            candidates.add(first - 1)
    for n in candidates:
        exact = _sat_rne(n * scale)
        device = _sat_rne(_f32(_f32(n) * multiplier))
        if exact != device:
            return False
    return True


def _constant(e, state, values, dtype):
    serial = state.get('_affine_constant_serial', 0)
    state['_affine_constant_serial'] = serial + 1
    return e.global_data('_affine_const_' + str(serial), values, dtype)


def _weighted(e, name, offset, n, weight, state):
    if weight == 1:
        _load(e, name, offset, 1, n)
        return
    while weight:
        part = min(weight, 127) if weight > 0 else max(weight, -127)
        weight -= part
        diagonal = [part if i == j else 0 for i in range(n) for j in range(n)]
        key = ('_diagonal', n, part)
        tensor = state.get(key)
        if tensor is None:
            tensor = _constant(e, state, diagonal, 'i8')
            state[key] = tensor
        # WS, untransposed A/B, row strides one, unit scale.
        e.issue(0, (ONE << 32) | (1 << 16) | 4, 1 << 48)
        _ldconfig(e, n, 'i8')
        e.issue(2, e.ptr(name, offset), _pack(0, 1, n))
        e.fence()
        _ldconfig(e, n, 'i8')
        e.issue(2, e.ptr(tensor), _pack(DIM, n, n))
        e.fence()
        e.issue(6, _pack(DIM, n, n), _pack(ADD, 1, n))
        e.issue(4, _pack(0, 1, n), _pack(0xFFFFFFFF, 1, n))
        e.fence()


def _affine(e, c, state):
    o, a = c['operands'], dict(c.get('attributes', {}))
    dst = o['dst']
    if 'acc_scale_tensor' in a:
        return False
    srcs = [o[k] for k in ('x', 'y') if k in o]
    weights = [int(x) for x in a.get('weights', [1] * len(srcs))]
    if len(weights) != len(srcs) or any(abs(x) > 381 for x in weights):
        return False
    if any(_width(_dtype(e, x)) != 1 for x in srcs):
        return False
    shape = e.tensors[dst]['shape']
    count = prod(shape)
    if not count:
        return False
    counts = [prod(e.tensors[s]['shape']) for s in srcs]
    period = min(counts) if counts else count
    if any(count % x for x in counts) or any(x not in (count, period) for x in counts):
        return False
    for src in srcs:
        ss = e.tensors[src]['shape']
        if len(ss) > len(shape) or (ss and list(shape[-len(ss):]) != list(ss)):
            return False
    operand_scales = a.get('operand_scales')
    if operand_scales is not None:
        if not a.get('per_operand_rounding') or len(operand_scales) != len(srcs) or any(w != 1 for w in weights):
            return False
        if any(not isfinite(float(v)) or not 0 < float(v) <= 1 for v in operand_scales):
            return False
    constant = int(a.get('constant', 0))
    scale = _scale(a)
    relu = a.get('activation', 'none') == 'relu'
    column_scales = a.get('acc_scale_by_column')
    if column_scales is not None:
        columns = shape[-1] if shape else 1
        if not isinstance(column_scales, list) or len(column_scales) != columns or period % columns:
            return False
        if any(k in a for k in ('acc_scale', 'readout_scale', 'scale_numerator', 'scale_denominator')):
            return False
        if any(not isinstance(s, (int, float)) or not isfinite(s) for s in column_scales):
            return False
        scales = [Fraction(s) for s in column_scales]
    else:
        columns = None
        scales = [scale]
    if scale < 0 and column_scales is None:
        if relu:
            return False
        weights = [-x for x in weights]
        constant = -constant
        scale = -scale
        scales = [scale]
    bound = abs(constant) + sum(abs(w) * 128 for w in weights)
    if bound >= 1 << 31 or any(not _admit_readout(e, dst, s, bound, a) for s in scales):
        return False
    widened_relu = _width(_dtype(e, dst)) == 4 and relu
    if widened_relu:
        # Raw i32 MVOUT has no activation. A byte ReLU followed by exact byte
        # widening is identical iff the complete positive interval fits int8.
        upper = constant + sum(w * (127 if w >= 0 else -128) for w in weights)
        if upper > 127 or any(s != 1 for s in scales):
            return False
        temporary = state.get('_relu_widen_scratch')
        if temporary is None:
            temporary = '_relu_widen_scratch'
            e.tensors[temporary] = {'shape': [DIM], 'dtype': 'i8', 'role': 'intermediate'}
            e.ptr(temporary)
            state['_relu_widen_scratch'] = temporary
    seed = _constant(e, state, [constant] * DIM, 'i32') if constant else None

    def tile(outer, inner, n, multiplier):
        offset = e.add(e.mul(outer, period), inner)
        if seed is None:
            _zero(e, 1, n)
        else:
            _load(e, seed, 0, 1, n, accumulate=False)
        for source_index, (src, w, size) in enumerate(zip(srcs, weights, counts)):
            source_offset = inner if size == period else offset
            if operand_scales is None:
                _weighted(e, src, source_offset, n, w, state)
            else:
                load_bits = struct.unpack('<I', struct.pack('<f', float(operand_scales[source_index])))[0]
                e.issue(0, (load_bits << 32) | (DIM << 16) | 5, 0)
                e.issue(2, e.ptr(src, source_offset), _pack(ADD, 1, n))
                e.fence()
        if widened_relu:
            _store(e, temporary, 0, 1, n, n, Fraction(1), True)
            _load(e, temporary, 0, 1, n, accumulate=False)
            _store(e, dst, offset, 1, n, n * 4)
        else:
            _store(e, dst, offset, 1, n, n * _width(_dtype(e, dst)), multiplier, relu)

    parts = []
    if column_scales is not None:
        start = 0
        while start < columns:
            bits = struct.pack('<f', float(scales[start]))
            stop = start + 1
            while stop < columns and struct.pack('<f', float(scales[stop])) == bits:
                stop += 1
            parts.append((start, stop, scales[start]))
            start = stop

    def outer_body(outer):
        if column_scales is None:
            full = period // DIM * DIM
            if full:
                e.loop(0, full, DIM, lambda inner: tile(outer, inner, DIM, scale))
            if full < period:
                tile(outer, full, period - full, scale)
        else:
            def row_body(row):
                for start, stop, multiplier in parts:
                    full = (stop - start) // DIM * DIM + start
                    if full > start:
                        e.loop(start, full, DIM, lambda col: tile(outer, e.add(row, col), DIM, multiplier))
                    if full < stop:
                        tile(outer, e.add(row, full), stop - full, multiplier)
            e.loop(0, period, columns, row_body)
    e.loop(0, count // period, 1, outer_body)
    return True


def _bias(e, c, state):
    o, a = c['operands'], c.get('attributes', {})
    src, bias, dst = o['src'], o['bias'], o['dst']
    if any(_width(_dtype(e, n)) != 4 for n in (src, bias, dst)):
        return False
    shape = e.tensors[dst]['shape']
    if len(shape) != 2 or e.tensors[bias]['shape'] != [shape[1]]:
        return False
    m, n = shape
    def rows(i):
        def tile(j, cols):
            offset = e.add(e.mul(i, n), j)
            _load(e, src, offset, 1, cols, accumulate=False)
            _load(e, bias, j, 1, cols)
            _store(e, dst, offset, 1, cols, cols * 4)
        full = n // DIM * DIM
        if full:
            e.loop(0, full, DIM, lambda j: tile(j, DIM))
        if full < n:
            tile(full, n-full)
    e.loop(0, m, 1, rows)
    return True


def _reduce(e, c, state):
    o, a = c['operands'], c.get('attributes', {})
    src, dst = o['src'], o['dst']
    if 'acc_scale_tensor' in a or 'acc_scale_by_column' in a:
        return False
    ss, ds = e.tensors[src]['shape'], e.tensors[dst]['shape']
    rank = len(ss)
    dims = sorted({int(d) % rank for d in a.get('dims', list(range(rank)))}) if rank else []
    if _width(_dtype(e, src)) != 1 or not dims:
        return False
    parallel = [d for d in range(rank) if d not in dims]
    output_extents = [ss[d] for d in parallel]
    if prod(output_extents) != prod(ds):
        return False
    reduction_count = prod(ss[d] for d in dims)
    scale = _scale(a) / (reduction_count if a.get('average', False) else 1)
    initial = int(a.get('initial', 0))
    bound = reduction_count * 128 + abs(initial)
    if bound >= 1 << 31 or not _admit_readout(e, dst, scale, bound, a):
        return False
    strides = [prod(ss[i+1:]) for i in range(rank)]
    out_strides = [prod(output_extents[i+1:]) for i in range(len(parallel))]
    seed = _constant(e, state, [initial] * DIM, 'i32') if initial else None
    contiguous = bool(parallel and parallel[-1] == rank - 1)
    vector_axis = parallel[-1] if contiguous else None
    outer_axes = parallel[:-1] if contiguous else parallel

    def emit_tile(src_offset, dst_offset, cols):
        if seed is None:
            _zero(e, 1, cols)
        else:
            _load(e, seed, 0, 1, cols, accumulate=False)
        def red(axis_index, address):
            if axis_index == len(dims):
                _load(e, src, address, 1, cols)
                return
            axis = dims[axis_index]
            e.loop(0, ss[axis], 1,
                   lambda iv: red(axis_index+1, e.add(address, e.mul(iv, strides[axis]))))
        red(0, src_offset)
        _store(e, dst, dst_offset, 1, cols, cols * _width(_dtype(e, dst)), scale)

    def outer(i, source_offset, dest_offset):
        if i == len(outer_axes):
            if vector_axis is None:
                emit_tile(source_offset, dest_offset, 1)
                return
            width = ss[vector_axis]
            full = width // DIM * DIM
            if full:
                e.loop(0, full, DIM, lambda iv: emit_tile(e.add(source_offset, iv), e.add(dest_offset, iv), DIM))
            if full < width:
                emit_tile(e.add(source_offset, full), e.add(dest_offset, full), width-full)
            return
        axis = outer_axes[i]
        out_stride = out_strides[parallel.index(axis)]
        e.loop(0, ss[axis], 1, lambda iv: outer(i+1, e.add(source_offset, e.mul(iv, strides[axis])),
                                              e.add(dest_offset, e.mul(iv, out_stride))))
    outer(0, 0, 0)
    return True


def _prefix(e, c, state):
    o, a = c['operands'], c.get('attributes', {})
    src, dst = o['src'], o['dst']
    ss = e.tensors[src]['shape']
    if e.tensors[dst]['shape'] != ss or not ss:
        return False
    axis = int(a.get('axis', 0)) % len(ss)
    if _width(_dtype(e, src)) != 1 or _width(_dtype(e, dst)) != 4:
        return False
    initial = int(a.get('initial', 0))
    if ss[axis] * 128 + abs(initial) >= 1 << 31:
        return False
    # Keep a vector of prefix accumulators on the device. Each axis step adds
    # its byte row to the preceding accumulator without a DRAM round trip.
    inner = prod(ss[axis+1:])
    outer_count = prod(ss[:axis])
    seed = _constant(e, state, [initial] * DIM, 'i32') if initial else None
    def group(outer_index):
        def tile(inner_index, cols):
            if seed is None:
                _zero(e, 1, cols)
            else:
                _load(e, seed, 0, 1, cols, accumulate=False)
            def step(iv):
                offset = e.add(e.mul(e.add(e.mul(outer_index, ss[axis]), iv), inner), inner_index)
                _load(e, src, offset, 1, cols)
                _store(e, dst, offset, 1, cols, cols * 4)
            e.loop(0, ss[axis], 1, step)
        full = inner // DIM * DIM
        if full:
            e.loop(0, full, DIM, lambda iv: tile(iv, DIM))
        if full < inner:
            tile(full, inner-full)
    e.loop(0, outer_count, 1, group)
    return True


def lower(e, command, state):
    opcode = command['opcode']
    if opcode == 'AFFINE_MAP':
        return _affine(e, command, state)
    if opcode == 'BIAS_ADD':
        return _bias(e, command, state)
    if opcode == 'RESIDUAL_ADD':
        o, a = command['operands'], command.get('attributes', {})
        ls, rs = Fraction(str(a['lhs_scale'])), Fraction(str(a['rhs_scale']))
        if ls.denominator != 1 or rs.denominator != 1:
            return False
        mapped = {'opcode': 'AFFINE_MAP', 'operands': {'x': o['lhs'], 'y': o['rhs'], 'dst': o['dst']},
                  'attributes': {'weights': [int(ls), int(rs)], 'constant': 0,
                                 'activation': 'relu' if 'relu' in a.get('epilogue', []) else 'none'}}
        return _affine(e, mapped, state)
    if opcode == 'WINDOW_SUM' and command.get('attributes', {}).get('mode') == 'reduce':
        return _reduce(e, command, state)
    if opcode == 'PREFIX_SUM' or (opcode == 'WINDOW_SUM' and command.get('attributes', {}).get('mode') == 'prefix'):
        return _prefix(e, command, state)
    if opcode == 'WINDOW_SUM' and command.get('attributes', {}).get('mode') in ('window', 'adaptive', 'window_transpose', 'adaptive_transpose', 'fold'):
        return _windows(e, command, state)
    return False


def _window_plan(in_shape, out_shape, attrs):
    """Source-derived sparse sum map: (terms, final effective multiplier).

    Only tensor geometry is enumerated here. Every value is still fetched by
    device DMA. Identical affine runs are compressed before instruction emission.
    """
    mode = attrs['mode']
    ndim = len(in_shape)
    scale = _scale(attrs)
    kernel = list(attrs.get('kernel', [1] * ndim))
    stride = list(attrs.get('stride', [1] * ndim))
    padding = list(attrs.get('padding', [0] * ndim))
    dilation = list(attrs.get('dilation', [1] * ndim))
    if not all(len(x) == ndim for x in (kernel, stride, padding, dilation)):
        return None
    if any(x <= 0 for x in kernel + stride + dilation):
        return None
    istrides = [prod(in_shape[i+1:]) for i in range(ndim)]
    transpose = mode in ('window_transpose', 'adaptive_transpose')
    pooled_shape, image_shape = (in_shape, out_shape) if transpose else (out_shape, in_shape)
    windows = []
    for o in product(*(range(d) for d in pooled_shape)):
        if mode in ('adaptive', 'adaptive_transpose'):
            lows = [o[i] * image_shape[i] // pooled_shape[i] for i in range(ndim)]
            highs = [((o[i] + 1) * image_shape[i] + pooled_shape[i] - 1) // pooled_shape[i] for i in range(ndim)]
            axes = [list(range(lo, hi)) for lo, hi in zip(lows, highs)]
            count = prod(len(a) for a in axes)
        else:
            raw_axes = [[o[i] * stride[i] - padding[i] + t * dilation[i] for t in range(kernel[i])]
                        for i in range(ndim)]
            axes = [[j for j in raw_axes[i] if 0 <= j < image_shape[i]] for i in range(ndim)]
            if attrs.get('divisor_override') is not None:
                count = int(attrs['divisor_override'])
            elif attrs.get('count_include_pad', False):
                count = prod(sum(-padding[i] <= j < image_shape[i] + padding[i] for j in raw_axes[i])
                             for i in range(ndim))
            else:
                count = prod(len(a) for a in axes)
        if count <= 0:
            return None
        windows.append((list(product(*axes)), count))
    if not transpose:
        result = []
        for coords, count in windows:
            terms = [(sum(c * s for c, s in zip(coord, istrides)), 1) for coord in coords]
            result.append((terms, scale / count))
        return result
    # A transpose sums exact rational contributions before its one readout.
    # Convert all window reciprocals to integer weights under one common LCM.
    denominator = lcm(*(count for _, count in windows))
    result_terms = [[] for _ in range(prod(out_shape))]
    ostrides = [prod(out_shape[i+1:]) for i in range(ndim)]
    for i, (coords, count) in enumerate(windows):
        weight = denominator // count
        for coord in coords:
            target = sum(c * s for c, s in zip(coord, ostrides))
            result_terms[target].append((i, weight))
    return [(terms, scale / denominator) for terms in result_terms]


def _fold_plan(source_shape, dest_shape, attrs):
    # col2im storage is [N, C*prod(kernel), L], independent of layout spelling.
    if len(source_shape) != 3 or len(dest_shape) < 3:
        return None
    image = dest_shape[2:]
    ndim = len(image)
    kernel = attrs.get('kernel', [])
    stride = attrs.get('stride', [1] * ndim)
    padding = attrs.get('padding', [0] * ndim)
    dilation = attrs.get('dilation', [1] * ndim)
    if not all(len(x) == ndim for x in (kernel, stride, padding, dilation)):
        return None
    lattice = [(image[i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1
               for i in range(ndim)]
    if any(d <= 0 for d in lattice) or prod(lattice) != source_shape[-1]:
        return None
    if source_shape[0] != dest_shape[0] or source_shape[1] != dest_shape[1] * prod(kernel):
        return None
    lst = [prod(lattice[i+1:]) for i in range(ndim)]
    output = []
    for coord in product(*(range(d) for d in image)):
        terms = []
        for tap_index, tap in enumerate(product(*(range(k) for k in kernel))):
            shifted = [coord[i] + padding[i] - tap[i] * dilation[i] for i in range(ndim)]
            if any(shifted[i] % stride[i] for i in range(ndim)):
                continue
            grid = [shifted[i] // stride[i] for i in range(ndim)]
            if all(0 <= grid[i] < lattice[i] for i in range(ndim)):
                terms.append((tap_index * prod(lattice) + sum(c * s for c, s in zip(grid, lst)), 1))
        output.append((terms, _scale(attrs)))
    return output


def _runs(plan):
    """Compress static index maps into affine row runs and repeated run groups."""
    runs = []
    i = 0
    while i < len(plan):
        terms, scale = plan[i]
        signature = tuple(weight for _, weight in terms), scale
        end = i + 1
        delta = None
        if end < len(plan):
            other, other_scale = plan[end]
            if (tuple(w for _, w in other), other_scale) == signature:
                trial = tuple(b[0] - a[0] for a, b in zip(terms, other))
                if all(d >= 0 for d in trial):
                    delta = trial
        if delta is not None:
            while end < len(plan):
                other, other_scale = plan[end]
                if (tuple(w for _, w in other), other_scale) != signature:
                    break
                if any(other[t][0] != terms[t][0] + (end-i) * delta[t] for t in range(len(terms))):
                    break
                end += 1
        else:
            delta = (0,) * len(terms)
        runs.append((i, end-i, tuple(off for off, _ in terms), delta, signature))
        i = end
    buckets = {}
    for run in runs:
        buckets.setdefault((run[1], run[3], run[4]), []).append(run)
    groups = []
    for records in buckets.values():
        i = 0
        while i < len(records):
            first = records[i]
            end = i+1
            delta = None
            if end < len(records):
                other = records[end]
                delta = (other[0]-first[0], tuple(b-a for a,b in zip(first[2], other[2])))
                if any(d < 0 for d in delta[1]):
                    delta = None
            if delta is not None:
                while end < len(records):
                    other = records[end]
                    if other[0] != first[0] + (end-i)*delta[0] or any(
                        other[2][t] != first[2][t] + (end-i)*delta[1][t] for t in range(len(first[2]))):
                        break
                    end += 1
            else:
                delta = (0, (0,) * len(first[2]))
            groups.append((first, end-i, delta))
            i=end
    return groups


def _weighted_block(e, src, offset, rows, cols, stride, weight, state):
    if weight == 1:
        _load(e, src, offset, rows, cols, stride=stride)
        return
    # Positive integer weights multiply in the mesh before adding to i32.
    # No byte load multiplier can saturate away any information.
    if _width(_dtype(e, src)) != 1:
        raise ValueError('weighted i32 source requires a separate proven component construction')
    while weight:
        part = min(weight, 127)
        weight -= part
        key = ('_diagonal', cols, part)
        tensor = state.get(key)
        if tensor is None:
            tensor = _constant(e, state,
                               [part if i == j else 0 for i in range(cols) for j in range(cols)], 'i8')
            state[key] = tensor
        e.issue(0, (ONE << 32) | (1 << 16) | 4, 1 << 48)
        _ldconfig(e, stride, 'i8')
        e.issue(2, e.ptr(src, offset), _pack(0, rows, cols))
        e.fence()
        _ldconfig(e, cols, 'i8')
        e.issue(2, e.ptr(tensor), _pack(DIM, cols, cols))
        e.fence()
        e.issue(6, _pack(DIM, cols, cols), _pack(ADD, rows, cols))
        e.issue(4, _pack(0, rows, cols), _pack(0xFFFFFFFF, rows, cols))
        e.fence()


def _windows(e, c, state):
    o, a = c['operands'], c.get('attributes', {})
    src, dst = o['src'], o['dst']
    if 'acc_scale_tensor' in a or 'acc_scale_by_column' in a:
        return False
    ss, ds = e.tensors[src]['shape'], e.tensors[dst]['shape']
    if len(ss) < 3 or len(ds) < 3 or ss[0] != ds[0]:
        return False
    if a['mode'] != 'fold' and len(ds) != len(ss):
        return False
    sw, dw = _width(_dtype(e, src)), _width(_dtype(e, dst))
    if sw not in (1, 4) or dw not in (1, 4):
        return False
    fold = a['mode'] == 'fold'
    layout = a.get('layout', 'nhwc')
    if layout not in ('nhwc', 'nchw'):
        return False
    nchw = fold or layout == 'nchw'
    if fold:
        plan = _fold_plan(ss, ds, a)
        channels = ds[1]
        invol = prod(ss[1:]) // channels
        outvol = prod(ds[2:])
    else:
        channels = ss[1] if nchw else ss[-1]
        if (ds[1] if nchw else ds[-1]) != channels:
            return False
        in_spatial = ss[2:] if nchw else ss[1:-1]
        out_spatial = ds[2:] if nchw else ds[1:-1]
        plan = _window_plan(in_spatial, out_spatial, a)
        invol, outvol = prod(in_spatial), prod(out_spatial)
    if plan is None or len(plan) != outvol:
        return False
    input_bound = 128 if sw == 1 else int(a.get('input_bound', e.tensors[src].get('value_bound', 0)))
    if input_bound <= 0:
        return False
    for terms, scale in plan:
        bound = sum(abs(weight) for _, weight in terms) * input_bound
        if bound >= 1 << 31 or not _admit_readout(e, dst, scale, bound, a):
            return False
        if sw == 4 and any(w != 1 for _, w in terms):
            return False
    groups = _runs(plan)

    def batch(nc):
        source_outer = e.mul(nc, invol) if nchw else e.mul(nc, invol * channels)
        dest_outer = e.mul(nc, outvol) if nchw else e.mul(nc, outvol * channels)
        def channel_tile(channel, cols):
            for first, repetitions, group_delta in groups:
                outstart, length, source_starts, row_steps, signature = first
                weights, scale = signature
                def repeat(g):
                    dest_start = e.add(outstart, e.mul(g, group_delta[0]))
                    def tile(row, rows):
                        _zero(e, rows, cols)
                        for t, (source_start, row_step, weight) in enumerate(zip(source_starts, row_steps, weights)):
                            local = e.add(e.add(source_start, e.mul(g, group_delta[1][t])), e.mul(row, row_step))
                            source = e.add(source_outer, local if nchw else e.add(e.mul(local, channels), channel))
                            _weighted_block(e, src, source, rows, cols,
                                            row_step * (1 if nchw else channels) * sw, weight, state)
                        outindex = e.add(dest_start, row)
                        dest = e.add(dest_outer, outindex if nchw else e.add(e.mul(outindex, channels), channel))
                        _store(e, dst, dest, rows, cols, (1 if nchw else channels) * dw, scale,
                               a.get('activation') == 'relu' or 'relu' in a.get('epilogue', []))
                    full = length // DIM * DIM
                    if full:
                        e.loop(0, full, DIM, lambda row: tile(row, DIM))
                    if full < length:
                        tile(full, length-full)
                e.loop(0, repetitions, 1, repeat)
        if nchw:
            channel_tile(0, 1)
        else:
            full = channels // DIM * DIM
            if full:
                e.loop(0, full, DIM, lambda channel: channel_tile(channel, DIM))
            if full < channels:
                channel_tile(full, channels-full)
    e.loop(0, ss[0] * channels if nchw else ss[0], 1, batch)
    return True
