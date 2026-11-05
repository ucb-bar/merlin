"""Tensor movement lowering through the accelerator's DMA datapaths.

Index arithmetic below acts only on coordinates. Tensor data is never loaded into
host registers: even irregular gathers issue a device load and device readout.
"""
from fractions import Fraction
import math
import struct

from device import transfer


def _prod(shape):
    return math.prod(shape)


def _width(dtype):
    return {'i8': 1, 'int8': 1, 'i32': 4, 'int32': 4}.get(dtype)


def _dtype(dtype):
    return {'int8': 'i8', 'int32': 'i32'}.get(dtype, dtype)


def _coords(g, index, shape):
    """Unflatten a C-order index, with no dependence on tensor contents."""
    result = []
    stride = _prod(shape)
    for dim in shape:
        stride //= dim
        result.append(g.urem(g.udiv(index, stride), dim))
    return result


def _linear(g, coords, shape):
    index = 0
    for coord, dim in zip(coords, shape):
        index = g.add(g.mul(index, dim), coord)
    return index


def _cmp(g, pred, a, b):
    return g.value(f'icmp {pred} i64 {a}, {b}')


def _select(g, pred, a, b, dtype='i64'):
    return g.value(f'select i1 {pred}, {dtype} {a}, {dtype} {b}')


def _clamp_index(g, index, size):
    index = _select(g, _cmp(g, 'slt', index, 0), 0, index)
    return _select(g, _cmp(g, 'sge', index, size), size - 1, index)


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def _scale(attrs, input_dtype, output_dtype):
    """Admit a scale only when its finite input domain proves readout equality."""
    if attrs.get('zero_point', 0) != 0:
        return None
    if attrs.get('rounding', 'half_to_even') not in ('half_to_even', 'half_even'):
        return None
    if 'scale_numerator' in attrs:
        ratio = Fraction(int(attrs['scale_numerator']), int(attrs['scale_denominator']))
    else:
        ratio = Fraction(float(attrs.get('scale', 1.0)))
    if ratio == 1:
        return 1.0
    if input_dtype != 'i8' or output_dtype != 'i8':
        return None
    if attrs.get('quant_min', -128) != -128 or attrs.get('quant_max', 127) != 127:
        return None
    scale = _f32(float(ratio))
    if not math.isfinite(scale):
        return None
    for value in range(-128, 128):
        exact = max(-128, min(127, round(value * ratio)))
        device = max(-128, min(127, round(_f32(value * scale))))
        if exact != device:
            return None
    return scale


def _copy_linear(g, src, dst, count, src_dtype, dst_dtype, scale=1.0):
    src_width, dst_width = _width(src_dtype), _width(dst_dtype)
    full, tail = divmod(count, 16)
    def chunk(iv):
        offset = g.mul(iv, 16)
        transfer(g, g.gep(src, g.mul(offset, src_width)),
                 g.gep(dst, g.mul(offset, dst_width)), 1, 16,
                 16 * src_width, 16 * dst_width,
                 input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)
    g.loop(full, chunk)
    if tail:
        offset = full * 16
        transfer(g, g.gep(src, offset * src_width),
                 g.gep(dst, offset * dst_width), 1, tail,
                 16 * src_width, 16 * dst_width,
                 input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)


def _literal_tile(g, value, dtype):
    """A DMA source row, with no scalar payload initialization."""
    if value == 0:
        return None
    name = '@movement_literal_' + str(g.counter)
    g.counter += 1
    values = ', '.join(f'{dtype} {int(value)}' for _ in range(16))
    g.globals.append(f'{name} = private constant [16 x {dtype}] [{values}], align 64')
    return name


def _fill(g, dst, count, dtype, value):
    literal = _literal_tile(g, value, dtype)
    width = _width(dtype)
    full, tail = divmod(count, 256)
    def chunk(iv):
        offset = g.mul(iv, 256)
        transfer(g, literal, g.gep(dst, g.mul(offset, width)), 16,
                 16, 0, 16 * width,
                 input_dtype=dtype, output_dtype=dtype)
    g.loop(full, chunk)
    if tail:
        rows, columns = divmod(tail, 16)
        if rows:
            transfer(g, literal, g.gep(dst, full * 256 * width), rows,
                     16, 0, 16 * width, input_dtype=dtype, output_dtype=dtype)
        if columns:
            transfer(g, literal, g.gep(dst, (full * 256 + rows * 16) * width), 1,
                     columns, 0, 16 * width, input_dtype=dtype, output_dtype=dtype)


def _movement(g, cb, command, ptrs):
    operands, attrs = command['operands'], command.get('attributes', {})
    src_name, dst_name = operands['src'], operands['dst']
    src_info, dst_info = cb['tensors'][src_name], cb['tensors'][dst_name]
    src_dtype, dst_dtype = _dtype(src_info['dtype']), _dtype(dst_info['dtype'])
    src_width, dst_width = _width(src_dtype), _width(dst_dtype)
    if not src_width or not dst_width or (src_dtype == 'i32' and dst_dtype != 'i32'):
        return False
    scale = _scale(attrs, src_dtype, dst_dtype)
    if scale is None:
        return False
    src_shape, dst_shape = src_info['shape'], dst_info['shape']
    src, dst = ptrs[src_name], ptrs[dst_name]
    form = attrs.get('form', 'identity')
    if form in ('identity', 'reshape'):
        if _prod(src_shape) != _prod(dst_shape):
            return False
        _copy_linear(g, src, dst, _prod(dst_shape), src_dtype, dst_dtype, scale)
        return True
    allowed = ('permute', 'slice', 'pad', 'broadcast', 'repeat', 'upsample', 'as_strided')
    if form not in allowed:
        return False
    rank = len(src_shape)
    if form == 'permute':
        permutation = attrs.get('perm', [])
        if sorted(permutation) != list(range(rank)):
            return False
    if form in ('slice', 'pad', 'upsample') and rank != len(dst_shape):
        return False
    if form == 'slice' and (len(attrs.get('offsets', [])) != rank or len(attrs.get('strides', [])) != rank):
        return False
    if form == 'upsample' and len(attrs.get('scales', [])) != rank:
        return False
    if form == 'pad':
        low, high = attrs.get('low', []), attrs.get('high', [])
        mode = attrs.get('mode', 'constant')
        if len(low) != rank or len(high) != rank or mode not in ('constant', 'reflect', 'replicate'):
            return False
        if mode == 'reflect' and any((lo or hi) and (dim <= 1 or lo >= dim or hi >= dim)
                                     for dim, lo, hi in zip(src_shape, low, high)):
            return False
    if form == 'as_strided' and len(attrs.get('strides', [])) != len(dst_shape):
        return False
    if form in ('broadcast', 'repeat'):
        if rank > len(dst_shape):
            return False
        aligned_shape = [1] * (len(dst_shape) - rank) + list(src_shape)
        if form == 'broadcast' and any(s != d and s != 1 for s, d in zip(aligned_shape, dst_shape)):
            return False
    # Each output axis is a sequence of affine source segments. Iterating
    # those segments directly keeps coordinates in induction variables, and
    # never unflattens a scalar element counter with division or remainder.
    source_strides = [_prod(src_shape[i + 1:]) for i in range(rank)]
    destination_strides = [_prod(dst_shape[i + 1:]) for i in range(len(dst_shape))]
    if form == 'permute':
        axis_strides = [source_strides[i] for i in attrs['perm']]
    elif form in ('broadcast', 'repeat'):
        axis_strides = [_prod(aligned_shape[i + 1:]) for i in range(len(dst_shape))]
    elif form == 'as_strided':
        axis_strides = list(attrs['strides'])
    else:
        axis_strides = source_strides
    literal_value = int(attrs.get('value_code', 0))
    literal = _literal_tile(g, literal_value, src_dtype) if form == 'pad' and attrs.get('mode', 'constant') == 'constant' else None

    def add(a, b):
        return a if b == 0 else b if a == 0 else g.add(a, b)

    def mul(a, b):
        return 0 if a == 0 or b == 0 else a if b == 1 else b if a == 1 else g.mul(a, b)

    def run(source_index, output_index, count, source_step, filling):
        output = g.gep(dst, mul(output_index, dst_width))
        if filling:
            # Replay the immutable row (or the unit's zero-load source).
            full, tail = divmod(count, 16)
            def chunk(iv):
                transfer(g, literal, g.gep(output, mul(iv, 16 * dst_width)),
                         1, 16, 0, 16 * dst_width,
                         input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)
            g.loop(full, chunk)
            if tail:
                transfer(g, literal, g.gep(output, full * 16 * dst_width),
                         1, tail, 0, 16 * dst_width,
                         input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)
            return
        if source_step < 0:
            # DMA strides are unsigned. A reversed boundary is expressed by
            # individual source rows; short reflected edges have static sites.
            def reversed_element(iv):
                source = g.gep(src, mul(add(source_index, mul(iv, source_step)), src_width))
                transfer(g, source, g.gep(output, mul(iv, dst_width)),
                         1, 1, src_width, dst_width,
                         input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)
            if count <= 16:
                for i in range(count):
                    reversed_element(i)
            else:
                g.loop(count, reversed_element)
            return
        full, tail = divmod(count, 16)
        def chunk(iv, length=16):
            start = mul(iv, 16)
            source = g.gep(src, mul(add(source_index, mul(start, source_step)), src_width))
            destination = g.gep(output, mul(start, dst_width))
            if source_step == 1:
                transfer(g, source, destination, 1, length,
                         16 * src_width, 16 * dst_width,
                         input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)
            else:
                transfer(g, source, destination, length, 1,
                         source_step * src_width, dst_width,
                         input_dtype=src_dtype, output_dtype=dst_dtype, scale=scale)
        g.loop(full, chunk)
        if tail:
            chunk(full, tail)

    def axis(axis_index, source_base, destination_base, filling=False):
        if axis_index == len(dst_shape):
            run(source_base, destination_base, 1, 1, filling)
            return
        size = dst_shape[axis_index]
        source_stride = axis_strides[axis_index]
        destination_stride = destination_strides[axis_index]

        def segment(output_start, length, source_start, source_step=1, pad=False):
            if length <= 0:
                return
            sb = add(source_base, mul(source_start, source_stride))
            db = add(destination_base, mul(output_start, destination_stride))
            pad = filling or pad
            if axis_index == len(dst_shape) - 1:
                run(sb, db, length, source_step * source_stride, pad)
                return
            def next_coordinate(iv):
                axis(axis_index + 1,
                     add(sb, mul(iv, source_step * source_stride)),
                     add(db, mul(iv, destination_stride)), pad)
            g.loop(length, next_coordinate)

        if filling:
            segment(0, size, 0, 0, True)
        elif form == 'slice':
            segment(0, size, attrs['offsets'][axis_index], attrs['strides'][axis_index])
        elif form == 'broadcast':
            segment(0, size, 0, 0 if aligned_shape[axis_index] == 1 else 1)
        elif form == 'repeat':
            extent = aligned_shape[axis_index]
            full, tail = divmod(size, extent)
            g.loop(full, lambda repetition: segment(mul(repetition, extent), extent, 0))
            if tail:
                segment(full * extent, tail, 0)
        elif form == 'upsample':
            step = attrs['scales'][axis_index]
            if not isinstance(step, int) or step <= 0 or size != src_shape[axis_index] * step:
                raise ValueError('nearest upsample requires exact positive integral factors')
            g.loop(src_shape[axis_index], lambda coordinate: segment(mul(coordinate, step), step, coordinate, 0))
        elif form == 'pad':
            low, high, extent = attrs['low'][axis_index], attrs['high'][axis_index], src_shape[axis_index]
            mode = attrs.get('mode', 'constant')
            segment(0, low, low if mode == 'reflect' else 0,
                    -1 if mode == 'reflect' else 0, mode == 'constant')
            segment(low, extent, 0)
            segment(low + extent, high, extent - 2 if mode == 'reflect' else extent - 1,
                    -1 if mode == 'reflect' else 0, mode == 'constant')
        else:
            segment(0, size, 0)

    axis(0, int(attrs.get('storage_offset', 0)) if form == 'as_strided' else 0, 0)
    return True


def _concat(g, cb, command, ptrs):
    operands, attrs = command['operands'], command.get('attributes', {})
    tensors = cb['tensors']
    names = [operands[key] for key in sorted((key for key in operands if key.startswith('src') and key[3:].isdigit()), key=lambda key: int(key[3:]))]
    dst_name = operands['dst']
    shape = tensors[dst_name]['shape']
    rank = len(shape)
    axis = int(attrs['axis']) % rank
    dst_dtype = _dtype(tensors[dst_name]['dtype'])
    dst_width = _width(dst_dtype)
    if not names or not dst_width:
        return False
    scales = attrs.get('scales', [1.0] * len(names))
    if len(scales) != len(names):
        return False
    choices = []
    for name, scale in zip(names, scales):
        info = tensors[name]
        src_dtype = _dtype(info['dtype'])
        if not _width(src_dtype) or (src_dtype == 'i32' and dst_dtype != 'i32'):
            return False
        chosen_scale = _scale({'scale': scale}, src_dtype, dst_dtype)
        if chosen_scale is None:
            return False
        choices.append((name, info['shape'], src_dtype, chosen_scale))
    before, after = _prod(shape[:axis]), _prod(shape[axis + 1:])
    axis_start = 0
    for name, src_shape, src_dtype, scale in choices:
        per_outer = src_shape[axis] * after
        src_width = _width(src_dtype)
        output_start = axis_start * after
        def outer(index):
            source_base = g.gep(ptrs[name], g.mul(g.mul(index, per_outer), src_width))
            output_offset = g.add(g.mul(index, shape[axis] * after), output_start)
            output_base = g.gep(ptrs[dst_name], g.mul(output_offset, dst_width))
            _copy_linear(g, source_base, output_base, per_outer, src_dtype, dst_dtype, scale)
        g.loop(before, outer)
        axis_start += src_shape[axis]
    return True


def _insert_slice(g, cb, command, ptrs):
    operands, attrs = command['operands'], command.get('attributes', {})
    tensors = cb['tensors']
    dest_name, src_name, out_name = operands['dest'], operands['src'], operands['dst']
    dest_info, src_info, out_info = tensors[dest_name], tensors[src_name], tensors[out_name]
    dest_dtype, src_dtype, out_dtype = map(_dtype, (dest_info['dtype'], src_info['dtype'], out_info['dtype']))
    if not _width(dest_dtype) or not _width(src_dtype) or not _width(out_dtype):
        return False
    if (src_dtype == 'i32' or dest_dtype == 'i32') and out_dtype != 'i32':
        return False
    rank = len(src_info['shape'])
    if len(attrs.get('offsets', [])) != rank or len(attrs.get('strides', [])) != rank:
        return False
    if dest_info['shape'] != out_info['shape']:
        return False
    _copy_linear(g, ptrs[dest_name], ptrs[out_name], _prod(out_info['shape']), dest_dtype, out_dtype)
    g.fence()
    source_shape = src_info['shape']
    source_strides = [_prod(source_shape[i + 1:]) for i in range(rank)]
    output_strides = [_prod(out_info['shape'][i + 1:]) for i in range(rank)]
    output_base = sum(offset * stride for offset, stride in zip(attrs['offsets'], output_strides))

    def add(a, b):
        return a if b == 0 else b if a == 0 else g.add(a, b)

    def mul(a, b):
        return 0 if a == 0 or b == 0 else a if b == 1 else b if a == 1 else g.mul(a, b)

    def axis(depth, source_index, output_index):
        if depth == rank:
            transfer(g, g.gep(ptrs[src_name], mul(source_index, _width(src_dtype))),
                     g.gep(ptrs[out_name], mul(output_index, _width(out_dtype))),
                     1, 1, _width(src_dtype), _width(out_dtype),
                     input_dtype=src_dtype, output_dtype=out_dtype)
            return
        source_step = source_strides[depth]
        output_step = attrs['strides'][depth] * output_strides[depth]
        size = source_shape[depth]
        if depth == rank - 1 and output_step > 0:
            full, tail = divmod(size, 16)
            def chunk(iv, length=16):
                first = mul(iv, 16)
                source = g.gep(ptrs[src_name], mul(add(source_index, first), _width(src_dtype)))
                destination = g.gep(ptrs[out_name], mul(add(output_index, mul(first, output_step)), _width(out_dtype)))
                if output_step == 1:
                    transfer(g, source, destination, 1, length,
                             16 * _width(src_dtype), 16 * _width(out_dtype),
                             input_dtype=src_dtype, output_dtype=out_dtype)
                else:
                    transfer(g, source, destination, length, 1,
                             _width(src_dtype), output_step * _width(out_dtype),
                             input_dtype=src_dtype, output_dtype=out_dtype)
            g.loop(full, chunk)
            if tail:
                chunk(full, tail)
            return
        g.loop(size, lambda iv: axis(depth + 1,
               add(source_index, mul(iv, source_step)),
               add(output_index, mul(iv, output_step))))

    axis(0, 0, output_base)
    return True


def emit_movement(g, cb, command, ptrs):
    # A full-width DMA preserves all 32 storage bits, including f32 payloads.
    # This applies only to pure movement: scaling retains its declared dtype.
    touched=[n for n in command.get('operands',{}).values() if n in cb['tensors']]
    if touched and all(cb['tensors'][n]['dtype'] in ('f16', 'bf16') for n in touched) and command['opcode'] in ('MOVEMENT', 'CONCAT', 'INSERT_SLICE') and not any(k in command.get('attributes',{}) for k in ('scale','scales','scale_numerator')):
        # Keep the two bytes of a narrow floating storage value adjacent as
        # a trailing byte axis. No accumulator conversion sees its payload.
        attrs = dict(command.get('attributes', {}))
        form = attrs.get('form', 'identity')
        if command['opcode'] == 'MOVEMENT' and form not in ('identity', 'reshape', 'slice', 'permute', 'as_strided', 'broadcast', 'repeat', 'upsample'):
            return False
        shadow = dict(cb)
        shadow['tensors'] = {n: dict(t, dtype='i8', shape=list(t['shape']) + [2])
                             if n in touched else t for n, t in cb['tensors'].items()}
        rank = len(cb['tensors'][command['operands']['dst']]['shape'])
        if command['opcode'] == 'CONCAT':
            attrs['axis'] = int(attrs['axis']) % rank
        elif command['opcode'] == 'INSERT_SLICE' or form == 'slice':
            attrs['offsets'] = list(attrs['offsets']) + [0]
            attrs['strides'] = list(attrs['strides']) + [1]
        elif form == 'permute':
            attrs['perm'] = list(attrs['perm']) + [rank]
        elif form == 'as_strided':
            attrs['strides'] = [2 * stride for stride in attrs['strides']] + [1]
            attrs['storage_offset'] = 2 * int(attrs.get('storage_offset', 0))
        elif form == 'upsample':
            attrs['scales'] = list(attrs['scales']) + [1]
        attrs['output_dtype'] = 'i8'
        return emit_movement(g, shadow, dict(command, attributes=attrs), ptrs)
    if touched and all(cb['tensors'][n]['dtype']=='f32' for n in touched) and command['opcode'] in ('MOVEMENT','CONCAT','INSERT_SLICE') and not any(k in command.get('attributes',{}) for k in ('scale','scales','scale_numerator')):
        shadow=dict(cb);shadow['tensors']={n:dict(t, dtype='i32') if t['dtype']=='f32' else t for n,t in cb['tensors'].items()}
        cmd=dict(command);cmd['attributes']=dict(command.get('attributes',{}),output_dtype='i32')
        return emit_movement(g,shadow,cmd,ptrs)
    """Emit a supported movement command, returning False before unsupported work."""
    opcode = command['opcode']
    if opcode == 'MOVEMENT':
        return _movement(g, cb, command, ptrs)
    if opcode == 'CONCAT':
        return _concat(g, cb, command, ptrs)
    if opcode == 'INSERT_SLICE':
        return _insert_slice(g, cb, command, ptrs)
    if opcode == 'FILL':
        name = command['operands']['dst']
        info = cb['tensors'][name]
        dtype = _dtype(info['dtype'])
        if not _width(dtype):
            return False
        _fill(g, ptrs[name], _prod(info['shape']), dtype, int(command.get('attributes', {})['value_code']))
        return True
    return False
