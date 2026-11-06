"""Modular integer contractions using bounded byte products and raw i32 DMA.

For little-endian bytes s[j] interpreted as signed integers, the exact identity
is x = sum(s[j]*256**j) + sum((s[j]<0)*256**(j+1), j<3).
Every mesh product has K<=16 and magnitude <=262144. Recombination takes place
in the source i32 ring using full-width accumulator additions and byte shifts;
i8 results select the low storage byte instead of saturating a readout.
"""
from math import prod

from affine_reduction import _load, _zero, _store
from contraction import matmul, _axis
from integer_reduction import low_bytes, temporary


def components(e, src):
    shape = e.tensors[src]['shape']
    if e.tensors[src]['dtype'] == 'i8':
        return [(src, 0)]
    result = []
    for byte in range(4):
        signed = temporary(e, shape, 'i8', '_signed_byte')
        low_bytes(e, src, signed, prod(shape), byte=byte)
        result.append((signed, byte))
        if byte < 3:
            negative = temporary(e, shape, 'i8', '_negative_byte')
            weight = e.global_data('_integer_negative_weight', [-1], 'i8')
            seed = e.global_data('_integer_negative_seed', [128], 'i32')
            # RNE((128-s)/256) is exactly [s<0], including the even tie s=0.
            matmul(e, signed, weight, negative, prod(shape), 1, 1,
                   {'acc_scale': 1/256, 'epilogue': ['acc_scale']}, seed=seed)
            result.append((negative, byte+1))
    return result


def shifted(e, src, count, power):
    if not power:
        return src
    dst = temporary(e, [count], 'i32', '_shifted_component')
    def zero(start, cols):
        _zero(e, 1, cols)
        _store(e, dst, start, 1, cols, cols*4)
    _axis(e, count, zero)
    # Zero extension followed by a byte shift computes multiplication modulo 2^32.
    for byte in range(4-power):
        lane = temporary(e, [count], 'i8', '_shift_byte')
        low_bytes(e, src, lane, count, byte=byte)
        from affine_reduction import ACC, ONE, _pack
        def scatter(start, rows):
            e.issue(0, (ONE << 32) | (16 << 16) | (1 << 8) | 5, 1)
            e.issue(2, e.ptr(lane, start), _pack(ACC, rows, 1))
            e.fence()
            e.issue(0, 2, (ONE << 32) | 4)
            e.issue(3, e.ptr(dst, e.add(e.mul(start, 4), byte+power)), _pack(ACC, rows, 1))
            e.fence()
        _axis(e, count, scatter)
    return dst


def add_tensor(e, src, dst, count, initialize=False):
    def block(start, cols):
        if not initialize:
            _load(e, dst, start, 1, cols, accumulate=False)
        _load(e, src, start, 1, cols, accumulate=not initialize)
        _store(e, dst, start, 1, cols, cols*4)
    _axis(e, count, block)


def scale_tensor(e, src, coefficient, count):
    if coefficient == 1:
        return src
    dst = temporary(e, [count], 'i32', '_scaled_integer')
    if coefficient == 0:
        def zero(start, cols):
            _zero(e, 1, cols)
            _store(e, dst, start, 1, cols, cols*4)
        _axis(e, count, zero)
        return dst
    word = coefficient & 0xffffffff
    weights = []
    for byte in range(4):
        raw = (word >> (8*byte)) & 255
        signed = raw if raw < 128 else raw-256
        if signed:weights.append((signed, byte))
        if raw >= 128 and byte < 3:weights.append((1, byte+1))
    part = temporary(e, [count,1], 'i32', '_scale_partial')
    first = True
    for source, power in components(e, src):
        for weight, wp in weights:
            if power+wp >= 4:
                continue
            literal = e.global_data('_integer_coefficient_'+str(weight), [weight], 'i8')
            matmul(e, source, literal, part, count, 1, 1)
            add_tensor(e, shifted(e, part, count, power+wp), dst, count, initialize=first)
            first = False
    return dst


def gathered_product(e, lhs, rhs, dst, n, gather):
    """Contract affine source taps directly from DRAM into scratchpad tiles.

    Each mesh pass has K=1, so its PE result is bounded by 16384. The
    accumulator sums these products in the i32 ring, without byte readout.
    """
    from contraction import ACC, DIM, _config, _load, _zero, _compute, _store
    shape, split = gather['shape'], gather['split']
    strides = gather['strides']
    output_shape, tap_shape = shape[:split], shape[split:]
    _config(e)

    def tile(source, row, rows):
        def column(ni, cols):
            _zero(e, rows, cols)
            def taps(axis, offset, weight_row):
                if axis == len(tap_shape):
                    _load(e, lhs, offset, 0, rows, 1, strides[split-1])
                    _load(e, rhs, e.add(e.mul(weight_row,n),ni), DIM, 1, cols, n, state=1)
                    _compute(e, rows, 1, cols, True)
                    e.fence()
                    return
                step = prod(tap_shape[axis+1:])
                e.loop(0,tap_shape[axis],1,lambda iv: taps(axis+1,
                    e.add(offset,e.mul(iv,strides[split+axis])),
                    e.add(weight_row,e.mul(iv,step))))
            taps(0,source,0)
            _store(e,dst,e.add(e.mul(row,n),ni),rows,cols,n)
        _axis(e,n,column)

    def outer(axis, source, row):
        if axis == len(output_shape)-1:
            _axis(e,output_shape[-1],lambda iv,rows: tile(
                e.add(source,e.mul(iv,strides[axis])),e.add(row,iv),rows))
            return
        step = prod(output_shape[axis+1:])
        e.loop(0,output_shape[axis],1,lambda iv: outer(axis+1,
            e.add(source,e.mul(iv,strides[axis])),e.add(row,e.mul(iv,step))))
    outer(0,gather['storage_offset'],0)


def pack_integer_rhs(e, src, k, n):
    """Pack a contraction's transposed RHS using exact-width device DMA."""
    from contraction import ACC, _load, _store
    dst=temporary(e,[k,n],e.tensors[src]['dtype'],'_packed_integer_rhs')
    def column(ki):
        def rows(ni,nr):
            _load(e,src,e.add(e.mul(ni,k),ki),ACC,nr,1,k,state=2)
            e.fence()
            _store(e,dst,e.add(e.mul(ki,n),ni),nr,1,1)
        _axis(e,n,rows)
    e.loop(0,k,1,column)
    return dst


def add_column_bias(e, bias, dst, rows, cols):
    from contraction import ACC, _load, _store
    def row(ri, nr):
        def column(ci, nc):
            offset=e.add(e.mul(ri,cols),ci)
            _load(e,dst,offset,ACC,nr,nc,cols,state=2)
            e.fence()
            _load(e,bias,ci,ACC | (1 << 30),nr,nc,0,state=2)
            e.fence()
            _store(e,dst,offset,nr,nc,cols)
        _axis(e,cols,column)
    _axis(e,rows,row)


def lower(e, command, state):
    a = command.get('attributes', {})
    if not a.get('integer_wrap'):
        return False
    from integer_addressing import AddressEmitter
    e=AddressEmitter(e)
    if command['opcode']=='RESIDUAL_ADD' and a.get('integer_wrap'):
        o=command['operands'];dst=o['dst'];count=prod(e.tensors[dst]['shape'])
        target=temporary(e,[count],'i32','_integer_epilogue')
        for j,(key,weight) in enumerate(zip(('lhs','rhs'),(a['lhs_scale'],a['rhs_scale']))):
            part=scale_tensor(e,o[key],int(weight),count)
            if e.tensors[part]['dtype']=='i8':
                wide=temporary(e,[count],'i32','_wide_epilogue')
                def widen(start,cols):
                    _load(e,part,start,1,cols,accumulate=False)
                    _store(e,wide,start,1,cols,cols*4)
                _axis(e,count,widen)
                part=wide
            add_tensor(e,part,target,count,initialize=j==0)
        if e.tensors[dst]['dtype']=='i8':low_bytes(e,target,dst,count)
        else:add_tensor(e,target,dst,count,initialize=True)
        return True
    if command['opcode'] != 'BATCHED_MATMUL' or not a.get('integer_wrap'):
        return False
    o = command['operands']
    lhs, rhs, dst = o['a'], o['w'], o['dst']
    ash, bsh = e.tensors[lhs]['shape'], e.tensors[rhs]['shape']
    gather=a.get('lhs_gather')
    if gather:
        m,k,n=a['matrix_shape']
        batches=1
        if prod(gather['shape'][gather['split']:]) != k or e.tensors[dst]['shape'] != [m,n]:
            raise ValueError('gathered integer contraction dimensions disagree')
        if gather.get('transpose_rhs'):
            rhs=pack_integer_rhs(e,rhs,k,n)
    else:
        m,k,n=ash[-2],ash[-1],bsh[-1]
        batches=prod(ash[:-2])
    if not gather and (ash[:-2] != bsh[:-2] or bsh[-2] != k):
        raise ValueError('integer contraction requires equal static batch prefixes')
    count = batches * m * n
    target = temporary(e, e.tensors[dst]['shape'], 'i32', '_integer_product')
    partial = temporary(e, e.tensors[dst]['shape'], 'i32', '_integer_partial')
    def zero(start, cols):
        _zero(e, 1, cols)
        _store(e, target, start, 1, cols, cols*4)
    _axis(e, count, zero)
    left_parts, right_parts = components(e, lhs), components(e, rhs)
    for left, lp in left_parts:
        for right, rp in right_parts:
            power = lp + rp
            if power >= (1 if e.tensors[dst]['dtype']=='i8' else 4):
                continue
            if gather:
                gathered_product(e,left,right,partial,n,gather)
                add_tensor(e,shifted(e,partial,count,power),target,count)
                continue
            def chunk(ki, kr):
                def batch(bi):
                    matmul(e, left, right, partial, m, kr, n,
                           a_offset=e.add(e.mul(bi,m*k),ki),
                           b_offset=e.add(e.mul(bi,k*n),e.mul(ki,n)),
                           c_offset=e.mul(bi,m*n), a_stride=k)
                e.loop(0, batches, 1, batch)
                part = shifted(e, partial, count, power)
                add_tensor(e, part, target, count)
            # Each retained component sum is bounded even for arbitrarily long K.
            # Recombination between chunks has exactly the source modular width.
            _axis(e, k, chunk)
    seed = o.get('seed')
    if seed:
        # A byte seed must be widened by DMA, without a saturating output stage.
        if e.tensors[seed]['dtype'] == 'i8':
            wide = temporary(e, [count], 'i32', '_integer_seed')
            def widen(start, cols):
                _load(e, seed, start, 1, cols, accumulate=False)
                _store(e, wide, start, 1, cols, cols*4)
            _axis(e, count, widen)
            seed = wide
        add_tensor(e, seed, target, count)
    if o.get('column_bias'):
        add_column_bias(e,o['column_bias'],target,batches*m,n)
    if e.tensors[dst]['dtype'] == 'i8':
        low_bytes(e, target, dst, count)
    else:
        add_tensor(e, target, dst, count, initialize=True)
    return True
