"""Typed modular sums on the accumulator data path, with no host payload work."""
from math import prod

from affine_reduction import _load, _zero, _store, _constant
from contraction import _axis


def temporary(e, shape, dtype, stem):
    name = stem + '_' + str(len(e.tensors))
    e.tensors[name] = {'shape': list(shape), 'dtype': dtype, 'role': 'intermediate'}
    e.ptr(name)
    return name


def low_bytes(e, src, dst, count, source_width=4, dest_width=1, byte=0):
    """DMA a storage byte per element; narrowing is truncation, never saturation."""
    from affine_reduction import ACC, ONE, _pack
    def block(start, rows):
        e.issue(0, (ONE << 32) | (16 << 16) | (1 << 8) | 5, source_width)
        e.issue(2, e.ptr(src, e.add(e.mul(start, source_width), byte)), _pack(ACC, rows, 1))
        e.fence()
        e.issue(0, 2, (ONE << 32) | dest_width)
        e.issue(3, e.ptr(dst, e.mul(start, dest_width)), _pack(ACC, rows, 1))
        e.fence()
    _axis(e, count, block)


def lower(e, command, state):
    a = command.get('attributes', {})
    if command['opcode'] != 'WINDOW_SUM' or not a.get('integer_wrap'):
        return False
    from integer_addressing import AddressEmitter
    e=AddressEmitter(e)
    src, dst = (command['operands'][k] for k in ('src', 'dst'))
    ss, ds = e.tensors[src]['shape'], e.tensors[dst]['shape']
    narrow = e.tensors[dst]['dtype'] == 'i8'
    target = temporary(e, ds, 'i32', '_integer_sum') if narrow else dst
    initial = int(a.get('initial', 0))
    seed = _constant(e, state, [initial] * 16, 'i32') if initial else None

    def initialize(cols):
        if seed:
            _load(e, seed, 0, 1, cols, accumulate=False)
        else:
            _zero(e, 1, cols)

    if a['mode'] == 'prefix':
        if ds!=ss or not ss:
            raise ValueError('integer prefix result must retain the input shape')
        axis = int(a['axis']) % len(ss)
        inner = prod(ss[axis+1:])
        def outer(oi):
            def tile(ii, cols):
                initialize(cols)
                def step(ri):
                    off = e.add(e.mul(e.add(e.mul(oi, ss[axis]), ri), inner), ii)
                    _load(e, src, off, 1, cols)
                    _store(e, target, off, 1, cols, cols * 4)
                e.loop(0, ss[axis], 1, step)
            _axis(e, inner, tile)
        e.loop(0, prod(ss[:axis]), 1, outer)
    else:
        dims = sorted(set(a['dims']))
        if any(type(d) is not int or not 0<=d<len(ss) for d in dims):
            raise ValueError('integer reduction axes must be static input dimensions')
        parallel = [d for d in range(len(ss)) if d not in dims]
        if prod(ss[d] for d in parallel) != prod(ds):
            raise ValueError('integer reduction result does not match retained axes')
        strides = [prod(ss[d+1:]) for d in range(len(ss))]
        contiguous = bool(parallel and parallel[-1] == len(ss)-1)
        outer_axes = parallel[:-1] if contiguous else parallel
        def tile(source, dest, cols):
            initialize(cols)
            def reduce_axis(j, off):
                if j == len(dims):
                    _load(e, src, off, 1, cols)
                else:
                    d = dims[j]
                    e.loop(0, ss[d], 1, lambda iv: reduce_axis(j+1, e.add(off, e.mul(iv, strides[d]))))
            reduce_axis(0, source)
            _store(e, target, dest, 1, cols, cols * 4)
        def outer(j, source, dest):
            if j == len(outer_axes):
                if contiguous:
                    _axis(e, ss[-1], lambda iv, cols: tile(e.add(source, iv), e.add(dest, iv), cols))
                else:
                    tile(source, dest, 1)
                return
            d = outer_axes[j]
            stride = prod(ss[k] for k in parallel[parallel.index(d)+1:])
            e.loop(0, ss[d], 1, lambda iv: outer(j+1, e.add(source, e.mul(iv, strides[d])), e.add(dest, e.mul(iv, stride))))
        outer(0, 0, 0)
    if narrow:
        low_bytes(e, target, dst, prod(ds))
    return True
