"""Standalone signed-byte max pooling through accumulator load and pooled store.

A complete valid window is gathered into consecutive accumulator rows. The
store sees a one-element output grid and never reads a padded or unwritten tap.
"""
from math import prod

DIM = 16
ACC = 1 << 31
ONE = 0x3f800000


def _geometry(e, c, state):
    attrs = c.get('attributes', {})
    if c.get('opcode') != 'WINDOW_SUM' or attrs.get('mode') != 'max':
        return None
    operands = c.get('operands', {})
    src, dst = operands.get('src'), operands.get('dst')
    if src not in e.tensors or dst not in e.tensors:
        return None
    source, output = e.tensors[src], e.tensors[dst]
    if source['dtype'] not in ('i8', 'int8','i32','int32') or output['dtype'] not in ('i8', 'int8'):
        return None
    if source['dtype'] in ('i32','int32'):
        bounds=state.get('value_bounds',{}).get(src)
        if bounds is None or bounds[0]<-128 or bounds[1]>127:return None
    try:
        h, w = attrs['pool_in_dims']
        ph, pw = attrs['pool_size']
        sh, sw = attrs['pool_stride']
    except (KeyError, TypeError, ValueError):
        return None
    if any(type(v) is not int for v in (h, w, ph, pw, sh, sw)):
        return None
    if min(h, w, ph, pw, sh, sw) < 1 or max(h, w) > 255:
        return None
    if ph != pw or sh != sw or ph > 3 or sh > 3 or h < ph or w < pw:
        return None
    # The selected store gives physical padding the value zero. A source
    # max initialized to -128 cannot generally substitute zero, so this path
    # admits only complete source windows with no physical padded taps.
    if list(attrs.get('pool_padding', [0, 0, 0, 0])) != [0, 0, 0, 0]:
        return None
    if attrs.get('initial', -128) != -128:
        return None
    source_shape = source['shape']
    if len(source_shape) not in (2, 3, 4) or not source_shape:
        return None
    channels = source_shape[-1]
    source_count = prod(source_shape)
    if type(channels) is not int or channels < 1 or source_count % (h*w*channels):
        return None
    batches = source_count // (h*w*channels)
    if batches < 1:
        return None
    oh, ow = (h-ph)//sh+1, (w-pw)//sw+1
    if prod(output['shape']) != batches*oh*ow*channels:
        return None
    return src, dst, h, w, ph, pw, sh, sw, channels, batches, oh, ow


def lower(e, command, state):
    """Emit a legal unpadded max window, or refuse before emitting commands."""
    geometry = _geometry(e, command, state)
    if geometry is None:
        return False
    src, dst, h, w, ph, pw, sh, sw, channels, batches, oh, ow = geometry
    # The accumulator is input-width loaded at scale1, preserving signed i8
    # exactly. Its row stride is the input's physical channel width.
    source_width=4 if e.tensors[src]['dtype'] in ('i32','int32') else 1
    e.issue(0, (ONE << 32) | (DIM << 16) | (1 << 8) | ((1 << 2) if source_width==1 else 0) | 1, channels*source_width)
    # All taps were gathered, so the physical input grid is ph x pw and the
    # pooled output grid contains one valid element. Field widths come from
    # ConfigMvoutRs1: size/stride2, grid extents8, scale in rs2 upper32.
    store_config = (2 | (sh << 4) | (ph << 6) | (1 << 24)
                    | (1 << 32) | (1 << 40) | (ph << 48) | (pw << 56))
    e.issue(0, store_config, (ONE << 32) | channels)

    def batch(bi):
        def y(yi):
            def x(xi):
                source_row = e.add(e.mul(bi, h*w),
                                   e.add(e.mul(yi, sh*w), e.mul(xi, sw)))
                target_row = e.add(e.mul(bi, oh*ow),
                                   e.add(e.mul(yi, ow), xi))

                def columns(ci, count):
                    for ky in range(ph):
                        source = e.add(e.mul(e.add(source_row, ky*w), channels), ci)
                        local = ACC | (ky*pw)
                        word = local | (count << 32) | (pw << 48)
                        e.issue(2, e.ptr(src, e.mul(source,source_width)), word)
                    # This fence is mandatory for a scratchpad-source pool;
                    # retain it for this accumulator gather as well so every
                    # load has completed before the multi-tap store begins.
                    e.fence()
                    target = e.add(e.mul(target_row, channels), ci)
                    e.issue(3, e.ptr(dst, target), ACC | (count << 32) | (1 << 48))
                    e.fence()

                bulk = channels//DIM*DIM
                e.loop(0, bulk, DIM, lambda ci: columns(ci, DIM))
                if channels != bulk:
                    columns(bulk, channels-bulk)
            e.loop(0, ow, 1, x)
        e.loop(0, oh, 1, y)
    e.loop(0, batches, 1, batch)
    return True
