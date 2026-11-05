"""Max-window lowering using the target's pooled device readout.

Each output window is staged as consecutive scratchpad rows, then one pooled
STORE takes its maximum. Coordinate and descriptor arithmetic stay on the host;
tensor values remain on the DMA and pooled-store datapaths throughout.
"""
import math

from device import config_ld, descriptor, fbits


def _geometry(shape_in, shape_out, attrs):
    if attrs.get('layout', 'nhwc').lower() not in ('nhwc', 'channels_last'):
        raise NotImplementedError('pooled-store lowering requires channels-last input')
    dims = attrs.get('pool_in_dims')
    if dims is None:
        if len(shape_in) == 4:
            dims = shape_in[1:3]
        elif len(shape_in) == 3:
            dims = shape_in[:2]
        else:
            raise NotImplementedError('flattened pooling requires pool_in_dims')
    if len(dims) != 2:
        raise NotImplementedError('pooled store supports two spatial dimensions')
    height, width = map(int, dims)
    kernel = attrs.get('pool_size', attrs.get('kernel'))
    stride = attrs.get('pool_stride', attrs.get('stride'))
    padding = attrs.get('pool_padding', attrs.get('padding', [0, 0, 0, 0]))
    if isinstance(padding, int):
        padding = [padding] * 4
    if any(padding):
        raise NotImplementedError('pooled-store padding has no derived numeric pad value')
    if isinstance(kernel, int):
        kernel = [kernel, kernel]
    if isinstance(stride, int):
        stride = [stride, stride]
    if kernel is None or stride is None or len(kernel) != 2 or len(stride) != 2:
        raise NotImplementedError('pool_size and pool_stride must have two explicit dimensions')
    ph, pw = map(int, kernel)
    sh, sw = map(int, stride)
    if ph != pw or ph < 1 or ph > 3:
        raise NotImplementedError('pooled-store window must be square and fit its 2-bit size field')
    if min(sh, sw) < 1 or min(height, width) < 1:
        raise ValueError('pool extents and strides must be positive')
    channels = int(shape_in[-1])
    count_in = math.prod(shape_in)
    if count_in % (height * width * channels):
        raise ValueError('pool_in_dims do not divide the source rows')
    batch = count_in // (height * width * channels)
    oh, ow = (height - ph) // sh + 1, (width - pw) // sw + 1
    if min(oh, ow) < 1 or math.prod(shape_out) != batch * oh * ow * channels:
        raise ValueError('pooled output shape does not match the declared window geometry')
    if int(shape_out[-1]) != channels:
        raise ValueError('pooled output channels must match input')
    return batch, height, width, channels, ph, pw, sh, sw, oh, ow


def emit_maxpool(g, src, dst, shape_in, shape_out, attrs,
                 input_dtype='i8', output_dtype='i8'):
    """Emit channels-last i8 max pooling with no undeclared host arithmetic.

    Geometry is checked before any IR is appended. A flattened [rows, channels]
    tensor supplies ``pool_in_dims``; rank-three and rank-four tensors may supply
    the spatial dimensions directly in their shape. Unsupported geometry raises
    NotImplementedError so the caller can explicitly decline that source region.
    """
    if input_dtype not in ('i8', 'int8') or output_dtype not in ('i8', 'int8'):
        raise NotImplementedError('scratchpad pooled-store currently supports i8 values only')
    batch, height, width, channels, ph, pw, sh, sw, oh, ow = _geometry(shape_in, shape_out, attrs)
    channel_tiles = (channels + 15) // 16

    def output_position(sample, oy, ox):
        position = g.add(g.mul(g.add(g.mul(sample, oh), oy), ow), ox)
        iy, ix = g.mul(oy, sh), g.mul(ox, sw)

        def channel_tile(channel, cols):
            g.fence()
            config_ld(g, channels, dtype='i8')

            def window_row(ky):
                row = g.add(g.mul(sample, height), g.add(iy, ky))
                pixel = g.add(g.mul(row, width), ix)
                offset = g.add(g.mul(pixel, channels), channel)
                local_row = g.mul(ky, pw)
                g.rocc(2, g.ptrint(g.gep(src, offset)), descriptor(g, local_row, pw, cols))
            g.loop(ph, window_row)
            # The contract explicitly requires this fence for a scratchpad-source
            # pooled store; its dependency tracker otherwise assumes accumulator
            # source banks and may let the store race these scratchpad loads.
            g.fence()
            # ConfigMvoutRs1 fields from the selected RTL facts: a one-output
            # patch with ph x pw source pixels, no padding and unit pool stride.
            config = (2 | (1 << 4) | (ph << 6) | (1 << 24)
                      | (1 << 32) | (1 << 40) | (ph << 48) | (pw << 56))
            g.rocc(0, config, (fbits(1.0) << 32) | channels)
            output_offset = g.add(g.mul(position, channels), channel)
            g.rocc(3, g.ptrint(g.gep(dst, output_offset)), descriptor(g, 0, 1, cols))
            g.fence()
        if channels // 16:
            g.loop(channels // 16, lambda tile: channel_tile(g.mul(tile, 16), 16))
        if channels % 16:
            channel_tile(channels // 16 * 16, channels % 16)
    g.loop(batch, lambda sample:
           g.loop(oh, lambda oy:
                  g.loop(ow, lambda ox: output_position(sample, oy, ox))))
