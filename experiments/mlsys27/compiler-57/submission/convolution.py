"""Shape-driven device contractions and bounded NHWC convolution gathering.

Only loop indices and addresses are computed by the scalar core. Gathered
activation and transpose bytes travel through Gemmini's DMA path.
"""
from math import prod

from device import (ACC, ACCUM, FULL, GARBAGE, DIM, matmul, transfer, bias_stride,
                    config_ex, config_ld, config_st, descriptor, fbits)


def _resolve(cb, name):
    aliases = {
        c['operands']['dst']: c['operands']['src']
        for c in cb.get('commands', []) if c['opcode'] == 'RES_PACK'
    }
    visited = set()
    while name in aliases:
        if name in visited:
            raise ValueError('cyclic resident tensor aliases')
        visited.add(name)
        name = aliases[name]
    return name


def _dtype_size(dtype):
    if dtype in ('i8', 'int8', 'ui8', 'u8'):
        return 1
    if dtype in ('i32', 'int32'):
        return 4
    raise ValueError('device contraction supports i8 operands and i8/i32 results')


def _is_i8(tensor):
    return tensor.get('dtype') in ('i8', 'int8')


def _transfer(g, source, dest, cols):
    transfer(g, source, dest, 1, cols, cols, cols,
             input_dtype='i8', output_dtype='i8', scale=1.0)


def _transpose(g, ptr, rows, cols):
    """Transpose an i8 matrix by device DMA without host tensor arithmetic."""
    output = g.alloca(rows * cols)
    def column(j):
        def row_block(begin, count):
            source = g.gep(ptr, g.add(g.mul(begin, cols), j))
            destination = g.gep(output, g.add(g.mul(j, rows), begin))
            transfer(g, source, destination, count, 1, cols, 1,
                     input_dtype='i8', output_dtype='i8')
        if rows // DIM:
            g.loop(rows // DIM, lambda block: row_block(g.mul(block, DIM), DIM))
        if rows % DIM:
            row_block(rows // DIM * DIM, rows % DIM)
    g.loop(cols, column)
    g.fence()
    return output


def _batched(g, cb, command, ptrs):
    op = command['operands']
    attrs = command.get('attributes', {})
    a_name, w_name, y_name = op['a'], _resolve(cb, op['w']), op['dst']
    a, w, y = (cb['tensors'][name] for name in (a_name, w_name, y_name))
    ash, wsh, ysh = a['shape'], w['shape'], y['shape']
    if not (_is_i8(a) and _is_i8(w)):
        return False
    if len(ash) < 3 or len(ash) != len(wsh) or len(ash) != len(ysh):
        raise ValueError('batched matmul requires equal rank >= 3')
    if ash[:-2] != wsh[:-2] or ash[:-2] != ysh[:-2]:
        raise ValueError('batched matmul does not broadcast batch dimensions')
    m, k, n = ash[-2], ash[-1], wsh[-1]
    if wsh[-2] != k or ysh[-2:] != [m, n]:
        raise ValueError('batched contraction dimensions disagree')
    dtype = y['dtype']
    if dtype not in ('i32', 'int32'):
        return False
    bias_name = op.get('bias', attrs.get('bias'))
    seed = None
    seed_stride = 0
    batched_seed = False
    if bias_name is not None:
        info = cb['tensors'][bias_name]
        seed = ptrs[bias_name]
        if info['shape'] == ysh and info['dtype'] in ('i32', 'int32'):
            seed_stride = n * 4
            batched_seed = True
        else:
            seed_stride = bias_stride(info, m, n)
    def batch(i):
        ap = g.gep(ptrs[a_name], g.mul(i, m * k))
        wp = g.gep(ptrs[w_name], g.mul(i, k * n))
        yp = g.gep(ptrs[y_name], g.mul(i, m * n * 4))
        bp = g.gep(seed, g.mul(i, m * n * 4)) if batched_seed else seed
        matmul(g, ap, wp, yp, m, k, n, attrs, output_dtype='i32',
               bias=bp, bias_row_stride=seed_stride)
    g.loop(prod(ash[:-2]), batch)
    g.fence()
    return True


def _attention(g, cb, command, ptrs):
    op = command['operands']
    attrs = command.get('attributes', {})
    qk = command['opcode'] == 'ATTENTION_QK'
    a_name = op['q'] if qk else op['p']
    w_name = _resolve(cb, op['k'] if qk else op['v'])
    y_name = op['dst']
    a, w, y = (cb['tensors'][name] for name in (a_name, w_name, y_name))
    if not (_is_i8(a) and _is_i8(w)):
        return False
    if len(a['shape']) != 2 or len(w['shape']) != 2:
        return False
    m, k = a['shape']
    if qk:
        n, wk = w['shape']
        if wk != k:
            raise ValueError('attention QK head dimensions disagree')
        weight = _transpose(g, ptrs[w_name], n, k)
    else:
        wk, n = w['shape']
        if wk != k:
            raise ValueError('attention PV sequence dimensions disagree')
        weight = ptrs[w_name]
    if y['shape'] != [m, n]:
        raise ValueError('attention destination shape disagrees')
    dtype = attrs.get('output_dtype', y['dtype'])
    if dtype not in ('i8', 'i32', 'int8', 'int32'):
        return False
    bias_name = op.get('bias', attrs.get('bias'))
    seed_stride = bias_stride(cb['tensors'][bias_name], m, n) if bias_name is not None else 0
    matmul(g, ptrs[a_name], weight, ptrs[y_name], m, k, n,
           attrs, output_dtype='i8' if _dtype_size(dtype) == 1 else 'i32',
           bias=ptrs.get(bias_name), bias_row_stride=seed_stride)
    g.fence()
    return True


def _conv(g, cb, command, ptrs):
    op = command['operands']
    attrs = command.get('attributes', {})
    x_name = op['ifm']
    w_name = _resolve(cb, op['weight'])
    y_name = op['dst']
    x, w, y = (cb['tensors'][name] for name in (x_name, w_name, y_name))
    if not (_is_i8(x) and _is_i8(w)):
        return False
    if attrs.get('layout', 'nhwc') != 'nhwc':
        return False
    epilogue = attrs.get('epilogue', [])
    if len(x['shape']) != 4 or len(w['shape']) != 2:
        return False
    batch, height, width, ci = x['shape']
    kh, kw, kernel_ci, co = attrs['kernel']
    sh, sw = attrs.get('stride', [1, 1])
    dh, dw = attrs.get('dilation', [1, 1])
    pt, pl, pb, pr = attrs.get('padding', [0, 0, 0, 0])
    if ci != kernel_ci or w['shape'] != [kh * kw * ci, co]:
        raise ValueError('convolution kernel/input/weight shapes disagree')
    if min(kh, kw, ci, co, sh, sw, dh, dw) <= 0 or min(pt, pl, pb, pr) < 0:
        raise ValueError('invalid convolution geometry')
    ho = (height + pt + pb - dh * (kh - 1) - 1) // sh + 1
    wo = (width + pl + pr - dw * (kw - 1) - 1) // sw + 1
    if min(ho, wo) <= 0:
        raise ValueError('convolution has no output positions')
    dtype = attrs.get('output_dtype', y['dtype'])
    if dtype not in ('i8', 'i32', 'int8', 'int32'):
        return False
    pooled = 'maxpool' in epilogue
    unpooled_shape = [batch * ho * wo, co]
    if pooled:
        if epilogue[-1] != 'maxpool' or _dtype_size(dtype) != 1:
            return False
        if attrs.get('pool_in_dims') != [ho, wo]:
            raise ValueError('pool_in_dims must match the convolution output geometry')
        # Validate the complete pooling geometry before appending any code.
        from pooling import _geometry, emit_maxpool
        _geometry(unpooled_shape, y['shape'], attrs)
    elif prod(y['shape']) != batch * ho * wo * co:
        raise ValueError('convolution destination does not match output geometry')
    bias_name = op.get('bias', attrs.get('bias'))
    if ('bias_add' in epilogue or 'bias' in epilogue) and bias_name not in ptrs:
        raise ValueError('convolution bias stage requires a tensor pointer')
    bias = ptrs.get(bias_name)
    seed_stride = bias_stride(cb['tensors'][bias_name], batch * ho * wo, co) if bias_name is not None else 0
    kernel_attrs = dict(attrs)
    kernel_attrs['epilogue'] = [stage for stage in epilogue if stage != 'maxpool']
    output = g.alloca(prod(unpooled_shape)) if pooled else ptrs[y_name]
    k = kh * kw * ci
    output_dtype = 'i8' if _dtype_size(dtype) == 1 else 'i32'
    if kh == kw == sh == sw == dh == dw == 1 and not any((pt, pl, pb, pr)):
        matmul(g, ptrs[x_name], ptrs[w_name], output,
               batch * height * width, ci, co, kernel_attrs,
               output_dtype=output_dtype, bias=bias, bias_row_stride=seed_stride)
    else:
        # Gather one K panel directly into the scratchpad. No im2col tensor is
        # written to DRAM: A occupies rows 0..15 and B occupies rows 16..31.
        # Every panel contributes into the same accumulator output tile.
        stages = kernel_attrs.get('epilogue', [])
        supported = {'bias', 'bias_add', 'acc_scale', 'relu'}
        if any(stage not in supported for stage in stages):
            raise NotImplementedError('convolution epilogue ' + str(stages))
        if any(key in kernel_attrs for key in ('scale_numerator', 'scale_denominator', 'zero_point', 'quant_min', 'quant_max')):
            raise NotImplementedError('exact rational convolution readout needs a separate device construction')
        scale = float(kernel_attrs.get('acc_scale', 1.0)) if 'acc_scale' in stages else 1.0
        scales = kernel_attrs.get('acc_scale_by_column')
        relu = 'relu' in stages
        full_output = output_dtype == 'i32'
        if full_output and (relu or scales is not None or scale != 1.0):
            raise NotImplementedError('full i32 convolution readout applies no scale or activation')
        if scales is None:
            runs = [(0, co, scale)]
        else:
            if len(scales) != co:
                raise ValueError('convolution per-column scale count must equal output channels')
            runs = []
            begin = 0
            for index in range(1, co + 1):
                if index == co or fbits(scales[index]) != fbits(scales[begin]):
                    runs.append((begin, index-begin, float(scales[begin])))
                    begin = index
        negative = any(value < 0 for _, _, value in runs)
        if relu and negative and 'acc_scale' in stages and stages.index('relu') < stages.index('acc_scale'):
            raise NotImplementedError('negative scaling after full-width relu needs a separate exact device construction')
        narrowed = g.alloca(DIM*DIM) if relu and negative else None
        dataflow = kernel_attrs.get('dataflow', 'WS')
        if dataflow not in ('WS', 'OS'):
            raise ValueError('unknown convolution dataflow ' + str(dataflow))
        padded = any((pt, pl, pb, pr))
        channel_tiles = (ci + DIM - 1)//DIM
        static_sites = batch*ho*((wo+DIM-1)//DIM)*kh*kw*channel_tiles*((co+DIM-1)//DIM) <= 256
        config_ex(g, dataflow)

        def tile_channels(width, callback, start=0):
            full, tail = divmod(width, DIM)
            if static_sites:
                for index in range(full):
                    callback(start + index*DIM, DIM)
            elif full:
                g.loop(full, lambda index:callback(g.add(start,g.mul(index,DIM)),DIM))
            if tail:
                callback(start + full*DIM, tail)

        def output_row(sample, oh):
            def output_tile(ow, rows):
                output_index = g.add(g.mul(g.add(g.mul(sample,ho),oh),wo),ow)

                def output_columns(column, cols, tile_scale):
                    # A zero load and an unshrunk i32 bias load both seed the
                    # accumulator once. All K panels then use accumulate=1.
                    g.fence()
                    config_ld(g,seed_stride if bias is not None else 0,state=2,dtype='i32')
                    seed = g.gep(bias,g.add(g.mul(output_index,seed_stride),g.mul(column,4))) if bias is not None else 0
                    g.rocc(14,g.ptrint(seed),descriptor(g,ACC,rows,cols))
                    g.fence()

                    def tap(kr,kc):
                        ih = g.sub(g.add(g.mul(oh,sh),g.mul(kr,dh)),pt)
                        ix = g.sub(g.add(g.mul(ow,sw),g.mul(kc,dw)),pl)
                        if padded:
                            first=g.min(rows,g.udiv(g.add(g.max(0,g.sub(0,ix)),sw-1),sw))
                            end=g.min(rows,g.udiv(g.add(g.max(0,g.sub(width,ix)),sw-1),sw))
                            valid_rows=g.max(0,g.sub(end,first))
                            if isinstance(ih,int) and isinstance(valid_rows,int):
                                valid=0<=ih<height and valid_rows>0
                            else:
                                valid=g.value(f'and i1 {g.icmp("ult",ih,height)}, {g.icmp("ugt",valid_rows,0)}')
                        else:
                            first,valid_rows=0,rows

                        def channel(ch,terms):
                            g.fence()
                            if padded:
                                config_ld(g,0,state=0,dtype='i8')
                                g.rocc(2,0,descriptor(g,0,rows,terms))
                                g.fence()

                            def gather():
                                source_x=g.add(ix,g.mul(first,sw))
                                pixel=g.add(g.mul(g.add(g.mul(sample,height),ih),width),source_x)
                                source=g.gep(ptrs[x_name],g.add(g.mul(pixel,ci),ch))
                                config_ld(g,sw*ci,state=0,dtype='i8')
                                if isinstance(valid_rows,int):
                                    g.rocc(2,g.ptrint(source),descriptor(g,first,valid_rows,terms))
                                else:
                                    # Runtime border spans keep constant transfer
                                    # dimensions at every emitted instruction site.
                                    g.loop(valid_rows,lambda row:g.rocc(2,g.ptrint(g.gep(source,g.mul(row,sw*ci))),descriptor(g,g.add(first,row),1,terms)))
                                g.fence()
                            if padded:
                                if type(valid) is bool:
                                    if valid:gather()
                                else:g.conditional(valid,gather)
                            else:gather()
                            tap_index=g.add(g.mul(kr,kw),kc)
                            weight_index=g.add(g.mul(g.add(g.mul(tap_index,ci),ch),co),column)
                            weights=g.gep(ptrs[w_name],weight_index)
                            config_ld(g,co,state=1,dtype='i8')
                            g.rocc(1,g.ptrint(weights),descriptor(g,DIM,terms,cols))
                            g.fence()
                            if dataflow=='WS':
                                g.rocc(6,descriptor(g,DIM,terms,cols),descriptor(g,ACC|ACCUM,rows,cols))
                                g.rocc(4,descriptor(g,0,rows,terms),descriptor(g,GARBAGE,rows,cols))
                            else:
                                g.rocc(6,descriptor(g,GARBAGE,rows,cols),descriptor(g,ACC|ACCUM,rows,cols))
                                g.rocc(4,descriptor(g,0,rows,terms),descriptor(g,DIM,terms,cols))
                            g.fence()
                        tile_channels(ci,channel)
                    if static_sites:
                        for kr in range(kh):
                            for kc in range(kw):tap(kr,kc)
                    else:
                        g.loop(kh,lambda kr:g.loop(kw,lambda kc:tap(kr,kc)))
                    destination=g.gep(output,g.mul(g.add(g.mul(output_index,co),column),_dtype_size(dtype)))
                    post_relu=relu and tile_scale<0
                    config_st(g,DIM if post_relu else co*_dtype_size(dtype),tile_scale,relu and not post_relu)
                    g.rocc(3,g.ptrint(narrowed if post_relu else destination),descriptor(g,ACC|(FULL if full_output else 0),rows,cols))
                    g.fence()
                    if post_relu:
                        transfer(g,narrowed,destination,rows,cols,DIM,co,'i8','i8',relu=True)
                for begin,run_width,tile_scale in runs:
                    tile_channels(run_width,lambda column,cols,tile_scale=tile_scale:output_columns(column,cols,tile_scale),begin)
            tile_channels(wo,output_tile)
        if static_sites:
            for sample in range(batch):
                for oh in range(ho):output_row(sample,oh)
        else:
            g.loop(batch,lambda sample:g.loop(ho,lambda oh:output_row(sample,oh)))
    g.fence()
    if pooled:
        # The contraction has already applied scale/activation exactly once;
        # pooling consumes those narrowed device results without another scale.
        emit_maxpool(g, output, ptrs[y_name], unpooled_shape, y['shape'], attrs,
                     input_dtype='i8', output_dtype='i8')
        g.fence()
    return True


def emit_contraction(g, cb, command, ptrs):
    """Emit a supported whole contraction; return False before emitting if absent."""
    opcode = command['opcode']
    if opcode == 'CONV2D':
        return _conv(g, cb, command, ptrs)
    if opcode == 'BATCHED_MATMUL':
        return _batched(g, cb, command, ptrs)
    if opcode in ('ATTENTION_QK', 'ATTENTION_PV'):
        return _attention(g, cb, command, ptrs)
    return False
