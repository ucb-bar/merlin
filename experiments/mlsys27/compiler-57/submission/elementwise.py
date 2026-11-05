"""Device implementations of integer maps and reductions.

All arithmetic on tensor elements is performed by Gemmini. Host operations
construct descriptors, iteration coordinates and literal coefficient tensors.
Exact rational narrowing uses integer threshold comparisons on the device when
its rounding cannot be represented by one binary32 readout.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import prod
import math
import struct

from device import (ACC, ACCUM, FULL, config_ld, config_st, descriptor, fbits,
                    matmul, transfer)


def _dtype(t):
    return {'int8': 'i8', 'int32': 'i32'}.get(t, t)


def _tiles(g, count, body):
    """Keep each transfer site's row count constant, including a short tail."""
    full, tail = divmod(count, 16)
    if full:
        g.loop(full, lambda index: body(g.mul(index, 16), 16))
    if tail:
        body(full * 16, tail)


def _rational(attrs, key='readout_scale'):
    if 'scale_numerator' in attrs or 'scale_denominator' in attrs:
        return Fraction(int(attrs['scale_numerator']), int(attrs['scale_denominator']))
    return Fraction(float(attrs.get(key, 1.0)))


def _seed(g, ptr, value, rows, addr=ACC, stride=0):
    config_ld(g, stride, dtype='i32')
    g.rocc(2, g.ptrint(ptr), descriptor(g, addr, rows, 1))
    g.fence()


def _load(g, ptr, rows, dtype='i8', stride=1, address=ACC):
    config_ld(g, stride, dtype=dtype)
    g.rocc(2, g.ptrint(ptr), descriptor(g, address, rows, 1))
    g.fence()


def _store(g, ptr, rows, dtype='i32', stride=4, address=ACC,
           scale=1.0, relu=False):
    config_st(g, stride, scale, relu)
    g.rocc(3, g.ptrint(ptr), descriptor(g, address | (FULL if dtype == 'i32' else 0), rows, 1))
    g.fence()


def _literal_ptr(g, values, dtype='i32'):
    """DMA literals from immutable module data; null is the device zero row."""
    values = tuple(int(value) for value in values)
    if not values or (len(values) == 1 and values[0] == 0):
        return 0
    cache = getattr(g, '_elementwise_literals', None)
    if cache is None:
        cache = g._elementwise_literals = {}
    key = (dtype, values)
    if key not in cache:
        label = '@elementwise_literal_' + str(g.counter)
        g.counter += 1
        body = ', '.join(dtype + ' ' + str(value) for value in values)
        g.globals.append(label + ' = private constant [' + str(len(values)) +
                         ' x ' + dtype + '] [' + body + '], align 64')
        cache[key] = label
    return cache[key]


def _constant_ptr(g, value):
    return _literal_ptr(g, [value])


def _scratch(g, size):
    """Device-owned staging has module lifetime and needs no host allocation."""
    label = '@elementwise_scratch_' + str(g.counter)
    g.counter += 1
    g.globals.append(label + ' = internal global [' + str(max(1, size)) +
                     ' x i8] zeroinitializer, align 64')
    return label


@lru_cache(maxsize=256)
def _prove_f32_readout(numerator,denominator,low_bound,high_bound,scale_bits,relu):
    """Exhaust the source-proved integer domain before choosing device RNE.

    Each integer converts exactly to binary32. A binary32 scale has at most
    24 significant bits, so the binary64 product is exact before the explicit
    binary32 rounding below. No tensor values are evaluated by host code.
    """
    if (type(low_bound) is not int or type(high_bound) is not int or
        high_bound<low_bound or high_bound-low_bound+1>65536 or
        max(abs(low_bound),abs(high_bound))>=1<<24 or denominator<=0 or
        (relu and numerator<0)):
        return False
    scale=struct.unpack('<f',struct.pack('<I',scale_bits))[0]
    if math.isnan(scale):return False
    for value in range(low_bound,high_bound+1):
        value=max(0,value) if relu else value
        quotient,remainder=divmod(value*numerator,denominator)
        twice=remainder*2
        if twice>denominator or (twice==denominator and quotient&1):quotient+=1
        expected=min(127,max(-128,quotient))
        exact_product=value*scale
        try:
            rounded_product=struct.unpack('<f',struct.pack('<f',exact_product))[0]
        except OverflowError:
            rounded_product=math.copysign(math.inf,exact_product)
        if math.isnan(rounded_product):return False
        actual=(127 if rounded_product>=127 else -128 if rounded_product<=-128
                else round(rounded_product))
        if actual!=expected:return False
    return True


def _proved_readout_scale(ratio,low_bound,high_bound,relu):
    """Return the actual emitted f32 scale only after a bounded proof."""
    if (type(low_bound) is not int or type(high_bound) is not int or
        high_bound<low_bound or high_bound-low_bound+1>65536 or
        max(abs(low_bound),abs(high_bound))>=1<<24 or (relu and ratio<0)):
        return None
    try:
        bits=fbits(float(ratio))
    except OverflowError:
        # Scale conversion must use the same accepted bits as CONFIG_ST.
        # An unavailable scale leaves the existing threshold path in place.
        return None
    if not _prove_f32_readout(ratio.numerator,ratio.denominator,
                             low_bound,high_bound,bits,relu):return None
    return struct.unpack('<f',struct.pack('<I',bits))[0]


def _exact_readout(g, source, destination, count, attrs, low_bound, high_bound,
                   output_dtype='i8'):
    """Quantize integer accumulator data using target arithmetic only.

    Comparisons use relu plus saturating int8 readout: an integer z > 0
    becomes 127 at scale 127, and zero otherwise. A subsequent 1/127 load
    makes that flag exactly one. Summing flags implements exact rational
    half-even quantization independently of binary32 scale approximation.
    """
    ratio = _rational(attrs)
    relu = attrs.get('activation', 'none') == 'relu'
    if output_dtype == 'i32':
        if ratio != 1 or attrs.get('zero_point', 0):
            raise NotImplementedError('i32 readout must preserve integer scale')
        if not relu:
            def copy_block(start, rows):
                transfer(g, g.gep(source, g.mul(start, 4)),
                         g.gep(destination, g.mul(start, 4)), rows, 1, 4, 4, 'i32', 'i32')
            _tiles(g, count, copy_block)
            return
        # max(z,0) = sum_j clamp(z-127*j,0,127), retaining i32 sums.
        pieces = max(0, (high_bound + 126)//127)
        # Keep source growth bounded. These exact shifts are source constants,
        # never scalar values computed or loaded by the host kernel.
        if pieces > 65536:
            raise NotImplementedError('exact full-width ReLU exceeds bounded literal shift table')
        shifts = _literal_ptr(g, [-127*j for j in range(pieces)])
        flags = _scratch(g, 16)
        def relu_block(start, rows):
            src = g.gep(source,g.mul(start,4))
            _seed(g, 0, 0, rows, ACC+16)
            def piece(j):
                _load(g,src,rows,'i32',4,ACC)
                _seed(g,g.gep(shifts,g.mul(j,4)),0,rows,ACC|ACCUM)
                _store(g,flags,rows,'i8',1,ACC,1.0,True)
                _load(g,flags,rows,'i8',1,(ACC|ACCUM)+16)
            g.loop(pieces,piece)
            _store(g,g.gep(destination,g.mul(start,4)),rows,'i32',4,ACC+16)
        _tiles(g, count, relu_block)
        return
    if output_dtype != 'i8':
        raise NotImplementedError('integer readout dtype '+output_dtype)
    lo,hi = int(attrs.get('quant_min',-128)),int(attrs.get('quant_max',127))
    if not -128 <= lo <= hi <=127:
        raise NotImplementedError('readout clamp outside i8')
    zp = int(attrs.get('zero_point',0))
    bound = max(abs(low_bound),abs(high_bound))
    # This bridge is exact: conversion and product have <=24 significant
    # integer bits and the denominator is a binary power.
    dyadic = ratio.denominator & (ratio.denominator-1) == 0
    simple=(dyadic and bound!=0 and bound*abs(ratio.numerator)<1<<24 and not (relu and ratio<0))
    proved_scale=None
    if lo==-128 and hi==127 and zp==0 and not simple:
        proved_scale=_proved_readout_scale(ratio,low_bound,high_bound,relu)
    if lo == -128 and hi ==127 and zp ==0 and (simple or proved_scale is not None):
        scale=float(ratio) if simple else proved_scale
        def simple_block(start, rows):
            _load(g,g.gep(source,g.mul(start,4)),rows,'i32',4)
            _store(g,g.gep(destination,start),rows,'i8',1,ACC,scale,relu)
        _tiles(g, count, simple_block)
        return
    if ratio ==0:
        value=min(hi,max(lo,zp))
        seed=_constant_ptr(g,value)
        def zeros(start, rows):
            _seed(g,seed,value,rows)
            _store(g,g.gep(destination,start),rows,'i8',1)
        _tiles(g, count, zeros)
        return
    numerator,denominator=abs(ratio.numerator),ratio.denominator
    thresholds=[]
    initial=lo if ratio>0 else hi
    step=1 if ratio>0 else -1
    effective_low=max(0,low_bound) if relu else low_bound
    effective_high=max(0,high_bound) if relu else high_bound
    for t in range(lo,hi):
        q=t-zp
        v=Fraction((2*q+1)*denominator,2*numerator)
        if ratio>0:
            threshold= -((-v.numerator)//v.denominator) if q%2 else v.numerator//v.denominator+1
        else:
            v=-v
            threshold= -((-v.numerator)//v.denominator) if q%2==0 else v.numerator//v.denominator+1
        if threshold<=effective_low:
            initial+=step
        elif threshold<=effective_high:
            thresholds.append(threshold)
    if any(not -(1<<31) < low_bound+1-threshold < (1<<31) or
           not -(1<<31) < high_bound+1-threshold < (1<<31) for threshold in thresholds):
        raise NotImplementedError('threshold comparison would overflow i32')
    threshold_ptr=_literal_ptr(g,[1-t for t in thresholds])
    seed=_constant_ptr(g,initial)
    flags=_scratch(g,16)
    def block(start, rows):
        src=g.gep(source,g.mul(start,4))
        _seed(g,seed,initial,rows,ACC+16)
        def threshold(j):
            _load(g,src,rows,'i32',4)
            _seed(g,g.gep(threshold_ptr,g.mul(j,4)),0,rows,ACC|ACCUM)
            _store(g,flags,rows,'i8',1,ACC,127.0,True)
            config_ld(g,1,dtype='i8',scale=step/127.0)
            g.rocc(2,g.ptrint(flags),descriptor(g,(ACC|ACCUM)+16,rows,1))
            g.fence()
        g.loop(len(thresholds),threshold)
        _store(g,g.gep(destination,start),rows,'i8',1,ACC+16)
    _tiles(g, count, block)


def _nested_coords(g,shape,body):
    def axis(index,coords):
        if index==len(shape):
            body(coords);return
        g.loop(shape[index],lambda coordinate:axis(index+1,coords+[coordinate]))
    axis(0,[])


def _leading_broadcast_info(tensors,names,dst):
    """Prove each source is a suffix tensor repeated over leading axes."""
    output=list(tensors[dst]['shape']);result=[]
    for name in names:
        shape=list(tensors[name]['shape'])
        while shape and shape[0]==1:shape=shape[1:]
        if any(extent<=0 for extent in shape) or len(shape)>len(output):return None
        if shape and shape!=output[-len(shape):]:return None
        result.append((prod(shape),shape))
    return result


def _broadcast_blocks(g,tensors,names,dst,info,body):
    """Tile suffix rows without modulo coordinates or per-lane gathers."""
    output=list(tensors[dst]['shape']);count=prod(output)
    if all(elements in (1,count) for elements,_ in info):
        _tiles(g,count,lambda start,rows:body(start,rows,
               {name:0 if elements==1 else start for name,(elements,_) in zip(names,info)}))
        return
    period,shape=min((item for item in info if item[0]>1),key=lambda item:item[0])
    rank=len(shape);outer_shape=output[:-rank]
    def outer(coords):
        destination_base=g.mul(_flat(g,coords,outer_shape),period)
        def block(start,rows):
            offsets={}
            for name,(elements,source_shape) in zip(names,info):
                if elements==1:offsets[name]=0;continue
                prefix_rank=len(source_shape)-rank
                prefix_coords=coords[-prefix_rank:] if prefix_rank else []
                source_base=g.mul(_flat(g,prefix_coords,source_shape[:prefix_rank]),period)
                offsets[name]=g.add(source_base,start)
            body(g.add(destination_base,start),rows,offsets)
        _tiles(g,period,block)
    _nested_coords(g,outer_shape,outer)


def _affine(g,cb,c,ptrs):
    attrs=c.get('attributes',{})
    operands=c['operands']
    names=[operands['x']]+([operands['y']] if 'y' in operands else [])
    tensors=cb['tensors']; dst=operands['dst']
    count=prod(tensors[dst]['shape'])
    weights=[int(w) for w in attrs['weights']]
    if len(weights)!=len(names):return False
    broadcast_info=_leading_broadcast_info(tensors,[name for name,weight in zip(names,weights) if weight],dst)
    if broadcast_info is None:return False
    if any(_dtype(tensors[name]['dtype'])!='i8' for name in names):
        # Full-width unscaled operand sums are handled by accumulator loads.
        if all(_dtype(tensors[name]['dtype']) in ('i8','i32') for name in names) and all(_dtype(tensors[name]['dtype'])!='i32' or w==1 for name,w in zip(names,weights)) and int(attrs.get('constant',0))==0 and attrs.get('activation','none')=='none' and _rational(attrs)==1 and _dtype(tensors[dst]['dtype'])=='i32':
            local=dict(cb);local['tensors']=dict(tensors);pointers=dict(ptrs);summands=[]
            for j,(name,weight) in enumerate(zip(names,weights)):
                if _dtype(tensors[name]['dtype'])=='i32':summands.append(name);continue
                temporary='_operand_extension_'+str(g.counter)+'_'+str(j)
                local['tensors'][temporary]={'shape':tensors[dst]['shape'],'dtype':'i32','role':'intermediate'}
                pointers[temporary]=_scratch(g,count*4)
                mapping={'opcode':'AFFINE_MAP','operands':{'x':name,'dst':temporary},'attributes':{'weights':[weight],'constant':0,'activation':'none','readout_scale':1.0,'output_dtype':'i32'}}
                if not _affine(g,local,mapping,pointers):return False
                summands.append(temporary)
            return _sum_i32(g,local,c,pointers,summands)
        return False
    dtype=_dtype(tensors[dst]['dtype'])
    if dtype not in ('i8','i32'):return False
    slots=[]
    for name,weight in zip(names,weights):
        if abs(weight)>381:return False
        while weight:
            piece=min(127,weight) if weight>0 else max(-128,weight)
            slots.append((name,piece));weight-=piece
    constant=int(attrs.get('constant',0))
    lower=constant+sum((-128*w if w>=0 else 127*w) for w in weights)
    upper=constant+sum((127*w if w>=0 else -128*w) for w in weights)
    if lower<-(1<<31) or upper>=1<<31:return False
    identity=(dtype=='i32' and _rational(attrs)==1 and
              attrs.get('activation','none')=='none' and not attrs.get('zero_point',0))
    temp=ptrs[dst] if identity else _scratch(g,count*4)
    if slots:
        k=len(slots)
        coeff=_literal_ptr(g,[w for _,w in slots],'i8')
        bias=_constant_ptr(g,constant) if constant else None
        if k==1 and prod(tensors[slots[0][0]]['shape'])==count:
            matmul(g,ptrs[slots[0][0]],coeff,temp,count,1,1,{},'i32',bias)
            if not identity:
                _exact_readout(g,temp,ptrs[dst],count,attrs,lower,upper,dtype)
            return True
        packed=_scratch(g,16*k)
        slot_names=[name for name,_ in slots]
        slot_info=_leading_broadcast_info(tensors,slot_names,dst)
        def block(start,rows,offsets):
            for j,(name,_) in enumerate(slots):
                elements=prod(tensors[name]['shape'])
                transfer(g,g.gep(ptrs[name],offsets[name]),g.gep(packed,j),rows,1,
                         0 if elements==1 else 1,k,'i8','i8')
            matmul(g,packed,coeff,g.gep(temp,g.mul(start,4)),rows,k,1,{},'i32',bias)
        _broadcast_blocks(g,tensors,slot_names,dst,slot_info,block)
    else:
        seed=_constant_ptr(g,constant)
        def fill(start, rows):
            _seed(g,seed,constant,rows)
            _store(g,g.gep(temp,g.mul(start,4)),rows,'i32',4)
        _tiles(g, count, fill)
    if not identity:
        _exact_readout(g,temp,ptrs[dst],count,attrs,lower,upper,dtype)
    return True


def _sum_i32(g,cb,c,ptrs,names):
    tensors=cb['tensors'];dst=c['operands']['dst'];count=prod(tensors[dst]['shape'])
    if _dtype(tensors[dst]['dtype'])!='i32':return False
    broadcast_info=_leading_broadcast_info(tensors,names,dst)
    if broadcast_info is None:return False
    def block(start, rows):
        for j,name in enumerate(names):
            elements=prod(tensors[name]['shape'])
            if elements==count:
                _load(g,g.gep(ptrs[name],g.mul(start,4)),rows,'i32',4,ACC if j==0 else ACC|ACCUM)
            elif elements==1:
                _load(g,ptrs[name],rows,'i32',0,ACC if j==0 else ACC|ACCUM)
            else:
                # Keep a compact per-element loop when a channel vector repeats.
                raise NotImplementedError('channel-vector full-width sum uses scalar route')
        _store(g,g.gep(ptrs[dst],g.mul(start,4)),rows,'i32',4)
    if any(prod(tensors[name]['shape']) not in (1,count) for name in names):
        def repeated(start,rows,offsets):
            for j,(name,(elements,_)) in enumerate(zip(names,broadcast_info)):
                source=g.gep(ptrs[name],g.mul(offsets[name],4))
                _load(g,source,rows,'i32',0 if elements==1 else 4,ACC if j==0 else ACC|ACCUM)
            _store(g,g.gep(ptrs[dst],g.mul(start,4)),rows,'i32',4)
        _broadcast_blocks(g,tensors,names,dst,broadcast_info,repeated)
    else:_tiles(g, count, block)
    return True


def _coords(g,index,shape):
    coords=[]
    for axis,extent in enumerate(shape):
        stride=prod(shape[axis+1:])
        coords.append(g.urem(g.udiv(index,stride),extent))
    return coords


def _flat(g,coords,shape):
    result=0
    for coord,extent in zip(coords,shape):result=g.add(g.mul(result,extent),coord)
    return result


def _window_axis_groups(input_extent,output_extent,mode,kernel=0,stride=1,padding=0,include_pad=False):
    """Partition a static axis into affine starts and one exact window size.

    Interior windows use one compact loop. Border/adaptive partitions retain
    constant DMA lengths and denominators without decoding flat coordinates.
    """
    if min(input_extent,output_extent)<=0:return None
    if mode=='window':
        if kernel<=0 or stride<=0 or padding<0:return None
        begin=min(output_extent,max(0,(padding+stride-1)//stride))
        end=min(output_extent,max(begin,(input_extent+padding-kernel)//stride+1))
        if begin+output_extent-end>4096:return None
        result=[]
        def border(index):
            start=index*stride-padding
            lo=max(0,start);hi=min(input_extent,start+kernel)
            size=max(0,hi-lo)
            return (index,1,lo,1,size,kernel if include_pad else max(1,size))
        result.extend(border(index) for index in range(begin))
        if end>begin:result.append((begin,end-begin,begin*stride-padding,stride,kernel,kernel))
        result.extend(border(index) for index in range(end,output_extent))
        return result
    quotient,remainder=divmod(input_extent,output_extent)
    if not remainder:
        return [(0,output_extent,0,quotient,quotient,max(1,quotient))]
    if output_extent>4096:return None
    starts=[index*input_extent//output_extent for index in range(output_extent)]
    sizes=[((index+1)*input_extent+output_extent-1)//output_extent-starts[index] for index in range(output_extent)]
    result=[];begin=0
    while begin<output_extent:
        step=starts[begin+1]-starts[begin] if begin+1<output_extent and sizes[begin+1]==sizes[begin] else 1
        end=begin+1
        while end<output_extent and sizes[end]==sizes[begin] and starts[end]-starts[end-1]==step:end+=1
        result.append((begin,end-begin,starts[begin],step,sizes[begin],max(1,sizes[begin])))
        begin=end
    return result


def _vector_window_sum(g,cb,c,ptrs):
    """Accumulate whole channel/width lane blocks through strided DMA."""
    attrs=c.get('attributes',{});operands=c['operands'];tensors=cb['tensors']
    src,dst=operands['src'],operands['dst']
    ishape=list(tensors[src]['shape']);oshape=list(tensors[dst]['shape'])
    mode=attrs['mode'];layout=attrs.get('layout','nhwc');odtype=_dtype(tensors[dst]['dtype'])
    if len(ishape)!=len(oshape) or not 3<=len(ishape)<=5:return False
    if layout=='nchw':
        if ishape[:2]!=oshape[:2]:return False
        spatial_axes=list(range(2,len(ishape)))
    elif layout=='nhwc':
        if ishape[0]!=oshape[0] or ishape[-1]!=oshape[-1]:return False
        spatial_axes=list(range(1,len(ishape)-1))
    else:return False
    kernel=list(attrs.get('kernel',[]));stride=list(attrs.get('stride',[]))
    padding=list(attrs.get('padding',[0]*len(spatial_axes)))
    if len(padding)==2*len(spatial_axes):padding=padding[:len(spatial_axes)]
    if mode=='window' and (len(kernel)!=len(spatial_axes) or len(stride)!=len(spatial_axes) or len(padding)!=len(spatial_axes)):return False
    groups=[]
    for index,axis in enumerate(spatial_axes):
        axis_groups=_window_axis_groups(ishape[axis],oshape[axis],mode,
                                       kernel[index] if mode=='window' else 0,
                                       stride[index] if mode=='window' else 1,
                                       padding[index] if mode=='window' else 0,
                                       bool(attrs.get('count_include_pad',False)))
        if axis_groups is None:return False
        groups.append(axis_groups)
    # A fixed partition budget keeps irregular public geometry from generating
    # unbounded code; the generic loop fallback remains available beyond it.
    if prod(len(axis_groups) for axis_groups in groups)>4096:return False
    override=attrs.get('divisor_override')
    if override is not None and int(override)<=0:raise ValueError('window divisor must be positive')
    max_terms=prod(max(group[4] for group in axis_groups) for axis_groups in groups)
    if layout=='nchw':
        sites=ishape[0]*ishape[1]*prod(oshape[axis] for axis in spatial_axes[:-1])*((oshape[-1]+15)//16)*max(1,max_terms)
    else:
        sites=ishape[0]*prod(oshape[axis] for axis in spatial_axes)*((ishape[-1]+15)//16)*max(1,max_terms)
    static=sites<=256
    uniform_divisor=int(override) if override is not None else 1
    if override is None:
        for axis_groups in groups:
            denominators={group[5] for group in axis_groups}
            if len(denominators)!=1:
                uniform_divisor=None;break
            uniform_divisor*=next(iter(denominators))
    # A common denominator permits one tensor-wide readout, retaining a
    # single exact threshold loop even when spatial groups have short lanes.
    raw=_scratch(g,prod(oshape)*4 if uniform_divisor is not None else 64)
    def run(extent,body):
        if static:
            for index in range(extent):body(index)
        else:g.loop(extent,body)
    def tiles(extent,body):
        full,tail=divmod(extent,16)
        run(full,lambda index:body(g.mul(index,16),16))
        if tail:body(full*16,tail)
    def seed(rows,cols):
        config_ld(g,0,dtype='i32')
        g.rocc(2,0,descriptor(g,ACC,rows,cols));g.fence()
    def load(coords,rows,cols,row_stride):
        source=g.gep(ptrs[src],_flat(g,coords,ishape))
        config_ld(g,row_stride,dtype='i8')
        g.rocc(2,g.ptrint(source),descriptor(g,ACC|ACCUM,rows,cols));g.fence()
    def finish(output_coords,rows,cols,divisor,terms):
        index=_flat(g,output_coords,oshape)
        raw_destination=g.gep(raw,g.mul(index,4)) if uniform_divisor is not None else raw
        config_st(g,cols*4)
        g.rocc(3,g.ptrint(raw_destination),descriptor(g,ACC|FULL,rows,cols));g.fence()
        if uniform_divisor is not None:return
        ep=dict(attrs);ratio=_rational(attrs)/int(override if override is not None else divisor)
        ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
        destination=g.gep(ptrs[dst],g.mul(_flat(g,output_coords,oshape),4 if odtype=='i32' else 1))
        _exact_readout(g,raw,destination,rows*cols,ep,-128*terms,127*terms,odtype)
    def taps(starts,sizes,coords,axis_index,body):
        if axis_index==len(spatial_axes):body(coords);return
        axis=spatial_axes[axis_index]
        def term(index):
            selected=list(coords);selected[axis]=g.add(starts[axis_index],index)
            taps(starts,sizes,selected,axis_index+1,body)
        run(sizes[axis_index],term)
    def spatial(axis_index,coords,starts,sizes,divisor,body):
        if axis_index==len(spatial_axes):body(coords,starts,sizes,divisor);return
        axis=spatial_axes[axis_index]
        for begin,count,start,step,size,denominator in groups[axis_index]:
            # NCHW's final spatial axis is handled as vector width lanes.
            if layout=='nchw' and axis_index==len(spatial_axes)-1:
                def block(offset,lanes,begin=begin,start=start,step=step,size=size,denominator=denominator):
                    selected=list(coords);selected[axis]=g.add(begin,offset)
                    body(selected,starts+[g.add(start,g.mul(offset,step))],sizes+[size],divisor*denominator,lanes,step)
                tiles(count,block)
            else:
                def position(index,begin=begin,start=start,step=step,size=size,denominator=denominator):
                    selected=list(coords);selected[axis]=g.add(begin,index)
                    spatial(axis_index+1,selected,starts+[g.add(start,g.mul(index,step))],sizes+[size],divisor*denominator,body)
                run(count,position)
    if layout=='nchw':
        def sample(batch):
            def channel(channel):
                coords=[0]*len(ishape);coords[0]=batch;coords[1]=channel
                def block(output_coords,starts,sizes,divisor,lanes,step):
                    seed(lanes,1)
                    taps(starts,sizes,coords,0,lambda selected:load(selected,lanes,1,step))
                    finish(output_coords,lanes,1,divisor,prod(sizes))
                spatial(0,coords,[],[],1,block)
            run(ishape[1],channel)
        run(ishape[0],sample)
    else:
        def sample(batch):
            coords=[0]*len(ishape);coords[0]=batch
            def position(output_coords,starts,sizes,divisor):
                def channel(begin,lanes):
                    selected=list(coords);selected[-1]=begin
                    outcoords=list(output_coords);outcoords[-1]=begin
                    seed(1,lanes)
                    taps(starts,sizes,selected,0,lambda selected:load(selected,1,lanes,lanes))
                    finish(outcoords,1,lanes,divisor,prod(sizes))
                tiles(ishape[-1],channel)
            spatial(0,coords,[],[],1,position)
        run(ishape[0],sample)
    if uniform_divisor is not None:
        ep=dict(attrs);ratio=_rational(attrs)/uniform_divisor
        ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
        _exact_readout(g,raw,ptrs[dst],prod(oshape),ep,-128*max_terms,127*max_terms,odtype)
    return True


def _product(g,cb,c,ptrs):
    """Multiply byte lanes through a device-built diagonal contraction."""
    operands=c['operands'];tensors=cb['tensors']
    left,right,dst=operands['x'],operands['y'],operands['dst']
    x,y,z=(tensors[name] for name in (left,right,dst))
    if (_dtype(x['dtype'])!='i8' or _dtype(y['dtype'])!='i8' or
        _dtype(z['dtype'])!='i32' or x['shape']!=y['shape'] or x['shape']!=z['shape']):
        return False
    from movement import _fill
    diagonal=_scratch(g,256)
    def block(begin,terms):
        # Zero and diagonal writes both travel through Gemmini's DMA path.
        # The only nonzero in RHS column j is the corresponding Y lane j.
        _fill(g,diagonal,terms*terms,'i8',0)
        transfer(g,g.gep(ptrs[right],begin),diagonal,terms,1,1,terms+1,'i8','i8')
        source=g.gep(ptrs[left],begin)
        destination=g.gep(ptrs[dst],g.mul(begin,4))
        matmul(g,source,diagonal,destination,1,terms,terms,{},'i32')
    _tiles(g,prod(z['shape']),block)
    return True


def _reduce_i32(g,cb,c,ptrs):
    """Sum full-width source values with accumulator-load addition only."""
    attrs=c.get('attributes',{});operands=c['operands'];tensors=cb['tensors']
    src,dst=operands['src'],operands['dst']
    ishape=list(tensors[src]['shape']);oshape=list(tensors[dst]['shape'])
    if (_dtype(tensors[src]['dtype'])!='i32' or _dtype(tensors[dst]['dtype'])!='i32' or
        attrs.get('mode')!='reduce' or attrs.get('average',False) or
        _rational(attrs)!=1 or attrs.get('activation','none')!='none' or
        attrs.get('relu',False) or attrs.get('zero_point',0) or attrs.get('epilogue',[]) or
        'quant_min' in attrs or 'quant_max' in attrs):
        return False
    requested=list(attrs.get('dims',[]))
    if not ishape and requested:return False
    dims=sorted({int(axis)%len(ishape) for axis in requested})
    kept=[axis for axis in range(len(ishape)) if axis not in dims]
    count=prod(oshape);reduced=prod(ishape[axis] for axis in dims)
    if prod(ishape[axis] for axis in kept)!=count:return False
    static=count*max(1,reduced)<=256
    def run(extent,body):
        if static:
            for index in range(extent):body(index)
        else:g.loop(extent,body)
    def nested(axes,coords,body):
        if not axes:
            body(coords);return
        axis=axes[0]
        def one(index):
            selected=list(coords);selected[axis]=index
            nested(axes[1:],selected,body)
        run(ishape[axis],one)
    lane_axis=kept[-1] if kept else None
    lane_extent=ishape[lane_axis] if kept else 1
    lane_stride=4*prod(ishape[lane_axis+1:]) if kept else 4
    def output(base):
        def lanes(begin,rows):
            coords=list(base)
            if kept:coords[lane_axis]=begin
            _seed(g,0,0,rows)
            def term(selected):
                source=g.gep(ptrs[src],g.mul(_flat(g,selected,ishape),4))
                _load(g,source,rows,'i32',lane_stride,ACC|ACCUM)
            nested(dims,coords,term)
            output_coords=[coords[axis] for axis in kept]
            index=_flat(g,output_coords,[ishape[axis] for axis in kept])
            destination=g.gep(ptrs[dst],g.mul(index,4))
            _store(g,destination,rows,'i32',4)
        full,tail=divmod(lane_extent,16)
        run(full,lambda index:lanes(g.mul(index,16),16))
        if tail:lanes(full*16,tail)
    nested(kept[:-1],[0]*len(ishape),output)
    return True


def _window_sum(g,cb,c,ptrs):
    attrs=c.get('attributes',{});operands=c['operands'];tensors=cb['tensors']
    src,dst=operands['src'],operands['dst']
    ishape=list(tensors[src]['shape']);oshape=list(tensors[dst]['shape'])
    idtype=_dtype(tensors[src]['dtype']);odtype=_dtype(tensors[dst]['dtype'])
    if idtype=='i32':return _reduce_i32(g,cb,c,ptrs)
    if idtype!='i8' or odtype not in ('i8','i32'):return False
    mode=attrs['mode'];count=prod(oshape)
    if mode=='reduce':
        dims=sorted({int(d)%len(ishape) for d in attrs['dims']})
        kept=[d for d in range(len(ishape)) if d not in dims]
        reduced=prod(ishape[d] for d in dims)
        if prod(ishape[d] for d in kept)!=count:return False
        ep=dict(attrs)
        ratio=_rational(attrs)/(reduced if attrs.get('average',False) else 1)
        ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
        prefix=bool(dims) and dims==list(range(len(dims)))
        suffix=bool(dims) and dims==list(range(len(ishape)-len(dims),len(ishape)))
        if prefix or suffix:
            # A contiguous reduction needs no coordinate decomposition or
            # scalar gathers. Flattening a prefix gives [R,C], and flattening
            # a suffix gives [C,R]. The ordinary bounded matmul scheduler
            # performs the sums against a literal byte-one vector on Gemmini.
            label='@reduction_ones_'+str(g.counter)
            g.counter+=1
            g.globals.append(label+' = private constant ['+str(reduced)+
                             ' x i8] c"'+'\\01'*reduced+'", align 64')
            identity=(odtype=='i32' and ratio==1 and
                      attrs.get('activation','none')=='none' and
                      not attrs.get('zero_point',0))
            raw=ptrs[dst] if identity else _scratch(g,count*4)
            schedule={'_static_k_passes':count<=16}
            if prefix:
                matmul(g,label,ptrs[src],raw,1,reduced,count,schedule,'i32')
            else:
                matmul(g,ptrs[src],label,raw,count,reduced,1,schedule,'i32')
            if not identity:
                _exact_readout(g,raw,ptrs[dst],count,ep,-128*reduced,127*reduced,odtype)
            return True
        identity=(odtype=='i32' and ratio==1 and
                  attrs.get('activation','none')=='none' and not attrs.get('zero_point',0))
        raw=ptrs[dst] if identity else _scratch(g,count*4)
        if not kept:
            transfer(g,ptrs[src],raw,1,1,1,4,'i8','i32')
        elif dims and dims==list(range(dims[0],dims[-1]+1)):
            # A middle reduction is a batch of [1,R] @ [R,inner]
            # contractions; no output coordinate decomposition is needed.
            outer=prod(ishape[:dims[0]]);inner=prod(ishape[dims[-1]+1:])
            ones=_literal_ptr(g,[1]*reduced,'i8')
            def batch(index):
                source=g.gep(ptrs[src],g.mul(index,reduced*inner))
                destination=g.gep(raw,g.mul(index,inner*4))
                matmul(g,ones,source,destination,1,reduced,inner,{'_static_k_passes':count<=16},'i32')
            g.loop(outer,batch)
        else:
            # Disjoint reduction axes retain one contiguous logical output
            # lane block. Nested induction variables replace flattened
            # coordinates for both kept axes and reduction axes.
            lane_axis=kept[-1]
            lane_extent=ishape[lane_axis]
            lane_stride=prod(ishape[lane_axis+1:])
            static=count*reduced<=256
            def run(extent,body):
                if static:
                    for index in range(extent):body(index)
                else:g.loop(extent,body)
            def nested(axes,coords,body):
                if not axes:
                    body(coords);return
                axis=axes[0]
                def one(index):
                    selected=list(coords);selected[axis]=index
                    nested(axes[1:],selected,body)
                run(ishape[axis],one)
            def output(base):
                def lanes(begin,rows):
                    base=list(base_coords);base[lane_axis]=begin
                    _seed(g,0,0,rows)
                    def term(coords):
                        source=g.gep(ptrs[src],_flat(g,coords,ishape))
                        _load(g,source,rows,'i8',lane_stride,ACC|ACCUM)
                    nested(dims,base,term)
                    output_coords=[base[d] for d in kept]
                    destination=g.gep(raw,g.mul(_flat(g,output_coords,[ishape[d] for d in kept]),4))
                    _store(g,destination,rows,'i32',4)
                base_coords=base
                full,tail=divmod(lane_extent,16)
                run(full,lambda index:lanes(g.mul(index,16),16))
                if tail:lanes(full*16,tail)
            nested(kept[:-1],[0]*len(ishape),output)
        if not identity:
            _exact_readout(g,raw,ptrs[dst],count,ep,-128*reduced,127*reduced,odtype)
        return True
    if mode not in ('window','adaptive'):return False
    if _vector_window_sum(g,cb,c,ptrs):return True
    seed=_constant_ptr(g,0);raw=_scratch(g,count*4)
    layout=attrs.get('layout','nhwc')
    if layout=='nchw':spatial_axes=list(range(2,len(ishape)))
    elif layout=='nhwc':spatial_axes=list(range(1,len(ishape)-1))
    else:return False
    kernel=list(attrs.get('kernel',[]));stride=list(attrs.get('stride',[]));padding=list(attrs.get('padding',[0]*len(spatial_axes)))
    if mode=='window' and (len(kernel)!=len(spatial_axes) or len(stride)!=len(spatial_axes)):return False
    if mode=='window' and len(padding)==2*len(spatial_axes):padding=padding[:len(spatial_axes)]
    def output(oc):
        oi=_flat(g,oc,oshape);starts=[];sizes=[]
        for j,axis in enumerate(spatial_axes):
            if mode=='window':
                starts.append(g.sub(g.mul(oc[axis],stride[j]),padding[j]));sizes.append(kernel[j])
            else:
                a=g.udiv(g.mul(oc[axis],ishape[axis]),oshape[axis])
                z=g.udiv(g.add(g.mul(g.add(oc[axis],1),ishape[axis]),oshape[axis]-1),oshape[axis])
                starts.append(a);sizes.append(g.sub(z,a))
        _seed(g,seed,0,1)
        def term(taps):
            coords=list(oc);valid='true'
            for j,axis in enumerate(spatial_axes):
                value=g.add(starts[j],taps[j]);coords[axis]=value
                valid=g.value(f'and i1 {valid}, {g.icmp("ult",value,ishape[axis])}')
            g.conditional(valid,lambda:_load(g,g.gep(ptrs[src],_flat(g,coords,ishape)),1,'i8',1,ACC|ACCUM))
        _nested_coords(g,sizes,term)
        _store(g,g.gep(raw,g.mul(oi,4)),1,'i32',4)
    _nested_coords(g,oshape,output)
    # Constant denominator windows have one shared exact readout. Border-varying
    # denominators are handled below per output with address-only geometry.
    if mode=='window' and attrs.get('count_include_pad',False):
        divisor=int(attrs.get('divisor_override',prod(kernel)))
        ep=dict(attrs);ratio=_rational(attrs)/divisor
        ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
        _exact_readout(g,raw,ptrs[dst],count,ep,-128*prod(kernel),127*prod(kernel),odtype)
        return True
    # Use compile-time boundary classes, emitting one compact guarded loop per
    # distinct denominator rather than data-dependent tensor arithmetic.
    from itertools import product
    axis_counts=[]
    for j,axis in enumerate(spatial_axes):
        if mode=='adaptive':
            quotient,remainder=divmod(ishape[axis],oshape[axis])
            values={quotient} if remainder==0 else {quotient+1,quotient+2}
        else:
            edge=min(oshape[axis],kernel[j]+2)
            samples=set(range(edge)) | set(range(max(0,oshape[axis]-edge),oshape[axis]))
            samples.add(oshape[axis]//2)
            values=set()
            for o in samples:
                a=o*stride[j]-padding[j];z=a+kernel[j]
                values.add(max(0,min(z,ishape[axis])-max(0,a)))
        axis_counts.append(values)
    unique=sorted({max(1,prod(v)) for v in product(*axis_counts)})
    if 'divisor_override' in attrs:unique=[int(attrs['divisor_override'])]
    for divisor in unique:
        ep=dict(attrs);ratio=_rational(attrs)/divisor
        ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
        def narrow(oc,divisor=divisor,ep=ep):
            oi=_flat(g,oc,oshape);actual=1
            for j,axis in enumerate(spatial_axes):
                if mode=='window':a=g.sub(g.mul(oc[axis],stride[j]),padding[j]);z=g.add(a,kernel[j])
                else:a=g.udiv(g.mul(oc[axis],ishape[axis]),oshape[axis]);z=g.udiv(g.add(g.mul(g.add(oc[axis],1),ishape[axis]),oshape[axis]-1),oshape[axis])
                valid_start=g.max(a,0);valid_end=g.min(z,ishape[axis]);actual=g.mul(actual,g.max(0,g.sub(valid_end,valid_start)))
            if 'divisor_override' in attrs:actual=int(attrs['divisor_override'])
            g.conditional(g.icmp('eq',actual,divisor),lambda:_exact_readout(g,g.gep(raw,g.mul(oi,4)),g.gep(ptrs[dst],g.mul(oi,4 if odtype=='i32' else 1)),1,ep,-128*prod(ishape),127*prod(ishape),odtype))
        _nested_coords(g,oshape,narrow)
    return True



def _prefix_sum(g,cb,c,ptrs):
    operands=c['operands'];attrs=c.get('attributes',{});tensors=cb['tensors']
    src,dst=operands['src'],operands['dst']
    shape=list(tensors[src]['shape']);axis=int(attrs.get('axis',0))%len(shape)
    if _dtype(tensors[src]['dtype'])!='i8' or _dtype(tensors[dst]['dtype'])!='i32':return False
    if list(tensors[dst]['shape'])!=shape:return False
    outer,extent,inner=prod(shape[:axis]),shape[axis],prod(shape[axis+1:])
    if not extent or not inner or not outer:return True
    # Every block is a triangular matrix contraction. Its previous block's
    # final i32 row seeds all rows of the next contraction, so both source
    # growth and literal storage remain independent of the prefix extent.
    full,tail=divmod(extent,16)
    triangle16=_literal_ptr(g,[int(k<=row) for row in range(16) for k in range(16)],'i8') if full else None
    triangle_tail=_literal_ptr(g,[int(k<=row) for row in range(tail) for k in range(tail)],'i8') if tail else None
    def batch(batch_index):
        input_base=g.gep(ptrs[src],g.mul(batch_index,extent*inner))
        output_base=g.gep(ptrs[dst],g.mul(batch_index,extent*inner*4))
        def block(begin,rows,triangle,carry):
            source=g.gep(input_base,g.mul(begin,inner))
            destination=g.gep(output_base,g.mul(begin,inner*4))
            seed=g.gep(output_base,g.mul(g.sub(begin,1),inner*4)) if carry else None
            matmul(g,triangle,source,destination,rows,rows,inner,{},'i32',seed)
        if full:
            block(0,16,triangle16,False)
            g.loop(full-1,lambda index:block(g.mul(g.add(index,1),16),16,triangle16,True))
        if tail:
            block(full*16,tail,triangle_tail,bool(full))
    g.loop(outer,batch)
    return True

def emit_elementwise(g,cb,command,ptrs):
    """Emit one supported map/reduction; False means no implementation claimed."""
    op=command['opcode'];attrs=command.get('attributes',{});operands=command.get('operands',{})
    if op=='DEVICE_PRODUCT':return _product(g,cb,command,ptrs)
    if op=='DEVICE_PREFIX_SUM':return _prefix_sum(g,cb,command,ptrs)
    if op=='AFFINE_MAP':return _affine(g,cb,command,ptrs)
    if op=='BIAS_ADD':return _sum_i32(g,cb,command,ptrs,[operands['src'],operands['bias']])
    if op=='RESIDUAL_ADD':
        if int(attrs.get('bound_lsb',0))!=0:return False
        a,b=Fraction(float(attrs['lhs_scale'])),Fraction(float(attrs['rhs_scale']))
        from math import gcd
        common=a.denominator*b.denominator//gcd(a.denominator,b.denominator)
        wa,wb=a.numerator*(common//a.denominator),b.numerator*(common//b.denominator)
        factor=gcd(abs(wa),abs(wb)) or 1
        mapped={'opcode':'AFFINE_MAP','operands':{'x':operands['lhs'],'y':operands['rhs'],'dst':operands['dst']},'attributes':{'weights':[wa//factor,wb//factor],'constant':0,'activation':'relu' if 'relu' in attrs.get('epilogue',[]) else 'none','scale_numerator':str(factor),'scale_denominator':str(common),'output_dtype':attrs.get('output_dtype',cb['tensors'][operands['dst']]['dtype'])}}
        return _affine(g,cb,mapped,ptrs)
    if op=='WINDOW_SUM':return _window_sum(g,cb,command,ptrs)
    return False
