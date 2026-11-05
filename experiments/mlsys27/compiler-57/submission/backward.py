"""Exact adjoint window reductions and overlap fold on the device."""
from math import prod,lcm
from fractions import Fraction
from itertools import product
from device import ACC,ACCUM,FULL,config_ld,config_st,descriptor
from elementwise import _flat,_constant_ptr,_seed,_load,_store,_exact_readout,_rational


def _nested_coords(g,shape,body):
    """Carry source coordinates in induction variables, without unflattening."""
    def axis(index,coords):
        if index==len(shape):
            body(coords);return
        g.loop(shape[index],lambda coordinate:axis(index+1,coords+[coordinate]))
    axis(0,[])


def _transpose_axis_regions(source_extent,destination_extent,mode,kernel,stride,padding,include_pad):
    """Static intervals with exactly the same contributing source windows."""
    if min(source_extent,destination_extent)<=0 or source_extent>65536:return None
    events={0:[],destination_extent:[]}
    for index in range(source_extent):
        if mode=='adaptive_transpose':
            lo=index*destination_extent//source_extent
            hi=((index+1)*destination_extent+source_extent-1)//source_extent
        else:
            lo=index*stride-padding;hi=lo+kernel
        first=max(0,lo);end=min(destination_extent,hi)
        if end<=first:continue
        divisor=kernel if mode=='window_transpose' and include_pad else end-first
        events.setdefault(first,[]).append((index,divisor,True))
        events.setdefault(end,[]).append((index,divisor,False))
    points=sorted(events);active={};regions=[];sites=0
    for position,first in enumerate(points[:-1]):
        for index,divisor,add in events[first]:
            if add:active[index]=divisor
            else:active.pop(index,None)
        end=points[position+1]
        if end==first:continue
        contributors=tuple(sorted(active.items()))
        sites+=max(1,len(contributors))
        if sites>65536:return None
        regions.append((first,end,contributors))
    return regions


def _transpose_dma(g,cb,c,ptrs):
    """Broadcast each proven source window across a region of output lanes."""
    a=c.get('attributes',{});mode=a['mode'];o=c['operands'];src,dst=o['src'],o['dst']
    ss=list(cb['tensors'][src]['shape']);ds=list(cb['tensors'][dst]['shape'])
    layout=a.get('layout','nhwc')
    if len(ss)!=len(ds) or not 3<=len(ds)<=5:return False
    if layout=='nchw':
        if ss[:2]!=ds[:2]:return False
        axes=list(range(2,len(ds)))
    elif layout=='nhwc':
        if ss[0]!=ds[0] or ss[-1]!=ds[-1]:return False
        axes=list(range(1,len(ds)-1))
    else:return False
    kernel=list(a.get('kernel',[]));stride=list(a.get('stride',[]));padding=list(a.get('padding',[0]*len(axes)))
    if len(padding)==2*len(axes):padding=padding[:len(axes)]
    if mode=='window_transpose':
        if len(kernel)!=len(axes) or len(stride)!=len(axes) or len(padding)!=len(axes):return False
        if min(kernel+stride)<=0 or min(padding)<0:return False
    regions=[]
    for index,axis in enumerate(axes):
        observed=_transpose_axis_regions(ss[axis],ds[axis],mode,
                                       kernel[index] if mode=='window_transpose' else 0,
                                       stride[index] if mode=='window_transpose' else 1,
                                       padding[index] if mode=='window_transpose' else 0,
                                       bool(a.get('count_include_pad',False)))
        if observed is None:return False
        regions.append(observed)
    # Every emitted region and source-contributor site is included in this
    # product budget. Larger geometries retain the original bounded loops.
    if prod(sum(max(1,len(item[2])) for item in axis) for axis in regions)>65536:return False
    if 'divisor_override' in a:
        common=int(a['divisor_override'])
        if common<=0:return False
        maximum_weight=prod(max(len(item[2]) for item in axis) for axis in regions)
    else:
        # Actual source windows determine the exact denominator. Independent
        # axes factor the LCM of all Cartesian-product window sizes.
        axis_divisors=[{size for _,_,contributors in axis for _,size in contributors} for axis in regions]
        common=prod(lcm(*sizes) for sizes in axis_divisors)
        maximum=Fraction(common)
        for axis in regions:
            maximum*=max(sum((Fraction(1,size) for _,size in contributors),Fraction()) for _,_,contributors in axis)
        if maximum.denominator!=1:return False
        maximum_weight=maximum.numerator
    if common>1000000:raise NotImplementedError('backward window common denominator too large')
    bound=128*maximum_weight
    if bound>=1<<31:raise NotImplementedError('backward window weighted sum can exceed int32')
    raw=g.alloca(prod(ds)*4)
    def tiles(extent,body):
        full,tail=divmod(extent,16)
        if full:g.loop(full,lambda index:body(g.mul(index,16),16))
        if tail:body(full*16,tail)
    def block(dc,contributors,rows,channel,cols):
        dc=list(dc)
        if layout=='nhwc':dc[-1]=channel
        config_ld(g,0,dtype='i32')
        g.rocc(2,0,descriptor(g,ACC,rows,cols));g.fence()
        for source_windows in product(*contributors):
            sc=list(dc)
            for axis,(index,_) in zip(axes,source_windows):sc[axis]=index
            divisor=int(a['divisor_override']) if 'divisor_override' in a else prod(size for _,size in source_windows)
            coefficient=common//divisor
            pointer=g.gep(ptrs[src],_flat(g,sc,ss));address=g.ptrint(pointer)
            config_ld(g,0,dtype='i8')
            def add(_):
                g.rocc(2,address,descriptor(g,ACC|ACCUM,rows,cols));g.fence()
            g.loop(coefficient,add)
        destination=g.gep(raw,g.mul(_flat(g,dc,ds),4))
        config_st(g,4 if layout=='nchw' else ds[-1]*4)
        g.rocc(3,g.ptrint(destination),descriptor(g,ACC|FULL,rows,cols));g.fence()
    def spatial(index,dc,contributors):
        axis=axes[index]
        for first,end,active in regions[index]:
            if index==len(axes)-1:
                def width(offset,rows,first=first,active=active):
                    selected=list(dc);selected[axis]=g.add(first,offset)
                    roster=contributors+[active]
                    if layout=='nhwc':tiles(ds[-1],lambda channel,cols:block(selected,roster,rows,channel,cols))
                    else:block(selected,roster,rows,0,1)
                tiles(end-first,width)
            else:
                def coordinate(offset,first=first,active=active):
                    selected=list(dc);selected[axis]=g.add(first,offset)
                    spatial(index+1,selected,contributors+[active])
                g.loop(end-first,coordinate)
    def sample(batch):
        dc=[0]*len(ds);dc[0]=batch
        if layout=='nchw':
            def channel(index):
                selected=list(dc);selected[1]=index
                spatial(0,selected,[])
            g.loop(ds[1],channel)
        else:spatial(0,dc,[])
    g.loop(ds[0],sample)
    ep=dict(a);ratio=_rational(a)/common
    ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
    _exact_readout(g,raw,ptrs[dst],prod(ds),ep,-bound,bound,cb['tensors'][dst]['dtype'])
    return True


def emit_backward(g,cb,c,ptrs):
    if c['opcode']!='WINDOW_SUM':return False
    a=c.get('attributes',{});mode=a.get('mode')
    if mode not in ('window_transpose','adaptive_transpose','fold'):return False
    o=c['operands'];src=o['src'];dst=o['dst'];ss=cb['tensors'][src]['shape'];ds=cb['tensors'][dst]['shape']
    if cb['tensors'][src]['dtype']!='i8' or cb['tensors'][dst]['dtype'] not in ('i8','i32'):return False
    if mode=='fold':return _fold(g,cb,c,ptrs)
    if _transpose_dma(g,cb,c,ptrs):return True
    axes=list(range(2,len(ds))) if a.get('layout','nhwc')=='nchw' else list(range(1,len(ds)-1))
    kernel=a.get('kernel',[]);stride=a.get('stride',[]);padding=a.get('padding',[0]*len(axes))
    if len(padding)==2*len(axes):padding=padding[:len(axes)]
    classes=[]
    for j,ax in enumerate(axes):
        if mode=='adaptive_transpose':
            q,r=divmod(ds[ax],ss[ax]);counts={q} if r==0 else {q+1,q+2}
        elif a.get('count_include_pad',False):counts={kernel[j]}
        else:
            edge=min(ss[ax],kernel[j]+2)
            samples=set(range(edge))|set(range(max(0,ss[ax]-edge),ss[ax]))|{ss[ax]//2}
            counts={max(1,min(k*stride[j]-padding[j]+kernel[j],ds[ax])-max(k*stride[j]-padding[j],0)) for k in samples}
        classes.append(counts)
    divisors={int(a['divisor_override'])} if 'divisor_override' in a else {prod(v) for v in product(*classes)}
    common=lcm(*divisors)
    if common>1000000:raise NotImplementedError('backward window common denominator too large')
    zero=_constant_ptr(g,0);raw=g.alloca(prod(ds)*4)
    def output(dc):
        oi=_flat(g,dc,ds);_seed(g,zero,0,1)
        def term(source_spatial):
            sc=list(dc);inside='true';count=1
            for j,(ax,p) in enumerate(zip(axes,source_spatial)):
                sc[ax]=p
                if mode=='adaptive_transpose':
                    lo=g.udiv(g.mul(p,ds[ax]),ss[ax]);hi=g.udiv(g.add(g.mul(g.add(p,1),ds[ax]),ss[ax]-1),ss[ax])
                else:lo=g.sub(g.mul(p,stride[j]),padding[j]);hi=g.add(lo,kernel[j])
                contains=g.value(f'and i1 {g.icmp("sge",dc[ax],lo)}, {g.icmp("slt",dc[ax],hi)}')
                inside=g.value(f'and i1 {inside}, {contains}')
                size=kernel[j] if mode=='window_transpose' and a.get('count_include_pad',False) else g.sub(g.min(hi,ds[ax]),g.max(lo,0))
                count=g.mul(count,size)
            if 'divisor_override' in a:count=int(a['divisor_override'])
            def add():
                pointer=g.gep(ptrs[src],_flat(g,sc,ss))
                g.loop(g.udiv(common,count),lambda _: _load(g,pointer,1,'i8',1,ACC|ACCUM))
            g.conditional(inside,add)
        _nested_coords(g,[ss[ax] for ax in axes],term)
        _store(g,g.gep(raw,g.mul(oi,4)),1,'i32',4)
    _nested_coords(g,ds,output)
    ep=dict(a);ratio=_rational(a)/common;ep['scale_numerator']=str(ratio.numerator);ep['scale_denominator']=str(ratio.denominator)
    bound=128*prod(ss[ax] for ax in axes)*common
    _exact_readout(g,raw,ptrs[dst],prod(ds),ep,-bound,bound,cb['tensors'][dst]['dtype'])
    return True


def _fold(g,cb,c,ptrs):
    """Fold contiguous column-grid tiles using statically proved tap domains.

    Output coordinates are partitioned by stride residue and by tap-boundary
    intervals. Within a rectangle every contributing tap is known at compile
    time, and its grid address is affine in the nested induction variables.
    """
    from device import FULL, config_ld, config_st, descriptor
    from movement import _fill
    a=c['attributes'];o=c['operands']
    ss=cb['tensors'][o['src']]['shape'];ds=cb['tensors'][o['dst']]['shape']
    kernel=list(a['kernel']);stride=list(a['stride']);padding=list(a['padding'])
    dilation=list(a.get('dilation',[1]*len(kernel)))
    spatial=ds[2:];rank=len(spatial)
    if len(padding)==2*rank:
        padding=padding[:rank]
    if not rank or not all(len(v)==rank for v in (kernel,stride,padding,dilation)):
        raise ValueError('fold requires complete static spatial geometry')
    if any(v<=0 for v in kernel+stride+dilation):
        raise ValueError('fold kernel, stride and dilation must be positive')
    grid=[(d+2*p-di*(k-1)-1)//s+1 for d,p,di,k,s in zip(spatial,padding,dilation,kernel,stride)]
    if any(v<=0 for v in grid) or ss!=[ds[0],ds[1]*prod(kernel),prod(grid)]:
        raise ValueError('fold source geometry mismatch')
    dtype=cb['tensors'][o['dst']]['dtype']
    direct=(dtype=='i32' and _rational(a)==1 and
            a.get('activation','none')=='none' and not a.get('zero_point',0))
    raw=ptrs[o['dst']] if direct else g.alloca(prod(ds)*4)
    # Residues with no contributing tap, including dilation holes, stay zero.
    _fill(g,raw,prod(ds),'i32',0)
    grid_strides=[prod(grid[j+1:]) for j in range(rank)]
    out_strides=[prod(spatial[j+1:]) for j in range(rank)]

    def add(x,y):
        return x if y==0 else y if x==0 else g.add(x,y)
    def mul(x,y):
        return 0 if x==0 or y==0 else x if y==1 else y if x==1 else g.mul(x,y)

    # Each axis lists disjoint (residue, q-low, q-high, tap-index/offset)
    # intervals, with physical output coordinate residue + stride*q.
    axis_regions=[]
    for dim,k,step,pad,dil,grid_extent in zip(spatial,kernel,stride,padding,dilation,grid):
        residues=sorted({(tap*dil-pad)%step for tap in range(k)})
        regions=[]
        for residue in residues:
            if residue>=dim:
                continue
            count=(dim-1-residue)//step+1
            taps=[];boundaries={0,count}
            for tap in range(k):
                start=tap*dil-pad-residue
                if start%step:
                    continue
                qstart=start//step
                low=max(0,qstart);high=min(count,qstart+grid_extent)
                if low<high:
                    taps.append((tap,qstart,low,high))
                    boundaries.update((low,high))
            boundaries=sorted(boundaries)
            for low,high in zip(boundaries,boundaries[1:]):
                active=[(tap,qstart) for tap,qstart,tl,th in taps if tl<=low and high<=th]
                if active:
                    regions.append((residue,low,high,active))
        axis_regions.append(regions)
    regions=list(product(*axis_regions))
    # The emitted sites depend on kernel geometry, never on tensor area.
    sites=sum(prod(len(axis[3]) for axis in region) for region in regions)
    if sites>65536:
        raise NotImplementedError('fold static tap-domain partition exceeds bounded emission budget')
    static_coordinates=prod(ds)*prod(kernel)<=16384 and sites<=1024
    def loop(count,body):
        if static_coordinates and count<=16:
            for coordinate in range(count):
                body(coordinate)
        else:
            g.loop(count,body)

    def plane(n,channel):
        plane_index=add(mul(n,ds[1]),channel)
        output_plane=mul(plane_index,prod(spatial))
        input_plane=mul(plane_index,prod(kernel)*prod(grid))
        for region in regions:
            active_taps=list(product(*(axis[3] for axis in region)))
            matrix_rows=rank>=2 and stride[-1]==1
            row_axis=rank-2 if matrix_rows else rank-1
            outer_axes=list(range(row_axis))
            def tile(qcoords,rows,columns):
                output_index=output_plane
                for j,q in enumerate(qcoords):
                    coordinate=add(region[j][0],mul(q,stride[j]))
                    output_index=add(output_index,mul(coordinate,out_strides[j]))
                config_ld(g,0,dtype='i32')
                g.rocc(2,0,descriptor(g,ACC,rows,columns));g.fence()
                for taps in active_taps:
                    tap_index=0;source_index=input_plane
                    for j,(tap,offset) in enumerate(taps):
                        tap_index=tap_index*kernel[j]+tap
                        source_index=add(source_index,mul(g.sub(qcoords[j],offset) if offset else qcoords[j],grid_strides[j]))
                    source_index=add(source_index,tap_index*prod(grid))
                    config_ld(g,grid[-1] if matrix_rows else 1,dtype='i8')
                    g.rocc(2,g.ptrint(g.gep(ptrs[o['src']],source_index)),
                           descriptor(g,ACC|ACCUM,rows,columns));g.fence()
                output_row_stride=(stride[-2]*spatial[-1] if matrix_rows else stride[-1])*4
                config_st(g,output_row_stride)
                g.rocc(3,g.ptrint(g.gep(raw,mul(output_index,4))),
                       descriptor(g,ACC|FULL,rows,columns));g.fence()

            def blocks(axis_index,coords):
                low,high=region[axis_index][1:3]
                full,tail=divmod(high-low,16)
                def block(iv,length=16):
                    q=add(low,mul(iv,16))
                    if matrix_rows:
                        wlow,whigh=region[-1][1:3]
                        wfull,wtail=divmod(whigh-wlow,16)
                        def width(wiv,width=16):
                            tile(coords+[q,add(wlow,mul(wiv,16))],length,width)
                        loop(wfull,width)
                        if wtail:
                            width(wfull,wtail)
                    else:
                        tile(coords+[q],length,1)
                loop(full,block)
                if tail:
                    block(full,tail)
            def outer(depth,coords):
                if depth==len(outer_axes):
                    blocks(row_axis,coords);return
                low,high=region[depth][1:3]
                loop(high-low,lambda iv:outer(depth+1,coords+[add(low,iv)]))
            outer(0,[])
    loop(ds[0],lambda n:loop(ds[1],lambda channel:plane(n,channel)))
    if not direct:
        _exact_readout(g,raw,ptrs[o['dst']],prod(ds),a,-128*prod(kernel),127*prod(kernel),dtype)
    return True
