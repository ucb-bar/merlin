"""Byte contractions lowered to bounded weight-stationary Gemmini tiles.

Every mesh pass has K <= 16, bounding the full signed-byte dot product
inside the selected signed 20-bit PE result. Accumulation between passes uses
32-bit accumulator memory, and full-width stores bypass the scale unit.
"""
import math
import struct
from functools import wraps
from resident_panels import prepare as _prepare_panels, pack as _pack_panels, address as _panel_address, source_address as _source_address

DIM = 16
ACC = 1 << 31
ACCUMULATE = 1 << 30
FULL = 1 << 29
GARBAGE = (1 << 32) - 1
ONE = 0x3f800000


def _readout_owner(function):
    """Assign a deferred readout to the actual source commit operation."""
    @wraps(function)
    def owned(e,*args,**kwargs):
        source=getattr(e,'current_source',None)
        region=getattr(e,'current_region',None)
        stage=getattr(e,'current_stage',None)
        e.current_source=getattr(e,'_readout_source',source)
        e.current_region=getattr(e,'_readout_region',region)
        e.current_stage=getattr(e,'_readout_stage',stage)
        try:
            return function(e,*args,**kwargs)
        finally:
            e.current_source=source
            e.current_region=region
            e.current_stage=stage
    return owned


def _f32(v):
    return struct.unpack('<I', struct.pack('<f', float(v)))[0]


def _size(dtype):
    return {'i8': 1, 'int8': 1, 'i32': 4, 'int32': 4}.get(dtype)


def _offset(e, base, index, stride=1):
    return e.add(base, e.mul(index, stride))


def _word(address, rows, cols):
    return address | (cols << 32) | (rows << 48)


def _config(e, scale=1.0, relu=False):
    if not getattr(e, '_contraction_started', False):
        e.issue(7, 0, 0)
        e._contraction_started = True
    # Packed field offsets are retained from ConfigExRs1/ConfigExRs2.
    e.issue(0, (ONE << 32) | (1 << 16) | 4 | (int(relu) << 3), 1 << 48)


def _load(e, tensor, offset, local, rows, cols, stride, state=0, scale=1.0, dtype=None, typed_address=False):
    """Load a dense row block; offset/stride are in elements of tensor storage."""
    dt = dtype or e.tensors[tensor]['dtype']
    size = _size(dt)
    if size is None:
        raise ValueError('contraction load requires byte or signed i32 storage')
    shrink = size == 1 and bool(local & ACC)
    rs1 = (_f32(scale) << 32) | (DIM << 16) | (1 << 8) | (state << 3) | (int(shrink) << 2) | 1
    e.issue(0, rs1, e.mul(stride, size))
    byteoffset=e.mul(offset,size)
    pointer=_source_address(e,tensor,byteoffset) if typed_address else e.ptr(tensor,byteoffset)
    e.issue([2, 1, 14][state], pointer, _word(local, rows, cols))


def _zero(e, rows, cols):
    # Unshrunk full-i32 zero loads cannot inherit a preceding byte load state.
    e.issue(0, (ONE << 32) | (DIM << 16) | (1 << 8) | (2 << 3) | 1, 0)
    e.issue(14, 0, _word(ACC, rows, cols))
    e.fence()


def _compute(e, mr, kr, nr, accumulate=False, local=0, b_local=DIM):
    dest = ACC | (ACCUMULATE if accumulate else 0) | local
    b_word=_word(b_local,kr,nr) if isinstance(b_local,int) else e.add(b_local,(nr<<32)|(kr<<48))
    e.issue(6, b_word, _word(dest, mr, nr))
    e.issue(4, _word(0, mr, kr), _word(GARBAGE, mr, nr))


@_readout_owner
def _store(e, tensor, offset, rows, cols, stride, scale=1.0, relu=False, scale_bits=None, local=0):
    size = _size(e.tensors[tensor]['dtype'])
    if size not in (1, 4):
        raise ValueError('contraction output requires i8 or i32 storage')
    bits=_f32(scale) if scale_bits is None else scale_bits
    e.issue(0, 2 | (int(relu) << 2), e.add(e.mul(stride, size), e.mul(bits,1<<32)))
    e.issue(3, e.ptr(tensor, e.mul(offset, size)), _word(ACC | (FULL if size == 4 else 0) | local, rows, cols))
    e.fence()


def _axis(e, extent, callback):
    """One bounded loop for full tiles plus one statically sized tail."""
    full = extent // DIM * DIM
    if full:
        e.loop(0, full, DIM, lambda iv: callback(iv, DIM))
    if full != extent:
        callback(full, extent-full)


def _admit_output(e, dst, attrs):
    dtype = e.tensors[dst]['dtype']
    if _size(dtype) not in (1,4):
        return False
    stages = attrs.get('epilogue', [])
    if isinstance(stages,str):
        stages = [x.strip() for x in stages.split(',') if x.strip()]
    if any(s not in ('bias','bias_add','relu','acc_scale','maxpool') for s in stages):
        return False
    if any(k in attrs for k in ('scale_numerator','scale_denominator','zero_point','quant_min','quant_max','rounding')):
        return False
    scale = attrs.get('acc_scale',1.0)
    if not isinstance(scale,(float,int)) or not math.isfinite(scale):
        return False
    modes=sum(k in attrs for k in ('acc_scale','acc_scale_by_column','acc_scale_tensor'))
    if modes>1:
        return False
    scales=attrs.get('acc_scale_by_column',[scale])
    if not isinstance(scales,list) or not scales or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in scales):
        return False
    if 'acc_scale_by_column' in attrs and len(scales)!=e.tensors[dst]['shape'][-1]:
        return False
    if any(v<0 for v in scales) and 'relu' in stages:
        return False
    if 'acc_scale_tensor' in attrs:
        name=attrs['acc_scale_tensor']
        if name not in e.tensors or e.tensors[name]['dtype']!='f32' or math.prod(e.tensors[name]['shape'])!=1 or not hasattr(e,'load_scale_word'):
            return False
    if _size(dtype) == 4 and (any(v!=1 for v in scales) or 'relu' in stages or 'maxpool' in stages or 'acc_scale_tensor' in attrs):
        # Full i32 readout bypasses all activation and scaling hardware.
        return False
    return True


@_readout_owner
def _scale_parts(e,n,attrs):
    """Contiguous column runs having the exact same binary32 multiplier bits."""
    if 'acc_scale_tensor' in attrs:
        return [(0,n,e.load_scale_word(attrs['acc_scale_tensor']))]
    if 'acc_scale_by_column' not in attrs:
        return [(0,n,_f32(attrs.get('acc_scale',1.0)))]
    values=attrs['acc_scale_by_column']
    if len(values)!=n:
        raise ValueError('column readout scale extent differs from output columns')
    parts=[]
    start=0
    while start<n:
        bits=_f32(values[start])
        stop=start+1
        while stop<n and _f32(values[stop])==bits:
            stop+=1
        parts.append((start,stop,bits))
        start=stop
    return parts


def _columns(e,parts,callback):
    for start,stop,bits in parts:
        full=(stop-start)//DIM*DIM
        if full:
            e.loop(start,start+full,DIM,lambda ni: callback(ni,DIM,bits))
        if start+full!=stop:
            callback(start+full,stop-start-full,bits)


def _pool_valid(e,m,n,dst,attrs):
    """Admit only pool fields whose geometry and numerical ordering are proven."""
    if 'maxpool' not in attrs.get('epilogue',[]):
        return True
    if _size(e.tensors[dst]['dtype'])!=1:
        return False
    try:
        h,w=attrs['pool_in_dims']
        ph,pw=attrs['pool_size']
        sh,sw=attrs['pool_stride']
        padding=attrs.get('pool_padding',[0,0,0,0])
    except (KeyError,ValueError,TypeError):
        return False
    if any(not isinstance(v,int) for v in (h,w,ph,pw,sh,sw)):
        return False
    if h<ph or w<pw or max(h,w)>255 or min(h,w)<1 or m%(h*w):
        return False
    if ph!=pw or not 1<=ph<=3 or sh!=sw or not 1<=sh<=3 or padding!=[0,0,0,0]:
        return False
    scales=attrs.get('acc_scale_by_column',[attrs.get('acc_scale',1.0)])
    if any(v<0 for v in scales):
        return False
    oh=(h-ph)//sh+1; ow=(w-pw)//sw+1
    return math.prod(e.tensors[dst]['shape'])==m//(h*w)*oh*ow*n


@_readout_owner
def _pool_store(e,dst,offset,nr,stride,ph,pw,sh,scale_bits,relu):
    # Each group holds a complete source window at consecutive accumulator
    # rows. The output grid is exactly one element; all declared pool taps
    # are valid, with neither omitted initialization nor physical padding.
    rs1=2|(int(relu)<<2)|(sh<<4)|(ph<<6)|(1<<24)|(1<<32)|(1<<40)|(ph<<48)|(pw<<56)
    e.issue(0,rs1,e.add(stride,e.mul(scale_bits,1<<32)))
    e.fence()
    e.issue(3,e.ptr(dst,offset),_word(ACC,1,nr))
    e.fence()


def _matmul_pool(e,lhs,rhs,dst,m,k,n,attrs,a_offset,b_offset,c_offset,a_stride,b_stride,c_stride,seed,seed_offset,rhs_panel=None):
    h,w=attrs['pool_in_dims']; ph,pw=attrs['pool_size']; sh,sw=attrs['pool_stride']
    oh=(h-ph)//sh+1; ow=(w-pw)//sw+1
    parts=_scale_parts(e,n,attrs)
    _config(e)
    def batch(bi):
        def y(yi):
            def x(xi):
                target_row=e.add(e.mul(e.add(e.mul(bi,oh),yi),ow),xi)
                def col(ni,nr,scale_bits):
                    for py in range(ph):
                        for px in range(pw):
                            local=py*pw+px
                            source_row=e.add(e.mul(bi,h*w),e.add(e.mul(e.add(e.mul(yi,sh),py),w),e.add(e.mul(xi,sw),px)))
                            if seed is not None:
                                shape=e.tensors[seed]['shape']
                                so=ni if len(shape)==1 else e.add(seed_offset,e.add(e.mul(source_row,n),ni))
                                _load(e,seed,so,ACC|local,1,nr,0,state=2,typed_address=rhs_panel is not None)
                                e.fence()
                            def product(ki,kr,accumulate):
                                ao=e.add(a_offset,e.add(e.mul(source_row,a_stride),ki))
                                _load(e,lhs,ao,0,1,kr,a_stride,typed_address=rhs_panel is not None)
                                if rhs_panel is None:
                                    bo=e.add(b_offset,e.add(e.mul(ki,b_stride),ni))
                                    _load(e,rhs,bo,DIM,kr,nr,b_stride,state=1)
                                    b_local=DIM
                                else:
                                    b_local=_panel_address(e,rhs_panel,ki,ni)
                                _compute(e,1,kr,nr,accumulate,local,b_local)
                                e.fence()
                            product(0,min(k,DIM),seed is not None)
                            full=k//DIM*DIM
                            if full>DIM:
                                e.loop(DIM,full,DIM,lambda ki: product(ki,DIM,True))
                            if k>DIM and full!=k:
                                product(full,k-full,True)
                    target=e.add(c_offset,e.add(e.mul(target_row,c_stride),ni))
                    _pool_store(e,dst,target,nr,c_stride,ph,pw,sh,scale_bits,'relu' in attrs.get('epilogue',[]))
                _columns(e,parts,col)
            e.loop(0,ow,1,x)
        e.loop(0,oh,1,y)
    e.loop(0,m//(h*w),1,batch)


def matmul(e, lhs, rhs, dst, m, k, n, attrs=None, a_offset=0, b_offset=0,
           c_offset=0, a_stride=None, b_stride=None, c_stride=None, seed=None,
           transpose_b=False, seed_offset=None, rhs_panel=None):
    attrs = attrs or {}
    if transpose_b:
        raise ValueError('transposed RHS requires an explicit movement lowering')
    if e.tensors[lhs]['dtype'] not in ('i8','int8') or e.tensors[rhs]['dtype'] not in ('i8','int8'):
        raise ValueError('mesh contraction operands must have signed byte storage')
    if not _admit_output(e,dst,attrs):
        raise ValueError('contraction readout attributes require a separate exact lowering')
    if any(not isinstance(s,int) or s<=0 for s in (m,k,n)):
        raise ValueError('contraction extents must be positive static integers')
    if not _pool_valid(e,m,n,dst,attrs):
        raise ValueError('pool geometry or readout order has no admitted store lowering')
    a_stride = k if a_stride is None else a_stride
    b_stride = n if b_stride is None else b_stride
    c_stride = n if c_stride is None else c_stride
    seed_offset=c_offset if seed_offset is None else seed_offset
    if rhs_panel is not None and (rhs_panel['tensor']!=rhs or rhs_panel['k']!=k or rhs_panel['n']!=n or b_offset!=0 or b_stride!=n):
        rhs_panel=None
    if 'maxpool' in attrs.get('epilogue',[]):
        return _matmul_pool(e,lhs,rhs,dst,m,k,n,attrs,a_offset,b_offset,c_offset,a_stride,b_stride,c_stride,seed,seed_offset,rhs_panel)
    stages=attrs.get('epilogue',[])
    relu='relu' in stages
    parts=_scale_parts(e,n,attrs)
    _config(e)
    def row(mi,mr):
        def col(ni,nr,scale_bits):
            if seed is not None:
                shape=e.tensors[seed]['shape']
                if len(shape)==1:
                    so=ni
                    ss=0
                else:
                    so=e.add(seed_offset,e.add(e.mul(mi,n),ni))
                    ss=n
                _load(e,seed,so,ACC,mr,nr,ss,state=2,typed_address=rhs_panel is not None)
                e.fence()
            def product(ki,kr,accumulate):
                ao=e.add(a_offset,e.add(e.mul(mi,a_stride),ki))
                _load(e,lhs,ao,0,mr,kr,a_stride,typed_address=rhs_panel is not None)
                if rhs_panel is None:
                    bo=e.add(b_offset,e.add(e.mul(ki,b_stride),ni))
                    _load(e,rhs,bo,DIM,kr,nr,b_stride,state=1)
                    b_local=DIM
                else:
                    b_local=_panel_address(e,rhs_panel,ki,ni)
                _compute(e,mr,kr,nr,accumulate,b_local=b_local)
                e.fence()
            first=min(k,DIM)
            product(0,first,seed is not None)
            full=k//DIM*DIM
            if full>DIM:
                e.loop(DIM,full,DIM,lambda ki: product(ki,DIM,True))
            if k>DIM and full!=k:
                product(full,k-full,True)
            co=e.add(c_offset,e.add(e.mul(mi,c_stride),ni))
            _store(e,dst,co,mr,nr,c_stride,relu=relu,scale_bits=scale_bits)
        _columns(e,parts,col)
    _axis(e,m,row)


def _direct(e, command):
    op=command['opcode']; o=command.get('operands',{}); a=command.get('attributes',{})
    if op=='ATTENTION_PV':
        lhs,rhs=o['p'],o['v']
    else:
        lhs,rhs=o.get('a',o.get('lhs')),o.get('w',o.get('rhs'))
    dst=o['dst']
    if lhs not in e.tensors or rhs not in e.tensors or dst not in e.tensors:
        return False
    ash=e.tensors[lhs]['shape']; bsh=e.tensors[rhs]['shape']
    if len(ash)<2 or len(bsh)<2 or ash[-1]!=bsh[-2]:
        return False
    m,k,n=ash[-2],ash[-1],bsh[-1]
    if ash[:-2] != bsh[:-2]:
        return False
    if not _admit_output(e,dst,a) or not _pool_valid(e,m,n,dst,a):
        return False
    if any(e.tensors[t]['dtype'] not in ('i8','int8') for t in (lhs,rhs)):
        return False
    batches=math.prod(ash[:-2])
    output_m=m
    if 'maxpool' in a.get('epilogue',[]):
        h,w=a['pool_in_dims']; ph,pw=a['pool_size']; sh,sw=a['pool_stride']
        output_m=m//(h*w)*((h-ph)//sh+1)*((w-pw)//sw+1)
    def batch(bi):
        matmul(e,lhs,rhs,dst,m,k,n,a,e.mul(bi,m*k),e.mul(bi,k*n),e.mul(bi,output_m*n),seed=o.get('bias',a.get('bias',a.get('accumulator_seed'))),seed_offset=e.mul(bi,m*n),rhs_panel=command.get('_rhs_panel'))
    if batches==1:
        batch(0)
    else:
        e.loop(0,batches,1,batch)
    return True


def _attention_qk(e,command):
    o=command.get('operands',{}); a=command.get('attributes',{})
    lhs=o.get('q'); rhs=o.get('k'); dst=o.get('dst')
    if any(t not in e.tensors for t in (lhs,rhs,dst)):
        return False
    ash=e.tensors[lhs]['shape']; bsh=e.tensors[rhs]['shape']
    if len(ash)!=2 or len(bsh)!=2 or ash[1]!=bsh[1] or not _admit_output(e,dst,a):
        return False
    if any(e.tensors[t]['dtype'] not in ('i8','int8') for t in (lhs,rhs)):
        return False
    m,k=ash; n=bsh[0]
    temporary='_qk_transpose_'+rhs+'_'+dst
    e.tensors[temporary]={'shape':[k,n],'dtype':'i8','role':'intermediate'}
    # A column gather is a rows-by-one transfer. Accumulator rows preserve
    # its order, and a one-byte DRAM store stride packs the transposed row.
    # This uses normal accumulator readout, never the forbidden spad-store.
    def column(ki):
        def row(ni,nr):
            source=e.add(e.mul(ni,k),ki)
            target=e.add(e.mul(ki,n),ni)
            _load(e,rhs,source,ACC,nr,1,k,state=2)
            _store(e,temporary,target,nr,1,1)
        _axis(e,n,row)
    e.loop(0,k,1,column)
    matmul(e,lhs,temporary,dst,m,k,n,a)
    return True


def _conv(e,command):
    o=command.get('operands',{}); a=command.get('attributes',{})
    src=o.get('ifm'); weight=o.get('weight'); dst=o.get('dst')
    if any(t not in e.tensors for t in (src,weight,dst)):
        return False
    if a.get('layout','nhwc')!='nhwc' or not _admit_output(e,dst,a) or 'maxpool' in a.get('epilogue',[]):
        return False
    if any(e.tensors[t]['dtype'] not in ('i8','int8') for t in (src,weight)):
        return False
    shape=e.tensors[src]['shape']
    if len(shape)!=4 or len(a.get('kernel',[]))!=4:
        return False
    bn,h,w,ci=shape; kh,kw,kci,co=a['kernel']
    if ci!=kci:
        return False
    sh,sw=a.get('stride',[1,1]); dh,dw=a.get('dilation',[1,1])
    pt,pl,pb,pr=a.get('padding',[0,0,0,0])
    ho=(h+pt+pb-dh*(kh-1)-1)//sh+1
    wo=(w+pl+pr-dw*(kw-1)-1)//sw+1
    if ho<=0 or wo<=0 or not hasattr(e,'cmp') or not hasattr(e,'select'):
        return False
    seed=o.get('bias',a.get('bias',a.get('accumulator_seed')))
    if seed is not None:
        if seed not in e.tensors or _size(e.tensors[seed]['dtype'])!=4:
            return False
        seed_shape=e.tensors[seed]['shape']
        if seed_shape!=[co] and math.prod(seed_shape)!=bn*ho*wo*co:
            return False
    if math.prod(e.tensors[weight]['shape'])!=kh*kw*ci*co:
        return False
    if math.prod(e.tensors[dst]['shape'])!=bn*ho*wo*co:
        return False
    # A tap's in-bounds output columns form one static interval. Partition
    # the output width at those interval endpoints, so every tap is either
    # valid for a complete row block or is a complete zero block. This
    # avoids per-element coordinates, select chains and command issue even
    # at the left/right borders. The number of regions depends on the
    # source kernel width, never on output tensor volume.
    valid=[]
    cuts={0,wo}
    for kx in range(kw):
        low=max(0,min(wo,-((kx*dw-pl)//sw)))
        high=max(0,min(wo,(w-1+pl-kx*dw)//sw+1))
        valid.append((low,high))
        cuts.update((low,high))
    cuts=sorted(cuts)
    regions=[(cuts[i],cuts[i+1]) for i in range(len(cuts)-1) if cuts[i]<cuts[i+1]]
    parts=_scale_parts(e,co,a)
    _config(e)
    def batch(bi):
        image_row_base=e.mul(bi,h)
        output_row_base=e.mul(bi,ho)
        def y(yi):
            output_y=e.mul(e.add(output_row_base,yi),wo)
            yorigin=e.mul(yi,sh)
            input_y=[]
            for ky in range(kh):
                iy=e.add(yorigin,ky*dh-pt)
                input_y.append((iy,e.cmp('sge',iy,0),e.cmp('slt',iy,h)))
            def region(start,stop):
                # In dense NHWC storage, consecutive unit-dilation taps
                # have consecutive channel spans. For a valid group starting
                # at kx, flattened channel j addresses
                #   ((image_y * W + output_x * sw + kx - pl) * ci) + j.
                # Advancing output_x by one adds sw*ci bytes, exactly the
                # existing multi-row DMA stride. The corresponding RHS row
                # is ((ky*kw+kx)*ci+j), so the same ordered products result.
                # Partition endpoints prove every grouped tap is horizontal
                # in-bounds for every output row in this region. Invalid
                # taps retain their original complete zero-load path.
                groups=[]
                kx=0
                while kx<kw:
                    low,high=valid[kx]
                    x_valid=start>=low and stop<=high
                    end=kx+1
                    if dw==1 and x_valid:
                        while end<kw:
                            next_low,next_high=valid[end]
                            if not (start>=next_low and stop<=next_high):
                                break
                            end+=1
                    groups.append((kx,(end-kx)*ci,x_valid))
                    kx=end
                def x(xi,mr):
                    row=e.add(output_y,xi)
                    source_x=e.mul(xi,sw)
                    def col(ni,nr,scale_bits):
                        if seed:
                            if e.tensors[seed]['shape']==[co]:
                                seed_offset=ni
                                seed_stride=0
                            else:
                                seed_offset=e.add(e.mul(row,co),ni)
                                seed_stride=co
                            _load(e,seed,seed_offset,ACC,mr,nr,seed_stride,state=2)
                            e.fence()
                        else:
                            _zero(e,mr,nr)
                        for ky in range(kh):
                            iy,y_low,y_high=input_y[ky]
                            image_y=e.mul(e.add(image_row_base,iy),w)
                            for kx,group_channels,x_valid in groups:
                                source_row=e.mul(e.add(image_y,e.add(source_x,kx*dw-pl)),ci) if x_valid else None
                                def channels(ki,kr):
                                    if x_valid:
                                        pointer=e.ptr(src,e.add(source_row,ki))
                                        pointer=e.select(y_low,pointer,0)
                                        pointer=e.select(y_high,pointer,0)
                                    else:
                                        pointer=0
                                    e.issue(0,(ONE<<32)|(DIM<<16)|(1<<8)|1,sw*ci)
                                    e.issue(2,pointer,_word(0,mr,kr))
                                    wi=e.add(e.mul(e.add((ky*kw+kx)*ci,ki),co),ni)
                                    _load(e,weight,wi,DIM,kr,nr,co,state=1)
                                    _compute(e,mr,kr,nr,True)
                                    e.fence()
                                # _axis bounds every mesh pass to K<=DIM,
                                # retaining the signed-20-bit partial-sum
                                # proof even when a group spans many taps.
                                _axis(e,group_channels,channels)
                        out=e.add(e.mul(row,co),ni)
                        _store(e,dst,out,mr,nr,co,relu='relu' in a.get('epilogue',[]),scale_bits=scale_bits)
                    _columns(e,parts,col)
                full=(stop-start)//DIM*DIM
                if full:
                    e.loop(start,start+full,DIM,lambda xi:x(xi,DIM))
                if start+full<stop:
                    x(start+full,stop-start-full)
            for start,stop in regions:
                region(start,stop)
        e.loop(0,ho,1,y)
    e.loop(0,bn,1,batch)
    return True


def lower(e,command,state):
    """Lower an ABI command. Return False before emission for unsupported routes."""
    op=command['opcode']; o=command.get('operands',{}); a=command.get('attributes',{})
    if op=='RES_PACK':
        if '_resident_panel_plan' not in state:
            state['_resident_panel_plan']=_prepare_panels(e.cb,e.tensors)
        panels=state.setdefault('_resident_panels',{})
        panels.pop(o['dst'],None)
        panel=state['_resident_panel_plan'].get(id(command))
        if panel is not None:
            _pack_panels(e,panel)
            panels[o['dst']]=panel
        state.setdefault('resident',{})[o['dst']]=o['src']
        return True
    if op=='EVICT':
        state.setdefault('_resident_panels',{}).pop(o.get('handle'),None)
        state.setdefault('resident',{}).pop(o.get('handle'),None)
        return True
    if op in ('MATMUL_RESIDENT','MATMUL'):
        rhs=state.get('resident',{}).get(o['rhs'],o['rhs'])
        state.setdefault('pending',{})[o['dst']]={
            'lhs':o['lhs'],'rhs':rhs,'attributes':a,
            'rhs_panel':state.get('_resident_panels',{}).get(o['rhs']) if op=='MATMUL_RESIDENT' else None,
            'source':getattr(e,'current_source',None),
            'region':getattr(e,'current_region',None),
            'stage':getattr(e,'current_stage',None)}
        return True
    if op=='COMMIT':
        pending=state.get('pending',{}).get(o.get('src'))
        if pending is None:
            return False
        direct={'opcode':'MATMUL_DIRECT','operands':{'lhs':pending['lhs'],'rhs':pending['rhs'],'dst':o['dst']},'attributes':dict(pending['attributes'],**a),'_rhs_panel':pending.get('rhs_panel')}
        bias=o.get('bias',a.get('bias',a.get('accumulator_seed')))
        if bias:
            direct['operands']['bias']=bias
        source=getattr(e,'current_source',None)
        region=getattr(e,'current_region',None)
        stage=getattr(e,'current_stage',None)
        missing=object()
        previous_readout_source=getattr(e,'_readout_source',missing)
        previous_readout_region=getattr(e,'_readout_region',missing)
        previous_readout_stage=getattr(e,'_readout_stage',missing)
        e._readout_source=source
        e._readout_region=region
        e._readout_stage=stage
        e.current_source=pending['source']
        e.current_region=pending['region']
        e.current_stage=pending['stage']
        try:
            return _direct(e,direct)
        finally:
            e.current_source=source
            e.current_region=region
            e.current_stage=stage
            if previous_readout_source is missing:
                del e._readout_source
            else:
                e._readout_source=previous_readout_source
            if previous_readout_region is missing:
                del e._readout_region
            else:
                e._readout_region=previous_readout_region
            if previous_readout_stage is missing:
                del e._readout_stage
            else:
                e._readout_stage=previous_readout_stage
    if op in ('BATCHED_MATMUL','ATTENTION_PV','MATMUL_DIRECT'):
        return _direct(e,command)
    if op=='ATTENTION_QK':
        return _attention_qk(e,command)
    if op=='CONV2D':
        weight=state.get('resident',{}).get(o.get('weight'),o.get('weight'))
        return _conv(e,dict(command,operands=dict(o,weight=weight)))
    return False
