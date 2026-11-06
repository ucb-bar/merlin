"""Exact tensor movement through the accelerator accumulator DMA path.

Coordinates below come entirely from static tensor layouts and nested induction
variables. Scalar payloads are never read or written by the host.
"""
from itertools import product
from math import prod

DIM = 16
ACC = 0x80000000
FULL = 0x20000000
ONE = 0x3f800000


def _dtype(dtype):
    return {'int8':'i8','int16':'i16','int32':'i32','int64':'i64',
            'uint8':'u8','uint16':'u16','uint32':'u32','uint64':'u64'}.get(dtype, dtype)


def _bytes(dtype):
    return {'i1':1,'i8':1,'u8':1,'i16':2,'u16':2,'f16':2,'bf16':2,'i32':4,'u32':4,'f32':4,'i64':8,'u64':8,'f64':8}[_dtype(dtype)]


def _strides(shape):
    out = []
    stride = 1
    for n in reversed(shape):
        out.append(stride)
        stride *= n
    return out[::-1]


def _add(e, a, b):
    if a == 0:
        return b
    if b == 0:
        return a
    return e.add(a, b)


def _mul(e, a, b):
    if b == 0 or a == 0:
        return 0
    if b == 1:
        return a
    if a == 1:
        return b
    return e.mul(a, b)


def _config(e, source_stride, dest_stride, shrink):
    # ConfigMvinRs1: scale at 32, block stride at 16, shrink at 2,
    # config type LD at 0. Its rs2 is the DRAM row stride in bytes.
    e.issue(0, (ONE << 32) | (DIM << 16) | (1 << 8) | (int(shrink) << 2) | 1, source_stride)
    # ConfigMvoutRs1 type ST=2; ConfigMvoutRs2 scale bits and row stride.
    e.issue(0, 2, (ONE << 32) | dest_stride)


def _transfer(e, src, dst, src_offset, dst_offset, cols, rows, full):
    load = (rows << 48) | (cols << 32) | ACC
    store = load | (FULL if full else 0)
    e.issue(2, 0 if src is None else e.ptr(src, src_offset), load)
    e.issue(3, e.ptr(dst, dst_offset), store)
    # This also orders a following load that reuses these accumulator rows.
    e.fence()


def _linear_copy(e, src, dst, length, src_offset, dst_offset, source_width, dest_width, shrink, full):
    """Dense storage copy, up to 256 values per DMA pair with bounded code."""
    if length <= 0:
        return
    _config(e, DIM*source_width, DIM*dest_width, shrink)
    bulk = (length // (DIM*DIM)) * (DIM*DIM)
    def tile(i):
        _transfer(e, src, dst,
                  _add(e, src_offset, _mul(e, i, source_width)),
                  _add(e, dst_offset, _mul(e, i, dest_width)),
                  DIM, DIM, full)
    e.loop(0, bulk, DIM*DIM, tile)
    left = length-bulk
    rows = left//DIM
    if rows:
        _transfer(e, src, dst, _add(e, src_offset, bulk*source_width),
                  _add(e, dst_offset, bulk*dest_width), DIM, rows, full)
    tail = left % DIM
    if tail:
        at = bulk + rows*DIM
        _transfer(e, src, dst, _add(e, src_offset, at*source_width),
                  _add(e, dst_offset, at*dest_width), tail, 1, full)


def _copy_rect(e, src, dst, shape, source_strides, dest_strides,
               source_offset=0, dest_offset=0):
    """Copy a static rectangular affine map; offsets/strides are in elements."""
    source_dtype = _dtype(e.tensors[src]['dtype'])
    dest_dtype = _dtype(e.tensors[dst]['dtype'])
    shape = list(shape)
    ss, ds = list(source_strides), list(dest_strides)
    sw, dw = _bytes(source_dtype), _bytes(dest_dtype)
    source_offset *= sw
    dest_offset *= dw
    ss = [x*sw for x in ss]
    ds = [x*dw for x in ds]
    if source_dtype == 'i8' and dest_dtype == 'i32':
        shrink, full = True, True
    elif source_dtype == dest_dtype and sw == 4:
        # Raw 32-bit storage is unchanged, including floating NaN payloads.
        shrink, full = False, True
    elif source_dtype == dest_dtype:
        # Wider exact storage can be moved as bytes, without arithmetic.
        if sw > 1:
            shape.append(sw)
            ss.append(1)
            ds.append(1)
        sw = dw = 1
        shrink, full = True, False
    else:
        raise ValueError('unsupported movement storage conversion '+source_dtype+' -> '+dest_dtype)
    # Drop unit extents and merge compatible contiguous coordinate axes.
    axes = [(n,s,d) for n,s,d in zip(shape,ss,ds) if n != 1]
    i = len(axes)-2
    while i >= 0:
        a,b = axes[i], axes[i+1]
        if a[1] == b[0]*b[1] and a[2] == b[0]*b[2]:
            axes[i:i+2] = [(a[0]*b[0], b[1], b[2])]
        i -= 1
    if not axes:
        _linear_copy(e, src, dst, 1, source_offset, dest_offset, sw, dw, shrink, full)
        return
    if len(axes)==1 and axes[0][1] == sw and axes[0][2] == dw:
        _linear_copy(e, src, dst, axes[0][0], source_offset, dest_offset, sw, dw, shrink, full)
        return
    # A common contiguous axis supplies columns. Another positive-stride axis
    # supplies rows; zero source stride deliberately replays a source row.
    columns = [i for i,(_,s,d) in enumerate(axes) if s==sw and d==dw]
    ci = max(columns, key=lambda i:axes[i][0]) if columns else None
    row_axes = [i for i,(_,s,d) in enumerate(axes) if i != ci and s>=0 and d>=0]
    ri = max(row_axes, key=lambda i:axes[i][0]) if row_axes else None
    outer = [x for i,x in enumerate(axes) if i not in (ci,ri)]
    cn,cs,cd = axes[ci] if ci is not None else (1,0,0)
    rn,rs,rd = axes[ri] if ri is not None else (1,0,0)
    _config(e, rs, rd, shrink)
    def region(depth, so, do):
        if depth < len(outer):
            n,s,d = outer[depth]
            def each(iv):
                region(depth+1, _add(e,so,_mul(e,iv,s)), _add(e,do,_mul(e,iv,d)))
            e.loop(0,n,1,each)
            return
        def rows_loop(rv, nr):
            def cols_loop(cv, nc):
                _transfer(e,src,dst,
                          _add(e,_add(e,so,_mul(e,rv,rs)),_mul(e,cv,cs)),
                          _add(e,_add(e,do,_mul(e,rv,rd)),_mul(e,cv,cd)), nc,nr,full)
            cb = cn//DIM*DIM
            e.loop(0,cb,DIM,lambda cv:cols_loop(cv,DIM))
            if cn-cb:
                cols_loop(cb,cn-cb)
        rb = rn//DIM*DIM
        e.loop(0,rb,DIM,lambda rv:rows_loop(rv,DIM))
        if rn-rb:
            rows_loop(rb,rn-rb)
    region(0,source_offset,dest_offset)


def _identity(e, src, dst):
    shape = e.tensors[src]['shape']
    if prod(shape) != prod(e.tensors[dst]['shape']):
        raise ValueError('movement reshape changes element count')
    _copy_rect(e,src,dst,[prod(shape)],[1],[1])


def _literal_row(e, values, dtype):
    name='__movement_fill_'+str(len(e.tensors))
    while name in e.tensors or name in e.pointers:
        name+='_' 
    return e.global_data(name,values,dtype)


def _integer_byte_fill(e, dst, value, dtype):
    """Fill complete integer storage words by replaying one literal byte row.

    Every supported width divides the 16-byte row, so replay preserves the
    little-endian word phase without a runtime modulo or per-element cast.
    """
    width = _bytes(dtype)
    nbytes = prod(e.tensors[dst]['shape']) * width
    if nbytes == 0:
        return
    source = None
    if value != 0:
        word = value.to_bytes(width, 'little', signed=dtype.startswith('i'))
        pattern = word * (DIM // width)
        signed_bytes = [b if b < 128 else b - 256 for b in pattern]
        source = _literal_row(e, signed_bytes, 'i8')
    # Zero uses an unshrunk accumulator load from DRAM address zero. A literal
    # byte row uses signed-byte input-width loading; unit narrow readout is an
    # exact byte identity, including bytes whose high bit is set.
    _config(e, 0, DIM, source is not None)
    bulk = nbytes // (DIM*DIM) * (DIM*DIM)
    e.loop(0, bulk, DIM*DIM,
           lambda iv: _transfer(e, source, dst, 0, iv, DIM, DIM, False))
    remain = nbytes - bulk
    rows = remain // DIM
    if rows:
        _transfer(e, source, dst, 0, bulk, DIM, rows, False)
    tail = remain % DIM
    if tail:
        _transfer(e, source, dst, 0, bulk + rows*DIM, tail, 1, False)


def _fill(e, dst, value):
    dtype = _dtype(e.tensors[dst]['dtype'])
    if dtype in ('f16', 'bf16', 'f32', 'f64'):
        # The selected software domain admits exact literal storage movement,
        # but the frozen FILL command encodes only an integer value_code in
        # an integer output dtype. There is no floating literal-word field.
        # A Python number would lose signed zero and NaN payload identity;
        # reinterpreting an integer code as such a word invents semantics.
        raise ValueError('floating-storage FILL cannot be represented by the selected command-buffer ABI: value_code and output_dtype are integer-only; no exact literal storage-word field is declared')
    if dtype in ('i1', 'int1'):
        raise ValueError('one-bit FILL lacks a selected caller byte-storage definition')
    if dtype not in ('i8','i16','i32','i64','u8','u16','u32','u64'):
        raise ValueError('fill storage width unsupported: '+dtype)
    if type(value) is not int:
        raise ValueError('FILL requires an integer value_code in the selected command-buffer ABI')
    bits = _bytes(dtype) * 8
    minimum = -(1 << (bits-1)) if dtype.startswith('i') else 0
    maximum = (1 << (bits-1)) - 1 if dtype.startswith('i') else (1 << bits) - 1
    if not minimum <= value <= maximum:
        raise ValueError('FILL value_code is outside the destination integer storage domain')
    if dtype not in ('i8', 'i32'):
        return _integer_byte_fill(e, dst, value, dtype)
    width = _bytes(dtype)
    full = width == 4
    n = prod(e.tensors[dst]['shape'])
    if n == 0:
        return
    name = None
    if value != 0:
        name = _literal_row(e,[value]*DIM,dtype)
    # A zero address takes the zero-load path with shrink disabled even for
    # byte output. Nonzero rows are DMA literals, replayed with stride zero.
    shrink = name is not None and width == 1
    _config(e,0,DIM*width,shrink)
    bulk = n//(DIM*DIM)*(DIM*DIM)
    def bulk_tile(iv):
        _transfer(e,name,dst,0,_mul(e,iv,width),DIM,DIM,full)
    e.loop(0,bulk,DIM*DIM,bulk_tile)
    remain = n-bulk
    rows = remain//DIM
    if rows:
        _transfer(e,name,dst,0,bulk*width,DIM,rows,full)
    if remain%DIM:
        _transfer(e,name,dst,0,(bulk+rows*DIM)*width,remain%DIM,1,full)


def _movement(e,c):
    o,a = c['operands'],c.get('attributes',{})
    src,dst = o['src'],o['dst']
    source_shape = list(e.tensors[src]['shape'])
    shape = list(e.tensors[dst]['shape'])
    ss, ds = _strides(source_shape), _strides(shape)
    form = a.get('form','identity')
    scale = a.get('scale',1)
    if scale != 1 or ('scale_numerator' in a and int(a['scale_numerator']) != int(a['scale_denominator'])):
        raise ValueError('scaled movement requires arithmetic lowering')
    if form in ('identity','reshape','copy'):
        _identity(e,src,dst)
    elif form in ('permute','permutation'):
        p = a['perm']
        if len(p)!=len(shape) or sorted(p)!=list(range(len(source_shape))):
            raise ValueError('invalid movement permutation')
        _copy_rect(e,src,dst,shape,[ss[i] for i in p],ds)
    elif form == 'slice':
        offsets = a['offsets']
        strides = a.get('strides',[1]*len(source_shape))
        sizes = a.get('sizes',shape)
        # A rank-reducing slice retains its selected unit dimensions in the
        # affine copy map; the row-major destination still has the same extent.
        mapshape = list(sizes)
        if prod(mapshape) != prod(shape):
            raise ValueError('slice size/result mismatch')
        _copy_rect(e,src,dst,mapshape,[s*t for s,t in zip(ss,strides)],_strides(mapshape),sum(i*s for i,s in zip(offsets,ss)))
    elif form == 'as_strided':
        _copy_rect(e,src,dst,shape,a['strides'],ds,int(a.get('storage_offset',0)))
    elif form == 'broadcast':
        pad = len(shape)-len(source_shape)
        if pad < 0:
            raise ValueError('broadcast lowers rank')
        aligned = [1]*pad+source_shape
        aligned_strides = [0]*pad+ss
        for s,d in zip(aligned,shape):
            if s not in (1,d):
                raise ValueError('invalid broadcast extent')
        _copy_rect(e,src,dst,shape,[0 if n==1 else s for n,s in zip(aligned,aligned_strides)],ds)
    elif form == 'repeat':
        repeats = list(a['repeats'])
        pad = len(repeats)-len(source_shape)
        sshape = [1]*pad+source_shape
        sstrides = _strides(sshape)
        dims, srcst, dstst = [],[],[]
        for n,r,s,d in zip(sshape,repeats,sstrides,ds):
            dims.extend([r,n]); srcst.extend([0,s]); dstst.extend([n*d,d])
        _copy_rect(e,src,dst,dims,srcst,dstst)
    elif form == 'upsample':
        scales = a['scales']
        dims, srcst, dstst = [],[],[]
        for n,r,s,d in zip(source_shape,scales,ss,ds):
            dims.extend([n,r]); srcst.extend([s,0]); dstst.extend([r*d,d])
        _copy_rect(e,src,dst,dims,srcst,dstst)
    elif form == 'pad':
        low,high = list(a['low']),list(a['high'])
        mode = a.get('mode','constant')
        if mode == 'constant':
            _fill(e,dst,a['value_code'])
            _copy_rect(e,src,dst,source_shape,ss,ds,0,sum(i*s for i,s in zip(low,ds)))
        elif mode in ('reflect','replicate'):
            pieces=[]
            for n,l,h in zip(source_shape,low,high):
                axis=[]
                if l:
                    axis.append((l,l if mode=='reflect' else 0,-1 if mode=='reflect' else 0,0))
                axis.append((n,0,1,l))
                if h:
                    axis.append((h,n-2 if mode=='reflect' else n-1,-1 if mode=='reflect' else 0,l+n))
                pieces.append(axis)
            for selection in product(*pieces):
                extent=[p[0] for p in selection]
                srcst=[p[2]*s for p,s in zip(selection,ss)]
                so=sum(p[1]*s for p,s in zip(selection,ss))
                do=sum(p[3]*s for p,s in zip(selection,ds))
                _copy_rect(e,src,dst,extent,srcst,ds,so,do)
        else:
            raise ValueError('unknown pad mode '+str(mode))
    else:
        raise ValueError('unknown movement form '+str(form))


def _concat(e,c):
    o,a = c['operands'],c.get('attributes',{})
    if a.get('scales') and any(x!=1 for x in a['scales']):
        raise ValueError('scaled concat requires arithmetic lowering')
    dst = o['dst']; shape = e.tensors[dst]['shape']; ds = _strides(shape)
    axis = int(a['axis'])%len(shape)
    offset = 0
    sources = sorted(((int(k[3:]),v) for k,v in o.items() if k.startswith('src')))
    for _,src in sources:
        sshape=e.tensors[src]['shape']
        _copy_rect(e,src,dst,sshape,_strides(sshape),ds,0,offset*ds[axis])
        offset += sshape[axis]


def _insert(e,c):
    o,a = c['operands'],c.get('attributes',{})
    src,dst,dest = o['src'],o['dst'],o['dest']
    _identity(e,dest,dst)
    shape=e.tensors[src]['shape']; ds=_strides(e.tensors[dst]['shape'])
    _copy_rect(e,src,dst,shape,_strides(shape),[s*t for s,t in zip(ds,a['strides'])],0,sum(i*s for i,s in zip(a['offsets'],ds)))


def lower(e, command, state):
    """Emit one recognized movement command; return False for other families."""
    op=command['opcode']
    if op == 'MOVEMENT':
        _movement(e,command)
    elif op == 'FILL':
        _fill(e,command['operands']['dst'],command.get('attributes',{})['value_code'])
    elif op == 'CONCAT':
        _concat(e,command)
    elif op == 'INSERT_SLICE':
        _insert(e,command)
    else:
        return False
    return True
