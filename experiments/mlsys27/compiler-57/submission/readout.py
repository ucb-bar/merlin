"""Standalone accumulator readouts and exact mixed-width integer sums."""
from fractions import Fraction
from math import prod,isfinite
import struct
from affine_reduction import _load,_zero,_store,_weighted,_admit_readout,DIM,ACC,FULL,ONE,_pack,_width


def _bounds(e,name,state):
    return state.get('value_bounds',{}).get(name,(-128,127) if _width(e.tensors[name]['dtype'])==1 else (-(1<<31),(1<<31)-1))


def _parts(e,dst,a):
    n=e.tensors[dst]['shape'][-1] if e.tensors[dst]['shape'] else 1
    if 'acc_scale_by_column' in a:
        scales=a['acc_scale_by_column']
        if len(scales)!=n:return None
    else:scales=[a.get('acc_scale',a.get('readout_scale',1))]*n
    if 'scale_numerator' in a:
        scales=[Fraction(int(a['scale_numerator']),int(a['scale_denominator']))]*n
    runs=[];start=0
    for i in range(1,n+1):
        if i==n or scales[i]!=scales[start]:
            runs.append((start,i,scales[start]));start=i
    return n,runs


def _store_result(e,dst,offset,cols,scale,relu,bounds,state):
    width=_width(e.tensors[dst]['dtype'])
    if width==4 and relu:
        temp='_readout_relu_tile'
        e.tensors.setdefault(temp,{'shape':[DIM],'dtype':'i8','role':'intermediate'})
        _store(e,temp,0,1,cols,cols,scale,True)
        _load(e,temp,0,1,cols,accumulate=False)
        _store(e,dst,offset,1,cols,cols*4)
    else:
        _store(e,dst,offset,1,cols,cols*width,scale,relu)


def _commit(e,c,state):
    o,a=c['operands'],c.get('attributes',{})
    src,dst=o.get('src'),o.get('dst')
    if src not in e.tensors or dst not in e.tensors:return False
    sw,dw=_width(e.tensors[src]['dtype']),_width(e.tensors[dst]['dtype'])
    if sw not in (1,4) or dw not in (1,4):return False
    if prod(e.tensors[src]['shape'])!=prod(e.tensors[dst]['shape']):return False
    ep=a.get('epilogue',[])
    if any(s not in ('relu','acc_scale') for s in ep):return False
    relu='relu' in ep
    if 'acc_scale_tensor' in a:return False
    parts=_parts(e,dst,a)
    if parts is None:return False
    n,runs=parts
    bounds=_bounds(e,src,state)
    bound=int(a.get('input_bound',max(abs(x) for x in bounds)))
    for _,_,scale in runs:
        if not isfinite(float(scale)):return False
        if dw==4 and (scale!=1 or (relu and bounds[1]>127)):return False
        if dw==1 and not _admit_readout(e,dst,Fraction(scale),bound,a):return False
    def row(i):
        base=e.mul(i,n)
        for begin,end,scale in runs:
            def tile(j,cols):
                offset=e.add(base,j)
                _load(e,src,offset,1,cols,accumulate=False)
                _store_result(e,dst,offset,cols,scale,relu,bounds,state)
            full=(end-begin)//DIM*DIM
            if full:e.loop(begin,begin+full,DIM,lambda j:tile(j,DIM))
            if begin+full<end:tile(begin+full,end-begin-full)
    e.loop(0,prod(e.tensors[dst]['shape'])//n,1,row)
    state.setdefault('value_bounds',{})[dst]=(0,max(0,bounds[1])) if relu else bounds
    return True


def _residual(e,c,state):
    o,a=c['operands'],c.get('attributes',{})
    srcs=[o['lhs'],o['rhs']];dst=o['dst']
    weights=[a['lhs_scale'],a['rhs_scale']]
    if any(not isinstance(w,(int,float)) or int(w)!=w or abs(w)>381 for w in weights):return False
    weights=[int(w) for w in weights]
    if any(_width(e.tensors[n]['dtype']) not in (1,4) for n in srcs+[dst]):return False
    if any(_width(e.tensors[n]['dtype'])==4 and w not in (0,1) for n,w in zip(srcs,weights)):return False
    if _width(e.tensors[dst]['dtype'])!=4:return False
    count=prod(e.tensors[dst]['shape']);sizes=[prod(e.tensors[s]['shape']) for s in srcs]
    if any(count%x for x in sizes):return False
    period=min(sizes)
    if any(size not in (count,period) for size in sizes):return False
    bounds=[0,0]
    for name,w in zip(srcs,weights):
        low,high=_bounds(e,name,state)
        bounds[0]+=min(low*w,high*w);bounds[1]+=max(low*w,high*w)
    if bounds[0]<-(1<<31) or bounds[1]>(1<<31)-1:return False
    relu='relu' in a.get('epilogue',[])
    if relu and bounds[1]>127:return False
    def outer(i):
        base=e.mul(i,period)
        def tile(j,cols):
            offset=e.add(base,j)
            _zero(e,1,cols)
            for name,w,size in zip(srcs,weights,sizes):
                if not w:continue
                index=j if size==period else offset
                if _width(e.tensors[name]['dtype'])==4:_load(e,name,index,1,cols,accumulate=True)
                else:_weighted(e,name,index,cols,w,state)
            _store_result(e,dst,offset,cols,Fraction(1),relu,bounds,state)
        full=period//DIM*DIM
        if full:e.loop(0,full,DIM,lambda j:tile(j,DIM))
        if full<period:tile(full,period-full)
    e.loop(0,count//period,1,outer)
    state.setdefault('value_bounds',{})[dst]=(0,max(0,bounds[1])) if relu else tuple(bounds)
    return True


def lower(e,c,state):
    # A COMMIT consumes an accumulator handle, which the contraction lowerer
    # owns. Ordinary tensors do not acquire accumulator identity by a load.
    if c['opcode']=='COMMIT':return False
    if c['opcode']=='RESIDUAL_ADD':return _residual(e,c,state)
    return False
