"""Device readout from full-width integer tensors with scalar or column scales."""
from math import prod
from device import ACC, descriptor, config_ld, config_st, config_st_bits, fbits


def emit_readout(g,cb,c,ptrs):
    if c['opcode']!='DEVICE_READOUT':return False
    o=c['operands'];a=c.get('attributes',{})
    src=o['src'];dst=o['dst'];ts=cb['tensors'][src];td=cb['tensors'][dst]
    if ts['dtype'] not in ('i8','i32') or td['dtype']!='i8':return False
    shape=td['shape'];n=shape[-1] if shape else 1;m=prod(shape[:-1]) if shape else 1
    size=4 if ts['dtype']=='i32' else 1
    values=a.get('gain_values');gain=o.get('gain')
    if values is not None:
        if not isinstance(values,list):values=[values]
        if len(values) not in (1,n):raise ValueError('readout scale must be scalar or per column')
        if len(values)==1:values=values*n
        groups=[]
        for j,v in enumerate(values):
            if groups and fbits(groups[-1][2])==fbits(v):groups[-1][1]+=1
            else:groups.append([j,1,v])
    elif gain in ptrs:
        scale_n=prod(cb['tensors'][gain]['shape'])
        if scale_n not in (1,n):raise ValueError('readout scale must be scalar or per column')
        groups=None
    else:raise ValueError('readout scale is unbound')
    def block(row,col,rows,cols,constant=None,bits=None):
        source=g.gep(ptrs[src],g.mul(g.add(g.mul(row,n),col),size))
        dest=g.gep(ptrs[dst],g.add(g.mul(row,n),col))
        g.fence();config_ld(g,n*size,dtype=ts['dtype'])
        g.rocc(2,g.ptrint(source),descriptor(g,ACC,rows,cols));g.fence()
        if bits is None:config_st(g,n,constant,a.get('relu',False))
        else:config_st_bits(g,n,bits,a.get('relu',False))
        g.rocc(3,g.ptrint(dest),descriptor(g,ACC,rows,cols));g.fence()
    def rows(col,cols,constant=None,bits=None):
        if m//16:g.loop(m//16,lambda ri:block(g.mul(ri,16),col,16,cols,constant,bits))
        if m%16:block(m//16*16,col,m%16,cols,constant,bits)
    if groups is not None:
        for start,count,value in groups:
            if count//16:g.loop(count//16,lambda ci,start=start,value=value:rows(g.add(start,g.mul(ci,16)),16,value))
            if count%16:rows(start+count//16*16,count%16,value)
    else:
        def column(j):
            bits=g.load(g.gep(ptrs[gain],g.mul(j if scale_n>1 else 0,4)),'i32')
            wide=g.value(f'zext i32 {bits} to i64')
            rows(j,1,bits=wide)
        g.loop(n,column)
    return True
