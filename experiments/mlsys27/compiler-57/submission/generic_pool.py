"""Source-affine maximum reductions using bounded hardware pooling trees."""
from math import prod
from device import transfer
from pooling import emit_maxpool
from host import _parse_map, _index


def emit_generic_pool(g,cb,c,ptrs):
    if c['opcode']!='DEVICE_POOL':return False
    o=c['operands'];a=c['attributes'];r=a['record']
    src=o['src'];dst=o['dst'];ish=cb['tensors'][src]['shape'];osh=cb['tensors'][dst]['shape']
    if cb['tensors'][src]['dtype']!='i8' or cb['tensors'][dst]['dtype']!='i8':return False
    if len(osh)!=2 or len(r['iterator_types'])!=3 or r['reduction_dims']!=[2]:return False
    if len(ish)!=2 or len(r['ins'])<2 or len(r['ins'][1]['shape'])!=1:return False
    mapping=_parse_map(r['indexing_maps'][0]);output_map=_parse_map(r['indexing_maps'][-1])
    from xdsl.ir.affine import AffineDimExpr
    if len(output_map.results)!=2 or any(not isinstance(d,AffineDimExpr) or d.position!=i for i,d in enumerate(output_map.results)):return False
    if not isinstance(mapping.results[-1],AffineDimExpr) or mapping.results[-1].position!=1:return False
    m,n=osh;reduced=prod(r['ins'][1]['shape'])
    if reduced<1:raise ValueError('empty maximum reduction')
    current=g.alloca(16);packed=g.alloca(64)
    pool_attrs={'pool_in_dims':[2,2],'pool_size':[2,2],'pool_stride':[2,2],'pool_padding':[0,0,0,0]}
    def row(ri):
        def tile(col,cols):
            for j in range(16):g.store(g.gep(current,j),-128,'i8')
            def chunk(ti):
                transfer(g,current,packed,1,16,16,16,'i8','i8')
                for j in range(3):
                    red=g.add(g.mul(ti,3),j)
                    target=g.gep(packed,(j+1)*16)
                    for q in range(16):g.store(g.gep(target,q),-128,'i8')
                    def gather(red=red,target=target):
                        index=_index(g,mapping,ish,[ri,col,red])
                        transfer(g,g.gep(ptrs[src],index),target,1,cols,ish[-1],16,'i8','i8')
                    g.conditional(g.icmp('ult',red,reduced),gather)
                emit_maxpool(g,packed,current,[4,16],[1,16],pool_attrs)
            g.loop((reduced+2)//3,chunk)
            output=g.gep(ptrs[dst],g.add(g.mul(ri,n),col))
            transfer(g,current,output,1,cols,16,n,'i8','i8')
        if n//16:g.loop(n//16,lambda ci:tile(g.mul(ci,16),16))
        if n%16:tile(n//16*16,n%16)
    g.loop(m,row)
    return True
