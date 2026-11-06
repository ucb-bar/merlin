"""Conservative lifetime planning and DMA packing for reused dense byte RHSs."""
import struct
from rtl_isa import DIM, SCRATCHPAD_ROWS

_TRANSIENT_ROWS=2*DIM
_ALLOWED={'RES_PACK','EVICT','MATMUL','MATMUL_RESIDENT','COMMIT','FENCE'}
_ONE=0x3f800000


def _aligned_readout(attrs,n):
    scales=attrs.get('acc_scale_by_column')
    if scales is None:
        return True
    if not isinstance(scales,list) or len(scales)!=n:
        return False
    try:
        bits=[struct.pack('<f',float(v)) for v in scales]
    except (TypeError,ValueError,OverflowError):
        return False
    return all(i%DIM==0 or bits[i]==bits[i-1] for i in range(1,n))


def prepare(cb,tensors):
    """Return source-command identity -> an internal, nonoverlapping panel map.

    Only epochs with at least two actually read-out resident contractions are
    eligible. All code between pack and eviction belongs to this contraction
    lowerer and writes transient SPAD rows [0,2*DIM), or a separately allocated
    panel. Other opcode families conservatively retain the streaming path.
    """
    commands=cb.get('commands',[])
    live={};captures={};records=[];writers={}
    for index,command in enumerate(commands):
        op=command.get('opcode');operands=command.get('operands',{})
        for key,name in operands.items():
            if key.startswith('dst') and isinstance(name,str) and name in tensors:
                writers.setdefault(name,[]).append(index)
        if op=='RES_PACK':
            handle=operands.get('dst')
            if handle in live:
                live[handle]['end']=index-1
            record={'start':index,'end':len(commands)-1,'command':command,
                    'handle':handle,'tensor':operands.get('src'),'users':[]}
            records.append(record);live[handle]=record
        elif op=='EVICT':
            record=live.pop(operands.get('handle'),None)
            if record is not None:
                record['end']=index
        elif op in ('MATMUL','MATMUL_RESIDENT'):
            record=live.get(operands.get('rhs')) if op=='MATMUL_RESIDENT' else None
            if record is None:
                captures[operands.get('dst')]=None
            else:
                user={'index':index,'command':command,'commits':[]}
                record['users'].append(user)
                captures[operands.get('dst')]=(record,user)
        elif op=='COMMIT':
            capture=captures.get(operands.get('src'))
            if capture is not None:
                capture[1]['commits'].append(index)
    planned={};allocated=[]
    for record in records:
        source=tensors.get(record['tensor'],{})
        shape=source.get('shape',[])
        if source.get('dtype') not in ('i8','int8') or len(shape)!=2:
            continue
        k,n=shape
        if any(type(d) is not int or d<=0 for d in shape):
            continue
        if record['command'].get('attributes',{}).get('layout')!='packed_rhs':
            continue
        start,end=record['start'],record['end']
        if any(c.get('opcode') not in _ALLOWED for c in commands[start:end+1]):
            continue
        if any(start<=writer<=end for writer in writers.get(record['tensor'],[])):
            continue
        users=record['users']
        if len(users)<2 or any(not user['commits'] for user in users):
            continue
        eligible=True
        for user in users:
            command=user['command'];operands=command.get('operands',{})
            lhs=tensors.get(operands.get('lhs'),{})
            ashape=lhs.get('shape',[])
            if lhs.get('dtype') not in ('i8','int8') or len(ashape)!=2 or ashape[1]!=k:
                eligible=False;break
            if any(commit>end for commit in user['commits']):
                eligible=False;break
            last=max(user['commits'])
            if any(user['index']<writer<=last for writer in writers.get(operands.get('lhs'),[])):
                eligible=False;break
            for commit in user['commits']:
                # Match the actual deferred dispatch merge: the COMMIT may
                # override a producer attribute, or add a new partition or
                # orientation constraint. Every effective readout must keep
                # its N origins on the prepacked panel boundaries.
                effective=dict(command.get('attributes',{}),
                               **commands[commit].get('attributes',{}))
                if not _aligned_readout(effective,n) or any(effective.get(key) for key in ('transpose_rhs','transpose_b','transpose_lhs','transpose_a')):
                    eligible=False;break
            if not eligible:
                break
        if not eligible:
            continue
        k_tiles=(k+DIM-1)//DIM;n_tiles=(n+DIM-1)//DIM
        rows=k_tiles*n_tiles*DIM
        # SCRATCHPAD_ROWS is the selected RTL generator's conservative bank
        # depth. Keeping all allocations below it is valid even without using
        # the larger aggregate row capacity across the remaining banks.
        allocated=[p for p in allocated if p['end']>=start]
        base=_TRANSIENT_ROWS
        for other in sorted(allocated,key=lambda p:p['base']):
            if base+rows<=other['base']:
                break
            base=max(base,other['base']+other['rows'])
        if base+rows>SCRATCHPAD_ROWS:
            continue
        panel={'tensor':record['tensor'],'k':k,'n':n,'k_tiles':k_tiles,
               'n_tiles':n_tiles,'rows':rows,'base':base,'end':end}
        allocated.append(panel)
        planned[id(record['command'])]=panel
    return planned


def _tiles(e,extent,callback,unroll=False):
    if unroll:
        for tile,start in enumerate(range(0,extent,DIM)):
            callback(tile,start,min(DIM,extent-start))
        return
    full=extent//DIM
    if full:
        e.loop(0,full,1,lambda tile:callback(tile,e.mul(tile,DIM),DIM))
    if extent%DIM:
        callback(full,full*DIM,extent%DIM)


def source_address(e,source,offset):
    """Retain a declared allocation root through actual byte GEP addressing."""
    if source not in e.names or offset==0:
        return e.ptr(source,offset)
    index=e.const(offset)
    pointer=e.fresh()
    e.emit(f'{pointer} = llvm.getelementptr %arg{e.names.index(source)}[{index}] : (!llvm.ptr, i64) -> !llvm.ptr, i8')
    address=e.fresh()
    e.emit(f'{address} = llvm.ptrtoint {pointer} : !llvm.ptr to i64')
    return address


def pack(e,panel):
    """Load each logical RHS tile once into its allocated persistent rows."""
    source=panel['tensor'];k=panel['k'];n=panel['n']
    e.fence()
    e.issue(0,(_ONE<<32)|(DIM<<16)|(1<<8)|(1<<3)|1,n)
    # A bounded small panel emits literal DMA addresses and extents. Larger
    # matrices retain counted loops, so code size never grows with their volume.
    unroll=panel['k_tiles']*panel['n_tiles']<=DIM
    def columns(nt,ni,nr):
        def inner(kt,ki,kr):
            # Tile (nt,kt) owns exactly DIM rows. Its actual kr<=DIM rows
            # and nr<=DIM columns stay inside that allocation, including tails.
            local=e.add(panel['base'],e.mul(e.add(e.mul(nt,panel['k_tiles']),kt),DIM))
            offset=e.add(e.mul(ki,n),ni)
            descriptor=e.add(local,(nr<<32)|(kr<<48))
            e.issue(1,source_address(e,source,offset),descriptor)
        _tiles(e,k,inner,unroll)
    _tiles(e,n,columns,unroll)
    e.fence()


def address(e,panel,ki,ni):
    """Address a proven aligned panel without flattened coordinate division.

    ni=nt*DIM and ki=kt*DIM, so base+ni*k_tiles+ki equals
    base+(nt*k_tiles+kt)*DIM. Both source origins are supplied by the
    existing bounded contraction tile loops, with original K/N tail extents.
    """
    return e.add(panel['base'],e.add(e.mul(ni,panel['k_tiles']),ki))
