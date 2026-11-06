"""Outline stable repeated captured byte RHS tensors as resident handles.

The pass changes only residency commands and handle operands. Original tensor
buffers, declaration order, source identities and arithmetic commands retain
those identities. Physical panel admission remains the contraction lowerer's
separate capacity, lifetime and transfer proof.
"""
from bisect import bisect_right

_CONTRACTION_COMMANDS={'RES_PACK','EVICT','MATMUL','MATMUL_RESIDENT','COMMIT','FENCE'}
_PLAIN_ATTRIBUTES={'accumulator_seed'}


def _plain_dense(command,tensors,packed,encodings):
    if command.get('opcode')!='MATMUL':
        return False
    operands=command.get('operands',{})
    if set(operands)!={'lhs','rhs','dst'}:
        return False
    lhs=tensors.get(operands['lhs'],{});rhs=tensors.get(operands['rhs'],{})
    if operands['rhs'] in packed or operands['rhs'] in encodings or 'base' in rhs:
        return False
    ashape=lhs.get('shape',[]);bshape=rhs.get('shape',[])
    if lhs.get('dtype') not in ('i8','int8') or rhs.get('dtype') not in ('i8','int8'):
        return False
    if len(ashape)!=2 or len(bshape)!=2 or ashape[1]!=bshape[0]:
        return False
    if any(type(d) is not int or d<=0 for d in ashape+bshape):
        return False
    # Every supported captured source is row-major. Unknown orientation or
    # stride declarations are deliberately left on the ordinary streaming path.
    if any(key not in _PLAIN_ATTRIBUTES for key in command.get('attributes',{})):
        return False
    return True


def normalize(cb):
    if cb.get('declined'):
        return cb
    commands=cb.get('commands',[]);tensors=cb.get('tensors',{})
    encodings=cb.get('params',{}).get('storage_encodings',{})
    if not isinstance(encodings,dict):
        # A non-map encoding has no established tensor-specific dense binding.
        return cb
    packed={c.get('operands',{}).get('src') for c in commands if c.get('opcode')=='RES_PACK'}
    writers={};used=set(tensors)
    for index,command in enumerate(commands):
        for key,name in command.get('operands',{}).items():
            if isinstance(name,str):
                used.add(name)
                if key.startswith('dst') and name in tensors:
                    writers.setdefault(name,[]).append(index)
    captures={};records=[]
    for index,command in enumerate(commands):
        operands=command.get('operands',{});opcode=command.get('opcode')
        if opcode in ('MATMUL','MATMUL_RESIDENT'):
            record=None
            if _plain_dense(command,tensors,packed,encodings):
                rhs=operands['rhs']
                record={'index':index,'command':command,'rhs':rhs,'commits':[],
                        'epoch':bisect_right(writers.get(rhs,[]),index)}
                records.append(record)
            captures[operands.get('dst')]=record
        elif opcode=='COMMIT':
            record=captures.get(operands.get('src'))
            if record is not None:
                record['commits'].append(index)
    groups={}
    for record in records:
        if not record['commits']:
            continue
        start=record['index'];end=max(record['commits']);rhs=record['rhs']
        if any(start<=i<=end for i in writers.get(rhs,[])):
            continue
        # Pending captures read their lhs at commit in the emitted program.
        # Retain only the same immutable operand epoch the existing lowerer uses.
        lhs=record['command']['operands']['lhs']
        if any(start<i<=end for i in writers.get(lhs,[])):
            continue
        groups.setdefault((rhs,record['epoch']),[]).append(record)
    before={};after={};replacements={};serial=0
    for (rhs,_epoch),users in groups.items():
        if len(users)<2:
            continue
        start=min(u['index'] for u in users)
        end=max(max(u['commits']) for u in users)
        if any(start<=i<=end for i in writers.get(rhs,[])):
            continue
        if any(c.get('opcode') not in _CONTRACTION_COMMANDS for c in commands[start:end+1]):
            continue
        # Every readout must preserve the ordinary dense RHS orientation.
        if any(any(key in commands[i].get('attributes',{}) for key in
                   ('transpose_rhs','transpose_b','transpose_lhs','transpose_a','rhs_stride','lhs_stride'))
               for user in users for i in user['commits']):
            continue
        while True:
            handle='_captured_rhs_'+str(serial);serial+=1
            if handle not in used:break
        used.add(handle)
        first=commands[start]
        metadata={key:value for key,value in first.items() if key.startswith('_')}
        before.setdefault(start,[]).append({'opcode':'RES_PACK','operands':{'src':rhs,'dst':handle},
                                           'attributes':{'layout':'packed_rhs'},**metadata})
        for user in users:
            original=user['command']
            replacements[user['index']]={**original,'opcode':'MATMUL_RESIDENT',
                                         'operands':{**original['operands'],'rhs':handle}}
        last_metadata={key:value for key,value in commands[end].items() if key.startswith('_')}
        after.setdefault(end,[]).append({'opcode':'EVICT','operands':{'handle':handle},**last_metadata})
    if replacements:
        outlined=[]
        for index,command in enumerate(commands):
            outlined.extend(before.get(index,[]))
            outlined.append(replacements.get(index,command))
            outlined.extend(after.get(index,[]))
        cb['commands']=outlined
    return cb
