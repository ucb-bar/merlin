"""Agent-authored scheduling, pointer ABI and operation dispatch."""
from math import prod
import struct
from device import Emitter, matmul, config_ex, bias_stride
from movement import emit_movement
from convolution import emit_contraction

PUBLISHED = {'COMMIT','BIAS_ADD','RESIDUAL_ADD','ATTENTION_QK','ATTENTION_PV','CONV2D','BATCHED_MATMUL','MOVEMENT'}


def argument_names(cb):
    if cb.get('kernel_abi',{}).get('kind')=='whole_program':
        return [a['tensor'] for a in cb['kernel_abi']['args']]
    tensors=cb['tensors']
    produced={c.get('operands',{}).get('dst') for c in cb['commands']}
    derived=set()
    for key in ('derived_tensors','derivations'):
        v=cb.get('params',{}).get(key,{})
        if isinstance(v,dict): derived.update(v)
    inputs=[n for n,t in tensors.items() if n not in produced and n not in derived and t.get('role') not in ('output','intermediate')]
    outputs=[]
    for c in cb['commands']:
        n=c.get('operands',{}).get('dst')
        if c['opcode'] in PUBLISHED and n in tensors and n not in outputs:
            outputs.append(n)
    outputs.extend(n for n,t in tensors.items() if t.get('role')=='output' and n not in outputs)
    if 'outputs' in cb: outputs=[n for n in outputs if n in cb['outputs']]
    return inputs+outputs


def _bytes(t):
    dt=t['dtype']
    return prod(t['shape'])*({'i8':1,'i16':2,'i32':4,'i64':8,'f16':2,'bf16':2,'f32':4,'f64':8}.get(dt,1))


def _literal_bytes(values, dtype):
    """Pack immutable storage for the selected little-endian pointer ABI."""
    if dtype in ('i8','i16','i32','i64'):
        width = int(dtype[1:])
        return b''.join((int(value) % (1 << width)).to_bytes(width // 8, 'little')
                        for value in values)
    if dtype in ('f16','f32','f64'):
        format = {'f16':'<e', 'f32':'<f', 'f64':'<d'}[dtype]
        result = bytearray()
        for value in values:
            try:
                result.extend(struct.pack(format, float(value)))
            except OverflowError:
                result.extend(struct.pack(format, float('-inf') if value < 0 else float('inf')))
        return bytes(result)
    if dtype == 'bf16':
        result = bytearray()
        for value in values:
            # Round the parsed scalar directly, avoiding a binary32
            # intermediate which can round a BF16 midpoint twice.
            bits = struct.unpack('<Q', struct.pack('<d', float(value)))[0]
            sign = (bits >> 63) << 15
            exponent = (bits >> 52) & 0x7ff
            fraction = bits & ((1 << 52) - 1)
            if exponent == 0x7ff:
                narrowed = sign | 0x7f80 | (0x40 if fraction else 0)
            elif exponent == 0 or exponent - 1023 < -134:
                narrowed = sign
            else:
                power = exponent - 1023
                shift = 45 if power >= -126 else -power - 81
                mantissa = (1 << 52) | fraction
                quotient, remainder = divmod(mantissa, 1 << shift)
                half = 1 << (shift - 1)
                quotient += remainder > half or remainder == half and quotient & 1
                if power < -126:
                    narrowed = sign | quotient
                else:
                    if quotient == 256:
                        power += 1
                        quotient = 128
                    narrowed = sign | (0x7f80 if power > 127 else
                                       ((power + 127) << 7) | (quotient - 128))
            result.extend(narrowed.to_bytes(2, 'little'))
        return bytes(result)
    raise ValueError('literal storage type is not implemented: ' + dtype)


def _publish_commands(cb):
    """Retain every issued task in the public command vocabulary.

    Generic maps/reductions carry the precise implementation opcode and typed
    source record. They are still commands: neither a helper nor a literal
    producer may disappear into a private params instruction list.
    """
    kinds = {'HOST_LINALG':'VECTOR_MAP', 'DEVICE_READOUT':'VECTOR_MAP',
             'DEVICE_PREFIX_SUM':'VREDUCE', 'DEVICE_POOL':'VREDUCE',
             'DEVICE_CONSTANT':'FILL', 'DEVICE_PRODUCT':'VECTOR_MAP'}
    placements = []
    for c in cb['commands']:
        opcode = c['opcode']
        attrs = c.setdefault('attributes', {})
        host = opcode == 'HOST_LINALG' or opcode in ('SOFTMAX','RMSNORM','ROPE')
        if opcode in kinds:
            attrs['implementation_opcode'] = opcode
            c['opcode'] = kinds[opcode]
            if opcode == 'DEVICE_CONSTANT':
                literal = cb['params']['constant_tensors'][c['operands']['dst']]
                attrs['literal_values'] = literal['values']
                attrs['output_dtype'] = literal['dtype']
                attrs['op'] = 'literal_materialization'
                if len(literal['values']) == 1 and isinstance(literal['values'][0], int):
                    attrs['value_code'] = literal['values'][0]
                else:
                    # A dense literal is a generic materialization task, not
                    # the scalar integer FILL operation's uniform value.
                    c['opcode'] = 'VECTOR_MAP'
            elif opcode == 'DEVICE_PREFIX_SUM':
                attrs['op'] = 'prefix_sum'
            elif opcode == 'DEVICE_READOUT':
                attrs['op'] = 'accumulator_readout'
            elif opcode == 'DEVICE_PRODUCT':
                attrs['op'] = 'integer_product'
            elif opcode == 'HOST_LINALG':
                attrs['op'] = attrs.get('record', {}).get('kind', 'linalg.generic')
        # The frozen JSON schema uses the shared activation-stage list even
        # though the affine-map prose names a scalar kind. Preserve both.
        if isinstance(attrs.get('activation'), str):
            kind = attrs['activation']
            attrs['activation_kind'] = kind
            attrs['activation'] = [] if kind == 'none' else [kind]
        if opcode == 'FENCE':
            continue
        placement = {'lane':'host' if host else 'gemmini_mesh',
                     'route':'H' if host else 'A', 'op':c['opcode'],
                     'reason':('floating-point island outside the integer datapath' if host
                               else 'device DMA, accumulator or systolic instruction lowering')}
        for key in ('source_op_index','source_region_id'):
            if key in attrs:
                placement['region_id' if key == 'source_region_id' else key] = attrs[key]
        if placement not in placements:
            placements.append(placement)
    cb['params']['lane_placement'] = placements


def _declare_dependency_fences(cb):
    """Declare every DRAM producer/consumer barrier in the persisted stream."""
    pending = set()
    commands = []
    tensors = cb['tensors']
    for command in cb['commands']:
        if command['opcode'] == 'FENCE':
            commands.append(command)
            pending.clear()
            continue
        operands = command.get('operands', {})
        attributes = command.get('attributes', {})
        reads = {name for key, name in operands.items()
                 if not key.startswith('dst') and name in tensors}
        bias = attributes.get('bias')
        if isinstance(bias, str) and bias in tensors:
            reads.add(bias)
        if pending & reads:
            owner = {key:attributes[key] for key in ('source_op_index','source_region_id')
                     if key in attributes}
            commands.append({'opcode':'FENCE', 'operands':{}, 'attributes':owner})
            pending.clear()
        commands.append(command)
        pending.update(name for key, name in operands.items()
                       if key.startswith('dst') and name in tensors)
    cb['commands'] = commands


def compile_buffer(cb):
    if 'kernel_abi' not in cb:
        # Raw interface chains expose their terminal values. Named tensor
        # results consumed by later commands remain compiler-owned storage.
        produced = {c.get('operands', {}).get('dst') for c in cb['commands']}
        consumed = {name for c in cb['commands']
                    for key, name in c.get('operands', {}).items() if key != 'dst'}
        consumed.update(c.get('attributes', {}).get('bias') for c in cb['commands'])
        explicit_outputs = set(cb.get('outputs', []))
        for name in (produced & consumed & set(cb['tensors'])) - explicit_outputs:
            cb['tensors'][name]['role'] = 'intermediate'
        if 'outputs' not in cb:
            cb['outputs'] = [name for name, tensor in cb['tensors'].items()
                             if tensor.get('role') == 'output']
    if any('requant' in c.get('attributes',{}).get('epilogue',[]) for c in cb['commands']):
        cb['commands']=[]
        cb['declined']={'op':'requant','reason':'integer round-half-up requant is not admitted by this target readout contract'}
    if cb.get('declined'):
        if cb.get('kernel_abi',{}).get('kind')=='whole_program':
            # Refusing a source creates no compiler intermediates. Retain its
            # explicit input/output boundary, including every declared result,
            # without exposing unmaterialized payload values as ABI buffers.
            boundary={arg['tensor'] for arg in cb['kernel_abi']['args']}
            cb['tensors']={name:tensor for name,tensor in cb['tensors'].items()
                           if name in boundary}
            for key in ('constant_tensors','compiler_temporaries','source_values',
                        'global_program_plan','compiler_storage','lowered_instructions'):
                cb.get('params',{}).pop(key,None)
        return Emitter(argument_names(cb)).finish()
    from attribution import prepare, marker, annotate
    task_map=prepare(cb)
    _declare_dependency_fences(cb)
    cb['compiler_pointer_abi'] = {'version':1, 'arguments':argument_names(cb)}
    g=Emitter(argument_names(cb))
    g.fence()
    g.rocc(7,0,0)
    config_ex(g)
    g.fence()
    ptrs=dict(g.pointers)
    tensors=cb['tensors']
    # Residency follows repeated source SSA use, with a read-only boundary proof.
    # A scratchpad frame may outlive one contraction only when no command can
    # replace the tensor's contents during this invocation.
    lhs_uses={}
    written={name for command in cb['commands']
             for key,name in command.get('operands',{}).items()
             if key.startswith('dst')}
    if cb.get('kernel_abi',{}).get('kind')=='whole_program':
        readonly={arg['tensor'] for arg in cb['kernel_abi']['args']
                  if arg.get('access')=='read'}
    else:
        readonly=set(g.pointers)-written
    for command in cb['commands']:
        if command['opcode'] in ('MATMUL','MATMUL_RESIDENT'):
            lhs=command.get('operands',{}).get('lhs')
            lhs_uses[lhs]=lhs_uses.get(lhs,0)+1
    g.shared_input_pointers={ptrs[name] for name,uses in lhs_uses.items()
                             if uses>1 and name in readonly-written
                             and name in ptrs and tensors[name]['dtype']=='i8'}
    div_literals={t['tensor'] for t in cb.get('params',{}).get('compiler_temporaries',[])
                  if t.get('purpose')=='literal divisor folded into exact rational device readout'}
    for n,t in tensors.items():
        if n not in ptrs and n not in div_literals: ptrs[n]=g.alloca(max(64,_bytes(t)))
    constant_sources={}
    temporary_names={t['tensor'] for t in cb.get('params',{}).get('compiler_temporaries',[])}
    for name,constant in cb.get('params',{}).get('constant_tensors',{}).items():
        vals=constant['values'];dtype=constant['dtype'];count=prod(constant['shape'])
        if len(vals)==1 and count>1:vals=vals*count
        if len(vals)!=count:raise ValueError('constant tensor extent mismatch')
        data = _literal_bytes(vals, dtype)
        if any(data):
            label='@constant_'+str(g.counter);g.counter+=1
            body=''.join('\\'+format(value,'02X') for value in data)
            g.globals.append(label+' = private constant ['+str(len(data))+' x i8] c"'+body+'", align 64')
        else:
            label = None
        constant_sources[name]=label
        if name in temporary_names:
            ptrs[name]=label
    aliases={}
    pending={}
    records=[]
    current_task=None
    for idx,c in enumerate(cb['commands']):
        code=c['opcode']; op=c.get('operands',{}); attrs=c.get('attributes',{})
        before=len(g.lines)
        source=attrs.get('source_op_index')
        if task_map is not None and source in task_map and current_task!=task_map[source]:
            current_task=task_map[source]
            marker(g,current_task,source)
        if code=='DEVICE_CONSTANT':
            from movement import _copy_linear, _fill
            name=op['dst']
            if constant_sources[name] is None:
                _fill(g,ptrs[name],_bytes(tensors[name]),'i8',0)
            else:
                _copy_linear(g,constant_sources[name],ptrs[name],_bytes(tensors[name]),'i8','i8')
        elif code=='RES_PACK':
            aliases[op['dst']]=aliases.get(op['src'],op['src'])
        elif code in ('MATMUL','MATMUL_RESIDENT'):
            pending[op['dst']]=(op['lhs'],aliases.get(op['rhs'],op['rhs']),attrs)
        elif code=='COMMIT':
            if op['src'] not in pending:
                raise NotImplementedError('COMMIT source has no contraction '+op['src'])
            an,wn,ma=pending[op['src']]
            ash=tensors[an]['shape']; wsh=tensors[wn]['shape']
            if len(ash)!=2 or len(wsh)!=2 or ash[1]!=wsh[0]:
                raise ValueError('matmul requires [M,K] x [K,N]')
            if tensors[an]['dtype']!='i8' or tensors[wn]['dtype']!='i8':
                raise NotImplementedError('contraction operands must be i8')
            at={**ma,**attrs}
            bn=op.get('bias',attrs.get('bias'))
            if bn is not None and bn not in ptrs: raise ValueError('unknown bias '+bn)
            seed_stride = bias_stride(tensors[bn], ash[0], wsh[1]) if bn is not None else 0
            if 'maxpool' in at.get('epilogue',[]):
                from pooling import emit_maxpool
                pre=g.alloca(ash[0]*wsh[1])
                raw=dict(at);raw['epilogue']=[s for s in at['epilogue'] if s!='maxpool']
                matmul(g,ptrs[an],ptrs[wn],pre,ash[0],ash[1],wsh[1],raw,'i8',ptrs.get(bn),seed_stride)
                emit_maxpool(g,pre,ptrs[op['dst']],[ash[0],wsh[1]],tensors[op['dst']]['shape'],at)
            else:
                matmul(g,ptrs[an],ptrs[wn],ptrs[op['dst']],ash[0],ash[1],wsh[1],at,tensors[op['dst']]['dtype'],ptrs.get(bn),seed_stride)
        elif code=='EVICT':
            pass
        elif code=='FENCE':
            g.fence()
        else:
            from readout import emit_readout
            done=emit_readout(g,cb,c,ptrs)
            if not done:
                from generic_pool import emit_generic_pool
                done=emit_generic_pool(g,cb,c,ptrs)
            if not done: done=emit_movement(g,cb,c,ptrs)
            if not done: done=emit_contraction(g,cb,c,ptrs)
            if not done:
                from backward import emit_backward
                done=emit_backward(g,cb,c,ptrs)
            if not done:
                from elementwise import emit_elementwise
                done=emit_elementwise(g,cb,c,ptrs)
            if not done:
                from div_scalar import emit_quantized_div
                if code == 'DEVICE_QUANTIZED_DIV':
                    done=emit_quantized_div(g,cb,c,ptrs)
            if not done:
                from host import emit_host
                done=emit_host(g,cb,c,ptrs)
            if not done: raise NotImplementedError('no lowering for '+code)
        records.append({'command_index':idx,'line_start':before,'line_end':len(g.lines)})
    g.fence()
    lowered=g.finish()
    if task_map is not None:lowered=annotate(lowered,cb)
    temporary_names={t['tensor'] for t in cb.get('params',{}).get('compiler_temporaries',[])}
    _publish_commands(cb)
    if temporary_names:
        cb['params']['compiler_storage']={n:cb['tensors'].pop(n) for n in temporary_names}
        cb['params'].pop('compiler_temporaries',None)
    return lowered
