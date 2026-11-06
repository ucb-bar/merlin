"""Preserve exact initializer routes without changing the caller's source values.

Readonly per-column biases use the modeled matching COMMIT operand/attribute
route. A full matrix can remain an accumulator_seed only after proving its
original typed MAC-plus-bias SSA binding; that narrow case requests the complete
source whole-program boundary rather than misdeclaring a vector bias stage.
"""
_MATMULS=('MATMUL','MATMUL_RESIDENT')
_BIAS_STAGES=('bias','bias_add')


def _decline(cb,op,reason):
    cb['commands']=[]
    cb['declined']={'op':op,'reason':reason}
    return cb


def _matrix_shape(command,tensors,resident):
    operands=command.get('operands',{})
    lhs=tensors.get(operands.get('lhs'),{})
    rhs_name=operands.get('rhs')
    rhs=tensors.get(resident.get(rhs_name,rhs_name),{})
    ashape=lhs.get('shape',[]);bshape=rhs.get('shape',[])
    if len(ashape)!=2 or len(bshape)!=2:
        return None
    m,k=ashape;n=bshape[-1]
    if bshape[0]!=k or any(type(d) is not int or d<=0 for d in (m,k,n)):
        return None
    return m,n


def _is_i32_bias(name,shape,tensors):
    spec=tensors.get(name,{})
    if spec.get('dtype') not in ('i32','int32'):
        return False
    if shape is None:
        return False
    m,n=shape
    return spec.get('shape') in ([n],[m,n])


def _uniform_literal_value(value,ops):
    """Read only original typed scalar/dense literal definitions."""
    from xdsl.dialects.builtin import IntegerAttr,DenseIntOrFPElementsAttr
    owner=value.owner
    if owner not in ops or value not in owner.results:
        return None
    if owner.name in ('tensor.splat','linalg.fill'):
        if not owner.operands:
            return None
        scalar=owner.operands[0]
        scalar_owner=scalar.owner
        if scalar_owner not in ops or ops.index(scalar_owner)>=ops.index(owner) or str(scalar.type)!='i32':
            return None
        if scalar_owner.name!='arith.constant':
            return None
        attribute=scalar_owner.properties.get('value',scalar_owner.attributes.get('value'))
        if not isinstance(attribute,IntegerAttr):
            return None
        result=attribute.value.data
    elif owner.name=='arith.constant':
        attribute=owner.properties.get('value',owner.attributes.get('value'))
        if not isinstance(attribute,DenseIntOrFPElementsAttr):
            return None
        values=list(attribute.get_values())
        if not values or any(type(x) is not int or x!=values[0] for x in values):
            return None
        result=values[0]
    else:
        return None
    return result if type(result) is int and -(1<<31)<=result<(1<<31) else None


def _full_seed_source_binding(cb,command,seed,shape,resident,writers,command_index):
    """Prove an exact original readonly input or immutable literal seed epoch."""
    ops=cb.get('_source_ops');names=cb.get('_source_names');args=cb.get('_source_args')
    if not isinstance(ops,list) or not isinstance(names,dict) or not isinstance(args,list):
        return None
    index=command.get('_source_op_index')
    if type(index) is not int or not 0<=index<len(ops) or shape is None:
        return None
    spec=cb.get('tensors',{}).get(seed,{})
    if spec.get('shape')!=list(shape) or spec.get('dtype')!='i32':
        return None
    candidates=[value for value,name in names.items() if name==seed]
    if len(candidates)!=1:
        return None
    seed_value=candidates[0]
    typ=seed_value.type
    if not hasattr(typ,'get_shape') or not hasattr(typ,'element_type'):
        return None
    if list(typ.get_shape())!=list(shape) or str(typ.element_type)!='i32':
        return None
    binding={'accumulator':command['operands']['dst'],'seed_tensor':seed,'seed_shape':list(shape),
             'matmul_source_op_index':index,'matmul_source_result_index':0}
    if seed_value in args:
        if spec.get('role') not in ('input','weight','bias','scale') or seed in writers:
            return None
        binding.update({'seed_storage':'readonly_argument','seed_argument_index':args.index(seed_value)})
    else:
        if spec.get('role')!='intermediate':
            return None
        owner=seed_value.owner
        if owner not in ops or ops.index(owner)>=index:
            return None
        literal=_uniform_literal_value(seed_value,ops)
        if literal is None:
            return None
        seed_writes=[(i,c) for i,c in enumerate(cb.get('commands',[]))
                     if any(key.startswith('dst') and name==seed for key,name in c.get('operands',{}).items())]
        if len(seed_writes)!=1:
            return None
        materialization_index,materialization=seed_writes[0]
        attributes=materialization.get('attributes',{})
        if materialization_index>=command_index or materialization.get('opcode')!='FILL':
            return None
        if materialization.get('operands')!={'dst':seed} or materialization.get('_source_op_index')!=ops.index(owner):
            return None
        if attributes.get('output_dtype')!='i32' or type(attributes.get('value_code')) is not int or attributes['value_code']!=literal:
            return None
        binding.update({'seed_storage':'internal_literal','seed_source_op_index':ops.index(owner),
                        'seed_source_result_index':list(owner.results).index(seed_value),
                        'seed_materialization_command_index':materialization_index,'seed_literal_value':literal})
    from frontend_composition import recognize_mac,recognize_extension
    from frontend import _const,Unsupported
    source=ops[index]
    mac=recognize_mac(source)
    if mac is None or mac['transpose_rhs'] or len(source.results)!=1:
        return None
    operands=command.get('operands',{})
    rhs=operands.get('rhs');seen=set()
    while rhs in resident and rhs not in seen:
        seen.add(rhs);rhs=resident[rhs]
    if names.get(mac['lhs'])!=operands.get('lhs') or names.get(mac['rhs'])!=rhs:
        return None
    if mac['seed'] is seed_value:
        binding['source_seed_relation']='direct_initializer'
        return binding
    try:
        if _const(mac['seed'])!=0:
            return None
    except Unsupported:
        return None
    matches=[]
    for extension_index,operation in enumerate(ops):
        extension=recognize_extension(operation)
        if extension is not None and extension.get('bias') is seed_value and extension.get('accumulator') is source.results[0]:
            if extension_index<=index or len(operation.results)!=1:
                return None
            result_type=operation.results[0].type
            if list(result_type.get_shape())!=list(shape) or str(result_type.element_type)!='i32':
                return None
            matches.append(extension_index)
    if len(matches)!=1:
        return None
    binding.update({'source_seed_relation':'separate_integer_bias','extension_source_op_index':matches[0],
                    'extension_source_result_index':0})
    return binding


def _batched_full_seed_binding(cb,command,command_index):
    """Prove the complete original initializer of a canonical batched MAC."""
    from frontend_composition import recognize_batched_mac
    tensors=cb.get('tensors',{});ops=cb.get('_source_ops');names=cb.get('_source_names');args=cb.get('_source_args')
    if not isinstance(ops,list) or not isinstance(names,dict) or not isinstance(args,list):
        return None
    operands=command.get('operands',{});attrs=command.get('attributes',{})
    if set(operands)!={'a','w','dst'}:
        return None
    source_index=command.get('_source_op_index')
    if type(source_index) is not int or not 0<=source_index<len(ops):
        return None
    seed_name=attrs.get('accumulator_seed')
    if not isinstance(seed_name,str) or seed_name not in tensors:
        return None
    source=ops[source_index]
    mac=recognize_batched_mac(source)
    if mac is None or mac.get('transpose_rhs') or len(source.results)!=1:
        return None
    lhs,rhs,seed=mac['lhs'],mac['rhs'],mac['seed']
    if names.get(lhs)!=operands['a'] or names.get(rhs)!=operands['w'] or names.get(seed)!=seed_name or names.get(source.results[0])!=operands['dst']:
        return None
    if any(value is not seed and name==seed_name for value,name in names.items()):
        return None
    specs=[]
    for value,name,dtype in ((lhs,operands['a'],'i8'),(rhs,operands['w'],'i8'),(seed,seed_name,'i32'),(source.results[0],operands['dst'],'i32')):
        typ=value.type;spec=tensors.get(name,{})
        if not hasattr(typ,'get_shape') or not hasattr(typ,'element_type') or str(typ.element_type)!=dtype:
            return None
        shape=list(typ.get_shape())
        if len(shape)<3 or any(type(d) is not int or d<=0 for d in shape) or spec.get('dtype')!=dtype or spec.get('shape')!=shape:
            return None
        specs.append(shape)
    ashape,wshape,shape,dshape=specs
    if len({len(s) for s in specs})!=1 or ashape[:-2]!=wshape[:-2] or shape[:-2]!=ashape[:-2] or dshape!=shape:
        return None
    if ashape[-1]!=wshape[-2] or shape[-2:]!=[ashape[-2],wshape[-1]]:
        return None
    if set(attrs)-{'accumulator_seed','output_dtype','epilogue'} or attrs.get('output_dtype','i32')!='i32' or attrs.get('epilogue',[])!=[]:
        return None
    seed_writes=[(i,c) for i,c in enumerate(cb.get('commands',[]))
                 if any(key.startswith('dst') and name==seed_name for key,name in c.get('operands',{}).items())]
    binding={'command_opcode':'BATCHED_MATMUL','accumulator':operands['dst'],'seed_tensor':seed_name,
             'seed_shape':shape,'batch_shape':shape[:-2],'matmul_source_op_index':source_index,
             'matmul_source_result_index':0,'source_seed_relation':'direct_batched_initializer'}
    if seed in args:
        if tensors[seed_name].get('role') not in ('input','weight','bias','scale') or seed_writes:
            return None
        binding.update({'seed_storage':'readonly_argument','seed_argument_index':args.index(seed)})
    else:
        if tensors[seed_name].get('role')!='intermediate':
            return None
        producer=seed.owner
        if producer not in ops or ops.index(producer)>=source_index:
            return None
        literal=_uniform_literal_value(seed,ops)
        if literal is None or len(seed_writes)!=1:
            return None
        writer_index,writer=seed_writes[0]
        a=writer.get('attributes',{})
        if writer_index>=command_index or writer.get('opcode')!='FILL' or writer.get('_source_op_index')!=ops.index(producer):
            return None
        if writer.get('operands')!={'dst':seed_name} or a.get('output_dtype')!='i32' or type(a.get('value_code')) is not int or a['value_code']!=literal:
            return None
        binding.update({'seed_storage':'internal_literal','seed_source_op_index':ops.index(producer),
                        'seed_source_result_index':list(producer.results).index(seed),
                        'seed_materialization_command_index':writer_index,'seed_literal_value':literal})
    return binding


def _preload_routes(cb):
    """Bind recognized bias reads to the selected resident-contraction route."""
    tensors=cb.get('tensors',{})
    commands=cb.get('commands',[])
    records=[];live={};writers={};used_names=set(tensors)
    for index,command in enumerate(commands):
        operands=command.get('operands',{})
        for key,name in operands.items():
            if isinstance(name,str):
                used_names.add(name)
                if key.startswith('dst') and name in tensors:
                    writers.setdefault(name,[]).append(index)
        if command.get('opcode') in _MATMULS:
            record={'index':index,'command':command,'commits':[],'biases':set()}
            records.append(record)
            live[operands.get('dst')]=record
        elif command.get('opcode')=='COMMIT':
            record=live.get(operands.get('src'))
            if record is not None:
                record['commits'].append(index)
                if operands.get('bias') is not None:
                    record['biases'].add(operands['bias'])
    before={};after={};serial=0
    for record in records:
        if not record['biases']:
            continue
        for seed in record['biases']:
            spec=tensors[seed]
            if spec.get('role') in ('intermediate','output') or seed in writers:
                return _decline(cb,'accumulator_seed',
                    'Compiler has no selected bias-role preload declaration for an internally produced or writable i32 initializer; its internal storage cannot be exposed as a readonly external bias')
            if spec.get('role') not in ('input','weight','bias','scale'):
                return _decline(cb,'accumulator_seed',
                    'Readonly initializer has no declared external input role for a bias preload binding')
            # Dictionary insertion order and the read-only pointer boundary
            # are unchanged: input/weight/bias/scale are all logical inputs.
            spec['role']='bias'
        command=record['command']
        if command['opcode']=='MATMUL_RESIDENT':
            continue
        shape=_matrix_shape(command,tensors,{})
        if shape is None:
            return _decline(cb,'MATMUL','A biased generic contraction needs a proved rank-2 dense RHS before entering the selected resident preload route')
        index=record['index'];last=record['commits'][-1]
        operands=command['operands']
        for tensor in (operands['lhs'],operands['rhs']):
            if any(index<writer<=last for writer in writers.get(tensor,[])):
                return _decline(cb,'MATMUL','Contraction operand storage changes between capture and its final readout; resident packing cannot preserve that tensor epoch')
        while True:
            handle='_seed_resident_'+str(serial);serial+=1
            if handle not in used_names:
                break
        used_names.add(handle)
        metadata={key:value for key,value in command.items() if key.startswith('_')}
        before[index]={'opcode':'RES_PACK',
            'operands':{'src':operands['rhs'],'dst':handle},
            'attributes':{'layout':'packed_rhs'},**metadata}
        command['opcode']='MATMUL_RESIDENT'
        command['operands']={**operands,'rhs':handle}
        last_metadata={key:value for key,value in commands[last].items() if key.startswith('_')}
        after.setdefault(last,[]).append({'opcode':'EVICT','operands':{'handle':handle},**last_metadata})
    output=[]
    for index,command in enumerate(commands):
        if index in before:
            output.append(before[index])
        output.append(command)
        output.extend(after.get(index,[]))
    cb['commands']=output
    return cb


def normalize(cb):
    """Canonicalize bias operands and model safe deferred matmul initializers.

    Accumulator names and tensor storage names have independent epochs. A new
    MATMUL replaces only that live accumulator; a tensor write invalidates a
    captured initializer for later commits of the older accumulator.
    """
    if cb.get('declined'):
        return cb
    tensors=cb.get('tensors',{})
    resident={};pending={};epochs={}
    writers={name for c in cb.get('commands',[]) for key,name in c.get('operands',{}).items()
             if key.startswith('dst') and name in tensors}
    full_bindings=[]
    commands=[]
    for command_index,original in enumerate(cb.get('commands',[])):
        command=dict(original)
        operands=dict(command.get('operands',{}))
        attrs=dict(command.get('attributes',{}))
        opcode=command.get('opcode')
        if opcode=='RES_PACK':
            src=operands.get('src');seen=set()
            while src in resident and src not in seen:
                seen.add(src);src=resident[src]
            resident[operands.get('dst')]=src
        elif opcode=='EVICT':
            resident.pop(operands.get('handle'),None)
        elif opcode=='BATCHED_MATMUL' and attrs.get('accumulator_seed') is not None:
            binding=_batched_full_seed_binding(cb,command,command_index)
            if binding is None:
                return _decline(cb,opcode,'Batched initializer requires its exact original signed-byte MAC and full signed-i32 seed SSA, with equal static batch prefixes and an immutable readonly argument or sole preceding source-owned uniform device fill')
            full_bindings.append(binding)
        elif opcode in _MATMULS:
            handle=operands.get('dst')
            shape=_matrix_shape(command,tensors,resident)
            seed=attrs.pop('accumulator_seed',None)
            full_binding=None
            if seed is not None:
                if not isinstance(seed,str) or not _is_i32_bias(seed,shape,tensors):
                    return _decline(cb,opcode,'Matmul initializer must be a signed-i32 full result tensor or per-column vector with a proved contraction shape')
                if tensors[seed].get('role') in ('intermediate','output') and seed not in epochs:
                    return _decline(cb,opcode,'Matmul initializer is read before its tensor storage has been produced')
                if tensors[seed].get('shape')==list(shape):
                    full_binding=_full_seed_source_binding(cb,command,seed,shape,resident,writers,command_index)
                    if full_binding is None:
                        return _decline(cb,opcode,'Full-matrix accumulator seed lacks an exact original typed SSA binding and immutable readonly-input or device-materialized literal epoch for the whole-program caller boundary')
                    attrs['accumulator_seed']=seed
                    full_bindings.append(full_binding)
            pending[handle]={'seed':seed,'epoch':epochs.get(seed,0),'shape':shape,'full_binding':full_binding}
        elif opcode=='COMMIT':
            handle=operands.get('src')
            capture=pending.get(handle)
            # The command ABI binds bias as an operand; the selected preload
            # route also names its operand_attribute as "bias". Keep the two
            # identical bindings for one addition, never two bias stages.
            attribute_bias=attrs.get('bias')
            operand_bias=operands.get('bias')
            if attribute_bias is not None:
                if not isinstance(attribute_bias,str) or attribute_bias not in tensors:
                    return _decline(cb,opcode,'Commit bias attribute does not name a declared tensor')
                if operand_bias is not None and operand_bias!=attribute_bias:
                    return _decline(cb,opcode,'Commit has conflicting operand and attribute bias tensors')
                operands['bias']=attribute_bias
                operand_bias=attribute_bias
            stages=attrs.get('epilogue',[])
            if not isinstance(stages,list) or any(not isinstance(s,str) for s in stages):
                return _decline(cb,opcode,'Commit epilogue must be an ordered stage list')
            bias_stages=sum(s in _BIAS_STAGES for s in stages)
            if operand_bias is not None:
                if bias_stages!=1:
                    return _decline(cb,opcode,'Commit bias operand requires exactly one explicit bias stage')
                if capture is not None and not _is_i32_bias(operand_bias,capture['shape'],tensors):
                    return _decline(cb,opcode,'Commit bias tensor must have signed-i32 full-result or per-column shape')
            elif bias_stages:
                return _decline(cb,opcode,'Commit declares a bias stage without its public bias operand')
            if capture is not None and capture['seed'] is not None:
                seed=capture['seed']
                if epochs.get(seed,0)!=capture['epoch']:
                    return _decline(cb,opcode,'Matmul initializer storage was overwritten after accumulator capture; moving the seed to readout would change its tensor epoch')
                if operand_bias is not None:
                    return _decline(cb,opcode,'A retained matmul initializer and an additional commit bias need a separately materialized exact sum')
                if capture.get('full_binding') is None:
                    operands['bias']=seed
                    attrs['bias']=seed
                    attrs['epilogue']=['bias_add']+list(stages)
                # A source-bound full matrix stays on the original MATMUL's
                # accumulator_seed. It is not a vector bias epilogue.
            elif operand_bias is not None:
                attrs['bias']=operand_bias
        if 'operands' in command or operands:
            command['operands']=operands
        if 'attributes' in command or attrs:
            command['attributes']=attrs
        commands.append(command)
        # Reads precede destination writes within a command, including an
        # output that intentionally reuses the initializer's storage name.
        for key,name in operands.items():
            if key.startswith('dst') and name in tensors:
                epochs[name]=epochs.get(name,0)+1
    cb['commands']=commands
    if full_bindings:
        cb['_requires_full_seed_program']=True
        cb['_full_seed_source_bindings']=full_bindings
    return _preload_routes(cb)
