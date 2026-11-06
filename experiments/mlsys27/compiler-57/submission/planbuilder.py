"""Source-bound task ownership for typed whole-program lowering."""
import hashlib


def _runtime_scale_source_boundary(cb,ops,names,args):
    """Admit a whole-program boundary only for original runtime scale operands.

    A scalar CONFIG_ST operand is read by the emitted kernel. It cannot be
    represented by replacing that input with a numerical command-buffer
    attribute. Keep its source tensor and caller binding instead.
    """
    commands=[c for c in cb['commands'] if 'acc_scale_tensor' in c.get('attributes',{})]
    if not commands:
        return False
    tensors=cb['tensors']
    bound={}
    original=list(args)+[v for op in ops for v in op.results]
    for value in original:
        name=names.get(value)
        if name in tensors:
            if name in bound and bound[name] is not value:
                return False
            bound[name]=value
    if set(tensors)!=set(bound):
        return False
    for name,spec in tensors.items():
        typ=bound[name].type
        if not hasattr(typ,'get_shape') or not hasattr(typ,'element_type'):
            return False
        if list(typ.get_shape())!=spec['shape'] or str(typ.element_type)!=spec['dtype']:
            return False
    for command in commands:
        name=command['attributes']['acc_scale_tensor']
        value=bound.get(name)
        index=command.get('_source_op_index')
        if value not in args or not isinstance(index,int) or not 0<=index<len(ops):
            return False
        if str(value.type.element_type)!='f32' or list(value.type.get_shape()) not in ([],[1]):
            return False
        source_op=ops[index]
        if value not in source_op.operands:
            return False
        destination=command.get('operands',{}).get('dst')
        if not any(names.get(result)==destination for result in source_op.results):
            return False
    return True


def _full_seed_source_boundary(cb,ops,names,args):
    """Keep full accumulator seeds at their exact source/caller boundary."""
    if not cb.get('_requires_full_seed_program'):
        return False
    from frontend_composition import recognize_mac,recognize_extension,recognize_readout
    from xdsl.dialects.builtin import IntegerAttr,DenseIntOrFPElementsAttr
    tensors=cb['tensors']
    bound={}
    for value in list(args)+[v for op in ops for v in op.results]:
        name=names.get(value)
        if name in tensors:
            if name in bound and bound[name] is not value:
                return False
            bound[name]=value
    if set(tensors)!=set(bound):
        return False
    for name,spec in tensors.items():
        typ=bound[name].type
        if not hasattr(typ,'get_shape') or not hasattr(typ,'element_type'):
            return False
        if list(typ.get_shape())!=spec['shape'] or str(typ.element_type)!=spec['dtype']:
            return False

    def literal_zero(value):
        owner=value.owner
        if getattr(owner,'name',None) in ('linalg.fill','tensor.splat'):
            return literal_zero(owner.operands[0])
        if getattr(owner,'name',None)!='arith.constant':
            return False
        attr=owner.properties.get('value',owner.attributes.get('value'))
        if isinstance(attr,IntegerAttr):
            return attr.value.data==0
        return isinstance(attr,DenseIntOrFPElementsAttr) and all(x==0 for x in attr.get_values())

    def uniform_integer(value):
        """Read only an original uniform integer literal, never runtime data."""
        owner=value.owner
        if getattr(owner,'name',None) in ('linalg.fill','tensor.splat'):
            return uniform_integer(owner.operands[0])
        if getattr(owner,'name',None)!='arith.constant':
            return None
        attr=owner.properties.get('value',owner.attributes.get('value'))
        if isinstance(attr,IntegerAttr):
            return attr.value.data
        if isinstance(attr,DenseIntOrFPElementsAttr):
            values=list(attr.get_values())
            if values and type(values[0]) is int and all(x==values[0] for x in values):
                return values[0]
        return None

    def immutable_materialized_seed(seed_name,seed,mac_index,command_index):
        """Prove a single source-owned device fill and its retained epoch."""
        if tensors[seed_name].get('role')!='intermediate':
            return False
        owner=seed.owner
        if getattr(owner,'name',None) not in ('arith.constant','tensor.splat','linalg.fill') or owner not in ops:
            return False
        producer_index=ops.index(owner)
        if producer_index>=mac_index or seed not in owner.results:
            return False
        number=uniform_integer(seed)
        if number is None:
            return False
        writes=[(i,c) for i,c in enumerate(commands)
                if any(k.startswith('dst') and n==seed_name for k,n in c.get('operands',{}).items())]
        # A rewritten or delayed seed has a different storage epoch even when
        # the source SSA spelling is unchanged. Keep this initial route closed.
        if len(writes)!=1 or writes[0][0]>=command_index:
            return False
        _,writer=writes[0]
        attrs=writer.get('attributes',{})
        return (writer['opcode']=='FILL'
                and writer.get('_source_op_index')==producer_index
                and writer.get('operands',{}).get('dst')==seed_name
                and attrs.get('output_dtype')=='i32'
                and type(attrs.get('value_code')) is int
                and attrs['value_code']==number)

    commands=cb['commands']

    def batched_seed_boundary(command,command_index,seed_name):
        """Bind a direct batched MAC to its original full i32 initializer."""
        from frontend_composition import recognize_batched_mac
        operands=command.get('operands',{})
        attrs=command.get('attributes',{})
        index=command.get('_source_op_index')
        if type(index) is not int or not 0<=index<len(ops):
            return False
        source=ops[index]
        mac=recognize_batched_mac(source)
        if mac is None or mac.get('transpose_rhs') or len(source.results)!=1:
            return False
        lhs,rhs,result=mac['lhs'],mac['rhs'],source.results[0]
        seed=bound.get(seed_name)
        if seed is None or mac['seed'] is not seed:
            return False
        if (names.get(lhs)!=operands.get('a') or names.get(rhs)!=operands.get('w')
                or names.get(result)!=operands.get('dst')
                or bound.get(operands.get('dst')) is not result
                or operands.get('dst')==seed_name):
            return False
        ash=list(lhs.type.get_shape());bsh=list(rhs.type.get_shape())
        dsh=list(result.type.get_shape());ssh=list(seed.type.get_shape())
        if (len(ash)<3 or len(bsh)!=len(ash) or len(dsh)!=len(ash)
                or ash[:-2]!=bsh[:-2] or ash[:-2]!=dsh[:-2]
                or any(type(n) is not int or n<=0 for n in ash+bsh+dsh)
                or ash[-1]!=bsh[-2] or dsh[-2:]!=[ash[-2],bsh[-1]] or ssh!=dsh):
            return False
        if (str(lhs.type.element_type)!='i8' or str(rhs.type.element_type)!='i8'
                or str(result.type.element_type)!='i32' or str(seed.type.element_type)!='i32'
                or attrs.get('output_dtype','i32')!='i32' or attrs.get('epilogue',[])!=[]
                or set(attrs)-{'accumulator_seed','output_dtype','epilogue'}
                or set(operands)!={'a','w','dst'}):
            return False
        spec=tensors[seed_name]
        if seed in args:
            return (spec.get('role') in ('input','weight','bias','scale')
                    and not any(any(k.startswith('dst') and n==seed_name
                                    for k,n in c.get('operands',{}).items()) for c in commands))
        return immutable_materialized_seed(seed_name,seed,index,command_index)

    resident={}
    seeds=[]
    for command_index,command in enumerate(commands):
        operands=command.get('operands',{})
        if command['opcode']=='RES_PACK':
            resident[operands['dst']]=resident.get(operands['src'],operands['src'])
        seed_name=command.get('attributes',{}).get('accumulator_seed')
        if seed_name is None:
            continue
        spec=tensors.get(seed_name,{})
        if command['opcode']=='BATCHED_MATMUL':
            if not batched_seed_boundary(command,command_index,seed_name):
                return False
            seeds.append(seed_name)
            continue
        if len(spec.get('shape',[]))!=2:
            return False
        if command['opcode'] not in ('MATMUL','MATMUL_RESIDENT'):
            return False
        seed=bound.get(seed_name)
        if seed is None or spec.get('dtype')!='i32':
            return False
        index=command.get('_source_op_index')
        if not isinstance(index,int) or not 0<=index<len(ops):
            return False
        if seed in args:
            if spec.get('role') not in ('input','weight','bias','scale'):
                return False
            if any(c.get('operands',{}).get('dst')==seed_name for c in commands):
                return False
        elif not immutable_materialized_seed(seed_name,seed,index,command_index):
            return False
        source_mac=ops[index]
        mac=recognize_mac(source_mac)
        if mac is None:
            return False
        rhs=resident.get(operands.get('rhs'),operands.get('rhs'))
        if names.get(mac['lhs'])!=operands.get('lhs') or names.get(mac['rhs'])!=rhs or mac['transpose_rhs']:
            return False
        if list(source_mac.results[0].type.get_shape())!=spec['shape']:
            return False
        result=source_mac.results[0]
        # Direct full initializers and a separately written exact i32 add both
        # retain their original arithmetic. No vector-bias stage is invented.
        source_results=[]
        if mac['seed'] is seed:
            source_results.append(result)
        elif literal_zero(mac['seed']):
            for later in ops[index+1:]:
                extension=recognize_extension(later)
                if extension is not None and extension.get('accumulator') is result and extension.get('bias') is seed:
                    source_results.append(later.results[0])
        if not source_results:
            return False
        commits=[c for c in commands if c['opcode']=='COMMIT' and c.get('operands',{}).get('src')==operands.get('dst')]
        if not commits:
            return False
        for commit in commits:
            ci=commit.get('_source_op_index')
            if not isinstance(ci,int) or not 0<=ci<len(ops):
                return False
            owner=ops[ci]
            destination=commit.get('operands',{}).get('dst')
            if not any(names.get(v)==destination for v in owner.results):
                return False
            readout=recognize_readout(owner)
            if not any(v in owner.results for v in source_results) and not (readout is not None and readout['src'] in source_results):
                return False
        seeds.append(seed_name)
    return bool(seeds)


def prepare(cb, source):
    ops=cb.get('_source_ops')
    if not ops or cb.get('declined'):
        return None
    names=cb['_source_names']
    args=cb['_source_args']
    mixed=any(c['opcode']=='HOST_REGION' for c in cb['commands'])
    runtime_scale=_runtime_scale_source_boundary(cb,ops,names,args)
    full_seed=_full_seed_source_boundary(cb,ops,names,args)
    # Typed modular integer routes retain their original program boundary.
    # The accelerator command vocabulary alone has saturating byte exits and
    # cannot stand in for the source's wrapping integer result semantics.
    integer=any(c.get('attributes',{}).get('integer_wrap') for c in cb['commands'])
    if cb.get('_requires_full_seed_program') and not full_seed:
        cb['declined']={'op':'full_matrix_accumulator_seed',
                        'reason':'Full matrix preload lacks its complete original source and whole-program pointer proof'}
        cb['commands']=[]
        return None
    if not mixed and not runtime_scale and not full_seed and not integer:
        return None
    host_indices={c['_source_op_index'] for c in cb['commands'] if c['opcode']=='HOST_REGION'}
    # Structural operations without commands remain with their adjacent route.
    owners={c.get('_source_op_index') for c in cb['commands']}
    kinds=['host' if i in host_indices else 'device' if i in owners else None for i in range(len(ops))]
    for i in range(len(kinds)-1,-1,-1):
        if kinds[i] is None:
            kinds[i]=kinds[i+1] if i+1<len(kinds) else 'device'
    groups=[]
    by_index={}
    for i,kind in enumerate(kinds):
        if not groups or groups[-1]['kind']!=kind:
            groups.append({'task_index':len(groups),'kind':kind,'source_op_indices':[], 'source_region_ids':[], 'reads':[], 'writes':[]})
        group=groups[-1];group['source_op_indices'].append(i);by_index[i]=group['task_index']
        rid=ops[i].attributes.get('prov.region_id')
        if rid is not None and rid.data not in group['source_region_ids']:
            group['source_region_ids'].append(rid.data)
    tensors=cb['tensors']
    entries=[names[v] for v in args]
    ordered=entries+[n for n in tensors if n not in entries]
    cb['kernel_abi']={'kind':'whole_program','args':[{'tensor':n,'access':'read' if n in entries else 'write'} for n in ordered], 'outputs':list(cb['outputs'])}
    for c in cb['commands']:
        i=c.get('_source_op_index',0)
        c['_task_index']=by_index[i]
    values=[]
    for i,op in enumerate(ops):
        for j,v in enumerate(op.results):
            if v in names and names[v] in tensors:
                values.append({'op_index':i,'result_index':j,'tensor':names[v]})
    for group in groups:
        produced=set()
        for i in group['source_op_indices']:
            for operand in ops[i].operands:
                n=names.get(operand)
                if n in tensors and n not in produced and n not in group['reads']:
                    group['reads'].append(n)
            for result in ops[i].results:
                n=names.get(result)
                if n in tensors:
                    produced.add(n)
                    if n not in group['writes']:group['writes'].append(n)
    plan={'schema':'mixed_program_plan_v1','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'source_op_count':len(ops), 'tasks':groups,'entry_bindings':entries,'source_values':values,'output_bindings':list(cb['outputs'])}
    cb.setdefault('params',{})['global_program_plan']=plan
    return plan


def finish(plan, e, ranges):
    if plan is None:return
    # Ranges index the emitted LLVM instruction schedule, including scalar host
    # instructions; labels and module data are not instructions.
    n=e.schedule_count+len(e.prologue)
    for task in plan['tasks']:
        tid=task['task_index']
        if tid not in ranges:
            raise ValueError('source task has no emitted instruction schedule')
        start,end=ranges[tid]
        task['instruction_start'],task['instruction_end']=start+len(e.prologue),end+len(e.prologue)
    first=plan['tasks'][0]['instruction_start']
    last=plan['tasks'][-1]['instruction_end']
    plan['schedule_instruction_count']=n
    plan['prologue_instruction_range']=[0,first]
    plan['epilogue_instruction_range']=[last,n]
