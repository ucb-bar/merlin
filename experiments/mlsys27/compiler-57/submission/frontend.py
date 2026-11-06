"""Typed source frontend and source-derived command construction.

The input readers in vendor_input are public grammar plumbing; lowering choices here
are authored for this backend. All unknown numerical forms are explicit declines.
"""
from __future__ import annotations
from fractions import Fraction
from math import prod
from vendor_input.interface_reader import parse_interface_mlir, _op_line, _parse_attr_block, _module_attrs, defined_mnemonics
from vendor_input.linalg_reader import make_linalg_context
from frontend_composition import recognize_mac, recognize_readout, recognize_operand_sum, recognize_extension, recognize_pool, recognize_batched_mac, register_batched_matmul
from frontend_broadcast import recognize_broadcast, register_broadcast
from quantized_float import recognize as recognize_quantized_float
from integer_arithmetic import recognize as recognize_integer_arithmetic
from promoted_integer import recognize as recognize_promoted_integer
from frontend_movement import recognize_movement
from xdsl.parser import Parser
from xdsl.dialects.builtin import TensorType, DenseIntOrFPElementsAttr, IntegerAttr, FloatAttr


class Unsupported(ValueError):
    pass


def _attr(op, name, default=None):
    return op.properties.get(name, op.attributes.get(name, default))


def _int(v):
    if isinstance(v, IntegerAttr):
        return v.value.data
    return int(v)


def _const(value):
    owner = value.owner
    if getattr(owner,'name',None) in ('tensor.splat', 'linalg.fill'):
        return _const(owner.operands[0])
    if getattr(owner,'name',None) == 'arith.constant':
        a = _attr(owner, 'value')
        if isinstance(a, IntegerAttr):
            return a.value.data
        if isinstance(a, FloatAttr):
            return a.value.data
        if isinstance(a, DenseIntOrFPElementsAttr):
            vals = list(a.get_values())
            if vals and all(v == vals[0] for v in vals):
                return vals[0]
    raise Unsupported('initializer is not a uniform literal')


def _spec(v):
    if not isinstance(v.type, TensorType):
        raise Unsupported('expected a ranked tensor')
    return {'shape': list(v.type.get_shape()), 'dtype': str(v.type.element_type)}


def _dense_uniform_floating_storage(value):
    """Compare original dense storage words, including zero signs/NaN bits."""
    dtype=_spec(value)['dtype']
    width={'f16':2,'bf16':2,'f32':4}.get(dtype)
    owner=value.owner
    if width is None or getattr(owner,'name',None)!='arith.constant':
        return None
    attr=_attr(owner,'value')
    if not isinstance(attr,DenseIntOrFPElementsAttr):
        return None
    raw=bytes(attr.data.data)
    if not raw or len(raw)!=prod(_spec(value)['shape'])*width:
        raise Unsupported('floating literal storage does not match its nonempty typed shape')
    first=raw[:width]
    return all(raw[i:i+width]==first for i in range(0,len(raw),width))


def _array(v):
    if v is None:
        return []
    if hasattr(v, 'get_values'):
        return list(v.get_values())
    return [_int(x) for x in v]


def _static_list(op, key):
    vals = _array(_attr(op, key))
    if any(x < 0 for x in vals):
        raise Unsupported('dynamic slice bounds are unsupported')
    return vals


def _affine_expr(v, args, env):
    """An exact integer linear form [weights..., constant] of scalar arguments."""
    if v in env:
        return env[v]
    n = len(args)
    if v in args:
        ans = [0] * (n + 1)
        ans[args.index(v)] = 1
        return ans
    op = v.owner
    if op.name == 'arith.constant':
        return [0]*n + [Fraction(_const(v))]
    if op.name in ('arith.extsi', 'arith.index_cast', 'arith.sitofp', 'arith.extf'):
        return _affine_expr(op.operands[0], args, env)
    if op.name in ('arith.addi', 'arith.addf', 'arith.subi', 'arith.subf'):
        a = _affine_expr(op.operands[0], args, env)
        b = _affine_expr(op.operands[1], args, env)
        sign = -1 if op.name in ('arith.subi', 'arith.subf') else 1
        return [x+sign*y for x,y in zip(a,b)]
    if op.name in ('arith.muli', 'arith.mulf'):
        a = _affine_expr(op.operands[0], args, env)
        b = _affine_expr(op.operands[1], args, env)
        if any(a[:-1]) and any(b[:-1]):
            raise Unsupported('runtime scalar product is not an affine map')
        return [x*b[-1] for x in a] if not any(b[:-1]) else [x*a[-1] for x in b]
    if op.name == 'arith.negf':
        return [-x for x in _affine_expr(op.operands[0], args, env)]
    raise Unsupported('unsupported scalar affine operation '+op.name)


def _floating_identity_source(op):
    """Recognize direct storage passthrough with exact indexing and bit identity.

    Arithmetic identities such as x*1 and x/1 can quiet signaling NaNs, so
    they do not acquire a byte-copy proof. Casts and cancellation chains also
    retain their separate numerical obligations.
    """
    if op.name not in ('linalg.generic','linalg.map') or len(op.results)!=1:
        return None
    result=op.results[0]
    if not isinstance(result.type,TensorType):
        return None
    dtype=str(result.type.element_type)
    if dtype not in ('f16','bf16','f32') or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:
        return None
    inputs=list(op.inputs)
    block=op.regions[0].block
    body=list(block.ops)
    if not body or body[-1].name!='linalg.yield' or len(body[-1].operands)!=1:
        return None
    rank=len(result.type.get_shape())
    maps=list(_attr(op,'indexing_maps',[]))
    if op.name=='linalg.generic':
        from xdsl.ir.affine import AffineDimExpr
        identity=tuple(AffineDimExpr(i) for i in range(rank))
        if len(maps)!=len(inputs)+1 or any('parallel' not in str(i) for i in _attr(op,'iterator_types',[])):
            return None
        if maps[-1].data.num_dims!=rank or tuple(maps[-1].data.results)!=identity:
            return None

    value=body[-1].operands[0]
    if value not in block.args:
        return None
    index=list(block.args).index(value)
    if index>=len(inputs) or str(value.type)!=dtype or _spec(inputs[index])!=_spec(result):
        return None
    if op.name=='linalg.generic' and (maps[index].data.num_dims!=rank or tuple(maps[index].data.results)!=identity):
        return None
    if any(scalar is not body[-1] and scalar.name!='arith.constant' for scalar in body):
        return None
    return inputs[index]


def _host_reason(op):
    """Select only explicitly retained host boundaries or nonlinear islands."""
    name=op.name
    if name=='builtin.unregistered':
        original=_attr(op,'op_name__')
        name=original.data if original is not None else name
    if name.startswith('quant_ext.'):
        return 'Selected quantization boundary retains scalar calibration and storage conversion'
    role=_attr(op,'prov.role');numerics=_attr(op,'prov.quant.numerics')
    if getattr(role,'data',None)=='readout' and getattr(numerics,'data',None)=='integer_readout':
        return 'Selected exact integer readout retains wide rational arithmetic and IEEE storage conversion on the host'
    body=[x for r in op.regions for x in r.walk()]
    nonlinear={'math.exp','math.exp2','math.log','math.log2','math.log10','math.sqrt','math.rsqrt','math.powf',
               'math.sin','math.cos','math.tanh','math.erf','math.expm1','math.log1p'}
    def original_mnemonic(scalar):
        if scalar.name=='builtin.unregistered':
            original=_attr(scalar,'op_name__')
            return original.data if original is not None else scalar.name
        return scalar.name
    if name in nonlinear or any(original_mnemonic(x) in nonlinear for x in body):
        # The caller separately checks host_lowering.supports(op) before emitting
        # a host task, including every original scalar operation in its body.
        return 'Nonlinear IEEE scalar operation is outside the declared byte mesh arithmetic'
    if op.name in ('linalg.generic','linalg.map'):
        inputs=list(getattr(op,'inputs',[]) or [])
        if len(op.regions)==1 and len(op.regions[0].blocks)==1:
            args=list(op.regions[0].block.args)
            runtime_args=set()
            for arg,tensor_input in zip(args,inputs):
                try: _const(tensor_input)
                except Unsupported: runtime_args.add(arg)
            for scalar in body:
                if scalar.name=='arith.mulf' and all(v in runtime_args for v in scalar.operands):
                    return 'Parallel runtime product is excluded from the selected native byte construction domain'
            def has_index_payload(value,seen):
                if value in seen: return False
                seen.add(value);owner=value.owner
                if getattr(owner,'name',None)=='linalg.index': return True
                if getattr(owner,'name',None)=='tensor.extract': return False
                return any(has_index_payload(v,seen) for v in getattr(owner,'operands',[]))
            if any(x.name in ('arith.sitofp','arith.uitofp') and has_index_payload(x.operands[0],set()) for x in body):
                return 'Index-generated numeric payload arithmetic is excluded from the selected native byte construction domain'
    family=_attr(op,'prov.family')
    if getattr(family,'data',None)=='normalization' and any(str(v.type.element_type).startswith('f') for v in op.results if isinstance(v.type,TensorType)):
        return 'Selected normalization island retains IEEE arithmetic on the host'
    return None


def attach_source_metadata(cb):
    """Bind command attribution to exact original source operation fields.

    This may run after command normalization. Only the retained operation index
    selects provenance; command kinds, output shapes and program names supply no
    region or composition identities.
    """
    operations=cb.get('_source_ops')
    records=cb.get('_interface_source_records')
    if operations is None and records is None:
        return cb
    count=len(operations) if operations is not None else len(records)
    cb['_source_op_count']=count
    correspondence=[]
    for command_index,command in enumerate(cb.get('commands',[])):
        index=command.get('_source_op_index')
        if type(index) is not int or not 0<=index<count:
            raise Unsupported('emitted command has no valid original source operation index')
        if operations is not None:
            source=operations[index]
            region=_attr(source,'prov.region_id')
            stage=_attr(source,'prov.composition_stage')
            region=getattr(region,'data',None)
            stage=getattr(stage,'data',None)
        else:
            region=records[index].get('prov.region_id')
            stage=records[index].get('prov.composition_stage')
        attrs=command.setdefault('attributes',{})
        for key in ('merlin.source_op_index','prov.region_id','prov.composition_stage'):
            attrs.pop(key,None)
        command.pop('_source_region_id',None)
        command.pop('_composition_stage',None)
        binding={'command_index':command_index,'source_op_index':index}
        fused=command.get('_fused_source_op_indices')
        if fused is not None:
            if any(type(i) is not int or not 0<=i<count for i in fused):
                raise Unsupported('fused command references an absent original source operation')
            binding['source_op_indices']=list(dict.fromkeys(fused))
        if region is not None:
            if not isinstance(region,str):
                raise Unsupported('source prov.region_id must be a string')
            command['_source_region_id']=region
            binding['region_id']=region
        if stage is not None:
            if not isinstance(stage,str):
                raise Unsupported('source prov.composition_stage must be a string')
            command['_composition_stage']=stage
            binding['composition_stage']=stage
        correspondence.append(binding)
    cb.setdefault('params',{})['command_sources']=correspondence
    return cb


def parse_source(text):
    if 'merlin_iface.' in text:
        modattrs=_parse_attr_block(_module_attrs(text))
        if modattrs.get('merlin_iface.version')!='0.1' or modattrs.get('merlin_iface.abi_version','0.1')!='0.1':
            raise Unsupported('unsupported interface or ABI version')
        cb = parse_interface_mlir(text, include_results=True)
        aliases={}
        source_records=[]
        command_indices=[]
        for raw in text.splitlines():
            op=_op_line(raw.strip())
            if op and op.get('mnemonic') in defined_mnemonics():
                attrs=_parse_attr_block(op['attrs_body'])
                index=len(source_records)
                source_records.append({k:v for k,v in attrs.items() if k.startswith('prov.')})
                if op['mnemonic']!='tensor': command_indices.append(index)
                if op.get('result'):
                    aliases[op['result']]=attrs.get('name',op['result'])
        if len(command_indices)!=len(cb['commands']):
            raise Unsupported('raw source operation inventory does not match emitted commands')
        cb['_interface_source_records']=source_records
        for c,index in zip(cb['commands'],command_indices):
            c['_source_op_index']=index
            c['operands']={k:aliases.get(v,v) for k,v in c.get('operands',{}).items()}
            # Source provenance belongs to the source ledger. Operation-specific
            # runtime attributes describe only mathematical/transfer semantics.
            if 'attributes' in c:
                c['attributes']={k:v for k,v in c['attributes'].items() if not k.startswith('prov.')}

        # The raw interface may chain named results; only final declared outputs
        # remain external pointers when used as internal intermediate values.
        dests = {c.get('operands', {}).get('dst') for c in cb['commands']}
        read = {v for c in cb['commands'] for k,v in c.get('operands',{}).items() if k != 'dst'}
        cb['outputs'] = [n for n,s in cb.get('tensors',{}).items() if s.get('role')=='output' and n not in read]
        for n in dests & read:
            if n in cb.get('tensors',{}) and cb['tensors'][n].get('role')=='output':
                cb['tensors'][n]['role']='intermediate'
        cb.setdefault('params',{}).setdefault('lane_placement',[{'region':'program','lane':'device'}])
        # Carry a mathematical source bound into overlap reductions. The bound
        # derives from byte operands and actual contraction K, never sample data.
        handles={};bounds={}
        for c in cb['commands']:
            o=c.get('operands',{});a=c.get('attributes',{})
            if c['opcode']=='RES_PACK': handles[o['dst']]=o['src']
            elif c['opcode'] in ('MATMUL','MATMUL_RESIDENT'):
                lhs=cb['tensors'].get(o['lhs'],{})
                rhs=cb['tensors'].get(handles.get(o['rhs'],o['rhs']),{})
                if lhs.get('dtype')=='i8' and rhs.get('dtype')=='i8': bounds[o['dst']]=16384*lhs['shape'][-1]
            elif c['opcode']=='COMMIT' and o.get('src') in bounds and not any(k in a for k in ('bias','accumulator_seed')) and not o.get('bias'):
                bounds[o['dst']]=bounds[o['src']] if a.get('output_dtype','i32')=='i32' else 128
            elif c['opcode']=='WINDOW_SUM' and o.get('src') in bounds:
                a['input_bound']=bounds[o['src']];c['attributes']=a
        return attach_source_metadata(cb)
    from normalize_prefix import normalize_prefix
    return attach_source_metadata(normalize_prefix(_parse_linalg(text)))


def _parse_linalg(text):
    context=make_linalg_context()
    register_batched_matmul(context)
    register_broadcast(context)
    module = Parser(context, text).parse_module()
    fns = [o for o in module.walk() if o.name=='func.func' and o.body.blocks]
    fn = next((f for f in fns if f.sym_name.data=='forward'), fns[0] if len(fns)==1 else None)
    if fn is None or len(fn.body.blocks)!=1:
        raise Unsupported('requires one source function with one entry block')
    block = fn.body.block
    ops = list(block.ops)
    if not ops or ops[-1].name!='func.return':
        raise Unsupported('source function has no complete return')
    cb = {'abi_version':'0.1','target':'gemmini','tensors':{},'commands':[], 'outputs':[], 'params':{'lane_placement':[]}}
    names = {}
    values = {}
    value_bounds = {}
    output_names = {}
    # The framework-level carrier publishes canonical Yn result identifiers.
    # Anonymous composition carriers publish their actual returned SSA names.
    # Both rules derive from the parsed source boundary, never its filename.
    canonical_results=_attr(module,'prov.level') is not None
    reserved={v.name_hint or ('A'+str(i)) for i,v in enumerate(block.args)}
    from semantic_outputs import gradient_roles
    semantic_names=gradient_roles(ops[:-1],list(block.args),list(ops[-1].operands)) or {}
    if any(name in reserved for name in semantic_names.values()):
        semantic_names={}
    for i,v in enumerate(ops[-1].operands):
        base=semantic_names.get(v) or ('Y'+str(i) if canonical_results else (v.name_hint or 'Y'+str(i)))
        name=base
        suffix=1
        while name in reserved:
            name=base+'_'+str(suffix);suffix+=1
        reserved.add(name)
        # Repeated source returns retain result order and distinct buffers.
        output_names.setdefault(v, []).append(name)
        cb['outputs'].append(name)
    for i,v in enumerate(block.args):
        n = v.name_hint or ('A'+str(i))
        names[v] = n
        cb['tensors'][n] = {**_spec(v),'role':'input'}
        if _spec(v)['dtype']=='i8': value_bounds[v]=128
    for i,op in enumerate(ops[:-1]):
        for j,v in enumerate(op.results):
            if isinstance(v.type,TensorType):
                n = output_names.get(v,['V'+str(i)+'_'+str(j)])[0]
                names[v]=n
                cb['tensors'][n]={**_spec(v),'role':'output' if v in output_names else 'intermediate'}
    materialized = set(block.args)
    current = [0]
    from frontend_integer import contraction_epilogues, gathered_contractions, column_biases
    integer_epilogues=contraction_epilogues(ops[:-1],output_names)
    integer_epilogue_ops={op for plan in integer_epilogues.values() for op in plan['absorbed']}
    absorbed_integer_ops=set()
    integer_gathers=gathered_contractions(ops[:-1],output_names)
    integer_biases=column_biases(ops[:-1],output_names)
    absorbed_integer_ops.update(op for plan in integer_gathers.values() for op in plan['absorbed'])
    absorbed_integer_ops.update(plan['op'] for plan in integer_biases.values())
    # A scalar reduction may be packed into a one-element public tensor.
    # Bind this shape-only boundary to the reduction producer itself.
    reduction_packs={}
    for candidate in ops[:-1]:
        if candidate.name!='tensor.from_elements' or len(candidate.operands)!=1 or len(candidate.results)!=1:
            continue
        scalar=candidate.operands[0];extract=scalar.owner
        if (getattr(extract,'name',None)!='tensor.extract' or len(extract.operands)!=1
                or getattr(extract.operands[0].owner,'name',None)!='linalg.reduce'
                or _spec(extract.operands[0])['shape']!=[] or prod(_spec(candidate.results[0])['shape'])!=1
                or len(list(extract.operands[0].uses))!=1 or len(list(scalar.uses))!=1):
            continue
        reduction_packs[extract.operands[0].owner]=candidate.results[0]
        absorbed_integer_ops.add(candidate)

    def emit(opcode, operands, attributes=None):
        attrs = dict(attributes or {})
        c = {'opcode':opcode,'operands':operands,'_source_op_index':current[0]}
        if attrs:
            c['attributes']=attrs
        cb['commands'].append(c)

    def contraction_literal(value,seen):
        if value in seen or value in output_names:
            return False
        seen=seen|{value}
        uses=list(value.uses)
        if not uses:return False
        for use in uses:
            consumer=use.operation
            if consumer.name in ('linalg.matmul','linalg.batch_matmul','linalg.matmul_transpose_b'):
                continue
            if consumer.name=='linalg.generic':
                mac=recognize_mac(consumer)
                extension=recognize_extension(consumer)
                if mac is not None and value in (mac['lhs'],mac['rhs'],mac['seed']):continue
                if extension is not None and value is extension.get('bias'):continue
            if consumer.name in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast','linalg.transpose'):
                if all(contraction_literal(v,seen) for v in consumer.results):continue
            return False
        return True

    def literal_tensor(value):
        owner=value.owner
        attr=_attr(owner,'value')
        if owner.name!='arith.constant' or not isinstance(attr,DenseIntOrFPElementsAttr):
            raise Unsupported('nonuniform initializer must be an original dense source literal')
        spec=_spec(value);dtype=spec['dtype'];shape=spec['shape']
        if not contraction_literal(value,set()):
            source_index=ops.index(owner)
            reason='Nonuniform original literal result construction is excluded from the selected native byte construction domain'
            cb['commands'].append({'opcode':'HOST_REGION','operands':{'dst':names[value]},'attributes':{'reason':reason},'_source_op_index':source_index,'_source_op':owner,'_value_names':names})
            rid=_attr(owner,'prov.region_id')
            cb['params']['lane_placement'].append({'lane':'host','region':getattr(rid,'data','source_'+str(source_index)),'reason':reason})
            materialized.add(value)
            return names[value]
        if dtype not in ('i8','i32'):
            raise Unsupported('literal contraction storage requires signed byte or signed-i32 values')
        data=list(attr.get_values())
        if len(data)!=prod(shape) or not data:
            raise Unsupported('dense literal shape and storage count disagree')
        bits=8 if dtype=='i8' else 32
        if any(type(x) is not int or not -(1<<(bits-1))<=x<(1<<(bits-1)) for x in data):
            raise Unsupported('dense source literal is outside its signed storage domain')
        source_index=ops.index(owner)
        result_index=list(owner.results).index(value)
        old_index=current[0];current[0]=source_index
        stem=names[value]+'_literal';serial=0
        def temporary(label,dims):
            nonlocal serial
            name=stem+'_'+label
            while name in cb['tensors']:
                serial+=1;name=stem+'_'+label+'_'+str(serial)
            cb['tensors'][name]={'shape':list(dims),'dtype':dtype,'role':'intermediate'}
            cb.setdefault('_temporary_sources',{})[name]={'source_op_index':source_index,'source_result_index':result_index,'purpose':'original_source_literal_materialization'}
            return name
        width=shape[-1] if shape else 1
        rows=[];row_cache={}
        try:
            for row_index in range(len(data)//width):
                row=tuple(data[row_index*width:(row_index+1)*width])
                if row in row_cache:
                    rows.append(row_cache[row]);continue
                frequencies={}
                for x in row:frequencies[x]=frequencies.get(x,0)+1
                base=max(frequencies,key=frequencies.get)
                current_row=temporary('row_'+str(row_index)+'_base',[1,width])
                emit('FILL',{'dst':current_row},{'value_code':base,'output_dtype':dtype})
                begin=0
                while begin<width:
                    end=begin+1
                    while end<width and row[end]==row[begin]:end+=1
                    if row[begin]!=base:
                        run=temporary('run_'+str(row_index)+'_'+str(begin),[1,end-begin])
                        emit('FILL',{'dst':run},{'value_code':row[begin],'output_dtype':dtype})
                        updated=temporary('row_'+str(row_index)+'_through_'+str(end),[1,width])
                        emit('INSERT_SLICE',{'dest':current_row,'src':run,'dst':updated},{'offsets':[0,begin],'strides':[1,1],'output_dtype':dtype})
                        current_row=updated
                    begin=end
                row_cache[row]=current_row;rows.append(current_row)
            level=0
            while len(rows)>1:
                next_rows=[]
                for begin in range(0,len(rows),8):
                    group=rows[begin:begin+8]
                    if len(group)==1:next_rows.append(group[0]);continue
                    result=temporary('cat_'+str(level)+'_'+str(begin),[sum(cb['tensors'][n]['shape'][0] for n in group),width])
                    emit('CONCAT',{**{'src'+str(j):n for j,n in enumerate(group)},'dst':result},{'axis':0,'output_dtype':dtype})
                    next_rows.append(result)
                rows=next_rows;level+=1
            emit('MOVEMENT',{'src':rows[0],'dst':names[value]},{'form':'reshape','output_dtype':dtype})
        finally:
            current[0]=old_index
        materialized.add(value);value_bounds[value]=max(abs(x) for x in data)
        return names[value]

    def tensor(v):
        if v not in names:
            raise Unsupported('unbound tensor SSA value')
        if v in materialized:
            return names[v]
        owner=v.owner
        if owner.name in ('arith.constant','tensor.splat','linalg.fill'):
            if _spec(v)['dtype'] in ('f16','bf16','f32'):
                uniform_storage=_dense_uniform_floating_storage(v)
                if uniform_storage is False:
                    return literal_tensor(v)
                raise Unsupported('Selected FILL ABI accepts only integer value_code and integer output_dtype; no exact floating-storage fill representation is declared')
            try:
                value = _const(v)
            except Unsupported:
                return literal_tensor(v)
            if _spec(v)['dtype'].startswith('f') or _spec(v)['dtype']=='bf16':
                raise Unsupported('floating literal fill has no byte storage lowering')
            if int(value)!=value:
                raise Unsupported('nonintegral literal fill')
            emit('FILL',{'dst':names[v]},{'value_code':int(value),'output_dtype':_spec(v)['dtype']})
            cb['commands'][-1]['_source_op_index']=ops.index(owner)
            materialized.add(v)
            value_bounds[v]=abs(int(value))
            return names[v]
        raise Unsupported('tensor read before complete source production')

    def copy(a,b,attrs=None):
        emit('MOVEMENT',{'src':tensor(a),'dst':names[b]},{'output_dtype':_spec(b)['dtype'],**(attrs or {})})
        materialized.add(b)
        if a in value_bounds: value_bounds[b]=value_bounds[a]

    def contraction(op,ins,outs,transpose_rhs=False):
        a,b=ins[:2]
        if _spec(a)['dtype']!='i8' or _spec(b)['dtype']!='i8':
            raise Unsupported('contraction operands need byte representation lowering')
        if _spec(op.results[0])['dtype']!='i32':
            raise Unsupported('contraction result needs exact i32 accumulation')
        if len(_spec(a)['shape'])>2:
            batch_proof=recognize_batched_mac(op)
            if batch_proof is None or batch_proof['lhs'] is not a or batch_proof['rhs'] is not b or not outs or batch_proof['seed'] is not outs[0]:
                raise Unsupported('batched contraction requires canonical signed batch indexing and scalar MAC semantics')
        attrs={}
        if outs:
            try:
                init=_const(outs[0])
            except Unsupported:
                init=None
            if init!=0:
                attrs['accumulator_seed']=tensor(outs[0])
        dest=names[op.results[0]]
        if len(_spec(a)['shape'])>2:
            # Nonzero initializers retain their original tensor SSA; the
            # source-bound batched-seed normalizer gates the caller contract.
            emit('BATCHED_MATMUL',{'a':tensor(a),'w':tensor(b),'dst':dest},{'output_dtype':'i32',**attrs})
        else:
            if op.name=='linalg.matmul_transpose_b' or transpose_rhs:
                old=tensor(b);shape=_spec(b)['shape'];tmp=dest+'_transpose'
                cb['tensors'][tmp]={'shape':list(reversed(shape)),'dtype':'i8','role':'intermediate'}
                emit('MOVEMENT',{'src':old,'dst':tmp},{'form':'permute','perm':[1,0],'output_dtype':'i8'})
                rhs=tmp
            else:
                rhs=tensor(b)
            acc=dest+'_acc'
            emit('MATMUL',{'lhs':tensor(a),'rhs':rhs,'dst':acc},attrs)
            emit('COMMIT',{'src':acc,'dst':dest},{'output_dtype':'i32','epilogue':[]})
        materialized.add(op.results[0])
        if not attrs.get('accumulator_seed'): value_bounds[op.results[0]]=16384*_spec(a)['shape'][-1]

    try:
        for i,op in enumerate(ops[:-1]):
            current[0]=i
            if op in absorbed_integer_ops:
                materialized.update(op.results)
                continue
            if not op.results or all(not isinstance(v.type,TensorType) for v in op.results):
                continue
            if op.name=='tensor.empty':
                continue
            if op.name in ('arith.constant','tensor.splat','linalg.fill'):
                # Initializers are materialized only when semantically consumed;
                # observable constant results are always produced.
                if any(v in output_names for v in op.results):
                    tensor(op.results[0])
                continue
            host_reason=_host_reason(op)
            if host_reason:
                from host_lowering import supports
                if not supports(op):
                    raise Unsupported(host_reason+'; scalar lowering is incomplete')
                refs=[]
                for value in list(op.operands)+[v for r in op.regions for inner in r.walk() for v in inner.operands]:
                    if value in names and value not in refs: refs.append(value)
                operands={'src'+str(j):names[v] for j,v in enumerate(refs)}
                operands.update({'dst'+str(j):names[v] for j,v in enumerate(op.results) if v in names})
                c={'opcode':'HOST_REGION','operands':operands,'attributes':{'reason':host_reason},'_source_op_index':i,'_source_op':op,'_value_names':names}
                cb['commands'].append(c)
                rid=_attr(op,'prov.region_id')
                cb['params']['lane_placement'].append({'lane':'host','region':getattr(rid,'data','source_'+str(i)),'reason':host_reason})
                materialized.update(v for v in op.results if v in names)
                continue
            ins=list(getattr(op,'inputs',[]) or [])
            outs=list(getattr(op,'outputs',[]) or [])
            from frontend_integer import recognize_integer_mac
            integer_mac=recognize_integer_mac(op)
            if integer_mac is not None:
                bias=integer_biases.get(op)
                result=bias['result'] if bias else op.results[0]
                gather=integer_gathers.get(op)
                operands={'a':tensor(gather['src'] if gather else integer_mac['lhs']),
                          'w':tensor(gather.get('rhs',integer_mac['rhs']) if gather else integer_mac['rhs']),
                          'dst':names[result]}
                try: zero_seed=_const(integer_mac['seed'])==0
                except Unsupported: zero_seed=False
                if not zero_seed:operands['seed']=tensor(integer_mac['seed'])
                attrs={'integer_wrap':True,'output_dtype':_spec(result)['dtype']}
                if gather:
                    attrs['lhs_gather']=gather['attributes']
                    attrs['matrix_shape']=_spec(integer_mac['lhs'])['shape']+[_spec(integer_mac['rhs'])['shape'][-1]]
                if bias:operands['column_bias']=tensor(bias['bias'])
                emit('BATCHED_MATMUL',operands,attrs)
                materialized.add(op.results[0])
                continue
            if op in integer_epilogue_ops:
                args=list(op.regions[0].block.args)
                body=list(op.regions[0].block.ops)
                env={}
                for arg,value in zip(args,ins):
                    try:env[arg]=[0]*len(args)+[int(_const(value))]
                    except Unsupported:pass
                form=_affine_expr(body[-1].operands[0],args,env)
                sources=[(value,int(weight)) for value,weight in zip(ins,form) if weight]
                if not sources:
                    emit('FILL',{'dst':names[op.results[0]]},{'value_code':int(form[-1]),'output_dtype':_spec(op.results[0])['dtype']})
                else:
                    if len(sources)==1:sources.append((sources[0][0],0))
                    operands={('lhs' if j==0 else 'rhs'):tensor(value) for j,(value,weight) in enumerate(sources)}
                    operands['dst']=names[op.results[0]]
                    emit('RESIDUAL_ADD',operands,{'lhs_scale':sources[0][1],'rhs_scale':sources[1][1],
                                                'integer_wrap':True,'output_dtype':_spec(op.results[0])['dtype']})
                materialized.add(op.results[0])
                continue
            broadcast=recognize_broadcast(op)
            if broadcast is not None:
                copy(broadcast['src'],op.results[0],broadcast['attributes'])
                continue
            movement=recognize_movement(op)
            if movement is not None:
                copy(movement['src'],op.results[0],movement['attributes'])
                continue
            if op.name in ('linalg.matmul','linalg.batch_matmul','linalg.matmul_transpose_b'):
                if op.name=='linalg.matmul' and len(ins)==2 and all(_spec(v)['dtype']=='i8' for v in ins) and _spec(op.results[0])['dtype']=='i32':
                    if recognize_mac(op) is None:
                        raise Unsupported('named byte matmul requires canonical signed indexing and cast semantics')
                contraction(op,ins,outs)
            elif op.name in ('tensor.collapse_shape','tensor.expand_shape','tensor.cast'):
                copy(op.operands[0],op.results[0],{'form':'reshape'})
            elif op.name=='tensor.extract_slice':
                offsets=_static_list(op,'static_offsets');sizes=_static_list(op,'static_sizes');strides=_static_list(op,'static_strides')
                copy(op.operands[0],op.results[0],{'form':'slice','offsets':offsets,'sizes':sizes,'strides':strides})
            elif op.name=='tensor.pad':
                low=_static_list(op,'static_low');high=_static_list(op,'static_high')
                body=list(op.regions[0].block.ops)
                if not body or body[-1].name!='tensor.yield' or len(body[-1].operands)!=1:
                    raise Unsupported('padding needs one literal yield')
                value=_const(body[-1].operands[0])
                copy(op.operands[0],op.results[0],{'form':'pad','low':low,'high':high,'mode':'constant','value_code':int(value)})
            elif op.name=='tensor.insert_slice':
                emit('INSERT_SLICE',{'src':tensor(op.operands[0]),'dest':tensor(op.operands[1]),'dst':names[op.results[0]]},{'offsets':_static_list(op,'static_offsets'),'strides':_static_list(op,'static_strides'),'output_dtype':_spec(op.results[0])['dtype']})
                materialized.add(op.results[0])
            elif op.name=='tensor.concat':
                emit('CONCAT',{**{'src'+str(j):tensor(v) for j,v in enumerate(op.operands)},'dst':names[op.results[0]]},{'axis':_int(_attr(op,'dim')),'output_dtype':_spec(op.results[0])['dtype']})
                materialized.add(op.results[0])
            elif op.name=='linalg.transpose':
                copy(ins[0],op.results[0],{'form':'permute','perm':_array(_attr(op,'permutation'))})
            elif op.name=='linalg.copy':
                copy(ins[0],op.results[0])
            elif op.name=='linalg.reduce':
                ins=list(op.operands[:1]);outs=list(op.operands[1:])
                region=op.regions[0].block
                body=list(region.ops)
                dtype=_spec(op.results[0])['dtype']
                if (len(ins)!=1 or len(outs)!=1 or len(region.args)!=2 or dtype not in ('i8','i32')
                        or _spec(ins[0])['dtype']!=dtype or len(body)!=2
                        or body[0].name!='arith.addi' or set(body[0].operands)!=set(region.args)
                        or body[-1].name!='linalg.yield' or tuple(body[-1].operands)!=tuple(body[0].results)):
                    raise Unsupported('named reduction requires an exact typed integer sum')
                dims=_array(_attr(op,'dimensions'))
                result=reduction_packs.get(op,op.results[0])
                emit('WINDOW_SUM',{'src':tensor(ins[0]),'dst':names[result]},
                     {'mode':'reduce','dims':dims,'initial':int(_const(outs[0])),
                      'integer_wrap':True,'output_dtype':dtype})
                materialized.add(op.results[0])
            elif op.name in ('linalg.generic','linalg.map'):
                identity_source=_floating_identity_source(op)
                if identity_source is not None:
                    copy(identity_source,op.results[0])
                    continue
                pool=recognize_pool(op)
                if pool is not None:
                    attrs={k:v for k,v in pool.items() if k!='src'}
                    attrs.update({'mode':'max','output_dtype':_spec(op.results[0])['dtype']})
                    emit('WINDOW_SUM',{'src':tensor(pool['src']),'dst':names[op.results[0]]},attrs)
                    materialized.add(op.results[0])
                    continue
                batched=recognize_batched_mac(op)
                if batched is not None:
                    contraction(op,[batched['lhs'],batched['rhs']],[batched['seed']])
                    continue
                mac = recognize_mac(op)
                if mac is not None:
                    contraction(op,[mac['lhs'],mac['rhs']],[mac['seed']],mac['transpose_rhs'])
                    continue
                extension = recognize_extension(op)
                if extension is not None:
                    src=extension['accumulator']
                    previous=next((c for c in reversed(cb['commands']) if c['opcode']=='COMMIT' and c['operands']['dst']==names[src]),None)
                    origin=next((c for c in reversed(cb['commands']) if previous and c['opcode'] in ('MATMUL','MATMUL_RESIDENT') and c['operands']['dst']==previous['operands']['src']),None)
                    if origin is None or origin.get('attributes',{}).get('accumulator_seed'):
                        raise Unsupported('extended contraction needs one unseeded retained accumulator')
                    if 'bias' in extension:
                        seed=tensor(extension['bias'])
                    else:
                        residual=extension['residual'];coefficient=extension['coefficient']
                        seed=names[op.results[0]]+'_seed'
                        cb['tensors'][seed]={**_spec(op.results[0]),'role':'intermediate'}
                        emit('AFFINE_MAP',{'x':tensor(residual),'dst':seed},{'weights':[coefficient],'constant':0,'activation':'none','output_dtype':'i32'})
                    acc=names[op.results[0]]+'_acc'
                    emit(origin['opcode'],{**origin['operands'],'dst':acc},{**origin.get('attributes',{}),'accumulator_seed':seed})
                    cb['commands'][-1]['_source_op_index']=origin['_source_op_index']
                    emit('COMMIT',{'src':acc,'dst':names[op.results[0]]},{'output_dtype':'i32','epilogue':[]})
                    if src not in output_names and len(list(src.uses))==1:
                        cb['commands'].remove(previous)
                        cb['commands'].remove(origin)
                    materialized.add(op.results[0])
                    continue
                operand_sum=recognize_operand_sum(op)
                if operand_sum is not None:
                    scales=[operand_sum['lhs_scale'],operand_sum['rhs_scale']]
                    attrs={'weights':[1,1],'constant':0,'activation':'none','output_dtype':'i32'}
                    if operand_sum['per_operand_rounding']:
                        attrs.update({'operand_scales':scales,'per_operand_rounding':True})
                    elif all(float(x).is_integer() for x in scales):
                        attrs['weights']=[int(x) for x in scales]
                    else:
                        raise Unsupported('single-rounding operand sum lacks an exact integer construction')
                    emit('AFFINE_MAP',{'x':tensor(operand_sum['lhs']),'y':tensor(operand_sum['rhs']),'dst':names[op.results[0]]},attrs)
                    materialized.add(op.results[0])
                    continue
                readout = recognize_readout(op)
                if readout is not None:
                    from frontend_admission import contraction_readout_refusal
                    refusal = contraction_readout_refusal(readout)
                    if refusal is not None:
                        raise Unsupported(refusal)
                    src=readout['src']
                    previous=next((c for c in reversed(cb['commands']) if c['opcode'] in ('COMMIT','AFFINE_MAP','WINDOW_SUM') and c['operands']['dst']==names[src]),None)
                    attrs={'output_dtype':'i8','epilogue':(['relu'] if readout['relu'] else [])+['acc_scale'],'activation':'relu' if readout['relu'] else 'none','readout_semantics':'f32_product_rne'}
                    if 'scales' in readout:
                        gains=readout['scales']
                        if all(x==gains[0] for x in gains): attrs['acc_scale']=gains[0]
                        else: attrs['acc_scale_by_column']=gains
                    else:
                        attrs['acc_scale_tensor']=names[readout['runtime_scale']]
                    if previous is not None and _spec(src)['dtype']=='i32' and src not in output_names and len(list(src.uses))==1:
                        cb['commands'].remove(previous)
                        emit(previous['opcode'],{**previous['operands'],'dst':names[op.results[0]]},{**previous.get('attributes',{}),**attrs})
                    elif previous is not None and previous['opcode']=='COMMIT':
                        emit('COMMIT',{'src':previous['operands']['src'],'dst':names[op.results[0]]},attrs)
                    elif _spec(src)['dtype']=='i8':
                        emit('AFFINE_MAP',{'x':tensor(src),'dst':names[op.results[0]]},{'weights':[1],'constant':0,**attrs})
                    else:
                        raise Unsupported('readout needs a retained exact integer producer')
                    materialized.add(op.results[0])
                    continue
                promoted=recognize_promoted_integer(op)
                if promoted is not None:
                    source,attrs=promoted
                    emit('DEVICE_PROMOTED_INTEGER',{'src':tensor(source),'dst':names[op.results[0]]},attrs)
                    materialized.add(op.results[0])
                    continue
                integer=recognize_integer_arithmetic(op)
                if integer is not None:
                    sources,attrs=integer
                    emit('DEVICE_INTEGER_AFFINE',{**{key:tensor(value) for key,value in zip(('x','y'),sources)},'dst':names[op.results[0]]},attrs)
                    materialized.add(op.results[0])
                    continue
                floating=recognize_quantized_float(op,list(block.args))
                if floating is not None:
                    source,attrs=floating
                    emit('DEVICE_QUANTIZED_FLOAT',{'src':tensor(source),'dst':names[op.results[0]]},attrs)
                    materialized.add(op.results[0])
                    continue
                region=op.regions[0].block
                body=list(region.ops)
                args=list(region.args)
                if len(op.results)!=1 or body[-1].name!='linalg.yield':
                    raise Unsupported('generic must yield one tensor')
                # Recognize sum and gated-prefix reductions from actual scalar SSA.
                iters=[str(x) for x in _attr(op,'iterator_types',[])]
                if any('reduction' in x for x in iters):
                    maps=list(_attr(op,'indexing_maps',[]))
                    from xdsl.ir.affine import AffineDimExpr
                    if len(ins)!=1 or len(outs)!=1 or not maps:
                        raise Unsupported('reduction needs one input and one initializer')
                    root=body[-1].operands[0].owner
                    if root.name not in ('arith.addi','arith.addf') or args[len(ins)] not in root.operands:
                        raise Unsupported('reduction yield is not accumulator plus source')
                    term=next(v for v in root.operands if v is not args[len(ins)])
                    initial=_const(outs[0])
                    if term.owner.name=='arith.select':
                        select=term.owner
                        cmp=select.operands[0].owner
                        if cmp.name!='arith.cmpi' or _int(_attr(cmp,'predicate')) not in (3,7):
                            raise Unsupported('prefix gate must be a nonstrict index comparison')
                        lhs,rhs=cmp.operands
                        if lhs.owner.name!='linalg.index' or rhs.owner.name!='linalg.index':
                            raise Unsupported('prefix gate does not compare loop indices')
                        rd=_int(_attr(lhs.owner,'dim'));od=_int(_attr(rhs.owner,'dim'))
                        if 'reduction' not in iters[rd] or 'parallel' not in iters[od]:
                            raise Unsupported('prefix index gate has wrong direction')
                        if _const(select.operands[2])!=0:
                            raise Unsupported('prefix exclusion is not literal zero')
                        form=_affine_expr(select.operands[1],args,{})
                        if form != [1,0,0]:
                            raise Unsupported('prefix source must be an exact widening of input')
                        srcmap=maps[0].data
                        axis=next((j for j,e in enumerate(srcmap.results) if isinstance(e,AffineDimExpr) and e.position==rd),None)
                        if axis is None:
                            raise Unsupported('prefix reduced input axis is unknown')
                        attrs={'mode':'prefix','axis':axis,'initial':int(initial),'output_dtype':_spec(op.results[0])['dtype']}
                    else:
                        form=_affine_expr(term,args,{})
                        if form != [1,0,0]:
                            raise Unsupported('sum term must be an exact widening of input')
                        dims=[j for j,e in enumerate(maps[0].data.results) if isinstance(e,AffineDimExpr) and 'reduction' in iters[e.position]]
                        attrs={'mode':'reduce','dims':dims,'average':False,'initial':int(initial),'output_dtype':_spec(op.results[0])['dtype']}
                    if _spec(ins[0])['dtype'] in ('i8','i32') and _spec(op.results[0])['dtype']==_spec(ins[0])['dtype']:
                        attrs['integer_wrap']=True
                    emit('WINDOW_SUM',{'src':tensor(ins[0]),'dst':names[op.results[0]]},attrs)
                    materialized.add(op.results[0])
                    continue
                yielded=body[-1].operands[0]
                activation='none'
                if getattr(yielded.owner,'name',None) in ('arith.maxsi','arith.maximumf','arith.maxnumf'):
                    a,b=yielded.owner.operands
                    try:
                        zero=_const(a)
                        yielded=b
                    except Unsupported:
                        zero=_const(b)
                        yielded=a
                    if zero!=0:
                        raise Unsupported('only rectification at literal zero is admitted')
                    activation='relu'
                scalar_env={}
                for arg,input_value in zip(args,ins):
                    try: scalar_value=_const(input_value)
                    except Unsupported: continue
                    scalar_env[arg]=[0]*len(args)+[Fraction(scalar_value)]
                expr=_affine_expr(yielded,args,scalar_env)
                if len(expr)>len(ins)+1 and any(expr[len(ins):-1]):
                    raise Unsupported('generic reads output initializer')
                if any(str(v.type) not in ('i8','i32') for v in args[:len(ins)]):
                    raise Unsupported('floating generic needs declared quantized representation')
                iters=[str(x) for x in _attr(op,'iterator_types',[])]
                if any('reduction' in x for x in iters):
                    raise Unsupported('generic reduction needs separate reduction lowering')
                weights=expr[:len(ins)]
                if len(ins)>2 or any(Fraction(x).denominator!=1 for x in weights+[expr[-1]]):
                    raise Unsupported('affine map requires exact bounded integer weights')
                if any(abs(int(x))>381 for x in weights):
                    raise Unsupported('affine weights exceed declared bound')
                maps=list(_attr(op,'indexing_maps',[]))
                from xdsl.ir.affine import AffineDimExpr
                if maps and any(tuple(x.data.results)!=tuple(AffineDimExpr(j) for j in range(x.data.num_dims)) for x in maps):
                    raise Unsupported('affine generic uses nonidentity indexing')
                emit('AFFINE_MAP',{**{'x' if j==0 else 'y':tensor(v) for j,v in enumerate(ins)},'dst':names[op.results[0]]},{'weights':[int(x) for x in weights],'constant':int(expr[-1]),'activation':activation,'output_dtype':_spec(op.results[0])['dtype'],'scale_numerator':'1','scale_denominator':'1'})
                materialized.add(op.results[0])
            else:
                raise Unsupported('no authored lowering for '+op.name)
        for v,dests in output_names.items():
            tensor(v)
            for dst in dests[1:]:
                cb['tensors'][dst]={**_spec(v),'role':'output'}
                emit('MOVEMENT',{'src':names[v],'dst':dst},{'form':'identity','output_dtype':_spec(v)['dtype']})
        # An argument returned directly needs its own fully written output.
        for v,dests in output_names.items():
            if v in block.args:
                for dst in dests:
                    cb['tensors'][dst]={**_spec(v),'role':'output'}
                    emit('MOVEMENT',{'src':names[v],'dst':dst},{'form':'identity','output_dtype':_spec(v)['dtype']})
        # Remove destination-only unmaterialized tensor.empty/fill values.
        keep={v for c in cb['commands'] for v in c.get('operands',{}).values()}
        keep.update(cb['outputs']);keep.update(names[v] for v in block.args)
        cb['tensors']={k:v for k,v in cb['tensors'].items() if k in keep}
        device_indices={c['_source_op_index'] for c in cb['commands'] if c['opcode']!='HOST_REGION'}
        cb['params']['lane_placement'].extend({'region':getattr(_attr(ops[j],'prov.region_id'),'data','source_'+str(j)),'lane':'device'} for j in sorted(device_indices))
        if not cb['params']['lane_placement']:
            cb['params']['lane_placement']=[{'region':'program','lane':'device'}]
        cb['_source_ops']=ops[:-1]
        cb['_source_op_count']=len(ops)-1
        cb['_source_names']=names
        cb['_source_args']=list(block.args)
        cb['_source_returns']=list(ops[-1].operands)
        return cb
    except Unsupported as exc:
        cb['commands']=[]
        cb['declined']={'op':ops[current[0]].name,'reason':str(exc)}
        cb['params']['lane_placement']=[]
        return cb
