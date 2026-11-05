"""LLVM lowering of independently host-eligible floating-point islands.

The caller owns placement: these routines never turn an integer contraction into
host work. All tensor loops remain runtime loops in the emitted program.
"""
from math import prod
import struct


def _ty(dtype):
    return {'f16': 'half', 'f32': 'float', 'f64': 'double', 'bf16': 'bfloat', 'index': 'i64'}.get(dtype, dtype)


def _size(dtype):
    ty = _ty(dtype)
    if ty in ('half', 'bfloat'):
        return 2
    if ty == 'float':
        return 4
    if ty == 'double':
        return 8
    return max(1, (int(ty[1:]) + 7) // 8)


def _float(value, ty='float'):
    value = float(value)
    if ty == 'float':
        value = struct.unpack('f', struct.pack('f', value))[0]
    return '0x' + struct.pack('>d', value).hex().upper()


def _load(g, ptr, dtype):
    return g.value('load '+_ty(dtype)+', ptr '+str(ptr)+', align 1')


def _store(g, ptr, value, dtype):
    g.emit('store '+_ty(dtype)+' '+str(value)+', ptr '+str(ptr)+', align 1')


def _at(g, ptr, index, dtype='f32'):
    return g.gep(ptr, g.mul(index, _size(dtype)))


def _fop(g, op, a, b, ty='float'):
    return g.value(op+' '+ty+' '+str(a)+', '+str(b))


def _intrinsic(g, name, args, ty='float'):
    suffix = {'half':'f16', 'float':'f32', 'double':'f64'}[ty]
    symbol = 'llvm.'+name+'.'+suffix
    g.declarations.add('declare '+ty+' @'+symbol+'('+', '.join([ty]*len(args))+')')
    return g.value('call '+ty+' @'+symbol+'('+', '.join(ty+' '+str(v) for v in args)+')')


def _maximum(g, a, b, ty='float'):
    return _intrinsic(g, 'maximum', [a,b], ty)


def _mark_host(cb, command):
    attrs = command.get('attributes', {})
    record = attrs.get('record', {})
    entry = {'route':'H', 'reason':'floating-point island requires operations outside the integer accelerator datapath', 'op':command.get('opcode','HOST_LINALG')}
    region = attrs.get('source_region_id') or record.get('prov',{}).get('region_id')
    if region:
        entry['region_id'] = region
    placements = cb.setdefault('params',{}).setdefault('lane_placement',[])
    if entry not in placements:
        placements.append(entry)


def emit_host(g, cb, command, ptrs):
    """Lower declared f32 softmax, RMS normalization, or rotary embedding."""
    op = command['opcode']
    if op == 'HOST_LINALG':
        record = command.get('attributes',{}).get('record')
        if record is None:
            return False
        if record.get('kind') in ('quant_ext.quantize_per_tensor','quant_ext.dequantize_per_tensor'):
            ok = _emit_quantization(g,cb,record,ptrs,command)
        elif any(op.get('regions',0) for op in (record.get('scalar_body') or {}).get('operations',[])):
            ok = _emit_typed_linalg_host(g,cb,record,ptrs,command)
        else:
            ok = emit_linalg_host(g,cb,record,ptrs,command)
        if ok:
            _mark_host(cb,command)
        return ok
    if op not in ('SOFTMAX','RMSNORM','ROPE'):
        return False
    o = command.get('operands',{})
    a = command.get('attributes',{})
    src,dst = o.get('src'),o.get('dst')
    if src not in ptrs or dst not in ptrs:
        return False
    tensors = cb['tensors']
    shape = tensors[src]['shape']
    if tensors[src]['dtype'] != 'f32' or tensors[dst]['dtype'] != 'f32' or not shape:
        return False
    if op == 'SOFTMAX':
        axis = int(a.get('axis',a.get('dim',-1))) % len(shape)
        outer,channels,inner = prod(shape[:axis]),shape[axis],prod(shape[axis+1:])
        maxbuf,sumbuf = g.alloca(4),g.alloca(4)
        def row(r):
            out_idx=g.udiv(r,inner)
            inner_idx=g.urem(r,inner)
            base=g.add(g.mul(out_idx,channels*inner),inner_idx)
            _store(g,maxbuf,_float(float('-inf')),'f32')
            def maximum(j):
                v=_load(g,_at(g,ptrs[src],g.add(base,g.mul(j,inner))),'f32')
                m=_load(g,maxbuf,'f32')
                _store(g,maxbuf,_maximum(g,m,v),'f32')
            g.loop(channels,maximum)
            _store(g,sumbuf,_float(0),'f32')
            m=_load(g,maxbuf,'f32')
            def exp_sum(j):
                idx=g.add(base,g.mul(j,inner))
                x=_load(g,_at(g,ptrs[src],idx),'f32')
                x=_fop(g,'fsub',x,m)
                e=_intrinsic(g,'exp',[x])
                _store(g,_at(g,ptrs[dst],idx),e,'f32')
                s=_load(g,sumbuf,'f32')
                _store(g,sumbuf,_fop(g,'fadd',s,e),'f32')
            g.loop(channels,exp_sum)
            total=_load(g,sumbuf,'f32')
            def divide(j):
                addr=_at(g,ptrs[dst],g.add(base,g.mul(j,inner)))
                e=_load(g,addr,'f32')
                _store(g,addr,_fop(g,'fdiv',e,total),'f32')
            g.loop(channels,divide)
        g.loop(outer*inner,row)
    elif op == 'RMSNORM':
        gamma=o.get('gamma')
        if gamma not in ptrs or tensors[gamma]['dtype']!='f32':
            return False
        channels=shape[-1]
        epsilon=_float(a.get('epsilon',a.get('eps',1e-6)))
        sumbuf=g.alloca(4)
        def row(r):
            _store(g,sumbuf,_float(0),'f32')
            base=g.mul(r,channels)
            def sums(j):
                x=_load(g,_at(g,ptrs[src],g.add(base,j)),'f32')
                square=_fop(g,'fmul',x,x)
                total=_load(g,sumbuf,'f32')
                _store(g,sumbuf,_fop(g,'fadd',total,square),'f32')
            g.loop(channels,sums)
            total=_load(g,sumbuf,'f32')
            mean=_fop(g,'fdiv',total,_float(channels))
            root=_intrinsic(g,'sqrt',[_fop(g,'fadd',mean,epsilon)])
            inv=_fop(g,'fdiv',_float(1),root)
            def normalized(j):
                idx=g.add(base,j)
                x=_load(g,_at(g,ptrs[src],idx),'f32')
                gain=_load(g,_at(g,ptrs[gamma],j),'f32')
                y=_fop(g,'fmul',_fop(g,'fmul',x,inv),gain)
                _store(g,_at(g,ptrs[dst],idx),y,'f32')
            g.loop(channels,normalized)
        g.loop(prod(shape[:-1]),row)
    else:
        # The legacy rotary operation must state the angle convention; no
        # implicit position or frequency convention can be inferred from shape.
        angles=a.get('angles')
        if not isinstance(angles,list) or len(angles)!=shape[-1]//2 or shape[-1]%2:
            return False
        interleaved=a.get('interleaved',False)
        channels=shape[-1]
        def row(r):
            base=g.mul(r,channels)
            # Static frequency parameters are literals, while rows remain looped.
            import math
            for j,angle in enumerate(angles):
                left=2*j if interleaved else j
                right=2*j+1 if interleaved else j+channels//2
                x=_load(g,_at(g,ptrs[src],g.add(base,left)),'f32')
                y=_load(g,_at(g,ptrs[src],g.add(base,right)),'f32')
                c,s=_float(math.cos(float(angle))),_float(math.sin(float(angle)))
                lo=_fop(g,'fsub',_fop(g,'fmul',x,c),_fop(g,'fmul',y,s))
                hi=_fop(g,'fadd',_fop(g,'fmul',y,c),_fop(g,'fmul',x,s))
                _store(g,_at(g,ptrs[dst],g.add(base,left)),lo,'f32')
                _store(g,_at(g,ptrs[dst],g.add(base,right)),hi,'f32')
        g.loop(prod(shape[:-1]),row)
    _mark_host(cb,command)
    return True


def _emit_quantization(g,cb,record,ptrs,command):
    """The source float-to-storage boundary is explicitly outside the i8 mesh."""
    ins=record.get('ins',[])
    results=record.get('results',[])
    if len(ins)!=3 or len(results)!=1:
        return False
    operands=command.get('operands',{})
    src=operands.get('ins0',_record_name(ins[0]))
    dst=operands.get('dst','value_'+str(record['id'])+'_0')
    if src not in ptrs or dst not in ptrs:
        return False
    attrs=record.get('attributes',{})
    quantize=record['kind']=='quant_ext.quantize_per_tensor'
    input_dtype=ins[0]['dtype']
    output_dtype=results[0]['dtype']
    if quantize and (input_dtype!='f32' or output_dtype!='i8'):
        return False
    if not quantize and (input_dtype!='i8' or output_dtype!='f32'):
        return False
    def parameter(index):
        item=ins[index]
        if 'const_value' in item:
            return item['const_value']
        name=operands.get('ins'+str(index),_record_name(item))
        constant=cb.get('params',{}).get('constant_tensors',{}).get(name)
        if constant and len(constant['values'])==1:
            return constant['values'][0]
        if name not in ptrs:
            raise ValueError('missing quantization parameter '+str(index))
        return _load(g,ptrs[name],item['dtype'])
    scale=parameter(1)
    zero=parameter(2)
    scale_type=_ty(ins[1]['dtype'])
    if isinstance(scale,(float,int)):
        scale=_float(scale,scale_type)
    zero_type=_ty(ins[2]['dtype'])
    if isinstance(zero,(float,int)):
        zero=str(int(zero))
    zero_float=g.value('sitofp '+zero_type+' '+str(zero)+' to double')
    if scale_type!='double':
        scale=g.value('fpext '+scale_type+' '+str(scale)+' to double')
    qmin=int(attrs.get('quant_min','-128').split(':')[0].strip())
    qmax=int(attrs.get('quant_max','127').split(':')[0].strip())
    def element(index):
        x=_load(g,_at(g,ptrs[src],index,input_dtype),input_dtype)
        if quantize:
            x=g.value('fpext float '+str(x)+' to double')
            scaled=_fop(g,'fdiv',x,scale,'double')
            rounded=_intrinsic(g,'roundeven',[scaled],'double')
            shifted=_fop(g,'fadd',rounded,zero_float,'double')
            lower=_intrinsic(g,'maximum',[shifted,_float(qmin,'double')],'double')
            clipped=_intrinsic(g,'minimum',[lower,_float(qmax,'double')],'double')
            result=g.value('fptosi double '+str(clipped)+' to i8')
        else:
            x=g.value('sitofp i8 '+str(x)+' to double')
            shifted=_fop(g,'fsub',x,zero_float,'double')
            scaled=_fop(g,'fmul',shifted,scale,'double')
            result=g.value('fptrunc double '+str(scaled)+' to float')
        _store(g,_at(g,ptrs[dst],index,output_dtype),result,output_dtype)
    g.loop(prod(results[0]['shape']),element)
    return True


def _record_name(record):
    source=record.get('source')
    if not source:
        return None
    if source[0]=='arg':
        return 'arg_'+str(source[1])
    return 'value_'+str(source[1])+'_'+str(record.get('result_index',0))


def _parse_map(text):
    from xdsl.context import Context
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import Builtin
    ctx=Context()
    ctx.load_dialect(Builtin)
    return Parser(ctx,text).parse_attribute().data


def _affine(g,expr,ivs):
    from xdsl.ir.affine import AffineDimExpr,AffineConstantExpr,AffineBinaryOpExpr
    if isinstance(expr,AffineDimExpr):
        return ivs[expr.position]
    if isinstance(expr,AffineConstantExpr):
        return expr.value
    if not isinstance(expr,AffineBinaryOpExpr):
        raise ValueError('unsupported symbolic affine indexing map')
    left,right=_affine(g,expr.lhs,ivs),_affine(g,expr.rhs,ivs)
    kind=expr.kind.name
    if kind=='Add':return g.add(left,right)
    if kind=='Mul':return g.mul(left,right)
    if kind=='Mod':return g.op('srem',left,right)
    if kind=='FloorDiv':return g.op('sdiv',left,right)
    if kind=='CeilDiv':return g.op('sdiv',g.add(left,g.sub(right,1)),right)
    raise ValueError('unsupported affine expression '+kind)


def _index(g,mapping,shape,ivs):
    index=0
    for expr,extent in zip(mapping.results,shape):
        index=g.add(g.mul(index,extent),_affine(g,expr,ivs))
    return index


def _constant(text,dtype):
    # Constants are structural scalar attributes printed by the admitted reader.
    from xdsl.context import Context
    from xdsl.dialects.builtin import Builtin,IntegerAttr,FloatAttr
    from xdsl.parser import Parser
    ctx=Context()
    ctx.load_dialect(Builtin)
    attr=Parser(ctx,text).parse_attribute()
    if isinstance(attr,IntegerAttr):
        return str(attr.value.data)
    if isinstance(attr,FloatAttr):
        return _float(attr.value.data,_ty(dtype))
    raise ValueError('unsupported scalar constant')


def _scalar(g,op,values,types,ivs):
    name=op['op']
    operands=[values[v] for v in op['operands']]
    operand_types=[types[v] for v in op['operands']]
    results=op['results']
    dtype=results[0] if results else 'f32'
    ty=_ty(dtype)
    attrs=op.get('attributes',{})
    binary={'arith.addf':'fadd','arith.subf':'fsub','arith.mulf':'fmul','arith.divf':'fdiv',
            'arith.remf':'frem','arith.addi':'add','arith.subi':'sub','arith.muli':'mul',
            'arith.divsi':'sdiv','arith.divui':'udiv','arith.remsi':'srem','arith.remui':'urem',
            'arith.andi':'and','arith.ori':'or','arith.xori':'xor','arith.shli':'shl',
            'arith.shrsi':'ashr','arith.shrui':'lshr'}
    if name=='arith.constant':
        return [_constant(attrs['value'],dtype)]
    if name in binary:
        return [g.value(binary[name]+' '+ty+' '+str(operands[0])+', '+str(operands[1]))]
    if name=='arith.negf':
        return [g.value('fneg '+ty+' '+str(operands[0]))]
    if name in ('arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf'):
        intr={'arith.maximumf':'maximum','arith.minimumf':'minimum','arith.maxnumf':'maxnum','arith.minnumf':'minnum'}[name]
        return [_intrinsic(g,intr,operands,ty)]
    if name in ('arith.maxsi','arith.maxui','arith.minsi','arith.minui'):
        pred={'arith.maxsi':'sgt','arith.maxui':'ugt','arith.minsi':'slt','arith.minui':'ult'}[name]
        cond=g.value('icmp '+pred+' '+ty+' '+str(operands[0])+', '+str(operands[1]))
        return [g.value('select i1 '+cond+', '+ty+' '+str(operands[0])+', '+ty+' '+str(operands[1]))]
    if name=='arith.select':
        return [g.value('select i1 '+str(operands[0])+', '+ty+' '+str(operands[1])+', '+ty+' '+str(operands[2]))]
    if name in ('arith.cmpf','arith.cmpi'):
        pred=int(attrs['predicate'].split(':')[0].strip())
        if name=='arith.cmpf':
            predicates=['false','oeq','ogt','oge','olt','ole','one','ord','ueq','ugt','uge','ult','ule','une','uno','true']
            instruction='fcmp'
        else:
            predicates=['eq','ne','slt','sle','sgt','sge','ult','ule','ugt','uge']
            instruction='icmp'
        input_ty=_ty(operand_types[0])
        return [g.value(instruction+' '+predicates[pred]+' '+input_ty+' '+str(operands[0])+', '+str(operands[1]))]
    casts={'arith.extsi':'sext','arith.extui':'zext','arith.trunci':'trunc',
           'arith.extf':'fpext','arith.truncf':'fptrunc','arith.sitofp':'sitofp','arith.uitofp':'uitofp',
           'arith.fptosi':'fptosi','arith.fptoui':'fptoui','arith.bitcast':'bitcast',
           'arith.index_cast':'sext','arith.index_castui':'zext'}
    if name in casts:
        input_ty=_ty(operand_types[0])
        cast=casts[name]
        if input_ty==ty:return operands[:1]
        if name in ('arith.index_cast','arith.index_castui'):
            cast=cast if int(input_ty[1:])<int(ty[1:]) else 'trunc'
        return [g.value(cast+' '+input_ty+' '+str(operands[0])+' to '+ty)]
    if name=='linalg.index':
        dim=int(attrs.get('dim',attrs.get('dimension','0')).split(':')[0].strip())
        return [ivs[dim]]
    intrinsics={'math.exp':'exp','math.exp2':'exp2','math.log':'log','math.log2':'log2',
                'math.sqrt':'sqrt','math.sin':'sin','math.cos':'cos','math.absf':'fabs',
                'math.ceil':'ceil','math.floor':'floor','math.trunc':'trunc','math.roundeven':'roundeven',
                'math.round':'round','math.powf':'pow','math.fma':'fma'}
    if name in intrinsics:
        return [_intrinsic(g,intrinsics[name],operands,ty)]
    if name=='math.rsqrt':
        root=_intrinsic(g,'sqrt',operands,ty)
        return [_fop(g,'fdiv',_float(1,ty),root,ty)]
    if name=='math.tanh':
        # Stable sign-symmetric formula avoids overflow for large magnitudes.
        mag=_intrinsic(g,'fabs',operands,ty)
        e=_intrinsic(g,'exp',[_fop(g,'fmul',mag,_float(-2,ty),ty)],ty)
        value=_fop(g,'fdiv',_fop(g,'fsub',_float(1,ty),e,ty),_fop(g,'fadd',_float(1,ty),e,ty),ty)
        g.declarations.add('declare '+ty+' @llvm.copysign.'+{'float':'f32','double':'f64'}[ty]+'('+ty+', '+ty+')')
        return [_intrinsic(g,'copysign',[value,operands[0]],ty)]
    raise ValueError('unsupported host scalar operation '+name)


def emit_linalg_host(g,cb,record,ptrs,command=None):
    """Lower a scalar-DAG float island; integer contraction never enters this path.

    Input/result pointer names may be supplied explicitly in command operands;
    otherwise the frontend's structural arg_N/value_ID_K binding is used.
    """
    if record.get('family')=='contraction' or record.get('kind') in ('linalg.matmul','linalg.batch_matmul'):
        return False
    body=record.get('scalar_body')
    if not body or not record.get('indexing_maps'):
        return False
    ins,outs=record.get('ins',[]),record.get('outs',[])
    result_types=record.get('results',[])
    if not any(item.get('dtype','').startswith(('f','bf')) for item in ins+outs+result_types):
        return False
    if body.get('effects') not in ([],None):
        return False
    maps=[_parse_map(text) for text in record['indexing_maps']]
    if len(maps)!=len(ins)+len(outs):
        return False
    from xdsl.ir.affine import AffineDimExpr
    ndim=maps[0].num_dims
    extents=[None]*ndim
    for item,mapping in zip(ins+outs,maps):
        for extent,expr in zip(item['shape'],mapping.results):
            if isinstance(expr,AffineDimExpr):
                current=extents[expr.position]
                if current is not None and current!=extent:
                    return False
                extents[expr.position]=extent
    if any(e is None or e<=0 for e in extents):
        return False
    opnds=(command or {}).get('operands',{})
    input_names=[opnds.get('ins'+str(i),_record_name(item)) for i,item in enumerate(ins)]
    output_names=[opnds.get('dst' if i==0 else 'dst'+str(i),'value_'+str(record['id'])+'_'+str(i)) for i in range(len(result_types))]
    if any(name not in ptrs for name,item in zip(input_names,ins) if 'const_value' not in item):
        return False
    if any(name not in ptrs for name in output_names):
        return False
    # Reductions require exactly the source's initialized outs values.
    if record.get('reduction_dims'):
        for k,(out,result) in enumerate(zip(outs,result_types)):
            count=prod(result['shape'])
            const=out.get('const_value')
            oldptr=ptrs.get(_record_name(out))
            if const is None and oldptr is None:
                return False
            def init(i,k=k,out=out,result=result,const=const,oldptr=oldptr):
                dtype=result['dtype']
                value=(_float(const,_ty(dtype)) if dtype.startswith('f') else str(int(const))) if const is not None else _load(g,_at(g,oldptr,i,dtype),dtype)
                _store(g,_at(g,ptrs[output_names[k]],i,dtype),value,dtype)
            g.loop(count,init)
    # Validate scalar vocabulary before emitting a loop, so unsupported code does
    # not leave a partially emitted task for a caller to accidentally accept.
    allowed={'arith.constant','arith.addf','arith.subf','arith.mulf','arith.divf','arith.remf','arith.negf',
             'arith.addi','arith.subi','arith.muli','arith.divsi','arith.divui','arith.remsi','arith.remui',
             'arith.andi','arith.ori','arith.xori','arith.shli','arith.shrsi','arith.shrui','arith.select',
             'arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf','arith.maxsi','arith.maxui','arith.minsi','arith.minui',
             'arith.cmpf','arith.cmpi','arith.extsi','arith.extui','arith.trunci','arith.extf','arith.truncf','arith.sitofp',
             'arith.uitofp','arith.fptosi','arith.fptoui','arith.bitcast','arith.index_cast','arith.index_castui','linalg.index',
             'math.exp','math.exp2','math.log','math.log2','math.sqrt','math.sin','math.cos','math.absf','math.ceil','math.floor',
             'math.trunc','math.roundeven','math.round','math.powf','math.fma','math.rsqrt','math.tanh'}
    if any(op['op'] not in allowed or op.get('regions',0) for op in body['operations']):
        return False
    def element(flat):
        ivs=[]
        for dim,extent in enumerate(extents):
            ivs.append(g.urem(g.udiv(flat,prod(extents[dim+1:])),extent))
        values,types={},{}
        for i,item in enumerate(ins):
            dtype=item['dtype']
            const=item.get('const_value')
            if const is not None:
                value=_float(const,_ty(dtype)) if dtype.startswith('f') else str(int(const))
            else:
                index=_index(g,maps[i],item['shape'],ivs)
                value=_load(g,_at(g,ptrs[input_names[i]],index,dtype),dtype)
            values['arg:'+str(i)]=value
            types['arg:'+str(i)]=dtype
        out_indices=[]
        for k,item in enumerate(outs):
            index=_index(g,maps[len(ins)+k],item['shape'],ivs)
            out_indices.append(index)
            dtype=item['dtype']
            values['arg:'+str(len(ins)+k)]=_load(g,_at(g,ptrs[output_names[k]],index,dtype),dtype)
            types['arg:'+str(len(ins)+k)]=dtype
        for i,item in enumerate(body.get('captures',[])):
            dtype=item['dtype']
            if 'const_value' not in item:
                raise ValueError('host scalar capture must be a scalar constant')
            const=item['const_value']
            values['capture:'+str(i)]=_float(const,_ty(dtype)) if dtype.startswith('f') else str(int(const))
            types['capture:'+str(i)]=dtype
        for i,op in enumerate(body['operations']):
            produced=_scalar(g,op,values,types,ivs)
            for k,(value,dtype) in enumerate(zip(produced,op['results'])):
                key='op:'+str(i)+':'+str(k)
                values[key]=value
                types[key]=dtype
        for k,ref in enumerate(body['yields']):
            dtype=result_types[k]['dtype']
            _store(g,_at(g,ptrs[output_names[k]],out_indices[k],dtype),values[ref],dtype)
    g.loop(prod(extents),element)
    return True


def _emit_typed_linalg_host(g,cb,record,ptrs,command):
    """Lower nested scalar SCF from the original parsed operation, retaining SSA.

    The compact inventory intentionally omits nested regions. This path reparses
    the same source bytes and joins every captured tensor to the full operation
    inventory; it never synthesizes normalization from a provenance nickname.
    """
    text=cb.get('_source_text')
    if not text or record.get('family')=='contraction':
        return False
    from xdsl.parser import Parser
    from xdsl.dialects.builtin import TensorType,IntegerAttr,FloatAttr
    from parser_support.linalg_iface import make_linalg_context,_op_name,_initialization_only,_ins_outs,_semantic_attributes
    module=Parser(make_linalg_context(),text).parse_module()
    definitions=[op for op in module.walk() if op.name=='func.func' and op.body.blocks]
    fn=next((op for op in definitions if op.sym_name.data=='forward'),None)
    if fn is None and len(definitions)==1:
        fn=definitions[0]
    if fn is None or len(fn.body.blocks)!=1:
        return False
    block=fn.body.blocks[0]
    direct=[op for op in block.ops if _op_name(op) not in ('func.return','linalg.yield')]
    payload=[op for op in direct if not _initialization_only(op)]
    top=payload[record['id']]
    if _op_name(top)!=record['kind'] or len(top.regions)!=1 or len(top.regions[0].blocks)!=1:
        return False
    input_values,output_values=_ins_outs(top)
    maps=[_parse_map(m) for m in record['indexing_maps']]
    if len(maps)!=len(input_values)+len(output_values):
        return False
    from xdsl.ir.affine import AffineDimExpr
    ndim=maps[0].num_dims
    extents=[None]*ndim
    for item,mapping in zip(record['ins']+record['outs'],maps):
        for extent,expr in zip(item['shape'],mapping.results):
            if isinstance(expr,AffineDimExpr):
                if extents[expr.position] not in (None,extent):
                    return False
                extents[expr.position]=extent
    if any(e is None or e<=0 for e in extents):
        return False
    names={}
    for i,arg in enumerate(block.args):
        if isinstance(arg.type,TensorType):
            names[arg]='arg_'+str(i)
    source_values=cb.get('params',{}).get('source_values',[])
    for binding in source_values:
        op=direct[binding['op_index']]
        names[op.results[binding.get('result_index',0)]]=binding['tensor']
    for i,op in enumerate(payload):
        for k,value in enumerate(op.results):
            names.setdefault(value,'value_'+str(i)+'_'+str(k))
    opnds=command.get('operands',{})
    output_names=[opnds.get('dst' if k==0 else 'dst'+str(k),names.get(value)) for k,value in enumerate(top.results)]
    if any(name not in ptrs for name in output_names):
        return False
    source_names={value:opnds.get('ins'+str(i),names.get(value)) for i,value in enumerate(input_values)}
    # Preflight the complete nested structure before emitting any task.
    scalar_supported={'arith.constant','arith.addf','arith.subf','arith.mulf','arith.divf','arith.remf','arith.negf',
        'arith.addi','arith.subi','arith.muli','arith.divsi','arith.divui','arith.remsi','arith.remui',
        'arith.andi','arith.ori','arith.xori','arith.shli','arith.shrsi','arith.shrui','arith.select',
        'arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf','arith.maxsi','arith.maxui','arith.minsi','arith.minui',
        'arith.cmpf','arith.cmpi','arith.extsi','arith.extui','arith.trunci','arith.extf','arith.truncf','arith.sitofp',
        'arith.uitofp','arith.fptosi','arith.fptoui','arith.bitcast','arith.index_cast','arith.index_castui','linalg.index',
        'math.exp','math.exp2','math.log','math.log2','math.sqrt','math.sin','math.cos','math.absf','math.ceil','math.floor',
        'math.trunc','math.roundeven','math.round','math.powf','math.fma','math.rsqrt','math.tanh'}
    structural={'scf.for','scf.if','scf.yield','linalg.yield','tensor.extract'}
    for inner in top.walk():
        if inner is top:
            continue
        kind=_op_name(inner)
        if kind not in scalar_supported|structural:
            return False
        if kind in scalar_supported and inner.regions:
            return False
        if kind=='scf.for':
            step_owner=inner.operands[2].owner
            attrs={**step_owner.attributes,**getattr(step_owner,'properties',{})}
            step_attr=attrs.get('value')
            if _op_name(step_owner)!='arith.constant' or not isinstance(step_attr,IntegerAttr) or step_attr.value.data<=0:
                return False
        if kind=='tensor.extract':
            tensor=inner.operands[0]
            if tensor not in names or names[tensor] not in ptrs:
                return False
    # Integer helper SSA is permitted here only as part of the float island.
    if not all(str(value.type.element_type) in ('f32','f64') for value in top.results):
        return False
    captures={}
    def literal(value):
        owner=value.owner
        kind=_op_name(owner)
        attrs={**owner.attributes,**getattr(owner,'properties',{})}
        if kind=='arith.constant':
            attr=attrs.get('value')
            if isinstance(attr,IntegerAttr):return str(attr.value.data)
            if isinstance(attr,FloatAttr):return _float(attr.value.data,_ty(str(value.type)))
        raise ValueError('unbound scalar capture '+kind)
    def read(value,env):
        return env[value] if value in env else literal(value)
    def lower_block(region_block,env,ivs):
        for inner in region_block.ops:
            kind=_op_name(inner)
            if kind in ('scf.yield','linalg.yield'):
                return [read(value,env) for value in inner.operands]
            if kind=='tensor.extract':
                tensor=inner.operands[0]
                shape=list(tensor.type.get_shape())
                index=0
                for extent,value in zip(shape,inner.operands[1:]):
                    index=g.add(g.mul(index,extent),read(value,env))
                dtype=str(tensor.type.element_type)
                env[inner.results[0]]=_load(g,_at(g,ptrs[names[tensor]],index,dtype),dtype)
            elif kind=='scf.for':
                lo,hi,step=[read(value,env) for value in inner.operands[:3]]
                initial=[read(value,env) for value in inner.operands[3:]]
                slots=[g.alloca(_size(str(value.type))) for value in inner.results]
                for slot,value,result in zip(slots,initial,inner.results):
                    _store(g,slot,value,str(result.type))
                distance=g.sub(hi,lo)
                nonnegative=g.max(distance,0)
                count=g.udiv(g.add(nonnegative,g.sub(step,1)),step)
                nested_block=inner.regions[0].blocks[0]
                def iteration(i):
                    nested=dict(env)
                    nested[nested_block.args[0]]=g.add(lo,g.mul(i,step))
                    for arg,slot in zip(nested_block.args[1:],slots):
                        nested[arg]=_load(g,slot,str(arg.type))
                    yielded=lower_block(nested_block,nested,ivs)
                    for slot,value,result in zip(slots,yielded,inner.results):
                        _store(g,slot,value,str(result.type))
                g.loop(count,iteration)
                for result,slot in zip(inner.results,slots):
                    env[result]=_load(g,slot,str(result.type))
            elif kind=='scf.if':
                slots=[g.alloca(_size(str(value.type))) for value in inner.results]
                def branch(region):
                    yielded=lower_block(region.blocks[0],dict(env),ivs)
                    for slot,value,result in zip(slots,yielded,inner.results):
                        _store(g,slot,value,str(result.type))
                g.conditional(read(inner.operands[0],env),lambda:branch(inner.regions[0]),
                              (lambda:branch(inner.regions[1])) if len(inner.regions)>1 else None)
                for result,slot in zip(inner.results,slots):
                    env[result]=_load(g,slot,str(result.type))
            else:
                keys={value:'v'+str(i) for i,value in enumerate(inner.operands)}
                values={key:read(value,env) for value,key in keys.items()}
                types={key:str(value.type) for value,key in keys.items()}
                scalar={'op':kind,'operands':[keys[value] for value in inner.operands],
                        'results':[str(value.type) for value in inner.results],
                        'attributes':_semantic_attributes(inner)}
                produced=_scalar(g,scalar,values,types,ivs)
                for result,value in zip(inner.results,produced):
                    env[result]=value
        raise ValueError('host scalar block has no yield')
    if record.get('reduction_dims'):
        # Generic nested bodies here may be pure parallel, but reduction bodies
        # still receive exactly their source initializer.
        for k,item in enumerate(record['outs']):
            value=item.get('const_value',item.get('init_value'))
            oldptr=ptrs.get(_record_name(item))
            dtype=item['dtype']
            if value is None and oldptr is None:
                return False
            def initialize(i,k=k,item=item,value=value,oldptr=oldptr,dtype=dtype):
                scalar=_float(value,_ty(dtype)) if value is not None else _load(g,_at(g,oldptr,i,dtype),dtype)
                _store(g,_at(g,ptrs[output_names[k]],i,dtype),scalar,dtype)
            g.loop(prod(item['shape']),initialize)
    def element(flat):
        ivs=[g.urem(g.udiv(flat,prod(extents[dim+1:])),extent) for dim,extent in enumerate(extents)]
        env={}
        body_block=top.regions[0].blocks[0]
        for i,item in enumerate(record['ins']):
            dtype=item['dtype']
            if 'const_value' in item:
                scalar=_float(item['const_value'],_ty(dtype)) if dtype.startswith('f') else str(int(item['const_value']))
            else:
                name=source_names[input_values[i]]
                index=_index(g,maps[i],item['shape'],ivs)
                scalar=_load(g,_at(g,ptrs[name],index,dtype),dtype)
            env[body_block.args[i]]=scalar
        indices=[]
        for k,item in enumerate(record['outs']):
            index=_index(g,maps[len(input_values)+k],item['shape'],ivs)
            indices.append(index)
            dtype=item['dtype']
            # A destination-only outs argument can remain poison; unused loads
            # have no semantic use and are removed by stock LLVM.
            env[body_block.args[len(input_values)+k]]=_load(g,_at(g,ptrs[output_names[k]],index,dtype),dtype)
        yielded=lower_block(body_block,env,ivs)
        for k,value in enumerate(yielded):
            dtype=record['results'][k]['dtype']
            _store(g,_at(g,ptrs[output_names[k]],indices[k],dtype),value,dtype)
    g.loop(prod(extents),element)
    return True
