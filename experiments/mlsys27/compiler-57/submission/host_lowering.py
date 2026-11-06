"""Explicit host-task lowering for independently ineligible scalar regions.

The frontend selects the host route. This module translates its original typed
scalar operations; it never selects a host route for a mesh-eligible operation.
"""
from __future__ import annotations
import math
import struct
from xdsl.dialects.builtin import TensorType, DenseIntOrFPElementsAttr, IntegerAttr, FloatAttr
from xdsl.ir.affine import AffineDimExpr, AffineConstantExpr, AffineBinaryOpExpr


def _attr(op,key,default=None):
    return op.properties.get(key,op.attributes.get(key,default))


def _name(op):
    if op.name=='builtin.unregistered':
        return _attr(op,'op_name__').data
    return op.name


def _type(t):
    return 'i64' if str(t)=='index' else str(t)


def _bytes(t):
    return {'i1':1,'i8':1,'i16':2,'bf16':2,'f16':2,'i32':4,'f32':4,'i64':8,'f64':8}[_type(t)]


def _dtype(v):
    return _type(v.type.element_type if isinstance(v.type,TensorType) else v.type)


_SCALAR = {
 'arith.constant','arith.addi','arith.subi','arith.muli','arith.divsi','arith.divui',
 'arith.remsi','arith.remui','arith.andi','arith.ori','arith.xori','arith.shli','arith.shrsi','arith.shrui',
 'arith.addf','arith.subf','arith.mulf','arith.divf','arith.negf','arith.cmpi','arith.cmpf','arith.select',
 'arith.extsi','arith.extui','arith.trunci','arith.extf','arith.truncf','arith.sitofp','arith.uitofp',
 'arith.fptosi','arith.fptoui','arith.bitcast','arith.index_cast','arith.index_castui',
 'arith.maxsi','arith.minsi','arith.maxui','arith.minui','arith.maximumf','arith.minimumf','arith.maxnumf','arith.minnumf',
 'math.exp','math.exp2','math.log','math.log2','math.log10','math.sqrt','math.rsqrt','math.absf','math.absi',
 'math.sin','math.cos','math.tanh','math.erf','math.expm1','math.log1p',
 'math.roundeven','math.floor','math.ceil','math.trunc','math.powf','math.fma',
 'linalg.index','linalg.yield','tensor.extract','scf.for','scf.yield'
}


def supports(op):
    if op.name=='arith.constant' and len(op.results)==1 and isinstance(op.results[0].type,TensorType):
        return isinstance(_attr(op,'value'),DenseIntOrFPElementsAttr)
    if _name(op) in ('quant_ext.quantize_per_tensor','quant_ext.dequantize_per_tensor',
                   'quant_ext.quantize_per_channel','quant_ext.dequantize_per_channel'):
        return len(op.operands)>=3 and len(op.results)==1
    if op.name not in ('linalg.generic','linalg.map','linalg.reduce') or not op.results:
        return False
    scalar_ops=[o for region in op.regions for o in region.walk()]
    if any(_name(o) not in _SCALAR for o in scalar_ops):
        return False
    # These three have no LLVM23 intrinsic. Select the standard host functions
    # only at their declared float/double boundary rather than inventing a
    # half-precision arithmetic or double-rounding policy.
    for scalar in scalar_ops:
        if _name(scalar) in ('math.erf','math.expm1','math.log1p'):
            if len(scalar.operands)!=1 or len(scalar.results)!=1:
                return False
            if _dtype(scalar.results[0]) not in ('f32','f64'):
                return False
            if _dtype(scalar.operands[0])!=_dtype(scalar.results[0]):
                return False
    return True


def _affine(e,expr,ivs):
    if isinstance(expr,AffineDimExpr): return ivs[expr.position]
    if isinstance(expr,AffineConstantExpr): return expr.value
    if isinstance(expr,AffineBinaryOpExpr):
        a,b=_affine(e,expr.lhs,ivs),_affine(e,expr.rhs,ivs)
        kind=expr.kind.name
        if kind=='Add': return e.add(a,b)
        if kind=='Mul': return e.mul(a,b)
        if kind=='FloorDiv': return e.udiv(a,b)
        if kind=='Mod': return e.urem(a,b)
        if kind=='CeilDiv': return e.udiv(e.add(a,e.sub(b,1)),b)
    raise ValueError('unsupported host affine index')


def _linear(e,indices,shape):
    answer=0
    for index,size in zip(indices,shape): answer=e.add(e.mul(answer,size),index)
    return answer


class Scalar:
    def __init__(self,e,names,state,ivs=()):
        self.e=e;self.names=names;self.state=state;self.ivs=ivs;self.values={}

    def instruction(self,text,typ):
        v=self.e.fresh();self.e.emit(f'{v} = {text}');return (v,typ)

    def constant(self,value,typ):
        if isinstance(value,(IntegerAttr,FloatAttr)):
            text=str(value)
        elif typ.startswith('i'):
            text=f'{int(value)} : {typ}'
        else:
            value=float(value)
            if math.isfinite(value): text=f'{value:.17e} : {typ}'
            elif typ=='f32': text=f'0x{struct.unpack("<I",struct.pack("<f",value))[0]:08X} : f32'
            else: text=f'0x{struct.unpack("<Q",struct.pack("<d",value))[0]:016X} : f64'
        return self.instruction(f'llvm.mlir.constant({text}) : {typ}',typ)

    def unary(self,op,a,dst=None):
        dst=dst or a[1]
        if op in ('sext','zext','trunc','fpext','fptrunc','sitofp','uitofp','fptosi','fptoui','bitcast'):
            return self.instruction(f'llvm.{op} {a[0]} : {a[1]} to {dst}',dst)
        return self.instruction(f'llvm.{op} {a[0]} : {dst}',dst)

    def binary(self,op,a,b):
        return self.instruction(f'llvm.{op} {a[0]}, {b[0]} : {a[1]}',a[1])

    def compare(self,pred,a,b,fp=False):
        return self.instruction(f'llvm.{"fcmp" if fp else "icmp"} "{pred}" {a[0]}, {b[0]} : {a[1]}','i1')

    def select(self,c,a,b):
        return self.instruction(f'llvm.select {c[0]}, {a[0]}, {b[0]} : i1, {a[1]}',a[1])

    def intrinsic(self,name,args,typ):
        operands=', '.join(x[0] for x in args);types=', '.join(x[1] for x in args)
        return self.instruction(f'llvm.call_intrinsic "llvm.{name}.{typ}"({operands}) : ({types}) -> {typ}',typ)

    def external_math(self,name,args,typ):
        if typ not in ('f32','f64'):
            raise ValueError('standard host math requires an f32 or f64 source boundary')
        symbol=name+('f' if typ=='f32' else '')
        types=[x[1] for x in args]
        self.e.declare_function(symbol,types,typ)
        operands=', '.join(x[0] for x in args)
        return self.instruction(f'llvm.call @{symbol}({operands}) : ({", ".join(types)}) -> {typ}',typ)

    def load(self,tensor,index,typ):
        address=self.e.ptr(tensor,self.e.mul(index,_bytes(typ)))
        p=self.instruction(f'llvm.inttoptr {self.e.const(address)} : i64 to !llvm.ptr','!llvm.ptr')
        return self.instruction(f'llvm.load {p[0]} : !llvm.ptr -> {typ}',typ)

    def store(self,tensor,index,value):
        address=self.e.ptr(tensor,self.e.mul(index,_bytes(value[1])))
        p=self.instruction(f'llvm.inttoptr {self.e.const(address)} : i64 to !llvm.ptr','!llvm.ptr')
        self.e.emit(f'llvm.store {value[0]}, {p[0]} : {value[1]}, !llvm.ptr')

    def materialize(self,value):
        """Keep original literal bytes in readonly module data."""
        name=self.names[value]
        done=self.state.setdefault('host_literals',set())
        if value in done: return name
        owner=value.owner
        if not hasattr(owner,'name'): return name
        if owner.name=='arith.constant' and isinstance(value.type,TensorType):
            attr=_attr(owner,'value')
            if isinstance(attr,DenseIntOrFPElementsAttr):
                # Dense storage is already little endian. Repacking its Python
                # floating values can quiet NaNs or change their payloads.
                raw=bytes(attr.data.data)
                if name in self.e.names:
                    _literal(self.e,owner,self.names,self.state)
                else:
                    self.e._storage(name,len(raw),raw)
                    done.add(value)
        elif owner.name=='tensor.splat':
            shape=value.type.get_shape();typ=_dtype(value);scalar=self.value(owner.operands[0])
            self.e.loop(0,math.prod(shape),1,lambda i:self.store(name,i,scalar));done.add(value)
        elif owner.name=='linalg.fill':
            shape=value.type.get_shape();scalar=self.value(owner.operands[0])
            self.e.loop(0,math.prod(shape),1,lambda i:self.store(name,i,scalar));done.add(value)
        return name

    def value(self,value):
        if value in self.values:return self.values[value]
        op=value.owner;name=_name(op);typ=_dtype(value)
        if name=='arith.constant':
            result=self.constant(_attr(op,'value'),typ)
        elif name=='scf.for':
            self.loop_results(op)
            return self.values[value]
        elif name=='linalg.index':
            dim=_attr(op,'dim').value.data;result=(self.e.const(self.ivs[dim]),'i64')
        elif name=='tensor.extract':
            tensor=op.operands[0];tn=self.materialize(tensor)
            indices=[self.value(v)[0] for v in op.operands[1:]]
            index=_linear(self.e,indices,tensor.type.get_shape())
            result=self.load(tn,index,typ)
        else:
            a=[self.value(x) for x in op.operands]
            integer={'addi':'add','subi':'sub','muli':'mul','divsi':'sdiv','divui':'udiv','remsi':'srem','remui':'urem','andi':'and','ori':'or','xori':'xor','shli':'shl','shrsi':'ashr','shrui':'lshr'}
            casts={'extsi':'sext','extui':'zext','trunci':'trunc','extf':'fpext','truncf':'fptrunc','sitofp':'sitofp','uitofp':'uitofp','fptosi':'fptosi','fptoui':'fptoui','bitcast':'bitcast'}
            short=name.split('.')[-1]
            if short in integer:result=self.binary(integer[short],a[0],a[1])
            elif short in ('addf','subf','mulf','divf'):result=self.binary({'addf':'fadd','subf':'fsub','mulf':'fmul','divf':'fdiv'}[short],a[0],a[1])
            elif short=='negf':result=self.unary('fneg',a[0])
            elif short in casts:result=self.unary(casts[short],a[0],typ)
            elif short.startswith('index_cast'):
                srcbits=int(a[0][1][1:]);dstbits=int(typ[1:])
                result=a[0] if srcbits==dstbits else self.unary('trunc' if srcbits>dstbits else ('zext' if short.endswith('ui') else 'sext'),a[0],typ)
            elif short=='select':result=self.select(*a)
            elif short=='cmpi':
                pred=['eq','ne','slt','sle','sgt','sge','ult','ule','ugt','uge'][_attr(op,'predicate').value.data]
                result=self.compare(pred,a[0],a[1])
            elif short=='cmpf':
                pred=['false','oeq','ogt','oge','olt','ole','one','ord','ueq','ugt','uge','ult','ule','une','uno','true'][_attr(op,'predicate').value.data]
                result=self.compare(pred,a[0],a[1],True)
            elif short in ('maxsi','minsi','maxui','minui'):
                pred=('s' if short.endswith('si') else 'u')+('gt' if short.startswith('max') else 'lt')
                result=self.select(self.compare(pred,a[0],a[1]),a[0],a[1])
            elif short in ('maximumf','minimumf','maxnumf','minnumf'):
                result=self.intrinsic({'maximumf':'maximum','minimumf':'minimum','maxnumf':'maxnum','minnumf':'minnum'}[short],a,typ)
            elif short=='rsqrt':
                sqrt=self.intrinsic('sqrt',a,typ);result=self.binary('fdiv',self.constant(1,typ),sqrt)
            elif short=='absi':
                zero=self.constant(0,typ);result=self.select(self.compare('slt',a[0],zero),self.binary('sub',zero,a[0]),a[0])
            elif name in ('math.erf','math.expm1','math.log1p'):
                result=self.external_math(short,a,typ)
            elif name.startswith('math.'):
                result=self.intrinsic({'absf':'fabs','powf':'pow'}.get(short,short),a,typ)
            else:raise ValueError('unimplemented explicit host scalar op '+name)
        self.values[value]=result
        return result

    def loop_results(self,op):
        """Lower scalar SCF iteration arguments to typed LLVM block arguments."""
        lower,upper,step=[self.value(v) for v in op.operands[:3]]
        init=[self.value(v) for v in op.operands[3:]]
        ident=self.e.fresh()[1:]
        head='host_for_'+ident;body='host_body_'+ident;tail='host_exit_'+ident
        iv=self.e.fresh();carried=[(self.e.fresh(),v[1]) for v in init]
        entry=[lower]+init
        self.e.emit('llvm.br ^'+head+'('+', '.join(v[0] for v in entry)+' : '+', '.join(v[1] for v in entry)+')')
        self.e.lines.append('  ^'+head+'('+', '.join(v[0]+': '+v[1] for v in [(iv,'i64')]+carried)+'):')
        condition=self.compare('slt',(iv,'i64'),upper)
        self.e.emit(f'llvm.cond_br {condition[0]}, ^{body}, ^{tail}')
        self.e.lines.append('  ^'+body+':')
        nested=Scalar(self.e,self.names,self.state,self.ivs)
        nested.values=dict(self.values)
        args=list(op.regions[0].block.args)
        nested.values.update(zip(args,[(iv,'i64')]+carried))
        term=list(op.regions[0].block.ops)[-1]
        yielded=[nested.value(v) for v in term.operands]
        nxt=nested.binary('add',(iv,'i64'),step)
        latch=[nxt]+yielded
        self.e.emit('llvm.br ^'+head+'('+', '.join(v[0] for v in latch)+' : '+', '.join(v[1] for v in latch)+')')
        self.e.lines.append('  ^'+tail+':')
        self.values.update(zip(op.results,carried))


def _generic(e,op,names,state):
    inputs=list(op.inputs);outputs=list(op.outputs)
    maps=list(_attr(op,'indexing_maps'))
    amap=[m.data for m in maps]
    nd=amap[0].num_dims
    extents=[None]*nd
    for value,m in zip(inputs+outputs,amap):
        for axis,expr in enumerate(m.results):
            if isinstance(expr,AffineDimExpr):
                extent=value.type.get_shape()[axis]
                prior=extents[expr.position]
                if prior is not None and prior!=extent: raise ValueError('inconsistent host iteration dimensions')
                extents[expr.position]=extent
    if any(x is None for x in extents):raise ValueError('host loop extent not determined by original maps')
    scalar=Scalar(e,names,state)
    for v in inputs+outputs:scalar.materialize(v)
    body=op.regions[0].block
    # Each reduction's original initializer is semantically observable.
    iters=list(_attr(op,'iterator_types',[]))
    reduction=any('reduction' in str(t) for t in iters)
    if reduction:
        for result,out in zip(op.results,outputs):
            if names[result]==names[out]:continue
            if getattr(out.owner,'name',None)=='tensor.empty':raise ValueError('reduction has undefined source initializer')
            count=math.prod(result.type.get_shape());typ=_dtype(result)
            e.loop(0,count,1,lambda i:scalar.store(names[result],i,scalar.load(names[out],i,typ)))
    def nest(dim,ivs):
        if dim<nd:
            e.loop(0,extents[dim],1,lambda iv:nest(dim+1,ivs+[iv]));return
        s=Scalar(e,names,state,ivs)
        for j,(v,m,arg) in enumerate(zip(inputs+outputs,amap,body.args)):
            coords=[_affine(e,x,ivs) for x in m.results]
            linear=_linear(e,coords,v.type.get_shape())
            target=names[op.results[j-len(inputs)]] if reduction and j>=len(inputs) else names[v]
            # Empty outs may legally be unused by the body.
            if arg.uses:s.values[arg]=s.load(target,linear,_dtype(v))
        for j,(y,result) in enumerate(zip(list(body.ops)[-1].operands,op.results)):
            val=s.value(y)
            oi=_linear(e,[_affine(e,x,ivs) for x in amap[len(inputs)+j].results],result.type.get_shape())
            s.store(names[result],oi,val)
    nest(0,[])


def _quant(e,op,names,state):
    src,scale,zp=op.operands[:3];dst=op.results[0]
    scalar=Scalar(e,names,state)
    for v in (src,scale,zp):scalar.materialize(v)
    shape=dst.type.get_shape();axisattr=_attr(op,'axis');axis=axisattr.value.data if axisattr else None
    if axis is not None and axis<0:axis+=len(shape)
    stride=math.prod(shape[axis+1:]) if axis is not None else None
    dequant='dequantize' in _name(op)
    def body(i):
        s=Scalar(e,names,state)
        q=s.load(names[src],i,_dtype(src))
        si=0 if axis is None else e.urem(e.udiv(i,stride),shape[axis])
        sc=s.load(names[scale],si,_dtype(scale));z=s.load(names[zp],si,_dtype(zp))
        fp=sc[1]
        if dequant:
            q=s.unary('sext',q,z[1]) if q[1]!=z[1] else q
            q=s.binary('sub',q,z);q=s.unary('sitofp',q,fp);q=s.binary('fmul',q,sc)
            if _dtype(dst)!=fp:q=s.unary('fptrunc' if fp=='f64' else 'fpext',q,_dtype(dst))
        else:
            if q[1]!=fp:q=s.unary('fpext' if fp=='f64' else 'fptrunc',q,fp)
            isnan=s.compare('uno',q,q,True)
            q=s.binary('fdiv',q,sc);q=s.intrinsic('roundeven',[q],fp)
            zf=s.unary('sitofp',z,fp);q=s.binary('fadd',q,zf)
            lo=_attr(op,'quant_min');hi=_attr(op,'quant_max')
            low=lo.value.data if lo is not None else -128;high=hi.value.data if hi is not None else 127
            q=s.intrinsic('maxnum',[q,s.constant(low,fp)],fp)
            q=s.intrinsic('minnum',[q,s.constant(high,fp)],fp)
            q=s.unary('fptosi',q,_dtype(dst))
            q=s.select(isnan,s.constant(0,_dtype(dst)),q)
        s.store(names[dst],i,q)
    e.loop(0,math.prod(shape),1,body)


def _reduce(e,op,names,state):
    src,init=op.operands;dst=op.results[0]
    s=Scalar(e,names,state)
    s.materialize(src);s.materialize(init)
    srcshape=src.type.get_shape();dstshape=dst.type.get_shape()
    dims=set(_attr(op,'dimensions').get_values())
    typ=_dtype(dst)
    e.loop(0,math.prod(dstshape),1,lambda i:s.store(names[dst],i,s.load(names[init],i,typ)))
    def visit(dim,ivs):
        if dim<len(srcshape):
            e.loop(0,srcshape[dim],1,lambda iv:visit(dim+1,ivs+[iv]));return
        oi=_linear(e,[iv for j,iv in enumerate(ivs) if j not in dims],dstshape)
        ii=_linear(e,ivs,srcshape)
        scalar=Scalar(e,names,state,ivs)
        args=list(op.regions[0].block.args)
        scalar.values[args[0]]=scalar.load(names[src],ii,_dtype(src))
        scalar.values[args[1]]=scalar.load(names[dst],oi,typ)
        yielded=list(op.regions[0].block.ops)[-1].operands[0]
        scalar.store(names[dst],oi,scalar.value(yielded))
    visit(0,[])


def _literal(e,op,names,state):
    """Construct original literal bytes without replacing a caller pointer."""
    value=op.results[0]
    attr=_attr(op,'value')
    if not isinstance(attr,DenseIntOrFPElementsAttr):
        raise ValueError('host literal requires original dense storage bytes')
    raw=bytes(attr.data.data)
    expected=math.prod(value.type.get_shape())*value.type.element_type.compile_time_size
    if len(raw)!=expected:
        raise ValueError('dense literal byte count does not match its typed shape')
    destination=names[value]
    source='__host_literal_'+str(len(state.setdefault('host_literal_buffers',{})))
    while source in e.tensors or source in e.pointers:
        source+='_' 
    # The readonly data has a distinct internal identity. The destination still
    # denotes exactly the pointer passed by the caller (or its owned scratch).
    e.tensors[source]={'shape':[len(raw)],'dtype':'i8','role':'intermediate'}
    e._storage(source,len(raw),raw)
    state['host_literal_buffers'][value]=source
    scalar=Scalar(e,names,state)
    e.loop(0,len(raw),1,lambda i:scalar.store(destination,i,scalar.load(source,i,'i8')))
    state.setdefault('host_literals',set()).add(value)


def lower(e,command,state):
    if command.get('opcode')!='HOST_REGION':return False
    op=command.get('_source_op');names=command.get('_value_names')
    if op is None or names is None or not supports(op):return False
    e.fence()
    if op.name=='arith.constant':_literal(e,op,names,state)
    elif _name(op).startswith('quant_ext.'):_quant(e,op,names,state)
    elif op.name=='linalg.reduce':_reduce(e,op,names,state)
    else:_generic(e,op,names,state)
    return True
