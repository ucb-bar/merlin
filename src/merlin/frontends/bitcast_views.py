"""Normalize observed equal-width dtype views with exclusively value-reading users.

The pinned frontend trace authenticates the dropped dtype literal and operation
identity. A returned/mutated alias, changed element width or missing trace stays
opaque. Pure value clients need only the selected element's storage bits.
"""


def lower_captured_bitcast_views(module, metadata):
    import hashlib,json
    from pathlib import Path
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import TensorType,IntegerType,Float16Type,BFloat16Type,Float32Type,Float64Type
    from xdsl.rewriter import Rewriter
    from merlin.llvmlower._ir import point
    # This normalization is for immutable tensor-level capture, before any
    # memory bridge can expose storage aliases or writes behind an SSA value.
    if any(op.name.startswith(('memref.','bufferization.')) for op in module.walk()):return 0
    record=(metadata or {}).get('frontend_trace') or {}
    if not record:return 0
    path=Path(record['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('frontend trace digest changed')
    trace=json.loads(raw);nodes={n['id']:n for n in trace.get('graphs',{}).get('prepared',{}).get('nodes',[])}
    names={'float16':Float16Type(),'bfloat16':BFloat16Type(),'float32':Float32Type(),'float64':Float64Type(),**{f'int{b}':IntegerType(b) for b in (8,16,32,64)}}
    def width(typ):
        if isinstance(typ,IntegerType):return typ.width.data
        return {Float16Type:16,BFloat16Type:16,Float32Type:32,Float64Type:64}.get(type(typ))
    def value_only_clients(value):
        pending=[value];seen=set()
        while pending:
            value=pending.pop()
            if value in seen:continue
            seen.add(value)
            if not value.uses:return False
            for use in value.uses:
                client=use.operation
                if client.name=='linalg.generic' and use.index<len(client.inputs) and value not in client.outputs and all(child.name=='linalg.yield' or child.name.startswith(('arith.','math.')) for child in client.body.block.ops):continue
                if client.name=='tensor.extract' and use.index==0:continue
                if client.name in {'tensor.cast','tensor.expand_shape','tensor.collapse_shape','tensor.reshape'} and use.index==0:
                    if any(not isinstance(r.type,TensorType) or r.type.element_type!=value.type.element_type for r in client.results):return False
                    pending.extend(client.results);continue
                return False
        return True
    lowered=set();count=0
    for call in list(module.walk()):
        if call.name!='func.call' or len(call.operands)!=1 or len(call.results)!=1:continue
        ids=call.attributes.get('prov.source_node_ids')
        if ids is None or len(ids)!=1:continue
        node=nodes.get(ids.data[0].data,{})
        if node.get('op')!='call_function' or node.get('target')!='aten.view.dtype':continue
        args=node.get('args',[])
        if len(args)!=2 or args[1].get('kind')!='dtype':continue
        src,dst=call.operands[0].type,call.results[0].type
        if not isinstance(src,TensorType) or not isinstance(dst,TensorType):continue
        if src.get_shape()!=dst.get_shape() or any(d<0 for d in src.get_shape()):continue
        expected=names.get(args[1].get('value','').removeprefix('torch.'))
        if expected!=dst.element_type or width(src.element_type)!=width(dst.element_type) or width(src.element_type) is None:continue
        if not value_only_clients(call.results[0]):continue
        # The call is an external placeholder, never a user function definition.
        callee=call.callee.string_value()
        decl=next((o for o in module.body.block.ops if o.name=='func.func' and o.sym_name.data==callee),None)
        if decl is None or decl.body.blocks:continue
        ops=[];result=point([call.operands[0]],dst,lambda args,emit:emit(arith.BitcastOp(args[0],dst.element_type)),ops)
        result.owner.attributes.update(call.attributes)
        Rewriter.replace_op(call,ops,[result]);lowered.add(callee);count+=1
    remaining={o.callee.string_value() for o in module.walk() if o.name=='func.call'}
    for op in list(module.body.block.ops):
        if op.name=='func.func' and op.sym_name.data in lowered-remaining and not op.body.blocks:Rewriter.erase_op(op)
    return count
