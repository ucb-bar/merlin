"""Move a per-tensor Q through closed byte-independent movement to its readout.

This removes an otherwise observable intermediate f32 rounding. Every live
path must end in the same declared quantizer; arithmetic and dtype casts are
not carriers. No sample values are inspected.
"""


def fuse_readout_quantizers(model):
    import torch
    import math
    from merlin.targetgen.declared_calibration import dtype_metadata
    carriers={'aten.'+name+'.default' for name in (
        'clone','alias','detach','view','reshape','_unsafe_view','permute','transpose','t',
        'squeeze','unsqueeze','flatten','expand','contiguous')}
    carriers.update({'aten.transpose.int','aten.squeeze.dim','aten.squeeze.dims','aten.flatten.using_ints'})
    proofs=[]
    declines=[]
    for node in list(model.graph.nodes):
        if node.op!='call_function' or str(node.target)!='m2m_q.exact_readout.default':continue
        schema=node.target._schema.arguments
        values={arg.name:node.args[i] if i<len(node.args) else node.kwargs.get(arg.name,arg.default_value) for i,arg in enumerate(schema)}
        if values['output_scale'] is not None or not node.users:continue
        pending=[node];seen=set();moves=[];quantizers=[];closed=True;blocked=None
        while pending and closed:
            source=pending.pop()
            if source in seen:continue
            seen.add(source)
            if not source.users:closed=False;break
            for user in source.users:
                target=str(user.target)
                if user.op!='call_function' or not user.args or user.args[0] is not source:
                    closed=False;blocked=str(user.target);break
                if target in ('quantized_decomposed.quantize_per_tensor.default','m2m_q.quantize.default'):
                    qschema=user.target._schema.arguments
                    args=[user.args[i] if i<len(user.args) else user.kwargs.get(a.name,a.default_value) for i,a in enumerate(qschema)]
                    if len(args)!=6:closed=False;blocked=target+': unknown quantizer schema';break
                    if target.startswith('quantized_decomposed.') and args[5] not in (torch.int8,torch.uint8):
                        closed=False;blocked='quantizer is not byte storage';break
                    unsigned=args[5] is torch.uint8 if target.startswith('quantized_decomposed.') else args[5]
                    params=(args[1],args[2],args[3],args[4],unsigned)
                    if any(isinstance(x,torch.fx.Node) for x in params):closed=False;break
                    scale,zp,low,high,unsigned=params
                    bounds=(0,255) if unsigned else (-128,127)
                    if type(unsigned) is not bool or type(scale) not in (int,float) or not math.isfinite(scale) or scale<=0 or any(type(x) is not int for x in (zp,low,high)) or not bounds[0]<=low<high<=bounds[1] or not bounds[0]<=zp<=bounds[1]:
                        closed=False;blocked='unproved static byte quantizer parameters';break
                    quantizers.append((user,params))
                elif target in carriers:
                    moves.append(user);pending.append(user)
                else:closed=False;blocked=target;break
        if not closed or not quantizers or any(p!=quantizers[0][1] for _,p in quantizers):
            declines.append(dict(readout=node.name,blocked=blocked,quantizers=len(quantizers),users=[str(u.target) for u in node.users]))
            continue
        scale,zp,low,high,unsigned=quantizers[0][1]
        values.update(output_scale=scale,zero_point=zp,quant_min=low,quant_max=high,unsigned=unsigned)
        node.args=tuple(values[arg.name] for arg in schema);node.kwargs={}
        dtype=torch.uint8 if unsigned else torch.int8
        for moved in [node,*moves]:
            if isinstance(moved.meta.get('val'),torch.Tensor):moved.meta['val']=dtype_metadata(moved.meta['val'],dtype)
        for quantizer,_ in quantizers:
            quantizer.replace_all_uses_with(quantizer.args[0]);model.graph.erase_node(quantizer)
        proofs.append(dict(readout=node.name,movement=[str(m.target) for m in moves],scale=scale,zero_point=zp,quant_min=low,quant_max=high,unsigned=unsigned,proof='same per-tensor quantizer on every live movement path; no intermediate floating rounding'))
    model._merlin_readout_quantizer_declines=declines
    model.graph.lint();model.recompile()
    return proofs
