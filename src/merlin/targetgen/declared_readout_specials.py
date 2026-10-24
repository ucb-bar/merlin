"""Keep nonfinite residual propagation outside exact integer readout.

The finite MAC and residual use the unchanged rational readout. Only outputs
whose broadcast residual is nonfinite select the declared byte-Q special code.
No runtime values are observed while constructing this graph.
"""


def retain_special_readouts(model, policy, *, floating_policy=None):
    if policy!='byte_storage_zero_nan_saturating_infinity':return 0
    import math
    import torch
    count=0
    proofs=[]
    for node in list(model.graph.nodes):
        if node.op!='call_function' or str(node.target)!='m2m_q.exact_readout.default':continue
        schema=node.target._schema.arguments
        values={arg.name:node.args[i] if i<len(node.args) else node.kwargs.get(arg.name,arg.default_value) for i,arg in enumerate(schema)}
        bias=values['bias']
        if bias is None:continue
        beta=values['beta']
        if type(beta) not in (int,float) or not math.isfinite(beta):
            raise ValueError('residual readout needs a finite literal coefficient')
        floating=values['output_scale'] is None
        if floating and beta and (floating_policy!='single_residual_quiet_word_f32' or beta<0):
            from merlin.targetgen.capture_refusals import NumericCaptureRefused
            raise NumericCaptureRefused('floating residual readout lacks a declared single-NaN word proof',code='unquantized_special_readout_unproved',details={'beta':beta,'policy':floating_policy})
        original_users=list(node.users)
        with model.graph.inserting_before(node):
            nan=model.graph.call_function(torch.ops.aten.isnan.default,(bias,))
            positive=model.graph.call_function(torch.ops.aten.eq.Scalar,(bias,float('inf')))
            negative=model.graph.call_function(torch.ops.aten.eq.Scalar,(bias,-float('inf')))
            zero=model.graph.call_function(torch.ops.aten.zeros_like.default,(bias,))
            safe=bias
            for mask in (nan,positive,negative):
                safe=model.graph.call_function(torch.ops.aten.where.self,(mask,zero,safe))
        values['bias']=safe
        node.args=tuple(values[arg.name] for arg in schema);node.kwargs={}
        tail=node
        if beta:
            if floating:
                with model.graph.inserting_after(tail):words=model.graph.call_function(torch.ops.aten.view.dtype,(bias,torch.int32))
                with model.graph.inserting_after(words):quiet=model.graph.call_function(torch.ops.aten.bitwise_or.Scalar,(words,1<<22))
                with model.graph.inserting_after(quiet):nan_value=model.graph.call_function(torch.ops.aten.view.dtype,(quiet,torch.float32))
                tail_anchor=nan_value
                selections=[(positive,float('inf')),(negative,0. if values['relu'] else -float('inf')),(nan,nan_value)]
            else:
                high,low=values['quant_max'],values['quant_min']
                if values['relu']:low=max(low,values['zero_point'])
                selections=[(positive,high if beta>0 else low),(negative,low if beta>0 else high),(nan,0)]
                tail_anchor=tail
            for mask,code in selections:
                with model.graph.inserting_after(tail_anchor):
                    fill=code if isinstance(code,torch.fx.Node) else model.graph.call_function(torch.ops.aten.full_like.default,(tail,code))
                with model.graph.inserting_after(tail_anchor if isinstance(code,torch.fx.Node) else fill):selected=model.graph.call_function(torch.ops.aten.where.self,(mask,fill,tail))
                tail=tail_anchor=selected
        for user in original_users:user.replace_input_with(node,tail)
        proofs.append(dict(readout=node.name,bias=bias.name,beta=beta,
                           dependence="declared readout bias broadcast to accumulator shape" if beta else "none: literal zero residual coefficient",
                           finite="original exact rational readout with nonfinite bias selected to zero",
                           special="only the corresponding nonfinite bias indices select a byte Q code",
                           floating=floating,quiet_nan_word='input f32 word OR quiet bit' if floating else None,
                           quant_min=values['quant_min'],quant_max=values['quant_max'],
                           zero_point=values['zero_point'],relu=values['relu'],nan_storage=0))
        count+=1
    model._merlin_special_readout_proofs=proofs
    model.graph.lint();model.recompile()
    return count
