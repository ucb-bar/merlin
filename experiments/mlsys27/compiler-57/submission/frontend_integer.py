"""Structural admission for same-width integer MACs and their linear epilogues."""
from fractions import Fraction
from math import prod

from xdsl.dialects.builtin import TensorType
from xdsl.ir.affine import AffineDimExpr


def gathered_contractions(ops, outputs):
    """Fuse private affine read/reshape producers into a typed integer MAC.

    Keep the source storage map, so the contraction gathers taps directly into
    scratchpad. In particular, no expanded im2col tensor is written to DRAM.
    """
    from frontend_movement import recognize_movement
    plans = {}
    for op in ops:
        mac = recognize_integer_mac(op)
        if mac is None or len(mac['lhs'].type.get_shape()) != 2:
            continue
        value = mac['lhs']
        absorbed = set()
        while getattr(value.owner, 'name', None) in ('tensor.collapse_shape', 'tensor.expand_shape'):
            owner = value.owner
            if value in outputs or len(list(value.uses)) != 1:
                break
            absorbed.add(owner)
            value = owner.operands[0]
        gather = recognize_movement(value.owner) if getattr(value.owner, 'name', None) == 'linalg.generic' else None
        if gather is None or value in outputs or len(list(value.uses)) != 1:
            continue
        shape = list(value.type.get_shape())
        m, k = mac['lhs'].type.get_shape()
        split = next((j for j in range(1,len(shape)) if prod(shape[:j]) == m and prod(shape[j:]) == k), None)
        strides = gather['attributes']['strides']
        if split is None or any(s < 0 for s in strides):
            continue
        absorbed.add(value.owner)
        plans[op] = {'src':gather['src'], 'absorbed':absorbed,
                     'attributes':{'shape':shape, 'split':split, 'strides':strides,
                                   'storage_offset':gather['attributes']['storage_offset']}}
        rhs = mac['rhs']
        transpose = rhs.owner
        if (getattr(transpose,'name',None) != 'linalg.transpose' or rhs in outputs
                or len(list(rhs.uses)) != 1):
            continue
        from frontend import _array, _attr
        if _array(_attr(transpose,'permutation')) != [1,0]:
            continue
        source = transpose.inputs[0]
        rhs_ops = {transpose}
        while getattr(source.owner,'name',None) in ('tensor.collapse_shape','tensor.expand_shape'):
            if source in outputs or len(list(source.uses)) != 1:
                break
            rhs_ops.add(source.owner)
            source = source.owner.operands[0]
        plans[op]['rhs'] = source
        plans[op]['attributes']['transpose_rhs'] = True
        absorbed.update(rhs_ops)
    return plans


def column_biases(ops, outputs):
    """Recognize a private MAC result plus a same-width column vector."""
    from frontend import _attr
    plans = {}
    for op in ops:
        if op.name != 'linalg.generic' or len(op.results) != 1:
            continue
        inputs, out = list(op.inputs), list(op.outputs)
        if len(inputs) != 2 or len(out) != 1:
            continue
        for i, value in enumerate(inputs):
            mac = value.owner
            if recognize_integer_mac(mac) is None or value in outputs or len(list(value.uses)) != 1:
                continue
            shape = list(value.type.get_shape())
            bias = inputs[1-i]
            if (op.results[0].type != value.type or out[0].type != value.type
                    or list(bias.type.get_shape()) != shape[-1:]
                    or bias.type.element_type != value.type.element_type):
                continue
            maps = _attr(op,'indexing_maps',[])
            identity = tuple(AffineDimExpr(j) for j in range(len(shape)))
            expected = [identity, identity, identity]
            expected[1-i] = (AffineDimExpr(len(shape)-1),)
            if len(maps) != 3 or any(m.data.num_dims != len(shape) or m.data.num_symbols or tuple(m.data.results) != ex for m, ex in zip(maps,expected)):
                continue
            iters = _attr(op,'iterator_types',[])
            if len(iters) != len(shape) or any('parallel' not in str(t) for t in iters):
                continue
            block = op.regions[0].block
            body = list(block.ops)
            if (len(block.args) != 3 or len(body) != 2 or body[0].name != 'arith.addi'
                    or set(body[0].operands) != set(block.args[:2])
                    or body[-1].name != 'linalg.yield' or tuple(body[-1].operands) != tuple(body[0].results)):
                continue
            plans[mac] = {'op':op,'bias':bias,'result':op.results[0]}
    return plans


def recognize_integer_mac(op):
    if getattr(op, 'name', None) not in ('linalg.generic', 'linalg.matmul', 'linalg.batch_matmul'):
        return None
    ins = list(op.inputs)
    outs = list(op.outputs)
    if len(ins)!=2 or len(outs)!=1 or len(op.results)!=1 or len(op.regions)!=1:
        return None
    values = ins+outs+list(op.results)
    if any(not isinstance(v.type, TensorType) for v in values):
        return None
    dtype = str(values[0].type.element_type)
    if dtype not in ('i8', 'i32') or any(str(v.type.element_type)!=dtype for v in values):
        return None
    ash,bsh,dsh = (list(v.type.get_shape()) for v in (ins[0],ins[1],op.results[0]))
    rank = len(ash)
    if (rank not in (2,3) or len(bsh)!=rank or len(dsh)!=rank
            or ash[:-2]!=bsh[:-2] or ash[:-2]!=dsh[:-2]
            or ash[-1]!=bsh[-2] or dsh[-2:]!=[ash[-2],bsh[-1]]
            or list(outs[0].type.get_shape())!=dsh):
        return None
    from frontend_composition import _attr, _signed_cast_selector
    if not _signed_cast_selector(op):
        return None
    maps = _attr(op, 'indexing_maps')
    if maps is None:
        return None
    expected = [list(range(rank-2))+[rank-2,rank],
                list(range(rank-2))+[rank,rank-1], list(range(rank))]
    for attr, positions in zip(maps, expected):
        m = attr.data
        if m.num_dims!=rank+1 or m.num_symbols or len(m.results)!=len(positions):
            return None
        if any(not isinstance(x,AffineDimExpr) or x.position!=p for x,p in zip(m.results,positions)):
            return None
    if len(maps)!=3:
        return None
    iters = _attr(op, 'iterator_types')
    if iters is not None and (len(iters)!=rank+1 or any('parallel' not in str(t) for t in list(iters)[:-1]) or 'reduction' not in str(list(iters)[-1])):
        return None
    block = op.regions[0].block
    body = list(block.ops)
    if len(block.args)!=3 or len(body)!=3:
        return None
    mul,add,yield_op = body
    a,b,seed = block.args
    if (mul.name!='arith.muli' or set(mul.operands)!={a,b}
            or add.name!='arith.addi' or set(add.operands)!={mul.results[0],seed}
            or yield_op.name!='linalg.yield' or tuple(yield_op.operands)!=tuple(add.results)
            or any(str(v.type)!=dtype for v in list(block.args)+list(mul.results)+list(add.results))):
        return None
    return {'lhs':ins[0], 'rhs':ins[1], 'seed':outs[0]}


def contraction_epilogues(ops, outputs):
    """Fuse a closed linear graph containing one modular contraction result.

    Only private intermediates of an observable contraction epilogue are folded;
    unrelated elementwise graphs remain with their existing lowering.
    """
    from frontend import _affine_expr, _const, _attr, Unsupported
    plans = {}
    for result in outputs:
        absorbed = set()
        dtype = str(result.type.element_type)
        def expression(value):
            try:
                return {}, int(_const(value))
            except Unsupported:
                pass
            owner = value.owner
            if getattr(owner, 'name', None)!='linalg.generic' or recognize_integer_mac(owner):
                return {value: 1}, 0
            ins, outs = list(owner.inputs), list(owner.outputs)
            if len(outs)!=1 or str(value.type.element_type)!=dtype:
                raise Unsupported('epilogue must preserve integer storage')
            maps = _attr(owner, 'indexing_maps', [])
            if len(maps)!=len(ins)+1 or any(tuple(m.data.results)!=tuple(AffineDimExpr(j) for j in range(m.data.num_dims)) for m in maps):
                raise Unsupported('contraction epilogue must use identity maps')
            if any('parallel' not in str(t) for t in _attr(owner,'iterator_types',[])):
                raise Unsupported('contraction epilogue must be parallel')
            block = owner.regions[0].block
            body = list(block.ops)
            args = list(block.args)
            if (len(ins)>2 or any(str(v.type)!=dtype for v in args)
                    or any(inner.name not in ('arith.constant','arith.addi','arith.subi','arith.muli','linalg.yield')
                           or any(str(v.type)!=dtype for v in inner.results) for inner in body)):
                raise Unsupported('contraction epilogue requires same-width modular integer arithmetic')
            env = {}
            for arg, src in zip(args, ins):
                try: env[arg] = [0]*len(args)+[int(_const(src))]
                except Unsupported: pass
            form = _affine_expr(body[-1].operands[0], args, env)
            if form[len(ins)] or any(Fraction(v).denominator!=1 for v in form):
                raise Unsupported('epilogue reads initializer or has fractional coefficients')
            weights, constant = {}, int(form[-1])
            absorbed.add(owner)
            for src, coefficient in zip(ins, form):
                terms, bias = expression(src)
                constant += int(coefficient)*bias
                for leaf, weight in terms.items():
                    weights[leaf] = weights.get(leaf, 0)+int(coefficient)*weight
            return weights, constant
        try:
            weights, constant = expression(result)
        except (Unsupported, AttributeError):
            continue
        macs = [v for v in weights if recognize_integer_mac(v.owner)]
        others = [v for v in weights if v not in macs]
        if len(macs)!=1 or len(others)>1 or constant or not absorbed:
            continue
        mac = macs[0]
        if any(v in outputs and v is not result for op in absorbed | {mac.owner} for v in op.results):
            continue
        if any(use.operation not in absorbed for op in absorbed | {mac.owner} for v in op.results if v is not result for use in v.uses):
            continue
        if others and (others[0].type != result.type):
            continue
        plans[mac.owner] = dict(result=result, alpha=weights[mac], residual=others[0] if others else None,
                                beta=weights[others[0]] if others else 0, absorbed=absorbed)
    return plans
