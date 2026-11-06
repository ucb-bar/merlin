"""Source-only certificate for the reviewed three-result gradient contraction.

The role vocabulary comes from operations.contraction form convolution_backward
in the selected software declaration. A certificate binds original return SSA
values; it neither creates aliases nor chooses a numerical/device lowering.
"""
from math import prod
from xdsl.dialects.builtin import TensorType,NoneAttr,IntegerAttr,DenseIntOrFPElementsAttr
from xdsl.ir.affine import AffineDimExpr
from frontend_composition import recognize_mac


class _NoMatch(Exception):
    pass


def _require(condition):
    if not condition:raise _NoMatch()


def _attr(op,key):
    _require(not (key in op.properties and key in op.attributes))
    return op.properties.get(key,op.attributes.get(key))


def _spec(value,dtype=None,rank=None):
    typ=value.type
    _require(isinstance(typ,TensorType) and isinstance(typ.encoding,NoneAttr))
    shape=tuple(typ.get_shape())
    _require(all(type(d) is int and d>0 for d in shape))
    _require(dtype is None or str(typ.element_type)==dtype)
    _require(rank is None or len(shape)==rank)
    return shape,str(typ.element_type)


def _map(op,expected,loop_count):
    attr=_attr(op,'indexing_maps')
    _require(attr is not None)
    maps=list(attr)
    _require(len(maps)==len(expected))
    for item,axes in zip(maps,expected):
        mapping=item.data
        _require(mapping.num_dims==loop_count and mapping.num_symbols==0 and len(mapping.results)==len(axes))
        _require(all(isinstance(expr,AffineDimExpr) and expr.position==axis for expr,axis in zip(mapping.results,axes)))


def _iterators(op,expected):
    values=_attr(op,'iterator_types')
    if values is None and op.name=='linalg.matmul':
        values=op.get_iterator_types()
    _require(values is not None and [getattr(a,'data',None) for a in values]==expected)


def _trivial_yield(op,argument_types):
    _require(len(op.regions)==1 and len(op.regions[0].blocks)==1)
    block=op.regions[0].block;body=list(block.ops)
    _require(tuple(str(a.type) for a in block.args)==tuple(argument_types))
    _require(len(body)==1 and body[0].name=='linalg.yield')
    _require(tuple(body[0].operands)==(block.args[0],))


def gradient_roles(ops,args,returns):
    """Return original return-SSA -> reviewed gradient role, or None.

    Requires the complete unique entry-block roster and actual ordered return.
    All original tensor arguments are dense signed-byte readonly operands; only
    exact tensor transposes, zero-seeded signed MACs and an outer signed sum
    contribute to the three results. No names/provenance classify the graph.
    """
    try:
        return _gradient_roles(list(ops),list(args),list(returns))
    except (_NoMatch,AttributeError,TypeError,ValueError,IndexError):
        return None


def _gradient_roles(ops,args,returns):
    _require(len(args)==3 and len(set(args))==3 and len(returns)==3 and len(set(returns))==3)
    block=args[0].owner
    _require(all(a.owner is block for a in args) and tuple(block.args)==tuple(args))
    original=list(block.ops)
    _require(original and original[-1].name=='func.return')
    terminator=original[-1]
    _require(tuple(terminator.operands)==tuple(returns))
    _require(tuple(ops) in (tuple(original),tuple(original[:-1])))
    ops=original[:-1];opset=set(ops)
    function=block.parent.parent
    _require(function.name=='func.func' and len(function.body.blocks)==1)
    function_type=_attr(function,'function_type')
    _require(tuple(function_type.inputs)==tuple(a.type for a in args))
    _require(tuple(function_type.outputs)==tuple(v.type for v in returns))
    for arg in args:_spec(arg,'i8',2)
    for value in returns:
        _spec(value,'i32')
        _require(value not in args and value.owner in opset and len(value.owner.results)==1 and value.owner.results[0] is value)

    def owned(value):
        owner=value.owner
        _require(owner in opset and len(owner.results)==1 and owner.results[0] is value)
        return owner

    def empty(value,visited):
        owner=owned(value);_require(owner.name=='tensor.empty' and not owner.operands and not owner.regions)
        _spec(value);visited.add(owner)

    def scalar_zero(value,visited):
        owner=owned(value)
        _require(owner.name=='arith.constant' and not owner.operands and not owner.regions and str(value.type)=='i32')
        attribute=_attr(owner,'value')
        _require(isinstance(attribute,IntegerAttr) and attribute.type==value.type and attribute.value.data==0)
        visited.add(owner)

    def zero(value,shape,visited):
        _require(_spec(value,'i32')[0]==shape)
        owner=owned(value);visited.add(owner)
        if owner.name=='arith.constant':
            attribute=_attr(owner,'value')
            _require(isinstance(attribute,DenseIntOrFPElementsAttr) and attribute.type==value.type)
            data=attribute.get_values()
            _require(len(data)==prod(shape) and all(type(x) is int and x==0 for x in data))
            _require(not owner.operands and not owner.regions)
        elif owner.name=='tensor.splat':
            _require(len(owner.operands)==1 and not owner.regions)
            scalar_zero(owner.operands[0],visited)
        elif owner.name=='linalg.fill':
            _require(len(owner.inputs)==1 and len(owner.outputs)==1)
            _require(len(owner.operands)==2)
            scalar_zero(owner.inputs[0],visited)
            _require(_spec(owner.outputs[0],'i32')[0]==shape)
            empty(owner.outputs[0],visited)
            _trivial_yield(owner,('i32','i32'))
        else:
            raise _NoMatch()

    def transpose(value,visited):
        owner=owned(value)
        _require(owner.name=='linalg.transpose' and len(owner.inputs)==1 and len(owner.outputs)==1 and len(owner.operands)==2)
        source=owner.inputs[0]
        _require(source in args)
        shape=_spec(source,'i8',2)[0]
        _require(_spec(value,'i8',2)[0]==shape[::-1])
        _require(_spec(owner.outputs[0],'i8',2)[0]==shape[::-1])
        permutation=_attr(owner,'permutation')
        _require(permutation is not None and tuple(permutation.get_values())==(1,0))
        _trivial_yield(owner,('i8','i8'))
        empty(owner.outputs[0],visited)
        visited.add(owner)
        return source

    def mac(value,visited):
        owner=owned(value)
        _require(owner.name in ('linalg.generic','linalg.matmul'))
        binding=recognize_mac(owner)
        _require(binding is not None and not binding['transpose_rhs'])
        _map(owner,((0,2),(2,1),(0,1)),3)
        _iterators(owner,['parallel','parallel','reduction'])
        lhs,rhs=binding['lhs'],binding['rhs']
        left=_spec(lhs,'i8',2)[0];right=_spec(rhs,'i8',2)[0]
        out=_spec(value,'i32',2)[0]
        _require(left[1]==right[0] and out==(left[0],right[1]))
        zero(binding['seed'],out,visited)
        visited.add(owner)
        return lhs,rhs

    def bias(value,visited):
        reduced=value;owner=owned(value)
        if owner.name=='tensor.expand_shape':
            _require(len(owner.operands)==1 and not owner.regions)
            reduced=owner.operands[0]
            source_shape=_spec(reduced,'i32',1)[0]
            _require(_spec(value,'i32',2)[0]==(1,source_shape[0]))
            reassociation=_attr(owner,'reassociation')
            _require(reassociation is not None and len(reassociation)==1)
            group=list(reassociation)[0]
            _require(tuple(x.value.data for x in group)==(0,1))
            extents=_attr(owner,'static_output_shape')
            _require(extents is not None and tuple(extents.get_values())==(1,source_shape[0]))
            visited.add(owner);owner=owned(reduced)
        _require(owner.name=='linalg.generic' and len(owner.inputs)==1 and len(owner.outputs)==1 and len(owner.operands)==2)
        dy=owner.inputs[0]
        _require(dy in args)
        shape=_spec(dy,'i8',2)[0]
        _require(_spec(reduced,'i32',1)[0]==(shape[1],))
        _map(owner,((0,1),(1,)),2)
        _iterators(owner,['reduction','parallel'])
        zero(owner.outputs[0],(shape[1],),visited)
        _require(len(owner.regions)==1 and len(owner.regions[0].blocks)==1)
        body_block=owner.regions[0].block;body=list(body_block.ops)
        _require(tuple(str(a.type) for a in body_block.args)==('i8','i32') and len(body)==3)
        widened,added,yield_op=body
        element,old=body_block.args
        _require(widened.name=='arith.extsi' and tuple(widened.operands)==(element,) and len(widened.results)==1 and str(widened.results[0].type)=='i32')
        _require(added.name=='arith.addi' and len(added.results)==1 and str(added.results[0].type)=='i32')
        _require(tuple(added.operands) in ((old,widened.results[0]),(widened.results[0],old)))
        _require(yield_op.name=='linalg.yield' and tuple(yield_op.operands)==(added.results[0],))
        visited.add(owner)
        return dy

    certificates=[]
    for db in returns:
        visited=set()
        try:
            dy=bias(db,visited)
            others=[v for v in returns if v is not db]
            bindings={value:mac(value,visited) for value in others}
            dx=next((v for v in others if bindings[v][0] is dy),None)
            dw=next((v for v in others if bindings[v][1] is dy),None)
            _require(dx is not None and dw is not None and dx is not dw)
            w=transpose(bindings[dx][1],visited)
            x=transpose(bindings[dw][0],visited)
            _require(len({x,dy,w})==3 and {x,dy,w}==set(args))
            xs=_spec(x,'i8',2)[0];dys=_spec(dy,'i8',2)[0];ws=_spec(w,'i8',2)[0]
            _require(xs[0]==dys[0] and xs[1]==ws[0] and dys[1]==ws[1])
            _require(_spec(dx,'i32',2)[0]==xs and _spec(dw,'i32',2)[0]==ws)
            # Every contributing producer is known pure. Only unused literal
            # scalars and static empty destination declarations may remain.
            for operation in ops:
                if operation in visited:continue
                if operation.name=='tensor.empty':
                    _require(len(operation.results)==1);empty(operation.results[0],visited)
                elif operation.name=='arith.constant':
                    _require(len(operation.results)==1 and not operation.operands and not operation.regions)
                    _require(not isinstance(operation.results[0].type,TensorType))
                    _require(isinstance(_attr(operation,'value'),IntegerAttr))
                else:
                    raise _NoMatch()
            certificates.append({dx:'grad_input',dw:'grad_weight',db:'grad_bias'})
        except _NoMatch:
            continue
    _require(len(certificates)==1)
    return certificates[0]
