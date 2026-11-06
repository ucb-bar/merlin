"""Prove rectangular storage reads from typed parallel linalg regions."""
from math import prod

from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.ir.affine import (
    AffineBinaryOpExpr, AffineBinaryOpKind, AffineConstantExpr, AffineDimExpr,
)
from frontend_broadcast import _attr, _plain_attributes, _spec


class _NotMovement(ValueError):
    pass


def _require(condition):
    if not condition:
        raise _NotMovement()


def _combine(lhs, rhs, multiply=False):
    if not multiply:
        return [a+b for a,b in zip(lhs,rhs)]
    _require(not any(lhs[:-1]) or not any(rhs[:-1]))
    return ([a*rhs[-1] for a in lhs] if not any(rhs[:-1])
            else [lhs[-1]*b for b in rhs])


def _map_expression(expr, rank):
    if isinstance(expr, AffineDimExpr):
        _require(0 <= expr.position < rank)
        return [int(i == expr.position) for i in range(rank)]+[0]
    if isinstance(expr, AffineConstantExpr):
        return [0]*rank+[expr.value]
    _require(isinstance(expr, AffineBinaryOpExpr)
             and expr.kind in (AffineBinaryOpKind.Add, AffineBinaryOpKind.Mul))
    return _combine(_map_expression(expr.lhs,rank), _map_expression(expr.rhs,rank),
                    expr.kind == AffineBinaryOpKind.Mul)


def _index_expression(value, block, rank, visited):
    _require(isinstance(value.type, IndexType))
    owner=value.owner
    _require(getattr(owner,'parent',None) is block)
    visited.add(owner)
    name=getattr(owner,'name',None)
    if name == 'linalg.index':
        dim=_attr(owner,'dim')
        _require(isinstance(dim,IntegerAttr) and 0 <= dim.value.data < rank)
        return [int(i == dim.value.data) for i in range(rank)]+[0]
    if name == 'arith.constant':
        literal=_attr(owner,'value')
        _require(isinstance(literal,IntegerAttr))
        return [0]*rank+[literal.value.data]
    _require(name in ('arith.addi','arith.subi','arith.muli')
             and len(owner.operands) == 2)
    lhs,rhs=(_index_expression(v,block,rank,visited) for v in owner.operands)
    if name == 'arith.subi':
        rhs=[-x for x in rhs]
    return _combine(lhs,rhs,name == 'arith.muli')


def _identity_map(mapping, rank):
    return (mapping.num_dims == rank and mapping.num_symbols == 0
            and tuple(mapping.results) == tuple(AffineDimExpr(i) for i in range(rank)))


def _storage_read(src, result, expressions):
    source_spec,result_spec=_spec(src),_spec(result)
    _require(source_spec is not None and result_spec is not None)
    source_shape,source_type=source_spec
    shape,result_type=result_spec
    _require(source_type == result_type and len(expressions) == len(source_shape))
    strides=[0]*len(shape)
    offset=0
    for axis,(extent,expr) in enumerate(zip(source_shape,expressions)):
        _require(len(expr) == len(shape)+1 and all(type(x) is int for x in expr))
        low=expr[-1]+sum(min(0,w*(n-1)) for w,n in zip(expr,shape))
        high=expr[-1]+sum(max(0,w*(n-1)) for w,n in zip(expr,shape))
        _require(0 <= low <= high < extent)
        step=prod(source_shape[axis+1:])
        offset+=expr[-1]*step
        strides=[s+w*step for s,w in zip(strides,expr)]
    return {'src':src,'result':result,'attributes':{
        'form':'as_strided','strides':strides,'storage_offset':offset}}


def _recognize(op):
    _require(op.name == 'linalg.generic' and len(op.results) == 1
             and len(op.regions) == 1 and len(op.regions[0].blocks) == 1)
    _require(_plain_attributes(op,{'indexing_maps','iterator_types','operandSegmentSizes','doc'}))
    result=op.results[0]
    spec=_spec(result)
    _require(spec is not None)
    rank=len(spec[0])
    inputs,outputs=list(op.inputs),list(op.outputs)
    block=op.regions[0].block
    body=list(block.ops)
    maps=_attr(op,'indexing_maps')
    iters=_attr(op,'iterator_types')
    _require(maps is not None and iters is not None)
    maps=list(maps)
    _require(len(outputs) == 1 and outputs[0].type == result.type
             and len(block.args) == len(inputs)+1 and maps is not None
             and len(maps) == len(inputs)+1 and iters is not None
             and [getattr(i,'data',None) for i in iters] == ['parallel']*rank
             and _identity_map(maps[-1].data,rank))
    _require(body and body[-1].name == 'linalg.yield' and len(body[-1].operands) == 1)
    yielded=body[-1].operands[0]
    if yielded in block.args:
        index=list(block.args).index(yielded)
        _require(index < len(inputs) and len(body) == 1)
        src=inputs[index]
        mapping=maps[index].data
        _require(mapping.num_dims == rank and mapping.num_symbols == 0)
        expressions=[_map_expression(expr,rank) for expr in mapping.results]
    else:
        extract=yielded.owner
        _require(getattr(extract,'name',None) == 'tensor.extract' and extract.parent is block)
        src=extract.operands[0]
        _require(src.owner is not block and _spec(src) is not None)
        visited={extract,body[-1]}
        expressions=[_index_expression(v,block,rank,visited) for v in extract.operands[1:]]
        _require(set(body) == visited)
    _require(yielded.type == result.type.element_type)
    return _storage_read(src,result,expressions)


def recognize_movement(op):
    """Return a bounded affine storage map, or leave other semantics untouched."""
    try:
        return _recognize(op)
    except _NotMovement:
        return None
