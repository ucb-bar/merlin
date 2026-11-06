"""Structural recognition of byte contractions and binary32 narrowing readouts.

Recognizers inspect typed scalar SSA and affine maps; source names and provenance
are never used to select a numerical implementation.
"""
from __future__ import annotations

import math
from typing import ClassVar
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IntegerAttr,
    TensorType,
)
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr


def _attr(op, name):
    return op.properties.get(name, op.attributes.get(name))


def _type(value):
    return str(value.type)


def _spec(value):
    t = value.type
    if not isinstance(t, TensorType):
        return None
    return tuple(t.get_shape()), str(t.element_type)


def _scalar_constant(value):
    op = value.owner
    if getattr(op, 'name', None) != 'arith.constant':
        return None
    a = _attr(op, 'value')
    if isinstance(a, (IntegerAttr, FloatAttr)):
        return a.value.data
    return None


def _dense_constant(value):
    op = value.owner
    if getattr(op, 'name', None) != 'arith.constant':
        return None
    a = _attr(op, 'value')
    if not isinstance(a, DenseIntOrFPElementsAttr):
        return None
    return list(a.get_values())


def _is_dim(e, n):
    return isinstance(e, AffineDimExpr) and e.position == n


def _map_is(m, *positions):
    return (
        len(m.results) == len(positions)
        and all(_is_dim(e, n) for e, n in zip(m.results, positions))
    )


def _generic_parts(op):
    if op.name != 'linalg.generic' or len(op.results) != 1:
        return None
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return None
    inputs = list(op.inputs)
    outputs = list(op.outputs)
    if len(outputs) != 1:
        return None
    maps_a = _attr(op, 'indexing_maps')
    iters_a = _attr(op, 'iterator_types')
    if maps_a is None or iters_a is None:
        return None
    maps = [a.data for a in maps_a]
    iters = [str(a) for a in iters_a]
    block = op.regions[0].block
    body = list(block.ops)
    if not body or body[-1].name != 'linalg.yield' or len(body[-1].operands) != 1:
        return None
    if len(block.args) != len(inputs) + len(outputs):
        return None
    return inputs, outputs, maps, iters, block, body


def _named_matmul_parts(op):
    """Read only the canonical signed named-matmul contract.

    The selected MLIR MatmulOp definition defaults cast to cast_signed and
    promotes both multiply operands to the accumulator type. The xDSL named-op
    parser's hidden region omits those promotions; its exact generated body is
    recognized below without treating arbitrary scalar bodies as equivalent.
    """
    from xdsl.dialects.linalg.ops import MatmulOp
    from xdsl.dialects.builtin import UnregisteredAttr
    if not isinstance(op,MatmulOp) or len(op.results)!=1:
        return None
    for key in ('indexing_maps','cast','iterator_types'):
        if key in op.attributes and key in op.properties:
            return None
    cast=_attr(op,'cast')
    if cast is not None:
        if not isinstance(cast,UnregisteredAttr) or cast.attr_name.data!='linalg.type_fn' or cast.value.data!='cast_signed':
            return None
    ins=list(op.inputs);outs=list(op.outputs)
    if len(ins)!=2 or len(outs)!=1 or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:
        return None
    maps_a=_attr(op,'indexing_maps')
    if maps_a is None:
        return None
    maps=[a.data for a in maps_a]
    if len(maps)!=3 or any(m.num_dims!=3 or m.num_symbols!=0 for m in maps):
        return None
    if not (_map_is(maps[0],0,2) and _map_is(maps[1],2,1) and _map_is(maps[2],0,1)):
        return None
    iters_a=_attr(op,'iterator_types')
    iters=[str(a) for a in iters_a] if iters_a is not None else ['parallel','parallel','reduction']
    if len(iters)!=3 or 'parallel' not in iters[0] or 'parallel' not in iters[1] or 'reduction' not in iters[2]:
        return None
    block=op.regions[0].block;body=list(block.ops)
    if len(block.args)!=3 or not body or body[-1].name!='linalg.yield' or len(body[-1].operands)!=1:
        return None
    return ins,outs,maps,iters,block,body


def _default_named_i8_body(block,body):
    """The exact public xDSL MatmulOp.get_hidden_region representation."""
    if len(body)!=3 or len(block.args)!=3:
        return False
    mul,add,yield_op=body
    lhs,rhs,seed=block.args
    return (
        mul.name=='arith.muli' and len(mul.results)==1
        and tuple(mul.operands)==(lhs,rhs) and _type(mul.results[0])=='i8'
        and add.name=='arith.addi' and len(add.results)==1
        and tuple(add.operands)==(mul.results[0],seed) and _type(add.results[0])=='i8'
        and yield_op.name=='linalg.yield'
        and tuple(yield_op.operands)==(add.results[0],)
    )


def recognize_mac(op):
    """Return rank-2 i8 x i8 -> i32 reduction operands and its seed, or None.

    Only the standard matmul iteration/indexing correspondence is accepted. The
    seed is returned unchanged, including a runtime tensor or nonzero literal.
    """
    named=op.name=='linalg.matmul'
    parts = _named_matmul_parts(op) if named else _generic_parts(op)
    if parts is None:
        return None
    ins, outs, maps, iters, block, body = parts
    if len(ins) != 2 or len(maps) != 3:
        return None
    if len(iters) != 3 or not all('parallel' in x for x in iters[:2]) or 'reduction' not in iters[2]:
        return None
    sa, sb, sd = _spec(ins[0]), _spec(ins[1]), _spec(op.results[0])
    if sa is None or sb is None or sd is None:
        return None
    if sa[1] != 'i8' or sb[1] != 'i8' or sd[1] != 'i32':
        return None
    if len(sa[0]) != 2 or len(sb[0]) != 2 or len(sd[0]) != 2:
        return None
    if not _map_is(maps[0], 0, 2) or not _map_is(maps[2], 0, 1):
        return None
    if _map_is(maps[1], 2, 1):
        transpose_rhs = False
        k, n = sb[0]
    elif _map_is(maps[1], 1, 2):
        transpose_rhs = True
        n, k = sb[0]
    else:
        return None
    if sa[0][1] != k or sd[0] != (sa[0][0], n) or _spec(outs[0]) != sd:
        return None
    a, b, dest = block.args
    if _type(a) != 'i8' or _type(b) != 'i8' or _type(dest) != 'i32':
        return None
    if named:
        if transpose_rhs or any(type(d) is not int or d<=0 for shape in (sa[0],sb[0],sd[0]) for d in shape):
            return None
        if _default_named_i8_body(block,body):
            return {'lhs':ins[0],'rhs':ins[1],'seed':outs[0],'transpose_rhs':False}
    root = body[-1].operands[0].owner
    if getattr(root, 'name', None) != 'arith.addi' or dest not in root.operands:
        return None
    term = next((x for x in root.operands if x is not dest), None)
    if term is None:
        return None
    mul = term.owner
    if getattr(mul, 'name', None) != 'arith.muli' or _type(term) != 'i32':
        return None
    used = {root, mul}

    def widened(value, arg):
        cast = value.owner
        if getattr(cast, 'name', None) != 'arith.extsi' or len(cast.operands) != 1:
            return False
        if cast.operands[0] is not arg or _type(value) != 'i32':
            return False
        used.add(cast)
        return True

    x, y = mul.operands
    if not ((widened(x, a) and widened(y, b)) or (widened(x, b) and widened(y, a))):
        return None
    if any(o not in used and o.name != 'arith.constant' for o in body[:-1]):
        return None
    return {'lhs': ins[0], 'rhs': ins[1], 'seed': outs[0], 'transpose_rhs': transpose_rhs}


def _signed_cast_selector(op):
    cast=_attr(op,'cast')
    if cast is None:return True
    from xdsl.dialects.builtin import UnregisteredAttr
    return (isinstance(cast,UnregisteredAttr) and cast.attr_name.data=='linalg.type_fn'
            and cast.is_type.data==0 and cast.value.data=='cast_signed')


def _signed_i32_mac_body(block,body):
    if len(block.args)!=3 or not body or body[-1].name!='linalg.yield' or len(body[-1].operands)!=1:
        return False
    lhs,rhs,seed=block.args
    if tuple(_type(v) for v in block.args)!=('i8','i8','i32'):
        return False
    root=body[-1].operands[0].owner
    if getattr(root,'name',None)!='arith.addi' or len(root.operands)!=2 or seed not in root.operands or _type(body[-1].operands[0])!='i32':
        return False
    term=next((v for v in root.operands if v is not seed),None)
    if term is None or _type(term)!='i32':return False
    product=term.owner
    if getattr(product,'name',None)!='arith.muli' or len(product.operands)!=2:
        return False
    used={root,product}
    def widened(value,arg):
        cast=value.owner
        if getattr(cast,'name',None)!='arith.extsi' or len(cast.operands)!=1 or cast.operands[0] is not arg or _type(value)!='i32':
            return False
        used.add(cast);return True
    a,b=product.operands
    if not ((widened(a,lhs) and widened(b,rhs)) or (widened(a,rhs) and widened(b,lhs))):
        return False
    return all(o in used or o.name=='arith.constant' for o in body[:-1])


_CAPTURED_BATCH_MATMUL=None


def register_batched_matmul(context):
    """Register the standard named batch source carrier missing in this xDSL."""
    global _CAPTURED_BATCH_MATMUL
    if _CAPTURED_BATCH_MATMUL is None:
        from xdsl.dialects.linalg.abstract_ops import NamedOperation
        from xdsl.dialects.linalg.ops import MatmulOp
        from xdsl.dialects.linalg.attrs import IteratorTypeAttr
        from xdsl.dialects.builtin import ArrayAttr,AffineMapAttr
        from xdsl.ir.affine import AffineMap
        from xdsl.irdl import irdl_op_definition,prop_def
        from xdsl.builder import Builder
        from xdsl.dialects.arith import ExtSIOp,MuliOp,AddiOp
        from xdsl.dialects.linalg.ops import YieldOp
        @irdl_op_definition
        class CapturedBatchMatmulOp(NamedOperation):
            name='linalg.batch_matmul'
            PRINT_ATTRS_IN_FRONT:ClassVar[bool]=True
            indexing_maps=prop_def(ArrayAttr[AffineMapAttr],default_value=ArrayAttr([
                AffineMapAttr(AffineMap.from_callable(lambda b,m,n,k:(b,m,k))),
                AffineMapAttr(AffineMap.from_callable(lambda b,m,n,k:(b,k,n))),
                AffineMapAttr(AffineMap.from_callable(lambda b,m,n,k:(b,m,n))),
            ]))
            @classmethod
            def get_hidden_region(cls,inputs,outputs):
                types=cls.body_arg_types((*inputs,*outputs))
                if tuple(map(str,types))!=('i8','i8','i32'):
                    return MatmulOp.get_hidden_region(inputs,outputs)
                @Builder.implicit_region(types)
                def region(args):
                    a=ExtSIOp(args[0],types[2]);b=ExtSIOp(args[1],types[2])
                    product=MuliOp(a.result,b.result)
                    value=AddiOp(product.result,args[2])
                    YieldOp(value.result)
                return region
            def get_default_indexing_maps(self):
                return tuple(a.data for a in self.indexing_maps)
            def get_indexing_maps(self):
                return self.indexing_maps
            def get_iterator_types(self):
                return ArrayAttr([IteratorTypeAttr.parallel(),IteratorTypeAttr.parallel(),
                                  IteratorTypeAttr.parallel(),IteratorTypeAttr.reduction()])
        _CAPTURED_BATCH_MATMUL=CapturedBatchMatmulOp
    context.load_op(_CAPTURED_BATCH_MATMUL)


def recognize_batched_mac(op):
    """Prove identical static batch axes and an exact signed byte MAC body.

    Returns the original lhs/rhs/seed SSA values and transpose_rhs=False. This
    proof never asserts that a nonzero seed has an admitted caller boundary.
    """
    from xdsl.dialects.builtin import NoneAttr
    named=op.name=='linalg.batch_matmul'
    if named:
        if _CAPTURED_BATCH_MATMUL is None or not isinstance(op,_CAPTURED_BATCH_MATMUL):return None
        if len(op.results)!=1 or len(op.regions)!=1 or len(op.regions[0].blocks)!=1:return None
        if not _signed_cast_selector(op):return None
        if any(k in op.attributes and k in op.properties for k in ('cast','indexing_maps','iterator_types')):return None
        ins=list(op.inputs);outs=list(op.outputs);block=op.regions[0].block;body=list(block.ops)
    else:
        parts=_generic_parts(op)
        if parts is None:return None
        ins,outs,_maps,_iters,block,body=parts
    if len(ins)!=2 or len(outs)!=1 or len(op.results)!=1:return None
    lhs,rhs,out=_spec(ins[0]),_spec(ins[1]),_spec(op.results[0])
    if lhs is None or rhs is None or out is None:return None
    if (lhs[1],rhs[1],out[1])!=('i8','i8','i32') or _spec(outs[0])!=out:return None
    if any(not isinstance(value.type.encoding,NoneAttr) for value in [*ins,*outs,*op.results]):return None
    rank=len(lhs[0])
    if rank<3 or len(rhs[0])!=rank or len(out[0])!=rank or (named and rank!=3):return None
    if any(type(d) is not int or d<=0 for shape in (lhs[0],rhs[0],out[0]) for d in shape):return None
    batch=rank-2
    if lhs[0][:-2]!=rhs[0][:-2] or lhs[0][:-2]!=out[0][:-2]:return None
    if lhs[0][-1]!=rhs[0][-2] or out[0][-2:]!=(lhs[0][-2],rhs[0][-1]):return None
    maps_a=_attr(op,'indexing_maps')
    if maps_a is None:return None
    maps=[a.data for a in maps_a]
    if len(maps)!=3 or any(m.num_dims!=batch+3 or m.num_symbols!=0 for m in maps):return None
    prefix=list(range(batch));m,n,k=batch,batch+1,batch+2
    if not (_map_is(maps[0],*prefix,m,k) and _map_is(maps[1],*prefix,k,n) and _map_is(maps[2],*prefix,m,n)):return None
    iters_a=_attr(op,'iterator_types')
    if iters_a is None:
        if not named:return None
        kinds=['parallel']*(batch+2)+['reduction']
    else:
        kinds=[getattr(a,'data',None) for a in iters_a]
    if kinds!=['parallel']*(batch+2)+['reduction']:return None
    if not _signed_i32_mac_body(block,body):return None
    return {'lhs':ins[0],'rhs':ins[1],'seed':outs[0],'transpose_rhs':False}


def recognize_readout(op):
    """Recognize f32-convert/product/RNE/saturation followed by an i8 result.

    Returns src, scales (literal per-column f32 values) or runtime_scale (a
    one-element f32 tensor), and relu. A runtime scale remains subject to the
    selected caller domain requiring a finite, nonnegative multiplier.
    """
    parts = _generic_parts(op)
    if parts is None:
        return None
    ins, outs, maps, iters, block, body = parts
    if len(ins) != 2 or len(maps) != 3 or len(iters) != 2 or any('parallel' not in x for x in iters):
        return None
    src, gain = ins
    ss, sg, sd = _spec(src), _spec(gain), _spec(op.results[0])
    if ss is None or sg is None or sd is None:
        return None
    if ss[1] not in ('i8', 'i32') or sg[1] != 'f32' or sd[1] != 'i8' or ss[0] != sd[0] or len(ss[0]) != 2:
        return None
    if _spec(outs[0]) != sd or not _map_is(maps[0], 0, 1) or not _map_is(maps[2], 0, 1):
        return None
    n = ss[0][1]
    column_gain = sg[0] == (n,) and _map_is(maps[1], 1)
    scalar_gain = sg[0] == (1,) and len(maps[1].results) == 1 and isinstance(maps[1].results[0], AffineConstantExpr) and maps[1].results[0].value == 0
    if not (column_gain or scalar_gain):
        return None
    x, g, old = block.args
    if (_type(x), _type(g), _type(old)) != (ss[1], 'f32', 'i8'):
        return None
    used = set()

    def unary(value, name, dtype):
        owner = value.owner
        if getattr(owner, 'name', None) != name or len(owner.operands) != 1 or _type(value) != dtype:
            return None
        used.add(owner)
        return owner.operands[0]

    def bound(value, names, constant):
        owner = value.owner
        if getattr(owner, 'name', None) not in names or len(owner.operands) != 2 or _type(value) != 'f32':
            return None
        a, b = owner.operands
        if _scalar_constant(a) == constant:
            used.add(owner)
            return b
        if _scalar_constant(b) == constant:
            used.add(owner)
            return a
        return None

    v = body[-1].operands[0]
    v = unary(v, 'arith.fptosi', 'i8')
    if v is None:
        return None
    v = bound(v, ('arith.minimumf', 'arith.minnumf'), 127.0)
    if v is None:
        return None
    v = bound(v, ('arith.maximumf', 'arith.maxnumf'), -128.0)
    if v is None:
        return None
    v = unary(v, 'math.roundeven', 'f32')
    if v is None:
        return None
    mul = v.owner
    if getattr(mul, 'name', None) != 'arith.mulf' or len(mul.operands) != 2 or _type(v) != 'f32' or g not in mul.operands:
        return None
    used.add(mul)
    v = next((a for a in mul.operands if a is not g), None)
    if v is None:
        return None
    activated = bound(v, ('arith.maximumf', 'arith.maxnumf'), 0.0)
    relu = activated is not None
    if relu:
        v = activated
    v = unary(v, 'arith.sitofp', 'f32')
    if v is not x:
        return None
    if any(o not in used and o.name != 'arith.constant' for o in body[:-1]):
        return None
    scales = _dense_constant(gain)
    if scales is not None:
        if len(scales) != sg[0][0] or any(not math.isfinite(v) for v in scales):
            return None
        if relu and any(v < 0 for v in scales):
            return None
        return {'src': src, 'scales': [float(v) for v in scales], 'relu': relu}
    if not scalar_gain:
        return None
    return {'src': src, 'runtime_scale': gain, 'relu': relu}


def _parallel_binary(op, input_dtypes, result_dtype):
    parts = _generic_parts(op)
    if parts is None:
        return None
    ins, outs, maps, iters, block, body = parts
    if len(ins) != 2 or len(maps) != 3 or len(iters) != 2 or any('parallel' not in x for x in iters):
        return None
    specs = [_spec(v) for v in (*ins, op.results[0])]
    if any(s is None for s in specs):
        return None
    a, b, dst = specs
    if (a[1], b[1]) != input_dtypes or dst[1] != result_dtype:
        return None
    if len(dst[0]) != 2 or a[0] != dst[0] or b[0] != dst[0] or _spec(outs[0]) != dst:
        return None
    if any(not _map_is(m, 0, 1) for m in maps):
        return None
    return parts


def recognize_operand_sum(op):
    """Return exact typed byte-sum operands, f32 gains and rounding position.

    Integer widening/add has unit gains. A floating scalar body may round once
    after adding scaled operands, or round each scaled operand before its exact
    integer sum. The latter returns per_operand_rounding=True; the scale values
    alone never imply that the two numerical models are interchangeable.
    """
    parts = _parallel_binary(op, ('i8', 'i8'), 'i32')
    if parts is None:
        return None
    ins, outs, maps, iters, block, body = parts
    left, right, dest = block.args
    if (_type(left), _type(right), _type(dest)) != ('i8', 'i8', 'i32'):
        return None
    used = set()

    def unary(value, name, dtype):
        o = value.owner
        if getattr(o, 'name', None) != name or len(o.operands) != 1 or _type(value) != dtype:
            return None
        used.add(o)
        return o.operands[0]

    def widened(value, arg):
        return unary(value, 'arith.extsi', 'i32') is arg

    def product(value, arg):
        mul = value.owner
        if getattr(mul, 'name', None) != 'arith.mulf' or len(mul.operands) != 2 or _type(value) != 'f32':
            return None
        a, b = mul.operands
        ca, cb = _scalar_constant(a), _scalar_constant(b)
        if ca is not None:
            source, scale = b, ca
        elif cb is not None:
            source, scale = a, cb
        else:
            return None
        if unary(source, 'arith.sitofp', 'f32') is not arg:
            return None
        if not math.isfinite(scale):
            return None
        used.add(mul)
        return float(scale)

    result = body[-1].operands[0]
    root = result.owner
    per_operand = False
    scales = None
    if getattr(root, 'name', None) == 'arith.addi' and _type(result) == 'i32':
        used.add(root)
        a, b = root.operands
        if (widened(a, left) and widened(b, right)) or (widened(a, right) and widened(b, left)):
            scales = (1.0, 1.0)
        else:
            # Both float products must round separately before converting to i32.
            def rounded_product(value, arg):
                v = unary(value, 'arith.fptosi', 'i32')
                if v is None:
                    return None
                v = unary(v, 'math.roundeven', 'f32')
                return None if v is None else product(v, arg)
            la, rb = rounded_product(a, left), rounded_product(b, right)
            if la is None or rb is None:
                la, rb = rounded_product(b, left), rounded_product(a, right)
            if la is not None and rb is not None:
                scales = (la, rb)
                per_operand = True
    elif getattr(root, 'name', None) == 'arith.fptosi':
        v = unary(result, 'arith.fptosi', 'i32')
        if v is None:
            return None
        v = unary(v, 'math.roundeven', 'f32')
        if v is None:
            return None
        add = v.owner
        if getattr(add, 'name', None) != 'arith.addf' or len(add.operands) != 2 or _type(v) != 'f32':
            return None
        used.add(add)
        a, b = add.operands
        la, rb = product(a, left), product(b, right)
        if la is None or rb is None:
            la, rb = product(b, left), product(a, right)
        if la is not None and rb is not None:
            scales = (la, rb)
    if scales is None:
        return None
    if any(o not in used and o.name != 'arith.constant' for o in body[:-1]):
        return None
    return {'lhs': ins[0], 'rhs': ins[1], 'lhs_scale': scales[0], 'rhs_scale': scales[1], 'per_operand_rounding': per_operand}


def recognize_extension(op):
    """Recognize i32 accumulator plus an exact integer byte multiple or i32 bias.

    The additive integer expression retains wrapping i32 source arithmetic. A
    rounded floating residual is intentionally absent from this exact recognizer.
    """
    parts = _generic_parts(op)
    if parts is None:
        return None
    ins, outs, maps, iters, block, body = parts
    if len(ins) != 2 or len(maps) != 3 or len(iters) != 2 or any('parallel' not in x for x in iters):
        return None
    ss, sr, sd = _spec(ins[0]), _spec(ins[1]), _spec(op.results[0])
    if ss is None or sr is None or sd is None:
        return None
    if ss[1] != 'i32' or sd[1] != 'i32' or sr[1] not in ('i8', 'i32'):
        return None
    if len(ss[0]) != 2 or ss != sd or _spec(outs[0]) != sd:
        return None
    if not _map_is(maps[0], 0, 1) or not _map_is(maps[2], 0, 1):
        return None
    full = sr[0] == ss[0] and _map_is(maps[1], 0, 1)
    column = sr[0] == (ss[0][1],) and _map_is(maps[1], 1)
    if not (full or column):
        return None
    acc, residual, old = block.args
    if (_type(acc), _type(residual), _type(old)) != ('i32', sr[1], 'i32'):
        return None
    root = body[-1].operands[0].owner
    if getattr(root, 'name', None) != 'arith.addi' or acc not in root.operands:
        return None
    term = next((v for v in root.operands if v is not acc), None)
    if term is None:
        return None
    used = {root}
    if sr[1] == 'i32':
        if term is not residual or any(o not in used and o.name != 'arith.constant' for o in body[:-1]):
            return None
        return {'accumulator': ins[0], 'bias': ins[1]}
    coefficient = 1
    mul = term.owner
    if getattr(mul, 'name', None) == 'arith.muli':
        a, b = mul.operands
        ca, cb = _scalar_constant(a), _scalar_constant(b)
        if isinstance(ca, int):
            term, coefficient = b, ca
        elif isinstance(cb, int):
            term, coefficient = a, cb
        else:
            return None
        used.add(mul)
    cast = term.owner
    if getattr(cast, 'name', None) != 'arith.extsi' or len(cast.operands) != 1 or cast.operands[0] is not residual or _type(term) != 'i32':
        return None
    used.add(cast)
    if any(o not in used and o.name != 'arith.constant' for o in body[:-1]):
        return None
    return {'accumulator': ins[0], 'residual': ins[1], 'coefficient': coefficient}


def _pool_linear_terms(expr):
    """Normalize a bounded affine row expression over floor/mod index atoms."""
    from xdsl.ir.affine import AffineBinaryOpExpr, AffineBinaryOpKind
    budget = [256]

    def normalize(e, depth):
        budget[0] -= 1
        if budget[0] < 0 or depth > 64:
            raise ValueError('pool indexing expression exceeds structural bound')
        if isinstance(e, AffineConstantExpr):
            return {('constant',): e.value} if e.value else {}
        if isinstance(e, AffineDimExpr):
            return {('dim', e.position): 1}
        if not isinstance(e, AffineBinaryOpExpr):
            raise ValueError('unsupported pool indexing expression')
        if e.kind == AffineBinaryOpKind.Add:
            a, b = normalize(e.lhs, depth+1), normalize(e.rhs, depth+1)
            for k, v in b.items():
                a[k] = a.get(k, 0) + v
            return {k:v for k,v in a.items() if v}
        if e.kind == AffineBinaryOpKind.Mul:
            if isinstance(e.lhs, AffineConstantExpr):
                scale, child = e.lhs.value, e.rhs
            elif isinstance(e.rhs, AffineConstantExpr):
                scale, child = e.rhs.value, e.lhs
            else:
                raise ValueError('nonlinear pool indexing expression')
            return {k:v*scale for k,v in normalize(child, depth+1).items() if v*scale}
        if e.kind in (AffineBinaryOpKind.FloorDiv, AffineBinaryOpKind.Mod):
            if not isinstance(e.lhs, AffineDimExpr) or not isinstance(e.rhs, AffineConstantExpr) or e.rhs.value <= 0:
                raise ValueError('pool floor/mod must have one loop index and a positive literal divisor')
            kind = 'floor' if e.kind == AffineBinaryOpKind.FloorDiv else 'mod'
            return {(kind, e.lhs.position, e.rhs.value):1}
        raise ValueError('unsupported pool indexing operator')

    try:
        return normalize(expr, 0)
    except ValueError:
        return None


def _initial_scalar(value):
    for _ in range(4):
        owner = value.owner
        name = getattr(owner, 'name', None)
        if name in ('linalg.fill', 'tensor.splat') and owner.operands:
            value = owner.operands[0]
            continue
        literal = _scalar_constant(value)
        if literal is not None:
            return literal
        if name == 'arith.constant':
            a = _attr(owner, 'value')
            if isinstance(a, DenseIntOrFPElementsAttr) and a.is_splat():
                return next(iter(a.get_values()))
        return None
    return None


def recognize_pool(op):
    """Recover unpadded byte max-pool geometry from its complete affine source map.

    The flattened source rows are a single H-by-W plane and the reduction domain
    is a PH-by-PW window. Geometry is derived from four floor/mod terms, with no
    enumeration of source or output tensor elements.
    """
    parts = _generic_parts(op)
    if parts is None:
        return None
    ins, outs, maps, iters, block, body = parts
    if len(ins) != 2 or len(maps) != 3 or len(iters) != 3 or not all('parallel' in x for x in iters[:2]) or 'reduction' not in iters[2]:
        return None
    ss, sw, sd = _spec(ins[0]), _spec(ins[1]), _spec(op.results[0])
    if ss is None or sw is None or sd is None:
        return None
    if ss[1] != 'i8' or sw[1] != 'i8' or sd[1] != 'i8' or _spec(outs[0]) != sd:
        return None
    if len(ss[0]) != 2 or len(sd[0]) != 2 or len(sw[0]) != 1 or ss[0][1] != sd[0][1]:
        return None
    if any(d <= 0 for d in (*ss[0], *sw[0], *sd[0])):
        return None
    if len(maps[0].results) != 2 or not _is_dim(maps[0].results[1],1) or not _map_is(maps[1],2) or not _map_is(maps[2],0,1):
        return None
    window_owner = ins[1].owner
    if getattr(window_owner, 'name', None) != 'arith.constant' or not isinstance(_attr(window_owner,'value'),DenseIntOrFPElementsAttr):
        return None
    source, unused_window, dest = block.args
    if (_type(source), _type(unused_window), _type(dest)) != ('i8','i8','i8'):
        return None
    root = body[-1].operands[0].owner
    if getattr(root,'name',None) != 'arith.maxsi' or len(root.operands)!=2 or set(root.operands)!={source,dest}:
        return None
    if any(o is not root and o.name != 'arith.constant' for o in body[:-1]):
        return None
    initial = _initial_scalar(outs[0])
    if initial != -128:
        return None
    terms = _pool_linear_terms(maps[0].results[0])
    if terms is None or len(terms)!=4:
        return None
    floor_out = [(k,v) for k,v in terms.items() if k[0]=='floor' and k[1]==0]
    floor_tap = [(k,v) for k,v in terms.items() if k[0]=='floor' and k[1]==2]
    mod_out = [(k,v) for k,v in terms.items() if k[0]=='mod' and k[1]==0]
    mod_tap = [(k,v) for k,v in terms.items() if k[0]=='mod' and k[1]==2]
    if any(len(rows)!=1 for rows in (floor_out,floor_tap,mod_out,mod_tap)):
        return None
    ko, row_jump = floor_out[0]
    kt, width = floor_tap[0]
    km, stride_w = mod_out[0]
    kw, tap_jump = mod_tap[0]
    out_w, pool_w = ko[2], kt[2]
    if out_w != km[2] or pool_w != kw[2] or tap_jump!=1 or width<=0 or stride_w<=0 or row_jump<=0:
        return None
    if row_jump % width or ss[0][0] % width or sw[0][0] % pool_w:
        return None
    stride_h = row_jump//width
    height = ss[0][0]//width
    pool_h = sw[0][0]//pool_w
    if pool_h<=0 or pool_h>height or pool_w>width:
        return None
    derived_out_w=(width-pool_w)//stride_w+1
    out_h=(height-pool_h)//stride_h+1
    if out_w!=derived_out_w or sd[0][0] != out_h*out_w:
        return None
    # Source-domain proof for every retained output and tap index.
    if (out_h-1)*stride_h+pool_h>height or (out_w-1)*stride_w+pool_w>width:
        return None
    return {'src':ins[0], 'pool_in_dims':[height,width], 'pool_size':[pool_h,pool_w], 'pool_stride':[stride_h,stride_w], 'pool_padding':[0,0,0,0], 'pool_pad_value':-128, 'initial':-128}
