"""Share a live floating pointwise result with its integer consumer.

Both original scalar programs and output maps survive in one tensor generic.
The floating result remains available to later skip/state consumers. Only pure
static parallel work can move across constants and private tensor constructors;
calls, reductions and earlier observers keep their original boundaries.
"""

from __future__ import annotations

import hashlib

from xdsl.dialects import arith, builtin, tensor
from xdsl.dialects.linalg.ops import GenericOp, YieldOp
from xdsl.ir import Block, Operation, Region
from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
from xdsl.utils.exceptions import ParseError

from .automatic_rounding import _container
from .quant_round_check import pure_scalar_body


def _pointwise(op):
    if (
        not isinstance(op, GenericOp)
        or len(op.results) != 1
        or len(op.outputs) != 1
        or not isinstance(op.outputs[0].owner, tensor.EmptyOp)
        or op.outputs[0].owner.attributes
        or op.body.block.args[-1].uses
        or not _container(op)
        or not pure_scalar_body(op.body.block, bounded_intrinsics=True)
        or any(iterator.data.value != "parallel" for iterator in op.iterator_types)
    ):
        return False
    rank = len(op.iterator_types)
    for value, attribute in zip(op.operands, op.indexing_maps, strict=True):
        typ, mapping = value.type, attribute.data
        if (
            not isinstance(typ, builtin.TensorType)
            or not isinstance(typ.encoding, builtin.NoneAttr)
            or any(dim <= 0 for dim in typ.get_shape())
            or mapping.num_symbols
            or mapping.num_dims != rank
            or len(mapping.results) != len(typ.get_shape())
        ):
            return False
        positions = []
        for extent, expr in zip(typ.get_shape(), mapping.results, strict=True):
            if isinstance(expr, AffineDimExpr):
                positions.append(expr.position)
            elif not isinstance(expr, AffineConstantExpr) or expr.value != 0 or extent != 1:
                return False
        if len(positions) != len(set(positions)):
            return False
    # No index-dependent scalar operations or arbitrary captured SSA values.
    for scalar in op.body.block.ops:
        for operand in scalar.operands:
            owner = operand.owner
            if operand not in op.body.block.args and getattr(owner, "parent", None) is not op.body.block:
                if not isinstance(owner, arith.ConstantOp):
                    return False
    return True


def _transparent(op):
    if isinstance(op, arith.ConstantOp):
        return True
    if isinstance(op, (tensor.EmptyOp, tensor.SplatOp)):
        return not op.attributes or all(key.startswith("prov.") for key in op.attributes)
    return _pointwise(op) and not op.results[0].type.get_shape()


def fuse_live_pointwise(module, *, max_scalar_operations=512, max_fusions=4096):
    """Mutate private IR, returning observations of eliminated tensor reads.

    Identity bridge maps keep this narrow: producer and consumer enumerate the
    same domain exactly once. Other inputs may broadcast and the integer output
    may use a different permutation. No arithmetic is cloned or reassociated.
    """
    if any(type(n) is not int or n < 0 for n in (max_scalar_operations, max_fusions)):
        raise ValueError("pointwise fusion requires nonnegative scalar and fusion budgets")
    module.verify()
    rows = []
    for consumer in tuple(module.walk()):
        if len(rows) >= max_fusions:
            break
        if not _pointwise(consumer) or not isinstance(consumer.results[0].type.get_element_type(), builtin.IntegerType):
            continue
        rank = len(consumer.iterator_types)
        output_map = consumer.indexing_maps.data[-1].data
        if not rank or sorted(
            expr.position if isinstance(expr, AffineDimExpr) else -1 for expr in output_map.results
        ) != list(range(rank)):
            continue
        identity = AffineMap.identity(rank)
        for index, source in enumerate(consumer.inputs):
            producer = source.owner
            if (
                not isinstance(producer, Operation)
                or not _pointwise(producer)
                or not isinstance(source.type.get_element_type(), builtin.AnyFloat)
                or producer.parent is not consumer.parent
                or len(producer.iterator_types) != rank
                or producer.indexing_maps.data[-1].data != identity
                or consumer.indexing_maps.data[index].data != identity
                or sum(value is source for value in consumer.inputs) != 1
                or len(tuple(source.uses)) < 2
            ):
                continue
            block = consumer.parent
            operations = tuple(block.ops)
            start, end = operations.index(producer), operations.index(consumer)
            if start >= end or any(not _transparent(op) for op in operations[start + 1:end]):
                continue
            if any(use.operation is not consumer and (
                use.operation.parent is not block or operations.index(use.operation) <= end
            ) for use in source.uses):
                continue
            if len(tuple(producer.body.block.ops)) + len(tuple(consumer.body.block.ops)) > max_scalar_operations:
                continue
            inputs = [*producer.inputs, *(value for i, value in enumerate(consumer.inputs) if i != index)]
            outputs = [producer.outputs[0], consumer.outputs[0]]
            body = Block(arg_types=[value.type.get_element_type() for value in (*inputs, *outputs)])
            values = dict(zip(producer.body.block.args[:-1], body.args[:len(producer.inputs)], strict=True))
            values[producer.body.block.args[-1]] = body.args[-2]
            for scalar in tuple(producer.body.block.ops)[:-1]:
                body.add_op(scalar.clone(value_mapper=values))
            floating = values.get(producer.body.block.last_op.operands[0], producer.body.block.last_op.operands[0])
            next_input = len(producer.inputs)
            for i, argument in enumerate(consumer.body.block.args[:-1]):
                if i == index:
                    values[argument] = floating
                else:
                    values[argument] = body.args[next_input]
                    next_input += 1
            values[consumer.body.block.args[-1]] = body.args[-1]
            for scalar in tuple(consumer.body.block.ops)[:-1]:
                body.add_op(scalar.clone(value_mapper=values))
            quantized = values.get(consumer.body.block.last_op.operands[0], consumer.body.block.last_op.operands[0])
            body.add_op(YieldOp(floating, quantized))
            maps = [*producer.indexing_maps.data[:-1], *(
                mapping for i, mapping in enumerate(consumer.indexing_maps.data[:-1]) if i != index
            ), producer.indexing_maps.data[-1], consumer.indexing_maps.data[-1]]
            fused = GenericOp(inputs, outputs, Region(body), maps, consumer.iterator_types,
                              result_types=[value.type for value in outputs])
            fused.attributes.update(consumer.attributes)
            fused.attributes["prov.fused_producer"] = builtin.DictionaryAttr(dict(producer.attributes))
            block.insert_op_before(fused, consumer)
            source.replace_all_uses_with(fused.results[0])
            consumer.results[0].replace_all_uses_with(fused.results[1])
            block.erase_op(consumer)
            block.erase_op(producer)
            elements = 1
            for dim in source.type.get_shape():
                elements *= dim
            rows.append({"shape": list(source.type.get_shape()), "shared_elements": elements,
                         "eliminated_read_bytes": elements * source.type.get_element_type().bitwidth // 8})
            break
    module.verify()
    return rows


def prepare_host_pointwise_fusion(source, *, enabled=False, serial=True):
    """Ordinary opt-in host fusion, before upstream bufferization."""
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.xdsl_dialects._common import text

    if type(enabled) is not bool or type(serial) is not bool:
        raise ValueError("pointwise fusion requires explicit Boolean selections")
    digest = hashlib.sha256(source.encode()).hexdigest()
    report = {"schema": "merlin.host_pointwise_fusion.v1", "rewrites": 0,
              "source_sha256": digest, "output_sha256": digest, "regions": []}
    if not enabled or not serial:
        return source, report
    try:
        module = parse_mlir_text(source)
    except ParseError:
        return source, report
    rows = fuse_live_pointwise(module)
    output = text(module, generic=True) if rows else source
    report.update(rewrites=len(rows), regions=rows, output_sha256=hashlib.sha256(output.encode()).hexdigest())
    return output, report
