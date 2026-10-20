"""Explicit V5 source exclusions from actual registered structured semantics.

Existing contraction hashes remain unchanged. Additional structured operations
must expose checked indexing maps, iterator semantics and their complete scalar
region. Iterator coordinates carry their actual enclosing-loop ownership. This
is private exclusion only, never numerical or executable qualification.
"""

from __future__ import annotations

from xdsl.dialects import builtin
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
from xdsl.dialects.linalg.ops import GenericOp, IndexOp

from merlin.common import mlir_query as Q

from . import operator_mixed_exclusion as M
from . import operator_structured_exclusion as R
from . import operator_typed_exclusion as T

SCHEMA = "merlin.phase0.registered_structured_source_exclusion.v1"
OP_SCHEMA = "merlin.phase0.registered_structured_exclusion_signature.v1"


def _index(op):
    parent = op.parent_op()
    if (type(op) is not IndexOp or type(parent) is not GenericOp or op.operands or op.regions
            or len(op.results) != 1 or not isinstance(op.results[0].type, builtin.IndexType)
            or set(op.properties) != {"dim"} or any(not k.startswith("prov.") for k in op.attributes)):
        raise ValueError("iterator coordinate exclusion has no exact registered scalar owner")
    dimension = op.dim.value.data
    if not 0 <= dimension < len(parent.get_iterator_types()):
        raise ValueError("iterator coordinate exclusion names no actual enclosing loop")


def _body(op):
    if not any(Q.op_name(inner) == "linalg.index" for inner in op.walk() if inner is not op):
        return T._body(op)
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("structured exclusion requires one complete owned scalar region")
    block = op.regions[0].block
    owners = {v: {"argument": i} for i, v in enumerate(block.args)}
    rows = []
    for ordinal, scalar in enumerate(block.ops):
        name = Q.op_name(scalar)
        if name == "linalg.index":
            _index(scalar)
        elif (scalar.name == "builtin.unregistered" or scalar.regions
              or not (name.startswith(("arith.", "math.")) or name == "linalg.yield")):
            raise ValueError("structured exclusion encountered unsupported scalar effects")
        if any(v not in owners for v in scalar.operands):
            raise ValueError("structured exclusion encountered unowned scalar operands")
        rows.append(dict(operation=name, operands=[owners[v] for v in scalar.operands],
                         result_types=[str(v.type) for v in scalar.results], attributes=T._attributes(scalar)))
        owners.update({v: {"operation": ordinal, "result": i} for i, v in enumerate(scalar.results)})
    if not rows or rows[-1]["operation"] != "linalg.yield":
        raise ValueError("structured exclusion encountered incomplete scalar results")
    return dict(argument_types=[str(v.type) for v in block.args], operations=rows)


def _structured(op, *, body_reader=_body):
    if not isinstance(op, LinalgStructuredOperation) or op.name == "builtin.unregistered":
        raise ValueError("named tensor exclusion has no registered structured semantics")
    maps, iterators = op.get_indexing_maps(), op.get_iterator_types()
    kinds = [v.data.value for v in iterators]
    if not kinds or any(k not in {"parallel", "reduction"} for k in kinds):
        raise ValueError("named tensor exclusion has unsupported iteration semantics")
    if (len(maps) != len(op.operands) or len(op.outputs) != len(op.results)
            or any(a.type != b.type for a, b in zip(op.outputs, op.results, strict=True))):
        raise ValueError("named tensor exclusion has incomplete result ownership")
    for value, mapping in zip(op.operands, maps, strict=True):
        typ = value.type
        if (not isinstance(typ, builtin.TensorType) or not isinstance(typ.encoding, builtin.NoneAttr)
                or any(d <= 0 for d in typ.get_shape()) or mapping.data.num_symbols
                or mapping.data.num_dims != len(kinds) or len(mapping.data.results) != len(typ.get_shape())):
            raise ValueError("named tensor exclusion has incomplete typed indexing maps")
    body = body_reader(op)
    block = op.regions[0].block
    yield_op = block.last_op
    if (len(block.args) != len(op.operands)
            or any(arg.type != value.type.get_element_type()
                   for arg, value in zip(block.args, op.operands, strict=True))
            or yield_op is None or len(yield_op.operands) != len(op.outputs)
            or any(v.type != out.type.get_element_type()
                   for v, out in zip(yield_op.operands, op.outputs, strict=True))):
        raise ValueError("named tensor exclusion has incomplete scalar result ABI")
    return maps, iterators, body, "reduction" in kinds


def module_signatures(module, *, body_reader=_body):
    """Project a complete verified AST; no unknown named operation is skipped."""
    module.verify()
    result = []
    for op in module.walk():
        name = Q.op_name(op)
        if name == "linalg.index":
            _index(op)
            continue
        if name == "linalg.reduce":
            result.append(R._reduction(op, body_reader=body_reader))
            continue
        if name == "linalg.generic":
            iterators = op.properties.get("iterator_types")
            if iterators is None:
                raise ValueError("generic exclusion has no explicit iterator semantics")
            if "reduction" not in [v.data.value for v in iterators]:
                continue
        elif name not in T._NAMED:
            if not name.startswith("linalg.") or name in {
                "linalg.yield", "linalg.transpose", "linalg.fill", "linalg.copy"
            }:
                continue
            maps, iterators, body, reduction = _structured(op, body_reader=body_reader)
            if reduction:
                result.append(T._hash(dict(schema=OP_SCHEMA, operation=name,
                    ordered_operands=[T._type(v) for v in op.operands],
                    ordered_results=[T._type(v) for v in op.results], attributes=T._attributes(op),
                    indexing_maps=[str(v) for v in maps], iterator_types=[str(v) for v in iterators],
                    scalar_body=body)))
            continue
        if op.name == "builtin.unregistered":
            raise ValueError("typed contraction exclusion has no registered semantics")
        result.append(T._hash(dict(schema=T.SCHEMA, operation=name,
            ordered_operands=[T._type(v) for v in op.operands],
            ordered_results=[T._type(v) for v in op.results], attributes=T._attributes(op), scalar_body=body_reader(op))))
    return tuple(result)


def signatures(source, *, max_bytes):
    if type(max_bytes) is not int or max_bytes < 1 or not isinstance(source, bytes) or len(source) > max_bytes:
        raise ValueError("registered source exclusion exceeds the selected compact parse budget")
    if b"merlin_iface.version" in source:
        return M.signatures(source, max_bytes=max_bytes)
    return module_signatures(Q.parse(source.decode("utf-8")))
