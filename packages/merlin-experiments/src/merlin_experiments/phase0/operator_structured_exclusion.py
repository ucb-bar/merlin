"""Explicit V4 exclusion grammar: checked mixed sources plus registered reductions.

This is conservative protected-source exclusion, never execution or arithmetic
admission. Original contraction hashes remain V1; complete explicit reductions
receive an additional typed hash on BOTH private intake and public publication.
"""

from __future__ import annotations

from xdsl.dialects import builtin
from xdsl.dialects.linalg.ops import ReduceOp

from merlin.common import mlir_query as Q

from . import operator_mixed_exclusion as M
from . import operator_typed_exclusion as T

SCHEMA = "merlin.phase0.checked_mixed_source_exclusion.v2"
REDUCTION_SCHEMA = "merlin.phase0.typed_reduction_exclusion_signature.v1"


def _reduction(op, *, body_reader=T._body):
    if (type(op) is not ReduceOp or len(op.operands) != 2 or len(op.results) != 1
            or set(op.properties) != {"dimensions"}
            or any(not key.startswith("prov.") for key in op.attributes)):
        raise ValueError("typed reduction exclusion has no complete registered tensor semantics")
    types = [value.type for value in (*op.operands, *op.results)]
    if any(not isinstance(typ, builtin.TensorType)
           or not isinstance(typ.encoding, builtin.NoneAttr)
           or any(n <= 0 for n in typ.get_shape()) for typ in types):
        raise ValueError("typed reduction exclusion needs complete unencoded static tensor storage")
    source, initial, result = types
    dimensions = tuple(op.dimensions.get_values())
    rank = len(source.get_shape())
    if (not dimensions or len(set(dimensions)) != len(dimensions)
            or any(not 0 <= dim < rank for dim in dimensions)
            or initial != result or source.get_element_type() != initial.get_element_type()
            or tuple(n for i, n in enumerate(source.get_shape()) if i not in dimensions) != initial.get_shape()):
        raise ValueError("typed reduction exclusion has invalid dimensions or result ownership")
    body = body_reader(op)
    block = op.regions[0].block
    yield_op = block.last_op
    element = source.get_element_type()
    if (len(block.args) != 2 or any(value.type != element for value in block.args)
            or yield_op is None or len(yield_op.operands) != 1 or yield_op.operands[0].type != element):
        raise ValueError("typed reduction exclusion has an incomplete scalar ABI")
    return T._hash({"schema": REDUCTION_SCHEMA, "operation": Q.op_name(op),
                    "ordered_operands": [T._type(v) for v in op.operands],
                    "ordered_results": [T._type(v) for v in op.results],
                    "attributes": T._attributes(op), "scalar_body": body})


def signatures(source, *, max_bytes):
    """Select exact structured semantics explicitly; unknown named ops refuse."""
    if type(max_bytes) is not int or max_bytes < 1 or not isinstance(source, bytes) or len(source) > max_bytes:
        raise ValueError("structured source exclusion exceeds the selected compact parse budget")
    if b"merlin_iface.version" in source:
        return M.signatures(source, max_bytes=max_bytes)
    module = Q.parse(source.decode("utf-8"))
    return module_signatures(module)


def module_signatures(module):
    """The identical hash projection over a complete independently parsed AST."""
    module.verify()
    result = []
    for op in Q.walk(module):
        name = Q.op_name(op)
        if name == "linalg.reduce":
            result.append(_reduction(op))
            continue
        if name == "linalg.generic":
            iterators = op.properties.get("iterator_types")
            if iterators is None:
                raise ValueError("generic exclusion has no explicit iterator semantics")
            if "reduction" not in [value.data.value for value in iterators]:
                continue
        elif name not in T._NAMED:
            if name.startswith("linalg.") and name not in {
                "linalg.yield", "linalg.transpose", "linalg.fill", "linalg.copy"
            }:
                raise ValueError("protected typed IR contains an unsupported named tensor operation")
            continue
        if op.name == "builtin.unregistered":
            raise ValueError("protected typed IR has no registered contraction semantics")
        result.append(T._hash({"schema": T.SCHEMA, "operation": name,
                              "ordered_operands": [T._type(v) for v in op.operands],
                              "ordered_results": [T._type(v) for v in op.results],
                              "attributes": T._attributes(op), "scalar_body": T._body(op)}))
    return tuple(result)
