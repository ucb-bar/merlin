"""Explicit V6 exclusion grammar for complete pure scalar capture dependencies.

The original registered region reader supplies SSA edges, effects and attributes.
Additional capture ownership is checked against the actual enclosing function,
dominance and complete pure producer DAG, including original constant storage.
This grants no arithmetic, compiler or execution qualification.
"""

from __future__ import annotations

import io

from xdsl.dialects import arith, builtin, math
from xdsl.ir import BlockArgument, OpResult

from merlin.common import mlir_query as Q
from merlin.targetgen.contract.linalg_iface import _effects, _scalar_body

from . import operator_dense_projection as P
from . import operator_initializer_identity as ID
from . import operator_mixed_exclusion as M
from . import operator_registered_exclusion as G
from . import operator_typed_exclusion as T

SCHEMA = "merlin.phase0.pure_capture_source_exclusion.v1"
BODY_SCHEMA = "merlin.phase0.pure_capture_scalar_dag.v1"
_SCALAR_ARITHMETIC = frozenset((*arith.Arith.operations, *math.Math.operations))


def _body(op, initializers):
    try:
        return G._body(op)
    except ValueError:
        pass  # Only this explicitly selected version may check the extension.
    if (len(op.regions) != 1 or len(op.regions[0].blocks) != 1
            or op.parent_op() is None or Q.op_name(op.parent_op()) != "func.func"):
        raise ValueError("captured scalar exclusion needs one immediate function-owned region")
    block = op.regions[0].block
    function = op.parent_op()
    if len(function.regions) != 1 or len(function.regions[0].blocks) != 1:
        raise ValueError("captured scalar exclusion has no unique function entry")
    entry = function.regions[0].block
    order = {candidate: i for i, candidate in enumerate(entry.ops)}
    seen, captures, active = set(block.args), {}, set()

    def pure(candidate, *, scalar=False):
        name = Q.op_name(candidate)
        if name == "linalg.index" and scalar:
            G._index(candidate)
            return
        # The retained registered arithmetic reader predates effects traits on
        # several conversion operations (e.g. sign extension). Its regionless
        # arithmetic/math definitions are pure value operations, as in the
        # original scalar grammar; other dialects require explicit empty effects.
        arithmetic = type(candidate) in _SCALAR_ARITHMETIC
        if (candidate.name == "builtin.unregistered" or candidate.regions
                or not name.startswith(("arith.", "math.", "tensor."))
                or name == "tensor.empty" or (_effects(candidate) != [] and not arithmetic)):
            raise ValueError("captured scalar exclusion encountered unknown or impure effects")

    def capture(value, consumer=op, depth=0):
        if isinstance(value, OpResult) and (
                value.owner.parent is not entry or consumer not in order or value.owner not in order
                or order[value.owner] >= order[consumer]):
            raise ValueError("captured scalar exclusion producer does not dominate its consumer")
        if isinstance(value, BlockArgument) and value.owner is not entry:
            raise ValueError("captured scalar exclusion encountered a foreign block argument")
        if value in captures:
            return captures[value]
        if depth > 64 or len(captures) + len(active) > 4096 or value in active:
            raise ValueError("captured scalar exclusion exceeds its acyclic producer bound")
        active.add(value)
        if isinstance(value, BlockArgument):
            result = dict(argument=value.index, type=str(value.type))
        elif isinstance(value, OpResult):
            owner = value.owner
            pure(owner)
            attributes = T._attributes(owner)
            constant = None
            if Q.op_name(owner) == "arith.constant":
                constant = initializers.get(owner) or ID.direct_identity(owner)
                if isinstance(owner.properties.get("value"), builtin.DenseResourceAttr) and constant is None:
                    raise ValueError("captured scalar exclusion has no original initializer identity")
                if constant is not None:
                    attributes.pop("value", None)
            result = dict(operation=Q.op_name(owner), result=value.index,
                          result_types=[str(v.type) for v in owner.results], attributes=attributes,
                          operands=[capture(v, owner, depth + 1) for v in owner.operands],
                          effects=[], constant=constant)
        else:
            raise ValueError("captured scalar exclusion has no actual SSA owner")
        active.remove(value)
        # Hash each complete node, bounding repeated DAG serialization without
        # discarding edges, result indices, types or dependency multiplicity.
        captures[value] = dict(producer_sha256=T._hash(result), type=str(value.type))
        return captures[value]

    for scalar in block.ops:
        if Q.op_name(scalar) != "linalg.yield":
            pure(scalar, scalar=True)
        elif scalar is not block.last_op or scalar.results or scalar.regions:
            raise ValueError("captured scalar exclusion has an invalid final yield")
        for value in scalar.operands:
            if value not in seen:
                capture(value)
        seen.update(scalar.results)
    body = _scalar_body(op, capture)
    if body is None:
        raise ValueError("captured scalar exclusion lost its complete scalar body")
    for row in body["operations"]:
        if row["op"] == "linalg.index":
            row["effects"] = []  # Exact enclosing iterator ownership checked above.
    body["effects"] = []
    return dict(schema=BODY_SCHEMA, **body)


def module_signatures(module, *, initializers=None):
    module.verify()
    return G.module_signatures(module, body_reader=lambda op: _body(op, initializers or {}))


def signatures(source, *, max_bytes):
    if type(max_bytes) is not int or max_bytes < 1 or not isinstance(source, bytes) or len(source) > max_bytes:
        raise ValueError("pure capture exclusion exceeds the selected original source budget")
    if b"merlin_iface.version" in source:
        return M.signatures(source, max_bytes=max_bytes)
    _, compact, constants = P.scan(io.BytesIO(source), max_input_bytes=max_bytes, max_compact_bytes=max_bytes)
    module = Q.parse(compact.decode("utf-8"))
    initializers = ID.identities(module, constants, max_decoded_bytes=4 << 20)
    return module_signatures(module, initializers=initializers)
