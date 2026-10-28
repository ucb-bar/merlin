"""Closed source proof for the i32 seed overflow guard that precedes a seeded integer contraction.

``m2m_q.int_mm_seeded(a, b, seed)`` is lowered as a chain, every member of which carries the same
``prov.aten``: a rank-1 guard ``linalg.generic`` over the ``[N]`` i32 seed that asserts each element lies in
``[-2^31 + b, 2^31 - 1 - b]``, a ``linalg.broadcast`` of the guarded seed to ``[M, N]``, and the seeded
contraction ``linalg.generic`` that starts from that broadcast. ``b = K * 128 * 128`` is the largest
magnitude an i8 x i8 reduction over ``K`` can add, so a seed inside the interval can never overflow the i32
accumulator. The guard is the only host work in the chain: the contraction itself is accelerator-admitted
and the broadcast is support lowering.

This screen proves, per parsed occurrence, that the guard is exactly that eight-operation body, that its
constants are the two ends of one interval whose half-width ``b`` is ``K * 16384`` for the ``K`` of the
contraction the guarded seed actually feeds (the broadcast and contraction are followed through the use
chain), and that the guarded value reaches nothing else. It grants no linked-build, ``cf.assert`` lowering
or execution qualification: how the host lowers an assertion inside a ``linalg.generic`` body is an open
obligation of the declaration that selects it.
"""

from __future__ import annotations

from merlin.common import mlir_query as mq

SCHEMA = "merlin.static_i32_seed_overflow_guard_source_body.v1"
OPERATION = "int32_seed_interval_assert"
FRONTEND_OP = "m2m_q.int_mm_seeded.default"
ASSERT_MESSAGE = "seeded contraction may overflow its int32 accumulator"
SIGNATURE = {"ordered_operand_dtypes": ["i32", "i32"], "ordered_result_dtypes": ["i32"], "ranks": [1]}
_BODY = (
    "arith.extsi",
    "arith.constant",
    "arith.cmpi",
    "arith.constant",
    "arith.cmpi",
    "arith.andi",
    "cf.assert",
    "linalg.yield",
)
_SGE, _SLE = 5, 3
_I32_MIN, _I32_MAX = -(1 << 31), (1 << 31) - 1
_PER_K = 128 * 128


class SeedGuardRefused(ValueError):
    """The parsed generic is not the declared closed seed overflow guard."""


def validate(body):
    """Validate the closed declaration; it selects a source proof and admits nothing."""
    if type(body) is not dict or set(body) != {"schema", "operation"} or body["schema"] != SCHEMA or body["operation"] != OPERATION:
        raise ValueError("seed guard needs one closed schema and operation")
    return dict(body)


def validate_row(row):
    """A seed-guard declaration names one frontend op, the exact rank-1 i32 ABI and a placement-only contract."""
    validate(row["source_body"])
    signature = row.get("signature") or {}
    if (
        row.get("ops") != [FRONTEND_OP]
        or {k: v for k, v in signature.items() if k != "family"} != SIGNATURE
        or signature.get("family") != "elementwise_map"
        or "families" in row
        or "family" in row
    ):
        raise ValueError("seed guard needs the seeded-contraction frontend op and the exact rank-1 i32 ABI")
    if not isinstance(row.get("numerical_contract"), dict) or row["numerical_contract"].get("status") != "unreviewed":
        raise ValueError("seed guard routing declares no numerical qualification")


def _static_type(value):
    shape, dtype = mq.type_shape_dtype(value.type)
    return list(shape), dtype


def _literal(value, width):
    owner = value.owner
    if mq.op_name(owner) != "arith.constant":
        raise SeedGuardRefused("guard bound is not an arith.constant")
    text = str(owner.properties.get("value", ""))
    head, _, tail = text.partition(" : ")
    if tail != f"i{width}":
        raise SeedGuardRefused("guard bound has an unexpected literal type")
    try:
        return int(head)
    except ValueError as exc:
        raise SeedGuardRefused("guard bound is not an integer literal") from exc


def _semantic_attributes(op):
    return {k: str(v) for k, v in {**op.attributes, **op.properties}.items() if not k.startswith("prov.")}


def _sole_user(value):
    users = {id(use.operation): use.operation for use in value.uses}
    if len(users) != 1 or sum(1 for _ in value.uses) != 1:
        raise SeedGuardRefused("guarded seed does not have exactly one use")
    return next(iter(users.values()))


def _consuming_contraction(guard):
    """Follow guard -> linalg.broadcast -> seeded contraction and return (contraction, K)."""
    from merlin.targetgen.application_inventory import exact_int_mm_generic_operation

    broadcast = _sole_user(guard.results[0])
    if (
        mq.op_name(broadcast) != "linalg.broadcast"
        or broadcast.operands[0] is not guard.results[0]
        or str(broadcast.properties.get("dimensions")) != "array<i64: 0>"
        or len(broadcast.operands) != 2
        or mq.op_name(broadcast.operands[1].owner) != "tensor.empty"
    ):
        raise SeedGuardRefused("guarded seed is not broadcast over the row axis into a fresh tensor")
    seed_shape = _static_type(guard.results[0])[0]
    wide_shape, wide_dtype = _static_type(broadcast.results[0])
    if wide_dtype != "i32" or len(wide_shape) != 2 or wide_shape[1] != seed_shape[0]:
        raise SeedGuardRefused("broadcast is not [N] -> [M, N] i32")
    contraction = _sole_user(broadcast.results[0])
    if (
        mq.op_name(contraction) != "linalg.generic"
        or mq.attr_str(contraction, "prov.aten") != FRONTEND_OP
        or mq.attr_str(contraction, "prov.accumulator_init") != "int32_seed"
        or len(contraction.operands) != 3
        or contraction.operands[2] is not broadcast.results[0]
        or not exact_int_mm_generic_operation(contraction, require_zero_init=False)
    ):
        raise SeedGuardRefused("broadcast seed does not initialize an exact seeded i8 x i8 -> i32 contraction")
    lhs_shape, rhs_shape = _static_type(contraction.operands[0])[0], _static_type(contraction.operands[1])[0]
    if len(lhs_shape) != 2 or len(rhs_shape) != 2 or lhs_shape[1] != rhs_shape[0] or rhs_shape[1] != seed_shape[0]:
        raise SeedGuardRefused("contraction operands do not match the guarded seed length")
    return contraction, lhs_shape[1]


def prove(op) -> dict:
    """Return the closed proof for one parsed occurrence or raise SeedGuardRefused."""
    if mq.op_name(op) != "linalg.generic" or mq.attr_str(op, "prov.aten") != FRONTEND_OP:
        raise SeedGuardRefused("occurrence is not a seeded-contraction linalg.generic")
    if (
        str(op.properties.get("operandSegmentSizes")) != "array<i32: 1, 1>"
        or len(op.operands) != 2
        or len(op.results) != 1
        or str(op.properties.get("indexing_maps")) != "[affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>]"
        or str(op.properties.get("iterator_types")) != "[#linalg.iterator_type<parallel>]"
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise SeedGuardRefused("guard is not a one-input, one-init, identity-mapped parallel rank-1 generic")
    shape, dtype = _static_type(op.operands[0])
    if dtype != "i32" or len(shape) != 1 or any(_static_type(v) != (shape, "i32") for v in (*op.operands, *op.results)):
        raise SeedGuardRefused("seed, init and result are not one static rank-1 i32 tensor type")
    if mq.op_name(op.operands[1].owner) != "tensor.empty":
        raise SeedGuardRefused("guard init is not a fresh tensor.empty")
    block = op.regions[0].blocks[0]
    inner = [o for o in block.ops]
    if tuple(mq.op_name(o) for o in inner) != _BODY or [str(a.type) for a in block.args] != ["i32", "i32"]:
        raise SeedGuardRefused("guard body is not the closed eight-operation interval assertion")
    if any(True for _ in block.args[1].uses):
        raise SeedGuardRefused("guard init block argument must be unread")
    ext, low, sge, high, sle, both, assertion, yielded = inner
    for member in inner:
        if member.regions or member.successors:
            raise SeedGuardRefused("guard body operation carries a region or successor")
    if (
        list(ext.operands) != [block.args[0]]
        or str(ext.results[0].type) != "i64"
        or _semantic_attributes(ext)
        or list(sge.operands) != [ext.results[0], low.results[0]]
        or list(sle.operands) != [ext.results[0], high.results[0]]
        or _semantic_attributes(sge) != {"predicate": str(_SGE) + " : i64"}
        or _semantic_attributes(sle) != {"predicate": str(_SLE) + " : i64"}
        or list(both.operands) != [sge.results[0], sle.results[0]]
        or _semantic_attributes(both)
        or list(assertion.operands) != [both.results[0]]
        or _semantic_attributes(assertion) != {"msg": f'"{ASSERT_MESSAGE}"'}
        or list(yielded.operands) != [block.args[0]]
        or _semantic_attributes(yielded)
    ):
        raise SeedGuardRefused("guard body does not wire the declared sign-extend, interval compare, assert and yield")
    lower, upper = _literal(low.results[0], 64), _literal(high.results[0], 64)
    bound = lower - _I32_MIN
    if bound <= 0 or upper != _I32_MAX - bound:
        raise SeedGuardRefused("guard bounds are not one interval symmetric about the i32 range")
    contraction, reduction = _consuming_contraction(op)
    if bound != reduction * _PER_K:
        raise SeedGuardRefused("guard half-width differs from the consuming contraction's K * 128 * 128")
    return {
        "seed_length": shape[0],
        "interval": [lower, upper],
        "half_width": bound,
        "reduction_extent": reduction,
        "consumer_ordinal_hint": "linalg.generic seeded contraction via linalg.broadcast",
    }


def screen(declaration, row, signature, source_operations, source_context):
    """Match every supplied parsed occurrence; an omitted source is never evidence."""
    del source_context
    body = declaration["source_body"]
    if source_operations is None or not source_operations:
        return {"status": "unknown", "reason": "seed guard source_body requires parsed source operations"}
    if (
        not isinstance(source_operations, tuple)
        or row.get("mlir_operation") != "linalg.generic"
        or row.get("frontend_op", FRONTEND_OP) != FRONTEND_OP
        or ("count" in row and (type(row["count"]) is not int or row["count"] != len(source_operations)))
        or len({id(op) for op in source_operations}) != len(source_operations)
    ):
        return {"status": "unsupported", "reason": "seed guard occurrence roster differs from source row"}
    if (
        signature.get("ordered_operand_dtypes") != SIGNATURE["ordered_operand_dtypes"]
        or signature.get("ordered_result_dtypes") != SIGNATURE["ordered_result_dtypes"]
        or signature.get("rank") != 1
    ):
        return {"status": "unsupported", "reason": "seed guard differs from observed rank-1 i32 ABI"}
    try:
        proofs = [prove(op) for op in source_operations]
    except SeedGuardRefused as exc:
        return {"status": "unsupported", "reason": f"seed guard structural proof refused: {exc}"}
    return {
        "status": "admitted",
        "reason": "every parsed occurrence is the declared seed overflow guard of its consuming contraction",
        "proof": {
            "schema": SCHEMA,
            "declaration": declaration["id"],
            "operation": body["operation"],
            "patterns": proofs,
            "scope": "source body identity and interval-to-K binding only; host lowering of the assertion and execution remain required",
        },
    }
