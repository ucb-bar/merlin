"""Closed source proof for a two-result f32 row-moments generic with a captured data operand.

Some layer-norm captures emit the moments as one ``linalg.generic`` with zero
inputs and two ``[R, 1]`` f32 inits whose body reads the ``[R, C]`` data tensor
through ``tensor.extract`` on a value captured from the enclosing scope. A typed
operand signature then describes only the two inits. This screen supplies the
missing source facts: the single captured data operand, its dtype and static
shape, the row index (``linalg.index`` 0 over the ``[R, 1]`` domain), and a
concrete enumeration showing every column index is in bounds and that the
whole body reads each of the ``C`` columns of its row exactly once (reduction
axis 1). The parsed body is pinned by a canonical digest (provenance
attributes excluded, SSA names normalized) so declared numerical evidence
binds to exactly the screened arithmetic.

It grants no linked-build, emitted-code, fused-multiply-add lowering or
all-input numerical qualification.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from itertools import product

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.common.jsonio import canonical_json

SCHEMA = "merlin.static_f32_row_moments_source_body.v1"
#: Rank-3 data operand ([B, S, C], rows indexed by linalg.index 0 and 1). A separate schema, so the
#: rank-2 canonical serialization and every digest pinned to it stay byte-identical.
SCHEMA_RANK3 = "merlin.static_f32_row_moments_rank3_source_body.v1"
SCHEMAS = (SCHEMA, SCHEMA_RANK3)
_SCHEMA_RANK = {SCHEMA: 2, SCHEMA_RANK3: 3}
OPERATION = "row_mean_biased_variance"
FRONTEND_OP = "aten.native_layer_norm.default"
SIGNATURE = {"ordered_operand_dtypes": ["f32", "f32"], "ordered_result_dtypes": ["f32", "f32"], "ranks": [2]}
SIGNATURE_RANK3 = {**SIGNATURE, "ranks": [3]}
_ALLOWED_BODY_OPS = frozenset(
    {
        "linalg.index",
        "linalg.yield",
        "arith.constant",
        "arith.addi",
        "arith.subi",
        "arith.muli",
        "arith.index_cast",
        "arith.sitofp",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "math.fma",
        "tensor.extract",
        "scf.for",
        "scf.yield",
    }
)
_IDENTITY_MAPS = "[affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>]"
_PARALLEL = "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>]"


def _identity_maps(rank):
    dims = ", ".join(f"d{i}" for i in range(rank))
    one = f"affine_map<({dims}) -> ({dims})>"
    return f"[{one}, {one}]"


def _parallel(rank):
    return "[" + ", ".join(["#linalg.iterator_type<parallel>"] * rank) + "]"
_MAX_ENUMERATION = 1 << 16


class RowMomentsRefused(ValueError):
    """The parsed generic is not the declared closed row-moments body."""


def validate(body):
    """Validate the closed declaration; it selects a source proof and admits nothing."""
    data = body.get("data_operand") if type(body) is dict else None
    shape = data.get("shape") if type(data) is dict else None
    rank = _SCHEMA_RANK.get(body.get("schema")) if type(body) is dict else None
    if (
        type(body) is not dict
        or set(body) != {"schema", "operation", "data_operand", "canonical_body_sha256"}
        or rank is None
        or body["operation"] != OPERATION
        or type(data) is not dict
        or set(data) != {"dtype", "shape", "reduction_axis"}
        or data["dtype"] != "f32"
        or data["reduction_axis"] != rank - 1
        or type(shape) is not list
        or len(shape) != rank
        or any(type(d) is not int or d <= 0 for d in shape)
        or not is_sha256(body["canonical_body_sha256"])
    ):
        raise ValueError("row moments needs one closed rank-2 f32 data operand, axis 1 and a body digest")
    return dict(body, data_operand=dict(data, shape=list(shape)))


def validate_row(row):
    """A row-moments declaration names one frontend op and the exact two-init f32 ABI."""
    validate(row["source_body"])
    expected = SIGNATURE_RANK3 if row["source_body"]["schema"] == SCHEMA_RANK3 else SIGNATURE
    if row.get("ops") != [FRONTEND_OP] or row.get("signature") != expected:
        raise ValueError("row moments needs the exact layer-norm frontend op and two-init f32 rank-2 ABI")
    if not isinstance(row.get("numerical_contract"), dict):
        raise ValueError("row moments needs an explicit numerical contract")


def _static_type(value):
    shape, dtype = mq.type_shape_dtype(value.type)
    return list(shape), dtype


def _inner_ops(op):
    inner = list(op.walk())
    if not inner or inner[0] is not op:
        raise RowMomentsRefused("generic walk does not start at the generic")
    return inner


def canonical_body(op, schema=SCHEMA) -> dict:
    """Serialize the generic without provenance and with position-based value names."""
    inner = _inner_ops(op)
    op_ids = {id(o): i for i, o in enumerate(inner)}
    block_ids = {}
    for o in inner:
        for region in o.regions:
            for block in region.blocks:
                block_ids[id(block)] = len(block_ids)

    def ref(value):
        owner = value.owner
        if id(owner) in op_ids:
            return ["result", op_ids[id(owner)], value.index]
        if id(owner) in block_ids:
            return ["argument", block_ids[id(owner)], value.index]
        return ["captured", str(value.type)]

    return {
        "schema": schema,
        "ops": [
            {
                "name": mq.op_name(o),
                "operands": [ref(v) for v in o.operands],
                "results": [str(r.type) for r in o.results],
                "properties": {k: str(v) for k, v in sorted(o.properties.items())},
                "attributes": {k: str(v) for k, v in sorted(o.attributes.items()) if not k.startswith("prov.")},
                "regions": [[[str(a.type) for a in b.args] for b in r.blocks] for r in o.regions],
            }
            for o in inner
        ],
    }


def canonical_body_sha256(op, schema=SCHEMA) -> str:
    payload = canonical_json(canonical_body(op, schema))
    return sha256(payload if isinstance(payload, bytes) else payload.encode()).hexdigest()


def _constant_index(value, inside):
    owner = value.owner
    if id(owner) not in inside or mq.op_name(owner) != "arith.constant" or str(value.type) != "index":
        raise RowMomentsRefused("loop bound or index term is not an in-body index constant")
    text = str(owner.properties.get("value", ""))
    head, _, tail = text.partition(" : ")
    if tail != "index":
        raise RowMomentsRefused("index constant has an unexpected literal")
    return int(head)


def _evaluate(value, env, inside):
    owner = value.owner
    if id(owner) in env and value.index == 0:
        return env[id(owner)]
    if id(owner) not in inside:
        raise RowMomentsRefused("column index depends on a value outside the screened body")
    name = mq.op_name(owner)
    if name == "arith.constant":
        return _constant_index(value, inside)
    if name in {"arith.addi", "arith.subi", "arith.muli"} and str(value.type) == "index":
        lhs, rhs = (_evaluate(v, env, inside) for v in owner.operands)
        return {"arith.addi": lhs + rhs, "arith.subi": lhs - rhs, "arith.muli": lhs * rhs}[name]
    raise RowMomentsRefused(f"column index uses unsupported {name}")


def _enclosing_loops(op, generic):
    loops = []
    parent = op.parent_op()
    while parent is not None and parent is not generic:
        if mq.op_name(parent) != "scf.for":
            raise RowMomentsRefused("data read is nested in a non-loop region")
        loops.append(parent)
        parent = parent.parent_op()
    if parent is not generic:
        raise RowMomentsRefused("data read is not inside the generic")
    return list(reversed(loops))


def prove(op, declaration: dict) -> dict:
    """Return the closed proof for one parsed occurrence or raise RowMomentsRefused."""
    schema = declaration.get("schema", SCHEMA)
    *lead, cols = declaration["data_operand"]["shape"]
    rank = len(lead) + 1
    if mq.op_name(op) != "linalg.generic" or mq.attr_str(op, "prov.aten") != FRONTEND_OP:
        raise RowMomentsRefused("occurrence is not a layer-norm linalg.generic")
    if (
        str(op.properties.get("operandSegmentSizes")) != "array<i32: 0, 2>"
        or len(op.operands) != 2
        or len(op.results) != 2
        or str(op.properties.get("indexing_maps")) != _identity_maps(rank)
        or str(op.properties.get("iterator_types")) != _parallel(rank)
        or len(op.regions) != 1
        or len(op.regions[0].blocks) != 1
    ):
        raise RowMomentsRefused("generic is not zero-input, two identity-mapped parallel inits")
    for value in (*op.operands, *op.results):
        if _static_type(value) != ([*lead, 1], "f32"):
            raise RowMomentsRefused("init or result is not a static [R, 1] f32 tensor")
    if any(mq.op_name(v.owner) != "tensor.empty" for v in op.operands):
        raise RowMomentsRefused("init is not a fresh tensor.empty")
    block = op.regions[0].blocks[0]
    if [str(a.type) for a in block.args] != ["f32", "f32"] or any(any(True for _ in a.uses) for a in block.args):
        raise RowMomentsRefused("init block arguments must be f32 and unread")
    inner = _inner_ops(op)
    inside = {id(o) for o in inner[1:]}
    inside_blocks = {id(b) for o in inner for r in o.regions for b in r.blocks}
    data = None
    extracts = []
    for o in inner[1:]:
        name = mq.op_name(o)
        if name not in _ALLOWED_BODY_OPS or o.successors:
            raise RowMomentsRefused(f"body uses {name} outside the closed row-moments vocabulary")
        if name == "linalg.index" and str(o.properties.get("dim")) not in {f"{d} : i64" for d in range(rank - 1)}:
            raise RowMomentsRefused("body indexes a non-row iteration dimension")
        for position, value in enumerate(o.operands):
            owner = value.owner
            if id(owner) in inside or id(owner) in inside_blocks:
                continue
            if name != "tensor.extract" or position != 0:
                raise RowMomentsRefused("body captures an outer value other than the extracted data operand")
            if data is None:
                data = value
            elif value is not data:
                raise RowMomentsRefused("body captures more than one data operand")
        if name == "tensor.extract":
            extracts.append(o)
    if data is None or not extracts:
        raise RowMomentsRefused("body reads no captured data operand")
    if _static_type(data) != ([*lead, cols], declaration["data_operand"]["dtype"]):
        raise RowMomentsRefused("captured data operand differs from the declared static f32 shape")
    reads = Counter()
    enumerated = 0
    for extract in extracts:
        if len(extract.operands) != rank + 1 or extract.operands[0] is not data or str(extract.results[0].type) != "f32":
            raise RowMomentsRefused(f"data read is not one rank-{rank} f32 scalar extract of the data operand")
        for axis in range(rank - 1):
            row_index = extract.operands[1 + axis].owner
            if (
                id(row_index) not in inside
                or mq.op_name(row_index) != "linalg.index"
                or row_index.parent_op() is not op
                or str(row_index.properties.get("dim")) != f"{axis} : i64"
            ):
                raise RowMomentsRefused(
                    "data row index is not linalg.index 0 of the [R, 1] domain"
                    if rank == 2
                    else f"data row index {axis} is not linalg.index {axis} of the leading domain"
                )
        loops = _enclosing_loops(extract, op)
        ranges = []
        for loop in loops:
            lower, upper, step = (_constant_index(v, inside) for v in loop.operands[:3])
            if step <= 0:
                raise RowMomentsRefused("loop step is not a positive constant")
            ranges.append(range(lower, upper, step))
        for ivs in product(*ranges):
            enumerated += 1
            if enumerated > _MAX_ENUMERATION:
                raise RowMomentsRefused("column enumeration exceeds the closed bound")
            env = {id(loop.regions[0].blocks[0]): iv for loop, iv in zip(loops, ivs, strict=True)}
            column = _evaluate(extract.operands[rank], env, inside)
            if not 0 <= column < cols:
                raise RowMomentsRefused(f"column index {column} is outside [0, {cols})")
            reads[column] += 1
    if reads != Counter(range(cols)):
        raise RowMomentsRefused("body does not read each column of its row exactly once")
    digest = canonical_body_sha256(op, schema)
    if digest != declaration["canonical_body_sha256"]:
        raise RowMomentsRefused("canonical body digest differs from the declared evidenced body")
    return {
        "data_operand": {"dtype": "f32", "shape": [*lead, cols], "reduction_axis": rank - 1},
        "row_index": "linalg.index 0 over [R, 1]" if rank == 2 else f"linalg.index 0..{rank - 2} over {lead} + [1]",
        "column_reads": cols,
        "extract_sites": len(extracts),
        "canonical_body_sha256": digest,
    }


def screen(declaration, row, signature, source_operations, source_context):
    """Match every supplied parsed occurrence; an omitted source is never evidence."""
    del source_context
    body = declaration["source_body"]
    if source_operations is None or not source_operations:
        return {"status": "unknown", "reason": "row moments source_body requires parsed source operations"}
    if (
        not isinstance(source_operations, tuple)
        or row.get("mlir_operation") != "linalg.generic"
        or row.get("frontend_op", FRONTEND_OP) != FRONTEND_OP
        or ("count" in row and (type(row["count"]) is not int or row["count"] != len(source_operations)))
        or len({id(op) for op in source_operations}) != len(source_operations)
    ):
        return {"status": "unsupported", "reason": "row moments occurrence roster differs from source row"}
    if (
        signature.get("ordered_operand_dtypes") != SIGNATURE["ordered_operand_dtypes"]
        or signature.get("ordered_result_dtypes") != SIGNATURE["ordered_result_dtypes"]
        or signature.get("rank") != _SCHEMA_RANK[body["schema"]]
    ):
        return {
            "status": "unsupported",
            "reason": f"row moments differs from observed two-init f32 rank-{_SCHEMA_RANK[body['schema']]} ABI",
        }
    try:
        proofs = [prove(op, body) for op in source_operations]
    except RowMomentsRefused as exc:
        return {"status": "unsupported", "reason": f"row moments structural proof refused: {exc}"}
    return {
        "status": "admitted",
        "reason": "every parsed occurrence is the declared row-moments body with proved data bounds",
        "proof": {
            "schema": body["schema"],
            "declaration": declaration["id"],
            "operation": body["operation"],
            "patterns": proofs,
            "scope": "source body identity and data index bounds only; host lowering, fused fma and execution remain required",
        },
    }
