"""Conservative typed-IR exclusion hashes, shared by private and public intake.

This is an exclusion projection, not a numerical equivalence or execution proof.
Initializer contents and unrelated graph names are deliberately omitted: a match
may reject additional public work, but cannot admit protected work or a compiler.
"""

from __future__ import annotations

import hashlib
import json

from xdsl.dialects import builtin

from merlin.common import mlir_query as Q

SCHEMA = "merlin.phase0.typed_ir_exclusion_signature.v1"
_NAMED = {
    "linalg.matmul", "linalg.batch_matmul", "linalg.matvec", "linalg.vecmat", "linalg.dot",
    "linalg.conv_2d_nhwc_hwcf", "linalg.conv_2d_nchw_fchw",
}


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _type(value):
    typ = value.type
    if not isinstance(typ, builtin.TensorType) or any(d <= 0 for d in typ.get_shape()):
        raise ValueError("typed exclusion needs complete static tensor operands and results")
    return {"shape": list(typ.get_shape()), "element_type": str(typ.get_element_type())}


def _attributes(op):
    return {key: str(value) for key, value in sorted({**op.attributes, **op.properties}.items())
            if not key.startswith("prov.") and key != "operandSegmentSizes"}


def _body(op):
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        raise ValueError("typed contraction exclusion needs one complete scalar region")
    block = op.regions[0].block
    owners = {value: {"argument": i} for i, value in enumerate(block.args)}
    rows = []
    for ordinal, scalar in enumerate(block.ops):
        name = Q.op_name(scalar)
        if scalar.regions or not (name.startswith(("arith.", "math.")) or name == "linalg.yield"):
            raise ValueError("typed contraction exclusion encountered unsupported scalar effects")
        if any(value not in owners for value in scalar.operands):
            raise ValueError("typed contraction exclusion encountered unowned scalar operands")
        rows.append({"operation": name, "operands": [owners[v] for v in scalar.operands],
                     "result_types": [str(v.type) for v in scalar.results], "attributes": _attributes(scalar)})
        owners.update({value: {"operation": ordinal, "result": i} for i, value in enumerate(scalar.results)})
    if not rows or rows[-1]["operation"] != "linalg.yield":
        raise ValueError("typed contraction exclusion encountered incomplete scalar results")
    return {"argument_types": [str(value.type) for value in block.args], "operations": rows}


def signatures(source, *, max_bytes):
    """Parse exact original IR under an explicit bound; never scrape shape text."""
    if type(max_bytes) is not int or max_bytes < 1 or not isinstance(source, bytes) or len(source) > max_bytes:
        raise ValueError("protected typed IR exceeds the selected parse budget")
    module = Q.parse(source.decode("utf-8"))
    module.verify()
    result = []
    for op in Q.walk(module):
        name = Q.op_name(op)
        if name == "linalg.generic":
            iterators = op.properties.get("iterator_types")
            if iterators is None:
                raise ValueError("generic exclusion has no explicit iterator semantics")
            if "reduction" not in [value.data.value for value in iterators]:
                continue
        elif name not in _NAMED:
            # Other linalg named operations can hide reductions behind implicit
            # indexing. Refuse rather than silently claim a complete projection.
            if name.startswith("linalg.") and name not in {
                "linalg.yield", "linalg.transpose", "linalg.fill", "linalg.copy"
            }:
                raise ValueError("protected typed IR contains an unsupported named tensor operation")
            continue
        result.append(_hash({"schema": SCHEMA, "operation": name,
                             "ordered_operands": [_type(v) for v in op.operands],
                             "ordered_results": [_type(v) for v in op.results],
                             "attributes": _attributes(op), "scalar_body": _body(op)}))
    return tuple(result)
