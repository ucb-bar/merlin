"""Narrow default-property compatibility at selected stock MLIR roundtrips.

MLIR 23 prints an explicit empty overflow-flag property on integer truncation.
The pinned xDSL definition represents the same unflagged operation without that
property. Nondefault flags cannot be represented by this compatibility rule.
"""

from __future__ import annotations


def unflagged_truncations(module):
    """Normalize only exact default-none properties on owned parsed operations.

    All candidates validate before any mutation. SSA operands/results, complete
    attributes and cast types remain identical; the original source bytes are
    never changed. The caller still verifies the complete resulting module.
    """
    from xdsl.dialects.arith import IntegerOverflowAttr, TruncIOp

    candidates = []
    for ordinal, op in enumerate(module.walk()):
        if not isinstance(op, TruncIOp):
            continue
        if "overflowFlags" in op.attributes:
            raise ValueError("integer truncation flag attributes are not a proved stock property")
        if "overflowFlags" not in op.properties:
            continue
        flags = op.properties["overflowFlags"]
        if type(flags) is not IntegerOverflowAttr or flags.data or set(op.properties) != {"overflowFlags"}:
            raise ValueError("stock integer truncation requires exact default-none overflow flags")
        check = op.clone()
        del check.properties["overflowFlags"]
        check.verify()
        candidates.append(
            (
                op,
                {
                    "ordinal": ordinal,
                    "operation": op.name,
                    "source_property": {"overflowFlags": str(flags)},
                    "operand_types": [str(v.type) for v in op.operands],
                    "result_types": [str(v.type) for v in op.results],
                    "attributes": {key: str(value) for key, value in op.attributes.items()},
                    "operand_and_result_identity_preserved": True,
                    "attributes_preserved": True,
                    "semantics": "unflagged integer truncation; no poison flags granted",
                },
            )
        )
    for op, _record in candidates:
        del op.properties["overflowFlags"]
    return [record for _op, record in candidates]
