"""The in-tree reference ``saturn`` target dialect (xDSL).

Data + one factory call: the op/type shape (pack / matmul / commit / release over packed_tensor /
accumulator) is synthesized by :func:`..factory.build_dialect` from the committed
``merlin/targets/saturn/contracts/dialect_plan.yaml`` — no hand-written IRDL classes. Saturn is the
chipyard RVV vector unit modeled as a multicore RV64GCV CPU: residency is a packed weight kept live
in memory, its matmul RHS may be plain (not type-constrained) and carries an optional vl-policy prop.
"""
from __future__ import annotations

from .._common import HAS_XDSL

DIALECT_NAME = "saturn"
OPS = ["pack", "matmul", "commit", "release"]
TYPES = ["packed_tensor", "accumulator"]

if HAS_XDSL:
    from .factory import build_dialect

    _BUILT = build_dialect("saturn", matmul_vl_policy=True)
    # exported op/type classes (names preserved for target_lowering._specs and dialect round-trip)
    PackOp = _BUILT.pack_op
    MatmulOp = _BUILT.matmul_op
    CommitOp = _BUILT.commit_op
    ReleaseOp = _BUILT.evict_op
    PackedTensorType = _BUILT.resident_type
    AccumulatorType = _BUILT.accumulator_type
    SATURN_DIALECT = _BUILT.dialect

    def get_dialect():
        return SATURN_DIALECT

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None
