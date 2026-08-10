"""The in-tree reference ``toynpu`` target dialect (xDSL).

Data + one factory call: the op/type shape (res_pack / matmul / commit / evict over
resident_tensor / accumulator) is synthesized by :func:`..factory.build_dialect` from the committed
``merlin/targets/toy_npu/contracts/dialect_plan.yaml`` — no hand-written IRDL classes. ToyNPU is an
NPU with real resident storage, so its matmul RHS is type-constrained to the resident tensor.
"""
from __future__ import annotations

from .._common import HAS_XDSL

DIALECT_NAME = "toynpu"
OPS = ["res_pack", "matmul", "commit", "evict", "vector_map", "vector_reduce"]
TYPES = ["resident_tensor", "accumulator"]

if HAS_XDSL:
    from .factory import build_dialect

    _BUILT = build_dialect("toy_npu", matmul_rhs_typed=True)
    # exported op/type classes (names preserved for target_lowering._specs and dialect round-trip)
    ResPackOp = _BUILT.pack_op
    MatmulOp = _BUILT.matmul_op
    CommitOp = _BUILT.commit_op
    EvictOp = _BUILT.evict_op
    VectorMapOp = _BUILT.vector_map_op
    VectorReduceOp = _BUILT.vector_reduce_op
    ResidentTensorType = _BUILT.resident_type
    AccumulatorType = _BUILT.accumulator_type
    TOYNPU_DIALECT = _BUILT.dialect

    def get_dialect():
        return TOYNPU_DIALECT

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None
