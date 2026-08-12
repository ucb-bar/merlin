"""The ``saturn`` reference target dialect (xDSL) — the OUT-OF-TREE package home.

Evicted from the shared library (``merlin/python/merlin/xdsl_dialects/targets/saturn.py``): a
reference target's dialect is now DATA its own package contributes, not a hardcoded entry in shared
lowering. Loaded by :func:`merlin.runtime.backends.base._load_oot_backend` as
``merlin._oot_dialects.saturn`` because the saturn target contract declares ``plugin.dialect: dialect``
and the dialect-discovery pass walks curated reference targets (see
:func:`merlin.xdsl_dialects.lowering.target_lowering._ensure_dialects_discovered`). Importing this
module runs its module-level self-registration (:func:`register_dialect_spec`) so the lowering registry
picks up saturn without shared code naming it.

Data + one factory call: the op/type shape (pack / matmul / commit / release over packed_tensor /
accumulator) is synthesized by :func:`merlin.xdsl_dialects.targets.factory.build_dialect` from the
committed ``merlin/targets/saturn/contracts/dialect_plan.yaml`` — no hand-written IRDL classes. Saturn
is the chipyard RVV vector unit modeled as a multicore RV64GCV CPU: residency is a packed weight kept
live in memory, its matmul RHS may be plain (not type-constrained) and carries an optional vl-policy
prop.

Parent imports are ABSOLUTE (``merlin.xdsl_dialects.*``) so the module resolves out-of-tree, loaded by
file path under a synthetic package name.
"""
from __future__ import annotations

import sys

from merlin.xdsl_dialects._common import HAS_XDSL

DIALECT_NAME = "saturn"
OPS = ["pack", "matmul", "commit", "release", "vector_map", "vector_reduce"]
TYPES = ["packed_tensor", "accumulator"]

# Target op -> Merlin-owned abstract command-buffer opcode. This is saturn-dialect DATA — it rides with
# the dialect (self-registered below), not baked into shared runtime_lowering. The command buffer is
# Merlin's; every target encodes onto the same opcode set, which is what keeps metrics comparable.
OPCODES = {
    "saturn.pack": "RES_PACK",
    "saturn.matmul": "MATMUL_RESIDENT",
    "saturn.commit": "COMMIT",
    "saturn.release": "EVICT",
    "saturn.vector_map": "VECTOR_MAP",
    "saturn.vector_reduce": "VREDUCE",
}

if HAS_XDSL:
    from merlin.xdsl_dialects.targets.factory import build_dialect

    _BUILT = build_dialect("saturn", matmul_vl_policy=True)
    # exported op/type classes (names preserved for target_lowering._specs and dialect round-trip)
    PackOp = _BUILT.pack_op
    MatmulOp = _BUILT.matmul_op
    CommitOp = _BUILT.commit_op
    ReleaseOp = _BUILT.evict_op
    VectorMapOp = _BUILT.vector_map_op
    VectorReduceOp = _BUILT.vector_reduce_op
    PackedTensorType = _BUILT.resident_type
    AccumulatorType = _BUILT.accumulator_type
    SATURN_DIALECT = _BUILT.dialect

    def get_dialect():
        return SATURN_DIALECT

    # Self-register the reference TargetSpec (and its op->opcode map) into the shared lowering registry.
    # This is the seam that makes saturn a DISCOVERED plugin: target_lowering._specs() ends up with the
    # built-in toynpu spec PLUS this one, without a saturn literal in shared lowering.
    from merlin.xdsl_dialects.lowering.target_lowering import TargetSpec, register_dialect_spec

    _SPEC = TargetSpec("saturn", sys.modules.get(__name__), PackOp, MatmulOp, CommitOp, ReleaseOp,
                       PackedTensorType, AccumulatorType,
                       vector_map_op=VectorMapOp, vector_reduce_op=VectorReduceOp)
    register_dialect_spec(_SPEC, opcodes=OPCODES)

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None
