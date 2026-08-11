"""Isolated Radiance target dialect (xDSL prototype plane) — run hand_v0.

Self-contained: loaded dynamically by ``merlin.targetgen.registry.load_target``, never imported from
the core tree and never hardcoded in a shared lowering table. What it models is the *compiler-level*
abstraction of a Muon SIMT tensor-core cluster, not its ISA.

Radiance is deliberately NOT a renamed Gemmini, because a portability claim proved against two
structurally identical targets proves nothing. Where a weight-stationary systolic array packs an
operand into a feed FIFO and streams against it, a SIMT cluster **stages** the operand into shared
scratchpad memory and then has each warp cooperate on the tile. The two differ in the facts the
compiler has to commit to:

* ``memory_model.shared_memory: true`` — a staged operand lives in the scratchpad, whose capacity is
  a derived fact (``1 << SMEM_LOG_SIZE`` from Radiance's own kernel headers), not a declared number.
* ``compiler_obligations: [must_map_to_warps]`` with ``capabilities.simt.lanes_per_warp: 16`` — every
  matmul MUST record the warp width it is mapped onto. That is why ``radiance.matmul`` *requires*
  ``lanes_per_warp`` while ``gemmini.matmul`` has no such property: an obligation that nothing has to
  record is not an obligation, and this is what makes the SIMT arm a real second shape.

The warp width is supplied by :func:`op_properties`, which reads it from this package's own contract
and fails closed. The core rebuild loop merges it without interpreting it, so the fact that Radiance
maps to warps at all never becomes core knowledge — and the kernel frontend never learns it either.
"""
from __future__ import annotations

from typing import Any

from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr, i64
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException

DIALECT_NAME = "radiance"
OPS = ["stage", "matmul", "commit", "release"]
TYPES = ["shared_tensor", "accumulator"]

# Epilogue stages the SIMT cluster can fuse. `requant` is deliberately absent: the contract routes
# requantization to the CONTAINED MX PE (`requant.ref: gemmini_mx.mx_requantizer`), so accepting it
# here would claim a capability this unit does not have.
KNOWN_EPILOGUE = {"bias", "bias_add", "relu"}

# The scratchpad is banked; a staged tile is addressed per bank. Layouts the stage op accepts.
KNOWN_LAYOUTS = {"packed_rhs", "shared_tile"}


@irdl_attr_definition
class SharedTensorType(ParametrizedAttribute, TypeAttribute):
    """An operand staged in the cluster's shared scratchpad, visible to every warp in the block."""

    name = "radiance.shared_tensor"
    element_type: Attribute


@irdl_attr_definition
class AccumulatorType(ParametrizedAttribute, TypeAttribute):
    """A tensor-core accumulator. The contract promises f32 accumulate for every float input."""

    name = "radiance.accumulator"
    element_type: Attribute


@irdl_op_definition
class StageOp(IRDLOperation):
    """Move an operand into shared scratchpad memory (the SIMT analogue of a resident pack)."""

    name = "radiance.stage"
    src = operand_def()
    layout = prop_def(StringAttr)
    res = result_def(SharedTensorType)

    def verify_(self) -> None:
        if self.layout.data not in KNOWN_LAYOUTS:
            raise VerifyException(
                f"radiance.stage layout {self.layout.data!r} not in {sorted(KNOWN_LAYOUTS)}")


@irdl_op_definition
class MatmulOp(IRDLOperation):
    """A tensor-core matmul, mapped onto warps of a recorded width.

    ``lanes_per_warp`` is required, not optional. Radiance's contract carries
    ``compiler_obligations: [must_map_to_warps]``, and the only way to hold a compiler to that is to
    make the mapping impossible to omit — a verifier that accepts a matmul with no warp width has
    silently discharged the obligation.
    """

    name = "radiance.matmul"
    lhs = operand_def()
    rhs = operand_def()
    lanes_per_warp = prop_def(IntegerAttr)
    acc = result_def(AccumulatorType)

    def verify_(self) -> None:
        lanes = self.lanes_per_warp.value.data
        if lanes < 1:
            raise VerifyException(f"radiance.matmul lanes_per_warp must be >= 1, got {lanes}")
        if lanes & (lanes - 1):
            raise VerifyException(
                f"radiance.matmul lanes_per_warp {lanes} is not a power of two — a warp is a "
                "power-of-two lane group on this cluster")


@irdl_op_definition
class CommitOp(IRDLOperation):
    """Write the accumulator back to global memory, optionally fusing an epilogue."""

    name = "radiance.commit"
    acc = operand_def(AccumulatorType)
    epilogue = prop_def(ArrayAttr)
    bias = opt_prop_def(StringAttr)
    requant_shift = opt_prop_def(IntegerAttr)
    output_dtype = opt_prop_def(StringAttr)
    out = result_def()

    def verify_(self) -> None:
        stages = []
        for entry in self.epilogue:
            stage = entry.data if isinstance(entry, StringAttr) else None
            if stage not in KNOWN_EPILOGUE:
                raise VerifyException(
                    f"radiance.commit epilogue stage {stage!r} not in {sorted(KNOWN_EPILOGUE)} — "
                    "requantization belongs to the contained MX PE, not the SIMT cluster")
            stages.append(stage)
        if ("bias" in stages or "bias_add" in stages) and self.bias is None:
            raise VerifyException("radiance.commit bias stage but no `bias` tensor name")


@irdl_op_definition
class ReleaseOp(IRDLOperation):
    """Release the scratchpad allocation holding a staged operand."""

    name = "radiance.release"
    handle = operand_def(SharedTensorType)


RADIANCE_DIALECT = Dialect(DIALECT_NAME, [StageOp, MatmulOp, CommitOp, ReleaseOp],
                           [SharedTensorType, AccumulatorType])

# The registry builds a TargetSpec from this mapping (decoupled from the core's built-in table).
SPEC_OPS = {"pack": StageOp, "matmul": MatmulOp, "commit": CommitOp, "evict": ReleaseOp,
            "resident_type": SharedTensorType, "accumulator_type": AccumulatorType}


def op_properties(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Contract-derived properties the core rebuild loop must put on this target's ops.

    FAILS CLOSED. ``must_map_to_warps`` without a declared ``lanes_per_warp`` is an incoherent
    contract, and guessing a warp width would produce a module that verifies while describing
    hardware that does not exist.
    """
    obligations = list(contract.get("compiler_obligations") or [])
    lanes = ((contract.get("capabilities") or {}).get("simt") or {}).get("lanes_per_warp")
    if "must_map_to_warps" in obligations and lanes is None:
        raise ValueError(
            "radiance contract declares compiler_obligations: [must_map_to_warps] but no "
            "capabilities.simt.lanes_per_warp — the obligation cannot be discharged, and a default "
            "warp width would be a fabricated hardware fact")
    if lanes is None:
        raise ValueError("radiance contract declares no capabilities.simt.lanes_per_warp")
    return {"matmul": {"lanes_per_warp": IntegerAttr(int(lanes), i64)},
            "pack": {}, "commit": {}, "evict": {}}


def get_dialect() -> Dialect:
    return RADIANCE_DIALECT
