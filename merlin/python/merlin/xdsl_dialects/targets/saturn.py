"""The in-tree reference ``saturn`` target dialect (xDSL).

Saturn (the chipyard RVV vector unit) modeled as a multicore RV64GCV CPU. Op/type
names mirror ``merlin/targets/saturn/contracts/dialect_plan.yaml``: pack / matmul /
commit / release over packed_tensor / accumulator. "Residency" is a packed weight
kept live in memory across the region; the runtime backend executes matmuls with the
hand-written RVV kernel, row-partitioned across harts.
"""
from __future__ import annotations

from .._common import HAS_XDSL

DIALECT_NAME = "saturn"
OPS = ["pack", "matmul", "commit", "release"]
TYPES = ["packed_tensor", "accumulator"]

# Must stay aligned with interface.KNOWN_EPILOGUE and the runtime engine.
KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "relu"}

if HAS_XDSL:
    from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_prop_def, prop_def, result_def)
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    @irdl_attr_definition
    class PackedTensorType(ParametrizedAttribute, TypeAttribute):
        """!saturn.packed_tensor<element_type> — packed weight live in memory."""
        name = "saturn.packed_tensor"
        element_type: Attribute

    @irdl_attr_definition
    class AccumulatorType(ParametrizedAttribute, TypeAttribute):
        """!saturn.accumulator<element_type> — e32 vector accumulator tile."""
        name = "saturn.accumulator"
        element_type: Attribute

    @irdl_op_definition
    class PackOp(IRDLOperation):
        """saturn.pack — layout-pack an immutable RHS (no-cost copy on CPU).

        Source abstraction: interface.resident_pack. Runtime: RES_PACK command
        (counted layout/budget event; data already in memory).
        """
        name = "saturn.pack"
        src = operand_def()
        layout = prop_def(StringAttr)
        res = result_def(PackedTensorType)

    @irdl_op_definition
    class MatmulOp(IRDLOperation):
        """saturn.matmul — RVV widening matmul -> accumulator.

        Source abstraction: interface.matmul. The RHS may be a packed tensor
        (resident across the region -> MATMUL_RESIDENT command) or a plain tensor
        re-fetched from memory each dispatch (-> MATMUL command); a CPU supports
        both, unlike the toy NPU. Executed by merlin_rvv_matmul_i8 (e8m1 loads ->
        vsext.vf4 -> e32m4 vmacc), rows partitioned across harts.
        """
        name = "saturn.matmul"
        lhs = operand_def()
        rhs = operand_def()
        vl_policy = opt_prop_def(StringAttr)
        acc = result_def(AccumulatorType)

    @irdl_op_definition
    class CommitOp(IRDLOperation):
        """saturn.commit — apply epilogue and commit an accumulator to a tensor.

        Source abstraction: interface.commit. Runtime: COMMIT command (epilogue
        stages in order, then output dtype conversion; semantics == engine).
        """
        name = "saturn.commit"
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
                        "saturn.commit epilogue stage %r not in %s"
                        % (stage, sorted(KNOWN_EPILOGUE)))
                stages.append(stage)
            if ("bias" in stages or "bias_add" in stages) and self.bias is None:
                raise VerifyException(
                    "saturn.commit epilogue has a bias stage but no `bias` tensor name")

    @irdl_op_definition
    class ReleaseOp(IRDLOperation):
        """saturn.release — release the packed tensor's working-set budget.

        Source abstraction: interface.resident_evict. Runtime: EVICT command.
        """
        name = "saturn.release"
        handle = operand_def(PackedTensorType)

    _OP_CLASSES = [PackOp, MatmulOp, CommitOp, ReleaseOp]
    _ATTR_CLASSES = [PackedTensorType, AccumulatorType]
    SATURN_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return SATURN_DIALECT

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None
