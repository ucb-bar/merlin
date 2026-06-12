"""The in-tree reference ``toynpu`` target dialect (xDSL).

Mirrors the TargetGen-generated dialect (op/type names from
``merlin/targets/toy_npu/contracts/dialect_plan.yaml``): res_pack / matmul / commit /
evict over resident_tensor / accumulator. The core pipeline lowers ``interface`` ops
into these; ``runtime`` lowering then encodes them as command-buffer appends.
"""
from __future__ import annotations

from .._common import HAS_XDSL

DIALECT_NAME = "toynpu"
OPS = ["res_pack", "matmul", "commit", "evict"]
TYPES = ["resident_tensor", "accumulator"]

# Must stay aligned with interface.KNOWN_EPILOGUE and the runtime engine.
KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "relu"}

if HAS_XDSL:
    from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_prop_def, prop_def, result_def)
    from xdsl.utils.exceptions import VerifyException
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    @irdl_attr_definition
    class ResidentTensorType(ParametrizedAttribute, TypeAttribute):
        """!toynpu.resident_tensor<element_type> — resident in target storage."""
        name = "toynpu.resident_tensor"
        element_type: Attribute

    @irdl_attr_definition
    class AccumulatorType(ParametrizedAttribute, TypeAttribute):
        """!toynpu.accumulator<element_type> — uncommitted accumulation state."""
        name = "toynpu.accumulator"
        element_type: Attribute

    @irdl_op_definition
    class ResPackOp(IRDLOperation):
        """toynpu.res_pack — pack + make an (immutable) RHS resident.

        Source abstraction: interface.resident_pack. Runtime: RES_PACK command.
        """
        name = "toynpu.res_pack"
        src = operand_def()
        layout = prop_def(StringAttr)
        res = result_def(ResidentTensorType)

    @irdl_op_definition
    class MatmulOp(IRDLOperation):
        """toynpu.matmul — matmul against a resident tensor -> accumulator.

        Source abstraction: interface.matmul. Runtime: MATMUL_RESIDENT command.
        """
        name = "toynpu.matmul"
        lhs = operand_def()
        rhs = operand_def(ResidentTensorType)
        acc = result_def(AccumulatorType)

    @irdl_op_definition
    class CommitOp(IRDLOperation):
        """toynpu.commit — apply epilogue and commit an accumulator to a tensor.

        Source abstraction: interface.commit. Runtime: COMMIT command (epilogue stages
        run in order, then the output dtype conversion).
        """
        name = "toynpu.commit"
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
                        "toynpu.commit epilogue stage %r not in %s"
                        % (stage, sorted(KNOWN_EPILOGUE)))
                stages.append(stage)
            if ("bias" in stages or "bias_add" in stages) and self.bias is None:
                raise VerifyException(
                    "toynpu.commit epilogue has a bias stage but no `bias` tensor name")

    @irdl_op_definition
    class EvictOp(IRDLOperation):
        """toynpu.evict — free resident storage.

        Source abstraction: interface.resident_evict. Runtime: EVICT command.
        """
        name = "toynpu.evict"
        handle = operand_def(ResidentTensorType)

    _OP_CLASSES = [ResPackOp, MatmulOp, CommitOp, EvictOp]
    _ATTR_CLASSES = [ResidentTensorType, AccumulatorType]
    TOYNPU_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return TOYNPU_DIALECT

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None
