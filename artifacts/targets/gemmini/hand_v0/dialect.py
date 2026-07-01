"""Isolated Gemmini target dialect (xDSL PROTOTYPE plane) — run hand_v0.

Self-contained: loaded dynamically by merlin.targetgen.registry.load_target, NOT imported
from the core merlin tree and NOT hardcoded in any shared lowering table. The compiler-level
abstraction (pack/matmul/commit/release over resident_tensor/accumulator) — NOT the ISA;
the mvin/preload/compute/mvout mapping lives in the codegen.

This is the xDSL prototype; once an architecture is settled, the package can EXPORT C++/ODS
(TableGen) into artifacts/ as the stable final form.
"""
from __future__ import annotations

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       operand_def, opt_prop_def, prop_def, result_def)
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

DIALECT_NAME = "gemmini"
OPS = ["pack", "matmul", "commit", "release"]
TYPES = ["resident_tensor", "accumulator"]
KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "relu"}


@irdl_attr_definition
class ResidentTensorType(ParametrizedAttribute, TypeAttribute):
    name = "gemmini.resident_tensor"
    element_type: Attribute


@irdl_attr_definition
class AccumulatorType(ParametrizedAttribute, TypeAttribute):
    name = "gemmini.accumulator"
    element_type: Attribute


@irdl_op_definition
class PackOp(IRDLOperation):
    name = "gemmini.pack"
    src = operand_def()
    layout = prop_def(StringAttr)
    res = result_def(ResidentTensorType)


@irdl_op_definition
class MatmulOp(IRDLOperation):
    name = "gemmini.matmul"
    lhs = operand_def()
    rhs = operand_def()
    vl_policy = opt_prop_def(StringAttr)
    acc = result_def(AccumulatorType)


@irdl_op_definition
class CommitOp(IRDLOperation):
    name = "gemmini.commit"
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
                    "gemmini.commit epilogue stage %r not in %s"
                    % (stage, sorted(KNOWN_EPILOGUE)))
            stages.append(stage)
        if ("bias" in stages or "bias_add" in stages) and self.bias is None:
            raise VerifyException("gemmini.commit bias stage but no `bias` tensor name")


@irdl_op_definition
class ReleaseOp(IRDLOperation):
    name = "gemmini.release"
    handle = operand_def(ResidentTensorType)


GEMMINI_DIALECT = Dialect(DIALECT_NAME, [PackOp, MatmulOp, CommitOp, ReleaseOp],
                          [ResidentTensorType, AccumulatorType])

# The registry builds a TargetSpec from this mapping (decoupled from the core's _specs()).
SPEC_OPS = {"pack": PackOp, "matmul": MatmulOp, "commit": CommitOp, "evict": ReleaseOp,
            "resident_type": ResidentTensorType, "accumulator_type": AccumulatorType}


def get_dialect() -> Dialect:
    return GEMMINI_DIALECT
