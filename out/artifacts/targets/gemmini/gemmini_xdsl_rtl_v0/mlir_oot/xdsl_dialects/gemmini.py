"""Verified high-level Gemmini target dialect used by the rewrite pass."""

from __future__ import annotations

from xdsl.dialects.builtin import ArrayAttr, StringAttr, TensorType
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    param_def,
    result_def,
)
from xdsl.utils.exceptions import VerifyException


@irdl_attr_definition
class ResidentTensorType(ParametrizedAttribute, TypeAttribute):
    name = "gemmini.resident_tensor"
    source: Attribute = param_def(Attribute)


@irdl_attr_definition
class AccumulatorType(ParametrizedAttribute, TypeAttribute):
    name = "gemmini.accumulator"
    element_type: Attribute = param_def(Attribute)


@irdl_op_definition
class TensorOp(IRDLOperation):
    name = "gemmini.tensor"
    logical_name = attr_def(StringAttr, attr_name="name")
    role = attr_def(StringAttr)
    result = result_def(TensorType)
    assembly_format = "attr-dict `:` type($result)"


@irdl_op_definition
class PackOp(IRDLOperation):
    name = "gemmini.pack"
    src = operand_def(TensorType)
    layout = attr_def(StringAttr)
    result = result_def(ResidentTensorType)
    assembly_format = "$src attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class MatmulOp(IRDLOperation):
    name = "gemmini.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(ResidentTensorType)
    result = result_def(AccumulatorType)
    assembly_format = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class CommitOp(IRDLOperation):
    name = "gemmini.commit"
    acc = operand_def(AccumulatorType)
    logical_name = attr_def(StringAttr, attr_name="name")
    epilogue = attr_def(ArrayAttr)
    output_dtype = attr_def(StringAttr)
    result = result_def(TensorType)
    assembly_format = "$acc attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class ReleaseOp(IRDLOperation):
    name = "gemmini.release"
    handle = operand_def(ResidentTensorType)
    assembly_format = "$handle attr-dict `:` `(` type($handle) `)` `->` `(` `)`"


@irdl_op_definition
class MovementOp(IRDLOperation):
    name = "gemmini.movement"
    src = operand_def(TensorType)
    logical_name = attr_def(StringAttr, attr_name="name")
    result = result_def(TensorType)
    assembly_format = "$src attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class Conv2DOp(IRDLOperation):
    name = "gemmini.conv2d"
    ifm = operand_def(TensorType)
    weight = operand_def(ResidentTensorType)
    result = result_def(TensorType)
    assembly_format = "$ifm `,` $weight attr-dict `:` functional-type(operands, results)"

    def verify_(self) -> None:
        if getattr(self.attributes.get("layout"), "data", None) != "nhwc":
            raise VerifyException("gemmini.conv2d requires NHWC")


@irdl_op_definition
class AttentionQKOp(IRDLOperation):
    name = "gemmini.attention_qk"
    q = operand_def(TensorType)
    k = operand_def(TensorType)
    result = result_def(TensorType)
    assembly_format = "$q `,` $k attr-dict `:` functional-type(operands, results)"


GEMMINI_DIALECT = Dialect(
    "gemmini",
    [TensorOp, PackOp, MatmulOp, CommitOp, ReleaseOp, MovementOp, Conv2DOp, AttentionQKOp],
    [ResidentTensorType, AccumulatorType],
)
