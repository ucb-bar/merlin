"""Typed xDSL definition of frozen ``merlin_iface`` grammar v0.1."""

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
class ResidentType(ParametrizedAttribute, TypeAttribute):
    name = "merlin_iface.resident"


@irdl_attr_definition
class AccType(ParametrizedAttribute, TypeAttribute):
    name = "merlin_iface.acc"
    element_type: Attribute = param_def(Attribute)


@irdl_op_definition
class TensorOp(IRDLOperation):
    name = "merlin_iface.tensor"
    logical_name = attr_def(StringAttr, attr_name="name")
    role = attr_def(StringAttr)
    result = result_def(TensorType)
    assembly_format = "attr-dict `:` type($result)"

    def verify_(self) -> None:
        if not self.logical_name.data:
            raise VerifyException("merlin_iface.tensor name must be nonempty")
        if self.role.data not in ("input", "weight", "bias"):
            raise VerifyException(f"unsupported tensor role {self.role.data!r}")
        if not self.result.type.get_shape() or any(d <= 0 for d in self.result.type.get_shape()):
            raise VerifyException("tensor shape must have positive static extents")


@irdl_op_definition
class ResidentPackOp(IRDLOperation):
    name = "merlin_iface.resident_pack"
    src = operand_def(TensorType)
    layout = attr_def(StringAttr)
    result = result_def(ResidentType)
    assembly_format = "$src attr-dict `:` functional-type(operands, results)"

    def verify_(self) -> None:
        if self.layout.data not in ("packed_rhs", "packed_conv_rhs"):
            raise VerifyException(f"unsupported resident layout {self.layout.data!r}")


@irdl_op_definition
class MatmulOp(IRDLOperation):
    name = "merlin_iface.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(ResidentType)
    result = result_def(AccType)
    assembly_format = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class CommitOp(IRDLOperation):
    name = "merlin_iface.commit"
    acc = operand_def(AccType)
    logical_name = attr_def(StringAttr, attr_name="name")
    epilogue = attr_def(ArrayAttr)
    output_dtype = attr_def(StringAttr)
    result = result_def(TensorType)
    assembly_format = "$acc attr-dict `:` functional-type(operands, results)"

    def verify_(self) -> None:
        stages = [getattr(stage, "data", None) for stage in self.epilogue]
        allowed = {"bias_add", "requant", "acc_scale", "relu", "maxpool"}
        if any(stage not in allowed for stage in stages):
            raise VerifyException(f"unsupported commit epilogue {stages}")
        if "acc_scale" in stages and "acc_scale" not in self.attributes:
            raise VerifyException("acc_scale epilogue requires acc_scale attribute")
        if "maxpool" in stages:
            missing = [k for k in ("pool_in_dims", "pool_size", "pool_stride") if k not in self.attributes]
            if missing:
                raise VerifyException("maxpool epilogue missing " + ", ".join(missing))
        if str(self.result.type.element_type) != self.output_dtype.data:
            raise VerifyException("commit output_dtype disagrees with tensor element type")


@irdl_op_definition
class EvictOp(IRDLOperation):
    name = "merlin_iface.evict"
    handle = operand_def(ResidentType)
    assembly_format = "$handle attr-dict `:` `(` type($handle) `)` `->` `(` `)`"


@irdl_op_definition
class MovementOp(IRDLOperation):
    name = "merlin_iface.movement"
    src = operand_def(TensorType)
    logical_name = attr_def(StringAttr, attr_name="name")
    result = result_def(TensorType)
    assembly_format = "$src attr-dict `:` functional-type(operands, results)"


@irdl_op_definition
class Conv2DOp(IRDLOperation):
    name = "merlin_iface.conv2d"
    ifm = operand_def(TensorType)
    weight = operand_def(ResidentType)
    logical_name = attr_def(StringAttr, attr_name="name")
    kernel = attr_def(ArrayAttr)
    stride = attr_def(ArrayAttr)
    padding = attr_def(ArrayAttr)
    dilation = attr_def(ArrayAttr)
    layout = attr_def(StringAttr)
    epilogue = attr_def(ArrayAttr)
    output_dtype = attr_def(StringAttr)
    result = result_def(TensorType)
    assembly_format = "$ifm `,` $weight attr-dict `:` functional-type(operands, results)"

    def verify_(self) -> None:
        if self.layout.data != "nhwc":
            raise VerifyException("Gemmini contract only defines NHWC conv2d")
        if len(self.kernel) != 4 or len(self.stride) != 2 or len(self.padding) != 4 or len(self.dilation) != 2:
            raise VerifyException("conv2d geometry has invalid arity")


@irdl_op_definition
class AttentionQKOp(IRDLOperation):
    name = "merlin_iface.attention_qk"
    q = operand_def(TensorType)
    k = operand_def(TensorType)
    logical_name = attr_def(StringAttr, attr_name="name")
    output_dtype = attr_def(StringAttr)
    result = result_def(TensorType)
    assembly_format = "$q `,` $k attr-dict `:` functional-type(operands, results)"

    def verify_(self) -> None:
        qs, ks = self.q.type.get_shape(), self.k.type.get_shape()
        if len(qs) != 2 or len(ks) != 2 or qs[1] != ks[1]:
            raise VerifyException("attention_qk contracts equal trailing dimensions")


MERLIN_IFACE_DIALECT = Dialect(
    "merlin_iface",
    [TensorOp, ResidentPackOp, MatmulOp, CommitOp, EvictOp, MovementOp, Conv2DOp, AttentionQKOp],
    [ResidentType, AccType],
)
