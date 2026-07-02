"""xDSL dialects for the Gemmini OOT backend.

Two IRDL dialects, authored with the xDSL framework (mirroring the idioms in
merlin's ``xdsl_dialects/targets/toynpu.py`` and ``interface.py`` — IRDL
op/type definitions, field-annotation attribute params, ``verify_`` checks):

* ``merlin_iface``  — the frozen input grammar (v0.1), parsed from the capsule.
* ``gemmini``       — our target dialect; the lowering pass rewrites
                      ``merlin_iface`` into it, and the command-buffer / RoCC
                      emitters walk it.

No ``merlin`` import: only the public ``xdsl`` library is used.
"""
from __future__ import annotations

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       operand_def, opt_prop_def, prop_def, result_def)
from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntegerAttr, StringAttr
from xdsl.utils.exceptions import VerifyException

KNOWN_EPILOGUE = {"bias_add", "bias", "requant", "acc_scale", "relu"}


# --------------------------------------------------------------------------- #
# merlin_iface (input) dialect
# --------------------------------------------------------------------------- #
@irdl_attr_definition
class ResidentType(ParametrizedAttribute, TypeAttribute):
    """!merlin_iface.resident — opaque handle to a packed/stationary weight."""
    name = "merlin_iface.resident"


@irdl_attr_definition
class AccType(ParametrizedAttribute, TypeAttribute):
    """!merlin_iface.acc<eltype> — opaque integer accumulator handle."""
    name = "merlin_iface.acc"
    element_type: Attribute


@irdl_op_definition
class IfTensorOp(IRDLOperation):
    name = "merlin_iface.tensor"
    tname = prop_def(StringAttr)
    role = prop_def(StringAttr)
    res = result_def()


@irdl_op_definition
class IfResidentPackOp(IRDLOperation):
    name = "merlin_iface.resident_pack"
    src = operand_def()
    layout = prop_def(StringAttr)
    res = result_def(ResidentType)


@irdl_op_definition
class IfMatmulOp(IRDLOperation):
    name = "merlin_iface.matmul"
    lhs = operand_def()
    rhs = operand_def(ResidentType)
    acc = result_def(AccType)


@irdl_op_definition
class IfMovementOp(IRDLOperation):
    name = "merlin_iface.movement"
    src = operand_def()
    tname = prop_def(StringAttr)
    res = result_def()


@irdl_op_definition
class IfConv2dOp(IRDLOperation):
    name = "merlin_iface.conv2d"
    ifm = operand_def()
    weight = operand_def(ResidentType)
    tname = prop_def(StringAttr)
    kernel = prop_def(ArrayAttr)
    stride = prop_def(ArrayAttr)
    padding = prop_def(ArrayAttr)
    dilation = prop_def(ArrayAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    layout = prop_def(StringAttr)
    res = result_def()


@irdl_op_definition
class IfCommitOp(IRDLOperation):
    name = "merlin_iface.commit"
    acc = operand_def(AccType)
    tname = prop_def(StringAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def()

    def verify_(self) -> None:
        stages = [e.data for e in self.epilogue if isinstance(e, StringAttr)]
        for s in stages:
            if s not in KNOWN_EPILOGUE:
                raise VerifyException(f"commit epilogue stage {s!r} unknown")
        if "acc_scale" in stages and self.acc_scale is None:
            raise VerifyException("commit epilogue has acc_scale but no acc_scale attr")


@irdl_op_definition
class IfEvictOp(IRDLOperation):
    name = "merlin_iface.evict"
    handle = operand_def(ResidentType)


IFACE_DIALECT = Dialect(
    "merlin_iface",
    [IfTensorOp, IfResidentPackOp, IfMatmulOp, IfMovementOp, IfConv2dOp,
     IfCommitOp, IfEvictOp],
    [ResidentType, AccType],
)


# --------------------------------------------------------------------------- #
# gemmini (target) dialect
# --------------------------------------------------------------------------- #
@irdl_attr_definition
class ResidentTensorType(ParametrizedAttribute, TypeAttribute):
    """!gemmini.resident_tensor — weight resident in the systolic array path."""
    name = "gemmini.resident_tensor"


@irdl_attr_definition
class AccumulatorType(ParametrizedAttribute, TypeAttribute):
    """!gemmini.accumulator — int32 accumulator state."""
    name = "gemmini.accumulator"


@irdl_op_definition
class GPackOp(IRDLOperation):
    name = "gemmini.pack"
    src = operand_def()
    layout = prop_def(StringAttr)
    res = result_def(ResidentTensorType)


@irdl_op_definition
class GMatmulOp(IRDLOperation):
    name = "gemmini.matmul"
    lhs = operand_def()
    rhs = operand_def(ResidentTensorType)
    acc = result_def(AccumulatorType)


@irdl_op_definition
class GConvOp(IRDLOperation):
    name = "gemmini.conv2d"
    ifm = operand_def()
    weight = operand_def(ResidentTensorType)
    tname = prop_def(StringAttr)
    kernel = prop_def(ArrayAttr)
    stride = prop_def(ArrayAttr)
    padding = prop_def(ArrayAttr)
    dilation = prop_def(ArrayAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    layout = prop_def(StringAttr)
    res = result_def()


@irdl_op_definition
class GMovementOp(IRDLOperation):
    name = "gemmini.movement"
    src = operand_def()
    tname = prop_def(StringAttr)
    res = result_def()


@irdl_op_definition
class GCommitOp(IRDLOperation):
    name = "gemmini.commit"
    acc = operand_def(AccumulatorType)
    tname = prop_def(StringAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def()


@irdl_op_definition
class GReleaseOp(IRDLOperation):
    name = "gemmini.release"
    handle = operand_def(ResidentTensorType)


GEMMINI_DIALECT = Dialect(
    "gemmini",
    [GPackOp, GMatmulOp, GConvOp, GMovementOp, GCommitOp, GReleaseOp],
    [ResidentTensorType, AccumulatorType],
)
