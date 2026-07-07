"""xDSL IRDL dialects for the Gemmini out-of-tree backend.

Two dialects, both authored here against ``xdsl`` directly (no ``merlin`` import):

* ``merlin_iface`` — the *input* dialect. A faithful in-memory model of the frozen
  ``merlin_iface`` v0.1 grammar (tensor / resident_pack / matmul / commit / evict, plus
  the ``movement`` and ``conv2d`` extension ops). Its verifiers enforce the contract's
  structural invariants (matmul inner-dim agreement, commit epilogue/dtype rules, …).

* ``gemmini`` — the *target* dialect. The recommended namespace from
  ``target_dialect_contract.yaml`` (pack / matmul / commit / release over
  ``!gemmini.resident_tensor`` and ``!gemmini.accumulator``), extended with
  ``gemmini.movement`` and ``gemmini.conv2d``.

The pattern (IRDL ops/types with ``verify_`` methods) mirrors the reference
``merlin.xdsl_dialects`` patterns we were given to study, but the code is our own and
imports only ``xdsl``.
"""
from __future__ import annotations

from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       operand_def, opt_prop_def, prop_def, result_def)
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import (ArrayAttr, FloatAttr, IntegerType, StringAttr,
                                   TensorType)

# Epilogue stages the COMMIT engine understands (command-buffer ABI).
KNOWN_EPILOGUE = {"bias", "bias_add", "requant", "acc_scale", "relu"}
KNOWN_OUTPUT_DTYPES = {"i8", "i16", "i32"}


def _tensor_of(t: Attribute):
    if isinstance(t, TensorType):
        return t
    inner = getattr(t, "element", None)
    return inner if isinstance(inner, TensorType) else None


# =========================================================================
# merlin_iface (input) dialect
# =========================================================================

@irdl_attr_definition
class IfaceResidentType(ParametrizedAttribute, TypeAttribute):
    """!merlin_iface.resident<tensor<...>> — opaque resident weight handle."""
    name = "merlin_iface.resident"
    element: Attribute


@irdl_attr_definition
class IfaceAccType(ParametrizedAttribute, TypeAttribute):
    """!merlin_iface.acc<tensor<...>> — opaque integer accumulator handle."""
    name = "merlin_iface.acc"
    element: Attribute


def _check_epilogue(op, stages_attr, dialect):
    stages = []
    for entry in stages_attr:
        stage = entry.data if isinstance(entry, StringAttr) else None
        if stage not in KNOWN_EPILOGUE:
            raise VerifyException(
                f"{dialect}.commit epilogue stage {stage!r} not in {sorted(KNOWN_EPILOGUE)}")
        stages.append(stage)
    return stages


@irdl_op_definition
class IfaceTensorOp(IRDLOperation):
    name = "merlin_iface.tensor"
    tname = prop_def(StringAttr, prop_name="tname")
    role = prop_def(StringAttr)
    res = result_def(TensorType)


@irdl_op_definition
class IfaceResidentPackOp(IRDLOperation):
    name = "merlin_iface.resident_pack"
    src = operand_def(TensorType)
    layout = prop_def(StringAttr)
    res = result_def(IfaceResidentType)


@irdl_op_definition
class IfaceMatmulOp(IRDLOperation):
    name = "merlin_iface.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(IfaceResidentType)
    acc = result_def(IfaceAccType)

    def verify_(self) -> None:
        lt = _tensor_of(self.lhs.type)
        rt = _tensor_of(self.rhs.type)
        if lt is not None and rt is not None:
            ls, rs = lt.get_shape(), rt.get_shape()
            if len(ls) == 2 and len(rs) == 2 and ls[1] != rs[0]:
                raise VerifyException(
                    f"merlin_iface.matmul inner dims disagree: {list(ls)} vs {list(rs)}")


@irdl_op_definition
class IfaceCommitOp(IRDLOperation):
    name = "merlin_iface.commit"
    acc = operand_def(IfaceAccType)
    tname = prop_def(StringAttr, prop_name="tname")
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def(TensorType)

    def verify_(self) -> None:
        stages = _check_epilogue(self, self.epilogue, "merlin_iface")
        if "acc_scale" in stages and self.acc_scale is None:
            raise VerifyException(
                "merlin_iface.commit epilogue has `acc_scale` but no acc_scale value")
        if self.output_dtype.data not in KNOWN_OUTPUT_DTYPES:
            raise VerifyException(
                f"merlin_iface.commit output_dtype {self.output_dtype.data!r} unknown")


@irdl_op_definition
class IfaceMovementOp(IRDLOperation):
    name = "merlin_iface.movement"
    src = operand_def(TensorType)
    tname = prop_def(StringAttr, prop_name="tname")
    res = result_def(TensorType)


@irdl_op_definition
class IfaceConv2dOp(IRDLOperation):
    name = "merlin_iface.conv2d"
    ifm = operand_def(TensorType)
    rhs = operand_def(IfaceResidentType)
    tname = prop_def(StringAttr, prop_name="tname")
    kernel = prop_def(ArrayAttr)
    stride = prop_def(ArrayAttr)
    padding = prop_def(ArrayAttr)
    dilation = prop_def(ArrayAttr)
    layout = prop_def(StringAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def(TensorType)


@irdl_op_definition
class IfaceEvictOp(IRDLOperation):
    name = "merlin_iface.evict"
    handle = operand_def(IfaceResidentType)


IFACE_DIALECT = Dialect(
    "merlin_iface",
    [IfaceTensorOp, IfaceResidentPackOp, IfaceMatmulOp, IfaceCommitOp,
     IfaceMovementOp, IfaceConv2dOp, IfaceEvictOp],
    [IfaceResidentType, IfaceAccType],
)


# =========================================================================
# gemmini (target) dialect
# =========================================================================

@irdl_attr_definition
class GemminiResidentTensorType(ParametrizedAttribute, TypeAttribute):
    """!gemmini.resident_tensor<tensor<...>> — packed weight resident in the array."""
    name = "gemmini.resident_tensor"
    element: Attribute


@irdl_attr_definition
class GemminiAccumulatorType(ParametrizedAttribute, TypeAttribute):
    """!gemmini.accumulator<tensor<...>> — i32 accumulator tile."""
    name = "gemmini.accumulator"
    element: Attribute


@irdl_op_definition
class GemminiTensorOp(IRDLOperation):
    name = "gemmini.tensor"
    tname = prop_def(StringAttr, prop_name="tname")
    role = prop_def(StringAttr)
    res = result_def(TensorType)


@irdl_op_definition
class GemminiPackOp(IRDLOperation):
    name = "gemmini.pack"
    src = operand_def(TensorType)
    layout = prop_def(StringAttr)
    res = result_def(GemminiResidentTensorType)


@irdl_op_definition
class GemminiMatmulOp(IRDLOperation):
    name = "gemmini.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(GemminiResidentTensorType)
    acc = result_def(GemminiAccumulatorType)


@irdl_op_definition
class GemminiCommitOp(IRDLOperation):
    name = "gemmini.commit"
    acc = operand_def(GemminiAccumulatorType)
    tname = prop_def(StringAttr, prop_name="tname")
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def(TensorType)

    def verify_(self) -> None:
        _check_epilogue(self, self.epilogue, "gemmini")


@irdl_op_definition
class GemminiMovementOp(IRDLOperation):
    name = "gemmini.movement"
    src = operand_def(TensorType)
    tname = prop_def(StringAttr, prop_name="tname")
    res = result_def(TensorType)


@irdl_op_definition
class GemminiConv2dOp(IRDLOperation):
    name = "gemmini.conv2d"
    ifm = operand_def(TensorType)
    rhs = operand_def(GemminiResidentTensorType)
    tname = prop_def(StringAttr, prop_name="tname")
    kernel = prop_def(ArrayAttr)
    stride = prop_def(ArrayAttr)
    padding = prop_def(ArrayAttr)
    dilation = prop_def(ArrayAttr)
    layout = prop_def(StringAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def(TensorType)


@irdl_op_definition
class GemminiReleaseOp(IRDLOperation):
    name = "gemmini.release"
    handle = operand_def(GemminiResidentTensorType)


GEMMINI_DIALECT = Dialect(
    "gemmini",
    [GemminiTensorOp, GemminiPackOp, GemminiMatmulOp, GemminiCommitOp,
     GemminiMovementOp, GemminiConv2dOp, GemminiReleaseOp],
    [GemminiResidentTensorType, GemminiAccumulatorType],
)
