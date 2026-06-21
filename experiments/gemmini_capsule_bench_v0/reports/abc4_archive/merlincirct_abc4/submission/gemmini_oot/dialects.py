"""xDSL dialects for the Gemmini OOT backend.

Two dialects:

* ``merlin_iface`` — the input dialect (mirrors the frozen interface grammar). Built by
  :mod:`parse_iface` from the ``*.interface.mlir`` text, then ``verify()``-ed (the ``parse``
  entrypoint).
* ``gemmini`` — the target dialect. Produced by the rewrite pass in :mod:`lower` and the
  source for both the command buffer (:mod:`cbuf`) and the RoCC kernel (:mod:`kernel`).

Numeric shape/dtype info travels on the builtin ``tensor`` result types; epilogue / conv
attributes travel as op properties so the downstream emitters are pure functions of the IR.
"""
from __future__ import annotations

from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       operand_def, opt_prop_def, prop_def, result_def)
from xdsl.utils.exceptions import VerifyException
from xdsl.dialects.builtin import (ArrayAttr, DictionaryAttr, FloatAttr, IntegerAttr,
                                   StringAttr, TensorType)

KNOWN_EPILOGUE = {"bias_add", "bias", "requant", "acc_scale", "relu"}
KNOWN_OUTPUT_DTYPES = {"i8", "i16", "i32"}


# --------------------------------------------------------------------------- iface types

@irdl_attr_definition
class ResidentType(ParametrizedAttribute, TypeAttribute):
    name = "merlin_iface.resident"


@irdl_attr_definition
class AccType(ParametrizedAttribute, TypeAttribute):
    name = "merlin_iface.acc"


@irdl_op_definition
class IfaceTensorOp(IRDLOperation):
    name = "merlin_iface.tensor"
    sym = prop_def(StringAttr)
    role = prop_def(StringAttr)
    res = result_def(TensorType)

    def verify_(self) -> None:
        if self.role.data not in ("weight", "input", "bias"):
            raise VerifyException(f"merlin_iface.tensor bad role {self.role.data!r}")


@irdl_op_definition
class IfaceResidentPackOp(IRDLOperation):
    name = "merlin_iface.resident_pack"
    src = operand_def(TensorType)
    sym = prop_def(StringAttr)
    layout = prop_def(StringAttr)
    res = result_def(ResidentType)


@irdl_op_definition
class IfaceMatmulOp(IRDLOperation):
    name = "merlin_iface.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(ResidentType)
    sym = prop_def(StringAttr)
    res = result_def(AccType)


@irdl_op_definition
class IfaceCommitOp(IRDLOperation):
    name = "merlin_iface.commit"
    acc = operand_def(AccType)
    sym = prop_def(StringAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def(TensorType)

    def verify_(self) -> None:
        for e in self.epilogue:
            if not isinstance(e, StringAttr) or e.data not in KNOWN_EPILOGUE:
                raise VerifyException(f"commit epilogue stage {e} unknown")
        if self.output_dtype.data not in KNOWN_OUTPUT_DTYPES:
            raise VerifyException(f"commit output_dtype {self.output_dtype.data!r} unknown")
        if any(isinstance(e, StringAttr) and e.data == "acc_scale" for e in self.epilogue) \
                and self.acc_scale is None:
            raise VerifyException("commit has acc_scale epilogue but no acc_scale value")


@irdl_op_definition
class IfaceMoveOp(IRDLOperation):
    name = "merlin_iface.movement"
    src = operand_def(TensorType)
    sym = prop_def(StringAttr)
    res = result_def(TensorType)


@irdl_op_definition
class IfaceConvOp(IRDLOperation):
    name = "merlin_iface.conv2d"
    ifm = operand_def(TensorType)
    rhs = operand_def(ResidentType)
    sym = prop_def(StringAttr)
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
    handle = operand_def(ResidentType)


IFACE_DIALECT = Dialect("merlin_iface", [
    IfaceTensorOp, IfaceResidentPackOp, IfaceMatmulOp, IfaceCommitOp, IfaceMoveOp,
    IfaceConvOp, IfaceEvictOp], [ResidentType, AccType])


# --------------------------------------------------------------------------- gemmini types

@irdl_attr_definition
class GResidentType(ParametrizedAttribute, TypeAttribute):
    name = "gemmini.resident"


@irdl_attr_definition
class GAccType(ParametrizedAttribute, TypeAttribute):
    name = "gemmini.acc"


@irdl_op_definition
class GTensorOp(IRDLOperation):
    """Leaf tensor in the command buffer's resource table (materialized by name)."""
    name = "gemmini.tensor"
    sym = prop_def(StringAttr)
    role = prop_def(StringAttr)
    res = result_def(TensorType)


@irdl_op_definition
class GPackOp(IRDLOperation):
    """RES_PACK — install a weight as the resident (stationary) operand."""
    name = "gemmini.pack"
    src = operand_def(TensorType)
    sym = prop_def(StringAttr)
    layout = prop_def(StringAttr)
    res = result_def(GResidentType)


@irdl_op_definition
class GMatmulOp(IRDLOperation):
    """MATMUL_RESIDENT — acc = lhs @ resolve(rhs). ``im2col`` present iff conv-derived."""
    name = "gemmini.matmul"
    lhs = operand_def(TensorType)
    rhs = operand_def(GResidentType)
    sym = prop_def(StringAttr)
    im2col = opt_prop_def(DictionaryAttr)
    res = result_def(GAccType)


@irdl_op_definition
class GCommitOp(IRDLOperation):
    """COMMIT — epilogue + dtype cast, accumulator -> output tensor."""
    name = "gemmini.commit"
    acc = operand_def(GAccType)
    sym = prop_def(StringAttr)
    epilogue = prop_def(ArrayAttr)
    output_dtype = prop_def(StringAttr)
    acc_scale = opt_prop_def(FloatAttr)
    res = result_def(TensorType)


@irdl_op_definition
class GMoveOp(IRDLOperation):
    """VECTOR_MAP(identity) — pure data movement (mvin/mvout copy)."""
    name = "gemmini.move"
    src = operand_def(TensorType)
    sym = prop_def(StringAttr)
    res = result_def(TensorType)


@irdl_op_definition
class GReleaseOp(IRDLOperation):
    """EVICT — release the resident weight."""
    name = "gemmini.release"
    handle = operand_def(GResidentType)


GEMMINI_DIALECT = Dialect("gemmini", [
    GTensorOp, GPackOp, GMatmulOp, GCommitOp, GMoveOp, GReleaseOp],
    [GResidentType, GAccType])
