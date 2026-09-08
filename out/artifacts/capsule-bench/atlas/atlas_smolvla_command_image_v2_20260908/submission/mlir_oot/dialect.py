"""Typed Atlas target dialect used as the lowering IR."""
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, opt_attr_def


def _op(mnemonic):
    @irdl_op_definition
    class AtlasOp(IRDLOperation):
        name = "atlas." + mnemonic
        detail = opt_attr_def(StringAttr)
        assembly_format = "attr-dict"
    return AtlasOp


DmaLoadOp = _op("dma_load")
TensorLoadOp = _op("tensor_load")
TransposeOp = _op("transpose")
WeightPushOp = _op("weight_push")
MatmulOp = _op("matmul")
AccumulatorPopOp = _op("accumulator_pop")
VectorUnaryOp = _op("vector_unary")
VectorBinaryOp = _op("vector_binary")
TensorStoreOp = _op("tensor_store")
DmaStoreOp = _op("dma_store")
HaltOp = _op("halt")

ATLAS_DIALECT = Dialect("atlas", [DmaLoadOp, TensorLoadOp, TransposeOp, WeightPushOp, MatmulOp,
                                   AccumulatorPopOp, VectorUnaryOp, VectorBinaryOp, TensorStoreOp,
                                   DmaStoreOp, HaltOp], [])
