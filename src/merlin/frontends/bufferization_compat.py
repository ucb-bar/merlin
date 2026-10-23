"""Bufferization tensor/memref correspondence without a fictitious layout tie.

A tensor fixes shape and element type, but carries no memref layout or memory
space. The upstream xDSL constraint infers an identity layout from that tensor,
rejecting legal dynamic strided buffers before host lowering can run.
"""

from xdsl.dialects.builtin import MemRefType, TensorType, UnitAttr, UnrankedMemRefType, UnrankedTensorType
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, result_def
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ToBufferOp(IRDLOperation):
    name = "bufferization.to_buffer"
    tensor = operand_def(TensorType | UnrankedTensorType)
    memref = result_def(MemRefType | UnrankedMemRefType)
    read_only = opt_prop_def(UnitAttr)
    assembly_format = "$tensor (`read_only` $read_only^)? `:` attr-dict type($tensor) `to` type($memref)"

    def verify_(self):
        source, result = self.tensor.type, self.memref.type
        ranked = isinstance(source, TensorType) and isinstance(result, MemRefType)
        unranked = isinstance(source, UnrankedTensorType) and isinstance(result, UnrankedMemRefType)
        if (
            source.element_type != result.element_type
            or not (ranked or unranked)
            or (ranked and source.get_shape() != result.get_shape())
        ):
            raise VerifyException("to_buffer requires matching tensor/memref shape and element type")


def dialect():
    from xdsl.dialects.bufferization import Bufferization

    return Dialect(
        "bufferization",
        [ToBufferOp if op.name == ToBufferOp.name else op for op in Bufferization.operations],
        list(Bufferization.attributes),
    )
