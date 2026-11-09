"""Missing xDSL bindings for standard parallel tensor MLIR operations.

These are SCF/tensor operations, with no target mapping selected or custom
Merlin semantics. Native lowering remains responsible for complete legality.
"""

from xdsl.dialects.builtin import ArrayAttr, DYNAMIC_INDEX, DenseArrayBase, IndexType, TensorType, i64
from xdsl.irdl import (
    AttrSizedOperandSegments, IRDLOperation, irdl_op_definition, operand_def,
    opt_prop_def, prop_def, region_def, traits_def, var_operand_def, var_result_def,
)
from xdsl.traits import IsTerminator, NoTerminator
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class ParallelInsertSliceOp(IRDLOperation):
    name = "tensor.parallel_insert_slice"
    source = operand_def(TensorType)
    dest = operand_def(TensorType)
    offsets = var_operand_def(IndexType)
    sizes = var_operand_def(IndexType)
    strides = var_operand_def(IndexType)
    static_offsets = prop_def(DenseArrayBase.constr(i64))
    static_sizes = prop_def(DenseArrayBase.constr(i64))
    static_strides = prop_def(DenseArrayBase.constr(i64))
    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def verify_(self):
        rank = self.dest.type.get_num_dims()
        for static, dynamic in ((self.static_offsets, self.offsets), (self.static_sizes, self.sizes),
                                (self.static_strides, self.strides)):
            if len(static) != rank or sum(n == DYNAMIC_INDEX for n in static.iter_values()) != len(dynamic):
                raise VerifyException("parallel slice static/dynamic index counts must match destination rank")


@irdl_op_definition
class InParallelOp(IRDLOperation):
    name = "scf.forall.in_parallel"
    body = region_def("single_block")
    traits = traits_def(IsTerminator(), NoTerminator())


@irdl_op_definition
class ForallOp(IRDLOperation):
    name = "scf.forall"
    dynamicLowerBound = var_operand_def(IndexType)
    dynamicUpperBound = var_operand_def(IndexType)
    dynamicStep = var_operand_def(IndexType)
    outputs = var_operand_def(TensorType)
    staticLowerBound = prop_def(DenseArrayBase.constr(i64))
    staticUpperBound = prop_def(DenseArrayBase.constr(i64))
    staticStep = prop_def(DenseArrayBase.constr(i64))
    mapping = opt_prop_def(ArrayAttr)
    results_ = var_result_def(TensorType)
    body = region_def("single_block")
    irdl_options = (AttrSizedOperandSegments(as_property=True),)

    def verify_(self):
        rank = len(self.staticLowerBound)
        if not rank or len(self.staticUpperBound) != rank or len(self.staticStep) != rank:
            raise VerifyException("forall bounds and steps must have matching nonempty ranks")
        for static, dynamic in ((self.staticLowerBound, self.dynamicLowerBound),
                                (self.staticUpperBound, self.dynamicUpperBound), (self.staticStep, self.dynamicStep)):
            if sum(n == DYNAMIC_INDEX for n in static.iter_values()) != len(dynamic):
                raise VerifyException("forall static/dynamic bound counts differ")
        if any(n != DYNAMIC_INDEX and n <= 0 for n in self.staticStep.iter_values()):
            raise VerifyException("forall static steps must be positive")
        types = tuple(value.type for value in self.outputs)
        if tuple(value.type for value in self.results) != types:
            raise VerifyException("forall results must match shared output types")
        if tuple(arg.type for arg in self.body.block.args) != (IndexType(),) * rank + types:
            raise VerifyException("forall block arguments must be indices followed by shared outputs")
        if not isinstance(self.body.block.last_op, InParallelOp):
            raise VerifyException("forall must end in scf.forall.in_parallel")


def register_parallel_tensor_ops(context):
    from xdsl.dialects.scf import Scf
    from xdsl.dialects.tensor import Tensor

    upstream = {op.name for dialect in (Scf, Tensor) for op in dialect.operations}
    for op in (ForallOp, InParallelOp, ParallelInsertSliceOp):
        if op.name not in upstream:
            context.load_op(op)
