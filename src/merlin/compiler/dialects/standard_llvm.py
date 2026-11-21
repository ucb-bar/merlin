"""Standard volatile memory fields; native LLVM owns their actual semantics."""

from xdsl.dialects import builtin, llvm
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, prop_def, result_def


@irdl_op_definition
class LoadOp(IRDLOperation):
    name = "llvm.load"
    ptr = operand_def(llvm.LLVMPointerType)
    dereferenced_value = result_def()
    alignment = opt_prop_def(builtin.IntegerAttr)
    ordering = prop_def(builtin.IntegerAttr, default_value=builtin.IntegerAttr(0, builtin.i64))
    volatile_ = opt_prop_def(builtin.UnitAttr)


def llvm_memory_dialect():
    """Register only the portable load/pointer leaf, without target intrinsics."""
    return Dialect("llvm", [LoadOp, llvm.IntToPtrOp], [llvm.LLVMPointerType])
