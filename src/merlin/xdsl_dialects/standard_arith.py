"""Scalar bindings for standard unsigned index casts.

LLVM a47bddccec30 ArithOps.td and Arith ops/invalid tests define the
zero-extension/truncation and optional nneg contract. Native MLIR owns
execution; this binding supplies no range or target index-width proof.
"""

from xdsl.dialects.builtin import IndexType, IntegerType, Signedness, UnitAttr
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, result_def, traits_def
from xdsl.traits import Pure
from xdsl.utils.exceptions import VerifyException


@irdl_op_definition
class IndexCastUIOp(IRDLOperation):
    name = "arith.index_castui"
    argument = operand_def()
    converted = result_def()
    nonNeg = opt_prop_def(UnitAttr)
    traits = traits_def(Pure())
    assembly_format = "$argument (`nneg` $nonNeg^)? attr-dict `:` type($argument) `to` type($converted)"

    def __init__(self, value, destination, *, non_negative=False):
        super().__init__(operands=[value], result_types=[destination],
                         properties={"nonNeg": UnitAttr()} if non_negative else {})

    def verify_(self):
        types = (self.argument.type, self.converted.type)
        if sum(isinstance(value, IndexType) for value in types) != 1:
            raise VerifyException("arith.index_castui requires exactly one scalar index type")
        integer = next(value for value in types if not isinstance(value, IndexType))
        if (not isinstance(integer, IntegerType) or integer.bitwidth < 1
                or integer.signedness.data != Signedness.SIGNLESS):
            raise VerifyException("arith.index_castui requires a nonzero signless integer")


def register_unsigned_index_cast(context):
    context.load_op(IndexCastUIOp)
