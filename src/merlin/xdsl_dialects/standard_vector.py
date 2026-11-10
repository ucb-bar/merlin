"""Complete standard scalar/string vector.print binding for xDSL 0.68.

Fields and invariants follow LLVM VectorOps.td at a47bddccec30255619bb8c37fa59700e661d4e66.
The installed xDSL binding requires one operand and omits strings/punctuation.
"""

from enum import StrEnum

from xdsl.dialects import builtin
from xdsl.ir import Dialect, EnumAttribute
from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                       opt_operand_def, opt_prop_def, traits_def)
from xdsl.traits import MemoryWriteEffect
from xdsl.utils.exceptions import VerifyException


class Punctuation(StrEnum):
    NONE = "no_punctuation"
    NEWLINE = "newline"
    OPEN = "open"
    CLOSE = "close"
    COMMA = "comma"


@irdl_attr_definition
class PunctuationAttr(EnumAttribute[Punctuation]):
    name = "vector.punctuation"

    def print_parameter(self, printer):
        with printer.in_angle_brackets():
            printer.print_string(self.data.value)

    @classmethod
    def parse_parameter(cls, parser):
        with parser.in_angle_brackets():
            return parser.parse_str_enum(Punctuation)


@irdl_op_definition
class PrintOp(IRDLOperation):
    name = "vector.print"
    source = opt_operand_def()
    punctuation = opt_prop_def(PunctuationAttr)
    stringLiteral = opt_prop_def(builtin.StringAttr)
    traits = traits_def(MemoryWriteEffect())

    def verify_(self):
        if self.stringLiteral is not None:
            if self.source is not None or self.punctuation is not None and self.punctuation.data != Punctuation.NEWLINE:
                raise VerifyException("standard string print cannot also have a value or nondefault punctuation")
        if self.source is not None and not isinstance(self.source.type,
                (builtin.IntegerType,builtin.IndexType,builtin.AnyFloat,builtin.VectorType)):
            raise VerifyException("standard print requires a numeric scalar or vector")


def literal(value):
    return PrintOp.build(operands=[[]],properties={"stringLiteral":builtin.StringAttr(value)})


def scalar(value):
    return PrintOp.build(operands=[[value]],properties={"punctuation":PunctuationAttr(Punctuation.NONE)})


def vector_dialect():
    return Dialect("vector",[PrintOp],[PunctuationAttr])
