"""Retain the standard named-matmul cast property missing from xDSL 0.68.

Fields follow native LinalgStructuredOps.td at
a47bddccec30255619bb8c37fa59700e661d4e66. Native parsing owns implicit
mixed-type regions except the checked signed i8-to-i32 implicit accumulator
body retained from the frontend. Explicit regions remain the caller's computation.
"""

from enum import StrEnum
from functools import cache

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, ShapedType, TensorType, i8, i32
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.linalg.abstract_ops import NamedOperation
from xdsl.dialects.linalg.ops import MatmulOp as XDSLMatmulOp
from xdsl.ir import Dialect, EnumAttribute
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (
    irdl_attr_definition,
    irdl_op_definition,
    opt_prop_def,
    prop_def,
    region_def,
    var_operand_def,
    var_result_def,
)
from xdsl.utils.exceptions import ParseError


class NativeMatmulRegionRequired(ParseError):
    """The direct Python builder cannot implement this implicit cast policy."""


class TypeFn(StrEnum):
    SIGNED = "cast_signed"
    UNSIGNED = "cast_unsigned"


@irdl_attr_definition
class TypeFnAttr(EnumAttribute[TypeFn]):
    name = "linalg.type_fn"

    def print_parameter(self, printer):
        with printer.in_angle_brackets():
            printer.print_string(self.data.value)

    @classmethod
    def parse_parameter(cls, parser):
        with parser.in_angle_brackets():
            value = parser.parse_identifier()
            if value not in TypeFn._value2member_map_:
                parser.raise_error("expected default or signed cast, or cast_unsigned")
            return TypeFn(value)


@irdl_op_definition
class MatmulOp(XDSLMatmulOp):
    # Decorated inherited fields are accessors; restate their original schema
    # when extending the operation definition with its native cast property.
    inputs = var_operand_def()
    outputs = var_operand_def(ShapedType)
    res = var_result_def(TensorType)
    body = region_def("single_block")
    indexing_maps = prop_def(
        ArrayAttr[AffineMapAttr],
        default_value=ArrayAttr(
            [
                AffineMapAttr(AffineMap.from_callable(lambda i, _j, k: (i, k))),
                AffineMapAttr(AffineMap.from_callable(lambda _i, j, k: (k, j))),
                AffineMapAttr(AffineMap.from_callable(lambda i, j, _k: (i, j))),
            ]
        ),
    )
    cast = opt_prop_def(TypeFnAttr)

    @classmethod
    def get_hidden_region(cls, inputs, outputs):
        types = cls.body_arg_types((*inputs, *outputs))
        if len(inputs) != 2 or len(outputs) != 1 or tuple(types) != (i8, i8, i32):
            return super().get_hidden_region(inputs, outputs)
        from xdsl.builder import Builder
        from xdsl.dialects import arith
        from xdsl.dialects.linalg.ops import YieldOp

        @Builder.implicit_region(types)
        def region(args):
            lhs = arith.ExtSIOp(args[0], i32)
            rhs = arith.ExtSIOp(args[1], i32)
            product = arith.MuliOp(lhs, rhs)
            total = arith.AddiOp(args[2], product)
            YieldOp(total)

        return region

    @classmethod
    def parse(cls, parser):
        op = super().parse(parser)
        types = cls.body_arg_types((*op.inputs, *op.outputs))
        if tuple(types) == (i8, i8, i32) and op.cast is not None and op.cast.data != TypeFn.SIGNED:
            try:
                parser.raise_error("implicit i8 matmul requires the default or signed cast")
            except ParseError as error:
                raise NativeMatmulRegionRequired(error.span, error.msg) from error
        return op

    def verify_(self):
        # The decorator verifies this complete extended schema. The original
        # class has no custom verifier, only a wrapper for its older schema.
        NamedOperation.verify_(self)


@cache
def linalg_dialect():
    return Dialect(
        "linalg",
        [MatmulOp if op.name == MatmulOp.name else op for op in Linalg.operations],
        [*Linalg.attributes, TypeFnAttr],
    )
