"""Retain the standard named-matmul cast property missing from xDSL 0.68.

Fields follow native LinalgStructuredOps.td at
a47bddccec30255619bb8c37fa59700e661d4e66. Native parsing owns implicit
mixed-type regions; this binding supplies no arithmetic rewrite or cast policy.
"""

from enum import StrEnum
from functools import cache

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, ShapedType, TensorType
from xdsl.dialects.linalg import Linalg
from xdsl.dialects.linalg.abstract_ops import NamedOperation
from xdsl.dialects.linalg.ops import MatmulOp as XDSLMatmulOp
from xdsl.ir import Dialect, EnumAttribute
from xdsl.ir.affine import AffineMap
from xdsl.irdl import (irdl_attr_definition, irdl_op_definition, opt_prop_def,
    prop_def, region_def, var_operand_def, var_result_def)


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
            return parser.parse_str_enum(TypeFn)


@irdl_op_definition
class MatmulOp(XDSLMatmulOp):
    # Decorated inherited fields are accessors; restate their original schema
    # when extending the operation definition with its native cast property.
    inputs = var_operand_def()
    outputs = var_operand_def(ShapedType)
    res = var_result_def(TensorType)
    body = region_def("single_block")
    indexing_maps = prop_def(ArrayAttr[AffineMapAttr], default_value=ArrayAttr([
        AffineMapAttr(AffineMap.from_callable(lambda i, _j, k: (i, k))),
        AffineMapAttr(AffineMap.from_callable(lambda _i, j, k: (k, j))),
        AffineMapAttr(AffineMap.from_callable(lambda i, j, _k: (i, j))),
    ]))
    cast = opt_prop_def(TypeFnAttr)

    def verify_(self):
        # The decorator verifies this complete extended schema. The original
        # class has no custom verifier, only a wrapper for its older schema.
        NamedOperation.verify_(self)


@cache
def linalg_dialect():
    return Dialect("linalg", [MatmulOp if op.name == MatmulOp.name else op for op in Linalg.operations],
        [*Linalg.attributes, TypeFnAttr])
