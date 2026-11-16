"""Upstream batch_matmul syntax missing from the selected xDSL Linalg dialect.

Reuse named matmul parsing and scalar region construction. Source consumers must
still check complete batch geometry, maps, arithmetic and initialization.
"""

from typing import ClassVar

from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
from xdsl.dialects.linalg.attrs import IteratorTypeAttr
from xdsl.dialects.linalg.ops import MatmulOp, NamedOperation
from xdsl.ir.affine import AffineMap
from xdsl.irdl import irdl_op_definition, prop_def


@irdl_op_definition
class BatchMatmulOp(NamedOperation):
    name = "linalg.batch_matmul"
    PRINT_ATTRS_IN_FRONT: ClassVar[bool] = True
    indexing_maps = prop_def(
        ArrayAttr[AffineMapAttr],
        default_value=ArrayAttr(
            [
                AffineMapAttr(AffineMap.from_callable(lambda b, m, n, k: (b, m, k))),
                AffineMapAttr(AffineMap.from_callable(lambda b, m, n, k: (b, k, n))),
                AffineMapAttr(AffineMap.from_callable(lambda b, m, n, k: (b, m, n))),
            ]
        ),
    )

    def __init__(self, inputs, outputs=(), res=None, attributes=None):
        super().__init__(
            ins=inputs,
            outs=outputs,
            result_types=[v.type for v in outputs if isinstance(v.type, TensorType)] if res is None else res,
            attributes=attributes,
            hidden_region=MatmulOp.get_hidden_region(inputs, outputs),
        )

    def get_default_indexing_maps(self):
        return tuple(m.data for m in self.indexing_maps)

    @classmethod
    def get_hidden_region(cls, inputs, outputs):
        return MatmulOp.get_hidden_region(inputs, outputs)

    def get_indexing_maps(self):
        return self.indexing_maps

    def get_iterator_types(self):
        return ArrayAttr([IteratorTypeAttr.parallel()] * 3 + [IteratorTypeAttr.reduction()])
