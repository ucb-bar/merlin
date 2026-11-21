"""Independently reconstruct checked runtime dense value reshapes.

Only actual typed tensors and checked source dimension operands participate.
Use standard collapse/expand, not a physical view or sampled shape. Review:
LLVM a47bddccec30255619bb8c37fa59700e661d4e66 TensorOps.cpp, tensor/ops.mlir,
and ArithOps.td's index-width extended multiplication contract.
"""

import math

from xdsl.dialects import arith, builtin, cf, tensor

POLICY = "dense-row-major-runtime-v1"
PROFILE = "t2m.execution-checked-call-views.v2"


class RuntimeViewContracts:
    def __init__(self, append):
        self.emit = append

    def constant(self, value):
        return self.emit(arith.ConstantOp.from_int_and_width(value, builtin.IndexType()), "runtime-view-constant")

    def require(self, predicate, message):
        self.emit(cf.AssertOp(predicate, "Triton executed call: runtime view " + message), "runtime-view-contract")

    def elements(self, extents):
        zero, one = self.constant(0), self.constant(1)
        empty = self.emit(arith.CmpiOp(zero, one, "eq"), "runtime-view-empty")
        for extent in extents:
            nonnegative = self.emit(arith.CmpiOp(extent, zero, "sge"), "runtime-view-nonnegative")
            self.require(nonnegative, "negative extent")
            is_zero = self.emit(arith.CmpiOp(extent, zero, "eq"), "runtime-view-zero")
            empty = self.emit(arith.OrIOp(empty, is_zero), "runtime-view-empty")
        product = one
        for extent in extents:
            factor = self.emit(arith.SelectOp(empty, one, extent), "runtime-view-factor")
            multiply = arith.MulUIExtendedOp(product, factor)
            self.emit(multiply, "runtime-view-product")
            product, high = multiply.results
            fits = self.emit(arith.CmpiOp(high, zero, "eq"), "runtime-view-no-overflow")
            signed = self.emit(arith.CmpiOp(product, zero, "sge"), "runtime-view-product-nonnegative")
            self.require(
                self.emit(arith.AndIOp(fits, signed), "runtime-view-product-valid"), "element product overflow"
            )
        return self.emit(arith.SelectOp(empty, zero, product), "runtime-view-element-count")

    def adapt(self, value, target, kernel_shape, dimensions, *, input_view):
        source = value.type
        if (
            not isinstance(source, builtin.TensorType)
            or not isinstance(target, builtin.TensorType)
            or not isinstance(source.encoding, builtin.NoneAttr)
            or not isinstance(target.encoding, builtin.NoneAttr)
            or source.element_type != target.element_type
        ):
            raise ValueError("runtime dense view changes original tensor storage or encoding")
        shape = target.get_shape()
        if not input_view and (sum(n < 0 for n in shape) > 1 or any(n < 0 for n in shape) and 0 in shape):
            raise ValueError("runtime dense result view has ambiguous source dimensions")
        if input_view and (
            len(shape) != len(kernel_shape) or any(isinstance(n, str) and n not in dimensions for n in kernel_shape)
        ):
            raise ValueError("runtime dense view lacks original kernel dimension operands")
        original_extents = []
        for axis, extent in enumerate(source.get_shape()):
            original_extents.append(
                self.constant(extent)
                if extent >= 0
                else self.emit(tensor.DimOp(value, self.constant(axis)), "runtime-view-actual-dimension")
            )
        count = self.elements(original_extents)
        if input_view:
            sizes = [dimensions[n] if isinstance(n, str) else self.constant(n) for n in kernel_shape]
        else:
            factor = self.elements([self.constant(n) for n in shape if n >= 0])
            zero, one = self.constant(0), self.constant(1)
            is_zero = self.emit(arith.CmpiOp(factor, zero, "eq"), "runtime-view-divisor-zero")
            safe = self.emit(arith.SelectOp(is_zero, one, factor), "runtime-view-safe-divisor")
            inferred = self.emit(arith.DivUIOp(count, safe), "runtime-view-inferred-dimension")
            sizes = [self.constant(n) if n >= 0 else inferred for n in shape]
        result_count = self.elements(sizes)
        equal = self.emit(arith.CmpiOp(count, result_count, "eq"), "runtime-view-equal-counts")
        self.require(equal, "element count mismatch")
        element = source.element_type

        def flat_type(shape):
            count = builtin.DYNAMIC_INDEX if any(n < 0 for n in shape) else math.prod(shape)
            return builtin.TensorType(element, [count])

        def reassociation(rank):
            return builtin.ArrayAttr(
                [builtin.ArrayAttr(builtin.IntegerAttr(i, builtin.i64) for i in range(rank))] if rank else []
            )

        rank = len(source.get_shape())
        if rank > 1:
            value = self.emit(
                tensor.CollapseShapeOp.build(
                    operands=[value],
                    result_types=[flat_type(source.get_shape())],
                    properties={"reassociation": reassociation(rank)},
                ),
                "runtime-view-collapse",
            )
        elif rank == 0:
            scalar = self.emit(
                tensor.ExtractOp.build(operands=[value, []], result_types=[element]), "runtime-view-scalar-read"
            )
            value = self.emit(tensor.FromElementsOp(scalar), "runtime-view-scalar-pack")
        flat = flat_type(shape)
        if value.type != flat:
            value = self.emit(tensor.CastOp(value, flat), "runtime-view-flat-cast")
        if len(shape) > 1:
            dynamic = [size for declared, size in zip(shape, sizes, strict=True) if declared < 0]
            value = self.emit(
                tensor.ExpandShapeOp(value, dynamic, reassociation(len(shape)), shape, target), "runtime-view-expand"
            )
        elif not shape:
            value = self.emit(
                tensor.CollapseShapeOp.build(
                    operands=[value], result_types=[target], properties={"reassociation": reassociation(0)}
                ),
                "runtime-view-scalar-collapse",
            )
        return value
