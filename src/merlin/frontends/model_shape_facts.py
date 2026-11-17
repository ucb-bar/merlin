"""Reconstruct tensor dimensions from compiler-owned standard MLIR.

LLVM a47bddcc TensorOps.cpp reifyResultShapes and SCF.cpp DimOfForallOp
define the destination/initial-output rules used here. Scalar shape transport
comes only from a checked caller adapter; tags and evaluator arrays supply no
shape identity. Unknown arithmetic, resizing loops and opaque calls stay unknown.
"""

from xdsl.dialects import arith, builtin, func, scf, tensor
from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation

from merlin.xdsl_dialects.standard_arith import IndexCastUIOp

from .source_index_ranges import unsigned_index_range

PROFILE = "t2m.symbolic-model-shapes.v1"


class ShapeFacts:
    def __init__(self):
        self.tensors = {}
        self.integers = {}
        self.indices = {}
        self.axes = {}
        self.visits = 0

    @classmethod
    def entry(cls, function):
        facts = cls()
        for argument, value in enumerate(function.body.block.args):
            if not isinstance(value.type, builtin.TensorType):
                continue
            if not isinstance(value.type.encoding, builtin.NoneAttr):
                continue
            shape = []
            for axis, size in enumerate(value.type.get_shape()):
                if size < 0:
                    size = f"d{argument}_{axis}"
                    facts.axes[size] = {"argument": argument, "axis": axis}
                shape.append(size)
            facts.tensors[value] = tuple(shape)
        return facts

    def _visit(self, depth):
        self.visits += 1
        if depth >= 64 or self.visits > 16384:
            raise ValueError("symbolic tensor shape analysis exceeded its finite source budget")

    def index(self, value, depth=0):
        self._visit(depth)
        if value in self.indices:
            return self.indices[value]
        operation = value.owner
        result = None
        if isinstance(operation, arith.ConstantOp) and isinstance(operation.value, builtin.IntegerAttr):
            result = operation.value.value.data
            if result < 0:
                return None
        elif isinstance(operation, tensor.DimOp):
            source, axis = operation.operands
            axis = self.index(axis, depth + 1)
            shape = self.shape(source, depth + 1)
            if shape is not None and type(axis) is int and 0 <= axis < len(shape):
                result = shape[axis]
        elif isinstance(operation, (arith.IndexCastOp, IndexCastUIOp)):
            source = operation.operands[0]
            if source in self.integers:
                result = self.integers[source]
            elif isinstance(source.owner, arith.ConstantOp) and isinstance(source.owner.value, builtin.IntegerAttr):
                number = source.owner.value.value.data
                if isinstance(source.type, builtin.IntegerType) and 0 <= number < 1 << (source.type.bitwidth - 1):
                    result = number
            elif isinstance(source.owner, (arith.IndexCastOp, IndexCastUIOp)):
                original = source.owner.operands[0]
                enclosure = unsigned_index_range(original, operation)
                if (
                    isinstance(original.type, builtin.IndexType)
                    and isinstance(source.type, builtin.IntegerType)
                    and enclosure is not None
                    and 0 <= enclosure[0] <= enclosure[1] < 1 << (source.type.bitwidth - 1)
                ):
                    result = self.index(original, depth + 1)
        elif isinstance(operation, (arith.AddiOp, arith.SubiOp, arith.MuliOp)):
            left, right = (self.index(operand, depth + 1) for operand in operation.operands)
            if isinstance(operation, (arith.AddiOp, arith.SubiOp)) and right == 0:
                result = left
            elif isinstance(operation, arith.AddiOp) and left == 0:
                result = right
            elif isinstance(operation, arith.MuliOp) and (left == 1 or right == 1):
                result = right if left == 1 else left
            elif isinstance(operation, arith.MuliOp) and (left == 0 or right == 0):
                result = 0
        if result is not None:
            self.indices[value] = result
        return result

    def shape(self, value, depth=0):
        self._visit(depth)
        if value in self.tensors:
            return self.tensors[value]
        type_ = value.type
        if not isinstance(type_, (builtin.TensorType, builtin.MemRefType)):
            return None
        if isinstance(type_, builtin.TensorType) and not isinstance(type_.encoding, builtin.NoneAttr):
            return None
        static = tuple(type_.get_shape())
        if all(size >= 0 for size in static):
            return static
        operation = value.owner
        result = None
        if isinstance(operation, tensor.EmptyOp) or getattr(operation, "name", None) in (
            "memref.alloc",
            "memref.alloca",
        ):
            sizes = iter(operation.operands if isinstance(operation, tensor.EmptyOp) else operation.dynamic_sizes)
            result = tuple(self.index(next(sizes), depth + 1) if size < 0 else size for size in static)
        elif isinstance(operation, tensor.CastOp) or getattr(operation, "name", None) in (
            "memref.cast",
            "bufferization.to_tensor",
            "bufferization.to_buffer",
        ):
            result = self.shape(operation.operands[0], depth + 1)
        elif isinstance(operation, tensor.InsertSliceOp):
            result = self.shape(operation.operands[1], depth + 1)
        elif getattr(operation, "name", None) == "scf.forall":
            result = self.shape(operation.outputs[value.index], depth + 1)
        elif isinstance(operation, LinalgStructuredOperation):
            destinations = [output for output in operation.outputs if isinstance(output.type, builtin.TensorType)]
            if value.index < len(destinations):
                result = self.shape(destinations[value.index], depth + 1)
        elif isinstance(operation, scf.ForOp):
            # A dynamic carried tensor may resize. Prove the actual yield keeps
            # every tensor shape; keep hypothetical block facts in a fork.
            local = ShapeFacts()
            local.tensors.update(self.tensors)
            local.integers.update(self.integers)
            initial = [
                self.shape(item, depth + 1) if isinstance(item.type, builtin.TensorType) else None
                for item in operation.iter_args
            ]
            for argument, shape in zip(operation.body.block.args[1:], initial, strict=True):
                if shape is not None:
                    local.tensors[argument] = shape
            terminal = operation.body.block.last_op
            if isinstance(terminal, scf.YieldOp):
                final = [
                    local.shape(item, depth + 1) if isinstance(item.type, builtin.TensorType) else None
                    for item in terminal.operands
                ]
                self.visits += local.visits
                if self.visits > 16384:
                    raise ValueError("symbolic tensor shape analysis exceeded its finite source budget")
                if final == initial and initial[value.index] is not None:
                    result = initial[value.index]
        elif isinstance(operation, scf.IfOp):
            alternatives = []
            for region in (operation.true_region, operation.false_region):
                terminal = region.block.last_op if len(region.blocks) == 1 else None
                if not isinstance(terminal, scf.YieldOp):
                    break
                alternatives.append(self.shape(terminal.operands[value.index], depth + 1))
            if len(alternatives) == 2 and alternatives[0] == alternatives[1]:
                result = alternatives[0]
        if result is not None and len(result) == len(static) and all(size is not None for size in result):
            if any(size >= 0 and size != actual for size, actual in zip(static, result, strict=True)):
                return None
            self.tensors[value] = result
            return result
        return None

    def contract(self, function):
        terminal = function.body.block.last_op
        if not isinstance(terminal, func.ReturnOp):
            raise TypeError("symbolic model shape contract requires one structured return")
        inputs = [self.shape(value) for value in function.body.block.args]
        outputs = [self.shape(value) for value in terminal.operands]
        if any(shape is None for shape in (*inputs, *outputs)):
            raise ValueError("symbolic model result dimensions need an executable source relationship")
        return {
            "schema": "t2m.runtime-shapes.v2",
            "inputs": [
                {"shape": list(shape), "dtype": str(value.type.element_type)}
                for value, shape in zip(function.body.block.args, inputs, strict=True)
            ],
            "outputs": [
                {"shape": list(shape), "dtype": str(value.type.element_type)}
                for value, shape in zip(terminal.operands, outputs, strict=True)
            ],
            "dimension_bindings": self.axes,
        }
