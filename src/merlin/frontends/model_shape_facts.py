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
EXPRESSION_PROFILE = "t2m.symbolic-model-shapes.v2"


class ShapeFacts:
    def __init__(self, *, expressions=False):
        self.tensors = {}
        self.integers = {}
        self.indices = {}
        self.axes = {}
        self.visits = 0
        self.expressions = expressions
        from .shape_expressions import Program

        self.program = Program()
        self.expression_words = {}
        self.arguments = ()
        self.axis_limits = {}

    @classmethod
    def entry(cls, function, *, expressions=False):
        facts = cls(expressions=expressions)
        facts.arguments = tuple(function.body.block.args)
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
        if result is None and self.expressions:
            from .shape_expression_ir import index

            result = index(self, value, depth)
        if result is not None:
            self.indices[value] = result
        return result

    def range_at(self, value, point):
        from .shape_expression_ir import range_at

        return range_at(self, value, point) if self.expressions else None

    def copy_context(self, other, point):
        from .shape_expression_ir import axis_range

        other.program.nodes.update(self.program.nodes)
        other.axes.update(self.axes)
        other.arguments = self.arguments
        for name in self.axes:
            interval = axis_range(self, name, point)
            if interval is not None:
                other.axis_limits[name] = interval

    def equal(self, left, right, point):
        if left == right:
            return True
        if not self.expressions or left is None or right is None:
            return False
        from xdsl.dialects import cf

        from .shape_expression_ir import axis_range, word
        from .source_index_ranges import _dominating_prefix

        assertions = []
        for operation in _dominating_prefix(point) or []:
            if isinstance(operation, cf.AssertOp):
                condition = word(self, operation.operands[0])
                if condition is not None:
                    assertions.append(condition)
        return self.program.equivalent(
            self.program.dimension(left),
            self.program.dimension(right),
            lambda name: axis_range(self, name, point),
            assertions=assertions,
        )

    def same_shape(self, left, right, point):
        return (
            left is not None
            and right is not None
            and len(left) == len(right)
            and all(self.equal(a, b, point) for a, b in zip(left, right, strict=True))
        )

    def check_contract(self, function, declared):
        actual = self.contract(function, force_expressions=declared.get("schema") == "t2m.runtime-shapes.v3")
        if declared.get("schema") != "t2m.runtime-shapes.v3" or not self.expressions:
            if declared != actual:
                raise ValueError("model shape contract differs from its executable source dimensions")
            return
        if (
            set(declared) != set(actual)
            or declared["inputs"] != actual["inputs"]
            or declared["dimension_bindings"] != actual["dimension_bindings"]
            or not isinstance(declared["outputs"], list)
            or len(declared["outputs"]) != len(actual["outputs"])
        ):
            raise ValueError("derived model shape contract changes its public input axes")
        if any(
            not isinstance(row, dict)
            or set(row) != {"shape", "dtype"}
            or not isinstance(row["shape"], list)
            or any(not (type(size) is int and 0 <= size < 2**63 or isinstance(size, str)) for size in row["shape"])
            for row in declared["outputs"]
        ):
            raise ValueError("derived model outputs require complete logical dimension declarations")
        from .shape_expressions import Program

        roots = [
            size
            for row in declared["outputs"]
            for size in row["shape"]
            if isinstance(size, str) and size not in self.axes
        ]
        program = Program.load(declared["shape_program"], self.axes, roots)
        self.program.nodes.update(program.nodes)
        for expected, observed in zip(declared["outputs"], actual["outputs"], strict=True):
            if (
                set(expected) != {"shape", "dtype"}
                or expected["dtype"] != observed["dtype"]
                or not self.same_shape(expected["shape"], observed["shape"], function.body.block.last_op)
            ):
                raise ValueError("derived model result expression differs from its executable source dimensions")

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
        elif self.expressions and isinstance(operation, tensor.ExtractSliceOp):
            sizes = tuple(operation.static_sizes.get_values())
            if len(sizes) == len(static):
                dynamic = iter(operation.sizes)
                result = tuple(self.index(next(dynamic), depth + 1) if size < 0 else size for size in sizes)
        elif getattr(operation, "name", None) == "scf.forall":
            result = self.shape(operation.outputs[value.index], depth + 1)
        elif isinstance(operation, LinalgStructuredOperation):
            destinations = [output for output in operation.outputs if isinstance(output.type, builtin.TensorType)]
            if value.index < len(destinations):
                result = self.shape(destinations[value.index], depth + 1)
        elif isinstance(operation, scf.ForOp):
            # A dynamic carried tensor may resize. Prove the actual yield keeps
            # every tensor shape; keep hypothetical block facts in a fork.
            local = ShapeFacts(expressions=self.expressions)
            local.program.nodes.update(self.program.nodes)
            local.axes.update(self.axes)
            local.arguments = self.arguments
            local.axis_limits.update(self.axis_limits)
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
                self.program.nodes.update(local.program.nodes)
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

    def contract(self, function, *, force_expressions=False):
        terminal = function.body.block.last_op
        if not isinstance(terminal, func.ReturnOp):
            raise TypeError("symbolic model shape contract requires one structured return")
        inputs = [self.shape(value) for value in function.body.block.args]
        outputs = [self.shape(value) for value in terminal.operands]
        if any(shape is None for shape in (*inputs, *outputs)):
            raise ValueError("symbolic model result dimensions need an executable source relationship")
        contract = {
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
        roots = [
            item for row in (*inputs, *outputs) for item in row if isinstance(item, str) and item in self.program.nodes
        ]
        if roots or force_expressions:
            contract.update(schema="t2m.runtime-shapes.v3", shape_program=self.program.export(roots))
        return contract
