"""Native typed non-NaN domains for optional ordered tensor reductions.

Bounds are conservative across rounding modes. Unknown producer, attribute,
argument or effect supplies no fact; no provenance hint supplies admission.
"""

SOURCE = r"""
class _OrderedReductionDomains:
    def __init__(self):
        self.memo = {}

    def ordinary(self, op, allowed=()):
        return all(
            name in allowed
            or name.startswith("prov.")
            or name == "fastmath"
            and str(op.attributes[name]) == "#arith.fastmath<none>"
            for name in op.attributes
        )

    def constant(self, attr):
        from torch_mlir import ir
        import math

        if not isinstance(attr, ir.FloatAttr):
            return None
        value = ir.FloatAttr(attr).value
        if math.isnan(value):
            return None
        return (math.isfinite(value), value >= 0, abs(value), value)

    def scalar(self, value, bindings, depth):
        from torch_mlir import ir

        if value in bindings:
            return bindings[value]
        if depth > 64 or not isinstance(value.type, ir.F32Type):
            return None
        op = value.owner
        if isinstance(op, ir.OpView):
            op = op.operation
        if not isinstance(op, ir.Operation):
            return None
        if op.name == "arith.constant" and self.ordinary(op, {"value"}):
            return self.constant(op.attributes["value"])
        if not self.ordinary(op):
            return None
        if op.name in {"arith.sitofp", "arith.uitofp"} and len(op.operands) == 1:
            typ = op.operands[0].type
            if isinstance(typ, ir.IntegerType) and 1 <= ir.IntegerType(typ).width <= 64:
                return (
                    True,
                    op.name == "arith.uitofp",
                    float(2 ** ir.IntegerType(typ).width),
                    None,
                )
            return None
        if op.name == "arith.select" and len(op.operands) == 3:
            a, b = [self.scalar(v, bindings, depth + 1) for v in op.operands[1:]]
            return (
                None
                if a is None or b is None
                else (a[0] and b[0], a[1] and b[1], max(a[2], b[2]), None)
            )
        if op.name == "math.exp" and len(op.operands) == 1:
            a = self.scalar(op.operands[0], bindings, depth + 1)
            return None if a is None else (False, True, float("inf"), None)
        if len(op.operands) != 2 or op.name not in {
            "arith.mulf",
            "arith.divf",
            "arith.addf",
            "arith.subf",
        }:
            return None
        a, b = [self.scalar(v, bindings, depth + 1) for v in op.operands]
        if a is None or b is None:
            return None
        if op.name == "arith.addf" and a[1] and b[1]:
            bound = a[2] + b[2]
            return (
                a[0] and b[0] and bound < 1.7e38,
                True,
                max(2 * bound, 2.0**-149),
                None,
            )
        if not a[0] or not b[0]:
            return None
        if op.name == "arith.divf":
            if b[3] is None or b[3] <= 0:
                return None
            bound, nonnegative = a[2] / b[3], a[1]
        elif op.name == "arith.mulf":
            bound, nonnegative = a[2] * b[2], a[1] and b[1]
        else:
            bound, nonnegative = a[2] + b[2], op.name == "arith.addf" and a[1] and b[1]
        return (bound < 1.7e38, nonnegative, max(2 * bound, 2.0**-149), None)

    def tensor(self, value, depth=0):
        from torch_mlir import ir

        if value in self.memo:
            return self.memo[value]
        if depth > 64 or len(self.memo) > 8192:
            return None
        self.memo[value] = None
        op = value.owner
        if isinstance(op, ir.OpView):
            op = op.operation
        if not isinstance(op, ir.Operation) or not isinstance(
            value.type, ir.RankedTensorType
        ):
            return None
        typ = ir.RankedTensorType(value.type)
        if not isinstance(typ.element_type, ir.F32Type):
            return None
        result = None
        if op.name == "arith.constant" and self.ordinary(op, {"value"}):
            attr = op.attributes["value"]
            if isinstance(attr, ir.DenseFPElementsAttr) and attr.is_splat:
                result = self.constant(attr.get_splat_value())
        elif op.name == "tensor.splat" and self.ordinary(op):
            result = self.scalar(op.operands[0], {}, depth + 1)
        elif op.name == "linalg.fill" and self.ordinary(op, {"operandSegmentSizes"}):
            result = self.scalar(op.operands[0], {}, depth + 1)
        elif op.name in {
            "tensor.cast",
            "tensor.extract_slice",
            "tensor.collapse_shape",
            "tensor.expand_shape",
        }:
            if self.ordinary(
                op,
                {
                    "static_offsets",
                    "static_sizes",
                    "static_strides",
                    "operandSegmentSizes",
                    "reassociation",
                    "static_output_shape",
                },
            ):
                result = self.tensor(op.operands[0], depth + 1)
        elif (
            op.name in {"linalg.generic", "linalg.reduce"}
            and len(op.results) == 1
            and len(op.regions) == 1
            and len(op.regions[0].blocks) == 1
        ):
            block = op.regions[0].blocks[0]
            body = list(block.operations)
            allowed = (
                {"dimensions"}
                if op.name == "linalg.reduce"
                else {"indexing_maps", "iterator_types", "operandSegmentSizes"}
            )
            if (
                not self.ordinary(op, allowed)
                or not body
                or body[-1].operation.name != "linalg.yield"
            ):
                return None
            iterators = (
                ["reduction"]
                if op.name == "linalg.reduce"
                else [str(x) for x in op.attributes["iterator_types"]]
            )
            pure = {
                "arith.constant",
                "arith.sitofp",
                "arith.uitofp",
                "arith.select",
                "arith.addf",
                "arith.subf",
                "arith.mulf",
                "arith.divf",
                "arith.cmpi",
                "arith.cmpf",
                "linalg.index",
                "math.exp",
            }
            if all(x == "#linalg.iterator_type<parallel>" for x in iterators):
                if not list(block.arguments[-1].uses) and all(
                    not child.regions and child.operation.name in pure
                    for child in body[:-1]
                ):
                    bindings = {
                        arg: self.tensor(v, depth + 1)
                        for arg, v in zip(block.arguments, op.operands[:-1])
                    }
                    result = self.scalar(body[-1].operands[0], bindings, depth + 1)
            elif (
                len(op.operands) == 2
                and len(block.arguments) == 2
                and len(body) == 2
                and body[0].operation.name in {"arith.maximumf", "arith.maxnumf"}
                and list(body[0].operands)
                in [list(block.arguments), list(block.arguments)[::-1]]
                and list(body[-1].operands) == list(body[0].results)
                and self.ordinary(body[0])
            ):
                a, b = [self.tensor(v, depth + 1) for v in op.operands]
                if a is not None and b is not None:
                    finite, positive, bound = (
                        a[0] and b[0],
                        a[1] and b[1],
                        max(a[2], b[2]),
                    )
                    if (
                        a[0]
                        and b[3] == -float("inf")
                        and all(
                            x > 0
                            for x in ir.RankedTensorType(op.operands[0].type).shape
                        )
                    ):
                        finite, positive, bound = True, a[1], a[2]
                    result = (finite, positive, bound, None)
        self.memo[value] = result
        return result
"""
