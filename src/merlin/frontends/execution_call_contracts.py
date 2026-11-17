"""Independently reconstruct executable typed kernel-call adapters.

Only original parent SSA, retained kernel types and declared ABI are inputs.
This reader imports neither the producer nor an evaluator. It establishes
call expansion correspondence, not equivalence of TTIR and its translation.
"""

import math

from xdsl.dialects import arith, builtin, cf, tensor


def scalar_attribute(value):
    if value is None:
        return builtin.UnitAttr()
    if type(value) is bool:
        return builtin.IntegerAttr(int(value), builtin.i1)
    if type(value) is int:
        return builtin.IntegerAttr(value, builtin.i64)
    if type(value) is float and math.isfinite(value):
        return builtin.FloatAttr(value, builtin.f64)
    if isinstance(value, str):
        return builtin.StringAttr(value)
    if isinstance(value, list):
        return builtin.ArrayAttr(scalar_attribute(item) for item in value)
    raise ValueError("executed call has an unsupported captured scalar attribute")


def _compatible(left, right):
    return (
        isinstance(left, builtin.TensorType)
        and isinstance(right, builtin.TensorType)
        and isinstance(left.encoding, builtin.NoneAttr)
        and isinstance(right.encoding, builtin.NoneAttr)
        and left.element_type == right.element_type
        and len(left.get_shape()) == len(right.get_shape())
        and all(a < 0 or b < 0 or a == b for a, b in zip(left.get_shape(), right.get_shape(), strict=True))
    )


class CallContracts:
    """One original call's expected executable guards and value adapters."""

    def __init__(self, append):
        self.append = append

    def constant(self, number, typ):
        return self.append(arith.ConstantOp.from_int_and_width(number, typ), "call-contract-constant")

    def check(self, condition, message):
        self.append(cf.AssertOp(condition, "Triton executed call: " + message), "call-contract")

    def index(self, value, target):
        changed = self.append(arith.IndexCastOp(value, target), "checked-call-index-cast")
        original = self.append(arith.IndexCastOp(changed, value.type), "checked-call-index-roundtrip")
        same = self.append(arith.CmpiOp(value, original, "eq"), "checked-call-index-equal")
        self.check(same, "integer/index conversion is not representable")
        return changed

    def tensor(self, value, target):
        if not _compatible(value.type, target):
            raise ValueError("executed call has incompatible tensor rank/storage/static dimensions")
        for axis, (a, b) in enumerate(zip(value.type.get_shape(), target.get_shape(), strict=True)):
            if a < 0 <= b:
                index = self.constant(axis, builtin.IndexType())
                size = self.append(tensor.DimOp(value, index), "call-actual-dimension")
                wanted = self.constant(b, builtin.IndexType())
                same = self.append(arith.CmpiOp(size, wanted, "eq"), "call-static-dimension")
                self.check(same, "tensor narrowing changes a static dimension")
        return value if value.type == target else self.append(tensor.CastOp(value, target), "checked-call-tensor-cast")

    def dimensions(self, value, buffer, dimensions):
        for axis, extent in enumerate(buffer["shape"]):
            if isinstance(extent, str):
                index = self.constant(axis, builtin.IndexType())
                actual = self.append(tensor.DimOp(value, index), "call-actual-dimension")
                same = self.append(arith.CmpiOp(actual, dimensions[extent], "eq"), "call-dimension-binding")
                self.check(same, "tensor dimension differs from scalar: " + buffer["name"])

    def view(self, value, target, buffer, record, attributes, *, role, dimensions=None):
        from .execution_runtime_views import POLICY as RUNTIME_POLICY
        from .execution_runtime_views import RuntimeViewContracts

        if record.get("tensor_views") == RUNTIME_POLICY and (
            role == "caller-to-kernel-view" or not _compatible(value.type, target)
        ):
            return RuntimeViewContracts(self.append).adapt(
                value, target, buffer["shape"], dimensions or {}, input_view=role == "caller-to-kernel-view"
            )
        if _compatible(value.type, target):
            return self.tensor(value, target)
        from .execution_call_views import POLICY, concrete, reshape

        if record.get("tensor_views") != POLICY:
            raise ValueError("executed call needs an explicit dense tensor view policy")
        logical = concrete(buffer, record, attributes, value.type.element_type)
        if role == "caller-to-kernel-view":
            return self.tensor(reshape(value, logical, self.append, role=role), target)
        return reshape(self.tensor(value, logical), target, self.append, role=role)

    def arguments(self, spec, function, record, by_name, attributes):
        scalars = spec.get("runtime_scalars", [])
        live, captured = record["runtime_operand_arguments"], record["runtime_scalar_arguments"]
        formal = {item["name"] for item in scalars}
        if (
            set(live) & set(captured)
            or set(live) | set(captured) != formal
            or set(live.values()) & set(attributes.data)
        ):
            raise ValueError("executed call scalar mappings do not partition the declared formals")
        tensor_inputs = [b for b in spec["buffers"] if b["effect"] != "write"]
        scalar_types = list(function.function_type.inputs)[len(tensor_inputs) :]
        values = {}
        for source, target in zip(scalars, scalar_types, strict=True):
            name = source["name"]
            if str(target) != source["dtype"]:
                raise ValueError("executed scalar type differs from the retained ABI")
            if name in live:
                value = by_name[live[name]]
                if isinstance(value.type, builtin.TensorType):
                    if (
                        not isinstance(value.type.encoding, builtin.NoneAttr)
                        or value.type.element_type != target
                        or any(n != 1 for n in value.type.get_shape())
                    ):
                        raise ValueError("executed scalar needs an exact singleton tensor")
                    indices = [self.constant(0, builtin.IndexType()) for _ in value.type.get_shape()]
                    value = self.append(
                        tensor.ExtractOp.build(operands=[value, indices], result_types=[target]), "call-scalar-read"
                    )
                elif value.type != target:
                    if not isinstance(value.type, builtin.IndexType) or str(target) not in ("i8", "i16", "i32", "i64"):
                        raise ValueError("executed call loses the original scalar type")
                    value = self.index(value, target)
            else:
                attribute = attributes.data[captured[name]]
                if target == builtin.i1:
                    if not isinstance(attribute, builtin.IntegerAttr) or attribute.type != builtin.i1:
                        raise ValueError("executed Boolean lacks its exact captured attribute")
                elif isinstance(target, builtin.IntegerType):
                    if not isinstance(attribute, builtin.IntegerAttr) or attribute.type != builtin.i64:
                        raise ValueError("executed integer lacks its captured integer attribute")
                    number = attribute.value.data
                    if not -(1 << (target.bitwidth - 1)) <= number < 1 << (target.bitwidth - 1):
                        raise ValueError("executed captured integer is not representable")
                    bounds = source.get("bounds")
                    if bounds is not None and not bounds[0] <= number <= bounds[1]:
                        raise ValueError("executed captured integer is outside its declared domain")
                    attribute = builtin.IntegerAttr(number, target)
                elif (
                    not isinstance(attribute, builtin.FloatAttr)
                    or attribute.type != builtin.f64
                    or not math.isfinite(attribute.value.data)
                    or target not in (builtin.f32, builtin.f64)
                ):
                    raise ValueError("executed float lacks its finite captured f64 attribute")
                value = self.append(arith.ConstantOp(attribute), "captured-call-scalar")
                if value.type != target:
                    value = self.append(arith.TruncFOp(value, target), "captured-call-scalar-rounding")
            if (bounds := source.get("bounds")) is not None:
                low, high = (self.constant(n, value.type) for n in bounds)
                lower = self.append(arith.CmpiOp(value, low, "sge"), "call-scalar-lower")
                upper = self.append(arith.CmpiOp(value, high, "sle"), "call-scalar-upper")
                valid = self.append(arith.AndIOp(lower, upper), "call-scalar-range")
                self.check(valid, "runtime scalar outside declared bounds: " + name)
            values[name] = value
        dimensions = {}
        for name in sorted({n for b in spec["buffers"] for n in b["shape"] if isinstance(n, str)}):
            value = values[name]
            valid = self.append(arith.CmpiOp(value, self.constant(0, value.type), "sge"), "call-size-nonnegative")
            self.check(valid, "negative tensor dimension: " + name)
            dimensions[name] = self.index(value, builtin.IndexType())
        arguments = []
        for source, target in zip(tensor_inputs, function.function_type.inputs, strict=False):
            value = by_name[record["source_operand_bindings"][source["name"]]]
            from .execution_runtime_views import POLICY as RUNTIME_POLICY

            if record.get("tensor_views") == RUNTIME_POLICY or not _compatible(value.type, target):
                value = self.view(
                    value, target, source, record, attributes, role="caller-to-kernel-view", dimensions=dimensions
                )
            self.dimensions(value, source, dimensions)
            arguments.append(self.tensor(value, target))
        arguments.extend(values[s["name"]] for s in scalars)
        return arguments, dimensions
