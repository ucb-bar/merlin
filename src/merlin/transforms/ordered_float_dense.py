"""Read an explicitly selected, positive-zero-seeded ordered floating dot.

This is source recognition, not a grant of target numerical equivalence. The
caller must qualify the selected recurrence and its input domain. No reduction
reassociation, padding, conversion or dtype-based policy inference is performed.
"""

from dataclasses import dataclass
from hashlib import sha256
from math import copysign

from xdsl.dialects import arith, builtin, func, math, tensor
from xdsl.dialects.linalg.attrs import IteratorType
from xdsl.dialects.linalg.ops import FillOp, GenericOp, YieldOp
from xdsl.ir.affine import AffineDimExpr, AffineMap


@dataclass(frozen=True)
class OrderedFloatPolicy:
    input_type: str
    accumulator_type: str
    identity: str
    input_domain: str

    def __post_init__(self):
        if any(type(v) is not str or not v.strip() for v in vars(self).values()):
            raise ValueError("ordered floating source needs explicit types, policy and domain")


@dataclass(frozen=True)
class OrderedFloatDense:
    source_sha256: str
    entry: str
    m: int
    k: int
    n: int
    rhs_transposed: bool
    policy: OrderedFloatPolicy


def _require(condition, reason):
    if not condition:
        raise ValueError("ordered floating dense: " + reason)


def read_ordered_float_dense(source: str, *, entry: str, policy: OrderedFloatPolicy) -> OrderedFloatDense:
    """Recognize a complete two-input contraction function without rewriting it.

    Both ordinary B[K,N] and transposed-storage W[N,K] maps are explicit in the
    result. Only exact extf/extf/fma/yield and an explicit positive-zero fill
    are admitted. The compiler must retain K order and the accumulator format
    across primitive boundaries. Unknown attributes and unused operations fail.
    """
    from merlin.frontends.linalg_mlir import parse_mlir_text

    _require(type(policy) is OrderedFloatPolicy, "a selected numerical policy is required")
    module = parse_mlir_text(source)
    module.verify()
    _require(not module.attributes, "module attributes are not admitted")
    functions = tuple(module.body.block.ops)
    _require(len(functions) == 1 and type(functions[0]) is func.FuncOp, "one function required")
    function = functions[0]
    _require(function.sym_name.data == entry and len(function.body.blocks) == 1, "entry differs or has control flow")
    _require(not function.attributes and set(function.properties) <= {
        "sym_name", "function_type", "arg_attrs", "res_attrs", "sym_visibility",
    }, "unknown function attributes")
    _require(function.res_attrs is None or all(not a.data for a in function.res_attrs), "result attributes")
    block = function.body.block
    _require(len(block.args) == 2, "two tensor inputs required")
    attrs = function.arg_attrs
    expected_domain = {"merlin.input_domain": builtin.StringAttr(policy.input_domain)}
    _require(attrs is not None and len(attrs) == 2 and all(a.data == expected_domain for a in attrs),
             "input domains differ from the selected contract")
    ops = tuple(block.ops)
    _require(tuple(type(op) for op in ops) == (arith.ConstantOp, tensor.EmptyOp, FillOp, GenericOp, func.ReturnOp),
             "expected constant, empty, fill, contraction and return")
    zero, empty, fill, dot, ret = ops
    _require(all(not op.attributes for op in (zero, empty, fill, ret)), "unknown initializer/return attributes")
    _require(set(zero.properties) == {"value"} and not empty.properties and
             set(fill.properties) <= {"operandSegmentSizes"} and not ret.properties,
             "unknown initializer/return properties")
    _require(type(zero.value) is builtin.FloatAttr and str(zero.result.type) == policy.accumulator_type,
             "initializer must have the accumulator format")
    z = zero.value.value.data
    _require(z == 0 and copysign(1, z) > 0, "initializer must be positive zero")
    _require(not empty.operands and list(fill.inputs) == [zero.result] and list(fill.outputs) == [empty.results[0]],
             "fill must initialize the complete static destination")
    _require(len(fill.body.blocks) == 1, "fill must have one scalar block")
    fill_body = fill.body.block
    fill_ops = tuple(fill_body.ops)
    _require(len(fill_body.args) == 2 and len(fill_ops) == 1 and type(fill_ops[0]) is YieldOp and
             list(fill_ops[0].operands) == [fill_body.args[0]] and not fill_ops[0].attributes and
             not fill_ops[0].properties, "fill body must yield its scalar initializer")
    _require(list(dot.inputs) == list(block.args) and list(dot.outputs) == [fill.results[0]] and
             list(ret.operands) == list(dot.results) and len(dot.results) == 1,
             "operand, initializer or returned result correspondence differs")
    values = (*block.args, empty.results[0], fill.results[0], dot.results[0])
    _require(all(type(v.type) is builtin.TensorType and type(v.type.encoding) is builtin.NoneAttr for v in values),
             "ordinary ranked tensors required")
    _require(tuple(str(v.type.element_type) for v in values) ==
             (policy.input_type, policy.input_type, policy.accumulator_type, policy.accumulator_type,
              policy.accumulator_type), "element formats differ")
    shapes = tuple(tuple(v.type.get_shape()) for v in values)
    _require(all(len(s) == 2 and all(d > 0 for d in s) for s in shapes), "positive static rank-two shapes required")
    _require(shapes[2] == shapes[3] == shapes[4], "destination/result shape differs")
    expected_attrs = {
        "merlin.numeric_semantics": builtin.StringAttr(policy.identity),
        "merlin.input_domain": builtin.StringAttr(policy.input_domain),
        "merlin.reduction_order": builtin.StringAttr("k-ascending"),
    }
    _require(dot.attributes == expected_attrs and set(dot.properties) == {
        "indexing_maps", "iterator_types", "operandSegmentSizes",
    }, "contraction policy or attributes differ")
    d0, d1, d2 = (AffineDimExpr(i) for i in range(3))
    left, right, out = tuple(a.data for a in dot.indexing_maps)
    _require(left == AffineMap(3, 0, (d0, d2)) and out == AffineMap(3, 0, (d0, d1)), "contraction maps differ")
    transposed = right == AffineMap(3, 0, (d1, d2))
    _require(transposed or right == AffineMap(3, 0, (d2, d1)), "RHS map is not a matrix")
    m, k = shapes[0]
    n, rk = shapes[1] if transposed else shapes[1][::-1]
    _require(k == rk and shapes[2] == (m, n), "matrix extents differ")
    _require(tuple(a.data for a in dot.iterator_types) == (
        IteratorType.PARALLEL, IteratorType.PARALLEL, IteratorType.REDUCTION,
    ), "iterator order differs")
    _require(len(dot.body.blocks) == 1, "one scalar body required")
    body = dot.body.block
    scalar = tuple(body.ops)
    _require(tuple(type(op) for op in scalar) == (arith.ExtFOp, arith.ExtFOp, math.FmaOp, YieldOp),
             "scalar recurrence must be extf, extf, fma, yield")
    a, b, fma, yld = scalar
    _require(len(body.args) == 3 and list(a.operands) == [body.args[0]] and list(b.operands) == [body.args[1]] and
             list(fma.operands) == [a.result, b.result, body.args[2]] and list(yld.operands) == [fma.result],
             "scalar operand correspondence differs")
    _require(all(str(op.results[0].type) == policy.accumulator_type for op in (a, b, fma)),
             "scalar accumulator format differs")
    _require(all(not op.attributes and set(op.properties) <= {"fastmath"} and
                 not getattr(op.properties.get("fastmath"), "data", ()) for op in scalar),
             "scalar flags or attributes would change the recurrence")
    return OrderedFloatDense(sha256(source.encode()).hexdigest(), entry, m, k, n, transposed, policy)
