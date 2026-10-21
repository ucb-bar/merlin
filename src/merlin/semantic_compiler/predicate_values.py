"""Exact bit-level predicate primitives shared by specialization and C emission."""

from .model import TensorType

PREDICATE_OPERATIONS = frozenset({"mask", "select", "f32_nonzero"})
_WIDTHS = {"i1": 1, "i8": 1, "i32": 4, "i64": 8, "fp8_e4m3": 1, "bf16": 2, "f32": 4, "f64": 8}


def ordered_float_compare(a, b, width, predicate):
    """Compare IEEE encodings without host arithmetic or NaN canonicalization."""
    magnitude = (1 << (width * 8 - 1)) - 1
    infinity = magnitude - ((1 << (23 if width == 4 else 52)) - 1)
    if (a & magnitude) > infinity or (b & magnitude) > infinity:
        return 0
    equal = a == b or (a & magnitude) == (b & magnitude) == 0
    sign_a, sign_b = a >> (width * 8 - 1), b >> (width * 8 - 1)
    less = not equal and (sign_a if sign_a != sign_b else a > b if sign_a else a < b)
    return int({"oeq": equal, "one": not equal, "olt": less, "ole": less or equal,
                "ogt": not (less or equal), "oge": not less}[predicate])


def predicate_widths(node, operands):
    """Check a same-shape primitive before evaluating declared constants.

    Index-map validation belongs to the caller. This function does not admit
    broadcasted constants or select a host execution route.
    """
    shape = node.type.shape
    boolean = TensorType(shape, "i1", "boolean")
    if any(item.type.shape != shape for item in operands):
        raise ValueError("constant predicate requires equal operand shapes")
    if node.op == "select":
        if (len(operands) != 3 or node.attrs or operands[0].type != boolean
                or any(item.type != node.type for item in operands[1:]) or node.type.dtype not in _WIDTHS):
            raise ValueError("constant select needs a Boolean condition and unchanged value types")
        width = _WIDTHS[node.type.dtype]
        return (1, width, width), width
    if node.op == "f32_nonzero":
        if (len(operands) != 1 or operands[0].type != TensorType(shape, "f32", "raw-bit-pattern")
                or node.type != boolean or node.attrs != (("numeric_semantics", "f32-une-zero-v1"),)):
            raise ValueError("constant truth conversion has different source semantics")
        return (4,), 1
    attrs = dict(node.attrs)
    if node.op != "mask" or set(attrs) != {"operation", "predicate"}:
        raise ValueError("constant mask has unknown source semantics")
    operation, predicate = attrs["operation"], attrs["predicate"]
    if operation == "arith.cmpi" and predicate in {"eq", "ne", "slt", "sle", "sgt", "sge"}:
        expected, arity = TensorType(shape, "i64", "signed-i64"), 2
    elif operation == "arith.cmpf" and predicate in {"oeq", "one", "olt", "ole", "ogt", "oge"}:
        if not operands or operands[0].type.dtype not in {"f32", "f64"}:
            raise ValueError("constant floating comparison needs IEEE f32 or f64 operands")
        expected, arity = TensorType(shape, operands[0].type.dtype, "raw-bit-pattern"), 2
    elif operation in {"arith.andi", "arith.xori", "arith.extui"} and predicate is None:
        expected, arity = boolean, 1 if operation == "arith.extui" else 2
    else:
        raise ValueError("constant predicate or operation is not admitted")
    output = TensorType(shape, "i64", "signed-i64") if operation == "arith.extui" else boolean
    if len(operands) != arity or any(item.type != expected for item in operands) or node.type != output:
        raise ValueError("constant mask types differ from its operation")
    return (_WIDTHS[expected.dtype],) * arity, _WIDTHS[output.dtype]


def predicate_element(node, arguments, widths):
    """Evaluate a validated primitive on raw integer encodings."""
    if node.op == "select":
        return arguments[1] if arguments[0] else arguments[2]
    if node.op == "f32_nonzero":
        return int((arguments[0] & 0x7FFFFFFF) != 0)
    attrs = dict(node.attrs)
    operation = attrs["operation"]
    if operation == "arith.cmpi":
        a, b = (v if v < 1 << 63 else v - (1 << 64) for v in arguments)
        return int({"eq": a == b, "ne": a != b, "slt": a < b, "sle": a <= b,
                    "sgt": a > b, "sge": a >= b}[attrs["predicate"]])
    if operation == "arith.cmpf":
        return ordered_float_compare(*arguments, widths[0], attrs["predicate"])
    if operation == "arith.extui":
        return arguments[0]
    a, b = arguments
    return a & b if operation == "arith.andi" else a ^ b
