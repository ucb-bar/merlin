"""Closed source forms for static integer sequences and f32 truth conversion."""

from .linalg_patterns import InvalidLinalgPattern, _checked_static_linalg_shell


class StaticIotaResourceLimit(RuntimeError):
    """A valid static sequence exceeds the compiler's literal element budget."""


def _closed_body(shell):
    for op in shell.body:
        if any(not key.startswith("prov.") for key in op.attributes):
            raise InvalidLinalgPattern("source scalar body has unknown attributes")
        for flag in ("fastmath", "overflowFlags"):
            value = op.properties.get(flag)
            if value is not None and value.data:
                raise InvalidLinalgPattern("source scalar body has arithmetic flags")


def recognize_f32_nonzero(op):
    """Prove x != 0 with IEEE unordered comparison, including NaNs and signed zero."""
    from .linalg_boolean_patterns import recognize_static_boolean_body

    pattern = recognize_static_boolean_body(op)
    if pattern.operation != "f32_nonzero_to_i1":
        raise InvalidLinalgPattern("truth conversion differs from unflagged f32 une zero")
    return pattern.shape


def recognize_static_iota(op, *, max_elements=1 << 20):
    """Fold a closed rank-one i64 start + index * step with no overflowing intermediate."""
    from xdsl.dialects.builtin import IntegerAttr

    shell = _checked_static_linalg_shell(op)
    if (
        shell.ordered_types != ("i64", "i64")
        or len(shell.shape) != 1
        or tuple(item.name for item in shell.body)
        != (
            "linalg.index",
            "arith.index_cast",
            "arith.constant",
            "arith.muli",
            "arith.constant",
            "arith.addi",
            "linalg.yield",
        )
    ):
        raise InvalidLinalgPattern("iota needs a closed rank-one i64 index expression")
    _closed_body(shell)
    index, cast, step, multiply, start, add, result = shell.body
    dim = index.properties.get("dim")
    literals = (start.properties.get("value"), step.properties.get("value"))
    if (
        set(index.properties) != {"dim"}
        or not isinstance(dim, IntegerAttr)
        or dim.value.data != 0
        or cast.properties
        or tuple(cast.operands) != (index.results[0],)
        or str(cast.results[0].type) != "i64"
        or any(not isinstance(value, IntegerAttr) or str(value.type) != "i64" for value in literals)
        or set(start.properties) != {"value"}
        or set(step.properties) != {"value"}
        or tuple(multiply.operands) != (cast.results[0], step.results[0])
        or tuple(add.operands) != (start.results[0], multiply.results[0])
        or not set(multiply.properties) <= {"overflowFlags"}
        or not set(add.properties) <= {"overflowFlags"}
        or tuple(result.operands) != (add.results[0],)
        or result.properties
    ):
        raise InvalidLinalgPattern("iota has an unknown expression or data dependency")
    first, stride = (value.value.data for value in literals)
    last_product = (shell.shape[0] - 1) * stride
    if not all(-(1 << 63) <= value < (1 << 63) for value in (first, stride, last_product, first + last_product)):
        raise InvalidLinalgPattern("iota intermediate may overflow signed i64")
    if shell.shape[0] > max_elements:
        raise StaticIotaResourceLimit("static iota exceeds the compile-time constant element budget")
    op.verify()
    return tuple(first + i * stride for i in range(shell.shape[0]))
