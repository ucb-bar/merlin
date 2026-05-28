"""Shared `iree.compiler.ir` helpers used by every v2 recognizer.

A recognizer is a free function:

    try_recognize(module, *, fp_dtype, mlir_text=None, **_) -> QnnGraphDesc | None

Returning `None` lets the dispatcher try the next recognizer; raising is
reserved for malformed inputs that *should* have matched but didn't (so
the dispatcher surfaces a useful error instead of silently falling
through).

**No regex.** Every helper uses the structured bindings API
(`ir.FloatAttr(...).value`, `DenseIntElementsAttr.get_splat_value()`,
iteration over dense element attrs, `DictAttr` indexed access).
"""

from __future__ import annotations

from typing import Any

from iree.compiler import ir


def find_func(module: Any) -> Any | None:
    """Return the first `func.func` op in the module, or None."""
    for op in module.body.operations:
        if op.operation.name == "func.func":
            return op
    return None


def func_name(func_op: Any) -> str:
    """Return the func.func's symbol name (without the `@` prefix)."""
    sym_attr = func_op.attributes["sym_name"]
    return ir.StringAttr(sym_attr).value


def func_arg_values(func_op: Any) -> list[Any]:
    return list(func_op.regions[0].blocks[0].arguments)


def shape_of(value: Any) -> tuple[int, ...]:
    """Return the static shape of an SSA value typed as RankedTensorType."""
    return tuple(value.type.shape)


def elem_dtype_of(value: Any) -> str:
    """Return a normalized element-type string (`"f32"`, `"i8"`, …)."""
    return str(value.type.element_type)


def is_ranked_tensor(value: Any) -> bool:
    ty = value.type
    return hasattr(ty, "shape") and hasattr(ty, "element_type")


def walk_inner_ops(func_op: Any):
    """Yield every op inside `func.func`'s entry block."""
    for region in func_op.operation.regions:
        for block in region.blocks:
            yield from block.operations


def has_op_in_func(func_op: Any, op_name: str) -> bool:
    for op in walk_inner_ops(func_op):
        if op.operation.name == op_name:
            return True
    return False


def has_any_op_in_func(func_op: Any, op_names: tuple[str, ...]) -> bool:
    return any(has_op_in_func(func_op, n) for n in op_names)


def find_named_op(func_op: Any, op_name: str) -> Any | None:
    for op in walk_inner_ops(func_op):
        if op.operation.name == op_name:
            return op
    return None


def find_named_ops(func_op: Any, op_name: str) -> list[Any]:
    return [op for op in walk_inner_ops(func_op) if op.operation.name == op_name]


def linalg_generic_body_op_names(generic_op: Any) -> set[str]:
    """Return the set of op names appearing inside a `linalg.generic`'s
    body region (excluding `linalg.yield`)."""
    names: set[str] = set()
    for region in generic_op.operation.regions:
        for block in region.blocks:
            for op in block.operations:
                if op.name == "linalg.yield":
                    continue
                names.add(op.name)
    return names


def find_tensor_constants(func_op: Any, *, rank: int, dtype: str) -> list[Any]:
    """Return every `arith.constant` op in `func_op` whose result is a
    ranked tensor of the given rank and element-type string."""
    out: list[Any] = []
    for op in walk_inner_ops(func_op):
        if op.operation.name != "arith.constant":
            continue
        res = op.results[0]
        if not is_ranked_tensor(res):
            continue
        if len(res.type.shape) == rank and str(res.type.element_type) == dtype:
            out.append(op)
    return out


def splat_constant_value(constant_op: Any) -> float | int | None:
    """Return the splat scalar of a tensor `arith.constant`, or None.

    Uses the structured `DenseElementsAttr.get_splat_value()` API; the
    returned typed attr is downcast to `FloatAttr` / `IntegerAttr` to
    extract the Python scalar. Returns None for non-splat dense attrs.
    """
    if "value" not in constant_op.attributes:
        return None
    v = constant_op.attributes["value"]
    if not getattr(v, "is_splat", False):
        return None
    splat = v.get_splat_value()
    # `splat` is either an `IntegerAttr` or a `FloatAttr`; both expose
    # `.value` after concrete-typing. We try float first because integer
    # element types still produce integer-typed splat attrs.
    try:
        return ir.IntegerAttr(splat).value
    except (ValueError, TypeError):
        pass
    try:
        return ir.FloatAttr(splat).value
    except (ValueError, TypeError):
        pass
    return None


def parse_dense_2d_attr(op: Any, attr_name: str) -> tuple[int, int] | None:
    """Read a `dense<[h,w]>` or `dense<N>` attribute as a (h, w) pair.

    Uses the `DenseElementsAttr` structural API: `is_splat` +
    `get_splat_value()` for splat dense attrs, otherwise indexed access
    via `[i]` (the bindings expose iteration through `__getitem__` +
    `__len__`, not `__iter__`).
    """
    if attr_name not in op.attributes:
        return None
    attr = op.attributes[attr_name]
    is_splat = getattr(attr, "is_splat", None)
    if is_splat is None:
        return None
    if is_splat:
        splat = attr.get_splat_value()
        try:
            v = int(ir.IntegerAttr(splat).value)
        except (ValueError, TypeError):
            return None
        return (v, v)
    # Non-splat: read via DenseElementsAttr type.shape (the underlying
    # tensor shape) and indexed access.
    try:
        n = len(attr)
    except TypeError:
        return None
    if n == 2:
        try:
            return (int(attr[0]), int(attr[1]))
        except (TypeError, ValueError):
            return None
    return None


def dict_attr_items(dict_attr: Any) -> dict[str, Any]:
    """Return a `DictAttr` as a Python `{name: typed_attr}` dict."""
    return {dict_attr[i].name: dict_attr[i].attr for i in range(len(dict_attr))}


def parse_qparams_attr(func_op: Any, attr_name: str):
    """Pull a `merlin.qnn.<attr_name> = {scale, offset}` dict off the func
    attrs as a `QuantParams`. Returns `None` when missing or malformed.
    """
    full_name = f"merlin.qnn.{attr_name}"
    if full_name not in func_op.attributes:
        return None
    dict_attr = func_op.attributes[full_name]
    items = dict_attr_items(dict_attr)
    if "scale" not in items or "offset" not in items:
        return None
    try:
        scale = ir.FloatAttr(items["scale"]).value
        offset = ir.IntegerAttr(items["offset"]).value
    except (ValueError, TypeError):
        return None
    from qnn_ir import QuantParams  # noqa: PLC0415  - lazy

    return QuantParams(scale=float(scale), offset=int(offset))


def integer_attr_value(op: Any, attr_name: str) -> int | None:
    """Read an `IntegerAttr` from `op.attributes[attr_name]`. Returns None
    if missing or not an integer attribute."""
    if attr_name not in op.attributes:
        return None
    try:
        return ir.IntegerAttr(op.attributes[attr_name]).value
    except (ValueError, TypeError):
        return None


def dense_to_bytes(constant_op: Any, dtype: str) -> bytes | None:
    """Extract a tensor `arith.constant`'s payload as raw little-endian
    bytes, regardless of splat-vs-non-splat representation.

    Iterates the `DenseElementsAttr` element-by-element via the
    bindings' indexed-access API (`attr[i]`); splat attrs are expanded
    by replicating the splat value across the tensor's element count.
    `dtype` selects the binary encoding:

      - ``"i8"`` / ``"u8"``  →  one byte per element (little-endian; for
        signed values the bit pattern matches a two's-complement i8).
      - ``"f32"``            →  4 little-endian bytes per element.
      - ``"i32"``            →  4 little-endian bytes per element.
    """
    import struct  # noqa: PLC0415  - lazy

    if "value" not in constant_op.attributes:
        return None
    attr = constant_op.attributes["value"]
    res_ty = constant_op.results[0].type
    if not hasattr(res_ty, "shape"):
        return None
    shape = list(res_ty.shape)
    n = 1
    for d in shape:
        n *= int(d)

    is_splat = getattr(attr, "is_splat", False)
    if is_splat:
        splat = attr.get_splat_value()
        try:
            try:
                v: int | float = ir.IntegerAttr(splat).value
            except (ValueError, TypeError):
                v = ir.FloatAttr(splat).value
        except (ValueError, TypeError):
            return None
        values: list[int | float] = [v] * n
    else:
        try:
            length = len(attr)
        except TypeError:
            return None
        if length != n:
            return None
        try:
            values = [attr[i] for i in range(n)]
        except (TypeError, IndexError):
            return None

    if dtype in ("i8", "int8"):
        return bytes((int(v) & 0xFF) for v in values)
    if dtype in ("u8", "uint8", "ui8"):
        return bytes(int(v) & 0xFF for v in values)
    if dtype in ("i32", "int32"):
        return struct.pack(f"<{n}i", *(int(v) for v in values))
    if dtype in ("f32", "float32"):
        return struct.pack(f"<{n}f", *(float(v) for v in values))
    return None
