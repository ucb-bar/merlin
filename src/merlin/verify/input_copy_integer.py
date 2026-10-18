"""Exact vectorized integer instructions for bounded copy-proof domains.

Only control/address integers enter here. Signed and unsigned no-wrap flags
are checked on every active lane before narrowing; unsupported widths and
operations refuse. Conservative host-i64 range bounds avoid NumPy overflow.
"""

import numpy as np
from xdsl.dialects.builtin import IntegerAttr, IntegerType, UnitAttr
from xdsl.dialects.llvm import ICmpPredicateFlag

from merlin.perf.constant_control_cfg import prove_cast_flags


def _width(value):
    if not isinstance(value.type, IntegerType) or not 1 <= value.type.width.data <= 64:
        raise ValueError("batched copy needs a bounded integer type")
    return value.type.width.data


def _unsigned(value, width):
    return value.view(np.uint64) if width == 64 else value & ((1 << width) - 1)


def _wrap(value, width):
    if width == 64:
        return value
    unsigned = value & ((1 << width) - 1)
    return np.where(unsigned >= 1 << (width - 1), unsigned - (1 << width), unsigned)


def _flags(op):
    result = 0
    for key in ("overflowFlags", "overflow_flags", "isExact", "isDisjoint"):
        attribute = op.properties.get(key)
        if attribute is None:
            continue
        if key == "isDisjoint" and isinstance(attribute, UnitAttr) and op.name == "llvm.or":
            continue  # The active-lane proof in integer() is mandatory before this helper.
        if isinstance(attribute, IntegerAttr):
            flags = attribute.value.data
        elif isinstance(getattr(attribute, "data", None), frozenset):
            names = {value.value for value in attribute.data}
            if names - {"nsw", "nuw"}:
                raise ValueError("batched copy has unsupported integer poison flags")
            flags = (1 if "nsw" in names else 0) | (2 if "nuw" in names else 0)
        else:
            raise ValueError("batched copy has unsupported integer poison flags")
        if flags and (key != "overflowFlags" or flags not in (1, 2, 3)):
            raise ValueError("batched copy has unsupported integer poison flags")
        result |= flags
    return result


def integer(op, args, mask):
    """Evaluate one integer operation without accepting opaque/tensor operands."""
    if any(not isinstance(value, (int, np.ndarray, np.integer)) for value in args):
        raise ValueError("tensor byte or pointer used in batched scalar arithmetic/control")
    if any(isinstance(value, np.ndarray) and value.dtype.kind not in "iu" for value in args):
        raise ValueError("batched copy has noninteger scalar data")
    width = _width(op.operands[0])
    shape = np.broadcast_shapes(np.shape(mask), *(np.shape(value) for value in args))
    selected = np.broadcast_to(mask, shape)
    arrays = [np.broadcast_to(np.asarray(value, dtype=np.int64), shape) for value in args]
    name = op.op_name.data if op.name == "builtin.unregistered" else op.name
    if name == "llvm.select":
        return np.where(arrays[0] != 0, arrays[1], arrays[2])
    disjoint = op.properties.get("isDisjoint")
    if isinstance(disjoint, UnitAttr):
        if name != "llvm.or" or len(arrays) != 2:
            raise ValueError("batched operation has an unsupported disjoint poison flag")
        if np.any((_unsigned(arrays[0], width) & _unsigned(arrays[1], width))[selected]):
            raise ValueError("batched copy violates disjoint-or poison flag")
    flags = _flags(op)
    if name == "llvm.icmp":
        predicate = ICmpPredicateFlag.from_int(op.properties["predicate"].value.data).value
        a, b = arrays if predicate.startswith("s") else [_unsigned(value, width) for value in arrays]
        operations = {"eq": np.equal, "ne": np.not_equal, "slt": np.less, "ult": np.less,
                      "sle": np.less_equal, "ule": np.less_equal, "sgt": np.greater,
                      "ugt": np.greater, "sge": np.greater_equal, "uge": np.greater_equal}
        return operations[predicate](a, b).astype(np.int64)
    if name in {"llvm.sext", "llvm.zext", "llvm.trunc"}:
        unsigned = _unsigned(arrays[0], width)[selected]
        signed = arrays[0][selected]
        # Both supported predicates are interval constraints. The actual
        # extrema cover every active lane, without trusting the flags.
        for u, s in ((int(unsigned.min()), int(signed.min())),
                     (int(unsigned.max()), int(signed.max()))):
            prove_cast_flags(op, u, s)
        value = _unsigned(arrays[0], width) if name == "llvm.zext" else arrays[0]
        if value.dtype == np.uint64 and np.any(value[selected] > (1 << 63) - 1):
            raise ValueError("batched integer conversion exceeds exact host range")
        return _wrap(value.astype(np.int64), _width(op.results[0]))
    if name in {"llvm.intr.umin", "llvm.intr.umax", "llvm.intr.smin", "llvm.intr.smax"}:
        if flags:
            raise ValueError("batched integer minimum/maximum has unproved poison flags")
        a, b = [_unsigned(value, width) for value in arrays] if name in {
            "llvm.intr.umin", "llvm.intr.umax"
        } else arrays
        value = np.minimum(a, b) if name.endswith("min") else np.maximum(a, b)
        return _wrap(value.astype(np.int64), _width(op.results[0]))
    if len(arrays) != 2:
        raise ValueError("batched copy has unsupported scalar operation")
    a, b = arrays
    low_a, high_a = int(a[selected].min()), int(a[selected].max())
    low_b, high_b = int(b[selected].min()), int(b[selected].max())
    if name == "llvm.add":
        low, high = low_a + low_b, high_a + high_b
        execute = np.add
    elif name == "llvm.sub":
        low, high = low_a - high_b, high_a - low_b
        execute = np.subtract
    elif name == "llvm.mul":
        bounds = [x * y for x in (low_a, high_a) for y in (low_b, high_b)]
        low, high = min(bounds), max(bounds)
        execute = np.multiply
    elif name == "llvm.shl":
        if not 0 <= low_b <= high_b < width or low_a < 0:
            raise ValueError("batched copy shift is poison or unsupported")
        low, high = low_a << low_b, high_a << high_b
        execute = np.left_shift
    elif name in {"llvm.and", "llvm.or", "llvm.xor"}:
        if flags:
            raise ValueError("batched bitwise operation has poison flags")
        return {"llvm.and": np.bitwise_and, "llvm.or": np.bitwise_or,
                "llvm.xor": np.bitwise_xor}[name](a, b)
    else:
        raise ValueError("batched copy has unsupported scalar operation")
    if not -(1 << 63) <= low <= high < 1 << 63:
        raise ValueError("batched copy arithmetic exceeds exact host integer range")
    value = execute(a, b)
    if flags & 1 and (np.any(value[selected] < -(1 << (width - 1)))
                      or np.any(value[selected] >= 1 << (width - 1))):
        raise ValueError("batched copy violates signed no-wrap poison flags")
    if flags & 2:
        unsigned_a, unsigned_b = _unsigned(a, width), _unsigned(b, width)
        amin, amax = int(unsigned_a[selected].min()), int(unsigned_a[selected].max())
        bmin, bmax = int(unsigned_b[selected].min()), int(unsigned_b[selected].max())
        if name == "llvm.add":
            lower, upper = amin + bmin, amax + bmax
        elif name == "llvm.sub":
            lower, upper = amin - bmax, amax - bmin
        elif name == "llvm.mul":
            lower, upper = amin * bmin, amax * bmax
        elif name == "llvm.shl":
            lower, upper = amin << low_b, amax << high_b
        else:
            raise ValueError("batched operation has unproved unsigned poison flags")
        if lower < 0 or upper >= 1 << width:
            raise ValueError("batched copy violates unsigned no-wrap poison flags")
    return _wrap(value, _width(op.results[0]))
