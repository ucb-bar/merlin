"""Exact integer GEMM patterns in upstream linalg-on-tensors IR.

Only dense row-major, from-zero i8 x i8 -> i32 contractions match.  A
provenance label is never sufficient: indexing maps, iterator kinds, the
integer region body, operand shapes and accumulator initialization are checked.
"""

from __future__ import annotations

from dataclasses import dataclass

from xdsl.dialects.builtin import TensorType
from xdsl.ir import Operation, SSAValue


@dataclass(frozen=True)
class IntegerGemm:
    batch: int
    m: int
    n: int
    k: int


_MAPS_2D = ((0, 2), (2, 1), (0, 1))
_MAPS_3D = ((0, 1, 3), (0, 3, 2), (0, 1, 2))


def _shape(value: SSAValue, dtype: str) -> tuple[int, ...] | None:
    ty = value.type
    if not isinstance(ty, TensorType) or str(ty.get_element_type()) != dtype:
        return None
    return tuple(int(x) for x in ty.get_shape())


def _zero_init(value: SSAValue) -> bool:
    producer = value.owner
    if not isinstance(producer, Operation) or producer.name not in ("linalg.fill", "tensor.splat"):
        return False
    scalar = producer.operands[0]
    src = scalar.owner
    if not isinstance(src, Operation) or src.name != "arith.constant":
        return False
    attr = src.properties.get("value")
    if attr is None:
        attr = src.attributes.get("value")
    data = getattr(getattr(attr, "value", None), "data", None)
    return data is not None and int(data) == 0


def _canonical_maps(op: Operation, rank: int) -> bool:
    maps = op.properties.get("indexing_maps")
    iters = op.properties.get("iterator_types")
    expected = _MAPS_2D if rank == 2 else _MAPS_3D
    if maps is None or iters is None or len(maps.data) != 3 or len(iters.data) != rank + 1:
        return False
    if [x.data.value for x in iters.data] != ["parallel"] * rank + ["reduction"]:
        return False
    for amap, indices in zip(maps.data, expected):
        data = amap.data
        if data.num_dims != rank + 1 or data.num_symbols != 0:
            return False
        if tuple(getattr(expr, "position", None) for expr in data.results) != indices:
            return False
    return True


def _canonical_body(op: Operation) -> bool:
    if len(op.regions) != 1 or len(op.regions[0].blocks) != 1:
        return False
    block = op.regions[0].blocks[0]
    ops = list(block.ops)
    if [x.name for x in ops] != ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"]:
        return False
    if len(block.args) != 3 or [str(x.type) for x in block.args] != ["i8", "i8", "i32"]:
        return False
    ext_a, ext_b, mul, add, result = ops
    return (
        list(ext_a.operands) == [block.args[0]]
        and list(ext_b.operands) == [block.args[1]]
        and str(ext_a.results[0].type) == "i32"
        and str(ext_b.results[0].type) == "i32"
        and list(mul.operands) == [ext_a.results[0], ext_b.results[0]]
        and str(mul.results[0].type) == "i32"
        and set(add.operands) == {mul.results[0], block.args[2]}
        and str(add.results[0].type) == "i32"
        and list(result.operands) == [add.results[0]]
    )


def match_integer_gemm(op: Operation) -> IntegerGemm | None:
    """Return logical batch/M/N/K only when this op has exact GEMM semantics."""
    if op.name not in ("linalg.matmul", "linalg.generic") or len(op.operands) != 3 or len(op.results) != 1:
        return None
    lhs = _shape(op.operands[0], "i8")
    rhs = _shape(op.operands[1], "i8")
    init = _shape(op.operands[2], "i32")
    out = _shape(op.results[0], "i32")
    if lhs is None or rhs is None or init is None or out != init or not _zero_init(op.operands[2]):
        return None
    if op.name == "linalg.matmul":
        if len(lhs) != 2 or len(rhs) != 2:
            return None
    else:
        if len(lhs) not in (2, 3) or len(rhs) != len(lhs) or not _canonical_maps(op, len(lhs)):
            return None
        if not _canonical_body(op):
            return None
    if len(lhs) == 2:
        m, k = lhs
        rk, n = rhs
        batch = 1
        expected_out = (m, n)
    else:
        batch, m, k = lhs
        rb, rk, n = rhs
        if rb != batch:
            return None
        expected_out = (batch, m, n)
    if min(batch, m, n, k) <= 0 or k != rk or out != expected_out:
        return None
    return IntegerGemm(batch, m, n, k)
