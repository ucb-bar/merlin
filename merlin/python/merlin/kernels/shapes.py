"""Observe the CONTRACTION SHAPES a module asks a micro-kernel to cover.

The micro-kernel space (:mod:`merlin.kernels.microkernel`) can pick a register block that masks no
parallel dim — but only if somebody tells it what the parallel extents ARE. That is this module: a
structural read of a linalg-on-tensors module into :class:`~merlin.kernels.microkernel.ContractionShape`.

Two forms have to be recognized, because the lowering pipeline sees both:

  * NAMED contraction ops (``linalg.matmul`` / ``linalg.batch_matmul``) — what an fp32 capture
    carries all the way down;
  * contraction GENERICS — what the int8 (W8A8) rewrite leaves behind (``apply_quant`` rebuilds each
    matmul as a ``linalg.generic`` with i8xi8->i32 arithmetic in the body). The RVV pipeline recovers
    the named op with ``linalg-specialize-generic-ops`` right before the transform interpreter runs
    (see ``llvmlower.pipeline.build_rvv_pipeline``), so a shape observer that only looked for named
    ops would report ZERO contractions for every int8 whole model — precisely the case the shape
    policy exists for.

A generic is classified the way ``linalg-specialize-generic-ops`` classifies it: by ITERATOR TYPES
and the OUTPUT indexing map, never by op name or body pattern matching. The parallel extents are
read off the output operand's shape (the output map of a contraction is a projected permutation of
the parallel dims), and the reduction extents off the input dims the output map does not cover. That
keeps the reader target-agnostic and independent of the dtype the rewrite chose.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .microkernel import ContractionShape

#: Named ops whose iteration space is (parallel..., reduction...) with the LAST TWO parallel dims
#: being the register-blocked pair. Value = number of leading BATCH (parallel) dims.
_NAMED: dict[str, int] = {
    "linalg.matmul": 0,
    "linalg.batch_matmul": 1,
}

#: How many batch dims -> the named op a contraction generic of that rank specializes to. This is
#: the same mapping ``linalg-specialize-generic-ops`` applies, kept as data so the policy sees the
#: op class the SCHEDULE will match rather than "linalg.generic".
_GENERIC_AS: dict[int, str] = {0: "linalg.matmul", 1: "linalg.batch_matmul"}


def _shape_of(value) -> "list[int] | None":
    from ..common.mlir_query import type_shape_dtype
    try:
        shape, _ = type_shape_dtype(value.type)
    except Exception:  # noqa: BLE001 — a non-shaped operand is simply not a contraction operand
        return None
    return list(shape) if shape and all(int(d) > 0 for d in shape) else None


def _iterator_types(op) -> "list[str] | None":
    """``["parallel", ..., "reduction"]`` for a ``linalg.generic``, or None when unreadable."""
    from ..common.mlir_query import _attr_tables
    for table in _attr_tables(op):
        it = table.get("iterator_types")
        if it is None:
            continue
        out: list[str] = []
        for entry in getattr(it, "data", ()):
            # xDSL models each entry as an enum/string attribute; both expose the name via `data`.
            inner = getattr(entry, "data", entry)
            name = getattr(inner, "data", inner)
            out.append(str(name))
        return out or None
    return None


def _generic_contraction(op) -> "ContractionShape | None":
    """A ``linalg.generic`` read as a contraction, or None when it is not one.

    Contraction test (structural): 2 shaped inputs + 1 shaped output, iterator types are some
    parallel dims followed by exactly one reduction dim, and the output rank equals the number of
    parallel dims — i.e. the output covers every parallel dim, which is what makes the trailing pair
    the register-blockable (M, N)."""
    its = _iterator_types(op)
    if not its or its.count("reduction") != 1 or its[-1] != "reduction":
        return None
    n_par = len(its) - 1
    if n_par < 2:
        return None
    ins = [s for s in (_shape_of(v) for v in op.operands) if s is not None]
    outs = [s for s in (_shape_of(v) for v in getattr(op, "results", ())) if s is not None]
    if len(ins) < 3 or not outs:
        return None
    out = outs[0]
    if len(out) != n_par:
        return None
    op_class = _GENERIC_AS.get(n_par - 2)
    if op_class is None:
        return None
    # The single reduction extent is the input dim the output does not carry: for both classes the
    # A operand is (batch..., M, K), so its last dim is K and its RANK equals the parallel-dim count
    # (2 for a matmul, 3 for a batch_matmul). Requiring that rank is what separates a contraction
    # from a reduce-with-broadcast generic that merely shares the iterator-type signature — without
    # it bitvla reported 33 "contractions" where `linalg-specialize-generic-ops` finds 19, and the
    # 14 phantoms (parallel (1, 32)) dragged the derived M block down to 1.
    a = ins[0]
    if len(a) != n_par:
        return None
    return ContractionShape(op=op_class, parallel=tuple(int(d) for d in out),
                            reduction=(int(a[-1]),))


def _named_contraction(op, name: str) -> "ContractionShape | None":
    outs = [s for s in (_shape_of(v) for v in getattr(op, "results", ())) if s is not None]
    ins = [s for s in (_shape_of(v) for v in op.operands) if s is not None]
    if not outs or not ins:
        return None
    out, a = outs[0], ins[0]
    batch = _NAMED[name]
    if len(out) != batch + 2 or len(a) != batch + 2:
        return None
    return ContractionShape(op=name, parallel=tuple(int(d) for d in out),
                            reduction=(int(a[-1]),))


def contraction_shapes(src: "str | Path | Any") -> list[ContractionShape]:
    """Every contraction in ``src`` (a path, MLIR text, or a parsed module), shapes included.

    Returns an EMPTY list rather than raising when the module cannot be parsed — a shape observer
    that fails must degrade to "I observed nothing", which makes the caller fall back to the
    shape-blind realization instead of failing a build over an unreadable capture."""
    from ..common import mlir_query as mq
    try:
        module = mq.parse(src)   # accepts an already-parsed module, MLIR text, or a path
    except Exception:  # noqa: BLE001
        return []
    found: list[ContractionShape] = []
    for op in mq.walk(module):
        name = mq.op_name(op)
        try:
            if name in _NAMED:
                cs = _named_contraction(op, name)
            elif name == "linalg.generic":
                cs = _generic_contraction(op)
            else:
                continue
        except Exception:  # noqa: BLE001
            continue
        if cs is not None:
            found.append(cs)
    return found
