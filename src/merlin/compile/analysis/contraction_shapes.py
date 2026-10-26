"""Observe the CONTRACTION SHAPES a module carries, and the extents a cover reads off them.

A register block, a tile cut or an output-fusion decision can only be legal for a contraction whose
parallel and reduction extents somebody has read. That is this module: a structural read of a
linalg-on-tensors module into :class:`ContractionShape`, plus :func:`contraction_extents`, the
``id(op) -> (m, k, n, rank)`` table region coverage, device grouping and output fusion consume.
It reads only the IR; no target, schedule or measured choice enters it.

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

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContractionShape:
    """One contraction a cover is asked to realize, named by ITERATOR TYPE.

    Deliberately NOT "M/N/K": a target-agnostic policy only needs to know which extents belong to
    PARALLEL iteration dims (the ones a register block tiles and whose tail must be masked) and which
    belong to REDUCTION dims (whose tail is harmless). ``op`` is the op class the target schedules
    separately (``linalg.matmul`` vs ``linalg.batch_matmul`` get independent tile sizes), so a policy
    can solve each class on its own instead of forcing one blocking on both.

    ``parallel`` is outer-to-inner as the op writes them (matmul -> (M, N); batch_matmul ->
    (B, M, N)), which is the order a target's tile-size vector uses.

    ``dtypes`` carries the element types as MLIR tokens, positionally ``(lhs, rhs, out)`` — the same
    triple a target contract's accumulate rule is written in (``{in, weight, acc}``), so a legality
    question can be asked of a contraction without re-reading the module. It is ``()`` when the
    observer could not read them, which is the honest reading for a synthetic shape or a policy that
    predates the field; a consumer that needs a dtype must therefore handle absence rather than
    assume one. Extents alone cannot answer legality on a unit that computes int8 and not fp32, and
    the accumulator token is what decides whether a reduction overflows.
    """

    op: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...] = ()
    dtypes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for f in ("parallel", "reduction"):
            v = getattr(self, f)
            if not isinstance(v, tuple) or any(int(x) <= 0 for x in v):
                raise ValueError(f"{f} must be a tuple of positive extents, got {v!r}")
        if not isinstance(self.dtypes, tuple) or any(not str(d) for d in self.dtypes):
            raise ValueError(f"dtypes must be a tuple of non-empty tokens, got {self.dtypes!r}")

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


def zero_initialised(op) -> bool:
    """Whether ``op``'s accumulator init is provably an all-zero fill.

    The microkernel writes its output rather than adding to it, so routing a contraction whose init
    carries live values would DROP that addend — a wrong answer, not a slow one. Answering "no" for
    anything not recognised is the fail-closed direction: an unrecognised init might be zero, and a
    contraction wrongly left on the vector path is merely slower.

    Structural: the init operand's defining op must be ``linalg.fill`` or ``tensor.splat``
    of an ``arith.constant`` zero, integer OR float. A block argument (whose ``owner`` is a
    ``Block``, not an ``Operation``) is rejected by the same check.

    The float case is not cosmetic. This predicate is a CORRECTNESS condition -- it decides whether the
    addend may be dropped -- and "is the init all zeros" has nothing to do with the element type. While
    it recognised integers only, every float contraction answered "no", so a float accelerator could
    never take a single layer through this path no matter what its datapath declared. That reads as a
    device with no work rather than as a predicate that cannot see it.
    """
    from xdsl.ir import Operation

    operands = list(getattr(op, "operands", ()))
    if len(operands) < 3:
        return False
    fill = operands[2].owner
    if not isinstance(fill, Operation) or getattr(fill, "name", "") not in {"linalg.fill", "tensor.splat"}:
        return False
    fill_operands = list(fill.operands)
    if not fill_operands:
        return False
    const = fill_operands[0].owner
    if not isinstance(const, Operation) or getattr(const, "name", "") != "arith.constant":
        return False
    return _is_zero_attr(getattr(const, "value", None))


def _is_zero_attr(attr) -> bool:
    """Whether ``attr`` is a zero constant of any numeric kind. Unrecognised => False (fail closed).

    Both signed zeros count: dropping a ``-0.0`` addend changes no result, since ``x + -0.0 == x`` for
    every finite ``x`` and ``0.0 + -0.0 == 0.0``. A NaN or a non-numeric attribute is not zero and is
    refused by the comparison itself.
    """
    if _int_attr_value(attr) == 0:
        return True
    value = getattr(attr, "value", None)
    raw = getattr(value, "data", value)
    if isinstance(raw, float):
        return raw == 0.0
    return False


def _int_attr_value(attr) -> int | None:
    """The integer an ``IntegerAttr`` carries, or None when it is not one.

    ``IntegerAttr`` nests its payload (``attr.value.data``); reaching through it explicitly rather than
    parsing ``str(attr)`` means a float or index attribute returns None instead of being misread.
    """
    inner = getattr(attr, "value", None)
    data = getattr(inner, "data", None)
    return data if isinstance(data, int) else None


def _shaped(value) -> "tuple[list[int], str] | None":
    """``(shape, dtype)`` of a shaped operand, or None when it is not one.

    The dtype is kept rather than dropped: it is the other half of a legality question (a unit that
    computes int8 and not fp32 rejects on element type, never on extents) and it names the
    accumulator, which is what decides whether a reduction overflows.
    """
    from merlin.common.mlir_query import type_shape_dtype

    try:
        shape, dtype = type_shape_dtype(value.type)
    except Exception:  # noqa: BLE001 — a non-shaped operand is simply not a contraction operand
        return None
    if not shape or not all(int(d) > 0 for d in shape):
        return None
    return list(shape), str(dtype)


def _shape_of(value) -> "list[int] | None":
    got = _shaped(value)
    return None if got is None else got[0]


def _iterator_types(op) -> "list[str] | None":
    """``["parallel", ..., "reduction"]`` for a ``linalg.generic``, or None when unreadable."""
    from merlin.common.mlir_query import _attr_tables

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


def indexing_maps(op) -> "list[list[Any]] | None":
    """``indexing_maps`` of a linalg op as one result-expression list per operand, or None.

    Read through the attribute tables rather than a typed accessor so it works on a module parsed with
    unregistered dialects as well as on a fully-typed xDSL ``GenericOp``. None means "could not read",
    which is NOT the same as "no maps" — callers must fail closed on it rather than proceed.
    """
    from merlin.common.mlir_query import _attr_tables

    maps = getattr(op, "indexing_maps", None)
    if maps is None:
        for table in _attr_tables(op):
            if "indexing_maps" in table:
                maps = table["indexing_maps"]
                break
    if maps is None:
        return None
    out: list[list[Any]] = []
    for m in getattr(maps, "data", ()):
        results = getattr(getattr(m, "data", None), "results", None)
        if results is None:
            return None
        out.append(list(results))
    return out or None


def _dim_position(expr) -> "int | None":
    """The iteration dim an affine map RESULT names, or None when the result is not a bare dim."""
    from xdsl.ir.affine import AffineDimExpr

    return int(expr.position) if isinstance(expr, AffineDimExpr) else None


def _generic_contraction(op) -> "ContractionShape | None":
    """A ``linalg.generic`` read as a contraction, or None when it is not one.

    Contraction test (structural): 2 shaped inputs + 1 shaped output, iterator types are some
    parallel dims followed by exactly one reduction dim, the output rank equals the number of
    parallel dims — i.e. the output covers every parallel dim, which is what makes the trailing pair
    the register-blockable (M, N) — and the two INPUT MAPS both contract over the reduction dim, with
    the A operand carrying it LAST.

    That last clause is not decoration; it is the assumption the reduction extent is read under. The
    test used to be ranks only, and a rank is not enough: an ``aten.bucketize`` boundary search has
    the identical signature — iterator types ``[parallel, parallel, reduction]``, output rank 2, and a
    rank-2 A operand — but its A map is ``(d0, d1, d2) -> (d0, d1)``, which does not mention the
    reduction dim at all. It was therefore reported as ``linalg.matmul`` with parallel (1, 32) and a
    reduction extent of 32 read off A's LAST dim, which is N, not K; its real reduction extent is the
    31 of its rank-1 boundary operand.

    A phantom is not a harmless over-report, because the block policy and the TAGGER are downstream of
    two different things: the policy prices this list, and the tagger runs after
    ``linalg-specialize-generic-ops``, which (correctly) refuses to name a bucketize a matmul. So a
    phantom is priced and can never be tagged, and ``perop_blocks.BlockAgreementError`` fails the whole
    build — MEASURED on smolvla int8: ``1 contraction(s) were priced by the block policy but not
    tagged: ['linalg.matmul:1x32:32']``, with the tagger reporting no untagged geometry at all because
    the op it was asked about is not a named contraction anywhere in the module.

    Checking the maps is what makes this observer classify the way the specialization pass does, which
    is what the two sides have to agree on. It is derived, not a shape allowlist: no extent appears in
    the test, so an M=1 contraction that IS one still prices and still tags.
    """
    its = _iterator_types(op)
    if not its or its.count("reduction") != 1 or its[-1] != "reduction":
        return None
    n_par = len(its) - 1
    if n_par < 2:
        return None
    ins = [s for s in (_shaped(v) for v in op.operands) if s is not None]
    outs = [s for s in (_shaped(v) for v in getattr(op, "results", ())) if s is not None]
    if len(ins) < 3 or not outs:
        return None
    out, out_dtype = outs[0]
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
    a, a_dtype = ins[0]
    if len(a) != n_par:
        return None
    # ... and the rank is STILL not enough (see the docstring): both inputs must actually contract
    # over the reduction dim, and A must carry it in the position the extent is read from. Unreadable
    # maps fail CLOSED — an observer that cannot see the maps has not verified anything, and pricing
    # an unverified op is what produced the un-taggable phantom.
    maps = indexing_maps(op)
    if maps is None or len(maps) < 3:
        return None
    red = n_par  # the reduction dim's position (it is the last one)
    if not maps[0] or _dim_position(maps[0][-1]) != red:
        return None
    if all(_dim_position(r) != red for r in maps[1]):
        return None
    return ContractionShape(
        op=op_class,
        parallel=tuple(int(d) for d in out),
        reduction=(int(a[-1]),),
        dtypes=(a_dtype, ins[1][1], out_dtype),
    )


def _named_contraction(op, name: str) -> "ContractionShape | None":
    outs = [s for s in (_shaped(v) for v in getattr(op, "results", ())) if s is not None]
    ins = [s for s in (_shaped(v) for v in op.operands) if s is not None]
    if not outs or not ins:
        return None
    (out, out_dtype), (a, a_dtype) = outs[0], ins[0]
    batch = _NAMED[name]
    if len(out) != batch + 2 or len(a) != batch + 2:
        return None
    # A named contraction always has a second shaped input (the RHS); when the printed form hides it
    # the dtype triple stays SHORT rather than repeating the LHS, so a consumer sees "not observed"
    # instead of a fabricated weight dtype.
    dtypes = (a_dtype, ins[1][1], out_dtype) if len(ins) > 1 else ()
    return ContractionShape(op=name, parallel=tuple(int(d) for d in out), reduction=(int(a[-1]),), dtypes=dtypes)


def observe_contractions(src: "str | Path | Any") -> "list[tuple[Any, ContractionShape]]":
    """Every contraction in ``src`` as ``(op, shape)`` pairs, in program order.

    The op handle is returned alongside the shape because a caller that wants more than extents — its
    ``prov.*`` provenance, its indexing maps, its position — would otherwise have to walk the module a
    second time and re-derive which ops the classification accepted, and the two walks could disagree.
    :func:`contraction_shapes` is the shape-only projection of this.

    Returns an EMPTY list rather than raising when the module cannot be parsed — a shape observer that
    fails must degrade to "I observed nothing", which makes the caller fall back to the shape-blind
    realization instead of failing a build over an unreadable capture."""
    from merlin.common import mlir_query as mq

    try:
        module = mq.parse(src)  # accepts an already-parsed module, MLIR text, or a path
    except Exception:  # noqa: BLE001
        return []
    found: list[tuple[Any, ContractionShape]] = []
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
            found.append((op, cs))
    return found


def contraction_shapes(src: "str | Path | Any") -> list[ContractionShape]:
    """Every contraction in ``src`` (a path, MLIR text, or a parsed module), shapes included."""
    return [cs for _, cs in observe_contractions(src)]


def contraction_extents(module) -> dict:
    """``id(op) -> (m, k, n, rank)`` for every contraction in a parsed ``module``.

    Two readers feed one table, keyed by op IDENTITY so a consumer joins it against its own walk:

    * :func:`observe_contractions` for contractions whose operands are indexed by plain iteration
      dims (named matmuls and their generic forms);
    * :func:`windowed_contraction_extents` for the windowed contractions it passes over (a
      convolution in generic form indexes its input by ``stride * d + tap``).

    The readers are independent: an observer failure never removes the windowed extents, and the
    windowed reader never overrides an op the observer already measured. ``m`` is the product of the
    leading parallel extents, ``n`` the last parallel extent and ``k`` the product of the reduction
    extents. An op whose extents cannot be read is absent, which a consumer reports as unknown rather
    than as zero; the table is the shared authority, so no consumer re-derives "which axes are parallel
    and which reduce" and silently disagrees with it.
    """
    import math

    out: dict = {}
    for op, cs in observe_contractions(module):
        par, red = list(cs.parallel), list(cs.reduction)
        if not par:
            continue
        m = math.prod(par[:-1]) if len(par) > 1 else par[0]
        n = par[-1] if len(par) > 1 else None
        k = math.prod(red) if red else None
        out[id(op)] = (m, k, n, len(par) + len(red))
    windowed_contraction_extents(module, out)
    return out


def windowed_contraction_extents(module, out: dict | None = None) -> dict:
    """Extents for the contractions :func:`observe_contractions` does not accept, read off their loops.

    The observer recognises a contraction whose operands are indexed by plain iteration dims. A
    convolution in generic form indexes its input by ``stride * d + tap``, so it is a contraction by
    structure (a reduction whose body multiplies and accumulates) that the observer passes over. On
    a captured CNN that was every convolution: 53 of 54 contractions carried no extents, so offload
    weighted by work could not see the model's work at all.

    Multiply-accumulates are the product of the loop extents; the split follows the iterator kinds
    (``m`` the leading parallel loops, ``n`` the last, ``k`` the reduction loops). Ops already in
    ``out`` are kept as they are. An op whose loops cannot be read stays absent. Fills and returns
    ``out`` (a new table when omitted).
    """
    import math

    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    out = {} if out is None else out
    for op in mq.walk(module):
        if id(op) in out or mq.op_name(op) != "linalg.generic":
            continue
        try:
            if cc.classify_generic(op) != "contraction":
                continue
            extents = cc.loop_extents(op)
            kinds = op.properties.get("iterator_types") or op.attributes.get("iterator_types")
            kinds = [str(kind) for kind in kinds]
        except Exception:  # noqa: BLE001 - an unreadable op stays absent, never a guess
            continue
        if not extents or len(extents) != len(kinds) or set(extents) != set(range(len(kinds))):
            continue
        par = [extents[i] for i, kind in enumerate(kinds) if "reduction" not in kind]
        red = [extents[i] for i, kind in enumerate(kinds) if "reduction" in kind]
        if not par or not red:
            continue
        m = math.prod(par[:-1]) if len(par) > 1 else par[0]
        n = par[-1] if len(par) > 1 else 1
        out[id(op)] = (m, math.prod(red), n, len(kinds))
    return out
