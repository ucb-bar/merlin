"""Which contractions did the matcher miss, and what do they cost.

A contraction that stays a `linalg.generic` never becomes `linalg.matmul`, so it never reaches a
matrix unit AND it never appears in the denominator when we report "what fraction of contraction
work is routed". Both halves of that are silent. This module makes the miss countable.

MEASURED, and the reason this exists. On `spectformer_int8_full` 16 generics are contractions the
matcher did not recognise -- 8 with loops `(4, 196, 196, reduce 64)` and 8 with
`(4, 196, 64, reduce 196)`, i.e. attention's Q.K^T and scores.V, the canonical matrix-unit workload.
They are 157.4 MMAC against 1702.2 MMAC of `linalg.matmul` work, so **8.5% of the model's contraction
MACs were invisible**, and the published "88.3% of contraction cycles are routed" and "the unit is at
96.6% of its Amdahl ceiling" were both computed against the under-count.

THE COST IS SHAPE-DEPENDENT, so do not carry 8.5% around as a constant. The same miss happens on
`gemma2_2b_int8_full_seq8` -- 26 pairs, one per layer -- but at seq=8 the scores are 8x8 and it is
0.03% of that model's MACs. The gap is systematic; whether it matters is set by sequence length.

CLASSIFY STRUCTURALLY, NEVER BY TAG. 751 of that model's 843 generics carry no `prov.family` /
`role` / `aten` / `module` at all, and the tags that do exist mislead in four documented ways. What a
loop nest computes is legible from `iterator_types` (is any dimension reduced) plus the body's arith
ops (is the accumulation a multiply-add).

GET EXTENTS FROM THE INDEXING MAPS. The tempting shortcut -- "the reduced extent is the one in the
inputs that is missing from the output" -- is wrong whenever a contracted extent also appears in the
output. `scores.V` has output `4x196x64` and contracts over a *second* 196, so the shortcut finds
nothing and undercounts by 196x (it reported 79.1 MMAC instead of 157.4). MACs are the product of ALL
loop extents, parallel and reduction alike, each read off an operand that carries that iteration dim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import mlir_query as mq

_MUL = {"arith.mulf", "arith.muli"}
_ADD = {"arith.addf", "arith.addi"}
_MAX = {"arith.maxf", "arith.maximumf", "arith.maxsi", "arith.maxnumf"}
_ABS = {"math.absf", "math.fabs"}

#: op names whose MACs we can price directly, mapped to nothing -- membership is the point.
MATMUL_OPS = ("linalg.matmul", "linalg.batch_matmul", "linalg.quantized_matmul")


@dataclass(frozen=True)
class UnloweredContraction:
    """A generic that computes a contraction the matcher did not recognise."""

    result_type: str
    loop_extents: tuple[tuple[int, int], ...]
    macs: int


@dataclass
class CoverageReport:
    lowered_macs: int = 0
    unlowered: list[UnloweredContraction] = field(default_factory=list)
    #: generics whose extents could not be derived. Counted and surfaced, never dropped -- an
    #: unpriceable op is a hole in the denominator and must not read as a zero.
    unpriceable: list[str] = field(default_factory=list)
    labels: dict[str, int] = field(default_factory=dict)

    @property
    def unlowered_macs(self) -> int:
        return sum(u.macs for u in self.unlowered)

    @property
    def total_macs(self) -> int:
        return self.lowered_macs + self.unlowered_macs

    @property
    def unlowered_share(self) -> float:
        """Fraction of all contraction MACs the matcher never saw. 0.0 when there is no work."""
        return (self.unlowered_macs / self.total_macs) if self.total_macs else 0.0


def _body_op_names(op) -> set[str]:
    if not op.regions or not op.regions[0].blocks:
        return set()
    return {mq.op_name(o) for o in op.regions[0].blocks[0].ops}


def _iterator_text(op) -> str:
    return str(op.properties.get("iterator_types") or op.attributes.get("iterator_types") or "")


def classify_generic(op) -> str:
    """What this `linalg.generic` computes, by structure alone."""
    reduces = "reduction" in _iterator_text(op)
    body = _body_op_names(op)
    if reduces and (body & _MUL) and (body & _ADD):
        return "contraction"
    if reduces and (body & _MAX) and (body & _ABS):
        return "absmax"
    if reduces and (body & _MAX):
        return "max-reduction"
    if reduces and (body & _ADD):
        return "sum-reduction"
    if reduces:
        return "other-reduction"
    if not body - {"linalg.yield"}:
        return "movement"          # a copy or broadcast: no arithmetic at all
    return "elementwise"


def loop_extents(op) -> dict[int, int] | None:
    """Iteration-dim index -> extent, read off the operands through their indexing maps.

    Returns None when the maps are absent or carry an expression this does not understand, so the
    caller can record the op as unpriceable rather than silently treating it as zero work.
    """
    maps = op.properties.get("indexing_maps") or op.attributes.get("indexing_maps")
    if maps is None:
        return None
    operands = list(op.operands)
    extents: dict[int, int] = {}
    for i, mattr in enumerate(maps):
        if i >= len(operands):
            break
        shape, _ = mq.type_shape_dtype(operands[i].type)
        for j, expr in enumerate(getattr(mattr.data, "results", ())):
            pos = getattr(expr, "position", None)     # a plain dim expr; anything else is skipped
            if pos is not None and j < len(shape):
                extents.setdefault(int(pos), int(shape[j]))
    return extents or None


def _product(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


def contraction_coverage(module: Any) -> CoverageReport:
    """Price every contraction in `module`, split by whether the matcher recognised it."""
    rep = CoverageReport()

    for g in mq.walk(module, "linalg.generic"):
        label = classify_generic(g)
        rep.labels[label] = rep.labels.get(label, 0) + 1
        if label != "contraction":
            continue
        ext = loop_extents(g)
        rtype = str(g.results[0].type) if g.results else "<no result>"
        if ext is None:
            rep.unpriceable.append(rtype)
            continue
        rep.unlowered.append(UnloweredContraction(
            result_type=rtype,
            loop_extents=tuple(sorted(ext.items())),
            macs=_product(ext.values()),
        ))

    for name in MATMUL_OPS:
        for op in mq.walk(module, name):
            if not op.results:
                continue
            out_shape, _ = mq.type_shape_dtype(op.results[0].type)
            ins = [mq.type_shape_dtype(o.type)[0] for o in op.operands]
            # A matmul contracts the last dimension of its first operand; that is the one extent the
            # result type does not carry.
            k = ins[0][-1] if ins and ins[0] else 1
            rep.lowered_macs += _product(out_shape) * int(k)

    return rep
