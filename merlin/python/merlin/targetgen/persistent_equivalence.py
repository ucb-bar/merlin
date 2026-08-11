"""Keep both implementations of a contraction in the IR and defer the choice to a costed extraction.

Routing today commits early: :func:`routing.select` picks a unit and the alternative is gone. That is
fine when the cost of each option is known at the moment of choosing, and wrong when a later pass changes
it — whether the epilogue can stay accumulator-resident, whether a quantize/dequantize pair fuses, and
whether the operands are already in the layout the unit wants are all decided *downstream* of the point
where routing currently commits.

xDSL 0.68 ships an ``equivalence`` dialect and the eqsat passes; this uses them for exactly one bounded
thing: **one contraction, with every legal implementation retained as an alternative in an e-class, and
the selection made by ``eqsat-extract`` against a cost model.** Bounded deliberately — the source paper
(Tamagoyaki, arXiv:2602.16707) restricts itself to pure straight-line functions and does not explore
structured control flow, so this covers a pure region and never a whole model and never control flow.

**What this does and does not establish.** With one e-class and no rewrite rules, extraction is an argmin
over the same costs :func:`routing.select` would have used, so it cannot make a better decision than
eager selection today, and claiming otherwise would be arithmetic dressed up as a result. What is
demonstrable, and is tested, is the *mechanism*: both implementations are present in the IR after
construction, and re-costing the same graph changes the extracted choice — i.e. the graph genuinely
carries both options rather than having quietly committed to one. That is the property a downstream pass
would need in order to influence the decision, and it is a precondition for H-EQ1 rather than evidence
for it.

**H-EQ2 (incremental re-saturation) is NOT exercised here.** Saturation needs rewrite rules to saturate
with; without them there is nothing to re-saturate, so ``saturate(E_parent, delta) == saturate(program,
parent | delta)`` is untested and :data:`HYPOTHESES` records it as such. The paper is silent on
incremental re-saturation, so it is not a result that can be inherited either.

Compile-time cost is reported as a first-class number, because the paper measures a 401x geomean slowdown
against egg and "the infrastructure retains both implementations; the performance benefit is not yet
established" has to stay sayable.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from . import routing as _routing

__all__ = ["COST_SCALE", "EGraphResult", "HYPOTHESES", "agreement", "alternatives_in",
           "build_egraph", "extract_choice", "hypothesis_status", "recost", "run_extraction",
           "select_by_extraction"]

#: Costs are attached as integers (the pass reads an ``IntAttr``), so a float cost is scaled and rounded.
#: 1000x keeps sub-cycle resolution, well past anything a cycle-level cost model distinguishes. Two
#: candidates whose scaled costs collide are a genuine tie and resolve by e-class operand order, which is
#: contract-declaration order — the same tie-break :func:`routing.select` uses.
COST_SCALE = 1000

#: What this prototype claims, and what it does not. Kept as data so a report cannot quietly promote a
#: hypothesis to a result: both entries are read by :func:`hypothesis_status` and asserted in the tests.
HYPOTHESES: dict[str, dict[str, Any]] = {
    "H-EQ1": {
        "claim": "persistent alternatives select better than early extraction",
        "status": "not_established",
        "why": ("with a single e-class and no rewrite rules, extraction is an argmin over the same costs "
                "eager selection uses, so it cannot decide differently. Establishing this needs a "
                "downstream pass that changes a candidate's cost after the graph is built; what is "
                "demonstrated here is the precondition — the alternatives survive in the IR and "
                "re-costing changes the choice"),
    },
    "H-EQ2": {
        "claim": "saturate(E_parent, delta) is equivalent to saturate(program, parent | delta)",
        "status": "not_exercised",
        "why": ("saturation requires rewrite rules; none are applied here, so there is nothing to "
                "re-saturate and no incremental-vs-scratch comparison to make. The source paper is "
                "silent on incremental re-saturation, so this cannot be inherited as a result either"),
    },
}


def hypothesis_status(name: str) -> str:
    """The recorded status of a hypothesis. Raises on an unknown name rather than returning 'unknown',
    which would let a typo read as a modest claim instead of a missing one."""
    if name not in HYPOTHESES:
        raise KeyError(f"no such hypothesis {name!r}; known: {sorted(HYPOTHESES)}")
    return str(HYPOTHESES[name]["status"])


@dataclass(frozen=True)
class EGraphResult:
    """The outcome of building and extracting from one contraction's e-graph."""

    chosen: str | None
    #: Every alternative that was retained, in the order it entered the e-class.
    alternatives: tuple[str, ...] = ()
    #: Scaled integer cost per alternative; absent for a candidate the cost model declined.
    costs: dict[str, int] = field(default_factory=dict)
    unscored: tuple[str, ...] = ()
    build_seconds: float = 0.0
    extract_seconds: float = 0.0
    gap: str | None = None

    @property
    def total_seconds(self) -> float:
        return self.build_seconds + self.extract_seconds

    def to_dict(self) -> dict[str, Any]:
        return {"chosen": self.chosen, "alternatives": list(self.alternatives),
                "costs": dict(self.costs), "unscored": list(self.unscored),
                "build_seconds": round(self.build_seconds, 6),
                "extract_seconds": round(self.extract_seconds, 6),
                "total_seconds": round(self.total_seconds, 6), "gap": self.gap}


def _context():
    from xdsl.context import Context
    from xdsl.dialects import builtin, equivalence, test

    ctx = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, equivalence.Equivalence, test.Test):
        ctx.load_dialect(dialect)
    return ctx


def build_egraph(demand: "_routing.OpDemand", candidates: Sequence["_routing.Candidate"],
                 cost_model: "_routing.CostModel"):
    """``(module, alternatives, costs, unscored)`` for one contraction.

    Each candidate becomes one operation carrying its measured cost, and all of them feed a single
    ``equivalence.class`` — so the IR holds every legal implementation at once instead of the one a
    router happened to prefer.

    A candidate the cost model declines is still added as an alternative but carries NO cost. That is the
    honest encoding: it remains a legal implementation, and it is not ranked, so extraction cannot prefer
    it for lack of data. Dropping it would erase a capability the target has.
    """
    from xdsl.dialects import equivalence as E, test
    from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp, StringAttr
    from xdsl.ir import Block, Region

    if not candidates:
        raise ValueError("refusing to build an e-graph with no alternatives; an empty e-class extracts "
                         "to nothing and would read as a routing decision")

    started = time.perf_counter()
    idx = IndexType()
    ops, names, costs, unscored = [], [], {}, []
    for cand in candidates:
        attrs: dict[str, Any] = {"unit": StringAttr(cand.unit),
                                 "kind": StringAttr(cand.kind),
                                 "exposure": StringAttr(cand.exposure)}
        score = cost_model(demand, cand)
        if score is None:
            unscored.append(cand.unit)
        else:
            scaled = int(round(float(score) * COST_SCALE))
            attrs[E.EQSAT_COST_LABEL] = IntAttr(scaled)
            costs[cand.unit] = scaled
        ops.append(test.TestOp(result_types=[idx], attributes=attrs))
        names.append(cand.unit)

    cls = E.ClassOp(*[o.results[0] for o in ops])
    body = Block([*ops, cls, E.YieldOp(cls.results[0])])
    module = ModuleOp([E.GraphOp([idx], Region([body]))])
    elapsed = time.perf_counter() - started
    return module, tuple(names), costs, tuple(unscored), elapsed


def alternatives_in(module) -> tuple[str, ...]:
    """The unit names still present as alternatives in ``module``, in order.

    Reading this before and after extraction is how "the alternatives are retained" stops being a claim:
    before, every legal implementation is there; after, one is.
    """
    from xdsl.dialects import equivalence as E

    out: list[str] = []
    for op in module.walk():
        if isinstance(op, E.AnyClassOp):
            continue
        unit = op.attributes.get("unit")
        if unit is not None:
            out.append(str(unit.data))
    return tuple(out)


def recost(module, costs: Mapping[str, float]) -> int:
    """Re-attach costs to an ALREADY-BUILT graph; returns how many alternatives were re-costed.

    This is the operation a downstream pass would perform — it learns, after the graph exists, that an
    implementation is cheaper or dearer than it looked (the epilogue fused, the operands turned out to be
    in the right layout already). Its existence is what makes the deferral testable: if construction had
    committed to a unit, re-costing could not change the outcome, and the graph would be a routing
    decision wearing an e-class.

    A unit absent from ``costs`` keeps whatever cost it had, so a partial update is a partial update
    rather than a silent removal of everything it did not mention.
    """
    from xdsl.dialects import equivalence as E
    from xdsl.dialects.builtin import IntAttr

    touched = 0
    for op in module.walk():
        if isinstance(op, E.AnyClassOp):
            continue
        unit = op.attributes.get("unit")
        if unit is None:
            continue
        name = str(unit.data)
        if name in costs:
            op.attributes[E.EQSAT_COST_LABEL] = IntAttr(int(round(float(costs[name]) * COST_SCALE)))
            touched += 1
    return touched


def run_extraction(module) -> tuple[str | None, float]:
    """Run ``eqsat-add-costs`` + ``eqsat-extract`` on ``module``; returns (chosen unit, seconds).

    The choice is READ BACK from the extracted IR rather than computed alongside it. Computing it here
    would make this :func:`routing.select` with extra steps, and would demonstrate nothing about whether
    the decision can be made from the graph.
    """
    from xdsl.transforms import eqsat_add_costs, eqsat_extract

    ctx = _context()
    started = time.perf_counter()
    eqsat_add_costs.EqsatAddCostsPass(default=None).apply(ctx, module)
    eqsat_extract.EqsatExtractPass().apply(ctx, module)
    elapsed = time.perf_counter() - started
    remaining = alternatives_in(module)
    return (remaining[0] if remaining else None), elapsed


def extract_choice(demand: "_routing.OpDemand", candidates: Sequence["_routing.Candidate"],
                   cost_model: "_routing.CostModel") -> EGraphResult:
    """Build the e-graph, run the real eqsat passes, and report which alternative survived.

    The choice is read back from the extracted IR rather than computed alongside it. That is the point:
    if it were computed here, this would be :func:`routing.select` with extra steps and would not
    demonstrate that the decision can be made from the graph.
    """
    module, names, costs, unscored, build_s = build_egraph(demand, candidates, cost_model)
    if not costs:
        # Nothing could be ranked. Fail closed rather than extract an arbitrary alternative.
        return EGraphResult(chosen=None, alternatives=names, costs=costs, unscored=unscored,
                            build_seconds=build_s,
                            gap=("the cost model declined every alternative, so extraction has nothing "
                                 "to minimise; the caller must fall back to declaration order"))
    chosen, extract_s = run_extraction(module)
    gap = None if chosen is not None else "extraction left no costed alternative in the IR"
    return EGraphResult(chosen=chosen, alternatives=names, costs=costs, unscored=unscored,
                        build_seconds=build_s, extract_seconds=extract_s, gap=gap)


def select_by_extraction(candidates: Sequence["_routing.RouteCandidates"],
                         cost_model: "_routing.CostModel") -> list["_routing.RouteResult"]:
    """A drop-in alternative to :func:`routing.select` that decides by extraction from an e-graph.

    Falls back to declaration order exactly where ``select`` does — on a gapped demand, and on one whose
    alternatives the cost model all declined — so the two are comparable rather than differing in their
    handling of missing data.
    """
    out: list[_routing.RouteResult] = []
    for entry in candidates:
        if entry.is_gapped:
            out.append(_routing.RouteResult(entry.demand, None, None, entry.gap))
            continue
        got = extract_choice(entry.demand, entry.candidates, cost_model)
        pick = got.chosen or entry.candidates[0].unit
        acc = next((c.acc for c in entry.candidates if c.unit == pick), None)
        out.append(_routing.RouteResult(entry.demand, pick, acc, None))
    return out


def agreement(candidates: Sequence["_routing.RouteCandidates"],
              cost_model: "_routing.CostModel") -> dict[str, Any]:
    """Compare extraction against eager selection over the same demands, and time both.

    Reported rather than asserted, and phrased as agreement rather than as a win: identical decisions are
    the EXPECTED outcome while there are no rewrite rules, and a disagreement would mean one of the two
    is not reading the cost model it claims to.
    """
    t0 = time.perf_counter()
    eager = _routing.select(candidates, cost_model)
    eager_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    extracted = select_by_extraction(candidates, cost_model)
    extract_s = time.perf_counter() - t0

    disagreements = [
        {"site": a.demand.site, "op": a.demand.op, "eager": a.unit, "extracted": b.unit}
        for a, b in zip(eager, extracted, strict=True) if a.unit != b.unit
    ]
    return {
        "n_demands": len(candidates),
        "agree": not disagreements,
        "disagreements": disagreements,
        "eager_seconds": round(eager_s, 6),
        "extraction_seconds": round(extract_s, 6),
        "slowdown": (round(extract_s / eager_s, 1) if eager_s > 0 else None),
        "hypotheses": {k: dict(v) for k, v in HYPOTHESES.items()},
    }
