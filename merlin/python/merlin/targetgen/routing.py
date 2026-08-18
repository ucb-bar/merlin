"""Datatype -> compute-unit routing — bind each op's (input, weight) formats to a legal unit.

Given a model's per-op format demands and a target's ``compute_units`` (the capability model in
:mod:`merlin.targetgen.compute_units`), assign each op to a compute unit that supports its op + its
input/weight formats + an accumulate rule for that combination — or record an **honest gap** when no
unit does (e.g. an RVV vector unit asked to run fp4). This is the generic, format-agnostic tooling
that is ours no matter the quantization format: it is a pure lookup over the manifest and contains no
target-specific logic (the target's own lowering/requant lives out-of-tree).

Mixed precision falls out naturally: an op with ``in_fmt != weight_fmt`` (e.g. fp16 activation + fp4
weight) routes iff a unit enumerates that ``(in, weight) -> acc`` rule — exactly how gemmini-mx's
``PE_MxMode`` enumerates the legal act x weight combinations.

**Legality is not profitability.** :func:`route` picks the first legal unit in declaration order, which
is right for a target whose units do not overlap and wrong for a hybrid one: an int8 matmul is legal on
both a vector unit and a matrix unit, and which one wins depends on the shape, the operand layout,
whether the epilogue can stay accumulator-resident, and dispatch overhead. So the decision is split in
two — :func:`route_candidates` enumerates *every* legal unit, and :func:`select` chooses among them with
a swappable cost model. :func:`route` stays as the first-candidate wrapper so existing callers are
unaffected, and the split is what makes an ablation possible: an ``eager`` model that always prefers the
matrix unit is a deliberately bad baseline to measure a real one against.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from merlin.targetgen import compute_units as _cu


@dataclass(frozen=True)
class OpDemand:
    """What one op needs: its op name and the format(s) of its operands.

    ``m``/``n``/``k`` are optional contraction extents. They are what a cost model needs and legality
    does not, so they stay optional: a demand that omits them is still routable, and a cost model that
    requires them must say so rather than substitute a default shape.
    """

    op: str
    in_fmt: str
    weight_fmt: str | None = None   # None for unary/elementwise ops
    site: str = ""                  # optional label (weight name / op id) for reporting
    m: int | None = None            # the op's real extents when known (a contraction's M x K x N), so a
    n: int | None = None            # whole-model matmul LAYER is compiled at its true shape (the backend
    k: int | None = None            # tiles it into DxD mesh tiles) rather than a single fixed tile

    @property
    def has_shape(self) -> bool:
        return None not in (self.m, self.n, self.k)


@dataclass(frozen=True)
class RouteResult:
    demand: OpDemand
    unit: str | None                # chosen compute unit, or None if gapped
    acc: str | None                 # accumulator token from the matched rule, if any
    gap: str | None                 # honest reason when unroutable


def _legal_on(unit: _cu.ComputeUnit, demand: OpDemand) -> tuple[bool, str | None]:
    """Is ``demand`` legal on ``unit``? Returns (legal, accumulator token)."""
    if not unit.supports_op(demand.op):
        return False, None
    if demand.in_fmt not in unit.dtypes:
        return False, None
    if demand.weight_fmt is not None and demand.weight_fmt not in unit.dtypes:
        return False, None
    if not unit.accumulate:
        return True, None
    for rule in unit.accumulate:
        if rule.inp == demand.in_fmt and (demand.weight_fmt is None or rule.weight == demand.weight_fmt):
            return True, rule.acc
    return False, None


def route(demands: list[OpDemand], units: list[_cu.ComputeUnit]) -> list[RouteResult]:
    """Route each demand to the first compute unit that supports it (composition resolved).

    Now a thin wrapper over ``select(route_candidates(...), first_candidate_cost)``. Written this way on
    purpose: "the refactor changed no behaviour" is true by construction here, rather than a claim held up
    by whichever cases the existing tests happen to cover.
    """
    return select(route_candidates(demands, units), first_candidate_cost)


# ---------------------------------------------------------------------------------------------
# Candidates, cost models, selection
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One legal placement of a demand, with what the cost model needs to judge it."""

    unit: str
    kind: str
    acc: str | None
    exposure: str


@dataclass(frozen=True)
class RouteCandidates:
    """Every legal unit for one demand, in declaration order, or an honest gap when there are none."""

    demand: OpDemand
    candidates: tuple[Candidate, ...] = ()
    gap: str | None = None

    @property
    def is_gapped(self) -> bool:
        return self.gap is not None


def _gap_text(d: OpDemand) -> str:
    wf = f" weight={d.weight_fmt}" if d.weight_fmt is not None else ""
    site = f" [{d.site}]" if d.site else ""
    return f"no compute unit supports op={d.op} in={d.in_fmt}{wf}{site}"


def route_candidates(demands: Sequence[OpDemand], units: Sequence[_cu.ComputeUnit], *,
                     target_endpoint_kind: str | None = None) -> list[RouteCandidates]:
    """Every legal (demand, unit) pairing — the input a cost model needs to have a choice at all.

    Order is contract-declaration order, so ``candidates[0]`` is exactly what :func:`route` would have
    picked. That is what lets the refactor be inert: the wrapper below selects the first candidate and
    reproduces the old behaviour by construction rather than by matching test expectations.
    """
    effective = [_cu.effective(u, list(units)) for u in units]
    out: list[RouteCandidates] = []
    for d in demands:
        legal: list[Candidate] = []
        for u in effective:
            ok, acc = _legal_on(u, d)
            if ok:
                legal.append(Candidate(unit=u.name, kind=u.kind, acc=acc,
                                       exposure=_cu.resolve_exposure(
                                           u, target_endpoint_kind=target_endpoint_kind)))
        out.append(RouteCandidates(demand=d, candidates=tuple(legal),
                                   gap=None if legal else _gap_text(d)))
    return out


#: A cost model scores one candidate for one demand: LOWER is better. Returning None means "I decline to
#: score this", which is not the same as scoring it badly — see :func:`select`.
CostModel = Callable[[OpDemand, Candidate], "float | None"]


def first_candidate_cost(demand: OpDemand, candidate: Candidate) -> float:
    """Score nothing; preserve declaration order. The behaviour :func:`route` has always had."""
    return 0.0


def eager_cost(demand: OpDemand, candidate: Candidate) -> float:
    """Prefer the widest datapath for every contraction, regardless of shape.

    A deliberately bad baseline. It exists to be beaten: routing every contraction onto a matrix unit is
    the obvious policy, it is wrong for the narrow shapes the workload census found in quantity, and an
    ablation needs the bad policy actually implemented rather than described.
    """
    return {"spatial": 0.0, "systolic": 0.0, "simt": 1.0, "vector": 2.0, "scalar": 3.0}.get(
        candidate.kind, 4.0)


@dataclass(frozen=True)
class MeasuredCost:
    """Cost from measured per-unit throughput, charging tile occupancy and layout explicitly.

    **A tiled unit is charged for the tile it occupies, not the elements it uses.** This is the whole
    reason a narrow extent is expensive: a unit with a 32-lane tile edge spends the same time on ``M = 1``
    as on ``M = 32``, because the tile is the unit of work. Costing a tiled unit as ``macs / peak_rate``
    instead credits it with the work it *would* have done at full occupancy, which makes every shape look
    good on it — including exactly the ones the workload census found to be numerous and negligible. A
    unit with no declared tile edge is costed elementwise, which is the right model for a vector unit
    whose lanes are set per-operation.

    Two things this does NOT do. It does not invent architectural constants: every number comes from a
    measurement table the caller supplies, and a unit absent from that table is DECLINED (None) rather
    than scored optimistically — an unmeasured unit that scored well would win routing decisions on the
    strength of having no data.

    And it charges for layout. A unit that indexes both operands K-major makes a contraction whose left
    operand is not already transposed pay a packing pass over ``k*m`` elements. Leaving that out is how a
    routing decision comes out in favour of a unit that then spends more time rearranging memory than
    computing.
    """

    macs_per_cycle: Mapping[str, float]
    dispatch_cycles: Mapping[str, float] = field(default_factory=dict)
    #: Cost per element of packing the left operand K-major, for units that require it.
    pack_cycles_per_element: Mapping[str, float] = field(default_factory=dict)
    #: Units that need the left operand K-major. Not a target fact — a property the contract declares.
    requires_k_major: frozenset[str] = frozenset()
    #: Logical tile edge per unit, for units that quantize work into tiles. Absent -> costed elementwise.
    #: Derived from the target (see ``kernels.opu_cert.logical_tile_edge``), never assumed here.
    tile_edge: Mapping[str, int] = field(default_factory=dict)
    #: Fixed cycles charged ONCE PER TILE PAIR, for units whose per-tile overhead (loading operands into
    #: the array, draining the accumulator out of it) does not scale with the reduction.
    #:
    #: This is separate from ``dispatch_cycles`` — which is charged once per operation — because the two
    #: amortize against completely different things, and collapsing them mismeasures by a large factor.
    #: MEASURED on the Saturn OPU at tile edge 64 (FPGA, 47/47-certified corpus): four shapes spanning 16
    #: to 64 tile pairs and K from 32 to 1024 fit ``cycles/pair = 29*K + ~2000`` with the asymptotic rate
    #: agreeing to within 5% (139.6 / 135.7 / 149.6 / 146.5 MACs per cycle). Without this term the SAME
    #: measurements imply rates from 39 to 135 depending only on which shape they were taken from -- a
    #: 3.45x spread that is entirely this overhead being amortized over different reduction lengths, and
    #: whichever single number were chosen would misprice every other shape.
    #:
    #: Defaults to zero, so a unit that does not declare one is costed exactly as before.
    tile_overhead_cycles: Mapping[str, float] = field(default_factory=dict)

    def __call__(self, demand: OpDemand, candidate: Candidate) -> float | None:
        rate = self.macs_per_cycle.get(candidate.unit)
        if rate is None or rate <= 0:
            return None                      # unmeasured: decline rather than guess
        if not demand.has_shape:
            return None                      # a cost model without extents would be scoring a wish
        m, n, k = float(demand.m), float(demand.n), float(demand.k)
        tile = int(self.tile_edge.get(candidate.unit, 0) or 0)
        if tile > 0:
            # Work is quantized: a partly-filled tile costs a full one.
            pairs = _ceil_div(demand.m, tile) * _ceil_div(demand.n, tile)
            steps = pairs * demand.k
            cost = float(steps) * (float(tile) * float(tile)) / rate
            # Per-tile-pair overhead, charged whatever the reduction length. Charging it here rather
            # than folding it into `rate` is what makes a SHORT reduction expensive on a tiled unit:
            # the measured 2000-odd cycles of load+drain per tile pair are ~65% of runtime at K=32 and
            # ~6% at K=1024, so a single blended rate cannot be right at both ends.
            cost += float(pairs) * float(self.tile_overhead_cycles.get(candidate.unit, 0.0))
        else:
            cost = m * n * k / rate
        cost += float(self.dispatch_cycles.get(candidate.unit, 0.0))
        if candidate.unit in self.requires_k_major:
            cost += float(self.pack_cycles_per_element.get(candidate.unit, 0.0)) * k * m
        return cost


def _ceil_div(a: int, b: int) -> int:
    return -(-int(a) // int(b))


COST_MODELS: dict[str, CostModel] = {
    "first": first_candidate_cost,
    "eager": eager_cost,
}


def select(candidates: Sequence[RouteCandidates], cost_model: CostModel = first_candidate_cost,
           *, context: Mapping[str, object] | None = None) -> list[RouteResult]:
    """Choose one unit per demand under ``cost_model``, preserving declaration order on ties.

    A candidate the model declines to score is kept as a fallback rather than dropped: declining means
    "no data", and a demand whose only legal unit is unmeasured must still route somewhere and say that
    it did so unscored. Dropping it would turn a missing measurement into a routing gap, which reads as
    a capability the target does not have.
    """
    results: list[RouteResult] = []
    for entry in candidates:
        if entry.is_gapped:
            results.append(RouteResult(entry.demand, None, None, entry.gap))
            continue
        best: tuple[float, int, Candidate] | None = None
        for i, cand in enumerate(entry.candidates):
            score = cost_model(entry.demand, cand)
            if score is None:
                continue
            if best is None or (score, i) < (best[0], best[1]):
                best = (score, i, cand)
        chosen = best[2] if best is not None else entry.candidates[0]
        results.append(RouteResult(entry.demand, chosen.unit, chosen.acc, None))
    return results


def explain(candidates: Sequence[RouteCandidates],
            cost_model: CostModel = first_candidate_cost) -> list[dict[str, object]]:
    """Per-demand scores for every candidate — so a routing decision can be inspected, not just taken."""
    out: list[dict[str, object]] = []
    for entry in candidates:
        scored = [{"unit": c.unit, "kind": c.kind, "exposure": c.exposure,
                   "score": cost_model(entry.demand, c)}
                  for c in entry.candidates]
        out.append({"op": entry.demand.op, "site": entry.demand.site,
                    "m": entry.demand.m, "n": entry.demand.n, "k": entry.demand.k,
                    "gap": entry.gap, "candidates": scored})
    return out


def route_target(demands: list[OpDemand], target_name: str) -> list[RouteResult]:
    """Route demands against a named target's contract ``compute_units`` (in-tree or out-of-tree)."""
    from merlin.targetgen import target_registry as tr

    units = _cu.compute_units(tr.load_contract(target_name))
    return route(demands, units)


# compute-unit kinds that ARE the accelerator (execute on-mesh); everything else runs on the scalar/vector
# lane. A demand that routes to none of a target's units is not a failure for a whole model — it means that
# op (a norm/activation/elementwise) runs on the scalar/RVV lane, not the mesh.
_MESH_KINDS = {"systolic", "spatial", "simt"}


def route_plan_on(demands: list[OpDemand], units: list[_cu.ComputeUnit]) -> dict:
    """Split a whole model's ops across a set of already-loaded ``units`` (the target-agnostic core of
    :func:`route_plan`). ``results`` preserves the input op ORDER — the whole-model splice walks it to
    co-schedule each op on its lane while handing activations between steps."""
    kind = {u.name: u.kind for u in units}
    results = route(demands, units)
    mesh, fallback, scalar_rvv = [], [], []
    for r in results:
        if r.unit and kind.get(r.unit) in _MESH_KINDS:
            mesh.append(r)
        elif r.unit:
            fallback.append(r)
        else:
            scalar_rvv.append(r)
    return {"mesh": mesh, "fallback": fallback, "scalar_rvv": scalar_rvv, "results": results}


def route_plan(demands: list[OpDemand], target_name: str) -> dict:
    """Split a whole model's ops across a target: which run on the accelerator MESH (systolic/spatial/simt
    unit), which on an in-contract vector/scalar unit, and which fall back to the scalar/RVV lane (no
    accelerator unit — an honest, expected outcome for norms/activations on a matmul-only mesh)."""
    from merlin.targetgen import target_registry as tr

    units = _cu.compute_units(tr.load_contract(target_name))
    return route_plan_on(demands, units)


def gaps(results: list[RouteResult]) -> list[RouteResult]:
    return [r for r in results if r.gap is not None]


def is_fully_routed(results: list[RouteResult]) -> bool:
    return all(r.gap is None for r in results)
