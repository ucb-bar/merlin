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
"""
from __future__ import annotations

from dataclasses import dataclass

from merlin.targetgen import compute_units as _cu


@dataclass(frozen=True)
class OpDemand:
    """What one op needs: its op name and the format(s) of its operands."""

    op: str
    in_fmt: str
    weight_fmt: str | None = None   # None for unary/elementwise ops
    site: str = ""                  # optional label (weight name / op id) for reporting
    m: int | None = None            # the op's real extents when known (a contraction's M x K x N), so a
    k: int | None = None            # whole-model matmul LAYER is compiled at its true shape (the backend
    n: int | None = None            # tiles it into DxD mesh tiles) rather than a single fixed tile


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
    """Route each demand to the first compute unit that supports it (composition resolved)."""
    effective = [_cu.effective(u, units) for u in units]
    results: list[RouteResult] = []
    for d in demands:
        chosen: str | None = None
        acc: str | None = None
        for u in effective:
            ok, a = _legal_on(u, d)
            if ok:
                chosen, acc = u.name, a
                break
        if chosen is None:
            wf = f" weight={d.weight_fmt}" if d.weight_fmt is not None else ""
            site = f" [{d.site}]" if d.site else ""
            gap = f"no compute unit supports op={d.op} in={d.in_fmt}{wf}{site}"
            results.append(RouteResult(d, None, None, gap))
        else:
            results.append(RouteResult(d, chosen, acc, None))
    return results


def route_target(demands: list[OpDemand], target_name: str) -> list[RouteResult]:
    """Route demands against a named target's contract ``compute_units`` (in-tree or out-of-tree)."""
    from merlin.targetgen import target_registry as tr

    units = _cu.compute_units(tr.load_contract(target_name))
    return route(demands, units)


# compute-unit kinds that ARE the accelerator (execute on-mesh); everything else runs on the scalar/vector
# lane. A demand that routes to none of a target's units is not a failure for a whole model — it means that
# op (a norm/activation/elementwise) runs on the scalar/RVV lane, not the mesh.
_MESH_KINDS = {"systolic", "spatial", "simt"}


def route_plan(demands: list[OpDemand], target_name: str) -> dict:
    """Split a whole model's ops across a target: which run on the accelerator MESH (systolic/spatial/simt
    unit), which on an in-contract vector/scalar unit, and which fall back to the scalar/RVV lane (no
    accelerator unit — an honest, expected outcome for norms/activations on a matmul-only mesh)."""
    from merlin.targetgen import target_registry as tr

    units = _cu.compute_units(tr.load_contract(target_name))
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


def gaps(results: list[RouteResult]) -> list[RouteResult]:
    return [r for r in results if r.gap is not None]


def is_fully_routed(results: list[RouteResult]) -> bool:
    return all(r.gap is None for r in results)
