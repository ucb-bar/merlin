"""Select and serialize current declaration-based host/device placement.

This preserves the existing format and policy, including its emulation report
and extent-keyed selector. It is not a heterogeneous execution admission: no
resource reservations, transfer/coherency plan, or synchronization are derived.
Use source-bound proposals for new routes that need exact operation identity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .candidates import HOST_DEVICE, _candidate, units_for


@dataclass(frozen=True)
class Placed:
    """One op's placement, and why."""

    demand: Any
    device: str  # a device name, or HOST_DEVICE
    unit: str | None  # the compute unit within it, when one was legal
    lane: str  # on_mesh | in_contract_vector_scalar | scalar_rvv_lane
    why: str
    #: The accumulate token the CHOSEN unit matched, exactly as ``routing.RouteResult.acc`` carries it.
    #: Not decoration: the mesh certifier compiles each placed contraction in its routed operand AND
    #: accumulate format, so a placement that omitted this could not stand in for a route result -- the
    #: tile would have to guess its accumulator, which is how an i8xi8 layer gets certified in the wrong
    #: precision. ``None`` means the unit declares no accumulate rule, which is a fact, not a gap.
    acc: str | None = None
    #: The router's own honest gap text when NO DEVICE unit accepted this op, else ``None``.
    #:
    #: Written by ``routing._gap_text`` rather than phrased here, so the two surfaces cannot drift into
    #: two different sentences for one fact. Note what it is NOT: a placement on the host is not a gap in
    #: the placement -- the host took the op and ``unit`` names which host unit did. It is a gap in the
    #: DEVICE, which is the question the coverage certificate asks of this field.
    gap: str | None = None
    #: True when the op landed on the host and the host does not natively carry its format -- i.e.
    #: the lowering has to emulate it. Today this is invisible; it is the honest reading of a gap.
    emulated: bool = False

    @property
    def on_device(self) -> bool:
        return self.device != HOST_DEVICE


@dataclass(frozen=True)
class Placement:
    """Every op placed, in the order the ops were given."""

    placed: tuple[Placed, ...] = ()

    def on_device(self) -> tuple[Placed, ...]:
        return tuple(p for p in self.placed if p.on_device)

    def on_host(self) -> tuple[Placed, ...]:
        return tuple(p for p in self.placed if not p.on_device)

    def emulated(self) -> tuple[Placed, ...]:
        """Ops the host took but cannot natively compute. A gap that used to be unreported."""
        return tuple(p for p in self.placed if p.emulated)

    def as_route_plan(self) -> dict:
        """This placement in ``routing.route_plan``'s own shape.

        The projection that lets one authority serve both surfaces. ``route_plan`` buckets by whether a
        legal unit was found and whether it is a mesh kind; a Placement already knows both, so the two
        can be compared op-for-op rather than argued about. Deliberately NOT the same objects:
        ``route_plan``'s ``results`` are ``RouteResult``s, and reconstructing those here would couple
        this module to routing's internals -- the buckets are what every consumer actually reads.
        """
        out: dict[str, list] = {"mesh": [], "fallback": [], "scalar_rvv": [], "results": []}
        for pl in self.placed:
            out["results"].append(pl)
            # BUCKET ON DEVICE-VS-HOST FIRST. `route_plan`'s `scalar_rvv` means "the host lane" and its
            # `fallback` means "a declared non-mesh DEVICE unit"; a Placement also knows about host
            # units, which routing does not, so keying on "was a unit legal" put host-vector placements
            # in `fallback` and the two surfaces disagreed about an op neither had misplaced.
            if not pl.on_device:
                out["scalar_rvv"].append(pl)
            elif pl.lane == "on_mesh":
                out["mesh"].append(pl)
            else:
                out["fallback"].append(pl)
        return out

    def to_dict(self) -> dict:
        """The record written beside a compile, one entry per placed op.

        ``emulated`` is the reason this file is worth writing: it is the case nothing reports today --
        an op the host took whose format the host cannot natively carry, so the lowering has to emulate
        it. Silent until now, because no surface had a field for it.
        """
        rows = []
        for pl in self.placed:
            d = pl.demand
            rows.append(
                {
                    "op": getattr(d, "op", None),
                    "in_fmt": getattr(d, "in_fmt", None),
                    "weight_fmt": getattr(d, "weight_fmt", None),
                    "site": getattr(d, "site", None),
                    "m": getattr(d, "m", None),
                    "n": getattr(d, "n", None),
                    "k": getattr(d, "k", None),
                    "rank": getattr(d, "rank", None),
                    "device": pl.device,
                    "unit": pl.unit,
                    "lane": pl.lane,
                    "acc": pl.acc,
                    "gap": pl.gap,
                    "why": pl.why,
                    "emulated": bool(pl.emulated),
                }
            )
        return {
            "placed": rows,
            "lanes": self.lanes(),
            "n_emulated": len(self.emulated()),
            "emulated": [r for r in rows if r["emulated"]],
        }

    def lanes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.placed:
            out[p.lane] = out.get(p.lane, 0) + 1
        return out


def _lane_for(kind: str | None, on_host: bool) -> str:
    from merlin.development.targetgen.routing import _MESH_KINDS

    if on_host:
        return "scalar_rvv_lane"
    return "on_mesh" if kind in _MESH_KINDS else "in_contract_vector_scalar"


def place(demands: Sequence[Any], system, *, cost: Callable[[Any, Any], float | None] | None = None) -> Placement:
    """Place every demand on the system, preferring a device and explaining each choice.

    ``cost`` scores a ``(demand, unit)`` pairing; ``None`` (the default) keeps declaration order,
    which is exactly what the contract router does today -- so this reproduces current placement by
    construction rather than by matching expectations, and a cost model becomes a change of argument
    rather than a change of pass. A cost model that DECLINES every candidate for a demand falls back
    to the first legal one rather than dropping it: declining to price is not declining to run.
    """
    # One legality predicate and one gap sentence, both borrowed rather than restated.
    from merlin.development.targetgen.routing import _gap_text, _legal_on  # noqa: PLC2701

    pairs = units_for(system)
    placed: list[Placed] = []

    for d in demands:
        # EVERY legal unit is a candidate, host included. Restricting the pool to devices whenever
        # one is legal would make the host unreachable to a cost model -- there would be nothing to
        # weigh it against -- and placement would still be "device if legal", which is the behaviour
        # this exists to replace. Device preference lives in the ORDER (units_for puts devices
        # first), so the unpriced path still picks a device by construction.
        legal = [(dev, u) for dev, u in pairs if _legal_on(u, d)[0]]
        pool = legal

        if not pool:
            # Nothing can take it, the host included -- today this is the silent case. The op still
            # goes to the host (the lowering will emulate it); what is new is that we say so.
            placed.append(
                Placed(
                    demand=d,
                    device=HOST_DEVICE,
                    unit=None,
                    lane="scalar_rvv_lane",
                    emulated=True,
                    gap=_gap_text(d),
                    why=f"no unit accepts op={getattr(d, 'op', '?')} "
                    f"in={getattr(d, 'in_fmt', '?')}; host must emulate it",
                )
            )
            continue

        chosen, why = pool[0], "first legal unit in declaration order"
        if cost is not None:
            # Score against a routing.Candidate, not the ComputeUnit: that is the protocol every cost
            # model in the repo already implements (it reads `candidate.unit` as a NAME), and passing
            # the unit object instead made "a cost model is an argument" true only of cost models
            # written for this function. MeasuredCost could not be passed at all.
            scored = [(c, i, p) for i, p in enumerate(pool) if (c := cost(d, _candidate(p[1], d))) is not None]
            if scored:
                best = min(scored, key=lambda t: (t[0], t[1]))
                chosen, why = best[2], f"lowest cost {best[0]:.4g} of {len(scored)} priced candidate(s)"
            else:
                why = "no candidate could be priced; kept the first legal unit"

        dev, unit = chosen
        on_host = dev == HOST_DEVICE
        placed.append(
            Placed(
                demand=d,
                device=dev,
                unit=unit.name,
                lane=_lane_for(unit.kind, on_host),
                acc=_legal_on(unit, d)[1],
                gap=_gap_text(d) if on_host else None,
                why=why if not on_host else f"{why} (no device accepted it)",
            )
        )

    return Placement(placed=tuple(placed))


def device_selector(placement: Placement) -> Callable[[Any], bool]:
    """The offload selector this placement implies: ``shape -> should this contraction move?``.

    The missing production caller. ``device_offload.rewrite_contractions_to_device`` takes ``select`` and
    moves nothing without one, and ``DeviceRouting.select`` defaults to ``None`` -- so the fused
    single-ELF path (host object + RVV kernels + device kernels, linked by the board build) has been
    reachable but never reached, and every whole-model run went through the Python interpreter instead.
    The reason was not a missing mechanism: it was that nobody had made the decision this closes over.

    Deliberately keyed on EXTENTS rather than on op identity. The offload pass walks a prepared module
    and the placement was derived from the capture, so the two hold different op handles for the same
    contraction; the extents are what both can state. ``ContractionShape`` gives ``(M, N)`` as the last
    two parallel dims and ``K`` as the first reduction dim, which is exactly what an ``OpDemand`` carries.

    Two ways it declines, both closed rather than open:

    * A contraction whose extents the demands never carried cannot be matched, so it is not moved. A
      selector that guessed here would move a contraction whose placement nobody decided.
    * If one extent triple was placed on a device in one site and on the host in another -- which a cost
      model may well do once one exists -- the triple is AMBIGUOUS through this keying, and ambiguity
      means decline. Moving it would silently apply one site's decision to another site.
    """
    seen: dict[tuple[int, int, int], set[bool]] = {}
    for pl in placement.placed:
        d = pl.demand
        m, k, n = getattr(d, "m", None), getattr(d, "k", None), getattr(d, "n", None)
        if m is None or k is None or n is None:
            continue
        seen.setdefault((int(m), int(k), int(n)), set()).add(bool(pl.on_device))
    decided = {key: (votes == {True}) for key, votes in seen.items()}

    def _select(shape) -> bool:
        try:
            key = (int(shape.parallel[-2]), int(shape.reduction[0]), int(shape.parallel[-1]))
        except Exception:  # noqa: BLE001 -- a shape this selector cannot read is not one it moves
            return False
        return decided.get(key, False)

    return _select
