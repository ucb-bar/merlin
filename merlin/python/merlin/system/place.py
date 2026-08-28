"""One placement decision, over the host and the devices together.

Three surfaces decide host-vs-device today and they disagree by construction, because each was
written for its own caller: the contract router matches an op name and two dtypes and then takes the
FIRST legal unit in declaration order; the dispatch runtime matches the IR structurally and ignores
declared units entirely; the offload pass carries its own dtype triple. A model can therefore be told
a layer belongs on the accelerator by one of them and off it by another, and nothing reconciles them.

What makes them reconcilable is representing the host. Today a routing gap silently BECOMES the host
lane -- the host is an absence, so no decision was ever recorded, and "this landed on the host because
nothing could take it" is indistinguishable from "this was placed on the host". Here the host is a
unit like any other, so every op has a placement WITH a reason, and the interesting case -- an op on
the host that the host cannot natively compute -- becomes visible instead of silent.

Deliberately conservative in one respect: the host is a LAST-RESORT candidate. Nothing that routes to
a device today stops routing, and no op that currently reaches the host is newly refused. The change
is that the result is now explained rather than inferred.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["HOST_DEVICE", "Placement", "Placed", "host_units", "place", "units_for"]

#: The device name a host placement carries. Not a target: it is the absence of one, named so a
#: report can say "host" without a caller testing for None.
HOST_DEVICE = "host"


@dataclass(frozen=True)
class Placed:
    """One op's placement, and why."""

    demand: Any
    device: str                      # a device name, or HOST_DEVICE
    unit: str | None                 # the compute unit within it, when one was legal
    lane: str                        # on_mesh | in_contract_vector_scalar | scalar_rvv_lane
    why: str
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

    def lanes(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for p in self.placed:
            out[p.lane] = out.get(p.lane, 0) + 1
        return out


def _host_dtypes() -> tuple[str, ...]:
    """The formats the host lowering declares it compiles. Read from the compiler's own declaration
    rather than listed here, so the two cannot drift."""
    try:
        from merlin.compile_cli import _RVV_DTYPES
        return tuple(_RVV_DTYPES)
    except Exception:            # noqa: BLE001
        return ()


def host_units(host) -> list[Any]:
    """The host as compute units: a scalar unit, and a vector unit when the board says it has one.

    ``ops=()`` means "every op", which is the honest declaration for a general-purpose core -- it is
    the DTYPE that constrains a host, not the operation. The vector unit is only synthesized when the
    board actually declares vector capability: a board that says nothing gets the scalar unit alone,
    because inventing a vector unit for it would place vector work on a core that traps on it.
    """
    from merlin.targetgen.compute_units import ComputeUnit

    dtypes = _host_dtypes()
    units = [ComputeUnit(name="host_scalar", kind="scalar", dtypes=dtypes, ops=())]
    if host is not None and host.vector_capable() is True:
        units.append(ComputeUnit(name="host_vector", kind="vector", dtypes=dtypes, ops=()))
    return units


def units_for(system) -> list[tuple[str, Any]]:
    """``[(device_name, unit)]`` for every unit in the system, devices first then the host.

    Devices precede the host so that declaration-order selection -- what the router does today when no
    cost model is supplied -- keeps preferring a device, and this stays inert on existing inputs.
    """
    from merlin.targetgen import compute_units as _cu, target_registry as _tr

    out: list[tuple[str, Any]] = []
    for dev in getattr(system, "devices", ()) or ():
        try:
            for u in _cu.compute_units(_tr.load_contract(dev.name)):
                out.append((dev.name, u))
        except Exception:        # noqa: BLE001 -- an unresolvable device contributes no units
            continue
    for u in host_units(getattr(system, "host", None)):
        out.append((HOST_DEVICE, u))
    return out


def _lane_for(kind: str | None, on_host: bool) -> str:
    from merlin.targetgen.routing import _MESH_KINDS

    if on_host:
        return "scalar_rvv_lane"
    return "on_mesh" if kind in _MESH_KINDS else "in_contract_vector_scalar"


def place(demands: Sequence[Any], system, *,
          cost: Callable[[Any, Any], float | None] | None = None) -> Placement:
    """Place every demand on the system, preferring a device and explaining each choice.

    ``cost`` scores a ``(demand, unit)`` pairing; ``None`` (the default) keeps declaration order,
    which is exactly what the contract router does today -- so this reproduces current placement by
    construction rather than by matching expectations, and a cost model becomes a change of argument
    rather than a change of pass. A cost model that DECLINES every candidate for a demand falls back
    to the first legal one rather than dropping it: declining to price is not declining to run.
    """
    from merlin.targetgen.routing import OpDemand, _legal_on  # noqa: PLC2701 -- one legality predicate

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
            placed.append(Placed(demand=d, device=HOST_DEVICE, unit=None,
                                 lane="scalar_rvv_lane", emulated=True,
                                 why=f"no unit accepts op={getattr(d, 'op', '?')} "
                                     f"in={getattr(d, 'in_fmt', '?')}; host must emulate it"))
            continue

        chosen, why = pool[0], "first legal unit in declaration order"
        if cost is not None:
            scored = [(c, i, p) for i, p in enumerate(pool)
                      if (c := cost(d, p[1])) is not None]
            if scored:
                best = min(scored, key=lambda t: (t[0], t[1]))
                chosen, why = best[2], f"lowest cost {best[0]:.4g} of {len(scored)} priced candidate(s)"
            else:
                why = "no candidate could be priced; kept the first legal unit"

        dev, unit = chosen
        on_host = dev == HOST_DEVICE
        placed.append(Placed(demand=d, device=dev, unit=unit.name,
                             lane=_lane_for(unit.kind, on_host),
                             why=why if not on_host else f"{why} (no device accepted it)"))

    return Placement(placed=tuple(placed))
