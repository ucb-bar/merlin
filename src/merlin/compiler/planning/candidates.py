"""Adapt current target contracts and host declarations into placement candidates.

This compatibility adapter preserves registry resolution and unknown handling.
Scalar/vector units are implementation choices, not proof of independent devices,
concurrency, transfer legality, numerical coverage or observed execution.
"""

from __future__ import annotations

from typing import Any

from merlin.compiler.backend.host import HOST_PRECISIONS

HOST_DEVICE = "host"


def _host_dtypes() -> tuple[str, ...]:
    """The host compilation modes; this is not full numerical admission."""
    return HOST_PRECISIONS


def host_units(host) -> list[Any]:
    """The host as compute units: a scalar unit, and a vector unit when the board says it has one.

    ``ops=()`` means "every op", which is the honest declaration for a general-purpose core -- it is
    the DTYPE that constrains a host, not the operation. The vector unit is only synthesized when the
    board actually declares vector capability: a board that says nothing gets the scalar unit alone,
    because inventing a vector unit for it would place vector work on a core that traps on it.
    """
    from merlin.development.targetgen.compute_units import ComputeUnit

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
    from merlin.development.targetgen import compute_units as _cu
    from merlin.development.targetgen import target_registry as _tr

    out: list[tuple[str, Any]] = []
    for dev in getattr(system, "devices", ()) or ():
        try:
            units = list(_cu.compute_units(_tr.load_contract(dev.name)))
            # RESOLVE COMPOSITION, exactly as routing does. A unit that `contains` another inherits its
            # capability, and `route_candidates` has always resolved that before testing legality. This
            # did not, so on any target declaring a containing unit the two surfaces disagreed about
            # what that unit can do -- and the test pinning them equal happens to use targets with no
            # composition, so nothing caught it.
            for u in (_cu.effective(u, units) for u in units):
                out.append((dev.name, u))
        except Exception:  # noqa: BLE001 -- an unresolvable device contributes no units
            continue
    for u in host_units(getattr(system, "host", None)):
        out.append((HOST_DEVICE, u))
    return out


def _candidate(unit, demand):
    """The `routing.Candidate` view of a unit, so any cost model in the repo can score it."""
    from merlin.development.targetgen.routing import Candidate, _legal_on  # noqa: PLC2701

    _legal, acc = _legal_on(unit, demand)
    return Candidate(unit=unit.name, kind=unit.kind, acc=acc, exposure=getattr(unit, "exposure", None))
