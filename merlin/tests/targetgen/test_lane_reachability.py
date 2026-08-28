"""A capsule may not require a routing lane its target cannot populate.

An interop capsule asserts COMPOSITION: part of a model on the accelerator, part on a lane the target
also owns. The lanes come from the routing plan's own partition, which is derived from the target's
DECLARED compute units:

    a unit of a mesh kind (systolic/spatial/simt)  -> on_mesh
    a unit that is not a mesh kind                 -> in_contract_vector_scalar
    an op routed to no unit at all                 -> scalar_rvv_lane   (always available: the host CPU)

So a target that declares only mesh-kind units can never put anything in `in_contract_vector_scalar`, and
a capsule requiring it there is unpassable however good the backend is -- a wall wearing the costume of a
capability test. That shipped: an interop capsule required that lane on a target whose every declared unit
is a mesh kind, and it could not have been passed by any submission.

The guard belongs at generation time, where the author can still fix it, not in a run where it fails
forever and reads as a compiler deficiency.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import pytest
import yaml

from merlin.common.paths import merlin_dir
from merlin.targetgen.routing import reachable_lanes, reachable_lanes_on


@dataclass
class _Unit:
    name: str
    kind: str


def test_a_mesh_only_target_cannot_offer_the_in_contract_lane():
    lanes = reachable_lanes_on([_Unit("mesh", "systolic"), _Unit("simt", "simt")])
    assert "on_mesh" in lanes
    assert "in_contract_vector_scalar" not in lanes, (
        "no declared unit is a non-mesh kind, so the router has nothing to put on that lane")


def test_a_target_with_a_non_mesh_unit_offers_it():
    lanes = reachable_lanes_on([_Unit("mxu", "systolic"), _Unit("vpu", "vector")])
    assert {"on_mesh", "in_contract_vector_scalar", "scalar_rvv_lane"} <= lanes


def test_the_host_lane_is_always_reachable():
    """An op routed to no unit falls to the host CPU, so this lane exists even with no units at all."""
    assert reachable_lanes_on([]) == {"scalar_rvv_lane"}


def test_a_unit_with_no_declared_kind_does_not_invent_a_lane():
    assert reachable_lanes_on([_Unit("mystery", None)]) == {"scalar_rvv_lane"}


@pytest.mark.parametrize("target", ["radiance", "gemmini", "atlas"])
def test_every_shipped_lane_declaration_is_reachable_on_its_target(target):
    """The end-to-end guard: what the corpus asks for must be something the target can do."""
    p = merlin_dir() / "contract/capsules/profiles" / f"{target}.yaml"
    if not p.exists():
        pytest.skip(f"no profile for {target}")
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    ents = [e for e in (d if isinstance(d, list) else d.get("capsules") or []) if isinstance(e, dict)]
    have = reachable_lanes(target)
    offenders = []
    for e in ents:
        want = (e.get("lanes") or {}).get("require") or []
        missing = [l for l in want if l not in have]
        if missing:
            offenders.append(f"{e.get('name')} requires {missing}, target offers {sorted(have)}")
    assert not offenders, "unpassable lane requirement(s):\n  " + "\n  ".join(offenders)


def test_every_target_that_ships_capstones_also_ships_an_interop_capsule():
    """A capstone says the accelerator ran the whole model; an interop capsule says the compiler can SPLIT
    one and still be right. A target with the first and not the second is only testing half the claim."""
    missing = []
    for target in ("radiance", "gemmini", "atlas"):
        p = merlin_dir() / "contract/capsules/profiles" / f"{target}.yaml"
        if not p.exists():
            continue
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        ents = [e for e in (d if isinstance(d, list) else d.get("capsules") or []) if isinstance(e, dict)]
        if not any(e.get("cat") == "model" for e in ents):
            continue
        if not any((e.get("lanes") or {}).get("require") for e in ents):
            missing.append(target)
    assert not missing, f"targets shipping capstones but no interop capsule: {missing}"
