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


def _experiment_targets() -> list[str]:
    """Every target that ships an experiment descriptor.

    DISCOVERED, never listed. A hardcoded roster is how a guard silently stops covering the target
    added after it was written -- and mx_gemmini is exactly that target here: it ships a corpus and was
    absent from the parametrization above, so nothing checked its lanes at all.
    """
    root = merlin_dir() / "experiments/capsule_bench/targets"
    return sorted(p.name for p in root.iterdir() if (p / "target_experiment.yaml").is_file())


def _materialized_lane_declarations(target: str) -> list[tuple[str, pathlib.Path, list[str]]]:
    """``(capsule name, path, required lanes)`` for every GRADED capsule on disk for ``target``.

    Mirrors ``boundary.corpus_boundaries``' selection rules -- the graded roots, the label filter, the
    descriptor's grading exclusions -- so this reads the same corpus the grader does.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    desc = merlin_dir() / "experiments/capsule_bench/targets" / target / "target_experiment.yaml"
    te = load_target_experiment(desc)
    exclude = set(getattr(te, "graded_exclude", ()) or ())
    out: list[tuple[str, pathlib.Path, list[str]]] = []
    for root in te.graded_roots():
        for cy in sorted(pathlib.Path(root).glob("*/capsule.yaml")):
            cap = yaml.safe_load(cy.read_text(encoding="utf-8")) or {}
            name = str(cap.get("name") or cy.parent.name)
            if name in exclude or str(cap.get("label")) not in {"public", "dev"}:
                continue
            want = [str(x) for x in ((cap.get("lanes") or {}).get("require") or [])]
            if want:
                out.append((name, cy.parent, want))
    return out


@pytest.mark.parametrize("target", _experiment_targets())
def test_every_materialized_lane_declaration_is_reachable_on_its_target(target):
    """The profile check above is necessary but NOT sufficient: it reads the SPEC, and the grader reads
    the CORPUS. Those diverge whenever a profile entry is renamed and the old materialized directory is
    not deleted -- and one had: radiance shipped a public, graded ``MX0_interop_scalar_lane`` requiring
    ``in_contract_vector_scalar`` on a target whose every declared unit is a mesh kind, long after the
    profile entry had been renamed to ``MX0_interop_rvv_lane_lstm`` with the reachable lane. Both guards
    passed, because neither of them looked at what actually ships.
    """
    try:
        have = reachable_lanes(target)
    except (FileNotFoundError, OSError) as exc:
        # A target contract is a GENERATED artifact under out/, absent in a fresh clone. "We could not
        # resolve this target's units" is not "its lanes are fine" -- skip loudly with the missing path
        # rather than let an unresolvable target read as a passing one.
        pytest.skip(f"cannot resolve declared units for {target!r}: {exc}")
    offenders = [
        f"{name} ({path}) requires {[l for l in want if l not in have]}, target offers {sorted(have)}"
        for name, path, want in _materialized_lane_declarations(target)
        if any(l not in have for l in want)
    ]
    assert not offenders, (
        "materialized capsule(s) requiring an unreachable lane -- unpassable by any submission:\n  "
        + "\n  ".join(offenders))


# --- the host lane is a declared fact, not an assumption ---------------------------------------------

def test_the_host_lane_is_not_added_unconditionally():
    """It used to be. A target declaring no host at all still offered `scalar_rvv_lane`, so a capsule
    requiring it was graded against a system model with no host in it and the lane looked available
    because nothing had asked whether one existed."""
    assert reachable_lanes_on([], host_declared=False) == set()
    assert reachable_lanes_on([], host_declared=True) == {"scalar_rvv_lane"}


def test_an_unanswerable_question_does_not_remove_a_lane():
    """`None` is "nobody asked", not "no host". Only a real negative may take a lane away."""
    assert reachable_lanes_on([], host_declared=None) == {"scalar_rvv_lane"}


def test_a_declared_host_lane_is_what_makes_the_lane_reachable_not_a_board():
    """The two are different facts, and conflating them is wrong in the direction that matters: a target
    can own a host lane while declaring no board, which is where four of six targets are. Keying
    reachability on the board would delete a lane those targets really have."""
    from merlin.targetgen.routing import host_board_gap, host_is_declared

    for target in ("gemmini", "atlas", "radiance"):
        assert host_is_declared(target) is True, f"{target} declares a host lane"
    # ...and a board is a SEPARATE question, answered per target rather than assumed: gemmini declares
    # one, and atlas needs none because it is self-hosted. Neither is a gap, and the lane is reachable
    # on both regardless -- which is the property this test exists for.
    assert host_board_gap("gemmini") is None


def test_a_self_hosted_target_needs_no_board_and_reports_no_gap():
    """WHETHER A BOARD IS NEEDED IS ITSELF DERIVED. A self-hosted target runs its own program on its own
    scalar core and its scalar/vector lane is in-contract, so there is no external host to declare and
    demanding a board would demand a fact the machine does not have.

    Reading "declares no host.board" as a gap on every target that has none was wrong in exactly that
    way: it counted four targets as under-modelled when two of them are self-hosted and correct."""
    from merlin.perf.profile import derive_profile
    from merlin.targetgen.routing import host_board_gap

    for target in ("atlas", "radiance"):
        trait = (derive_profile(target).traits or {}).get("self_hosted_program")
        if getattr(trait, "satisfied", None) is not True:
            pytest.skip(f"{target} is not self-hosted in this checkout; the case under test is absent")
        assert host_board_gap(target) is None, f"{target} is self-hosted; a board is inapplicable"


def test_an_undetermined_endpoint_is_reported_as_undetermined():
    """A target whose `self_hosted_program` trait is UNKNOWN is not thereby a self-hosted one, and its
    absent board is neither a gap nor correct until the trait is established."""
    from merlin.perf.profile import derive_profile
    from merlin.targetgen.routing import host_board_gap

    trait = (derive_profile("saturn_opu").traits or {}).get("self_hosted_program")
    if getattr(trait, "satisfied", None) is not None:
        pytest.skip("saturn_opu's endpoint is established in this checkout")
    assert "UNKNOWN" in (host_board_gap("saturn_opu") or "")
