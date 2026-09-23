"""The declared-vs-derived geometry check, exercised on every target rather than on one.

This comparison was first written inside one target's schedule binding, where it was correct and would
have been copied. The second copy is where two implementations drift, which is the shape of overfit the
repo's cardinal rule exists to prevent -- so it lives in the core now, and this file is what holds it to
being genuinely target-neutral: every assertion below is parametrized over the roster, and the ones that
name a target name it as DATA.
"""

from __future__ import annotations

import dataclasses

import pytest

from merlin.sched.mach.crosscheck import AXES, GeometryDisagreement, crosscheck_declared_geometry

pytestmark = pytest.mark.target("gemmini", "atlas", "muon", "toy_npu")

#: Targets the comparison is asked about. Each is here for a different outcome -- one where a machine
#: derives a geometry, several where none does -- so the suite covers both branches rather than whichever
#: one this checkout happens to produce.
TARGETS = ("gemmini", "atlas", "muon", "toy_npu")


def _derived(target: str):
    """The geometry a machine projects for ``target``, or ``None`` when none does."""
    from merlin.compile.scheduling.derive import geometry_from_machine
    from merlin.sched.mach.derive import derive

    try:
        return geometry_from_machine(derive(target))
    except Exception:  # noqa: BLE001
        return None


def _declared_from(geometry) -> dict[str, int]:
    return {axis: getattr(geometry, axis) for axis in AXES}


@pytest.mark.parametrize("target", TARGETS)
def test_every_target_either_agrees_or_says_why_it_could_not_look(target: str):
    """Three-state, per target, and never a silent pass."""
    geometry = _derived(target)
    declared = _declared_from(geometry) if geometry else dict.fromkeys(AXES, 1)
    verdict = crosscheck_declared_geometry(declared, target)
    if geometry is None:
        assert verdict.startswith("unavailable:"), verdict
        assert len(verdict) > len("unavailable:") + 10, "the reason it could not look is not recorded"
    else:
        assert verdict.startswith("agrees with"), verdict


def test_at_least_one_target_can_be_compared_and_at_least_one_cannot():
    """The existence proof.

    Every per-target assertion above is satisfied by a function that answers "unavailable" for
    everything. Requiring both outcomes to occur is what makes the parametrized test able to fail.
    """
    derivable = [t for t in TARGETS if _derived(t) is not None]
    assert derivable, "no target derives a geometry; the comparison asserts nothing in this checkout"
    assert len(derivable) < len(TARGETS), (
        "every target derived one; the unavailable branch is untested, and it is the branch a sandbox always takes"
    )


@pytest.mark.parametrize("axis", AXES)
def test_a_disagreement_on_any_axis_alone_raises_and_names_both_numbers(axis: str):
    """Parametrized per axis so an axis dropped from the comparison fails its own case.

    Asserted in a lump, a comparison that only ever checked one axis would pass.
    """
    target = next((t for t in TARGETS if _derived(t) is not None), None)
    if target is None:
        pytest.skip("no target derives a geometry in this checkout")
    declared = _declared_from(_derived(target))
    wrong = dict(declared, **{axis: declared[axis] + 1})
    with pytest.raises(GeometryDisagreement) as excinfo:
        crosscheck_declared_geometry(wrong, target)
    message = str(excinfo.value)
    assert f"{axis} declared={wrong[axis]} derived={declared[axis]}" in message, message
    assert "refuses rather than picking" in message, "the refusal does not say why it will not choose"


def test_an_unrecognised_axis_is_refused_rather_than_ignored():
    """A caller translating its own spelling must not be able to pass a name nothing compares.

    Silently ignoring one would mean an axis the caller believed was checked never was -- which is worse
    than not checking it, because the receipt would say it agreed.
    """
    with pytest.raises(GeometryDisagreement, match="not one of"):
        crosscheck_declared_geometry({"spad_rows": 16384}, "gemmini")


def test_no_target_named_is_a_distinct_state_from_agreement():
    assert crosscheck_declared_geometry({}, None) == "not checked: no target named"
    assert crosscheck_declared_geometry({}, "") == "not checked: no target named"


def test_the_core_comparison_names_no_target():
    """The property that makes this shareable, asserted where it is easy to lose.

    The repo-wide gate scans the whole core tree and carries a ratchet; this is the local, unratchetable
    statement that THIS file must never acquire a target literal -- because the moment it does, the
    comparison has quietly become one target's again.
    """
    import sys

    from merlin.common.paths import repo_root

    sys.path.insert(0, str(repo_root() / "build_tools" / "scripts"))
    from _target_roster import target_names

    source = (repo_root() / "merlin/python/merlin/sched/mach/crosscheck.py").read_text(encoding="utf-8")
    named = sorted(t for t in target_names(repo_root()) if t.lower() in source.lower())
    assert not named, f"the shared comparison names {named}; it is shared exactly because it names none"


def test_a_machine_whose_geometry_moved_is_caught_end_to_end():
    """The mutation, through the real derivation rather than a hand-built geometry.

    A derivation that reports the wrong array edge must make the comparison refuse -- this is the same
    property the target binding's own test asserts, checked here at the shared seam so it holds for
    every caller rather than for the one that happens to exist.
    """
    target = next((t for t in TARGETS if _derived(t) is not None), None)
    if target is None:
        pytest.skip("no target derives a geometry in this checkout")
    declared = _declared_from(_derived(target))

    import merlin.sched.mach.derive as derive_module

    real = derive_module.derive
    try:
        derive_module.derive = lambda t, **kw: dataclasses.replace(real(t, **kw), array_edge=declared["block"] * 2)
        with pytest.raises(GeometryDisagreement, match="block declared="):
            crosscheck_declared_geometry(declared, target)
    finally:
        derive_module.derive = real
