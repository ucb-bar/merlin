"""Two derivations of one target's on-chip geometry must not disagree.

The block-scheduling pass reads a target's facts through ``address_space`` and projects them onto a
``Geometry``; the machine model reads the SAME facts and describes more of them. Both now exist, which
means the repo has two answers to "how many operand rows does this device have" and nothing that
notices when they stop matching.

The 42 committed golden cells cannot notice: they hold the pass to a recorded instruction stream, so a
derivation change that moves BOTH derivations the same way leaves every cell green. That is exactly the
class this file catches, which is why it is worth its own test rather than a line in that one.
"""

from __future__ import annotations

import dataclasses

import pytest

from merlin.compile.scheduling import BlockScheduleError
from merlin.compile.scheduling.derive import geometry_from_address_space, geometry_from_machine
from merlin.sched.mach.derive import derive
from merlin.targetgen.address_space import derive_address_space

pytestmark = pytest.mark.target("gemmini", "gemmini_universal", "atlas", "muon", "toy_npu")

#: Every target either derivation can be asked about. Both a target that RESOLVES and one that REFUSES
#: belong here: a projection that refused everything would agree with a derivation that refused
#: everything, and only a resolving target makes the equality assertion say anything.
TARGETS = ("gemmini", "gemmini_universal", "atlas", "muon", "toy_npu", "saturn_opu_mxv256d128")


def _geometry(fn, arg):
    """``(geometry_without_sources, None)`` or ``(None, refusal)``. Sources differ by construction --
    each derivation cites its own path to the number -- so the comparison is over the numbers."""
    try:
        return dataclasses.replace(fn(arg), sources={}), None
    except BlockScheduleError as exc:
        return None, str(exc)


def _pair(target: str):
    return (
        _geometry(geometry_from_address_space, derive_address_space(target)),
        _geometry(geometry_from_machine, derive(target)),
    )


@pytest.mark.parametrize("target", TARGETS)
def test_the_two_derivations_agree_or_refuse_together(target: str):
    (space_geom, space_refusal), (machine_geom, machine_refusal) = _pair(target)
    if space_refusal is None and machine_refusal is None:
        assert space_geom == machine_geom, (
            f"{target}: the address space and the machine describe different geometry.\n"
            f"  from address_space: {space_geom}\n  from machine:       {machine_geom}"
        )
        return
    assert (space_refusal is None) == (machine_refusal is None), (
        f"{target}: one derivation resolved a geometry and the other refused. "
        f"address_space={space_refusal or space_geom}; machine={machine_refusal or machine_geom}"
    )


def test_at_least_one_target_resolves_and_at_least_one_refuses():
    """The anti-vacuity guard.

    Every assertion above is satisfied by two derivations that refuse everything, and a refactor that
    broke both identically would leave this file green. Requiring both outcomes to occur is what makes
    the parametrized test able to fail.
    """
    outcomes = {t: [refusal for _, refusal in _pair(t)] for t in TARGETS}
    resolved = [t for t, (a, b) in outcomes.items() if a is None and b is None]
    refused = [t for t, (a, b) in outcomes.items() if a is not None and b is not None]
    assert resolved, f"no target resolved a geometry; the comparison asserts nothing. {outcomes}"
    assert refused, f"no target refused; the refusal paths are untested. {outcomes}"


def test_the_comparison_baseline_derives_the_same_geometry_as_the_machine():
    """A THIRD derivation of the same facts exists, and it stays independent -- but checked.

    The comparison baseline derives its own geometry deliberately: a comparison must not let a
    machine-model choice of ours decide that the thing being compared against fails. That independence
    is worth keeping and is exactly why nothing would notice the two drifting apart, so this compares
    them without coupling them. A disagreement here says one of the two derivations is wrong; it does
    not say which, and neither should be changed to match the other without finding out.
    """
    voyager = pytest.importorskip("merlin.baselines.voyager")
    try:
        theirs = voyager.geometry_for("gemmini")
    except Exception as exc:  # noqa: BLE001 - unresolvable here is not a disagreement
        pytest.skip(f"the comparison baseline cannot derive a geometry in this checkout: {exc}")
    machine = derive("gemmini")
    ours = geometry_from_machine(machine)
    assert (theirs.dim, theirs.spad_rows, theirs.acc_rows) == (
        machine.array_edge,
        ours.operand_rows,
        ours.accumulator_rows,
    ), (
        f"two independent derivations of one target's geometry disagree: baseline "
        f"dim={theirs.dim} spad_rows={theirs.spad_rows} acc_rows={theirs.acc_rows}; machine "
        f"edge={machine.array_edge} operand_rows={ours.operand_rows} acc_rows={ours.accumulator_rows}"
    )


def test_the_projection_carries_the_machine_it_came_from():
    """A geometry that cannot say which machine produced it cannot be attributed to one, and a
    measurement keyed on it would be reused across two machines that differ."""
    geometry = geometry_from_machine(derive("gemmini"))
    assert geometry.sources.get("machine") == derive("gemmini").digest()


def test_a_perturbed_machine_moves_the_projection():
    """The mutation: if a projected field did not actually come from the machine, changing the machine
    would not change the projection, and the agreement test above would be comparing a constant."""
    machine = derive("gemmini")
    base = geometry_from_machine(machine)
    memories = tuple(
        dataclasses.replace(m, rows=(m.rows or 0) + 16) if not m.accumulates else m for m in machine.memories
    )
    moved = geometry_from_machine(dataclasses.replace(machine, memories=memories))
    assert moved.operand_rows == base.operand_rows + 16, "operand_rows does not come from the machine's memory"
    edged = dataclasses.replace(machine, array_edge=(machine.array_edge or 0) * 2)
    assert geometry_from_machine(edged).block == base.block * 2, "block does not come from the machine's array edge"
