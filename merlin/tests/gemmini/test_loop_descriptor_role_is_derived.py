"""The no-FSM rule names this target's hardware-loop instructions, derived from its own facts.

A sealed Phase 0 release prohibited ``loop_descriptor`` and resolved it to ZERO instructions: the
endpoint declaration that binds the role to this target's RTL names was read from the wrong place and
came back empty, so every later phase enforced a rule that forbade nothing. The count is not written
here; the expected set is read from the target's own derived name table -- every instruction its RTL
spells as a hardware loop (the ``LOOP_`` family: the weight-stationary matmul loop and the convolution
loop with their configuration words) -- and the role binding must cover exactly that set, through
both the core derivation and the Phase 0 taxonomy that is sealed into a release.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.target("gemmini")

TARGET = "gemmini"
ROLE = "loop_descriptor"
#: This target's own spelling of its hardware-loop family, as its RTL decode table names it.
LOOP_FAMILY = "LOOP_"


@pytest.fixture(scope="module")
def names() -> dict:
    from merlin.kernels.decode.rocc import funct_table_for

    try:
        table = funct_table_for(TARGET).get("names") or {}
    except Exception as exc:  # noqa: BLE001 -- no RTL facts on this host is a skip, not a pass
        pytest.skip(f"{TARGET} RTL facts unavailable: {type(exc).__name__}: {exc}")
    if not table:
        pytest.skip(f"{TARGET} RTL facts carry no instruction name table on this host")
    return {str(k): str(v) for k, v in table.items()}


def test_the_role_covers_exactly_the_targets_hardware_loop_family(names):
    from merlin.kernels.endpoints import endpoints_for

    endpoints = endpoints_for(TARGET)
    assert endpoints, "no compute endpoint binds this target's roles"
    bound = {name for name in names.values() for endpoint in endpoints if ROLE in endpoint.roles_of(name)}
    family = {name for name in names.values() if name.startswith(LOOP_FAMILY)}
    assert family, "the derived name table has no hardware-loop instructions; this test would prove nothing"
    assert bound == family, (sorted(bound - family), sorted(family - bound))


def test_the_phase0_policy_resolves_the_role_to_that_family(names):
    roles = pytest.importorskip("merlin_experiments.phase0.instruction_roles")
    taxonomy = roles.derive_role_taxonomy(TARGET)
    assert taxonomy["status"] == "derived", taxonomy.get("reason")
    policy = roles.resolve_policy([ROLE], taxonomy)
    assert policy["status"] == roles.RESOLVED and policy["vacuous_roles"] == []
    prohibited = {row["name"] for row in policy["prohibited_instructions"][ROLE]}
    assert prohibited == {name for name in names.values() if name.startswith(LOOP_FAMILY)}
    assert roles.enforcement_problems(policy, [ROLE]) == []
