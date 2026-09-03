"""Which tier a WHOLE-MODEL capsule can be cited at is derived from the target, never written down.

A model capsule never enters the tier ladder.  Its tier block is synthesised by ``_model_tier_map``
from the model's own layer accounting, and that block carries exactly ONE execution tier: the last
declared tier the target's capability manifest counts as RTL.  L0/L1 are honestly not_applicable (a
whole model has no command buffer to interpret) and every other declared tier is simply never emitted.

A consumer that gates on a tier by name therefore gates on a tier that may not exist.  Measured: an
Arm-4 performance campaign required ``L2 == "pass"`` and ``L3 == "pass"`` from a full-model admission
run.  A target whose RTL tiers are ``L3/L4/L5`` emits only ``L3``, so ``L2`` read ``None`` on a
FLAWLESS model run, admission failed, and the campaign raised before measuring a single cell.

``model_citable_rtl_tier`` is the supported way to ask which tier that is, and these tests pin that it
cannot drift away from what the emitter actually produces.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.target_experiment import load_capability_manifest

# Per-target edges: the point of the case is that these ladders DIFFER, and that neither is a literal
# anywhere in library code.  A target whose manifest cannot be resolved here is skipped by omission,
# not by a soft pass -- the parametrisation asserts at least one resolved.
_TARGETS = ("gemmini", "atlas", "radiance")


def _resolvable() -> list[str]:
    resolved = []
    for name in _TARGETS:
        try:
            load_capability_manifest(name)
        except Exception:  # noqa: BLE001 — target not present in this checkout
            continue
        if CR.rtl_tiers_of(name):
            resolved.append(name)
    return resolved


def test_at_least_one_target_manifest_resolves_so_the_rest_is_not_vacuous() -> None:
    assert _resolvable(), "no target's RTL tiers were derivable; every test below would be vacuous"


@pytest.mark.parametrize("target", _TARGETS)
def test_the_citable_tier_is_exactly_the_tier_the_emitter_produces(target: str) -> None:
    if target not in _resolvable():
        pytest.skip(f"{target}'s capability manifest is not present in this checkout")
    rtl = CR.rtl_tiers_of(target)
    perfect = {"matmul_layers_on_mesh": 3, "matmul_layers_host_fallback": 0}
    for declared in (["L0", "L1", "L2", "L3"], ["L2", "L3"], ["L0", "L1", "L2", "L3", "L4"]):
        citable = CR.model_citable_rtl_tier(declared, target)
        emitted = {name for name, tier in CR._model_tier_map(declared, target, perfect).items()
                   if not tier.not_applicable}
        if citable is None:
            # No declared tier is RTL here: the emitter still names a tier so a refusal stays
            # attributable, but that label is not a citable hardware verdict.
            assert not (emitted & rtl)
            continue
        assert emitted == {citable}, f"{target}: gate would name {citable}, emitter names {emitted}"
        assert citable in rtl
        assert citable == [t for t in declared if t in rtl][-1]


def test_a_declaration_with_no_rtl_tier_derives_nothing_rather_than_guessing() -> None:
    """FAIL CLOSED. An underivable ladder must yield ``None``, never a plausible tier name."""
    assert CR.model_citable_rtl_tier([], "gemmini") is None
    assert CR.model_citable_rtl_tier(["L2", "L3"], None) is None
    assert CR.model_citable_rtl_tier(["L2", "L3"], "") is None
    # An unresolvable manifest fails soft to an empty RTL set, which must read as "no citable tier"
    # rather than falling through to the last declared name.
    assert CR.rtl_tiers_of("no-such-target-exists") == frozenset()
    assert CR.model_citable_rtl_tier(["L2", "L3"], "no-such-target-exists") is None


def test_l0_and_l1_are_never_citable_because_a_model_has_no_command_buffer() -> None:
    resolved = _resolvable()
    assert resolved, "vacuous without a resolvable target"
    for target in resolved:
        assert CR.model_citable_rtl_tier(["L0", "L1"], target) is None
        tiers = CR._model_tier_map(["L0", "L1"], target, {"matmul_layers_on_mesh": 2,
                                                          "matmul_layers_host_fallback": 0})
        assert all(tier.not_applicable for tier in tiers.values())
