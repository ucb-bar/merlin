"""No shipped performance family may contradict itself.

Three families shipped with declarations no admissible measurement could satisfy: one gate demanded
at least two distinct values of a quantity its own comparand held equal, and two falsifiers fired on
a range or band the declaration never carried. Each still generated capsules, still measured, and
still reported -- they simply could never fail, which is the one thing a falsifier must be able to
do. Nothing caught it because nothing asserted it, so this does.
"""
from __future__ import annotations

import dataclasses

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.perf import claim_reach as CR

PROFILE = repo_root() / "merlin" / "contract" / "capsules" / "profiles" / "_perf.yaml"


def _families():
    document = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    emitted = [(s["id"], s["base"]["performance"]) for s in document.get("sweeps") or []]
    blocked = [(b["family"], b["performance"]) for b in document.get("blocked_unimplemented") or []
               if isinstance(b.get("performance"), dict)]
    return emitted, blocked


def _reach(performance):
    result = CR.family_reach(performance)
    return {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}


@pytest.mark.parametrize("family_id", [fid for fid, _ in _families()[0]])
def test_every_emitted_family_is_satisfiable(family_id):
    performance = dict(_families()[0])[family_id]
    reach = _reach(performance)
    assert reach["satisfiable"], (
        f"{family_id} cannot be satisfied by any admissible measurement: {reach['obstructions']}")


@pytest.mark.parametrize("family_id", [fid for fid, _ in _families()[1]])
def test_a_blocked_family_is_still_coherent(family_id):
    """Blocked means the target cannot run it, not that its declaration may be incoherent.

    A family recorded as blocked is expected to come back when the capability arrives, so it has to
    be sound now -- otherwise the block hides a contradiction that resurfaces later as a claim that
    cannot fail.
    """
    performance = dict(_families()[1])[family_id]
    reach = _reach(performance)
    assert reach["satisfiable"], f"{family_id} is blocked AND self-contradictory: {reach['obstructions']}"


def test_the_profile_names_no_target():
    """This template is shared by every target; one target's fact here is every target's bug."""
    text = PROFILE.read_text(encoding="utf-8").lower()
    for name in ("gemmini", "atlas", "radiance", "saturn", "mx_gemmini"):
        assert name not in text, f"the shared performance template names the target {name!r}"


def test_no_family_declares_a_predictive_claim_without_a_contract():
    """A PREDICTS family with no frozen acceptance is a claim nothing can decide."""
    for family_id, performance in _families()[0]:
        if performance.get("claim") == "PREDICTS":
            assert isinstance(performance.get("acceptance"), dict), (
                f"{family_id} claims PREDICTS with no frozen acceptance contract")
