"""No shipped performance family may contradict itself.

Three families shipped with declarations no admissible measurement could satisfy: one gate demanded
at least two distinct values of a quantity its own comparand held equal, and two falsifiers fired on
a range or band the declaration never carried. Each still generated capsules, still measured, and
still reported -- they simply could never fail, which is the one thing a falsifier must be able to
do. Nothing caught it because nothing asserted it, so this does.
"""
from __future__ import annotations

import dataclasses
import importlib

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


@pytest.mark.parametrize("family_id", [fid for fid, _ in _families()[0]])
def test_an_emitter_declared_existing_actually_exists(family_id):
    """``emitter.status: existing`` is a claim about this tree, so resolve it rather than trust it.

    MEASURED, and this is why the test exists: the synchronization family declared
    ``entry: merlin.perf.barrier_arms.pair_from_emitter`` with ``status: existing``. The module
    exists; that function never did, and neither did the ``retire knob`` its ``knobs`` named. Eleven
    capsules shipped declaring an emitter against nothing, so the family's second arm could not be
    built and ten of them measured one arm against no comparand. Generation succeeded, the capsules
    materialised, and the gap was invisible from every artifact -- an emitter is only consulted when
    someone tries to build the arm, and nobody had.

    Import-and-getattr is the whole check. It is cheap, it cannot pass vacuously, and it fails in the
    one direction that matters: a declaration naming something absent.
    """
    performance = dict(_families()[0])[family_id]
    emitter = performance.get("emitter") or {}
    if str(emitter.get("status")) != "existing":
        pytest.skip(f"{family_id} does not declare an existing emitter")
    entry = str(emitter.get("entry") or "")
    assert entry and not entry.startswith("new:"), (
        f"{family_id} declares status 'existing' with entry {entry!r}, which names work not yet done")
    module_path, _, attribute = entry.rpartition(".")
    assert module_path and attribute, f"{family_id}: entry {entry!r} is not a module path plus a name"
    module = importlib.import_module(module_path)
    assert hasattr(module, attribute), (
        f"{family_id} declares emitter {entry!r} as existing, but {module_path} defines no "
        f"{attribute!r}; the family cannot build its arms")


@pytest.mark.parametrize("family_id", [fid for fid, _ in _families()[1]])
def test_a_blocked_family_does_not_claim_its_emitter_exists(family_id):
    """The mirror direction: a family recorded as blocked must not also claim a working emitter, or
    the block reason and the emitter record disagree about whether the work is done."""
    performance = dict(_families()[1])[family_id]
    emitter = performance.get("emitter") or {}
    status, entry = str(emitter.get("status")), str(emitter.get("entry") or "")
    if status != "existing":
        return
    module_path, _, attribute = entry.rpartition(".")
    assert module_path and attribute and not entry.startswith("new:"), (
        f"{family_id} is blocked yet declares emitter {entry!r} as existing")
    module = importlib.import_module(module_path)
    assert hasattr(module, attribute), (
        f"{family_id} is blocked and its 'existing' emitter {entry!r} does not resolve either")
