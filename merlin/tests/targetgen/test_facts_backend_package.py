"""The RTL-facts accessor and the bundle generator must resolve to the SAME artifact.

An experiment target need not share a name with the backend package that holds its contracts — a SIMT
experiment is served by its core's package — which is why a descriptor may declare
``backend_package_dir``, and why :attr:`TargetExperiment.rtl_facts_pin` exists. The bundle generator
reads that declaration and mounts the right artifact into the agent sandbox.

The accessor did not. It assumed ``merlin/targets/<target>/contracts/rtl_facts/`` and, for a target
that declares otherwise, looked somewhere that can never exist. In the sandbox the purgeable cache is
not granted, so both lookups missed, regeneration produced ``facts: {}``, and every RTL generator the
CIRCT arm is granted failed with ``FactsEmpty`` — while its own bundle had mounted a populated artifact
one symlink away. Measured before the fix: 0 fact groups from the accessor against 7 in the mount.

These tests pin the invariant (accessor == bundle pin) and the fail-closed behaviour that let the
emptiness travel silently.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import merlin_dir, repo_root, targets_dir
from merlin.targetgen import target_experiment as TE
from merlin.targetgen.rtl import facts as F


def _descriptors():
    root = merlin_dir() / "experiments" / "capsule_bench" / "targets"
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):
        p = d / "target_experiment.yaml"
        if p.is_file():
            try:
                out.append(TE.load_target_experiment(p))
            except Exception:  # noqa: BLE001 — a malformed descriptor is a different test's problem
                pass
    return out


def test_every_descriptor_resolves_the_accessor_to_its_own_declared_pin():
    """The invariant that was broken: what the accessor finds is what the bundle mounts."""
    tes = _descriptors()
    if not tes:
        pytest.skip("no target-experiment descriptors in this checkout")
    checked = 0
    for te in tes:
        pin = repo_root() / te.rtl_facts_pin / "facts.json"
        if not pin.is_file():
            continue                      # this target ships no committed artifact; nothing to agree on
        got = F._committed_facts_path(te.target)
        assert got is not None, f"{te.target}: bundle mounts {pin} but the accessor found nothing"
        assert got.resolve() == pin.resolve(), (
            f"{te.target}: accessor resolved {got} but the bundle mounts {pin}")
        checked += 1
    if not checked:
        pytest.skip("no descriptor in this checkout ships a committed facts artifact")


def test_a_target_named_differently_from_its_backend_package_is_still_resolved():
    """The case the name convention alone cannot serve. Guarded, not skipped away: if some descriptor
    declares a backend package under another name, the accessor must follow the DECLARATION — and the
    name-convention path must be shown absent, so the test proves the declaration did the work."""
    off = [te for te in _descriptors()
           if te.backend_package_dir and te.backend_package != f"merlin/targets/{te.target}"]
    if not off:
        pytest.skip("no descriptor declares a backend package under a different name")
    for te in off:
        pin = repo_root() / te.rtl_facts_pin / "facts.json"
        if not pin.is_file():
            continue
        convention = targets_dir() / te.target / "contracts" / "rtl_facts" / "facts.json"
        assert not convention.is_file(), (
            f"{te.target}: the name-convention path exists, so this target no longer exercises the case")
        got = F._committed_facts_path(te.target)
        assert got is not None and got.resolve() == pin.resolve()
        body = json.loads(got.read_text()).get("facts") or {}
        assert body, f"{te.target}: the declared pin resolved but carries no facts"


def test_the_declared_pin_is_used_even_when_the_cache_is_cold(tmp_path, monkeypatch):
    """The sandbox condition. The purgeable cache is not granted inside the box, so a cold out-root is
    what the agent actually experiences; the committed pin is what has to answer."""
    off = [te for te in _descriptors()
           if te.backend_package_dir and (repo_root() / te.rtl_facts_pin / "facts.json").is_file()]
    if not off:
        pytest.skip("no descriptor declares a populated backend-package facts pin")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    te = off[0]
    assert not F.rtl_facts_path(te.target).is_file(), "fixture bug: the cold cache is not cold"
    body = F.load_facts(te.target).get("facts") or {}
    assert body, f"{te.target}: cold-cache load_facts returned an EMPTY body"


def test_an_empty_regeneration_fails_closed_instead_of_returning_nothing(tmp_path, monkeypatch):
    """``facts: {}`` used to be returned as success, so ``load_facts`` handed back an empty dict and
    only a caller routing through ``facts_body``/``decode_body`` ever noticed. Raise where it is made."""
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    fake = "notatarget_for_this_test"
    assert F._committed_facts_path(fake) is None, "fixture bug: the fake target ships an artifact"

    def _empty(p, target):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"inputs": {}, "facts": {}}), encoding="utf-8")

    monkeypatch.setattr(F, "_dump_facts_for_kind", _empty)
    monkeypatch.setattr(F, "_warn_if_degraded", lambda *a, **k: None)
    with pytest.raises(F.FactsEmpty):
        F.ensure_facts(fake)


def test_descriptor_for_prefers_the_env_pointer_only_when_it_names_that_target(monkeypatch, tmp_path):
    tes = _descriptors()
    if not tes:
        pytest.skip("no target-experiment descriptors in this checkout")
    te = tes[0]
    monkeypatch.delenv("MERLIN_TARGET_EXPERIMENT", raising=False)
    assert TE.descriptor_for(te.target) is not None
    bogus = tmp_path / "target_experiment.yaml"
    bogus.write_text("target: some_other_target\n")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", str(bogus))
    got = TE.descriptor_for(te.target)
    assert got is not None and got != bogus, "an env pointer for a DIFFERENT target must not win"
