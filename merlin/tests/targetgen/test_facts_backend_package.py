"""A target whose BACKEND PACKAGE is not named after it must still find its committed RTL-facts pin.

WHY THIS EXISTS. ``merlin/targets/<target>/contracts/rtl_facts/facts.json`` is only the pin location when
the experiment target and the package that serves it share a name. They do not always: a SoC experiment
can be served by its core's package, which is exactly why ``target_experiment.yaml`` carries
``backend_package_dir`` and says the mapping "cannot be inferred from the target name".

The BUNDLE GENERATOR already read that declaration and granted the right directory. The facts ACCESSOR
did not: it assumed the target name and looked somewhere that can never exist. Inside the agent sandbox
neither the external RTL checkout nor the purgeable introspect cache is mounted, so both lookups missed,
regeneration produced ``facts: {}``, ``rtl_backend.target_profile`` came back all-``None``, and every
RTL-derived authoring tool the assisted arm is granted raised ``FactsEmpty`` — a launch NO-GO for a
target whose own bundle mounted a perfectly good artifact.

So the invariant under test is agreement: where the sandbox MOUNTS a target's pin and where the accessor
LOOKS for it must be the same file.
"""
from __future__ import annotations

import json

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.rtl import facts as F
from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment


def _pin(root, package: str, declares: str):
    """Write a minimal committed facts pin for ``package`` whose body declares target ``declares``."""
    p = root / package / "contracts" / "rtl_facts" / "facts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema_version": "test/v0",
                             "facts": {"target": declares, "memories": [{"name": "smem", "bytes": 1}]}}),
                 encoding="utf-8")
    return p


@pytest.fixture()
def fake_targets_root(tmp_path, monkeypatch):
    """A targets root with ONE package whose name is not the target it serves."""
    root = tmp_path / "targets"
    root.mkdir()
    monkeypatch.setattr("merlin.common.paths.targets_dir", lambda: root)
    monkeypatch.setattr(F, "targets_dir", lambda: root, raising=False)
    return root


def test_pin_found_when_package_name_differs_from_target_name(fake_targets_root):
    """THE REGRESSION: the pin lives in a differently-named package and must still be found.

    Nothing here names a target; the artifact itself says which target it is about, which is the only
    thing available inside the sandbox (the descriptor and the target contract are not mounted there).
    """
    pin = _pin(fake_targets_root, "core_pkg", declares="soc_name")
    assert F._committed_facts_path("soc_name") == pin


def test_pin_is_refused_for_the_package_it_is_not_about(fake_targets_root):
    """FAIL CLOSED, not by directory. A pin sitting under ``<package>/`` is not automatically that
    package's own facts: matched by what it DECLARES, so asking for the package name gets nothing rather
    than another target's hardware facts."""
    _pin(fake_targets_root, "core_pkg", declares="soc_name")
    assert F._committed_facts_path("core_pkg") is None


def test_same_name_target_still_resolves_by_convention(fake_targets_root):
    """The common case is untouched: package and target share a name and the pin declares that target."""
    pin = _pin(fake_targets_root, "same_name", declares="same_name")
    assert F._committed_facts_path("same_name") == pin


def test_a_pin_that_declares_nothing_is_accepted_by_convention(fake_targets_root):
    """A pin whose body carries no ``target`` key is not evidence of a MISMATCH, so the naming
    convention still serves it — absence of a claim must not read as a contradictory claim."""
    root = fake_targets_root
    p = root / "quiet" / "contracts" / "rtl_facts" / "facts.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"facts": {"memories": []}}), encoding="utf-8")
    assert F._committed_facts_path("quiet") == p


def _descriptor_targets_with_pins():
    """(target, descriptor) for every shipped descriptor whose declared pin actually exists on disk."""
    out = []
    exp = repo_root() / "merlin" / "experiments"
    for cand in sorted(exp.glob("*/targets/*/target_experiment.yaml")):
        try:
            te = load_target_experiment(cand)
        except Exception:                                   # noqa: BLE001 - a broken descriptor is not this test
            continue
        if (repo_root() / te.rtl_facts_pin / "facts.json").is_file():
            out.append(pytest.param(te.target, id=te.target))
    return out


@pytest.mark.parametrize("target", _descriptor_targets_with_pins())
def test_mounted_pin_and_resolved_pin_are_the_same_file(target):
    """THE AGREEMENT. For every shipped target that has a committed pin, the directory the sandbox binder
    resolves for the bundle's ``rtl_facts`` grant and the file the accessor returns must be the same
    artifact. When they disagree the arm is handed bytes its own library cannot find — which is exactly
    how a target read as having empty RTL facts while its bundle mounted them."""
    te = load_target_experiment(descriptor_for(target))
    mounted = BW.resolve_grant(te.rtl_facts_pin, repo_root())
    resolved = F._committed_facts_path(target)
    assert resolved is not None, f"{target}: no committed pin resolved, but its descriptor declares one"
    assert resolved.parent.resolve() == mounted.resolve()


@pytest.mark.parametrize("target", _descriptor_targets_with_pins())
def test_facts_load_from_the_committed_pin_alone(target, monkeypatch, tmp_path):
    """With the purgeable cache unreachable (the sandbox's situation), the pin alone must serve non-empty
    facts — no regeneration, no external RTL checkout."""
    monkeypatch.setattr(F, "rtl_cache_dir", lambda t, ensure=False: tmp_path / "cold" / t)
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    body = (F.load_facts(target) or {}).get("facts") or {}
    assert body, f"{target}: the committed pin served no facts with the cache cold"
