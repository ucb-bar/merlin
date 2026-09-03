"""A directory name is not always the target name, and asking by either must work.

A capsule-bench descriptor sits in a short directory and declares the configuration-qualified name
that every generated artifact path uses. Nothing reconciled the two, and this repo has now paid for it
FOUR separate times:

* `check_conformance_coverage` reported `no_target_experiment` for two saturn targets and exited 0 --
  a gate reporting success for a question it never asked;
* the tracked conformance specs were audited under their filename rather than their declared `target:`;
* `generate_corpus.py --target <declared name>` dies with a missing-descriptor traceback;
* `boundary.profile_capsule` raised a bare `FileNotFoundError` from deep inside a caller's stack for a
  contract that exists perfectly well under the declared name.

So the hop belongs in the RESOLVER, where every caller inherits it, rather than in each caller. It is
tried LAST so it can never shadow a target that resolves on its own, and exactly once so a descriptor
naming itself cannot loop.
"""
from __future__ import annotations

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import target_registry as TR

_TARGETS = repo_root() / "merlin" / "experiments" / "capsule_bench" / "targets"


def _pairs() -> list[tuple[str, str]]:
    """``(directory, declared)`` for every descriptor whose two names differ."""
    out = []
    for desc in sorted(_TARGETS.glob("*/target_experiment.yaml")):
        try:
            doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        declared = str(doc.get("target") or "")
        if declared and declared != desc.parent.name:
            out.append((desc.parent.name, declared))
    return out


def test_the_declared_name_is_discoverable_from_the_directory_name():
    pairs = _pairs()
    if not pairs:
        pytest.skip("no target declares a name differing from its directory")
    for directory, declared in pairs:
        assert TR.declared_target_for(directory) == declared


def test_a_target_whose_names_agree_reports_no_hop():
    """`None` distinguishes "no hop available" from "hop to X"; conflating them hides the mapping."""
    agree = []
    for desc in sorted(_TARGETS.glob("*/target_experiment.yaml")):
        try:
            doc = yaml.safe_load(desc.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        if str(doc.get("target") or "") == desc.parent.name:
            agree.append(desc.parent.name)
    if not agree:
        pytest.skip("every descriptor declares a differing name")
    for name in agree:
        assert TR.declared_target_for(name) is None
    assert TR.declared_target_for("definitely_not_a_directory") is None


def test_resolving_by_either_name_finds_the_same_contract():
    """The property the four defects all wanted: either spelling reaches the generated package."""
    pairs = _pairs()
    if not pairs:
        pytest.skip("no target declares a name differing from its directory")
    checked = 0
    for directory, declared in pairs:
        by_declared = TR.resolve(declared)
        if not by_declared.contract_path.is_file():
            continue                       # this target's package is not generated in this checkout
        by_directory = TR.resolve(directory)
        assert by_directory.contract_path == by_declared.contract_path, (
            f"{directory!r} and {declared!r} resolved to different contracts")
        assert by_directory.contract_path.is_file()
        checked += 1
    if not checked:
        pytest.skip("no differing-name target has a generated package here")


def test_the_hop_never_shadows_a_target_that_resolves_on_its_own():
    """Tried last, so a real target keeps its own identity even if some directory names it."""
    info = TR.resolve("gemmini")
    assert info.name == "gemmini"


def test_a_missing_contract_says_which_target_and_which_path():
    """A bare FileNotFoundError from inside a caller's stack is not 'surfacing it honestly'."""
    info = TR.resolve("definitely_not_a_target")
    with pytest.raises(TR.TargetContractMissing) as exc:
        info.load_contract()
    msg = str(exc.value)
    assert "definitely_not_a_target" in msg
    assert "target_contract.yaml" in msg
    # And it must point the reader at the name mapping, since that is the usual cause.
    assert "declared" in msg
