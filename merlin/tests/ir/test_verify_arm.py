"""The verification-seam arm: it exists, it is exactly arm-4 plus one tool, and it did not widen arm-4.

The compiler-verification layer is a treatment, so the arm carrying it has to be attributable. These
pin the three ways that can quietly stop being true:

* the seam leaks into ``merlin_rtlchecks`` — arm-4 then differs from arm-3 in two ways at once and every
  arm-3-vs-arm-4 number already reported stops being comparable to the ones taken after;
* the verify arm's treatment is not actually a treatment — the merlin-opt driver sits inside the whole
  ``xdsl_dialects/`` directory that ``xdsl_kit`` already grants, so leaving it merely off the allow list
  grants it anyway and half the declared difference does not exist;
* the default ladder grows a sixth bundle — every committed bundle dir, run path and A/B report resolves
  against the five stems, so a sixth arriving by default changes the arm set under runs in flight.
"""
from __future__ import annotations

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.generate_bundles import _ARMS, _OPT_IN_ARMS, _arm_manifest, generate_bundles
from merlin.targetgen.target_experiment import load_target_experiment

# One real descriptor is enough: every verify_seam path is a literal shared by all targets, and the one
# target-varying grant in the arm (the RTL facts pin) reaches the manifest through a descriptor attribute.
_DESCRIPTOR = "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"


@pytest.fixture(scope="module")
def te():
    return load_target_experiment(repo_root() / _DESCRIPTOR)


def _sets(te, arm, **kw):
    m = _arm_manifest(te, arm, "bid", **kw)
    return ({e["path"] for e in m["allowed"]}, {e["path"] for e in m["denied"]}, m)


def test_the_seam_grants_paths_that_exist():
    """A grant naming a path that moved grants nothing, and the arm loses its whole treatment silently."""
    missing = [p for p in TR.TOOLS["verify_seam"].bundle_paths if not (repo_root() / p).exists()]
    assert not missing, f"verify_seam names paths that do not exist: {missing}"


def test_the_verify_arm_is_arm4_plus_exactly_the_seam():
    added = set(TR.ARM_TOOLS["merlin_verify"]) - set(TR.ARM_TOOLS["merlin_rtlchecks"])
    assert added == {"verify_seam"}
    assert set(TR.ARM_TOOLS["merlin_rtlchecks"]) <= set(TR.ARM_TOOLS["merlin_verify"])


def test_arm4_did_not_gain_the_seam():
    """The whole point of a new arm rather than a wider one: arm-4's tool set is untouched."""
    assert "verify_seam" not in TR.ARM_TOOLS["merlin_rtlchecks"]
    assert TR.ARM_TOOLS["merlin_rtlchecks"] == TR.ARM_TOOLS["merlin_assisted"] + ("rtl_generators",
                                                                                 "rtl_facts")


def test_the_seam_is_denied_not_merely_absent_on_the_arms_without_it(te):
    """``xdsl_kit`` grants ``xdsl_dialects/`` as a directory, which re-exposes the merlin-opt driver."""
    seam = set(TR.TOOLS["verify_seam"].bundle_paths)
    for arm in ("merlin_assisted", "merlin_rtlchecks", "merlin_eqsat"):
        _, denied, _ = _sets(te, arm)
        assert seam <= denied, f"{arm} does not mask {sorted(seam - denied)}"


def test_the_verify_arm_reaches_the_seam_and_keeps_arm4s_denials(te):
    allow_v, deny_v, _ = _sets(te, "merlin_verify")
    allow_4, deny_4, _ = _sets(te, "merlin_rtlchecks")
    seam = set(TR.TOOLS["verify_seam"].bundle_paths)
    assert seam <= allow_v and not (seam & deny_v), "the seam is granted and unmasked on its own arm"
    assert allow_v - allow_4 == seam, f"the verify arm gained more than the seam: {allow_v - allow_4 - seam}"
    assert deny_4 - deny_v == seam, f"the verify arm lost a denial beyond the seam: {deny_4 - deny_v - seam}"


def test_granting_the_seam_to_arm4_as_an_ablation_cell_reproduces_the_verify_arm(te):
    """The cell and the arm must be the same object, or an ablation sweep measures a different thing
    from the campaign it is meant to explain."""
    cell, cell_deny, _ = _sets(te, "merlin_rtlchecks", add_tools=("verify_seam",))
    arm, arm_deny, _ = _sets(te, "merlin_verify")
    assert cell == arm and cell_deny == arm_deny


def test_the_verify_arm_is_opt_in_and_the_default_ladder_is_unchanged(te):
    assert "merlin_verify" not in _ARMS and "merlin_verify" in _OPT_IN_ARMS
    default = generate_bundles(te, variant="hwbringup_v0")
    assert set(default) == {f"{stem}_hwbringup_v0" for stem in _ARMS.values()}
    opt_in = generate_bundles(te, variant="hwbringup_v0", arms=("merlin_verify",))
    assert list(opt_in) == ["merlin_assisted_verify_hwbringup_v0"]
    assert opt_in["merlin_assisted_verify_hwbringup_v0"]["arm"] == "merlin_verify"


def test_the_stem_keeps_the_assisted_seam_menu():
    """``generate_prompt._is_assisted_arm`` is a substring test on the bundle stem, so an assisted arm
    whose stem drops the substring is handed the raw-baseline prompt with none of its tools named."""
    from merlin.targetgen.generate_prompt import _is_assisted_arm
    assert _is_assisted_arm(_OPT_IN_ARMS["merlin_verify"])


def test_the_stem_resolves_back_to_its_own_arm_by_longest_match():
    """The launcher maps a bundle id back to a rung by longest matching stem, and every opt-in stem nests
    inside an existing one. Pin that the FULL table disambiguates it, so graduating the arm is a table
    swap in the resolver rather than a silent downgrade to the arm whose stem it is prefixed by."""
    from merlin.targetgen.generate_bundles import _ALL_ARMS
    bundle_id = f"{_OPT_IN_ARMS['merlin_verify']}_hwbringup_v0"
    best = max((a for a, s in _ALL_ARMS.items() if bundle_id.startswith(s + "_")),
               key=lambda a: len(_ALL_ARMS[a]))
    assert best == "merlin_verify"
    ladder_only = max((a for a, s in _ARMS.items() if bundle_id.startswith(s + "_")),
                      key=lambda a: len(_ARMS[a]))
    assert ladder_only == "merlin_assisted", ("the ladder-only table resolves this stem to another arm; "
                                              "the resolver must read the full table before the arm ships")


def test_an_unknown_arm_still_fails_closed(te):
    with pytest.raises(KeyError):
        generate_bundles(te, arms=("merlin_verfy",))
