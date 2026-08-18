"""Arm 5 — the equivalence seam as a declared treatment — and the fairness wiring that makes it an arm.

A new arm ships with ZERO automated fairness checking unless its differentiator is written down where a
gate reads it. These tests assert the three properties that make arm5 a comparison rather than a second
run of arm3: it HAS the seam, no other arm has it, and it differs from the arm it is compared against in
exactly that one declared way.

They also pin the two things a fifth arm could quietly break: the assisted-prompt substring convention it
relies on, and the bundle-stem collision it introduces (`merlin_assisted_eqsat` starts with
`merlin_assisted_`, which made the PLAIN merlin arm unresolvable in the pre-spend gate until the variant
tokens were excluded).
"""
from __future__ import annotations

import sys

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.generate_bundles import generate_bundles
from merlin.targetgen.target_experiment import load_target_experiment

_HARNESS = merlin_dir() / "experiments/capsule_bench/harness"
_DESC = merlin_dir() / "experiments/capsule_bench/targets/saturn_opu/target_experiment.yaml"
_EQSAT_TOKENS = ("contraction_egraph", "persistent_equivalence")


@pytest.fixture(scope="module")
def bundles():
    return generate_bundles(load_target_experiment(_DESC))


def _allowed(b):
    return [e["path"] for e in b.get("allowed", [])]


def _denied(b):
    return [e["path"] for e in b.get("denied", [])]


def _bundle(bundles, stem):
    return bundles[f"{stem}_hwbringup_v0"]


# --------------------------------------------------------------------------- the treatment exists
def test_the_eqsat_arm_is_generated(bundles):
    assert "merlin_assisted_eqsat_hwbringup_v0" in bundles


def test_the_eqsat_arm_gets_the_seam(bundles):
    allowed = " ".join(_allowed(_bundle(bundles, "merlin_assisted_eqsat")))
    for tok in _EQSAT_TOKENS:
        assert tok in allowed, f"the treatment ({tok}) is missing from the arm under test"


# --------------------------------------------------------------------------- nobody else gets it
@pytest.mark.parametrize("stem", ["raw_baseline", "merlin_assisted", "merlin_assisted_rtlchecks",
                                  "cpp_merlininfra"])
def test_no_other_arm_reaches_the_seam(bundles, stem):
    allowed = " ".join(_allowed(_bundle(bundles, stem)))
    for tok in _EQSAT_TOKENS:
        assert tok not in allowed, f"{stem} can reach {tok}: the comparison is against an arm that had it"


# --------------------------------------------------------------------------- exactly one difference
def test_the_eqsat_arm_differs_from_the_xdsl_arm_only_by_the_seam(bundles):
    """The load-bearing property. If arm5 also gained (or lost) something else, a difference in outcome
    would not attribute to the seam."""
    eq = _bundle(bundles, "merlin_assisted_eqsat")
    xd = _bundle(bundles, "merlin_assisted")
    extra = set(_allowed(eq)) - set(_allowed(xd))
    missing = set(_allowed(xd)) - set(_allowed(eq))
    assert not missing, f"arm5 LOST grants the xDSL arm has: {sorted(missing)}"
    assert all(any(t in p for t in _EQSAT_TOKENS) for p in extra), \
        f"arm5 gained something that is not the seam: {sorted(extra)}"
    assert set(_denied(eq)) == set(_denied(xd)), "the two arms must deny the same things"


def test_the_stem_keeps_the_assisted_prompt_convention():
    """`generate_prompt._is_assisted_arm` is a substring test, which is why the stem is
    `merlin_assisted_eqsat` — the arm inherits the assisted seam menu with no prompt edit."""
    from merlin.targetgen import generate_prompt as GP
    assert GP._is_assisted_arm("merlin_assisted_eqsat")
    assert not GP._is_assisted_arm("raw_baseline")


# --------------------------------------------------------------------------- the fairness gate knows
def test_the_moat_check_names_the_seam():
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import verify_no_cheat as V
    assert V.EQSAT_ARM == "merlin_assisted_eqsat"
    assert set(V.EQSAT_PATHS) == set(_EQSAT_TOKENS)
    assert V.EQSAT_ARM in V.MERLIN_ARMS, "arm5 must be in the merlin-arm set the parity checks walk"


def test_adding_the_variant_did_not_make_the_plain_merlin_arm_ambiguous(monkeypatch, tmp_path):
    """The collision this arm introduces: three bundle dirs start with `merlin_assisted_`. The gate must
    still resolve the plain arm rather than refusing because it found candidates."""
    if str(_HARNESS) not in sys.path:
        sys.path.insert(0, str(_HARNESS))
    import preflight as P

    for stem in ("merlin_assisted_hwbringup_v0", "merlin_assisted_rtlchecks_hwbringup_v0",
                 "merlin_assisted_eqsat_hwbringup_v0"):
        d = tmp_path / stem
        d.mkdir()
        (d / "input_bundle_manifest.yaml").write_text("bundle_id: x\n")
    monkeypatch.setattr(P.C, "BUNDLES", tmp_path)
    monkeypatch.setitem(P.RAE.ARM_BUNDLE, "merlin_assisted", "merlin_assisted_public_v0")  # absent here
    assert P._bundle_id_for("merlin_assisted") == "merlin_assisted_hwbringup_v0"


def test_the_hypotheses_stay_unestablished():
    """An arm5 win is evidence about agent productivity given the seam, never about eqsat being a better
    compiler strategy. The two hypotheses that would claim the latter must stay open."""
    from merlin.targetgen import persistent_equivalence as PE
    assert PE.HYPOTHESES["H-EQ1"]["status"] == "not_established"
    assert PE.HYPOTHESES["H-EQ2"]["status"] == "not_established"
