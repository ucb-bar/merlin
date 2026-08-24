"""A pass is not a pass: the score must say which KIND of tier carried it.

Two submissions both reported "20/20". One cleared the RTL tier on all 20; the other cleared it on 1,
because that tier was advisory when it ran. Nothing in the headline distinguished them, and the
flattering reading is the one that got quoted for days. These tests pin the distinction into the score.
"""
from __future__ import annotations


def _cap(name, status, tiers):
    return {"capsule": name, "label": "public", "status": status, "tiers": tiers}


def _rtl(status):
    return {"status": status, "derived_from_rtl": True}


def _cheap(status):
    return {"status": status, "derived_from_rtl": False}


def _evidence(graded):
    """The derivation under test, mirrored from capsule_grade.grade so the tests can drive it directly
    without standing up a package + oracle."""
    passed = [r for r in graded if r.get("status") == "pass"]
    rtl_backed = [r for r in passed
                  if any(isinstance(t, dict) and t.get("status") == "pass" and t.get("derived_from_rtl")
                         for t in (r.get("tiers") or {}).values())]
    return {"n_passed": len(passed), "rtl_backed": len(rtl_backed),
            "cheap_tier_only": len(passed) - len(rtl_backed)}


def test_the_derivation_matches_the_shipped_one():
    """Guard against the mirror above drifting from capsule_grade's real implementation."""
    import inspect

    from merlin.targetgen import capsule_grade
    src = inspect.getsource(capsule_grade.grade)
    assert "pass_evidence" in src, "the score must carry a pass_evidence block"
    assert "derived_from_rtl" in src, \
        "RTL-ness must be DERIVED from the tier record, never matched against a tier-name literal"
    for name in ("rtl_backed", "cheap_tier_only", "n_passed"):
        assert name in src, f"pass_evidence must report {name}"


def test_a_cheap_tier_only_suite_is_not_reported_as_rtl_backed():
    """The codex3 shape: 20 passes, RTL tier passing on exactly one of them."""
    graded = [_cap("A1_movement", "pass", {"L2": _cheap("pass"), "L3": _rtl("pass")})]
    graded += [_cap(f"C{i}", "pass", {"L2": _cheap("pass"), "L3": _rtl("fail")}) for i in range(19)]
    ev = _evidence(graded)
    assert ev["n_passed"] == 20
    assert ev["rtl_backed"] == 1, "only the capsule whose RTL tier passed is RTL-backed"
    assert ev["cheap_tier_only"] == 19, "the other 19 passed on cheap tiers and must say so"


def test_an_rtl_clean_suite_reports_every_pass_as_rtl_backed():
    """The codex/codex2/rb_gemrecreate1 shape: same 20/20 headline, entirely different evidence."""
    graded = [_cap(f"C{i}", "pass", {"L2": _cheap("pass"), "L3": _rtl("pass")}) for i in range(20)]
    ev = _evidence(graded)
    assert (ev["n_passed"], ev["rtl_backed"], ev["cheap_tier_only"]) == (20, 20, 0)


def test_rtl_ness_is_not_tied_to_a_tier_NAME():
    """A target whose RTL tier is L4 (atlas) must be described correctly by the same code that handles a
    target whose RTL tier is L3 (gemmini). Keying on the tier name is how this goes wrong quietly."""
    graded = [_cap(f"A{i}", "pass", {"L3": _cheap("pass"), "L4": _rtl("pass")}) for i in range(14)]
    ev = _evidence(graded)
    assert ev["rtl_backed"] == 14 and ev["cheap_tier_only"] == 0


def test_a_failing_capsule_never_counts_as_evidence():
    graded = [_cap("F0", "fail", {"L3": _rtl("fail")}), _cap("P0", "pass", {"L3": _rtl("pass")})]
    ev = _evidence(graded)
    assert (ev["n_passed"], ev["rtl_backed"]) == (1, 1)


# --- the model capsule's own tier attribution -------------------------------------------------------

def test_the_rtl_tier_is_derived_per_target_not_named():
    """`[x for x in declared if x not in ("L0","L1")]` with an `"L3"` fallback is three tier-name
    literals standing in for a fact the capability manifest already carries. It names the wrong tier
    confidently on any target whose ladder differs."""
    from merlin.targetgen.capsule_runner import _rtl_tiers_of
    seen = {t: _rtl_tiers_of(t) for t in ("gemmini", "atlas", "radiance")}
    assert all(seen.values()), f"every target must declare its RTL tiers: {seen}"
    assert len({frozenset(v) for v in seen.values()}) > 1, \
        f"the ladders differ between targets, so a single literal cannot be right for all: {seen}"
    assert _rtl_tiers_of(None) == frozenset(), "no target -> fail soft, never a guessed tier"
    assert _rtl_tiers_of("definitely_not_a_target") == frozenset()


def test_the_model_not_its_tiles_decides_the_model_capsules_tier():
    """A run with every layer on the host once reported '15 of 15 tiles passed'. The tile record proves
    the SHAPE runs; the capstone is a claim about THIS model."""
    import inspect

    from merlin.targetgen import capsule_runner
    src = inspect.getsource(capsule_runner._grade_model_capsule)
    assert "_model_ok" in src, "the model's own per-layer accounting must gate the tier"
    assert "matmul_layers_host_fallback" in src, "a fallen-back layer must not pass the tier"
    assert "tile_evidence" in src, "the tile record must be reported as separate, weaker evidence"
    assert '"L3"' not in src.split("_rtl_tiers_of")[-1], \
        "no tier-name literal may decide which tier the mesh oracle corresponds to"
