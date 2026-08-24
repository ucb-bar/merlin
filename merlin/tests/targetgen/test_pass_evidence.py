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


def _grade_model(*, on_mesh, fallback, tiles):
    """Grade a whole-model capsule against a synthetic compile_model result: `mesh_execution` is what
    happened to THIS MODEL's layers, `mesh_tile_verification` is a synthesized tile at each routed shape.
    """
    from merlin import compile_cli as CCLI
    from merlin.targetgen import capsule_runner as CR

    capsule = {"name": "M_probe", "kind": "model",
               "operation": {"op": "model", "attributes": {"model": "probe", "compile_dtype": "int8",
                                                           "dtype": "i8"}},
               "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
               "semantic": {"semantic_family": "contraction", "must_accelerate": True}}
    out = {"status": "verified", "verify": {"gate_ok": True},
           "mesh_tile_verification": tiles,
           "mesh_execution": {"target": "gemmini", "matmul_layers_routed": on_mesh + fallback,
                              "matmul_layers_on_mesh": on_mesh,
                              "matmul_layers_host_fallback": fallback}}
    real = CCLI.compile_model
    CCLI.compile_model = lambda *a, **k: out
    try:
        return CR._grade_model_capsule(capsule, target="gemmini", timeout=1)
    finally:
        CCLI.compile_model = real


def test_the_model_not_its_tiles_decides_the_model_capsules_tier():
    """A run with every layer on the host once reported '15 of 15 tiles passed'. The tile record proves
    the SHAPE runs; the capstone is a claim about THIS model. Asserted on the GRADE, not on the source
    text -- a behavioural claim that survives the code being rewritten under it."""
    _all_tiles_pass = {"n_tiles": 15, "n_passed": 15, "n_failed": 0,
                       "n_unavailable": 0, "n_unsynthesizable": 0}

    def _passed(r):
        return {t: v for t, v in ((k, (o or {}).get("status"))
                                  for k, o in (r.get("tiers") or {}).items()) if v == "pass"}

    on_host = _grade_model(on_mesh=0, fallback=15, tiles=_all_tiles_pass)
    assert _passed(on_host) == {}, "every layer ran on the host; certified tiles cannot pass the model"
    assert on_host["status"] != "pass"

    partial = _grade_model(on_mesh=14, fallback=1, tiles=_all_tiles_pass)
    assert _passed(partial) == {}, "a fallen-back layer must not pass the tier"
    assert partial["status"] != "pass"

    clean = _grade_model(on_mesh=15, fallback=0, tiles=_all_tiles_pass)
    assert list(_passed(clean)) == ["L3"], clean["tiers"]
    assert clean["status"] == "pass"
    # the tile record is reported BESIDE the verdict, as the separate and weaker evidence it is
    assert clean["tile_evidence"]["n_tiles"] == 15
    assert "shapes run, not that this model ran" in clean["tile_evidence"]["note"]


# --- the two tier RECORD SHAPES that coexist in one results list ------------------------------------

def test_both_tier_record_shapes_normalize():
    """An op capsule records a tier as a dict; a model capsule records it as a bare string. Every
    aggregation assumed the dict, which only ever crashed on a submission good enough to un-gate its
    model capsules -- after all 36 capsules had been simulated, so the run cost its full wall-clock and
    wrote no score at all."""
    from merlin.targetgen.capsule_common import tier_field, tier_status
    assert tier_status({"status": "pass", "cycles": 318}) == "pass"     # op capsule
    assert tier_status("pass") == "pass"                                # model capsule
    assert tier_status(None) is None and tier_status(123) is None
    assert tier_field({"status": "pass", "cycles": 318}, "cycles") == 318
    assert tier_field("pass", "cycles") is None, "the string form carries no fields, and must not raise"


def test_the_aggregators_do_not_reimplement_the_shape_check():
    """Both aggregators read the same results list, so both had the same bug. One normalizer."""
    import inspect

    from merlin.targetgen import capsule_grade, coverage_report
    for mod in (capsule_grade, coverage_report):
        src = inspect.getsource(mod)
        assert "tier_status" in src, f"{mod.__name__} must use the shared normalizer"
        assert '.get(t, {}).get("status")' not in src, \
            f"{mod.__name__} still assumes a tier is a dict"


def test_a_mixed_results_list_aggregates_without_raising():
    """The exact shape that crashed: op capsules with dict tiers beside a model capsule with strings."""
    from merlin.targetgen import coverage_report as CV
    results = [
        {"capsule": "A0", "kind": "isa", "label": "public", "status": "pass",
         "tiers": {"L2": {"status": "pass"}, "L3": {"status": "pass", "derived_from_rtl": True}}},
        {"capsule": "M0", "kind": "model", "label": "public", "status": "pass",
         "tiers": {"L3": "pass"}},                                      # <- the bare-string form
    ]
    cov = CV.aggregate(results, capsules=[], traces={}, target="gemmini")
    assert cov["by_tier_reached"]["L3"] == 2, "both shapes must be counted"
