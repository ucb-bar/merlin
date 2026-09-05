"""A whole model must EARN the tier below its cert tier, not be graded on the cert tier alone.

Both gemmini model capsules declare ``required_oracle_tiers: [L0, L1, L2, L3]``, and the tier map used
to emit only ``{L0, L1, L3}`` -- the screen rung was declared and then silently dropped. That is the
`not_run_is_not_pass` hole in its most invisible form: an ABSENT key does not read as a hole, it reads
as nothing at all, and a downstream gate that asks ``tiers["L2"] == "pass"`` gets ``None`` from a
capsule that never ran the tier.

These tests pin the three properties that close it: the screen tier is DERIVED from the target's own
contract (not a "spike" literal), it appears on every path including the ones that refuse, and a tile
that fails it never reaches the cert oracle.
"""
from __future__ import annotations

import pytest

from merlin.targetgen.capsule_runner import (_model_tier_map, _rtl_tiers_of, _screen_tiers_of)

DECLARED = ["L0", "L1", "L2", "L3"]
ON_MESH = {"matmul_layers_on_mesh": 12, "matmul_layers_host_fallback": 0}


def _screened(passed: int, failed: int = 0, unavailable: int = 0) -> dict:
    n = passed + failed + unavailable
    return {"n_screened": n, "n_screen_passed": passed, "n_screen_failed": failed,
            "n_screen_unavailable": unavailable}


def test_screen_tier_is_derived_from_the_contract_not_a_literal():
    """The screen is whatever the target's ``tier_sim`` declares below its RTL tiers."""
    screens = _screen_tiers_of("gemmini")
    assert screens, "gemmini declares an oracle tier below L3; the deriver found none"
    rtl = _rtl_tiers_of("gemmini")
    for tier, sim in screens:
        assert tier not in rtl, f"{tier} is an RTL tier and cannot be its own screen"
        assert sim, f"{tier} has no simulator declared"


def test_unresolvable_target_yields_no_screen_rather_than_a_guess():
    assert _screen_tiers_of(None) == ()
    assert _screen_tiers_of("no_such_target_xyz") == ()


def test_a_declared_screen_tier_is_never_dropped_from_the_tier_map():
    """The regression: L2 is declared, so it must appear -- with a verdict, whatever that verdict is."""
    for tile_exec in (_screened(12), _screened(11, failed=1), _screened(0, unavailable=12),
                      {"n_screened": 0, "n_screen_passed": 0}, {}, None):
        tiers = _model_tier_map(DECLARED, "gemmini", ON_MESH, tile_exec)
        assert "L2" in tiers, f"L2 was declared but dropped for tile_exec={tile_exec!r}"
        assert tiers["L2"].status in ("pass", "fail", "skipped", "unavailable")
        assert tiers["L2"].reason, "a tier verdict with no reason cannot be audited"


@pytest.mark.parametrize("tile_exec,expected", [
    (_screened(12), "pass"),
    (_screened(11, failed=1), "fail"),
    (_screened(0, unavailable=12), "unavailable"),
    ({"n_screened": 0, "n_screen_passed": 0, "n_screen_failed": 0,
      "n_screen_unavailable": 0}, "skipped"),
    # No tile record AT ALL: this grade synthesized nothing, so there was nothing to screen. A
    # legitimate skip -- distinct from tiles that were verified without being screened, below.
    ({}, "skipped"),
    (None, "skipped"),
    # Tiles WERE verified and carry no screen tally: the cheapest-first order really was skipped.
    ({"per_tile": [{"oracle_result": "pass"}, {"oracle_result": "pass"}]}, "unavailable"),
])
def test_the_four_screen_cases_stay_distinct(tile_exec, expected):
    """`pass`, `fail`, `skipped` and `unavailable` are four different facts about the same rung."""
    assert _model_tier_map(DECLARED, "gemmini", ON_MESH, tile_exec)["L2"].status == expected


def test_an_unrecorded_screen_is_unavailable_never_a_pass():
    """Absent must never read as satisfied -- the whole reason the dropped key was dangerous."""
    tiers = _model_tier_map(DECLARED, "gemmini", ON_MESH,
                            {"n_screened": None, "per_tile": [{"oracle_result": "pass"}]})
    assert tiers["L2"].status == "unavailable"
    assert "UNKNOWN" in (tiers["L2"].reason or "")


def test_no_tiles_is_a_skip_and_verified_tiles_without_a_screen_is_a_hole():
    """The two absences the first version collapsed, and why the distinction matters.

    Collapsing them made every model grade that does no tile verification read as INCOMPLETE -- a
    mandatory tier with no record -- which is a false alarm about grades that had nothing to screen.
    Keeping them apart is what makes the `unavailable` verdict mean something when it does fire.
    """
    no_tiles = _model_tier_map(DECLARED, "gemmini", ON_MESH, None)["L2"]
    assert no_tiles.status == "skipped" and no_tiles.not_applicable is True, \
        "a grade with no tiles must not be reported as an incomplete measurement"
    unscreened = _model_tier_map(DECLARED, "gemmini", ON_MESH,
                                 {"per_tile": [{"oracle_result": "pass"}]})["L2"]
    assert unscreened.status == "unavailable" and unscreened.not_applicable is False, \
        "a tile verified without a screen is a real hole and must stay one"


def test_a_partial_screen_tally_does_not_pass():
    """n_screened=12 with 10 passed and nothing else recorded leaves 2 unaccounted for."""
    tiers = _model_tier_map(DECLARED, "gemmini", ON_MESH,
                            {"n_screened": 12, "n_screen_passed": 10})
    assert tiers["L2"].status == "unavailable"
    assert "unaccounted" in (tiers["L2"].reason or "")


def test_the_screen_rung_cites_its_own_evidence_not_the_cert_rung_s():
    tiers = _model_tier_map(DECLARED, "gemmini", ON_MESH, _screened(12))
    assert tiers["L2"].evidence == "mesh_tile_verification.per_tile[].screen"
    assert tiers["L2"].derived_from_rtl is False, "a functional screen is not RTL-derived"
    assert tiers["L2"].cycle_accurate is False, "a functional screen is not cycle-accurate"


def test_the_model_grade_refuses_a_cert_with_no_screen_record():
    """`capsule_grade` must treat a missing screen tally as a violation, not an omission."""
    import inspect

    from merlin.targetgen import capsule_grade
    src = inspect.getsource(capsule_grade)
    for token in ("n_screened", "not_all_model_tiles_screened", "model_tile_screen_failed",
                  "model_tile_screen_oracle_unavailable"):
        assert token in src, f"the model grade does not check {token}"


def test_a_tile_that_fails_the_screen_does_not_reach_the_cert_oracle():
    """The gate, read off the source: the cert `certify` call is guarded by the screen verdict."""
    import inspect

    from merlin import compile_cli
    src = inspect.getsource(compile_cli._mesh_verify)
    assert 'rec["screen"]["status"] != "pass"' in src, "the screen does not gate the cert rung"
    gate = src.index('rec["screen"]["status"] != "pass"')
    assert "continue" in src[gate:gate + 900], "the screen gate does not skip the cert oracle"
    assert "_screen_tiers_of" in src, "the screen simulator is not derived from the contract"
    # Only CODE lines: the prose above the gate legitimately names the simulator it used to hardcode.
    code = "\n".join(line.split("#", 1)[0] for line in src.splitlines())
    assert "spike" not in code, "the screen names a simulator literal instead of reading the contract"
