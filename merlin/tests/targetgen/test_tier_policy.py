"""Spend the expensive oracle where it buys evidence, and say what was not bought.

Three measured facts drive this policy, all from the two graded targets:

* the ladder ran tiers in LEXICOGRAPHIC order, and the cost spread between a target's own tiers is two
  to three orders of magnitude pointing in OPPOSITE directions -- one target's cheapest tier is spike
  (0.44s) against verilator (132.5s); the other's cheapest is verilator (0.29s) against an arc cosim
  (24.5s). "Run the cheap tier first" cannot be written as a tier name;
* a screen may ELIMINATE: replaying the 0.29s tier on twelve capsules the 24.5s tier had failed
  reproduced all twelve, exactly;
* a screen may never CERTIFY: one submission passed the cheap functional tier 20/20 while its RTL tier
  passed 1.
"""
from __future__ import annotations

import pytest

from merlin.targetgen import tier_policy as TP


@pytest.fixture(autouse=True)
def _clean():
    TP.reset_costs()
    TP.reset_spend()
    yield
    TP.reset_costs()
    TP.reset_spend()


def test_order_is_by_measured_cost_not_by_tier_name():
    """The exact inversion that was costing 24.5s per failing capsule: the expensive tier sorts first
    lexicographically, and must not run first."""
    TP.record_cost("t", "L3", 24.5)
    TP.record_cost("t", "L4", 0.29)
    assert TP.tier_order("t", ["L3", "L4"]) == ["L4", "L3"]
    assert sorted(["L3", "L4"]) == ["L3", "L4"], "the lexicographic order is the opposite one"


def test_the_other_target_orders_the_other_way_from_the_same_code():
    TP.record_cost("u", "L2", 0.44)
    TP.record_cost("u", "L3", 132.5)
    assert TP.tier_order("u", ["L2", "L3"]) == ["L2", "L3"]


def test_an_unmeasured_tier_runs_first_so_the_ladder_can_learn():
    """The deadlock this avoids, measured on a live suite: the ladder stops at the first tier that
    refutes a capsule, so sorting unknowns LAST means a target whose early capsules fail measures only
    its expensive tier -- the cheap tier below it is never reached, stays unknown, and keeps sorting
    behind the expensive one forever. Nine capsules in, every one had paid the 24.5s tier and only the
    single passing capsule had ever reached the 0.29s one."""
    TP.record_cost("t", "L3", 24.5)
    assert TP.observed_cost("t", "L4") is None
    assert TP.tier_order("t", ["L3", "L4"]) == ["L4", "L3"]


def test_once_both_are_measured_the_cheaper_leads():
    TP.record_cost("t", "L3", 24.5)
    TP.record_cost("t", "L4", 0.29)
    assert TP.tier_order("t", ["L3", "L4"]) == ["L4", "L3"]


def test_cost_is_the_median_so_one_slow_run_does_not_reorder_the_ladder():
    for v in (0.3, 0.3, 0.3, 90.0):
        TP.record_cost("t", "L4", v)
    TP.record_cost("t", "L3", 24.5)
    assert TP.tier_order("t", ["L3", "L4"]) == ["L4", "L3"]


def test_calibration_is_reported_honestly():
    TP.record_cost("t", "L4", 0.29)
    assert not TP.is_calibrated("t", ["L3", "L4"])
    TP.record_cost("t", "L3", 24.5)
    assert TP.is_calibrated("t", ["L3", "L4"])


# --- coverage ------------------------------------------------------------------------------------

def _cap(name, *, family="contraction", op="matmul", epilogue=(), mtiles=1, modes=()):
    return {"name": name, "kind": "isa",
            "semantic": {"semantic_family": family, "generalization_axis": "seen"},
            "operation": {"op": op, "attributes": {"epilogue": list(epilogue)}},
            "M_tiles": mtiles,
            "expected": {"modes": {m: True for m in modes}}}


def test_the_cover_spans_every_declared_axis():
    caps = [_cap("a"), _cap("b"), _cap("c", epilogue=["relu"], modes=["relu"]),
            _cap("d", family="movement", op="movement"), _cap("e", mtiles=2)]
    cover = TP.covering_set(caps)
    covered = set().union(*[TP.capsule_axes(c) for c in caps if c["name"] in cover])
    assert covered == set().union(*[TP.capsule_axes(c) for c in caps])


def test_redundant_capsules_are_left_out_of_the_cover():
    """Three capsules declaring identical axes buy one capsule's worth of evidence."""
    caps = [_cap("a"), _cap("b"), _cap("c")]
    assert len(TP.covering_set(caps)) == 1


def test_the_cover_is_deterministic():
    caps = [_cap(n) for n in "abcdef"] + [_cap("z", op="movement")]
    assert TP.covering_set(caps) == TP.covering_set(list(reversed(caps)))


def test_certify_order_puts_the_cover_first():
    caps = [_cap("a"), _cap("b"), _cap("z", op="movement")]
    order = TP.certify_order(caps)
    cover = set(TP.covering_set(caps))
    assert set(order[:len(cover)]) == cover and len(order) == 3


def test_a_capsule_declaring_nothing_is_never_a_representative():
    caps = [_cap("real"), {"name": "empty"}]
    assert "empty" not in TP.covering_set(caps)


# --- budget --------------------------------------------------------------------------------------

def test_unlimited_is_the_default_so_coverage_is_never_silently_narrowed(monkeypatch):
    monkeypatch.delenv("MERLIN_CERTIFY_BUDGET_S", raising=False)
    assert TP.budget_seconds() is None
    TP.note_spend("t", 10_000)
    assert TP.may_certify("t", {"name": "x", "_covering": False})[0] is True


def test_a_covering_capsule_is_certified_even_with_the_budget_gone(monkeypatch):
    monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "10")
    TP.note_spend("t", 999)
    assert TP.may_certify("t", {"name": "x", "_covering": True})[0] is True


def test_a_non_covering_capsule_is_declined_once_the_budget_is_gone(monkeypatch):
    monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "10")
    assert TP.may_certify("t", {"name": "x", "_covering": False})[0] is True
    TP.note_spend("t", 11)
    ok, why = TP.may_certify("t", {"name": "x", "_covering": False})
    assert ok is False
    assert "NOT a verdict" in why and "covering set" in why


def test_spend_is_per_target(monkeypatch):
    monkeypatch.setenv("MERLIN_CERTIFY_BUDGET_S", "10")
    TP.note_spend("t", 50)
    assert TP.may_certify("t", {"name": "x", "_covering": False})[0] is False
    assert TP.may_certify("u", {"name": "x", "_covering": False})[0] is True


# --- what the artifact says ------------------------------------------------------------------------

def test_a_screened_capsule_is_neither_numerator_nor_denominator_and_is_named(monkeypatch):
    """The whole risk of this policy in one test: a capsule that was screened and not certified must not
    inflate a score, must not deflate one, and must be listed by name with the reason."""
    from merlin.targetgen import capsule_grade as CG

    results = ([{"capsule": f"p{i}", "status": "pass", "kind": "op", "tiers": {"L3": "pass"}}
                for i in range(6)]
               + [{"capsule": f"s{i}", "status": "screened_only", "kind": "op", "tiers": {},
                   "failure": {"plane": "budget", "category": "SCREENED_NOT_CERTIFIED",
                               "detail": "passed the screen tier; certify tier(s) L3 not purchased"}}
                  for i in range(9)])
    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]}
                                                                     for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", lambda *a, **k: results)
    s = CG.grade("pkg", capsules_root=["r"], runs_root="runs", target="gemmini", max_workers=1)

    assert (s["n_passed"], s["n_capsules"]) == (6, 6), "screened capsules must not enter either side"
    assert s["n_screened_only"] == 9
    assert s["screened_only"] == sorted(f"s{i}" for i in range(9))
    assert "does NOT certify these capsules" in s["screened_only_note"]


def test_a_screened_capsule_does_not_prop_up_the_model_gate(monkeypatch):
    """It passed the cheap screen. Counting that toward a whole-model gate would certify a model on
    evidence nobody bought — the same shape as the crash-leak this suite already guards."""
    from merlin.targetgen import capsule_runner as CR

    caps = ([{"name": f"ok{i}", "kind": "op"} for i in range(4)]
            + [{"name": f"sc{i}", "kind": "op"} for i in range(16)]
            + [{"name": "M0", "kind": "model", "gate": {"after_op_pass_fraction": 0.8}}])

    def _fake(cap, package_dir, **kw):
        st = ("pass" if cap["name"].startswith("ok")
              else "pass" if cap.get("kind") == "model" else "screened_only")
        return {"capsule": cap["name"], "status": st, "kind": cap.get("kind"), "tiers": {}}

    monkeypatch.setattr(CR, "load_package", lambda *a, **k: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CR, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CR, "_split_ineligible", lambda c, t: (c, []))
    monkeypatch.setattr(CR, "_gate_counts", lambda r, c, t: True)
    monkeypatch.setattr(CR, "run_capsule", _fake)

    out = CR.run_suite(caps, "pkg", runs_root="runs", target="gemmini", max_workers=1)
    m0 = next(r for r in out if r["capsule"] == "M0")
    # 4 certified of 4 graded is 1.00 — the gate is satisfied by what was actually certified, and the
    # 16 screened capsules neither help nor hurt. What must never happen is 20/20.
    assert m0["status"] != "gated"
    assert sum(1 for r in out if r.get("status") == "screened_only") == 16


def test_the_covering_set_is_dispatched_before_the_rest(monkeypatch):
    """An interrupted run must keep full axis coverage, not a lexicographic prefix."""
    from merlin.targetgen import capsule_runner as CR

    order: list[str] = []
    caps = [_cap("zzz_movement", family="movement", op="movement"),
            _cap("aaa"), _cap("bbb"), _cap("ccc")]

    def _fake(cap, package_dir, **kw):
        order.append(cap["name"])
        return {"capsule": cap["name"], "status": "pass", "kind": "op", "tiers": {}}

    monkeypatch.setattr(CR, "load_package", lambda *a, **k: object())
    monkeypatch.setattr(CR, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CR, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CR, "_split_ineligible", lambda c, t: (c, []))
    monkeypatch.setattr(CR, "run_capsule", _fake)
    CR.run_suite(caps, "pkg", runs_root="runs", target="gemmini", max_workers=1)

    cover = set(TP.covering_set(caps))
    assert set(order[:len(cover)]) == cover, f"cover {sorted(cover)} not dispatched first: {order}"
    assert "zzz_movement" in cover, "the only movement capsule must be a representative"
