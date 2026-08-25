"""A tier NAME is not evidence about what ran.

``rtl_backed`` was computed by asking whether a passing tier's name was in the target's declared
``rtl_tiers``. That works on a target whose L3 IS the elaborated Verilog and silently lies on one whose
L3 is a model: the ``external_backend`` ladder puts an *RTL-derived* arc cosim at L3 (its own source
calls it "the RTL-derived functional gold") and the elaborated-Verilog run at L4. Both counted as RTL
certification, so a suite could report every pass as RTL-backed while no elaborated RTL ever ran.

The fix is that the ORACLE declares what it was and the oracle's word outranks the tier name --
the seam ``capsule_runner`` already read (``oracle.get("derived_from_rtl", <tier-name default>)``) and
that muon's gsim adapter already used. This pins the three-state rule, because the trap is the middle
state: an explicit ``False`` must beat the name, while an ABSENT flag must still fall back to it (a
model capsule records its tier as a bare string and would otherwise lose its RTL credit).
"""
from __future__ import annotations

from merlin.targetgen import capsule_grade as CG
from merlin.targetgen.capsule_common import oracle_kind
from merlin.targetgen.capsule_runner import TierResult


def _score(monkeypatch, results, target="atlas"):
    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG, "source_experiment_env", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]}
                                                                     for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", lambda *a, **k: results)
    # every target in this test declares L3/L4/L5 as its RTL tiers BY NAME -- that is exactly the
    # classification the oracle's own word has to be able to override.
    monkeypatch.setattr(CG.CR, "_rtl_tiers_of", lambda *a, **k: frozenset({"L3", "L4", "L5"}))
    return CG.grade("pkg", capsules_root=["root"], runs_root="runs", target=target, max_workers=1)


def _op(name, tiers, status="pass"):
    return {"capsule": name, "kind": "op", "label": "public", "status": status,
            "tiers": tiers, "numeric": {"status": status}, "trace_check": {"status": status}}


_MODEL_L3 = {"status": "pass", "derived_from_rtl": False, "fidelity": "rtl_derived_model"}
_RTL_L4 = {"status": "pass", "derived_from_rtl": True, "fidelity": "elaborated_rtl"}


def test_an_explicit_denial_beats_the_tier_name(monkeypatch):
    """L3 is a declared RTL tier by NAME; the oracle says it was a model. The oracle wins."""
    s = _score(monkeypatch, [_op("A0", {"L3": _MODEL_L3})])
    ev = s["pass_evidence"]
    assert ev["n_passed"] == 1
    assert ev["rtl_backed"] == 0, "an RTL-derived MODEL is not RTL certification"
    assert ev["model_certified"] == 1


def test_an_absent_flag_still_falls_back_to_the_tier_name(monkeypatch):
    """The bare-string form states nothing, so the target's declared rtl_tiers still decide.

    This is the case the fallback exists for -- a model capsule records ``{"L3": "pass"}`` -- and
    removing it would strip RTL credit from every whole-model pass.
    """
    s = _score(monkeypatch, [_op("M1", {"L3": "pass"})])
    assert s["pass_evidence"]["rtl_backed"] == 1


def test_the_elaborated_rtl_tier_carries_the_pass(monkeypatch):
    """atlas's real shape: a model at L3 and the elaborated Verilog at L4. It IS RTL-backed, by L4."""
    s = _score(monkeypatch, [_op("A0", {"L3": _MODEL_L3, "L4": _RTL_L4})])
    ev = s["pass_evidence"]
    assert ev["rtl_backed"] == 1
    assert ev["model_certified"] == 0, "credited to the tier that earned it, not counted twice"
    assert ev["rtl_tiers_seen"] == ["L4"], "L3 denied being RTL, so it is not an RTL tier here"
    assert set(ev["fidelity_seen"]) == {"rtl_derived_model", "elaborated_rtl"}


def test_the_headline_says_a_model_oracle_carried_the_rest(monkeypatch):
    """'passed on cheap tiers only' is wrong for an RTL-derived cosim, and the headline is what is quoted."""
    s = _score(monkeypatch, [_op("A0", {"L3": _MODEL_L3, "L4": _RTL_L4}), _op("A1", {"L3": _MODEL_L3})])
    head = s["headline"]
    assert "RTL-backed 1/2" in head
    assert "model oracle only" in head, head
    assert "cheap tiers only" not in head, head


def test_a_failing_tier_is_never_rtl_evidence(monkeypatch):
    s = _score(monkeypatch, [_op("A0", {"L4": {"status": "fail", "derived_from_rtl": True}}, status="fail")])
    assert s["pass_evidence"]["rtl_backed"] == 0


def test_tier_records_carry_the_oracles_fidelity(monkeypatch):
    """The word has to survive into the per-capsule record, or the grade cannot read it back."""
    d = TierResult("L3", "pass", True, fidelity="rtl_derived_model").to_dict()
    assert d["fidelity"] == "rtl_derived_model"
    assert "fidelity" not in TierResult("L3", "pass", True).to_dict(), "absent stays absent"


def test_oracle_kind_accepts_both_shapes():
    """Enriching an adapter must not turn a recorded provenance STRING into a dict."""
    assert oracle_kind("atlas-verilator-rtl") == "atlas-verilator-rtl"
    assert oracle_kind({"kind": "atlas-verilator-rtl", "derived_from_rtl": True}) == "atlas-verilator-rtl"
    assert oracle_kind(None) is None
