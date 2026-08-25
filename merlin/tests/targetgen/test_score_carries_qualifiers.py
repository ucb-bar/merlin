"""A verdict's qualifiers have to travel with the verdict.

The rolled-up ``score_capsule.json`` -- the artifact that actually gets cited -- kept six fields per
capsule: capsule, label, status, numeric, trace, tiers. Every one of them flatters. What said what the
pass RESTS ON stayed behind in the per-capsule result several directories down: which contract
obligations the RUNTIME discharged on the backend's behalf, how many of the model's own layers reached
the accelerator versus fell back to the host, and which declared tiers never ran. Measured: a model
capsule whose result recorded two layers the backend could not fit -- split host-side so the model
could run at all -- rolled up as a bare ``pass``.
"""
from __future__ import annotations

from merlin.targetgen import capsule_grade as CG


_OBLIGATION = {"capacity_fit": {"discharged_by": "merlin runtime (host-side residency tiling)",
                                "n_layers": 2, "layers": [{"kernel": "k0"}, {"kernel": "k1"}]}}


def _score(monkeypatch, results):
    monkeypatch.setattr(CG, "load_package", lambda *a, **k: type("P", (), {"integrity_exempt": False})())
    monkeypatch.setattr(CG, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(CG, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(CG.CR, "discover_capsules", lambda *a, **k: [{"name": r["capsule"]}
                                                                     for r in results])
    monkeypatch.setattr(CG.CR, "run_suite", lambda *a, **k: results)
    return CG.grade("pkg", capsules_root=["root"], runs_root="runs", target="gemmini", max_workers=1)


def _model_result(**extra):
    return {"capsule": "M1", "kind": "model", "label": "public", "status": "pass",
            "tiers": {"L3": "pass"}, "numeric": {"status": "pass"},
            "trace_check": {"status": "pass"}, **extra}


def test_a_runtime_discharged_obligation_reaches_the_score_file(monkeypatch):
    s = _score(monkeypatch, [_model_result(contract_obligations=_OBLIGATION)])
    e = s["per_capsule"][0]
    assert e["status"] == "pass"
    cf = e["contract_obligations"]["capacity_fit"]
    assert cf["n_layers"] == 2
    assert cf["discharged_by"] == "merlin runtime (host-side residency tiling)"


def test_layers_on_mesh_versus_host_fallback_reaches_the_score_file(monkeypatch):
    s = _score(monkeypatch, [_model_result(
        mesh_execution={"matmul_layers_on_mesh": 35, "matmul_layers_host_fallback": 2,
                        "status": "partial", "per_layer": ["dropped-detail"] * 40})])
    e = s["per_capsule"][0]["mesh_execution"]
    assert (e["matmul_layers_on_mesh"], e["matmul_layers_host_fallback"]) == (35, 2)
    assert "per_layer" not in e, "carry the summary, not the whole per-layer dump"


def test_unexercised_tiers_reach_the_score_file(monkeypatch):
    s = _score(monkeypatch, [_model_result(tiers_unexercised={"L0": "no whole-model analogue"})])
    assert s["per_capsule"][0]["tiers_unexercised"] == {"L0": "no whole-model analogue"}


def test_an_op_capsule_is_not_padded_with_model_only_fields(monkeypatch):
    s = _score(monkeypatch, [{"capsule": "A0", "kind": "op", "label": "public", "status": "pass",
                              "tiers": {"L3": {"status": "pass"}}, "numeric": {"status": "pass"},
                              "trace_check": {"status": "pass"}}])
    assert set(s["per_capsule"][0]) == {"capsule", "label", "status", "numeric", "trace", "tiers"}


def test_a_gated_capsule_carries_why(monkeypatch):
    """'gated' with no reason reads as 'skipped'. What it means is that the op suite did not earn the
    right to run this capsule, and the fraction it fell short by is the whole content of the verdict."""
    s = _score(monkeypatch, [{"capsule": "M0", "kind": "model", "label": "public", "status": "gated",
                              "failure": {"plane": "gate", "category": "GATED",
                                          "detail": "whole-model capsule deferred: op pass fraction "
                                                    "0.67 < gate 0.8"}}])
    assert "0.67 < gate 0.8" in s["per_capsule"][0]["gate_reason"]


def test_the_headline_names_capsules_that_were_never_certified(monkeypatch):
    """The headline exists so a reader who copies it cannot drop the qualification. A certify budget
    introduced a NEW thing to drop: capsules that passed the cheap screen and were never measured
    against the RTL tier. A bare "17/18" over a suite with seven of those is the same failure the
    headline was built to prevent."""
    results = ([{"capsule": f"p{i}", "status": "pass", "kind": "op",
                 "tiers": {"L3": {"status": "pass", "derived_from_rtl": True}}} for i in range(6)]
               + [{"capsule": f"s{i}", "status": "screened_only", "kind": "op", "tiers": {}}
                  for i in range(7)])
    s = _score(monkeypatch, results)
    assert s["n_screened_only"] == 7
    assert "7 more screened only, NOT certified" in s["headline"], s["headline"]
    assert "covering set" in s["headline"]


def test_a_fully_certified_suite_headline_is_unchanged(monkeypatch):
    """No budget, nothing screened: the headline must not grow noise."""
    s = _score(monkeypatch, [{"capsule": "p0", "status": "pass", "kind": "op",
                              "tiers": {"L3": {"status": "pass", "derived_from_rtl": True}}}])
    assert "screened" not in s["headline"], s["headline"]
