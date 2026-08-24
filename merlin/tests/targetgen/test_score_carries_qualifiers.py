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
