"""Independently admitted models still reach their certificate tier.

Snapshot execution admits a model by request (no operator-fraction gate, no
prerequisite credit). A completed L2 model pass there must be promoted without
a staged gate voucher; the staged treatment keeps requiring its voucher.
"""
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.brokers import simjob
from merlin_experiments.phase1.feedback import loop_grading as G
from merlin_experiments.phase1.feedback import model_gate_vouchers, promotion, qa, snapshot_execution

from merlin.targetgen import capsule_common

INDEPENDENT = {"policy": "independent_requests_v1", "execution_mode": snapshot_execution.MODE,
               "prerequisite_credit": False}


def _promoter(tmp_path, monkeypatch):
    ws, frozen, runs = (tmp_path / name for name in ("ws", "frozen", "runs"))
    for root in (ws, frozen):
        (root / "submission").mkdir(parents=True)
    (ws / ".qa_channel").mkdir()
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_text("{}\n")
    inputs = SimpleNamespace(context=SimpleNamespace(descriptor=descriptor), public_root=tmp_path / "public",
                             contract=tmp_path / "contract", promotion_root=tmp_path / "policy")
    monkeypatch.setattr(G, "load_target_experiment",
                        lambda _: SimpleNamespace(target="fixture", effective_exclusions=lambda names: set()))
    monkeypatch.setattr(capsule_common, "discover_capsules", lambda *a, **kw: [{"name": "model", "kind": "model"}])
    monkeypatch.setattr(G, "_suite_for", lambda _: "suite")
    monkeypatch.setattr(qa, "_execution_digest_from_result", lambda _: None)
    issued, promoted = [], []
    monkeypatch.setattr(model_gate_vouchers, "issue", lambda *a, **kw: issued.append(kw["capsule"]["name"]))
    monkeypatch.setattr(promotion, "promote", lambda ws, ch, verdict, *a, **kw: promoted.append(verdict))
    callback = G._completed_loop_promoter(ws, frozen, runs, "L2", "L3", None, inputs=inputs)

    def complete(admission):
        row = {"capsule": "model", "kind": "model", "tiers": {"L2": {"status": "pass"}},
               "model_gate_admission": admission}
        path = runs / "runs/suite/model/capsule_result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row))
        callback(row)

    return complete, issued, promoted


def test_independent_admission_promotes_completed_model_without_voucher(tmp_path, monkeypatch):
    complete, issued, promoted = _promoter(tmp_path, monkeypatch)
    complete(INDEPENDENT)
    assert issued == []
    assert [row["capsule"] for verdict in promoted for row in verdict["per_capsule"]] == ["model"]


def test_staged_admission_still_issues_its_voucher(tmp_path, monkeypatch):
    monkeypatch.delenv(snapshot_execution.ENV, raising=False)
    complete, issued, promoted = _promoter(tmp_path, monkeypatch)
    complete({"passed": True, "fraction": 1.0, "stage": "L2"})
    assert issued == ["model"] and len(promoted) == 1


def _caps(tmp_path):
    root = tmp_path / "caps"
    for name, kind in (("OP", "op"), ("MODEL", "model")):
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / name / "capsule.yaml").write_text(f"kind: {kind}\n")
    return root


def test_simjob_certifies_independent_model_with_its_own_l2(tmp_path, monkeypatch):
    monkeypatch.setenv(snapshot_execution.ENV, snapshot_execution.MODE)
    monkeypatch.setattr(model_gate_vouchers, "lookup", lambda *a, **kw: pytest.fail("no voucher in this mode"))
    tiers, vouchers = simjob._promoted_model_continuation(
        tmp_path, ["OP", "MODEL"], _caps(tmp_path), "L3", digest="d", context=None, contract=None, cert_tier="L3")
    assert (tiers, vouchers) == ("L2,L3", [])
    # Operator-only promotions keep the requested tier.
    assert simjob._promoted_model_continuation(
        tmp_path, ["OP"], _caps(tmp_path), "L3", digest="d", context=None, contract=None, cert_tier="L3"
    ) == ("L3", [])


def test_simjob_staged_model_still_requires_voucher(tmp_path, monkeypatch):
    monkeypatch.delenv(snapshot_execution.ENV, raising=False)
    root = _caps(tmp_path)
    monkeypatch.setattr(model_gate_vouchers, "lookup", lambda *a, **kw: None)
    with pytest.raises(ValueError, match="voucher"):
        simjob._promoted_model_continuation(
            tmp_path, ["MODEL"], root, "L3", digest="d", context=None, contract=None, cert_tier="L3")
    monkeypatch.setattr(model_gate_vouchers, "lookup", lambda *a, **kw: tmp_path / "v.json")
    assert simjob._promoted_model_continuation(
        tmp_path, ["MODEL"], root, "L3", digest="d", context=None, contract=None, cert_tier="L3"
    ) == ("L2,L3", [tmp_path / "v.json"])


def test_independent_priority_needs_no_voucher(tmp_path, monkeypatch):
    monkeypatch.setenv(snapshot_execution.ENV, snapshot_execution.MODE)
    monkeypatch.setattr(simjob._TP, "_submission_digest", lambda _: "current")
    monkeypatch.setattr(model_gate_vouchers, "lookup", lambda *a, **kw: None)
    paths = []
    for index, name in enumerate(["OP", "MODEL"]):
        path = tmp_path / f"simreq_{index}.json"
        path.write_text(json.dumps({"capsules": name, "promoted": True, "submission_digest": "current"}))
        paths.append(path)
    ordered = simjob._current_source_first(paths, tmp_path, context=SimpleNamespace(),
                                          capsules_root=_caps(tmp_path), cert_tier="L3")
    assert ordered == [paths[1], paths[0]]
