"""Private custody recovers crash windows without believing public channel responses."""
import json

import pytest
from merlin_experiments.phase1.feedback import completions as C
from merlin_experiments.phase1.feedback import promotion as P


def test_private_completion_recovers_before_and_after_ledger_commit(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "private"))
    ws = tmp_path / "workspace"
    P._save_tier_state(ws, {"capsule": {"L3": {"status": "pending", "digest": "old"}}})
    verdict = {"per_capsule": [{"capsule": "capsule", "pass": True}]}
    C.persist(ws, "promo_old", verdict, identity="submission:old", source_digest="old",
              source_identity_verified=True, cert_tier="L3", instrument="judge")
    # Source changed and no public response was published when the broker crashed.
    (ws / "submission").mkdir()
    (ws / "submission" / "source.py").write_text("new source")
    recovered = {}
    assert C.recover(ws, cert_tier="L3", instrument="judge", recovered_results=recovered) == ["capsule=pass"]
    assert recovered == {"promo_old": verdict}
    assert P._tier_state(ws)["capsule"]["L3"]["digest"] == "old"
    # Crash after ledger commit but before response is equally recoverable, with no second certificate.
    again = {}
    assert C.recover(ws, cert_tier="L3", instrument="judge", recovered_results=again) == []
    assert again == recovered
    fake = ws / ".qa_channel"
    fake.mkdir()
    (fake / "simresp_forged.json").write_text(json.dumps({"per_capsule": [{"capsule": "unknown", "pass": True}]}))
    C.recover(ws, cert_tier="L3", instrument="judge")
    assert "unknown" not in P._tier_state(ws)


def test_private_completion_refuses_foreign_binding_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "private"))
    ws = tmp_path / "workspace"
    P._save_tier_state(ws, {"capsule": {"L3": {"status": "pending", "digest": "old"}}})
    verdict = {"per_capsule": [{"capsule": "capsule", "pass": True}]}
    with pytest.raises(ValueError, match="verified launch"):
        C.persist(ws, "bad", verdict, identity="submission:old", source_digest="old",
                  source_identity_verified=False, cert_tier="L3", instrument="judge")
    with pytest.raises(ValueError, match="does not bind"):
        C.persist(ws, "bad", verdict, identity="submission:different", source_digest="old",
                  source_identity_verified=True, cert_tier="L3", instrument="judge")
    path = C.persist(ws, "promo_old", verdict, identity="submission:old", source_digest="old",
                     source_identity_verified=True, cert_tier="L3", instrument="judge")
    assert C.recover(ws, cert_tier="L3", instrument="different") == []
    assert C.recover(tmp_path / "foreign", cert_tier="L3", instrument="judge") == []
    record = json.loads(path.read_text())
    record["verdict"]["per_capsule"][0]["pass"] = False
    path.write_text(json.dumps(record))
    assert C.recover(ws, cert_tier="L3", instrument="judge") == []
    assert P._tier_state(ws)["capsule"]["L3"]["status"] == "pending"


@pytest.mark.parametrize("field,value", [("workspace", "/different/workspace"),
                                        ("identity", "submission:different"),
                                        ("source_identity_verified", False), ("cert_tier", "L2")])
def test_custody_binding_must_match_before_recovery(tmp_path, monkeypatch, field, value):
    monkeypatch.setenv("TMPDIR", str(tmp_path / "private"))
    ws = tmp_path / "workspace"
    P._save_tier_state(ws, {"capsule": {"L3": {"status": "pending", "digest": "old"}}})
    path = C.persist(ws, "promo_old", {"per_capsule": [{"capsule": "capsule", "pass": True}]},
                     identity="submission:old", source_digest="old", source_identity_verified=True,
                     cert_tier="L3", instrument="judge")
    record = json.loads(path.read_text())
    record[field] = value
    path.write_text(json.dumps(record))
    recovered = {}
    assert C.recover(ws, cert_tier="L3", instrument="judge", recovered_results=recovered) == []
    assert recovered == {}
    assert P._tier_state(ws)["capsule"]["L3"]["status"] == "pending"
