"""A required refusal is successful qualification, never a fabricated native tier."""
from io import StringIO

from merlin_experiments.phase1.feedback import promotion as P
from merlin_experiments.phase1.feedback.selfcheck import _successful_expected_refusal


def test_declared_refusal_passes_without_numeric_or_tier_measurement():
    result = {"status": "pass", "failure": None, "tiers": {},
              "declined": {"reason": "unsupported epilogue"}, "expected_refusal": {"status": "met"}}
    assert _successful_expected_refusal(result)
    assert result["tiers"] == {}
    assert not _successful_expected_refusal({**result, "expected_refusal": {"status": "violated"}, "status": "fail"})
    assert not _successful_expected_refusal({**result, "expected_refusal": None, "status": "declined"})


def test_successful_refusal_never_promotes_even_with_old_loop_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "submission_digests", lambda ws: ("bound", {}, []))
    monkeypatch.setattr(P, "decomposition", lambda ws: {"source": None, "inert": []})
    monkeypatch.setattr(P, "_mandatory_cert_members", lambda **kwargs: {"refusal"})
    monkeypatch.setattr(P, "cert_sim", lambda *args, **kwargs: "gsim")
    monkeypatch.setattr(P, "cert_instrument", lambda *args, **kwargs: "judge")
    P._save_tier_state(tmp_path, {"refusal": {"L2": {"status": "pass", "digest": "bound"}}})
    channel = tmp_path / ".qa_channel"
    channel.mkdir()
    verdict = {"per_capsule": [{"capsule": "refusal", "pass": True, "tiers": {},
                                "expected_refusal": {"status": "met"}}]}
    assert P.promote(tmp_path, channel, verdict, "L2", "L3", set(), StringIO(), context=None) == []
    assert list(channel.glob("simreq_*.json")) == []
