import pytest
from merlin_experiments.phase1.feedback.formal import private_gate_completion


def test_explicit_private_gate_deferral_is_not_a_pass():
    receipt = {"passed": False, "models": []}
    assert private_gate_completion(["model"], receipt, deferred_reason="operator deferred full models")
    assert receipt["status"] == "deferred" and receipt["passed"] is False


def test_missing_private_gate_stays_incomplete():
    assert not private_gate_completion(["model"], {"passed": False})
    assert not private_gate_completion([], {"passed": True})
    assert private_gate_completion(["model"], {"passed": True})
    with pytest.raises(ValueError):
        private_gate_completion(["model"], {"models": []}, deferred_reason="")
    with pytest.raises(ValueError):
        private_gate_completion(["model"], {"models": ["ran"]}, deferred_reason="defer")
