"""Joint routing keeps large feedback early and full model execution late."""

import sys
from types import ModuleType

from merlin_experiments.phase1.brokers import model_check
from merlin_experiments.phase1.feedback import public_scalability as F


def control(monkeypatch, *, selected=True):
    module = ModuleType("merlin_experiments.phase1.joint_control")
    module.selected = lambda: {} if selected else None
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def test_joint_large_suite_does_not_wait_for_core_success(tmp_path, monkeypatch):
    control(monkeypatch)
    monkeypatch.setattr(F, "selected", lambda _: (tmp_path, {}))
    calls = []

    def grade(*args, **kwargs):
        calls.append(kwargs)
        return {"passed": True, "observed": {}, "score": {}}

    monkeypatch.setattr(F, "grade", grade)
    verdict = {"all_pass": False}
    F.attach(verdict, tmp_path, context=object(), output=tmp_path / "out", timeout=1)
    assert len(calls) == 1
    assert verdict["public_scalability"]["passed"] is True
    assert verdict["all_pass"] is False  # Large passes do not erase core failures.


def test_legacy_waits_for_core_before_large_suite(tmp_path, monkeypatch):
    control(monkeypatch, selected=False)
    monkeypatch.setattr(F, "selected", lambda _: (tmp_path, {}))
    monkeypatch.setattr(F, "grade", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not execute")))
    verdict = {"all_pass": False}
    F.attach(verdict, tmp_path, context=object(), output=tmp_path / "out", timeout=1)
    assert verdict["public_scalability"]["status"] == "waiting_for_core"


def test_early_full_model_request_refuses_before_any_subprocess(tmp_path, monkeypatch):
    owner = control(monkeypatch)
    def refuse(*, engine):
        assert engine == "spike"
        raise RuntimeError("outside final window")
    owner.require_full_model = refuse
    monkeypatch.setattr(model_check.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    result = model_check.execute_visible(tmp_path, {"action": "check", "model": "resnet50", "split": "visible", "mode": "full"})
    assert result == {"status": "deferred", "error": "outside final window"}
