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


def test_final_model_uses_capture_python_and_bound_configuration(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from merlin.integrations import model2mlir

    owner = control(monkeypatch)
    owner.require_full_model = lambda **_: 120
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("version: 1\n")
    bundle = tmp_path / "bundle"
    (bundle / "configuration").mkdir(parents=True)
    config = bundle / "configuration" / "models.yaml"
    config.write_text("configuration is selected by the operator\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("MERLIN_MODEL_CHECK_BUNDLE", str(bundle))
    monkeypatch.setattr(model_check, "target_from_contract", lambda: "test")
    monkeypatch.setattr(model2mlir, "capture_python", lambda: Path("/selected/model/python"))
    probed = []
    monkeypatch.setattr(model_check, "capture_import_failure", lambda python, **_: probed.append(python))
    calls = []

    def launch(argv, **_):
        calls.append(argv)
        Path(argv[argv.index("--out") + 1]).write_text(json.dumps({
            "schema": model_check.REPORT_SCHEMA, "command": "check", "model": "resnet50",
            "split": "visible", "agent_view": True, "status": "pass", "exit_code": 0,
        }))
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    monkeypatch.setattr(model_check.subprocess, "Popen", launch)
    result = model_check.execute_visible(tmp_path, {"action": "check", "model": "resnet50",
                                                    "split": "visible", "mode": "full"},
                                         snapshot_source=submission)
    assert result["status"] == "pass"
    argv = calls[0]
    assert argv[0] == "/selected/model/python"
    assert argv[argv.index("--models") + 1] == str(config)
    assert argv[argv.index("--full-comparison") + 1] == "quantized-reference"
    assert argv[argv.index("--engine") + 1] == "spike"
    assert argv[argv.index("--timeout-s") + 1] == "120"
    assert probed == ["/selected/model/python"]


def test_final_model_without_child_dependencies_fails_fast_as_harness_error(tmp_path, monkeypatch):
    from pathlib import Path
    from merlin.integrations import model2mlir

    owner = control(monkeypatch)
    owner.require_full_model = lambda **_: 120
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("version: 1\n")
    bundle = tmp_path / "bundle"
    (bundle / "configuration").mkdir(parents=True)
    (bundle / "configuration" / "models.yaml").write_text("selected\n")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("MERLIN_MODEL_CHECK_BUNDLE", str(bundle))
    monkeypatch.setattr(model_check, "target_from_contract", lambda: "test")
    monkeypatch.setattr(model2mlir, "capture_python", lambda: Path("/selected/model/python"))
    monkeypatch.setattr(model_check, "capture_import_failure",
                        lambda python, **_: "ModuleNotFoundError: No module named 'jsonschema'")
    monkeypatch.setattr(model_check.subprocess, "Popen",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not launch")))
    result = model_check.execute_visible(tmp_path, {"action": "check", "model": "resnet50",
                                                    "split": "visible", "mode": "full"},
                                         snapshot_source=submission)
    assert result["status"] == "error" and result["model"] == "resnet50"
    assert "jsonschema" in result["error"] and "/selected/model/python" in result["error"]


def test_capture_import_probe_reports_the_missing_module(tmp_path, monkeypatch):
    import os

    model_check._READY.clear()
    broken = tmp_path / "python"
    broken.write_text("#!/bin/sh\necho \"ModuleNotFoundError: No module named 'jsonschema'\" >&2\nexit 1\n")
    broken.chmod(0o755)
    assert model_check.capture_import_failure(str(broken)) == "ModuleNotFoundError: No module named 'jsonschema'"
    assert str(broken) not in model_check._READY
    # The interpreter that runs these tests has every child dependency; success is remembered.
    assert model_check.capture_import_failure(sys.executable, env=dict(os.environ)) is None
    assert model_check.capture_import_failure(sys.executable) is None
    assert sys.executable in model_check._READY
    assert model_check.capture_import_failure(str(tmp_path / "missing")).startswith("FileNotFoundError")
