"""The bounded study preserves legacy deadlines and defers, never invents, grades."""
import json
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import joint_control as J
from merlin_experiments.phase1 import study_protocol as S


def policy():
    return dict(version=1, duration_s=15300, drain_s=900, final_model_s=900,
                study_protocol=dict(version=1, author_s=14400, evaluation="deferred",
                    scope="selected_capsules", snapshot_seconds=[0, 1800, 3600, 7200, 14400]))


def test_exact_four_hour_author_clock_and_private_model_refusal(tmp_path, monkeypatch):
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()))
    selected = J.policy(path)
    state = dict(policy=selected, started_at_epoch=1000)
    monkeypatch.setattr(J, "selected", lambda: state)
    assert J.remaining(drain=True, now=15399) == 1
    assert J.remaining(drain=True, now=15400) == 0
    assert J.remaining(now=15400) == 900
    with pytest.raises(RuntimeError, match="outside"):
        J.require_full_model(now=15400)
    assert "4 hours" in S.prompt(selected)


@pytest.mark.parametrize("change", [
    {"author_s": 10800}, {"evaluation": "inline"}, {"scope": "all_models"},
    {"snapshot_seconds": [0, 14400, 1800]}, {"snapshot_seconds": [0, 1800]},
])
def test_ambiguous_study_policy_refuses(change):
    selected = policy()
    selected["study_protocol"].update(change)
    with pytest.raises(ValueError):
        S.validate(selected)


def test_snapshot_is_complete_readonly_and_ignores_build_state(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "compiler.py").write_text("x=1\n")
    (source / "build").mkdir()
    (source / "build" / "kernel.o").write_bytes(b"build state")
    (source / "CMakeCache.txt").write_text("old build root")
    (source / ".ninja_log").write_text("old log")
    target = tmp_path / "snapshots" / "0"
    row = S.snapshot(source, target, offset_s=0, started_at=1000, now=1001)
    assert row["status"] == "frozen" and row["observed_elapsed_s"] == 1
    assert (target / "compiler.py").read_bytes() == (source / "compiler.py").read_bytes()
    assert not (target / "build").exists()
    assert not (target / "CMakeCache.txt").exists()
    assert not (target / ".ninja_log").exists()
    assert (target / "compiler.py").stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="already"):
        S.snapshot(source, target, offset_s=0, started_at=1000)


def test_moving_source_never_becomes_a_coherent_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "compiler.py").write_text("x=1\n")
    sequence = iter(["before", "changed"])
    monkeypatch.setattr(S, "_source_hash", lambda _: next(sequence))
    target = tmp_path / "snapshots" / "0"
    with pytest.raises(ValueError, match="changed"):
        S.snapshot(source, target, offset_s=0, started_at=1000)
    assert not target.exists()


def test_deferred_completion_pins_source_without_calling_final_grader(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_completion as C
    from merlin_experiments.phase1.telemetry import report
    workspace, run = tmp_path / "workspace", tmp_path / "run"
    (workspace / "submission").mkdir(parents=True)
    run.mkdir()
    (workspace / "submission" / "compiler.py").write_text("x=1\n")
    state = dict(policy=policy(), started_at_epoch=1000, policy_sha256="pinned-policy")
    monkeypatch.setattr(J, "selected", lambda: state)
    monkeypatch.setattr(J, "remaining", lambda **_: 0)
    monkeypatch.setattr(report, "_emit_run_timing", lambda *_: {"complete": False})
    prepared = SimpleNamespace(run_dir=run, workspace=workspace, verify_inputs=lambda: None)
    def forbidden(*_):
        pytest.fail("deferred study executed final coverage inside author run")
    assert C.complete(prepared, verdict={"all_pass": True}, rounds=[],
                      workflow_conformant=True, final_coverage=forbidden) == 0
    summary = json.loads((run / "joint_campaign_summary.json").read_text())
    assert summary["numeric_all_pass"] is None
    assert summary["final_cheap_coverage"]["status"] == "deferred"
    assert summary["final_validation"]["scope"] == ["selected_capsules", "frozen_transfer_panel"]
    assert summary["legacy_certificate_issued"] is False
