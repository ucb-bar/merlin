"""Observation failures retain stage/budget diagnostics without private details."""
from merlin_experiments.phase1.feedback.joint_performance_report import project
from merlin_experiments.phase2 import archive_count_observation as observer


def test_failed_runtime_retains_stage_and_never_claims_target_execution(tmp_path):
    result = observer.observe(tmp_path / "unused.elf", runtime=object(), policy=None,
                              directory=tmp_path / "unused", input_sha256="unused",
                              remaining_s=1, numerical_pass=True)
    assert result["status"] == "UNKNOWN"
    assert result["failure_stage"] == "runtime_validation"
    assert not result["deadline_exhausted"]
    assert result["wall_seconds"] >= 0
    assert not (tmp_path / "unused").exists()


def test_deadline_stage_projection_does_not_publish_private_exception_or_cycles():
    raw = dict(status="UNKNOWN", failure_stage="annotated_execution", deadline_exhausted=True,
               wall_seconds=120.5, detail="private path and reference", failure_category="CensusUnavailable")
    result = project(None, raw, {}, selection_sha256="a" * 64, reason="observation unavailable")
    assert result["observation_diagnostics"]["failure_stage"] == "annotated_execution"
    assert result["observation_diagnostics"]["deadline_exhausted"]
    assert result["observation_diagnostics"]["host_wall_seconds"] == 120.5
    assert "private path" not in str(result)
    assert result["status"] == "UNKNOWN" and result["estimated_whole_soc_cycles"] is None
    assert not result["timing_qualified"]
    raw["failure_stage"] = "/private/path"
    assert "observation_diagnostics" not in project(None, raw, {}, selection_sha256="a" * 64, reason="unknown")
