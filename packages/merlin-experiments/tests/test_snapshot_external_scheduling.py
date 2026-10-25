"""Independent model admission changes scheduling, never a model verdict."""
from types import SimpleNamespace

from merlin.targetgen import capsule_runner as runner


def test_requested_model_runs_without_cohort_and_retains_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_COMPILER_EXECUTION_MODE", "snapshot_external_v1")
    monkeypatch.setenv("MERLIN_OPERATOR_GRADE_PROCESSES", "0")
    monkeypatch.setattr(runner, "load_package", lambda *a, **k: SimpleNamespace(directory=tmp_path))
    monkeypatch.setattr(runner, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(runner, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_candidate_boundary_readiness", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_split_ineligible", lambda caps, target: (caps, []))
    monkeypatch.setattr(runner._tier_policy, "covering_set", lambda caps: [])
    monkeypatch.setattr(runner._tier_policy, "priced_tiers", lambda target: {"L2"})
    calls = []

    def grade(cap, *args, **kwargs):
        calls.append(cap["name"])
        return {"capsule": cap["name"], "kind": "model", "status": "fail",
                "numeric": {"status": "fail", "mismatch_count": 1}}

    monkeypatch.setattr(runner, "run_capsule", grade)
    model = {"name": "requested", "kind": "model", "gate": {"after_op_pass_fraction": .8}}
    rows = runner._run_suite([model], tmp_path, runs_root=tmp_path / "runs", target="fixture",
                             max_workers=2, oracle_adapters={"L2": object()})
    assert calls == ["requested"]
    assert rows[0]["status"] == "fail"
    assert rows[0]["numeric"]["mismatch_count"] == 1
    admission = rows[0]["model_gate_admission"]
    assert admission["policy"] == "independent_requests_v1"
    assert admission["prerequisite_credit"] is False
    assert "fraction" not in admission and "passed" not in admission
