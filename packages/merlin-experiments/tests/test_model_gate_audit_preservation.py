"""Unexecuted model gates stay distinct from missing native execution evidence."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import selfcheck

from merlin.targetgen import capsule_grade as grade


def _gated(name="model"):
    return {
        "capsule": name, "kind": "model", "label": "public", "status": "gated",
        "failure": {"plane": "gate", "category": "GATED",
                    "detail": "whole-model capsule deferred: op pass fraction 0.00 < gate 0.8"},
    }


def test_direct_execution_enforcement_preserves_unexecuted_gate(monkeypatch):
    row = _gated()
    before = deepcopy(row)
    monkeypatch.setattr(grade, "model_execution_check", lambda *a, **k: pytest.fail("gated model cannot be audited"))
    assert grade.enforce_model_execution_check(row, {}, target="synthetic") is row
    assert row == before
    assert "model_execution_check" not in row and "tiers" not in row


def test_executed_owned_model_without_receipt_still_refuses(monkeypatch):
    from merlin.targetgen import native_model_execution

    monkeypatch.setattr(native_model_execution, "independent_frozen_source_eligibility",
                        lambda *a, **k: {"n_eligible": 1})
    row = {"kind": "model", "status": "pass", "numeric": {"status": "pass"}}
    capsule = {"semantic": {"must_accelerate": False}, "lanes": {"require": ["on_mesh"]}}
    result = grade.enforce_model_execution_check(row, capsule, target="synthetic")
    assert result["status"] == "incomplete"
    assert result["model_execution_check"]["kind"] == "owned_host_model_execution"
    assert "receipt or original frozen capsule is absent" in result["model_execution_check"]["detail"]


@pytest.fixture
def gated_grade(tmp_path, monkeypatch):
    capsules = [{"name": f"model{i}", "kind": "model", "label": "public",
                 "gate": {"after_op_pass_fraction": 0.8}} for i in range(6)]
    monkeypatch.setattr(grade, "source_experiment_env", lambda target: {})
    tool = tmp_path / "compiler.py"
    tool.write_text("# fixture compiler entrypoint\n")
    monkeypatch.setattr(grade, "load_package", lambda *a, **k: SimpleNamespace(integrity_exempt=False, tool=tool))
    monkeypatch.setattr(grade, "integrity_scan", lambda *a, **k: None)
    monkeypatch.setattr(grade, "build_package", lambda *a, **k: None)
    monkeypatch.setattr(grade.CR, "discover_capsules", lambda *a, **k: capsules)
    monkeypatch.setattr(grade, "_materialized_cohort_admission", lambda *a: None)
    monkeypatch.setattr(grade, "model_execution_check", lambda *a, **k: pytest.fail("deferred model was audited"))
    monkeypatch.setattr(grade.CR, "suite_for", lambda target: "fixture-suite")

    def suite(caps, package, **kwargs):
        # Use the actual scheduler gate, exercising both grade completion hooks.
        rows = grade.CR._run_streaming_model_gate(
            [], caps, lambda *a: pytest.fail("gated model was executed"), max_workers=1,
            target="synthetic", oracle_adapters={}, on_complete=kwargs.get("on_capsule_complete"),
        )
        kwargs["post_suite"](rows)
        return rows

    monkeypatch.setattr(grade.CR, "run_suite", suite)
    return capsules


def test_grade_completion_and_final_audit_keep_all_model_gates(tmp_path, gated_grade):
    completed, selected = [], []
    score = grade.grade(
        tmp_path / "package", capsules_root=tmp_path / "capsules", runs_root=tmp_path / "result",
        target="synthetic", max_workers=1, oracle_adapters={},
        on_capsule_complete=lambda row: completed.append(deepcopy(row)),
        on_cohort_ready=selected.append,
    )
    assert selected == [6]  # Selected denominator survives all six model gates.
    assert len(completed) == 6 and all(row["status"] == "gated" for row in completed)
    assert score["n_gated_deferred"] == 6 and score["n_capsules"] == score["n_passed"] == 0
    assert all(row["status"] == "gated" and row["gate_reason"] for row in score["per_capsule"])
    assert not list((tmp_path / "result").rglob("capsule_result.json"))
    feedback = selfcheck._gated_without_result_rows(
        score, requested={cap["name"] for cap in gated_grade},
        models={cap["name"] for cap in gated_grade}, represented=set(),
    )
    assert len(feedback) == 6 and all(row["pass"] is False for row in feedback)


def test_model_only_selfcheck_reports_redacted_model_gate(tmp_path, monkeypatch, capsys, gated_grade):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    package, capsules = tmp_path / "submission", tmp_path / "capsules"
    package.mkdir()
    (package / "manifest.yaml").write_text("commands: {}\n")
    capsules.mkdir()
    monkeypatch.setattr(selfcheck, "resolve_context", lambda *a: SimpleNamespace(repo=tmp_path, target="synthetic"))
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *a: ("synthetic", "fixture"))
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a, **k: ({"L2": object()}, "spike"))
    monkeypatch.setattr(selfcheck, "oracle_kwargs", lambda *a: {})
    monkeypatch.setattr(selfcheck, "_streaming_completion", lambda **k: None)
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *a: None)
    result = selfcheck.main(
        ["--submission", str(package), "--sim", "spike", "--workers", "1"],
        context=object(), capsules_root=capsules,
    )
    output = json.loads(capsys.readouterr().out)
    assert result == 1 and output["model_gate"] is True
    assert output["all_pass"] is False and output["n_passed"] == 0 and output["n_capsules"] == 6
    assert len(output["per_capsule"]) == 6
    assert all(row["status"] == "gated" and row["pass"] is False for row in output["per_capsule"])
    assert "MODEL GATE" in output["note"] and "HARNESS" not in output["note"]
    assert not output.get("no_results") and not output.get("build_failed")
    assert str(tmp_path) not in json.dumps(output)
