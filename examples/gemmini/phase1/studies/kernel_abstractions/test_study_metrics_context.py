"""Automatic private context and scratch milestones need no initial executable."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def protocol():
    path = Path(__file__).parents[5] / "packages/merlin-experiments/src/merlin_experiments/phase1/study_protocol.py"
    return module("merlin_experiments.phase1._study_context_under_test", path)


def state():
    return {"started_at_epoch": 1000, "policy": {
        "duration_s": 15300, "drain_s": 900,
        "study_protocol": {"version": 1, "author_s": 14400, "evaluation": "deferred",
                           "scope": "selected_capsules", "snapshot_seconds": [0, 14400]}}}


def test_context_writes_original_clock_public_ids_and_source_without_fake_elf(tmp_path):
    api = protocol()
    run, ws = tmp_path / "run", tmp_path / "workspace"
    run.mkdir()
    index = {"snapshots": [{"offset_s": 0, "status": "frozen", "compiler_sha256": "initial",
                             "observed_elapsed_s": 0.25}]}
    value = api.write_metrics_context(ws, run, state(), index,
        public_roster={"core": ["A"], "scalability": ["B"], "programs": ["C"]},
        environment={"backend_origin": {"schema": "merlin.empty_backend_origin.v1"}})
    assert value["started_at"] == 1000 and value["observation_end_at"] == 15400
    assert value["public_case_ids"] == ["A", "B", "C"]
    assert value["initial_source_sha256"] == "initial"
    assert value["backend_origin"] == "scratch" and value["initial_case_elf_sha256"] == {}
    assert not value["author_visible"] and not value["qualification_issued"]
    assert json.loads((run / "study-metrics-context.json").read_text()) == value
    # An operator's separately qualified initial-case evidence survives resume.
    value["initial_case_elf_sha256"] = {"A": "retained-seed-elf"}
    (run / "study-metrics-context.json").write_text(json.dumps(value))
    resumed = api.write_metrics_context(ws, run, state(), index,
        public_roster=value["public_cases_by_suite"],
        environment={"backend_origin": {"schema": "merlin.empty_backend_origin.v1"}})
    assert resumed["initial_case_elf_sha256"] == {"A": "retained-seed-elf"}
    changed = state()
    changed["started_at_epoch"] += 1
    with pytest.raises(ValueError, match="resume"):
        api.write_metrics_context(ws, run, changed, index,
            public_roster=value["public_cases_by_suite"], environment={})


def test_seeded_origin_wins_over_inherited_empty_origin_and_unknown_snapshot_stays_unknown(tmp_path):
    api = protocol()
    index = {"snapshots": [{"offset_s": 0, "status": "unavailable", "observed_elapsed_s": 1.0}]}
    selected = state()
    selected["policy"]["compiler_origin"] = {"source": "reviewed-fresh6"}
    value = api.write_metrics_context(tmp_path / "ws", tmp_path, selected, index,
        public_roster={"core": ["public"]}, environment={"backend_origin": {"schema": "merlin.empty_backend_origin.v1"}})
    assert value["backend_origin"] == "seeded" and value["initial_source_sha256"] is None


def test_actual_roster_uses_only_admitted_public_suites_and_exclusions(tmp_path, monkeypatch):
    api = protocol()
    from merlin_experiments.phase1.feedback import public_programs, public_scalability

    from merlin.targetgen import capsule_common, target_experiment
    experiment = SimpleNamespace(graded_roots=lambda: ["selected"], effective_exclusions=lambda _: ["exclude"])
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda _: experiment)
    def discover(roots, *, labels, contract):
        assert labels == {"public", "dev"} and roots == ["selected"]
        return [{"name": "core"}, {"name": "exclude"}]
    monkeypatch.setattr(capsule_common, "discover_capsules", discover)
    monkeypatch.setattr(public_scalability, "selected", lambda _: (None, {"cases": {"scale": {}}}))
    monkeypatch.setattr(public_programs, "selected", lambda _: (None, {}, {"program": None}))
    value = api._metrics_roster(SimpleNamespace(descriptor="descriptor", repo=tmp_path))
    assert value == {"core": ["core"], "scalability": ["scale"], "programs": ["program"]}


def test_scratch_numeric_execution_milestone_needs_no_nonexistent_initial_elf(tmp_path):
    api = module("study_metrics_scratch_under_test", Path(__file__).with_name("study_metrics.py"))
    context = {"started_at": 100, "observation_end_at": 200, "initial_source_sha256": "initial",
               "initial_case_elf_sha256": {}, "public_case_ids": ["G"], "backend_origin": "scratch"}
    event = {"kind": "case_result", "at": 130, "attempt_id": "one", "source_sha256": "authored",
             "case": "G", "numeric_status": "pass", "tiers": {"L2": "pass"}}
    result = api.extract_metrics(tmp_path, context=context,
        decomposition={"provider_usage_events": []}, observations=[event])
    assert result["milestones"]["first_correct_from_scratch"]["elapsed_s"] == 30
    assert result["milestones"]["first_correct_from_scratch"]["elf_sha256"] is None
    assert result["milestones"]["first_correct_changed_kernel"]["elapsed_s"] is None
    # Source-only/L1 observations and seeded runs do not become scratch results.
    for bad_context, bad_event in [(context, {**event, "tiers": {"L1": "pass"}}),
                                   ({**context, "backend_origin": "seeded"}, event),
                                   (context, {**event, "source_sha256": "initial"})]:
        measured = api.extract_metrics(tmp_path, context=bad_context,
            decomposition={"provider_usage_events": []}, observations=[bad_event])
        assert measured["milestones"]["first_correct_from_scratch"]["elapsed_s"] is None


def test_native_row_adapter_keeps_execution_tier_for_the_scratch_milestone():
    api = module("study_metrics_row_under_test", Path(__file__).with_name("study_metrics.py"))
    row = api._row_event({"capsule": "G", "numeric_status": "pass", "tiers": {"L2": "pass"},
                          "execution_digest": "executed"}, attempt="one", source="source", at=130)
    assert row["tiers"] == {"L2": "pass"} and row["execution_digest"] == "executed"
