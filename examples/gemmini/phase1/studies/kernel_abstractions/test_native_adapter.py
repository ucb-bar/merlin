"""Native artifact adapters do not launch authors, custom graders or hardware."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.tree_hash import hash_tree


def module():
    directory = Path(__file__).parent
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("native_adapter_under_test", directory / "native_adapter.py")
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        return api
    finally:
        sys.path.pop(0)


def native_fixture(tmp_path, monkeypatch):
    api = module()
    directory = Path(__file__).parent
    # Sibling production helpers are also imported lazily by the adapter.
    monkeypatch.syspath_prepend(str(directory))
    def dump(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return api.pin(path)
    run = tmp_path / "native-run"
    source = run / "joint_frozen_submission"
    source.mkdir(parents=True)
    (source / "compiler.py").write_text("# frozen compiler\n")
    (run / "joint_campaign_summary.json").write_text(json.dumps({
        "completion_reason": "freeze_drain", "frozen_compiler": str(source),
        "compiler_sha256": hash_tree(source)["sha256"]}))
    rules = dump("rules.json", {"rules": "derived from public premises"})
    settings = dump("settings.json", {"selected": "final"})
    descriptor, facts, environment = dump("descriptor.json", {}), dump("facts.json", {}), dump("env.json", {})
    deps, native_environment = dump("deps.json", {"compiler_sha256": hash_tree(source)["sha256"]}), dump("native-environment.json", {})
    runtime = dump("runtime.json", {"python": api.pin(Path(sys.executable).resolve()), "repo": str(tmp_path),
        "descriptor": descriptor, "rtl_facts": facts, "environment": environment,
        "dependency_selection": deps, "run_environment": native_environment, "workers": 2, "timeout_s": 900})
    subjects = dump("subjects.json", {"subjects": [{"id": "run-00", "run_dir": str(run), "snapshot": "final",
        "rules": rules, "settings": settings, "frozen_unix": 1000, "runtime_config": runtime}]})
    cases = [{"case": f"case{i}", "family": "gemm", "split": "heldout", "source": facts,
              "fixtures": {}} for i in range(34)]
    panel = dump("panel.json", {"schema": "merlin.study.transfer_panel.v1", "cases": cases})
    artifacts = dump("artifacts.json", {"cells": []})
    measurement = dump("measurement.json", {"snapshot": "final", "domain": {}, "toolchain": {}})
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "selection.json").write_text("{}")
    from merlin_experiments.corpus import public_scalability

    from merlin.targetgen import target_experiment
    monkeypatch.setattr(target_experiment, "load_target_experiment", lambda *_: SimpleNamespace(descriptor_sha256="descriptor-digest"))
    monkeypatch.setattr(public_scalability, "selected", lambda *_: (suite, {"cases": {c["case"]: {} for c in cases}}))
    return api, subjects, panel, artifacts, measurement, run


def test_native_final_freeze_builds_ordinary_worker_commands_and_full_roster(tmp_path, monkeypatch):
    api, subjects, panel, artifacts, measurement, run = native_fixture(tmp_path, monkeypatch)
    result = api.plan(panel["path"], subjects["path"], artifacts["path"], measurement["path"], tmp_path / "private")
    assert result["schema"] == api.SCHEMA and result["executed"] is False
    assert len(result["cases"]) == len(result["jobs"][0]["cases"]) == 34
    command = result["jobs"][0]["argv"]
    assert command[command.index("-m") + 1] == "merlin_experiments.phase1.feedback.selfcheck"
    assert command[command.index("--suite") + 1] == "scalability"
    assert command[command.index("--tiers") + 1] == "L0,L1,L2"
    assert not any("author" in token for token in command)
    selection = json.loads((tmp_path / "private/physical-selection.json").read_text())
    assert selection["compiled"] == [] and selection["subjects"] == ["run-00"]
    assert selection["evaluation_plan"]["sha256"] == api.pin(tmp_path / "private/plan.json")["sha256"]


def test_missing_native_results_keep_all_34_cells_without_passes(tmp_path, monkeypatch):
    api, subjects, panel, artifacts, measurement, run = native_fixture(tmp_path, monkeypatch)
    api.plan(panel["path"], subjects["path"], artifacts["path"], measurement["path"], tmp_path / "private")
    result = api.collect(tmp_path / "private/plan.json")
    assert len(result["cells"]) == 34
    assert all(row["state"] == "missing" and row["physical"]["cycles"] is None for row in result["cells"])


def test_native_checkpoint_reads_exact_declared_index_and_actual_time(tmp_path, monkeypatch):
    api, subjects, panel, artifacts, measurement, run = native_fixture(tmp_path, monkeypatch)
    source = run / "joint_frozen_submission"
    (run / "study_snapshots").mkdir()
    (run / "study_snapshots/index.json").write_text(json.dumps({"started_at_epoch": 1000, "snapshots": [
        {"offset_s": 1800, "observed_elapsed_s": 1801.5, "source": str(source),
         "compiler_sha256": hash_tree(source)["sha256"], "status": "frozen"}]}))
    row = api.read(subjects)["subjects"][0]
    row["snapshot"] = 1800
    selected = api.freeze_subject(row)
    assert selected["freeze"]["frozen_unix"] == 2801.5
    row["snapshot"] = 3600
    with pytest.raises(ValueError, match="unavailable"):
        api.freeze_subject(row)


def test_changed_native_source_and_incomplete_native_completion_refuse(tmp_path, monkeypatch):
    api, subjects, panel, artifacts, measurement, run = native_fixture(tmp_path, monkeypatch)
    row = api.read(subjects)["subjects"][0]
    (run / "joint_frozen_submission/compiler.py").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        api.freeze_subject(row)
    summary = json.loads((run / "joint_campaign_summary.json").read_text())
    summary["completion_reason"] = "interrupted"
    (run / "joint_campaign_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="boundary"):
        api.freeze_subject(row)
