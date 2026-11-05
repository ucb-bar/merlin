"""Prospective operator preparation uses no candidate, paid author or target."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase1 import launch_config as L

HERE = Path(__file__).parent


def module(name):
    spec = importlib.util.spec_from_file_location("test_successor_" + name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_selected_allocation_has_no_eighteen_cpu_assumption():
    operator = module("launch_phase1")
    compute = [0, 1, 2, 3, *range(24, 32)]
    launch = SimpleNamespace(
        variables={}, document={"cpu": {"author": {"cpus": [4, 5]}, "compute_pool": {"cpus": compute}}}
    )
    selected = operator.assigned_cpus(launch, available=range(32))
    assert selected == {0, 1, 2, 3, 4, 5, *range(24, 32)}
    with pytest.raises(ValueError, match="unavailable"):
        operator.assigned_cpus(launch, available=range(24))
    launch.document["cpu"]["compute_pool"]["cpus"].append(4)
    with pytest.raises(ValueError, match="overlap"):
        operator.assigned_cpus(launch, available=range(32))


def test_template_selects_exact_clocks_empty_origin_and_separate_external_owner(tmp_path):
    launch = L.build(
        L.load(HERE / "el4-successor.yaml"),
        task_root=tmp_path,
        prep=tmp_path / "prep",
        run_id="fresh",
        environ={"HOME": str(tmp_path / "home")},
    )
    document = launch.document
    assert document["budgets"]["authoring_s"] == 10800
    assert document["budgets"]["supervisor_backstop_s"] == 14400
    assert launch.variables["SUPERVISOR_FINAL_RESERVE_S"] == "3600"
    assert document["validation_scope"] == "capsules"
    assert document["origin"]["resume"] is False and document["origin"]["seed_submission"] is None
    assert "--resume" not in launch.argv and "--seed-submission" not in launch.argv
    assert launch.argv[launch.argv.index("--validation-scope") + 1] == "capsules"
    assert set(document["cpu"]["author"]["cpus"]) == {4, 5}
    assert set(launch.manager_cpus) == {0, 1, 2, 3, *range(24, 32)}
    assert document["external_validation"]["start_after"] == "capsule_owner_terminal"


@pytest.fixture
def world(tmp_path, monkeypatch):
    task = tmp_path / "task"
    repo = task / "merlin"
    repo.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    prep = task / ".tmp/prep"
    # Real config renderer with synthetic selected files; no fake qualification.
    launch = L.build(
        L.load(HERE / "el4-successor.yaml"),
        task_root=task,
        prep=prep,
        run_id="fresh",
        environ={"HOME": str(tmp_path / "home")},
    )
    for label, (name, _) in launch.pins.items():
        if label == "errata":
            continue
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture " + label)
    for name in ("descriptor", "corpus_seal", "bundle_manifest"):
        path = launch.path(launch.document["inputs"][name])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "compiler_tool_profile": {
                        "schema": "fixture",
                        "sha256": "a" * 64,
                        "common_substrate": "merlin.common_compiler_substrate.v6",
                        "shared_scheduling_library": {"schema": "synthetic_sdk"},
                    }
                }
            )
        )
    launch.path(launch.document["runtime_pins"]["shared_sdk_manifest"]["path"]).write_text(
        json.dumps({"schema": "synthetic_sdk"})
    )
    launch.path(launch.document["inputs"]["corpus_seal"]).write_text(
        json.dumps({"review": {"reviewed_by": "synthetic fixture", "note": "No corpus authority."}})
    )
    # Files under the selected repo are committed; external metadata remains outside it.
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"],
        cwd=repo,
        check=True,
    )
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return task, repo, prep, launch


def test_binding_reopens_exact_inputs_without_a_candidate_or_grading(world):
    task, repo, prep, launch = world
    output = task / ".tmp/bound"
    result = module("prepare_successor").bind(
        HERE / "el4-successor.yaml", task_root=task, prep=prep, run_id="fresh", output=output
    )
    assert result["controller_started"] is False and result["candidate_qualified"] is False
    assert result["candidate_required_for_preparation"] is False
    assert result["sdk_admission"] == "pending_native_session"
    document = yaml.safe_load((output / "config.yaml").read_text())
    assert document["variables"]["SOURCE_REVISION"] == result["source_revision"]
    assert all(row["sha256"] != "0" * 64 for row in document["runtime_pins"].values())
    assert json.loads((output / "rendered/launch-argv.json").read_text())["origin"]["seed_submission"] is None
    assert not launch.path(launch.variables["RUN"]).exists()
    with pytest.raises(ValueError, match="fresh"):
        module("prepare_successor").bind(
            HERE / "el4-successor.yaml", task_root=task, prep=prep, run_id="fresh", output=output
        )


def test_binding_rejects_dirty_source_and_missing_runtime(world):
    task, repo, prep, launch = world
    changed = repo / "new.py"
    changed.write_text("changed")
    with pytest.raises(ValueError, match="commit"):
        module("prepare_successor").bind(
            HERE / "el4-successor.yaml", task_root=task, prep=prep, run_id="fresh", output=task / ".tmp/dirty"
        )
    changed.unlink()
    launch.path(launch.document["runtime_pins"]["spike"]["path"]).unlink()
    with pytest.raises(OSError):
        module("prepare_successor").bind(
            HERE / "el4-successor.yaml", task_root=task, prep=prep, run_id="fresh", output=task / ".tmp/absent"
        )
    assert not (task / ".tmp/absent").exists()


def test_binding_refuses_a_different_selected_sdk_export(world):
    task, repo, prep, launch = world
    sdk_manifest = launch.path(launch.document["runtime_pins"]["shared_sdk_manifest"]["path"])
    sdk_manifest.write_text(json.dumps({"schema": "a_different_sdk"}))
    output = task / ".tmp/wrong-sdk"
    with pytest.raises(ValueError, match="differs from the corpus"):
        module("prepare_successor").bind(
            HERE / "el4-successor.yaml", task_root=task, prep=prep, run_id="fresh", output=output
        )
    assert not output.exists()
