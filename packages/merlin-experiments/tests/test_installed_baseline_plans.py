"""Installed baseline command and operator-input identity without executing providers."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from merlin_experiments import SpecError, load_spec
from merlin_experiments.runner import preflight, resolve_plan, resume, run


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path))
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    for name in ("task", "bundle", "corpus/isa", "merlin/contract", "merlin/schemas"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "task/TASK_realistic.md").write_text("Authored task\n")
    (tmp_path / "bundle/input_bundle_manifest.yaml").write_text(
        "bundle_id: raw_baseline_fixture\nallowed: []\nhost_inputs: []\n"
    )
    (tmp_path / "timing.json").write_text(json.dumps({"target": "fixture"}))
    (tmp_path / "target.yaml").write_text("target: fixture\ncapsule_corpus: corpus/isa\n")
    config = dict(
        descriptor="target.yaml",
        bundle="raw_baseline_fixture",
        bundle_manifest="bundle/input_bundle_manifest.yaml",
        oracle_timing="timing.json",
        arm="raw_baseline",
        model="fixture",
        effort="high",
        max_wall_s=60,
        round_timeout=60,
    )
    path = tmp_path / "experiment.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                id="installed-baseline",
                target="fixture",
                phases={"1": dict(adapter="capsule_bench", config=config)},
            )
        )
    )
    return path


def select_treatment(path, treatment):
    document = yaml.safe_load(path.read_text())
    config = document["phases"]["1"]["config"]
    config["treatment"] = treatment
    config["arm"] = "merlin_assisted" if treatment == "rtlchecks" else "raw_baseline"
    path.write_text(yaml.safe_dump(document))


@pytest.mark.parametrize("treatment", ["baseline", "rtlchecks"])
def test_baseline_resolves_installed_entrypoint_without_native_tree(baseline, treatment):
    select_treatment(baseline, treatment)
    plan = resolve_plan(load_spec(baseline), run_dir=baseline.parent / "run")
    command = plan["phases"]["1"]
    assert command["argv"][1:3] == ["-m", "merlin_experiments.phase1"]
    assert command["module"] == "merlin_experiments.phase1"
    assert Path(command["entrypoint"]).name == "__main__.py"
    assert "--bundle-manifest" in command["argv"]
    assert "--descriptor" in command["argv"]
    assert "--repo" in command["argv"]
    assert not any(key.startswith("phase1:native:") for key in plan["input_paths"])
    if treatment == "rtlchecks":
        assert command["argv"][command["argv"].index("--treatment") + 1] == treatment


def test_optional_timing_appearance_refuses_preflight(baseline):
    plan = resolve_plan(load_spec(baseline), run_dir=baseline.parent / "run")
    scripts = baseline.parent / "scripts"
    scripts.mkdir()
    (scripts / ".oracle_timing.fixture.json").write_text("{}")
    assert not preflight(plan)["configuration_ready"]


@pytest.mark.parametrize("appearance", [False, True])
@pytest.mark.parametrize("task_override", [False, True])
def test_absent_authored_tasks_are_bound_and_appearance_refuses_resume(
    baseline,
    monkeypatch,
    appearance,
    task_override,
):
    root = baseline.parent
    selected = root / "generated-prompt-resources"
    selected.mkdir()
    descriptor = root / "target.yaml"
    descriptor.write_text(descriptor.read_text() + "resources_root: generated-prompt-resources\n")
    tasks = selected / "task"
    if task_override:
        tasks = root / "separate-tasks"
        descriptor.write_text(descriptor.read_text() + "task_root: separate-tasks\n")
        (selected / "task").mkdir()
        (selected / "task/TASK_realistic.md").write_text("Wrong fallback prompt\n")
    plan = resolve_plan(load_spec(baseline), run_dir=root / "run")
    assert plan["phase1_operator_inputs"]["phase1:operator:task"] is None
    assert "phase1:operator:task" not in plan["input_paths"]
    assert preflight(plan)["configuration_ready"]
    launches = []
    popen = subprocess.Popen

    def harmless_process(argv, **kwargs):
        launches.append(argv)
        return popen([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", harmless_process)
    assert run(plan) == 7
    frozen = json.loads((root / "run/resolved-plan.json").read_text())
    assert frozen["phase1_operator_inputs"]["phase1:operator:task"] is None
    assert "phase1:operator:task" not in frozen["input_paths"]
    if appearance:
        tasks.mkdir()
        (tasks / "TASK_realistic.md").write_text("New authored prompt\n")
        with pytest.raises(SpecError, match="operator input membership"):
            resume(root / "run")
        assert len(launches) == 1
    else:
        assert resume(root / "run") == 7
        assert len(launches) == 2
        assert not tasks.exists()


@pytest.mark.parametrize("mutation", ["task", "environment", "timing", "root"])
@pytest.mark.parametrize("task_override", [False, True])
def test_explicit_resource_ownership_is_bound_in_frozen_plan(baseline, monkeypatch, mutation, task_override):
    root = baseline.parent
    resources = root / "authored"
    (resources / "task").mkdir(parents=True)
    (resources / "task/TASK_realistic.md").write_text("Selected task\n")
    (resources / "scripts").mkdir()
    descriptor = root / "target.yaml"
    descriptor.write_text(descriptor.read_text() + "resources_root: authored\n")
    tasks = resources / "task"
    if task_override:
        tasks = root / "separate-tasks"
        tasks.mkdir()
        (tasks / "TASK_realistic.md").write_text("Selected external task\n")
        descriptor.write_text(descriptor.read_text() + "task_root: separate-tasks\n")
    plan = resolve_plan(load_spec(baseline), run_dir=root / "run")
    assert plan["input_paths"]["phase1:operator:task"] == str(tasks)
    assert plan["phase1_operator_inputs"]["phase1:operator:environment"] is None
    launches = []
    popen = subprocess.Popen

    def harmless_process(argv, **kwargs):
        launches.append(argv)
        return popen([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", harmless_process)
    assert run(plan) == 7
    if mutation == "task":
        (tasks / "TASK_realistic.md").write_text("Changed selected task\n")
    elif mutation == "environment":
        (resources / "experiment.env").write_text("SELECTED=changed\n")
    elif mutation == "timing":
        (resources / "scripts/.oracle_timing.fixture.json").write_text("{}")
    else:
        descriptor.write_text(descriptor.read_text().replace("authored", "different-root"))
    with pytest.raises(SpecError):
        resume(root / "run")
    assert len(launches) == 1


def test_timing_alias_retargeting_refuses_even_with_identical_bytes(baseline):
    root = baseline.parent
    scripts = root / "scripts"
    scripts.mkdir()
    first, second = root / "first.json", root / "second.json"
    first.write_text("{}")
    second.write_text("{}")
    alias = scripts / ".oracle_timing.fixture.json"
    alias.symlink_to(first)
    plan = resolve_plan(load_spec(baseline), run_dir=root / "run")
    alias.unlink()
    alias.symlink_to(second)
    assert not preflight(plan)["configuration_ready"]


@pytest.mark.parametrize("treatment", ["baseline", "rtlchecks"])
def test_baseline_resume_keeps_frozen_installed_command(baseline, monkeypatch, treatment):
    select_treatment(baseline, treatment)
    root = baseline.parent
    calls = []

    class FakeProcess:
        pid = 2**30

        def __init__(self, argv, **kwargs):
            calls.append(list(argv))

        def wait(self):
            return 7

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    plan = resolve_plan(load_spec(baseline), run_dir=root / "run")
    assert run(plan) == 7
    frozen = (root / "run/resolved-plan.json").read_bytes()
    assert resume(root / "run") == 7
    assert calls[0] == plan["phases"]["1"]["argv"]
    assert calls[1] == calls[0] + ["--resume"]
    assert (root / "run/resolved-plan.json").read_bytes() == frozen


def test_historical_rtlchecks_resume_keeps_recorded_native_command(baseline, monkeypatch):
    select_treatment(baseline, "rtlchecks")
    root = baseline.parent
    plan = resolve_plan(load_spec(baseline), run_dir=root / "run")
    script = root / "historical_rtlchecks.py"
    script.write_text("# historical entrypoint; not executed\n")
    command = plan["phases"]["1"]
    command.update(module=None, entrypoint=str(script))
    command["argv"] = [sys.executable, str(script), "--schedule", "continuous"]
    plan["input_paths"]["phase1:entrypoint"] = str(script)
    calls = []

    class FakeProcess:
        pid = 2**30

        def __init__(self, argv, **kwargs):
            calls.append(list(argv))

        def wait(self):
            return 7

    monkeypatch.setattr(subprocess, "Popen", FakeProcess)
    assert run(plan) == 7
    frozen = (root / "run/resolved-plan.json").read_bytes()
    assert resume(root / "run") == 7
    assert calls == [command["argv"], command["argv"] + ["--resume"]]
    assert (root / "run/resolved-plan.json").read_bytes() == frozen


@pytest.mark.parametrize("mutation", ["task", "bundle", "grant", "timing", "optional_timing", "environment"])
def test_resume_refuses_operator_input_changes_before_launch(baseline, monkeypatch, mutation):
    root = baseline.parent
    grant = root / "grant.txt"
    grant.write_text("original")
    manifest = root / "bundle/input_bundle_manifest.yaml"
    manifest.write_text(manifest.read_text().replace("allowed: []", "allowed:\n- path: grant.txt"))
    plan = resolve_plan(load_spec(baseline), run_dir=root / "run")
    calls = []
    popen = subprocess.Popen

    def harmless_process(argv, **kwargs):
        calls.append(argv)
        return popen([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", harmless_process)
    assert run(plan) == 7
    paths = {
        "task": root / "task/TASK_realistic.md",
        "bundle": root / "bundle/tools.txt",
        "grant": grant,
        "timing": root / "timing.json",
        "optional_timing": root / "scripts/.oracle_timing.fixture.json",
        "environment": root / "experiment.env",
    }
    selected = paths[mutation]
    selected.parent.mkdir(exist_ok=True)
    selected.write_text("changed")
    with pytest.raises(SpecError, match="operator input membership|frozen input changed"):
        resume(root / "run")
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["module", "module_removed", "entrypoint", "argv", "importroot", "descriptor"])
def test_installed_command_substitution_refuses_preflight(baseline, mutation):
    plan = resolve_plan(load_spec(baseline), run_dir=baseline.parent / "run")
    command = plan["phases"]["1"]
    if mutation == "module":
        command["module"] = "other.module"
    elif mutation == "module_removed":
        command["module"] = None
    elif mutation == "entrypoint":
        command["entrypoint"] = str(baseline)
    elif mutation == "argv":
        command["argv"][2] = "other.module"
    elif mutation == "importroot":
        command["env"]["PYTHONPATH"] = str(baseline.parent)
    else:
        command["argv"][command["argv"].index("--descriptor") + 1] = str(baseline)
    assert not preflight(plan)["configuration_ready"]


@pytest.mark.parametrize("missing", ["bundle", "bundle_manifest", "oracle_timing"])
@pytest.mark.parametrize("treatment", ["baseline", "rtlchecks"])
def test_baseline_requires_explicit_inputs_without_native_fallback(baseline, missing, treatment):
    select_treatment(baseline, treatment)
    document = yaml.safe_load(baseline.read_text())
    del document["phases"]["1"]["config"][missing]
    baseline.write_text(yaml.safe_dump(document))
    with pytest.raises(SpecError, match="installed Phase 1 requires explicit inputs"):
        load_spec(baseline)


@pytest.mark.parametrize("treatment", ["baseline", "rtlchecks"])
@pytest.mark.parametrize("flag", ["--bundle", "--arm", "--treatment"])
@pytest.mark.parametrize("mutation", ["replace", "duplicate", "remove"])
def test_selected_phase1_policy_flags_are_bound_to_definition(baseline, treatment, flag, mutation):
    select_treatment(baseline, treatment)
    plan = resolve_plan(load_spec(baseline), run_dir=baseline.parent / "run")
    assert preflight(plan)["configuration_ready"]
    argv = plan["phases"]["1"]["argv"]
    # Older installed baseline commands omitted this flag and use the same default.
    if flag == "--treatment" and flag not in argv:
        assert treatment == "baseline"
        argv.extend([flag, treatment])
    index = argv.index(flag)
    if mutation == "remove":
        del argv[index : index + 2]
    elif mutation == "duplicate":
        argv.extend([flag, argv[index + 1]])
    else:
        replacements = {
            "--bundle": "different_bundle",
            "--arm": "raw_baseline" if treatment == "rtlchecks" else "merlin_assisted",
            "--treatment": "baseline" if treatment == "rtlchecks" else "rtlchecks",
        }
        argv[index + 1] = replacements[flag]
    allowed_default = treatment == "baseline" and flag == "--treatment" and mutation == "remove"
    assert preflight(plan)["configuration_ready"] is allowed_default


def test_older_baseline_without_treatment_selection_retains_default(baseline):
    document = yaml.safe_load(baseline.read_text())
    assert "treatment" not in document["phases"]["1"]["config"]
    plan = resolve_plan(load_spec(baseline), run_dir=baseline.parent / "run")
    argv = plan["phases"]["1"]["argv"]
    if "--treatment" in argv:
        index = argv.index("--treatment")
        assert argv[index + 1] == "baseline"
        del argv[index : index + 2]
    assert preflight(plan)["configuration_ready"]
