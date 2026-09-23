"""Actual task writes and late tool observation, without providers or hardware."""

from dataclasses import FrozenInstanceError, replace

import pytest
from merlin_experiments.phase1 import source_inputs, treatments
from merlin_experiments.phase1 import task_staging as TS
from merlin_experiments.phase1.context import InvocationContext


def config(root, target="synthetic"):
    root.mkdir(parents=True)
    descriptor = root / "explicit-name.yaml"
    descriptor.write_text(f"target: {target}\n")
    context = InvocationContext(root, descriptor, root, target, root / "runs", root / "reports", root / "bundles", ())
    bundle = root / "chosen-bundle"
    bundle.mkdir()
    return TS.TaskStagingConfig(context, "raw_baseline_hwbringup_v0", bundle, "realistic")


def stage(conf, arm="raw_baseline"):
    ws, run = conf.context.repo / "workspace", conf.context.repo / "run"
    ws.mkdir(exist_ok=True)
    run.mkdir(exist_ok=True)
    TS.callbacks(conf).stage_task(
        arm,
        ws,
        run,
        sandbox="none",
        task_scope={"required_public_dev_capsules": 7, "held_out_capsules": 3},
        policy_root=None,
    )
    assert (ws / "TASK.md").read_bytes() == (run / "TASK.md").read_bytes()
    return ws


@pytest.mark.parametrize("arm", ["raw_baseline", "merlin_assisted"])
@pytest.mark.parametrize("task_override", [False, True])
def test_actual_authored_task_explicit_bundle_and_descriptor(tmp_path, monkeypatch, arm, task_override):
    conf = config(tmp_path / "operator-root")
    (conf.context.experiment / "task").mkdir()
    (conf.context.experiment / "task/TASK_realistic.md").write_text("operator task\n")
    if task_override:
        tasks = tmp_path / "selected-tasks"
        tasks.mkdir()
        (tasks / "TASK_realistic.md").write_text("operator task\n")
        (conf.context.experiment / "task/TASK_realistic.md").write_text("wrong sibling task\n")
        descriptor = conf.context.descriptor
        descriptor.write_text(descriptor.read_text() + f"task_root: {tasks}\n")
    (conf.bundle_dir / "STARTER_PROMPT.md").write_text("selected starter")
    (conf.bundle_dir / "TASK_ADDENDUM.md").write_text("selected addendum")
    for doc in TS.RI.MERLIN_WS_DOCS:
        (conf.bundle_dir / doc).write_text(f"selected {doc}")
    monkeypatch.setenv("MERLIN_TARGET_EXPERIMENT", "/absent/ambient.yaml")
    ws = stage(conf, arm)
    text = (ws / "TASK.md").read_text()
    assert text.startswith(
        "operator task\n\n\n---\n\n# Starter plan / approach for THIS arm (read this)\n\nselected starter"
    )
    assert "Required public/dev capsules: **7**" in text
    assert "Held-out capsules: **3**" in text
    assert "unsandboxed diagnostic override" in text
    assert ("selected TASK_ADDENDUM.md" in text) == (arm == "merlin_assisted")
    for doc in TS.RI.MERLIN_WS_DOCS:
        assert (ws / doc).exists() == (arm == "merlin_assisted")


def test_invocations_capture_language_and_do_not_share_bundle_warning_state(tmp_path, monkeypatch):
    first = config(tmp_path / "first", "first_target")
    second = config(tmp_path / "second", "second_target")
    for conf in (first, second):
        (conf.context.experiment / "task").mkdir()
        (conf.context.experiment / "task/TASK_realistic.md").write_text(conf.context.target)
    first = replace(first, language="cpp")
    monkeypatch.setenv("PILOT_LANG", "python")
    assert "first_target-opt" in (stage(first) / "TASK.md").read_text()
    assert "Language mandate" not in (stage(second) / "TASK.md").read_text()
    with pytest.raises(FrozenInstanceError):
        first.language = "python"


def test_tools_are_observed_after_callback_creation_and_again_after_changes(tmp_path):
    conf = config(tmp_path / "operator")
    callback = TS.callbacks(replace(conf, add_tools=("isa_query",), drop_tools=("old",))).resolved_tools
    tools = conf.bundle_dir / "tools.txt"
    tools.write_text("old\nselfcheck\n")
    assert callback() == ("selfcheck", "isa_query")
    tools.write_text("sim_job\n")
    assert callback() == ("sim_job", "isa_query")


def test_callbacks_are_actual_inventoried_source_functions(tmp_path):
    conf = config(tmp_path / "operator")
    owners = source_inputs.record(repo=conf.context.repo, entrypoint=conf.context.repo / "external.py")
    selected = TS.callbacks(conf)
    for function in (selected.stage_task, selected.resolved_tools):
        identity = treatments.callback_reference(function, owners, label="task staging")
        assert identity["module"] == TS.__name__
        assert identity["source_input"] == "phase1:source:task_staging.py"


@pytest.mark.parametrize("experiment", ["realistic", "full"])
@pytest.mark.parametrize("arm", ["raw_baseline", "merlin_assisted"])
def test_generated_task_composition_keeps_starter_and_addendum_rules(tmp_path, monkeypatch, experiment, arm):
    from merlin.targetgen import capsule_runner, generate_prompt

    conf = replace(config(tmp_path / "operator"), experiment=experiment)
    calls = []

    def render(te, manifest, mode, selected_arm, granted_tools):
        calls.append(
            (
                te.descriptor_path if hasattr(te, "descriptor_path") else te.target,
                manifest,
                mode,
                selected_arm,
                granted_tools,
            )
        )
        return "generated task"

    monkeypatch.setattr(generate_prompt, "render_prompt", render)
    monkeypatch.setattr(TS, "load_capability_manifest", lambda target: {"target": target})
    monkeypatch.setattr(capsule_runner, "qa_loop_adapters", lambda *args, **kwargs: {"L1": object()})
    monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *args, **kwargs: {"L2": object()})
    (conf.bundle_dir / "allowed_files.txt").write_text("merlin/tool.py\nother/file\n")
    (conf.bundle_dir / "STARTER_PROMPT.md").write_text("must not duplicate generated starter")
    (conf.bundle_dir / "TASK_ADDENDUM.md").write_text("selected addendum")
    ws = stage(conf, arm)
    text = (ws / "TASK.md").read_text()
    assert calls[0][1:] == ({"target": "synthetic"}, experiment, arm, {"merlin/tool.py"})
    assert text.startswith("generated task")
    assert "must not duplicate" not in text
    assert ("selected addendum" in text) == (arm == "merlin_assisted")
    assert ("## Grading tiers" in text) == (experiment == "full")
    if experiment == "full":
        assert "fast RTL oracle tier (L1)" in text
        assert "checkpoint (L2)" in text
