"""Actual task writes and late tool observation, without providers or hardware."""

from dataclasses import FrozenInstanceError, replace

import pytest
from merlin_experiments.phase1 import source_inputs, treatments
from merlin_experiments.phase1 import task_staging as TS
from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.levels import LEVELS

from merlin.targetgen.generalization_prompt import GENERAL_COMPILER_CONTRACT_V3


def test_gemmini_full_model_task_is_short_and_allows_edits_during_checks(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import resource_prompt

    conf = config(tmp_path / "operator", "gemmini")
    conf = replace(conf, experiment="full", bundle_id="merlin_assisted_rtlchecks_hwbringup_v0")
    (conf.bundle_dir / "tools.txt").write_text("xdsl_kit\ncca_spine\n")
    monkeypatch.setattr(TS, "load_target_experiment", lambda _: object())
    ws, run = conf.context.repo / "workspace", conf.context.repo / "run"
    ws.mkdir()
    run.mkdir()
    TS.build_task(
        "merlin_assisted",
        ws,
        run,
        config=conf,
        task_scope={"required_public_dev_capsules": 120, "held_out_capsules": 20},
    )
    resource_prompt.append_resource_guidance(ws, run, object())
    task = (ws / "TASK.md").read_text()
    assert len(task.encode()) <= 12 * 1024
    assert all(name in task for name in ("ResNet50", "SmolVLA", "TinyLlama", "numerics.approximations"))
    assert "task/PHASE2_TARGET.md" in task
    assert "public capsule L3 gSIM qualification" in task
    assert "per-kernel model gSIM replay is not requested" in task
    assert "--engine spike" in task and "--engine gsim" not in task
    assert "You may continue editing while a self-check or grade runs" in task
    # The advertised focused check is the certifying one; a single-engine screen is named as such.
    assert "`python agent_selfcheck.py --submission submission --capsules A,B`" in task
    assert "certified pass/fail per capsule" in task and "cannot certify" in task
    assert task == (run / "TASK.md").read_text()


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


@pytest.mark.parametrize("arm", [level["bundle_arm"] for level in LEVELS])
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
    assert text.count(GENERAL_COMPILER_CONTRACT_V3) == 1
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


@pytest.mark.parametrize(
    "level",
    [level for level in LEVELS if level["id"] not in {"EL5", "EL6"}],
    ids=[level["id"] for level in LEVELS if level["id"] not in {"EL5", "EL6"}],
)
def test_actual_tool_inventory_follows_selected_level_and_explicit_ablation(tmp_path, level):
    from merlin.targetgen.generate_bundles import _ALL_ARMS

    conf = config(tmp_path / "operator")
    conf = replace(conf, bundle_id=_ALL_ARMS[level["bundle_arm"]] + "_synthetic")
    names = TS._TR.ARM_TOOLS[level["bundle_arm"]]
    (conf.bundle_dir / "tools.txt").write_text("\n".join(names))
    text = TS.task_tool_inventory_block(conf)
    for name in names:
        assert f"- `{name}`:" in text
        assert TS._TR.spec(name).blurb in text
    assert "agent_selfcheck.py" in text and "await_verdict.py" in text
    # The selected file and add/drop arguments override the label. This guards
    # against advertising extra capabilities merely because an EL is named.
    dropped = replace(conf, drop_tools=names)
    assert "No additional treatment tools" in TS.task_tool_inventory_block(dropped)
    (conf.bundle_dir / "tools.txt").write_text("")
    assert "No additional treatment tools" in TS.task_tool_inventory_block(conf)


def test_callbacks_are_actual_inventoried_source_functions(tmp_path):
    conf = config(tmp_path / "operator")
    owners = source_inputs.record(repo=conf.context.repo, entrypoint=conf.context.repo / "external.py")
    contract_source = owners["inputs"]["phase1:startup:general_compiler_contract"]
    assert contract_source["sha256"] == source_inputs.fingerprint(contract_source["path"])
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
    assert text.count(GENERAL_COMPILER_CONTRACT_V3) == 1
    assert "must not duplicate" not in text
    assert ("selected addendum" in text) == (arm == "merlin_assisted")
    assert ("## Grading tiers" in text) == (experiment == "full")
    if experiment == "full":
        assert "fast RTL oracle tier (L1)" in text
        assert "checkpoint (L2)" in text
