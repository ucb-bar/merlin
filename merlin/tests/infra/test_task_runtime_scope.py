"""The task served to an agent records facts from this launch, not historical prose."""

from __future__ import annotations

import sys

import pytest
from merlin_experiments.phase1 import task_staging as TS

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.target_experiment import load_target_experiment

sys.path.insert(0, str(repo_root() / "merlin/experiments/capsule_bench/harness"))


def _loop():
    import run_baseline_qa_loop as loop

    return loop


def _gemmini():
    return load_target_experiment(
        repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/target_experiment.yaml"
    )


def test_runtime_scope_matches_descriptor_discovery_not_historical_literals():
    loop = _loop()
    te = _gemmini()
    contract = repo_root() / "merlin/contract"
    public = discover_capsules(te.graded_roots(), labels={"public", "dev"}, contract=contract)
    public = [cap for cap in public if cap.get("name") not in set(te.graded_exclude)]
    hidden = discover_capsules(te.hidden_roots(), labels={"hidden"}, contract=contract)

    scope = TS.task_runtime_scope(te, "bwrap", context=loop.C.CONTEXT)

    assert scope["required_public_dev_capsules"] == len(public) != 20
    assert scope["held_out_capsules"] == len(hidden) != 5
    assert scope["sandbox"] == "bwrap"


@pytest.mark.parametrize("experiment", ["full", "realistic"])
def test_every_task_shape_gets_the_authoritative_runtime_block(experiment, tmp_path, monkeypatch):
    loop = _loop()
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()

    TS.build_task(
        "raw_baseline",
        ws,
        run_dir,
        sandbox="bwrap",
        config=TS.TaskStagingConfig(
            loop.C.CONTEXT,
            loop.RX.ARM_BUNDLE["raw_baseline"],
            loop.C.BUNDLES / loop.RX.ARM_BUNDLE["raw_baseline"],
            experiment,
        ),
    )

    task = (ws / "TASK.md").read_text()
    scope = TS.task_runtime_scope(_gemmini(), "bwrap", context=loop.C.CONTEXT)
    assert "Runtime scope (generated for this launch; authoritative)" in task
    assert f"Required public/dev capsules: **{scope['required_public_dev_capsules']}**" in task
    assert f"Held-out capsules: **{scope['held_out_capsules']}**" in task
    assert "Active sandbox: **`bwrap`**" in task
    assert "20 public capsules" not in task
    assert (run_dir / "TASK.md").read_text() == task


def test_unsandboxed_task_is_explicitly_untrusted():
    block = TS.task_runtime_scope_block(_gemmini(), "none", context=_loop().C.CONTEXT)
    assert "Active sandbox: **`none`**" in block
    assert "unsandboxed diagnostic override" in block
    assert "cannot support a trusted isolation claim" in block


def test_launch_agent_refuses_to_build_task_after_environment_setup(monkeypatch, tmp_path):
    from merlin_experiments.phase1.providers import execution as EX

    loop = _loop()
    monkeypatch.setattr(
        TS, "build_task", lambda *_args, **_kwargs: pytest.fail("launch must not regenerate the sealed task")
    )
    with pytest.raises(RuntimeError, match="sealed task is missing"):
        EX.launch(
            tmp_path / "ws",
            tmp_path / "run",
            "dummy",
            "low",
            "none",
            {},
            0,
            1,
            config=EX.ExecutionConfig(loop.C.CONTEXT, EX.ProviderConfig(), lambda: (), tmp_path / "timing"),
        )


def test_served_rtlchecks_task_and_tool_doc_have_no_stale_launch_claims(tmp_path, monkeypatch):
    loop = _loop()
    ws, run_dir = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run_dir.mkdir()

    TS.build_task(
        "merlin_assisted",
        ws,
        run_dir,
        sandbox="bwrap",
        config=TS.TaskStagingConfig(
            loop.C.CONTEXT,
            "merlin_assisted_rtlchecks_public_v0",
            loop.C.BUNDLES / "merlin_assisted_rtlchecks_public_v0",
            "full",
        ),
    )

    task = (ws / "TASK.md").read_text()
    tooling = (ws / "ALLOWED_MERLIN_TOOLS.md").read_text()
    served = task + "\n" + tooling
    scope = TS.task_runtime_scope(_gemmini(), "bwrap", context=loop.C.CONTEXT)
    assert f"Required public/dev capsules: **{scope['required_public_dev_capsules']}**" in task
    assert "Active sandbox: **`bwrap`**" in task
    assert "20 public capsules" not in served and "5 hidden" not in served
    assert "bwrap crashes" not in served and "the mode both arms run" not in served
    assert "does **not** select a sandbox" in tooling
    assert "environment.yaml" in tooling
