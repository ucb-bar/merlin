"""Served resource guidance reflects runtime limits without claiming active jobs."""

from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import resource_prompt as prompt


def test_shared_runtime_limits_and_scheduling_are_in_the_served_task(tmp_path, monkeypatch):
    environment = {
        "MERLIN_COMPUTE_CPUS": "3,4,5,6",
        "MERLIN_COMPUTE_CPU_SLOTS": str(tmp_path / "slots"),
        "MERLIN_CODEX_CPUS": "1,2",
    }
    options = SimpleNamespace(sim_max_jobs=4, codex_max_subagents=1)
    text = prompt.render_resource_block(options, environ=environment, manager_cpus=(0,), grade_workers=9)
    assert "worker ceiling: **4**" in text and "`--workers 4`" in text
    assert "CPU pool: `1,2`" in text and "CPU pool: `3,4,5,6`" in text
    assert "subagent ceiling: **1**" in text
    assert "not active-session counts" in text
    assert "same-request operator gate evidence" in text
    assert "duplicate full sweep" in text and "liveness diagnostics" in text
    ws, run = tmp_path / "workspace", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    (ws / "TASK.md").write_text("treatment-authored task\n")
    monkeypatch.setattr(prompt, "render_resource_block", lambda _: text)
    prompt.append_resource_guidance(ws, run, options)
    assert (ws / "TASK.md").read_text().startswith("treatment-authored task\n")
    assert (ws / "TASK.md").read_bytes() == (run / "TASK.md").read_bytes()


def test_legacy_unselected_limits_remain_explicitly_unspecified():
    text = prompt.render_resource_block(SimpleNamespace(), environ={}, manager_cpus=(7,), grade_workers=2)
    assert "`--workers 2`" in text
    assert "Native concurrency is engine-derived" in text
    assert "provider's limit" in text
    assert "inherited allocation" in text
    assert "15" not in text and "32" not in text


def test_unknown_shared_selection_or_out_of_boundary_pool_refuses(tmp_path):
    with pytest.raises(ValueError):
        prompt.render_resource_block(
            SimpleNamespace(), environ={"MERLIN_COMPUTE_CPUS": "3,4"}, manager_cpus=(0,), grade_workers=2
        )
    boundary = SimpleNamespace(cpu_ids=(0, 1), verify=lambda: None)
    with pytest.raises(ValueError, match="verified resource boundary"):
        prompt.render_resource_block(
            SimpleNamespace(), environ={"MERLIN_CODEX_CPUS": "2"}, manager_cpus=(0,), grade_workers=2, boundary=boundary
        )


def test_selected_build_budget_and_public_commands_are_grounded_in_actual_tools(monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability, compile_only

    options = SimpleNamespace(public_object_build_budget_s=12)
    text = prompt.render_resource_block(options, environ={}, manager_cpus=(7,), grade_workers=2)
    assert "12 seconds per sample" in text
    assert f"{codegen_scalability._MAX_BUILD_PROBE_BYTES:,} bytes" in text
    assert "--compile-only --capsules PUBLIC_ID_1,PUBLIC_ID_2 --timeout 120" in text
    assert "--scale-probe" not in text
    monkeypatch.setattr(compile_only, "SCALE_PROBE_FLAG", "--scale-probe", raising=False)
    text = prompt.render_resource_block(options, environ={}, manager_cpus=(7,), grade_workers=2)
    assert "--compile-only --scale-probe --capsules PUBLIC_ID_1 --timeout 120" in text
    assert "No captured tensor values" in text and "required trace gates" in text


@pytest.mark.parametrize("budget", [-1, 121, True])
def test_rendering_cannot_invent_an_invalid_selected_build_budget(budget):
    with pytest.raises(ValueError, match="selected 0..120-second budget"):
        prompt.render_resource_block(SimpleNamespace(public_object_build_budget_s=budget), environ={},
                                     manager_cpus=(7,), grade_workers=2)
