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

    monkeypatch.delattr(compile_only, "SCALE_PROBE_FLAG", raising=False)
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


def test_selected_original_public_resource_gate_is_visible_and_exact():
    from merlin.targetgen.public_codegen_budget import validate_policy

    policy = validate_policy({"required": True, "timeout_s": 91, "capsules": ["public_a", "public_b"],
                              "device_ir_budget": {"mode": "enforce", "max_code_bytes": 12345,
                                                   "max_inline_asm_sites": 37}})
    text = prompt.render_resource_block(SimpleNamespace(), environ={}, manager_cpus=(7,), grade_workers=2,
                                        public_codegen_policy=policy)
    assert "max_code_bytes=12345" in text and "max_inline_asm_sites=37" in text
    assert "--scale-probe --capsules public_a,public_b --timeout 91" in text
    assert "mandatory for completion: **True**" in text and "requests cannot relax" in text


def test_short_full_task_discloses_selected_public_resource_limits(tmp_path):
    from merlin.targetgen.public_codegen_budget import validate_policy

    ws, run = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    header = "# Phase 1: Gemmini compiler for complete models\n\nOriginal compact task.\n"
    (ws / "TASK.md").write_text(header)
    policy = validate_policy({"required": True, "timeout_s": 120, "capsules": ["public_a"],
                              "device_ir_budget": {"mode": "enforce", "max_code_bytes": 1048576}})
    prompt.append_resource_guidance(ws, run, object(), public_codegen_policy=policy)
    text = (ws / "TASK.md").read_text()
    assert text.startswith(header) and "max_code_bytes=1048576" in text
    assert "--capsules public_a --timeout 120" in text and "mandatory for completion: **True**" in text
    assert "qa/public_codegen_resources.json" in text
    assert "Resource allocation and scheduling" not in text
    assert (run / "TASK.md").read_text() == text
    assert len(text.encode()) < 12 * 1024


@pytest.mark.parametrize("joint,size,accepted", [(False, 30000, False), (True, 30000, True),
                                               (True, 66000, False)])
def test_joint_task_has_bounded_extended_budget_and_native_scheduling(tmp_path, monkeypatch,
                                                                    joint, size, accepted):
    from merlin_experiments.phase1 import joint_control, task_staging

    monkeypatch.setattr(joint_control, "selected", lambda: {} if joint else None)
    verified = []
    monkeypatch.setattr(task_staging, "verify_shared_pass_prompt", lambda *args: verified.append(args))
    monkeypatch.setenv("MERLIN_COMPUTE_CPUS", "9,10")
    monkeypatch.setenv("MERLIN_COMPUTE_CPU_SLOTS", str(tmp_path / "slots"))
    monkeypatch.setenv("MERLIN_CODEX_CPUS", "9,10")
    config = task_staging.TaskStagingConfig(None, "test", tmp_path, "full")
    ws, run = tmp_path / "ws", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    (ws / "TASK.md").write_text("# Phase 1: Gemmini compiler for complete models\n" + "x" * size)
    options = SimpleNamespace(sim_max_jobs=2, codex_max_subagents=16)
    if not accepted:
        with pytest.raises(RuntimeError, match="exceeds"):
            prompt.append_resource_guidance(ws, run, options, task_staging_config=config)
        return
    prompt.append_resource_guidance(ws, run, options, task_staging_config=config)
    text = (ws / "TASK.md").read_text()
    assert len(verified) == 1
    assert "shared compute affinity: `9,10`" in text
    assert "priority 1" in text and "not a selfcheck priority flag" in text
    assert "two-job limits still apply" in text
    assert (run / "TASK.md").read_text() == text


@pytest.mark.parametrize("budget", [-1, 121, True])
def test_rendering_cannot_invent_an_invalid_selected_build_budget(budget):
    with pytest.raises(ValueError, match="selected 0..120-second budget"):
        prompt.render_resource_block(SimpleNamespace(public_object_build_budget_s=budget), environ={},
                                     manager_cpus=(7,), grade_workers=2)


def test_full_compiler_joint_resources_do_not_require_shared_pass_staging(tmp_path, monkeypatch):
    from merlin_experiments.phase1 import joint_control as joint
    from merlin_experiments.phase1 import full_compiler_workspace as full

    policy = {'duration_s': 18000, 'drain_s': 300, 'final_model_s': 7200,
              'compiler_execution_mode': 'snapshot_external_v1',
              'gsim': {'max_jobs': 20, 'per_job_s': 1200},
              'gsim_correctness': {'capsules': ['public_case'], 'max_jobs': 20, 'per_job_s': 1200},
              'operator_measurement_policy': {'initial_diagnostics_max': 1}}
    monkeypatch.setattr(joint, 'selected', lambda: {'policy': policy})
    monkeypatch.setattr(full, 'selected', lambda: object())
    monkeypatch.setenv('MERLIN_COMPUTE_CPUS', '9,10')
    monkeypatch.setenv('MERLIN_COMPUTE_CPU_SLOTS', str(tmp_path / 'slots'))
    monkeypatch.setenv('MERLIN_CODEX_CPUS', '9,10')
    ws, run = tmp_path / 'workspace', tmp_path / 'run'
    ws.mkdir(); run.mkdir()
    task = ws / 'TASK.md'
    task.write_text('# Phase 1: Gemmini compiler for complete models\n')
    options = SimpleNamespace(compiler_access='full_compiler', compiler_execution='snapshot_external_v1',
                              sim_max_jobs=20, codex_max_subagents=16)
    prompt.append_resource_guidance(ws, run, options)
    assert 'calibration witnesses 20; required correctness 20' in task.read_text()
    assert 'two-job' not in task.read_text()
    assert task.read_bytes() == (run / 'TASK.md').read_bytes()
    text = joint.prompt_block()
    assert 'initial diagnostics 1' in text
    assert 'declared continuation retains' in text
    assert 'there is no final-model-window waiting requirement' in text
    assert 'mandatory pass order' in text and 'are editable' in text
    assert 'Shared library bytes are immutable' not in text
    # The selected-library treatment still needs its original verified staging.
    options.compiler_access = 'selected_library'
    with pytest.raises(ValueError, match='requires verified task staging'):
        prompt.append_resource_guidance(ws, run, options)
