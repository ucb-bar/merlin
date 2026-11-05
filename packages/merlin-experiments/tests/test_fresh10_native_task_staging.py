"""Fresh10 keeps the native contract before its capability disclosure is appended."""

import pytest
from merlin_experiments.phase1 import fresh10_disclosure as D
from merlin_experiments.phase1 import fresh10_profiles as P
from merlin_experiments.phase1 import task_staging as TS
from merlin_experiments.phase1.context import InvocationContext


def config(tmp_path, profile):
    repo = tmp_path / "repo"
    repo.mkdir()
    descriptor = repo / "experiment.yaml"
    descriptor.write_text("target: gemmini\n")
    bundle = repo / "bundle"
    bundle.mkdir()
    (bundle / "tools.txt").write_text("\n".join(P.TOOLS) + "\n")
    for doc in TS.RI.MERLIN_WS_DOCS:
        (bundle / doc).write_text("Legacy CCA and unrestricted pass-library instructions.\n")
    context = InvocationContext(
        repo, descriptor, repo, "gemmini", repo / "runs", repo / "reports", repo / "bundles", ()
    )
    return TS.TaskStagingConfig(context, "merlin_assisted_rtlchecks_public_v0", bundle, "full", author_profile=profile)


@pytest.mark.parametrize("profile", P.NAMES)
def test_actual_native_task_owner_keeps_full_model_and_cohort_contracts(profile, tmp_path, monkeypatch):
    from merlin_experiments.corpus import public_program_suite as programs
    from merlin_experiments.phase1 import full_compiler_workspace, joint_control
    from merlin_experiments.phase1.feedback import public_programs, public_scalability, snapshot_execution

    conf = config(tmp_path, profile)
    ws, run = conf.context.repo / "workspace", conf.context.repo / "run"
    ws.mkdir()
    run.mkdir()
    monkeypatch.setattr(TS, "load_target_experiment", lambda _: object())
    monkeypatch.setattr(joint_control, "selected", lambda: None)
    monkeypatch.setattr(snapshot_execution, "selected", lambda: None)
    monkeypatch.setattr(full_compiler_workspace, "selected", lambda: None)
    monkeypatch.setattr(programs, "selected", lambda _: ("selected-programs",))
    views = []
    monkeypatch.setattr(
        programs, "public_view", lambda selection, destination, **kw: views.append((selection, destination, kw))
    )
    scale = (
        "\n\n## Selected scalability obligations\n"
        "Source-bound placement and half-even saturation; loop_descriptor remains prohibited.\n"
    )
    program = (
        "\n\n## Selected connected-program obligations\nEvery selected public connected program remains required.\n"
    )
    campaign = (
        "\n\n## Selected campaign policy\n"
        "Continuous physical promotion and complete final validation remain required.\n"
    )
    monkeypatch.setattr(public_scalability, "prompt_block", lambda _: scale)
    monkeypatch.setattr(public_programs, "prompt_block", lambda _: program)
    monkeypatch.setattr(joint_control, "prompt_block", lambda: campaign)

    TS.callbacks(conf).stage_task(
        "merlin_assisted",
        ws,
        run,
        sandbox="bwrap",
        task_scope={"required_public_dev_capsules": 109, "held_out_capsules": 14},
        policy_root=None,
    )
    before = (ws / "TASK.md").read_text()
    assert before.startswith("# Phase 1: Gemmini compiler for complete models\n")
    assert "All six programs must compile and link" in before and "bit-exactly" in before
    assert all(
        name in before for name in ("ResNet50", "prefix_encode", "flow_denoise", "action_decode", "prefill", "decode")
    )
    assert all(block in before for block in (scale, program, campaign))
    assert len(views) == 1 and views[0][0] == "selected-programs"
    assert views[0][1] == ws / "public_program_sources"
    assert not any((ws / name).exists() for name in TS.RI.MERLIN_WS_DOCS)
    assert "## Selected fresh10 compiler capabilities" not in before

    selected = D.stage_task(ws, run, profile, object())
    task = (ws / "TASK.md").read_text()
    assert task.count("## Selected fresh10 compiler capabilities") == 1
    for heading in D.COMMON_CONTRACT_SECTIONS:
        assert task.count("## " + heading + "\n") == 1
    core = task.partition("## Selected fresh10 compiler capabilities")[0]
    assert all(term in core for term in ("source-bound placement", "half-even", "saturation", "W8A8"))
    assert all(block in task for block in (scale, program, campaign))
    assert "All six programs must compile and link" in task
    assert "bit-exactly" in task
    assert selected["common_core_bytes"] >= len(before.encode()) - 2
    assert task == (run / "TASK.md").read_text()


@pytest.mark.parametrize("profile", ("xs-el-v1/el1", "xs-el-v1/el2", "xs-el-v1/el3", "xs-el-v1/el4"))
def test_historical_construction_task_does_not_acquire_full_model_obligations(profile, tmp_path, monkeypatch):
    from merlin_experiments.corpus import public_program_suite as programs

    conf = config(tmp_path, profile)
    ws, run = conf.context.repo / "workspace", conf.context.repo / "run"
    ws.mkdir()
    run.mkdir()
    monkeypatch.setattr(
        TS, "full_model_task", lambda *a: pytest.fail("historical construction task must stay unchanged")
    )
    monkeypatch.setattr(
        programs, "selected", lambda *a: pytest.fail("historical construction task has no model cohort")
    )
    TS.callbacks(conf).stage_task("merlin_assisted", ws, run, sandbox="bwrap", task_scope=None, policy_root=None)
    body = (ws / "TASK.md").read_text()
    assert body.startswith("# Compiler construction ablation\n")
    assert "All six programs" not in body
    assert body == (run / "TASK.md").read_text()


@pytest.mark.parametrize("profile", P.NAMES)
def test_original_common_contract_sections_are_preserved_once(profile, tmp_path):
    ws, run = tmp_path / "workspace", tmp_path / "run"
    ws.mkdir()
    run.mkdir()
    (ws / "TASK.md").write_text("# Full native task\n\nOriginal common workload obligations.\n")
    D.stage_task(ws, run, profile, object())
    first = (ws / "TASK.md").read_text()
    # Exercise the already-present native delivery independently of disclosure:
    # a baseline TASK containing these original sections must not acquire copies.
    core = first.partition("## Selected fresh10 compiler capabilities")[0]
    (ws / "TASK.md").write_text(core)
    D.stage_task(ws, run, profile, object())
    task = (ws / "TASK.md").read_text()
    for heading in D.COMMON_CONTRACT_SECTIONS:
        assert task.count("## " + heading + "\n") == 1
    assert task.count("## Selected fresh10 compiler capabilities") == 1
    assert "source-bound placement" in task and "half-even" in task and "W8A8" in task
