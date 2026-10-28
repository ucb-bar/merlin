"""Continuation must preserve the deadline and avoid duplicate paid campaigns."""

import importlib.util
import json

import pytest

from merlin.common.paths import repo_root


@pytest.fixture
def study(monkeypatch):
    folder = repo_root() / "examples/gemmini/comparisons/tvm"
    monkeypatch.syspath_prepend(str(folder))
    spec = importlib.util.spec_from_file_location("firesim_study_test", folder / "run_firesim_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unqualified_old_adapter_is_rejected_before_execution(study, tmp_path):
    pool = tmp_path / "pool.json"
    pool.write_text(json.dumps(dict(measured_limit_enforced=True, moving_limit_enforced=True)))
    with pytest.raises(ValueError, match="qualification"):
        study.qualification(pool)


def test_runner_qualification_is_bound_to_sources_and_control_bytes(study, tmp_path):
    source = tmp_path / "adapter.py"
    source.write_text("unit test adapter")
    tests = {}
    for name in (
        "startup_exclusion",
        "same_elf_repeat_agreement",
        "wrong_oracle_rejection",
        "wall_timeout",
        "remote_cancellation",
    ):
        receipt = tmp_path / (name + ".json")
        receipt.write_text(json.dumps(dict(test=name, status="pass")))
        tests[name] = study.identity(receipt)
    proof = tmp_path / "qualification.json"
    proof.write_text(
        json.dumps(
            dict(
                schema="gemmini_firesim_runner_qualification_v1",
                timeout_scope="wall_execution", cycle_scope="fixed_warm_model_run",
                deployment_id="unit-test",
                adapter_source_sha256=[study.identity(source)["sha256"]],
                tests=tests,
            )
        )
    )
    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            dict(
                deployment_id="unit-test",
                runner_qualification=study.identity(proof),
                adapter_sources=[study.identity(source)],
            )
        )
    )
    assert study.qualification(pool)["deployment_id"] == "unit-test"
    receipt.write_text(json.dumps(dict(test=name, status="fail")))
    with pytest.raises(ValueError, match="Changed"):
        study.qualification(pool)
    source.write_text("changed adapter")
    config = json.loads(pool.read_text())
    config["adapter_sources"] = [study.identity(source)]
    pool.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="different installed"):
        study.qualification(pool)


def test_shared_ledger_skips_reserved_campaign_and_rejects_deadline_reset(study, tmp_path):
    (tmp_path / "program.json").write_text("fixture program")
    (tmp_path / "fixture.npz").write_bytes(b"fixture bytes")
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    row = dict(id="example", program="program.json", fixture="fixture.npz", trials=8, workload_keys=["key"])
    args = dict(bootstrap_seconds=10, ceiling_seconds=60, deadline=1000, clock_hz=100, pool=pool, cap=12)
    ledger = tmp_path / "ledger.json"
    assert study.reserve_campaigns(ledger, [row], tmp_path, **args) == [row]
    assert study.reserve_campaigns(ledger, [row], tmp_path, **args) == []
    with pytest.raises(ValueError, match="deadline"):
        study.reserve_campaigns(ledger, [row], tmp_path, **dict(args, deadline=2000))
    with pytest.raises(ValueError, match="cap"):
        study.reserve_campaigns(ledger, [row], tmp_path, **dict(args, cap=24))
    with pytest.raises(ValueError, match="cap"):
        study.reserve_campaigns(ledger, [dict(row, id="second")], tmp_path, **args)
    assert list(json.loads(ledger.read_text())["campaigns"]) == ["example"]


def test_shared_ledger_rejects_mutated_program(study, tmp_path):
    (tmp_path / "program.json").write_text("original")
    (tmp_path / "fixture.npz").write_bytes(b"fixture")
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    row = dict(id="example", program="program.json", fixture="fixture.npz", trials=4)
    args = dict(bootstrap_seconds=10, ceiling_seconds=60, deadline=1000, clock_hz=100, pool=pool, cap=8)
    ledger = tmp_path / "ledger.json"
    study.reserve_campaigns(ledger, [row], tmp_path, **args)
    (tmp_path / "program.json").write_text("different")
    with pytest.raises(ValueError, match="identity"):
        study.reserve_campaigns(ledger, [row], tmp_path, **args)


@pytest.mark.parametrize("option,value", [
    ("graph_fusion_level", 0), ("graph_cse", True),
])
def test_changed_graph_plan_cannot_reuse_campaign(study, tmp_path, option, value):
    (tmp_path / "program.json").write_text("program")
    (tmp_path / "fixture.npz").write_bytes(b"fixture")
    pool = tmp_path / "pool.json"
    pool.write_text("{}")
    row = dict(id="example", program="program.json", fixture="fixture.npz", trials=4)
    args = dict(bootstrap_seconds=10, ceiling_seconds=60, deadline=1000, clock_hz=30000000, pool=pool, cap=8)
    ledger = tmp_path / "ledger.json"
    study.reserve_campaigns(ledger, [row], tmp_path, **args)
    with pytest.raises(ValueError, match="identity"):
        study.reserve_campaigns(ledger, [dict(row, **{option: value})], tmp_path, **args)


def test_graph_options_reach_inner_campaign(study, tmp_path):
    from types import SimpleNamespace

    args = SimpleNamespace(pool=tmp_path / "pool", toolchain=tmp_path / "tools", target_clock_hz=30000000, candidate_bootstrap_seconds=10,
                           candidate_timeout_ceiling_seconds=60)
    row = dict(program="p", fixture="f", trials=4, graph_fusion_level=0,
               graph_cse=True, prepack_weights=False, bias_preload=False)
    command = study.command(row, tmp_path, args, tmp_path / "output", 1000)
    assert command[command.index("--graph-fusion-level") + 1] == "0"
    assert "--cost-model-warmup" not in command
    assert command[command.index("--batch") + 1] == "1"
    assert {"--graph-cse", "--no-prepack-weights", "--no-bias-preload"}.issubset(command)


def test_continuation_preserves_finish_active_policy(study, tmp_path):
    (tmp_path / "p").write_text("program")
    (tmp_path / "f").write_text("fixture")
    pool = tmp_path / "pool"
    pool.write_text("{}")
    row = dict(id="example", program="p", fixture="f", trials=1)
    args = dict(bootstrap_seconds=10, ceiling_seconds=60, deadline=1000, clock_hz=30000000, pool=pool, cap=1,
                finish_active_runs=True)
    ledger = tmp_path / "ledger.json"
    study.reserve_campaigns(ledger, [row], tmp_path, **args)
    assert study.reserve_campaigns(ledger, [row], tmp_path, **args) == []
    with pytest.raises(ValueError, match="completion policy"):
        study.reserve_campaigns(ledger, [row], tmp_path, **dict(args, ceiling_seconds=120))


@pytest.mark.parametrize("option", ["prepack_weights", "bias_preload"])
def test_removed_graph_preparation_is_refused_before_launch(study, tmp_path, option):
    from types import SimpleNamespace

    row = dict(id="old", program="missing-program", fixture="missing-fixture", trials=1,
               **{option: True})
    with pytest.raises(ValueError, match="Removed graph preparation"):
        study.command(row, tmp_path, SimpleNamespace(), tmp_path / "output", 1000)
    assert not (tmp_path / "output").exists()


def test_default_study_uses_only_stock_graph_options(study):
    assert study.graph_options({}) == dict(graph_fusion_level=2, prepack_weights=False,
                                          bias_preload=False, graph_cse=False)


def test_continuation_rejects_obsolete_xgb_warmup(study, tmp_path):
    (tmp_path / "p").write_text("program")
    (tmp_path / "f").write_text("fixture")
    (tmp_path / "pool").write_text("{}")
    row = dict(id="one", program="p", fixture="f", trials=25)
    options = dict(deadline=100, clock_hz=30_000_000, pool=tmp_path / "pool", cap=25,
                   bootstrap_seconds=10, ceiling_seconds=60)
    study.reserve_campaigns(tmp_path / "ledger", [row], tmp_path, **options)
    with pytest.raises(ValueError, match="ReplayTrace"):
        study.reserve_campaigns(tmp_path / "ledger", [dict(row, cost_model_warmup=4)], tmp_path, **options)


@pytest.mark.parametrize("trials", [0, 26, 55, True])
def test_replay_total_budget_is_bounded(study, trials):
    with pytest.raises(ValueError, match="25"):
        study.replay_trials(dict(trials=trials))


def test_graph_variants_share_chunk_trial_budget(study, tmp_path):
    (tmp_path / "p").write_text("program")
    (tmp_path / "f").write_text("fixture")
    (tmp_path / "pool").write_text("{}")
    opts = dict(deadline=100, clock_hz=30_000_000, pool=tmp_path / "pool", cap=100,
                bootstrap_seconds=10, ceiling_seconds=60)
    row = dict(id="first", program="p", fixture="f", trials=20)
    study.reserve_campaigns(tmp_path / "ledger", [row], tmp_path, **opts)
    with pytest.raises(ValueError, match="across graph plans"):
        study.reserve_campaigns(tmp_path / "ledger", [dict(row, id="second", trials=6, graph_cse=True)], tmp_path, **opts)
    assert len(json.loads((tmp_path / "ledger").read_text())["campaigns"]) == 1
