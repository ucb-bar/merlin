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


def test_observer_qualification_is_bound_to_sources_and_control_bytes(study, tmp_path):
    source = tmp_path / "adapter.py"
    source.write_text("unit test adapter")
    tests = {}
    for name in (
        "startup_exclusion",
        "same_elf_uncapped_agreement",
        "inflight_tightening_abort",
        "remote_cancellation",
    ):
        receipt = tmp_path / (name + ".json")
        receipt.write_text(json.dumps(dict(test=name, status="pass")))
        tests[name] = study.identity(receipt)
    proof = tmp_path / "qualification.json"
    proof.write_text(
        json.dumps(
            dict(
                cycle_limit_scope="active_measured",
                limit_refresh_scope="in_flight",
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
                observer_qualification=study.identity(proof),
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
    args = dict(deadline=1000, clock_hz=100, pool=pool, cap=12)
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
    args = dict(deadline=1000, clock_hz=100, pool=pool, cap=8)
    ledger = tmp_path / "ledger.json"
    study.reserve_campaigns(ledger, [row], tmp_path, **args)
    (tmp_path / "program.json").write_text("different")
    with pytest.raises(ValueError, match="identity"):
        study.reserve_campaigns(ledger, [row], tmp_path, **args)
