"""Real process observations survive failed publication without becoming authority."""

import hashlib
import json
import os

import pytest
from merlin_experiments.capture_execution import runtime_store
from merlin_experiments.capture_execution import sealed_failure as F
from merlin_experiments.capture_execution import sealed_m2m as M


def _issue_fixture(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    selected.mkdir()
    run = tmp_path / "run"
    plan = {
        "schema": M.SCHEMA,
        "status": "plan_only",
        **{
            key: str(selected / key)
            for key in ("m2m_root", "workload_root", "worker", "venv", "schemas_root", "base", "merlin_root")
        },
        "dtype": "fp32",
        "recipe": None,
        "worker_sha256": "worker",
        "max_snapshot_bytes": 1000,
        "estimate_bytes": 0,
        "system_libs": [],
    }
    identity = {"sha256": "snapshot", "bytes": 0}
    monkeypatch.setattr(M, "prepare_plan", lambda **_: plan)
    monkeypatch.setattr(runtime_store, "snapshot_space_requirement", lambda *_: (0, None))
    monkeypatch.setattr(runtime_store, "link_runtime", lambda *_, **__: None)
    monkeypatch.setattr(M, "_bwrap_binary", lambda _: selected / "bwrap")
    monkeypatch.setattr(M, "_probe_sandbox", lambda _: None)
    monkeypatch.setattr(M, "_stage_source", lambda *_: None)
    monkeypatch.setattr(M, "_validate_snapshots", lambda *_, **__: None)
    monkeypatch.setattr(M, "_tree", lambda _: [])
    monkeypatch.setattr(M, "_verify_staged_selection", lambda *_, **__: {})
    monkeypatch.setattr(M, "_file_digest", lambda _: "worker")
    monkeypatch.setattr(M, "_snapshot_rows", lambda _: identity)
    monkeypatch.setattr(M, "_snapshot_tree", lambda _: identity)

    def execute(_bwrap, _runtime, _source, output, *_args, **_kwargs):
        (output / "result.bin").write_bytes(b"actual capture")
        return M._process_record(
            0, b"actual stdout\x00\xff", b"warning\n", failure_dir=output.parent / (output.name + "-failure")
        )

    def execute_beside(_bwrap, _runtime, _source, outputs, *_args, **_kwargs):
        return tuple(execute(None, None, None, output) for output in outputs)

    monkeypatch.setattr(M, "_execute", execute)
    monkeypatch.setattr(M, "_execute_beside", execute_beside)
    return run, plan


@pytest.mark.parametrize("paired", [False, True])
def test_ordinary_issue_retains_process_and_output_on_materialization_refusal(tmp_path, monkeypatch, paired):
    run, plan = _issue_fixture(tmp_path, monkeypatch)

    def refuse(*_):
        raise M.SealedM2MError("materialization refused after both executions")

    monkeypatch.setattr(M, "_materialized_v2", refuse)
    with pytest.raises(M.SealedM2MError, match="materialization refused"):
        M._issue(plan, run, replay_cpus=(frozenset({0}), frozenset({1})) if paired else None)
    assert not (run / "sealed_m2m_pending.json").exists()
    failed = json.loads((run / "sealed_capture_unverified_failure.json").read_bytes())
    assert failed["status"] == "unverified" and failed["phase0_admission"] == "not_granted"
    assert failed["source_closure_verified"] is False and failed["recovery_supported"] is False
    assert failed["plan_sha256"] == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    outputs = [run / "capture"] + ([run / failed["replay"]] if paired else [])
    for output in outputs:
        assert (output / "result.bin").read_bytes() == b"actual capture"
        [diagnostic] = run.glob(output.name + "-execution-*")
        assert (diagnostic / "stdout.bin").read_bytes() == b"actual stdout\x00\xff"
        process = json.loads((diagnostic / "process.json").read_bytes())
        assert process["returncode"] == 0 and process["status"] == "unverified"
        assert process["stdout_sha256"] == hashlib.sha256(b"actual stdout\x00\xff").hexdigest()
        assert os.stat(diagnostic / "stdout.bin").st_mode & 0o777 == 0o600
    # Diagnostics neither stand in for a pending receipt nor authorize attestation.
    with pytest.raises(M.SealedM2MError, match="pending M2M receipt is absent"):
        M.replay_verify(run)


@pytest.mark.parametrize("refuse_replay", [False, True])
def test_ordinary_issue_deletes_replay_only_after_successful_comparison(tmp_path, monkeypatch, refuse_replay):
    run, plan = _issue_fixture(tmp_path, monkeypatch)

    def pending(**_):
        path = run / "sealed_m2m_pending.json"
        path.write_text("{}")
        return path

    def compare(_run, *, bwrap_binary, _executed):
        process, replay = _executed
        assert process["returncode"] == 0 and (replay / "result.bin").exists()
        if refuse_replay:
            raise M.SealedM2MError("independent materialized bytes differ")
        return {"status": "verified_sandbox_replay"}

    monkeypatch.setattr(M, "_write_pending", pending)
    monkeypatch.setattr(M, "replay_verify", compare)
    if refuse_replay:
        with pytest.raises(M.SealedM2MError, match="independent materialized bytes differ"):
            M._issue(plan, run, replay_cpus=(frozenset({0}), frozenset({1})))
    else:
        _, replay = M._issue(plan, run, replay_cpus=(frozenset({0}), frozenset({1})))
        assert replay["status"] == "verified_sandbox_replay"
    trees = [path for path in run.glob("m2m-replay-*") if path.is_dir() and "-execution-" not in path.name]
    assert bool(trees) is refuse_replay
    assert (run / "sealed_capture_unverified_failure.json").exists() is refuse_replay


def test_diagnostic_write_failure_preserves_original_error_and_replay(tmp_path, monkeypatch):
    run, replay = tmp_path / "run", tmp_path / "run/replay"
    replay.mkdir(parents=True)
    monkeypatch.setattr(F, "_write", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(RuntimeError, match="original refusal") as caught:
        with F.publication(run, replay=replay, plan={}, selection_sha256="s" * 64):
            raise RuntimeError("original refusal")
    assert replay.exists()
    assert "OSError" in caught.value.__notes__[0]


def test_raw_success_process_diagnostics_never_overwrite_prior_observations_or_follow_alias(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "capture-execution").symlink_to(elsewhere, target_is_directory=True)
    for stdout in (b"first", b"second"):
        M._process_record(0, stdout, b"", failure_dir=tmp_path / "capture-failure")
    assert list(elsewhere.iterdir()) == []
    records = list(tmp_path.glob("capture-execution-*/stdout.bin"))
    assert sorted(record.read_bytes() for record in records) == [b"first", b"second"]


def test_failed_process_diagnostics_refuse_existing_alias(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "capture-failure").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(FileExistsError):
        M._process_record(1, b"actual", b"", failure_dir=tmp_path / "capture-failure")
    assert list(elsewhere.iterdir()) == []
