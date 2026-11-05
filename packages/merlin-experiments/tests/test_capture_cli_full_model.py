"""The capture CLI selects v2 full-model inputs and writes the v3 execution attestation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from merlin_experiments import cli
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _file_digest
from merlin_experiments.phase0 import capture_execution_attestation as attestation
from merlin_experiments.phase0 import capture_selection


def _select_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "corpus",
        "capture",
        "select",
        "--m2m-root",
        str(tmp_path / "m2m"),
        "--workload-root",
        str(tmp_path / "workload"),
        "--venv",
        str(tmp_path / "venv"),
        "--run-dir",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "selection"),
        *extra,
    ]


def _recorded_select(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    monkeypatch.setattr(
        capture_selection, "select", lambda **kwargs: calls.append(kwargs) or {"sha256": "a" * 64, "schema": "x"}
    )
    return calls


def test_full_model_flags_map_onto_the_v2_selection(tmp_path: Path, monkeypatch, capsys):
    calls = _recorded_select(monkeypatch)
    argv = _select_argv(
        tmp_path,
        "--dtype",
        "int8",
        "--checkpoint",
        f"weights/model.safetensors={tmp_path / 'ckpt.safetensors'}",
        "--extra-input",
        f"tokenizer={tmp_path / 'tok'}",
        "--extra-input",
        f"calibration/batch.pt={tmp_path / 'batch.pt'}",
        "--loader-env",
        "MODEL_MODE=complete",
        "--loader-env",
        "EMPTY=",
        "--loader-env-unset",
        "TRUNCATE_LAYERS",
        "--execution-timeout-seconds",
        "7200",
    )
    assert cli.main(argv) == 0
    (call,) = calls
    assert call["checkpoint"] == tmp_path / "ckpt.safetensors"
    assert call["checkpoint_guest_member"] == "weights/model.safetensors"
    assert call["extra_inputs"] == {"tokenizer": tmp_path / "tok", "calibration/batch.pt": tmp_path / "batch.pt"}
    assert call["loader_env"] == {"MODEL_MODE": "complete", "EMPTY": "", "TRUNCATE_LAYERS": None}
    assert call["execution_timeout_seconds"] == 7200
    assert call["dtype"] == "int8"
    assert '"sha256"' in capsys.readouterr().out


def test_checkpoint_free_selection_passes_no_full_model_keywords(tmp_path: Path, monkeypatch):
    calls = _recorded_select(monkeypatch)
    assert cli.main(_select_argv(tmp_path)) == 0
    assert not {
        "checkpoint",
        "checkpoint_guest_member",
        "extra_inputs",
        "loader_env",
        "execution_timeout_seconds",
    } & set(calls[0])


@pytest.mark.parametrize(
    "extra",
    [
        ("--checkpoint", "no-member-separator"),
        ("--checkpoint", "=/abs/path"),
        ("--extra-input", "a=/x", "--extra-input", "a=/y"),
        ("--loader-env", "NAME"),
        ("--loader-env", "A=1", "--loader-env", "A=2"),
        ("--loader-env", "A=1", "--loader-env-unset", "A"),
        ("--loader-env-unset", "A=1"),
    ],
)
def test_malformed_full_model_flags_are_refused_before_selection(tmp_path: Path, monkeypatch, capsys, extra):
    calls = _recorded_select(monkeypatch)
    assert cli.main(_select_argv(tmp_path, *extra)) != 0
    assert calls == []


def _issued_v3_capture(tmp_path: Path, monkeypatch, kind: str = "single") -> tuple[Path, str, Path]:
    """A v2 selection plus the run a sealed issue would leave; the sandbox replay is simulated."""
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"pretrained checkpoint")
    bwrap = tmp_path / "bwrap"
    bwrap.write_bytes(b"bubblewrap")
    roots = {name: tmp_path / name for name in ("m2m", "workload", "merlin", "schemas", "venv", "base")}
    for root in roots.values():
        root.mkdir()
    worker = roots["merlin"] / "targetgen/worker.py"
    worker.parent.mkdir()
    worker.write_bytes(b"worker")
    plan = {
        "schema": sealed_m2m.SCHEMA_V3,
        "status": "plan_only",
        "dtype": "fp32",
        "recipe": None,
        "m2m_root": str(roots["m2m"]),
        "workload_root": str(roots["workload"]),
        "merlin_root": str(roots["merlin"]),
        "schemas_root": str(roots["schemas"]),
        "venv": str(roots["venv"]),
        "base": str(roots["base"]),
        "worker": str(worker),
        "system_libs": [],
        "execution_timeout_seconds": 3600,
        "selected_inputs": [{"role": "checkpoint", **sealed_m2m._input_selection(checkpoint, "weights.bin")}],
        "loader_env": {"MODEL_MODE": "complete"},
        "max_snapshot_bytes": 15_000_000_000,
    }
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_: dict(plan))
    monkeypatch.setattr(capture_selection, "_bwrap_binary", lambda *_: bwrap)
    run = tmp_path / "run"
    identity = capture_selection.select(
        m2m_root=roots["m2m"],
        workload_root=roots["workload"],
        worker=worker,
        venv=roots["venv"],
        schemas_root=roots["schemas"],
        run_dir=run,
        output_dir=tmp_path / "selection",
        checkpoint=checkpoint,
        checkpoint_guest_member="weights.bin",
        loader_env={"MODEL_MODE": "complete"},
        execution_timeout_seconds=3600,
    )
    selected = capture_selection.load(Path(identity["path"]), expected_sha256=identity["sha256"])
    capture = run / "capture"
    model = capture / "model.mlir" if kind == "single" else capture / "stages/complete/model.mlir"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"complete model")
    if kind == "single":
        (capture / "capture_receipt.json").write_bytes(b"complete receipt")
    (run / "snapshots/source").mkdir(parents=True)
    (run / "snapshots/guest-root").mkdir()
    run.chmod(0o700)
    materialized = {"kind": kind, "integer_contractions": 2}
    if kind == "session":
        materialized.update(
            session_contract_sha256="a" * 64,
            session_receipt_sha256="b" * 64,
            programs=[{"name": "complete", "model_sha256": _file_digest(model), "receipt_sha256": "c" * 64}],
        )
    monkeypatch.setattr(sealed_m2m, "_materialized_v3", lambda *_args, **_kwargs: materialized)
    monkeypatch.setattr(sealed_m2m, "_verify_staged_selection", lambda *_args, **_kwargs: {})
    pending = run / "sealed_m2m_pending.json"
    pending.write_text(
        json.dumps(
            {
                "schema": sealed_m2m.SCHEMA_V3,
                "status": "pending_replay",
                "capture_selection_sha256": identity["sha256"],
                "plan": plan,
                "policy_sha256": selected["sandbox_policy_sha256"],
                "issuer_sha256": selected["issuer_source_sha256"],
                "bwrap_sha256": selected["bwrap"]["sha256"],
                "source": sealed_m2m._snapshot_tree(run / "snapshots/source"),
                "guest_root": sealed_m2m._snapshot_tree(run / "snapshots/guest-root"),
                "output": sealed_m2m._snapshot_tree(capture),
                "materialized": materialized,
            }
        )
    )
    monkeypatch.setattr(
        sealed_m2m,
        "replay_verify",
        lambda *_args, **_kwargs: {
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "receipt_sha256": _file_digest(pending),
        },
    )
    return Path(identity["path"]), identity["sha256"], run


@pytest.mark.parametrize("kind", ["single", "session"])
def test_attest_writes_the_v3_attestation_the_private_gate_reads(tmp_path: Path, monkeypatch, capsys, kind):
    selection, digest, run = _issued_v3_capture(tmp_path, monkeypatch, kind)
    output = run / "attestation.json"
    argv = ["corpus", "capture", "attest", "--selection", str(selection), "--expected-sha256", digest]
    assert cli.main([*argv, "--output", str(output)]) == 0
    printed = json.loads(capsys.readouterr().out)
    raw = output.read_bytes()
    assert printed["capture_execution_attestation_sha256"] == _file_digest(output)
    assert output.stat().st_mode & 0o777 == 0o400
    document = json.loads(raw)
    attestation.require_verified_execution(document)
    # The fields the private full-model gate checks before trusting the attestation.
    assert document["issuer"] == "merlin.sealed_m2m_cpu.v3"
    assert Path(document["selection"]["path"]).resolve() == selection.resolve()
    assert document["selection"]["sha256"] == digest
    assert document["capture"]["kind"] == kind
    assert Path(document["capture"]["capture_path"]).resolve() == (run / "capture").resolve()
    assert document["capture"]["integer_contractions"] == 2
    # A second attestation never overwrites the first.
    assert cli.main([*argv, "--output", str(output)]) != 0
    assert output.read_bytes() == raw


@pytest.mark.parametrize("where", ["capture/attestation.json", "snapshots/attestation.json"])
def test_attestation_never_enters_the_attested_bytes(tmp_path: Path, monkeypatch, where):
    selection, digest, run = _issued_v3_capture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="may not enter"):
        capture_selection.attest(selection, expected_sha256=digest, output=run / where)
    with pytest.raises(ValueError, match="may not enter"):
        capture_selection.attest(selection, expected_sha256=digest, output=selection.parent / "attestation.json")


def test_attestation_refuses_changed_capture_bytes(tmp_path: Path, monkeypatch):
    selection, digest, run = _issued_v3_capture(tmp_path, monkeypatch)
    (run / "capture/model.mlir").write_bytes(b"edited model")
    with pytest.raises(ValueError):
        capture_selection.attest(selection, expected_sha256=digest, output=tmp_path / "attestation.json")
    assert not (tmp_path / "attestation.json").exists()


def test_issue_can_attest_the_fresh_capture(tmp_path: Path, monkeypatch, capsys):
    selection, digest, run = _issued_v3_capture(tmp_path, monkeypatch)
    issued = []
    monkeypatch.setattr(
        capture_selection,
        "issue",
        lambda path, *, expected_sha256: issued.append((path, expected_sha256)) or run / "sealed_m2m_pending.json",
    )
    output = tmp_path / "attestation.json"
    argv = ["corpus", "capture", "issue", "--selection", str(selection), "--expected-sha256", digest]
    assert cli.main([*argv, "--attestation-output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert issued == [(selection, digest)]
    assert result["phase0_admission"] == "not_granted"
    assert result["issuer"] == "merlin.sealed_m2m_cpu.v3"
    assert result["capture_execution_attestation"] == str(output)
    attestation.require_verified_execution(json.loads(output.read_bytes()))


def test_issue_refuses_an_existing_attestation_output_before_capture(tmp_path: Path, monkeypatch):
    output = tmp_path / "attestation.json"
    output.write_text("{}")
    issued = []
    monkeypatch.setattr(capture_selection, "issue", lambda *a, **k: issued.append(a))
    argv = ["corpus", "capture", "issue", "--selection", str(tmp_path / "s.json"), "--expected-sha256", "a" * 64]
    assert cli.main([*argv, "--attestation-output", str(output)]) != 0
    assert issued == []
    assert os.path.getsize(output) == 2


def test_checkpoint_free_selection_accepts_a_selected_timeout(tmp_path: Path, monkeypatch):
    calls = _recorded_select(monkeypatch)
    assert cli.main(_select_argv(tmp_path, "--execution-timeout-seconds", "900")) == 0
    assert calls[0]["execution_timeout_seconds"] == 900
    assert "checkpoint" not in calls[0] and "loader_env" not in calls[0]


def test_stage_fp32_selects_the_worker_option_and_absence_keeps_the_plan(tmp_path: Path, monkeypatch):
    calls = _recorded_select(monkeypatch)
    assert cli.main(_select_argv(tmp_path, "--stage-fp32")) == 0
    assert calls[0]["worker_options"] == {"stage_fp32": True}
    assert cli.main(_select_argv(tmp_path)) == 0
    assert "worker_options" not in calls[1]
